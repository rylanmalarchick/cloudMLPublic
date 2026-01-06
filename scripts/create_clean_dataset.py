#!/usr/bin/env python3
"""
Create Clean Integrated Features Datasets

This script creates two verified datasets:
1. Clean_933_Integrated_Features.hdf5 - Matches paper filtering (ocean, CBH 0.2-2km, single-layer)
2. Clean_3334_Integrated_Features.hdf5 - All samples (no filters except valid cloud)

CRITICAL: These datasets do NOT include 'inversion_height' which caused data leakage
in the original study (inversion_height = CBH - BLH directly encodes the target).

Features included (10 total):
- ERA5 raw: blh, t2m, d2m, sp, tcwv
- ERA5 derived: lcl, stability_index, moisture_gradient
- Geometric: sza_deg, saa_deg

Reference:
    Espy's LCL equation: LCL(m) ≈ 125 × (T - Td)
    See: Stull, R.B. (1988). An Introduction to Boundary Layer Meteorology.

Author: AgentBible-assisted development
Date: 2026-01-06
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional
import json

import h5py
import numpy as np
import xarray as xr

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Import AgentBible for provenance and validation
from agentbible.provenance import save_with_metadata, get_provenance_metadata
from agentbible import validate_finite

# Import our CBH-specific validators
from src.cbh_validators import (
    validate_no_leakage,
    validate_era5_nonzero,
    validate_lcl_cbh_consistency,
    compute_temporal_autocorrelation,
)
from src.cplCompareSub import cplTimeConvert

# =============================================================================
# Configuration
# =============================================================================

DATA_ROOT = Path("/home/rylan/Documents/research/NASA/programDirectory/data")
ERA5_DIR = Path("/media/rylan/two/research/NASA/ERA5_data_root/surface")
OUTPUT_DIR = PROJECT_ROOT / "outputs/preprocessed_data"

# Flight configuration (EXCLUDING 04Nov24 - no camera data)
FLIGHT_CONFIG = {
    '23Oct24': {
        'era5_date': '20241023',
        'cpl_file': 'CPL_L2_V1-02_01kmLay_259004_23oct24.hdf5',
        'irai_pattern': '*IRAI_L1B*.h5',
        'nav_file': 'CRS_20241023_nav.hdf',
        'year': 2024,
        'flight_id': 0,
    },
    '30Oct24': {
        'era5_date': '20241030',
        'cpl_file': 'CPL_L2_V1-02_01kmLay_259006_30oct24.hdf5',
        'irai_pattern': '*IRAI_L1B*.h5',
        'nav_file': 'CRS_20241030_nav.hdf',
        'year': 2024,
        'flight_id': 1,
    },
    # 04Nov24 EXCLUDED - no camera data available
    '10Feb25': {
        'era5_date': '20250210',
        'cpl_file': 'CPL_L2_V1-02_01kmLay_259015_10feb25.hdf5',
        'irai_pattern': '*IRAI_L1B*.h5',
        'nav_file': 'CRS_20250210_nav.hdf',
        'year': 2025,
        'flight_id': 2,
    },
    '12Feb25': {
        'era5_date': '20250212',
        'cpl_file': 'CPL_L2_V1-02_01kmLay_259016_12feb25.hdf5',
        'irai_pattern': '*IRAI_L1B*.h5',
        'nav_file': 'CRS_20250212_nav.hdf',
        'year': 2025,
        'flight_id': 3,
    },
    '18Feb25': {
        'era5_date': '20250218',
        'cpl_file': 'CPL_L2_V1-02_01kmLay_259017_18feb25.hdf5',
        'irai_pattern': '*IRAI_L1B*.h5',
        'nav_file': 'CRS_20250218_nav.hdf',
        'year': 2025,
        'flight_id': 4,
    },
}

# Paper filter criteria
PAPER_FILTERS = {
    'cbh_min_km': 0.2,
    'cbh_max_km': 2.0,
    'dem_value': 0.0,  # Exact ocean (DEM == 0)
    'single_layer_only': True,
    'temporal_tolerance_s': 0.5,
}


@dataclass
class Sample:
    """Container for a matched CPL-Camera-ERA5 sample."""
    flight_id: int
    flight_name: str
    sample_id: int
    cbh_km: float
    latitude: float
    longitude: float
    timestamp: float
    sza_deg: float
    saa_deg: float
    dem_m: float
    num_layers: int
    # ERA5 features
    blh: float
    t2m: float
    d2m: float
    sp: float
    tcwv: float
    # Derived features (NO inversion_height!)
    lcl: float
    stability_index: float
    moisture_gradient: float


def julian_day_to_timestamp(jd: float, year: int) -> float:
    """Convert Julian day (decimal) to Unix timestamp.
    
    Args:
        jd: Decimal Julian day (1.0 = midnight Jan 1)
        year: Year for the Julian day
        
    Returns:
        Unix timestamp (seconds since 1970-01-01)
    """
    base = datetime(year, 1, 1, tzinfo=None)
    dt = base + timedelta(days=jd - 1)
    return dt.timestamp()


def compute_solar_position(lat: float, lon: float, dt: datetime) -> tuple[float, float]:
    """Compute solar zenith and azimuth angles.
    
    Based on NOAA Solar Calculator algorithms.
    
    Reference:
        Meeus, J. (1991). Astronomical Algorithms. Willmann-Bell.
        
    Args:
        lat: Latitude in degrees
        lon: Longitude in degrees  
        dt: UTC datetime
        
    Returns:
        (sza_deg, saa_deg): Solar zenith and azimuth angles in degrees
    """
    lat_rad = np.radians(lat)
    doy = dt.timetuple().tm_yday
    
    # Fractional year (radians)
    gamma = 2 * np.pi / 365 * (doy - 1 + (dt.hour + dt.minute/60) / 24)
    
    # Equation of time (minutes)
    eqtime = 229.18 * (0.000075 + 0.001868 * np.cos(gamma) - 0.032077 * np.sin(gamma)
                       - 0.014615 * np.cos(2 * gamma) - 0.040849 * np.sin(2 * gamma))
    
    # Solar declination (radians)
    decl = (0.006918 - 0.399912 * np.cos(gamma) + 0.070257 * np.sin(gamma)
            - 0.006758 * np.cos(2 * gamma) + 0.000907 * np.sin(2 * gamma)
            - 0.002697 * np.cos(3 * gamma) + 0.00148 * np.sin(3 * gamma))
    
    # True solar time (minutes since midnight)
    time_offset = eqtime + 4 * lon
    tst = dt.hour * 60 + dt.minute + dt.second / 60 + time_offset
    
    # Hour angle (degrees)
    ha = (tst / 4) - 180
    ha_rad = np.radians(ha)
    
    # Solar zenith angle
    cos_sza = (np.sin(lat_rad) * np.sin(decl) + 
               np.cos(lat_rad) * np.cos(decl) * np.cos(ha_rad))
    sza = np.degrees(np.arccos(np.clip(cos_sza, -1, 1)))
    
    # Solar azimuth angle
    sin_sza = np.sin(np.radians(sza))
    if sin_sza > 0.01:
        cos_saa = (np.sin(decl) - np.sin(lat_rad) * cos_sza) / (np.cos(lat_rad) * sin_sza)
        saa = np.degrees(np.arccos(np.clip(cos_saa, -1, 1)))
        if ha > 0:
            saa = 360 - saa
    else:
        saa = 0
    
    return sza, saa


@validate_finite
def compute_lcl(t2m_k: float, d2m_k: float) -> float:
    """Compute Lifting Condensation Level using Espy's equation.
    
    LCL(m) ≈ 125 × (T - Td)
    
    Reference:
        Espy, J.P. (1836). Essays on Meteorology.
        Stull, R.B. (1988). An Introduction to Boundary Layer Meteorology.
        
    Args:
        t2m_k: 2-meter temperature in Kelvin
        d2m_k: 2-meter dewpoint temperature in Kelvin
        
    Returns:
        LCL height in meters (clipped to physical range)
    """
    t_celsius = t2m_k - 273.15
    td_celsius = d2m_k - 273.15
    lcl_m = 125.0 * (t_celsius - td_celsius)
    return max(0.0, min(lcl_m, 15000.0))  # Physical bounds


def compute_derived_features(t2m: float, d2m: float, blh: float) -> dict:
    """Compute derived atmospheric features.
    
    NOTE: Does NOT compute inversion_height (would be CBH - BLH = data leakage).
    
    Args:
        t2m: 2m temperature in Kelvin
        d2m: 2m dewpoint in Kelvin
        blh: Boundary layer height in meters
        
    Returns:
        Dictionary of derived features
    """
    lcl = compute_lcl(t2m, d2m)
    
    # Stability index: dewpoint depression normalized
    stability_index = (t2m - d2m) / 10.0
    
    # Moisture gradient: dewpoint depression per km of BLH
    if blh > 0:
        moisture_gradient = (t2m - d2m) / (blh / 1000.0)
    else:
        moisture_gradient = 0.0
    
    return {
        'lcl': lcl,
        'stability_index': stability_index,
        'moisture_gradient': moisture_gradient,
    }


def load_era5_data(date_str: str) -> Optional[xr.Dataset]:
    """Load ERA5 surface data for a specific date.
    
    Args:
        date_str: Date in YYYYMMDD format
        
    Returns:
        xarray Dataset or None if file doesn't exist
    """
    era5_file = ERA5_DIR / f"era5_surface_{date_str}.nc"
    if not era5_file.exists():
        print(f"  WARNING: ERA5 file not found: {era5_file}")
        return None
    return xr.open_dataset(era5_file)


def extract_era5_at_point(era5_ds: xr.Dataset, lat: float, lon: float, hour: int) -> dict:
    """Extract ERA5 values at a specific lat/lon/time.
    
    Args:
        era5_ds: ERA5 xarray Dataset
        lat: Latitude
        lon: Longitude
        hour: Hour of day (0-23)
        
    Returns:
        Dictionary of ERA5 values
    """
    # Handle longitude convention (ERA5 may use 0-360)
    if era5_ds.longitude.values.min() >= 0 and lon < 0:
        lon_query = lon + 360
    else:
        lon_query = lon
    
    # Find nearest indices
    lat_idx = int(np.argmin(np.abs(era5_ds.latitude.values - lat)))
    lon_idx = int(np.argmin(np.abs(era5_ds.longitude.values - lon_query)))
    time_idx = min(hour, len(era5_ds.valid_time) - 1)
    
    result = {}
    for var in ['blh', 't2m', 'd2m', 'sp', 'tcwv']:
        if var in era5_ds:
            result[var] = float(era5_ds[var].isel(
                valid_time=time_idx,
                latitude=lat_idx,
                longitude=lon_idx
            ).values)
    
    return result


def process_flight(
    flight_name: str,
    config: dict,
    apply_paper_filters: bool = True,
) -> list[Sample]:
    """Process a single flight and extract matched samples.
    
    Args:
        flight_name: Name of the flight (e.g., '23Oct24')
        config: Flight configuration dictionary
        apply_paper_filters: Whether to apply paper filtering criteria
        
    Returns:
        List of Sample objects
    """
    print(f"\n{'='*60}")
    print(f"Processing flight: {flight_name}")
    print(f"{'='*60}")
    
    flight_dir = DATA_ROOT / flight_name
    
    # Load CPL data
    cpl_file = flight_dir / config['cpl_file']
    if not cpl_file.exists():
        print(f"  ERROR: CPL file not found: {cpl_file}")
        return []
    
    print(f"  Loading CPL: {config['cpl_file']}")
    with h5py.File(cpl_file, 'r') as f:
        cpl_jd = f['layer_descriptor/Profile_Decimal_Julian_Day'][:, 0]
        cbh_all = f['layer_descriptor/Layer_Base_Altitude'][:, 0]
        num_layers = f['layer_descriptor/Number_Layers'][:]
        dem_m = f['layer_descriptor/DEM_Surface_Altitude'][:]
        cpl_lat = f['geolocation/CPL_Latitude'][:, 1]
        cpl_lon = f['geolocation/CPL_Longitude'][:, 1]
    
    # Convert Julian day to Unix timestamp
    cpl_times = cplTimeConvert(cpl_jd, str(cpl_file))
    
    print(f"    Total CPL profiles: {len(cpl_times)}")
    
    # Load camera timestamps
    irai_files = list(flight_dir.glob(config['irai_pattern']))
    if not irai_files:
        print(f"  ERROR: No IRAI files found matching {config['irai_pattern']}")
        return []
    
    irai_file = irai_files[0]
    print(f"  Loading IRAI: {irai_file.name}")
    with h5py.File(irai_file, 'r') as f:
        irai_times = f['Time/TimeUTC'][:]
    
    print(f"    Total IRAI frames: {len(irai_times)}")
    
    # Load ERA5 data
    era5_ds = load_era5_data(config['era5_date'])
    if era5_ds is None:
        return []
    print(f"  Loaded ERA5 for {config['era5_date']}")
    
    # Process each CPL profile
    samples = []
    stats = {
        'total': len(cpl_times),
        'valid_cbh': 0,
        'cbh_range': 0,
        'ocean': 0,
        'single_layer': 0,
        'temporal_match': 0,
    }
    
    for i in range(len(cpl_times)):
        cbh_km = cbh_all[i]
        
        # Filter: valid CBH
        if not np.isfinite(cbh_km) or cbh_km < 0:
            continue
        stats['valid_cbh'] += 1
        
        if apply_paper_filters:
            # Filter: CBH range
            if cbh_km < PAPER_FILTERS['cbh_min_km'] or cbh_km > PAPER_FILTERS['cbh_max_km']:
                continue
            stats['cbh_range'] += 1
            
            # Filter: Ocean only (DEM == 0)
            if dem_m[i] != PAPER_FILTERS['dem_value']:
                continue
            stats['ocean'] += 1
            
            # Filter: Single layer only
            if num_layers[i] != 1:
                continue
            stats['single_layer'] += 1
        
        # Temporal matching to camera
        time_diffs = np.abs(irai_times - cpl_times[i])
        closest_idx = np.argmin(time_diffs)
        dt = time_diffs[closest_idx]
        
        if apply_paper_filters and dt > PAPER_FILTERS['temporal_tolerance_s']:
            continue
        stats['temporal_match'] += 1
        
        # Get position and time
        lat = float(cpl_lat[i])
        lon = float(cpl_lon[i])
        timestamp = float(irai_times[closest_idx])
        dt_obj = datetime.utcfromtimestamp(timestamp)
        
        # Compute solar angles
        sza, saa = compute_solar_position(lat, lon, dt_obj)
        
        # Skip nighttime (SZA > 85)
        if sza > 85:
            continue
        
        # Extract ERA5 data
        era5_values = extract_era5_at_point(era5_ds, lat, lon, dt_obj.hour)
        
        if not all(v != 0 for v in era5_values.values()):
            continue  # Skip if any ERA5 value is missing
        
        # Compute derived features (NO inversion_height!)
        derived = compute_derived_features(
            era5_values['t2m'],
            era5_values['d2m'],
            era5_values['blh'],
        )
        
        sample = Sample(
            flight_id=config['flight_id'],
            flight_name=flight_name,
            sample_id=int(closest_idx),
            cbh_km=float(cbh_km),
            latitude=lat,
            longitude=lon,
            timestamp=timestamp,
            sza_deg=float(sza),
            saa_deg=float(saa),
            dem_m=float(dem_m[i]),
            num_layers=int(num_layers[i]),
            blh=era5_values['blh'],
            t2m=era5_values['t2m'],
            d2m=era5_values['d2m'],
            sp=era5_values['sp'],
            tcwv=era5_values['tcwv'],
            lcl=derived['lcl'],
            stability_index=derived['stability_index'],
            moisture_gradient=derived['moisture_gradient'],
        )
        samples.append(sample)
    
    era5_ds.close()
    
    # Print statistics
    print(f"\n  Filtering cascade:")
    print(f"    Total CPL profiles:     {stats['total']:6d}")
    print(f"    Valid CBH (>0, finite): {stats['valid_cbh']:6d}")
    if apply_paper_filters:
        print(f"    CBH in [0.2, 2.0] km:   {stats['cbh_range']:6d}")
        print(f"    Ocean only (DEM=0):     {stats['ocean']:6d}")
        print(f"    Single layer:           {stats['single_layer']:6d}")
        print(f"    Temporal match (<0.5s): {stats['temporal_match']:6d}")
    print(f"    Final samples:          {len(samples):6d}")
    
    return samples


def save_dataset(
    samples: list[Sample],
    output_file: Path,
    description: str,
    apply_paper_filters: bool,
) -> None:
    """Save samples to HDF5 with AgentBible provenance.
    
    Args:
        samples: List of Sample objects
        output_file: Path to output HDF5 file
        description: Description for provenance
        apply_paper_filters: Whether paper filters were applied
    """
    n = len(samples)
    
    if n == 0:
        print("ERROR: No samples to save!")
        return
    
    # Feature names (EXCLUDING inversion_height)
    feature_names = [
        'blh', 't2m', 'd2m', 'sp', 'tcwv',  # ERA5 raw
        'lcl', 'stability_index', 'moisture_gradient',  # ERA5 derived
        'sza_deg', 'saa_deg',  # Geometric
    ]
    
    # Validate: no leakage features
    validate_no_leakage(feature_names)
    
    # Extract arrays
    cbh_km = np.array([s.cbh_km for s in samples], dtype=np.float32)
    flight_ids = np.array([s.flight_id for s in samples], dtype=np.int32)
    sample_ids = np.array([s.sample_id for s in samples], dtype=np.int32)
    latitudes = np.array([s.latitude for s in samples], dtype=np.float32)
    longitudes = np.array([s.longitude for s in samples], dtype=np.float32)
    timestamps = np.array([s.timestamp for s in samples], dtype=np.float64)
    
    # ERA5 features
    blh = np.array([s.blh for s in samples], dtype=np.float32)
    t2m = np.array([s.t2m for s in samples], dtype=np.float32)
    d2m = np.array([s.d2m for s in samples], dtype=np.float32)
    sp = np.array([s.sp for s in samples], dtype=np.float32)
    tcwv = np.array([s.tcwv for s in samples], dtype=np.float32)
    
    # Derived features
    lcl = np.array([s.lcl for s in samples], dtype=np.float32)
    stability_index = np.array([s.stability_index for s in samples], dtype=np.float32)
    moisture_gradient = np.array([s.moisture_gradient for s in samples], dtype=np.float32)
    
    # Geometric features
    sza_deg = np.array([s.sza_deg for s in samples], dtype=np.float32)
    saa_deg = np.array([s.saa_deg for s in samples], dtype=np.float32)
    
    # Validate ERA5 features are non-zero
    era5_features = {
        'blh': blh, 't2m': t2m, 'd2m': d2m, 'sp': sp, 'tcwv': tcwv
    }
    validate_era5_nonzero(era5_features)
    
    # Check LCL-CBH consistency
    lcl_cbh = validate_lcl_cbh_consistency(lcl, cbh_km * 1000)
    print(f"\n  LCL-CBH consistency check:")
    print(f"    Correlation: {lcl_cbh['lcl_cbh_correlation']:.3f}")
    print(f"    LCL > CBH fraction: {lcl_cbh['lcl_above_cbh_fraction']*100:.1f}%")
    if lcl_cbh['warning']:
        print(f"    WARNING: {lcl_cbh['warning']}")
    
    # Compute temporal autocorrelation per flight
    print(f"\n  Temporal autocorrelation:")
    for fid in np.unique(flight_ids):
        mask = flight_ids == fid
        cbh_flight = cbh_km[mask]
        autocorr = compute_temporal_autocorrelation(cbh_flight)
        fname = [s.flight_name for s in samples if s.flight_id == fid][0]
        print(f"    {fname}: lag-1 = {autocorr['lag1_autocorr']:.3f}")
        if autocorr['warning']:
            print(f"      WARNING: {autocorr['warning']}")
    
    # Create flight mapping
    flight_mapping = {str(fid): fname for fid, fname in 
                      set((s.flight_id, s.flight_name) for s in samples)}
    
    # Get provenance metadata
    provenance = get_provenance_metadata(
        description=description,
        extra={
            'n_samples': n,
            'n_flights': len(np.unique(flight_ids)),
            'feature_names': feature_names,
            'apply_paper_filters': apply_paper_filters,
            'paper_filters': PAPER_FILTERS if apply_paper_filters else None,
            'excluded_flights': ['04Nov24'],
            'excluded_features': ['inversion_height'],
            'lcl_cbh_correlation': float(lcl_cbh['lcl_cbh_correlation']),
        },
    )
    
    # Save HDF5
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(output_file, 'w') as f:
        # Metadata group
        meta = f.create_group('metadata')
        meta.create_dataset('cbh_km', data=cbh_km)
        meta.create_dataset('flight_id', data=flight_ids)
        meta.create_dataset('sample_id', data=sample_ids)
        meta.create_dataset('latitude', data=latitudes)
        meta.create_dataset('longitude', data=longitudes)
        meta.create_dataset('timestamp', data=timestamps)
        
        # Geometric features
        geo = f.create_group('geometric_features')
        geo.create_dataset('sza_deg', data=sza_deg)
        geo.create_dataset('saa_deg', data=saa_deg)
        
        # Atmospheric features (NO inversion_height!)
        atmo = f.create_group('atmospheric_features')
        atmo.create_dataset('blh', data=blh)
        atmo.create_dataset('t2m', data=t2m)
        atmo.create_dataset('d2m', data=d2m)
        atmo.create_dataset('sp', data=sp)
        atmo.create_dataset('tcwv', data=tcwv)
        atmo.create_dataset('lcl', data=lcl)
        atmo.create_dataset('stability_index', data=stability_index)
        atmo.create_dataset('moisture_gradient', data=moisture_gradient)
        
        # Global attributes
        f.attrs['title'] = 'Clean Integrated CBH Features'
        f.attrs['description'] = description
        f.attrs['created'] = datetime.now().isoformat()
        f.attrs['n_samples'] = n
        f.attrs['n_flights'] = len(np.unique(flight_ids))
        f.attrs['flight_mapping'] = json.dumps(flight_mapping)
        f.attrs['feature_names'] = json.dumps(feature_names)
        f.attrs['apply_paper_filters'] = apply_paper_filters
        f.attrs['provenance'] = json.dumps(provenance)
        f.attrs['agentbible_version'] = '0.3.0'
        
        # CRITICAL: Document that inversion_height is excluded
        f.attrs['excluded_features'] = json.dumps(['inversion_height'])
        f.attrs['exclusion_reason'] = (
            'inversion_height = CBH - BLH directly encodes target variable, '
            'causing R² = 0.999 data leakage. NEVER include this feature.'
        )
    
    print(f"\n  Saved to: {output_file}")
    print(f"  Total samples: {n}")
    print(f"  Features: {feature_names}")


def main():
    """Create both 933-sample and 3334-sample clean datasets."""
    
    print("=" * 80)
    print("CREATING CLEAN CBH DATASETS")
    print("=" * 80)
    print(f"\nThis script creates verified datasets WITHOUT data leakage.")
    print(f"EXCLUDED: inversion_height (causes R² = 0.999 leakage)")
    print(f"EXCLUDED: Flight 04Nov24 (no camera data)")
    print("=" * 80)
    
    # =========================================================================
    # Dataset 1: 933-sample (paper filtering)
    # =========================================================================
    print("\n" + "=" * 80)
    print("DATASET 1: Paper-filtered (933 samples expected)")
    print("Filters: ocean only, CBH 0.2-2km, single-layer, 0.5s temporal match")
    print("=" * 80)
    
    samples_933 = []
    for flight_name, config in FLIGHT_CONFIG.items():
        flight_samples = process_flight(flight_name, config, apply_paper_filters=True)
        samples_933.extend(flight_samples)
    
    print(f"\nTotal 933-dataset samples: {len(samples_933)}")
    
    save_dataset(
        samples_933,
        OUTPUT_DIR / "Clean_933_Integrated_Features.hdf5",
        description=(
            "CBH dataset with paper filtering (ocean, CBH 0.2-2km, single-layer). "
            "EXCLUDES inversion_height to prevent data leakage."
        ),
        apply_paper_filters=True,
    )
    
    # =========================================================================
    # Dataset 2: All samples (no paper filters)
    # =========================================================================
    print("\n" + "=" * 80)
    print("DATASET 2: All valid samples (no paper filters)")
    print("Only filter: valid CBH, daytime, ERA5 available")
    print("=" * 80)
    
    samples_all = []
    for flight_name, config in FLIGHT_CONFIG.items():
        flight_samples = process_flight(flight_name, config, apply_paper_filters=False)
        samples_all.extend(flight_samples)
    
    print(f"\nTotal all-samples dataset: {len(samples_all)}")
    
    save_dataset(
        samples_all,
        OUTPUT_DIR / "Clean_All_Integrated_Features.hdf5",
        description=(
            "CBH dataset with all valid samples (no paper filters). "
            "EXCLUDES inversion_height to prevent data leakage."
        ),
        apply_paper_filters=False,
    )
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\n  Paper-filtered dataset: {len(samples_933)} samples")
    print(f"  All-samples dataset:    {len(samples_all)} samples")
    print(f"\n  Excluded features: inversion_height (data leakage)")
    print(f"  Excluded flights: 04Nov24 (no camera data)")
    print(f"\n  Output directory: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == '__main__':
    main()
