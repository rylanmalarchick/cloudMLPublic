#!/usr/bin/env python3
"""
Rebuild Verified Dataset from Raw CPL and ERA5 Data

This script creates a verified dataset by:
1. Extracting all valid CBH samples from CPL L2 files
2. Matching each sample to ERA5 atmospheric data
3. Computing derived features (LCL, stability, etc.)
4. Saving to a verified HDF5 file

Author: Automated rebuild
Date: 2026-01-06
"""

import h5py
import numpy as np
import xarray as xr
from pathlib import Path
from datetime import datetime, timedelta
import json

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
ERA5_DIR = Path("/media/rylan/two/research/NASA/ERA5_data_root/surface")
OUTPUT_FILE = PROJECT_ROOT / "outputs/preprocessed_data/Verified_Integrated_Features.hdf5"

# Flight date mapping (folder name -> YYYYMMDD)
FLIGHT_DATES = {
    '22Oct24': '20241022',
    '23Oct24': '20241023', 
    '30Oct24': '20241030',
    '04Nov24': '20241104',
    '10Feb25': '20250210',
    '12Feb25': '20250212',
    '18Feb25': '20250218',
}

# Flight ID mapping
FLIGHT_IDS = {
    '23Oct24': 0,
    '30Oct24': 1,
    '04Nov24': 2,
    '10Feb25': 3,
    '12Feb25': 4,
    '18Feb25': 5,
}

def julian_day_to_datetime(jd, year):
    """Convert Julian day (decimal) to datetime."""
    # Julian day is days since start of year
    base = datetime(year, 1, 1)
    return base + timedelta(days=jd - 1)

def compute_solar_position(lat, lon, dt_utc):
    """
    Compute solar zenith and azimuth angles from lat/lon and UTC datetime.
    
    Based on NOAA Solar Calculator algorithms.
    """
    lat_rad = np.radians(lat)
    
    # Day of year
    doy = dt_utc.timetuple().tm_yday
    
    # Fractional year (radians)
    gamma = 2 * np.pi / 365 * (doy - 1 + (dt_utc.hour + dt_utc.minute/60) / 24)
    
    # Equation of time (minutes)
    eqtime = 229.18 * (0.000075 + 0.001868 * np.cos(gamma) - 0.032077 * np.sin(gamma)
                       - 0.014615 * np.cos(2 * gamma) - 0.040849 * np.sin(2 * gamma))
    
    # Solar declination (radians)
    decl = (0.006918 - 0.399912 * np.cos(gamma) + 0.070257 * np.sin(gamma)
            - 0.006758 * np.cos(2 * gamma) + 0.000907 * np.sin(2 * gamma)
            - 0.002697 * np.cos(3 * gamma) + 0.00148 * np.sin(3 * gamma))
    
    # True solar time (minutes since midnight)
    time_offset = eqtime + 4 * lon
    tst = dt_utc.hour * 60 + dt_utc.minute + dt_utc.second / 60 + time_offset
    
    # Hour angle (degrees) - 0 at solar noon
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

def extract_cpl_samples(cpl_file: Path, flight_name: str):
    """Extract valid samples from a CPL L2 Layer file."""
    samples = []
    
    # Determine year from flight name
    if '24' in flight_name:
        year = 2024
    else:
        year = 2025
    
    with h5py.File(cpl_file, 'r') as f:
        lat = f['geolocation/CPL_Latitude'][:, 1]  # Center value
        lon = f['geolocation/CPL_Longitude'][:, 1]
        cbh = f['layer_descriptor/Layer_Base_Altitude'][:, 0]  # First layer (lowest)
        cth = f['layer_descriptor/Layer_Top_Altitude'][:, 0]  # First layer top
        
        # Get timestamps from Julian day
        jd = f['layer_descriptor/Profile_Decimal_Julian_Day'][:, 1]  # Center
        
        # Filter valid samples
        # CBH > 0.1 km (100m minimum - valid measurement)
        # CBH < 3 km (focus on boundary layer clouds)
        valid = (cbh > 0.1) & (cbh < 3.0)
        
        for i in np.where(valid)[0]:
            dt = julian_day_to_datetime(jd[i], year)
            
            # Compute solar angles from position and time
            sza, saa = compute_solar_position(lat[i], lon[i], dt)
            
            # Skip nighttime samples (SZA > 85)
            if sza > 85:
                continue
            
            samples.append({
                'flight': flight_name,
                'flight_id': FLIGHT_IDS.get(flight_name, -1),
                'lat': float(lat[i]),
                'lon': float(lon[i]),
                'sza_deg': float(sza),
                'saa_deg': float(saa),
                'cbh_km': float(cbh[i]),
                'cth_km': float(cth[i]) if cth[i] > 0 else np.nan,
                'timestamp': dt.timestamp(),
                'datetime': dt.isoformat(),
                'hour': dt.hour,
            })
    
    return samples

def load_era5_for_date(date_str: str):
    """Load ERA5 data for a specific date."""
    era5_file = ERA5_DIR / f"era5_surface_{date_str}.nc"
    if not era5_file.exists():
        return None
    return xr.open_dataset(era5_file)

def match_era5_to_sample(sample: dict, era5_ds):
    """Match ERA5 data to a sample location and time."""
    if era5_ds is None:
        return None
    
    lat, lon = sample['lat'], sample['lon']
    hour = sample['hour']
    
    # Find nearest grid point
    lat_idx = np.argmin(np.abs(era5_ds.latitude.values - lat))
    lon_idx = np.argmin(np.abs(era5_ds.longitude.values - lon))
    
    # Handle longitude convention (ERA5 uses 0-360, we might have -180 to 180)
    if lon < 0:
        lon_360 = lon + 360
        lon_idx_360 = np.argmin(np.abs(era5_ds.longitude.values - lon_360))
        if abs(era5_ds.longitude.values[lon_idx_360] - lon_360) < abs(era5_ds.longitude.values[lon_idx] - lon):
            lon_idx = lon_idx_360
    
    # Find nearest time
    time_idx = hour  # ERA5 is hourly, so hour index should work
    time_idx = min(time_idx, len(era5_ds.valid_time) - 1)
    
    # Extract values
    era5_data = {}
    for var in ['blh', 't2m', 'd2m', 'sp', 'tcwv']:
        if var in era5_ds:
            era5_data[var] = float(era5_ds[var].isel(
                valid_time=time_idx, 
                latitude=lat_idx, 
                longitude=lon_idx
            ).values)
    
    return era5_data

def compute_lcl(t2m: float, d2m: float) -> float:
    """
    Compute Lifting Condensation Level using Espy's equation.
    
    LCL ≈ 125 * (T - Td) meters
    
    Parameters
    ----------
    t2m : float
        2-meter temperature in Kelvin
    d2m : float
        2-meter dewpoint temperature in Kelvin
        
    Returns
    -------
    float
        LCL height in meters
    """
    t_celsius = t2m - 273.15
    td_celsius = d2m - 273.15
    lcl_m = 125.0 * (t_celsius - td_celsius)
    return max(0, lcl_m)  # LCL can't be negative

def compute_derived_features(era5_data: dict, cbh_km: float) -> dict:
    """Compute derived atmospheric features."""
    derived = {}
    
    t2m = era5_data.get('t2m', 288)
    d2m = era5_data.get('d2m', 283)
    blh = era5_data.get('blh', 1000)
    
    # LCL height
    derived['lcl'] = compute_lcl(t2m, d2m)
    
    # Stability index (simple: T-Td spread)
    derived['stability_index'] = t2m - d2m
    
    # Moisture gradient (dewpoint depression per km of BLH)
    if blh > 0:
        derived['moisture_gradient'] = (t2m - d2m) / (blh / 1000)
    else:
        derived['moisture_gradient'] = 0
    
    # Inversion height estimate (CBH - BLH relationship)
    cbh_m = cbh_km * 1000
    derived['inversion_height'] = cbh_m - blh
    
    return derived

def main():
    print("=" * 80)
    print("REBUILDING VERIFIED DATASET")
    print("=" * 80)
    
    # Step 1: Extract all CPL samples
    print("\n1. Extracting CPL samples...")
    cpl_files = sorted(DATA_DIR.rglob('CPL_L2_V1-02_01kmLay*.hdf5'))
    
    all_samples = []
    for cpl_file in cpl_files:
        flight_name = cpl_file.parent.name
        if flight_name not in FLIGHT_DATES:
            print(f"  Skipping {flight_name} (unknown flight)")
            continue
            
        era5_date = FLIGHT_DATES[flight_name]
        era5_file = ERA5_DIR / f"era5_surface_{era5_date}.nc"
        
        if not era5_file.exists():
            print(f"  Skipping {flight_name} (no ERA5 data for {era5_date})")
            continue
        
        samples = extract_cpl_samples(cpl_file, flight_name)
        print(f"  {flight_name}: {len(samples)} valid samples")
        all_samples.extend(samples)
    
    print(f"\nTotal CPL samples: {len(all_samples)}")
    
    # Step 2: Match ERA5 data
    print("\n2. Matching ERA5 data...")
    era5_cache = {}  # Cache loaded ERA5 datasets
    
    matched_samples = []
    for i, sample in enumerate(all_samples):
        if i % 1000 == 0:
            print(f"  Processing {i}/{len(all_samples)}...")
        
        flight_name = sample['flight']
        era5_date = FLIGHT_DATES[flight_name]
        
        # Load ERA5 (cached)
        if era5_date not in era5_cache:
            era5_cache[era5_date] = load_era5_for_date(era5_date)
        
        era5_ds = era5_cache[era5_date]
        era5_data = match_era5_to_sample(sample, era5_ds)
        
        if era5_data is None:
            continue
        
        # Compute derived features
        derived = compute_derived_features(era5_data, sample['cbh_km'])
        
        # Merge all data
        sample.update(era5_data)
        sample.update(derived)
        matched_samples.append(sample)
    
    # Close cached datasets
    for ds in era5_cache.values():
        if ds is not None:
            ds.close()
    
    print(f"\nMatched samples: {len(matched_samples)}")
    
    # Step 3: Create HDF5 file
    print("\n3. Creating verified HDF5 file...")
    
    n = len(matched_samples)
    
    # Prepare arrays
    cbh_km = np.array([s['cbh_km'] for s in matched_samples])
    flight_ids = np.array([s['flight_id'] for s in matched_samples])
    latitudes = np.array([s['lat'] for s in matched_samples])
    longitudes = np.array([s['lon'] for s in matched_samples])
    timestamps = np.array([s['timestamp'] for s in matched_samples])
    sample_ids = np.arange(n)
    
    # Geometric features
    sza_deg = np.array([s['sza_deg'] for s in matched_samples])
    saa_deg = np.array([s['saa_deg'] for s in matched_samples])
    
    # Atmospheric features
    t2m = np.array([s['t2m'] for s in matched_samples])
    d2m = np.array([s['d2m'] for s in matched_samples])
    blh = np.array([s['blh'] for s in matched_samples])
    sp = np.array([s['sp'] for s in matched_samples])
    tcwv = np.array([s['tcwv'] for s in matched_samples])
    lcl = np.array([s['lcl'] for s in matched_samples])
    stability_index = np.array([s['stability_index'] for s in matched_samples])
    moisture_gradient = np.array([s['moisture_gradient'] for s in matched_samples])
    inversion_height = np.array([s['inversion_height'] for s in matched_samples])
    
    # Write HDF5
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(OUTPUT_FILE, 'w') as f:
        # Metadata
        meta = f.create_group('metadata')
        meta.create_dataset('cbh_km', data=cbh_km)
        meta.create_dataset('flight_id', data=flight_ids)
        meta.create_dataset('latitude', data=latitudes)
        meta.create_dataset('longitude', data=longitudes)
        meta.create_dataset('timestamp', data=timestamps)
        meta.create_dataset('sample_id', data=sample_ids)
        
        # Flight mapping
        f.attrs['flight_mapping'] = json.dumps({v: k for k, v in FLIGHT_IDS.items()})
        f.attrs['created'] = datetime.now().isoformat()
        f.attrs['n_samples'] = n
        
        # Geometric features
        geo = f.create_group('geometric_features')
        geo.create_dataset('sza_deg', data=sza_deg)
        geo.create_dataset('saa_deg', data=saa_deg)
        
        # Atmospheric features
        atmo = f.create_group('atmospheric_features')
        atmo.create_dataset('t2m', data=t2m)
        atmo.create_dataset('d2m', data=d2m)
        atmo.create_dataset('blh', data=blh)
        atmo.create_dataset('sp', data=sp)
        atmo.create_dataset('tcwv', data=tcwv)
        atmo.create_dataset('lcl', data=lcl)
        atmo.create_dataset('stability_index', data=stability_index)
        atmo.create_dataset('moisture_gradient', data=moisture_gradient)
        atmo.create_dataset('inversion_height', data=inversion_height)
    
    print(f"\nSaved to: {OUTPUT_FILE}")
    
    # Step 4: Print summary statistics
    print("\n" + "=" * 80)
    print("DATASET SUMMARY")
    print("=" * 80)
    
    print(f"\nTotal samples: {n}")
    print(f"CBH range: [{cbh_km.min():.3f}, {cbh_km.max():.3f}] km")
    print(f"CBH mean: {cbh_km.mean():.3f} km, std: {cbh_km.std():.3f} km")
    
    print(f"\nSamples per flight:")
    for fid in np.unique(flight_ids):
        count = np.sum(flight_ids == fid)
        flight_name = {v: k for k, v in FLIGHT_IDS.items()}.get(fid, 'Unknown')
        print(f"  {flight_name}: {count}")
    
    print(f"\nAtmospheric features:")
    print(f"  t2m: [{t2m.min():.1f}, {t2m.max():.1f}] K (mean={t2m.mean():.1f})")
    print(f"  d2m: [{d2m.min():.1f}, {d2m.max():.1f}] K (mean={d2m.mean():.1f})")
    print(f"  blh: [{blh.min():.1f}, {blh.max():.1f}] m (mean={blh.mean():.1f})")
    print(f"  lcl: [{lcl.min():.1f}, {lcl.max():.1f}] m (mean={lcl.mean():.1f})")
    print(f"  tcwv: [{tcwv.min():.1f}, {tcwv.max():.1f}] kg/m² (mean={tcwv.mean():.1f})")
    
    print(f"\nGeometric features:")
    print(f"  sza: [{sza_deg.min():.1f}, {sza_deg.max():.1f}] deg")
    print(f"  saa: [{saa_deg.min():.1f}, {saa_deg.max():.1f}] deg")

if __name__ == '__main__':
    main()
