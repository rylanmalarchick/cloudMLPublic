#!/usr/bin/env python3
"""
Integrate ERA5 Atmospheric Features into HDF5

This script reads ERA5 NetCDF files and integrates atmospheric features
into the Integrated_Features.hdf5 file by:
1. Loading sample lat/lon/time from the HDF5 metadata
2. Finding the nearest ERA5 grid point and time for each sample
3. Extracting ERA5 variables (t2m, d2m, sp, blh, tcwv)
4. Computing derived features (LCL, stability_index, etc.)
5. Writing features back to the HDF5 file

Author: OpenCode
Date: 2025-01-06
"""

import sys
from pathlib import Path
from datetime import datetime
import argparse
import json

import h5py
import numpy as np
import netCDF4 as nc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Default paths
ERA5_ROOT = Path("/media/rylan/two/research/NASA/ERA5_data_root")
HDF5_PATH = PROJECT_ROOT / "outputs/preprocessed_data/Integrated_Features.hdf5"


def compute_lcl_height(t2m_k, d2m_k):
    """
    Compute Lifting Condensation Level (LCL) height using Espy's equation.
    
    LCL height (m) ≈ 125 * (T - Td)
    
    Where T and Td are in Celsius.
    
    Args:
        t2m_k: 2m temperature in Kelvin
        d2m_k: 2m dewpoint temperature in Kelvin
        
    Returns:
        lcl_m: LCL height in meters
    """
    t2m_c = t2m_k - 273.15
    d2m_c = d2m_k - 273.15
    lcl_m = 125.0 * (t2m_c - d2m_c)
    return np.clip(lcl_m, 0, 15000)  # Reasonable range


def compute_stability_index(t2m_k, d2m_k, sp_pa):
    """
    Compute a simple stability index based on dewpoint depression.
    
    Higher values indicate more unstable conditions (larger T-Td spread).
    
    Args:
        t2m_k: 2m temperature in Kelvin
        d2m_k: 2m dewpoint temperature in Kelvin
        sp_pa: Surface pressure in Pa
        
    Returns:
        stability_index: Dimensionless stability index
    """
    # Dewpoint depression (proxy for instability)
    dd = t2m_k - d2m_k
    
    # Normalize by a typical value
    stability_index = dd / 10.0  # Typical dd is 5-15K
    
    return stability_index


def compute_moisture_gradient(d2m_k, tcwv_kg_m2):
    """
    Compute a moisture gradient proxy from dewpoint and total column water vapor.
    
    Args:
        d2m_k: 2m dewpoint temperature in Kelvin
        tcwv_kg_m2: Total column water vapor in kg/m²
        
    Returns:
        moisture_gradient: Moisture gradient proxy
    """
    # Normalize both and combine
    d2m_norm = (d2m_k - 250) / 30  # Typical range 250-280K
    tcwv_norm = tcwv_kg_m2 / 30  # Typical range 0-60 kg/m²
    
    moisture_gradient = d2m_norm + tcwv_norm
    return moisture_gradient


def find_nearest_era5(lat, lon, timestamp, era5_data):
    """
    Find nearest ERA5 grid point and time for a sample.
    
    Args:
        lat: Sample latitude
        lon: Sample longitude  
        timestamp: Sample timestamp (Unix time)
        era5_data: Dict with ERA5 file data
        
    Returns:
        Dict with extracted ERA5 values
    """
    era5_lats = era5_data['latitude']
    era5_lons = era5_data['longitude']
    era5_times = era5_data['valid_time']
    
    # Find nearest lat/lon indices
    lat_idx = np.argmin(np.abs(era5_lats - lat))
    
    # Handle longitude wrapping (ERA5 uses 0-360 or -180 to 180)
    if era5_lons.min() >= 0:
        # ERA5 uses 0-360
        lon_query = lon if lon >= 0 else lon + 360
    else:
        lon_query = lon
    lon_idx = np.argmin(np.abs(era5_lons - lon_query))
    
    # Find nearest time index
    time_idx = np.argmin(np.abs(era5_times - timestamp))
    
    # Extract values
    result = {}
    for var in ['blh', 't2m', 'd2m', 'sp', 'tcwv']:
        if var in era5_data:
            data = era5_data[var]
            if data.ndim == 3:
                result[var] = float(data[time_idx, lat_idx, lon_idx])
            else:
                result[var] = float(data[lat_idx, lon_idx])
    
    return result


def load_era5_for_date(date_str, era5_root):
    """
    Load ERA5 data for a specific date.
    
    Args:
        date_str: Date string in format YYYYMMDD
        era5_root: Path to ERA5 data root
        
    Returns:
        Dict with ERA5 data arrays
    """
    surface_file = era5_root / "surface" / f"era5_surface_{date_str}.nc"
    
    if not surface_file.exists():
        return None
    
    ds = nc.Dataset(surface_file, 'r')
    
    data = {
        'latitude': ds.variables['latitude'][:],
        'longitude': ds.variables['longitude'][:],
        'valid_time': ds.variables['valid_time'][:],
        'blh': ds.variables['blh'][:],
        't2m': ds.variables['t2m'][:],
        'd2m': ds.variables['d2m'][:],
        'sp': ds.variables['sp'][:],
        'tcwv': ds.variables['tcwv'][:],
    }
    
    ds.close()
    return data


def integrate_era5_features(hdf5_path, era5_root, verbose=True):
    """
    Main function to integrate ERA5 features into HDF5.
    
    Args:
        hdf5_path: Path to Integrated_Features.hdf5
        era5_root: Path to ERA5 data root
        verbose: Print progress
    """
    if verbose:
        print("=" * 80)
        print("ERA5 Feature Integration")
        print("=" * 80)
        print(f"HDF5 file: {hdf5_path}")
        print(f"ERA5 root: {era5_root}")
    
    # Load sample metadata from HDF5
    with h5py.File(hdf5_path, 'r') as f:
        latitudes = f['metadata/latitude'][:]
        longitudes = f['metadata/longitude'][:]
        timestamps = f['metadata/timestamp'][:]
        n_samples = len(latitudes)
    
    if verbose:
        print(f"\nLoaded {n_samples} samples from HDF5")
        print(f"Lat range: [{latitudes.min():.4f}, {latitudes.max():.4f}]")
        print(f"Lon range: [{longitudes.min():.4f}, {longitudes.max():.4f}]")
    
    # Initialize feature arrays
    blh = np.zeros(n_samples, dtype=np.float32)
    t2m = np.zeros(n_samples, dtype=np.float32)
    d2m = np.zeros(n_samples, dtype=np.float32)
    sp = np.zeros(n_samples, dtype=np.float32)
    tcwv = np.zeros(n_samples, dtype=np.float32)
    
    # Cache for loaded ERA5 files
    era5_cache = {}
    
    # Process each sample
    if verbose:
        print("\nProcessing samples...")
    
    success_count = 0
    for i in range(n_samples):
        lat = latitudes[i]
        lon = longitudes[i]
        ts = timestamps[i]
        
        # Convert timestamp to date string
        dt = datetime.utcfromtimestamp(ts)
        date_str = dt.strftime("%Y%m%d")
        
        # Load ERA5 data for this date (cached)
        if date_str not in era5_cache:
            era5_data = load_era5_for_date(date_str, era5_root)
            era5_cache[date_str] = era5_data
            if verbose and era5_data is not None:
                print(f"  Loaded ERA5 for {date_str}")
        else:
            era5_data = era5_cache[date_str]
        
        if era5_data is None:
            if verbose and i < 10:
                print(f"  Sample {i}: No ERA5 data for {date_str}")
            continue
        
        # Find nearest ERA5 values
        try:
            values = find_nearest_era5(lat, lon, ts, era5_data)
            blh[i] = values.get('blh', 0)
            t2m[i] = values.get('t2m', 0)
            d2m[i] = values.get('d2m', 0)
            sp[i] = values.get('sp', 0)
            tcwv[i] = values.get('tcwv', 0)
            success_count += 1
        except Exception as e:
            if verbose and i < 10:
                print(f"  Sample {i}: Error - {e}")
    
    if verbose:
        print(f"\nSuccessfully extracted ERA5 for {success_count}/{n_samples} samples")
    
    # Compute derived features
    if verbose:
        print("\nComputing derived features...")
    
    lcl = compute_lcl_height(t2m, d2m)
    stability_index = compute_stability_index(t2m, d2m, sp)
    moisture_gradient = compute_moisture_gradient(d2m, tcwv)
    
    # Inversion height proxy (simple: scale of BLH)
    inversion_height = blh * 1.2  # Simple approximation
    
    if verbose:
        print(f"  LCL range: [{lcl.min():.1f}, {lcl.max():.1f}] m")
        print(f"  t2m range: [{t2m.min():.1f}, {t2m.max():.1f}] K")
        print(f"  d2m range: [{d2m.min():.1f}, {d2m.max():.1f}] K")
        print(f"  BLH range: [{blh.min():.1f}, {blh.max():.1f}] m")
    
    # Write to HDF5
    if verbose:
        print("\nWriting to HDF5...")
    
    with h5py.File(hdf5_path, 'r+') as f:
        atmo = f['atmospheric_features']
        
        # Update existing datasets
        atmo['blh'][...] = blh
        atmo['t2m'][...] = t2m
        atmo['d2m'][...] = d2m
        atmo['sp'][...] = sp
        atmo['tcwv'][...] = tcwv
        atmo['lcl'][...] = lcl
        atmo['stability_index'][...] = stability_index
        atmo['moisture_gradient'][...] = moisture_gradient
        atmo['inversion_height'][...] = inversion_height
        
        # Update metadata
        f.attrs['era5_integrated'] = True
        f.attrs['era5_integration_date'] = datetime.now().isoformat()
        f.attrs['era5_samples_success'] = success_count
    
    if verbose:
        print(f"  Updated {success_count} samples with ERA5 features")
        print("=" * 80)
        print("ERA5 Integration Complete!")
        print("=" * 80)
    
    return success_count


def main():
    parser = argparse.ArgumentParser(
        description='Integrate ERA5 atmospheric features into HDF5'
    )
    parser.add_argument('--hdf5', type=str, default=str(HDF5_PATH),
                       help='Path to Integrated_Features.hdf5')
    parser.add_argument('--era5-root', type=str, default=str(ERA5_ROOT),
                       help='Path to ERA5 data root')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress verbose output')
    
    args = parser.parse_args()
    
    hdf5_path = Path(args.hdf5)
    era5_root = Path(args.era5_root)
    
    if not hdf5_path.exists():
        print(f"ERROR: HDF5 file not found: {hdf5_path}")
        sys.exit(1)
    
    if not era5_root.exists():
        print(f"ERROR: ERA5 root not found: {era5_root}")
        sys.exit(1)
    
    success = integrate_era5_features(hdf5_path, era5_root, verbose=not args.quiet)
    
    if success == 0:
        print("WARNING: No samples were successfully integrated!")
        sys.exit(1)


if __name__ == '__main__':
    main()
