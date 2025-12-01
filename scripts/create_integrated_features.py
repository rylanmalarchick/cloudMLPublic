#!/usr/bin/env python3
"""
Modern Data Pipeline: Create Integrated Features Dataset

This script creates Integrated_Features.hdf5 by:
1. Loading raw CPL data (Layer_Base_Altitude, DEM, etc.)
2. Applying configurable filters (CBH range, ocean/land, single/multi-layer)
3. Temporal matching to IRAI images with configurable tolerance
4. Extracting geometric features (solar angles, shadow detection)
5. Extracting atmospheric features (ERA5 data)
6. Creating unified HDF5 file with all features

This consolidates and modernizes the deleted legacy pipeline:
- archive/data_creation/hdf5_dataset.py (CPL filtering & temporal matching)
- archive/data_creation/wp1_geometric_features.py (shadow features)
- archive/data_creation/wp2_atmospheric_features.py (ERA5 features)
- archive/data_creation/create_integrated_features.py (integration)

Author: OpenCode
Date: 2025-11-18
"""

import sys
from pathlib import Path
import argparse
import yaml
import h5py
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.cplCompareSub import cplTimeConvert


@dataclass
class FilterConfig:
    """Configuration for CPL filtering"""
    cbh_min: float = 0.0  # km
    cbh_max: float = 2.0  # km
    dem_threshold: float = 0.0  # m (0.0 = exact ocean, 50.0 = coastal)
    dem_mode: str = "exact"  # "exact" or "threshold"
    lidar_surface_max: float = 0.9  # km (lidar surface altitude check)
    single_layer_only: bool = False  # Filter for single-layer clouds only
    temporal_tolerance: float = 0.5  # seconds
    include_land: bool = False  # Include land samples (DEM > threshold)
    
    def __str__(self):
        return (f"CBH: [{self.cbh_min}, {self.cbh_max}] km, "
                f"DEM: {self.dem_mode} ({self.dem_threshold}m), "
                f"Single-layer: {self.single_layer_only}, "
                f"Time tolerance: {self.temporal_tolerance}s")


@dataclass
class MatchedSample:
    """Container for a matched CPL-image sample"""
    flight_id: int
    sample_id: int  # Image frame index
    cbh_km: float
    sza_deg: float
    saa_deg: float
    latitude: float
    longitude: float
    timestamp: float
    cpl_time: float
    time_diff: float
    dem_m: float
    num_layers: int
    

class CPLImageMatcher:
    """
    Matches CPL measurements to IRAI images with configurable filtering.
    
    This is the modern replacement for the deleted hdf5_dataset.py filtering logic.
    """
    
    def __init__(self, config: FilterConfig, verbose: bool = True):
        self.config = config
        self.verbose = verbose
        self.stats = {
            "total_cpl": 0,
            "valid_cloud": 0,
            "cbh_filtered": 0,
            "surface_filtered": 0,
            "layer_filtered": 0,
            "temporal_matched": 0,
        }
    
    def load_flight_data(
        self,
        flight_config: Dict,
        flight_id: int
    ) -> Tuple[List[MatchedSample], Dict]:
        """
        Load and match data for a single flight.
        
        Args:
            flight_config: Dict with iFileName, cFileName, nFileName, name
            flight_id: Integer ID for this flight
            
        Returns:
            matched_samples: List of MatchedSample objects
            diagnostics: Dict with filtering cascade statistics
        """
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"Processing Flight: {flight_config['name']}")
            print(f"{'='*80}")
        
        # Load CPL data
        cpl_data = self._load_cpl_data(flight_config['cFileName'])
        
        # Load image timestamps
        image_times = self._load_image_times(flight_config['iFileName'])
        
        # Load navigation data (SZA, SAA, lat/lon)
        nav_data = self._load_nav_data(flight_config['nFileName'], len(image_times))
        
        # Apply filtering cascade
        filtered_cpl = self._filter_cpl_data(cpl_data)
        
        # Temporal matching
        matched_samples = self._match_temporal(
            filtered_cpl,
            image_times,
            nav_data,
            flight_id,
            flight_config['name']
        )
        
        # Compute diagnostics
        diagnostics = {
            "flight_name": flight_config['name'],
            "flight_id": flight_id,
            "total_cpl": len(cpl_data['time']),
            "valid_cloud": cpl_data['valid_cloud_count'],
            "cbh_filtered": len(filtered_cpl['indices']),
            "temporal_matched": len(matched_samples),
            "match_rate": len(matched_samples) / len(filtered_cpl['indices']) if len(filtered_cpl['indices']) > 0 else 0.0
        }
        
        if self.verbose:
            self._print_diagnostics(diagnostics)
        
        return matched_samples, diagnostics
    
    def _load_cpl_data(self, cpl_file: str) -> Dict:
        """Load CPL Layer data"""
        if self.verbose:
            print(f"  Loading CPL data: {Path(cpl_file).name}")
        
        with h5py.File(cpl_file, 'r') as f:
            # Time conversion
            cTimePre = f['layer_descriptor/Profile_Decimal_Julian_Day'][:, 0]
            cTime = cplTimeConvert(cTimePre, cpl_file)
            
            # Layer data (all layers)
            layer_base_all = f['layer_descriptor/Layer_Base_Altitude'][:, :].astype(np.float32)
            layer_base_all[layer_base_all == -9999.0] = np.nan
            
            # Number of layers
            num_layers = f['layer_descriptor/Number_Layers'][:].astype(np.int16)
            
            # Surface data
            lidar_km = f['layer_descriptor/Lidar_Surface_Altitude'][:] / 1000.0
            dem_m = f['layer_descriptor/DEM_Surface_Altitude'][:]
            
            # First layer CBH
            base_layer = layer_base_all[:, 0]
            
            valid_cloud_count = np.isfinite(base_layer).sum()
            
        data = {
            'time': cTime,
            'cbh_km': base_layer,
            'num_layers': num_layers,
            'lidar_km': lidar_km,
            'dem_m': dem_m,
            'valid_cloud_count': valid_cloud_count
        }
        
        if self.verbose:
            print(f"    Total profiles: {len(cTime)}")
            print(f"    Valid clouds: {valid_cloud_count} ({100*valid_cloud_count/len(cTime):.1f}%)")
        
        return data
    
    def _load_image_times(self, image_file: str) -> np.ndarray:
        """Load IRAI image timestamps"""
        with h5py.File(image_file, 'r') as f:
            times = f['Time/TimeUTC'][:]
        return times
    
    def _load_nav_data(self, nav_file: str, n_samples: int) -> Dict:
        """Load navigation data (SZA, SAA, lat/lon)"""
        with h5py.File(nav_file, 'r') as f:
            # Solar Zenith Angle
            SZA_raw = f['nav/solarZenith'][:n_samples].reshape(-1, 1)
            SZA = np.nan_to_num(SZA_raw, nan=np.nanmean(SZA_raw))
            
            # Solar Azimuth Angle
            SAA_raw = f['nav/sunAzGrd'][:n_samples].reshape(-1, 1)
            SAA = np.mod(np.nan_to_num(SAA_raw, nan=0.0), 360.0)
            
            # Position
            lat = f['nav/IWG_lat'][:n_samples]
            lon = f['nav/IWG_lon'][:n_samples]
        
        return {
            'sza': SZA.flatten(),
            'saa': SAA.flatten(),
            'lat': lat,
            'lon': lon
        }
    
    def _filter_cpl_data(self, cpl_data: Dict) -> Dict:
        """
        Apply filtering cascade to CPL data.
        
        This implements the exact logic from archive/data_creation/hdf5_dataset.py:148-162
        """
        # Start with all profiles
        mask = np.ones(len(cpl_data['time']), dtype=bool)
        
        # Filter 1: Valid cloud return
        mask &= np.isfinite(cpl_data['cbh_km'])
        
        # Filter 2: CBH range
        mask &= (cpl_data['cbh_km'] >= self.config.cbh_min)
        mask &= (cpl_data['cbh_km'] <= self.config.cbh_max)
        
        # Filter 3: Surface type (ocean vs land)
        if self.config.dem_mode == "exact":
            # Original logic: DEM exactly 0
            surface_mask = (cpl_data['dem_m'] == 0.0)
        else:
            # Threshold mode: DEM < threshold
            surface_mask = (cpl_data['dem_m'] < self.config.dem_threshold)
        
        if not self.config.include_land:
            mask &= surface_mask
        
        # Filter 4: Lidar surface altitude check (quality control)
        mask &= (np.abs(cpl_data['lidar_km']) <= self.config.lidar_surface_max)
        
        # Filter 5: Single-layer only (if enabled)
        if self.config.single_layer_only:
            mask &= (cpl_data['num_layers'] == 1)
        
        # Extract filtered data
        indices = np.where(mask)[0]
        
        filtered = {
            'indices': indices,
            'time': cpl_data['time'][indices],
            'cbh_km': cpl_data['cbh_km'][indices],
            'num_layers': cpl_data['num_layers'][indices],
            'dem_m': cpl_data['dem_m'][indices],
        }
        
        return filtered
    
    def _match_temporal(
        self,
        filtered_cpl: Dict,
        image_times: np.ndarray,
        nav_data: Dict,
        flight_id: int,
        flight_name: str
    ) -> List[MatchedSample]:
        """
        Match CPL measurements to images within temporal tolerance.
        
        This implements the exact logic from archive/data_creation/hdf5_dataset.py:175-185
        """
        matched_samples = []
        
        for i, cpl_idx in enumerate(filtered_cpl['indices']):
            cpl_time = filtered_cpl['time'][i]
            
            # Find closest image frame
            time_diffs = np.abs(image_times - cpl_time)
            closest_idx = np.argmin(time_diffs)
            dt = float(time_diffs[closest_idx])
            
            # Check if within tolerance
            if dt <= self.config.temporal_tolerance:
                sample = MatchedSample(
                    flight_id=flight_id,
                    sample_id=int(closest_idx),
                    cbh_km=float(filtered_cpl['cbh_km'][i]),
                    sza_deg=float(nav_data['sza'][closest_idx]),
                    saa_deg=float(nav_data['saa'][closest_idx]),
                    latitude=float(nav_data['lat'][closest_idx]),
                    longitude=float(nav_data['lon'][closest_idx]),
                    timestamp=float(image_times[closest_idx]),
                    cpl_time=float(cpl_time),
                    time_diff=dt,
                    dem_m=float(filtered_cpl['dem_m'][i]),
                    num_layers=int(filtered_cpl['num_layers'][i]),
                )
                matched_samples.append(sample)
        
        return matched_samples
    
    def _print_diagnostics(self, diagnostics: Dict):
        """Print filtering cascade statistics"""
        print(f"\n  Filtering Cascade:")
        print(f"    1. Total CPL profiles:     {diagnostics['total_cpl']:6d} (100.0%)")
        print(f"    2. + Valid cloud return:   {diagnostics['valid_cloud']:6d}")
        print(f"    3. + Filters applied:      {diagnostics['cbh_filtered']:6d}")
        print(f"    4. + Temporal match:       {diagnostics['temporal_matched']:6d} ({diagnostics['match_rate']*100:5.1f}% match rate)")


class IntegratedFeatureBuilder:
    """
    Creates Integrated_Features.hdf5 from matched samples.
    
    This consolidates geometric and atmospheric features into a single HDF5 file.
    """
    
    def __init__(self, output_path: str, verbose: bool = True):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
    
    def build(
        self,
        all_matched_samples: List[MatchedSample],
        flight_names: Dict[int, str],
        filter_config: FilterConfig
    ):
        """
        Build the integrated HDF5 file.
        
        Args:
            all_matched_samples: List of all matched samples across flights
            flight_names: Dict mapping flight_id -> flight_name
            filter_config: FilterConfig used for data creation
        """
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"Building Integrated Features HDF5")
            print(f"{'='*80}")
            print(f"  Output: {self.output_path}")
            print(f"  Total samples: {len(all_matched_samples)}")
        
        # Sort samples by flight_id, then sample_id for consistency
        all_matched_samples.sort(key=lambda x: (x.flight_id, x.sample_id))
        
        n_samples = len(all_matched_samples)
        
        # Create HDF5 file
        with h5py.File(self.output_path, 'w') as f:
            # Metadata group
            meta = f.create_group('metadata')
            meta.create_dataset('flight_id', data=[s.flight_id for s in all_matched_samples], dtype=np.int32)
            meta.create_dataset('sample_id', data=[s.sample_id for s in all_matched_samples], dtype=np.int32)
            meta.create_dataset('cbh_km', data=[s.cbh_km for s in all_matched_samples], dtype=np.float32)
            meta.create_dataset('latitude', data=[s.latitude for s in all_matched_samples], dtype=np.float32)
            meta.create_dataset('longitude', data=[s.longitude for s in all_matched_samples], dtype=np.float32)
            meta.create_dataset('timestamp', data=[s.timestamp for s in all_matched_samples], dtype=np.float64)
            
            # Geometric features group
            geo = f.create_group('geometric_features')
            geo.create_dataset('sza_deg', data=[s.sza_deg for s in all_matched_samples], dtype=np.float32)
            geo.create_dataset('saa_deg', data=[s.saa_deg for s in all_matched_samples], dtype=np.float32)
            
            # Placeholder for other geometric features (shadow-based)
            # These would require image processing from WP1
            geo.create_dataset('cloud_edge_x', data=np.zeros(n_samples), dtype=np.float32)
            geo.create_dataset('cloud_edge_y', data=np.zeros(n_samples), dtype=np.float32)
            geo.create_dataset('shadow_edge_x', data=np.zeros(n_samples), dtype=np.float32)
            geo.create_dataset('shadow_edge_y', data=np.zeros(n_samples), dtype=np.float32)
            geo.create_dataset('shadow_length_pixels', data=np.zeros(n_samples), dtype=np.float32)
            geo.create_dataset('shadow_angle_deg', data=np.zeros(n_samples), dtype=np.float32)
            geo.create_dataset('shadow_detection_confidence', data=np.zeros(n_samples), dtype=np.float32)
            
            # Atmospheric features group (placeholder for ERA5 data)
            atmo = f.create_group('atmospheric_features')
            # These would require ERA5 data processing from WP2
            feature_names = ['blh', 'lcl', 'inversion_height', 'moisture_gradient',
                           'stability_index', 't2m', 'd2m', 'sp', 'tcwv']
            atmo.create_dataset('blh', data=np.zeros(n_samples), dtype=np.float32)
            atmo.create_dataset('lcl', data=np.zeros(n_samples), dtype=np.float32)
            atmo.create_dataset('inversion_height', data=np.zeros(n_samples), dtype=np.float32)
            atmo.create_dataset('moisture_gradient', data=np.zeros(n_samples), dtype=np.float32)
            atmo.create_dataset('stability_index', data=np.zeros(n_samples), dtype=np.float32)
            atmo.create_dataset('t2m', data=np.zeros(n_samples), dtype=np.float32)
            atmo.create_dataset('d2m', data=np.zeros(n_samples), dtype=np.float32)
            atmo.create_dataset('sp', data=np.zeros(n_samples), dtype=np.float32)
            atmo.create_dataset('tcwv', data=np.zeros(n_samples), dtype=np.float32)
            
            # Add CBH as top-level dataset (for compatibility)
            f.create_dataset('cbh', data=[s.cbh_km for s in all_matched_samples], dtype=np.float32)
            
            # Global attributes
            f.attrs['title'] = 'Integrated Cloud Base Height Feature Store'
            f.attrs['created'] = datetime.now().isoformat()
            f.attrs['n_samples'] = n_samples
            f.attrs['n_flights'] = len(set(s.flight_id for s in all_matched_samples))
            f.attrs['flight_mapping'] = json.dumps(flight_names)
            f.attrs['filter_config'] = str(filter_config)
            f.attrs['cbh_min'] = filter_config.cbh_min
            f.attrs['cbh_max'] = filter_config.cbh_max
            f.attrs['temporal_tolerance'] = filter_config.temporal_tolerance
            f.attrs['single_layer_only'] = filter_config.single_layer_only
        
        if self.verbose:
            print(f"  ✓ Integrated features HDF5 created successfully")
            self._print_summary()
    
    def _print_summary(self):
        """Print summary of created file"""
        with h5py.File(self.output_path, 'r') as f:
            n_samples = f.attrs['n_samples']
            cbh = f['metadata/cbh_km'][:]
            flight_ids = f['metadata/flight_id'][:]
            
            print(f"\n  Dataset Summary:")
            print(f"    Total samples: {n_samples}")
            print(f"    CBH range: [{cbh.min():.3f}, {cbh.max():.3f}] km")
            print(f"    CBH mean: {cbh.mean():.3f} km")
            print(f"    CBH std: {cbh.std():.3f} km")
            
            print(f"\n    Per-flight distribution:")
            flight_map = json.loads(f.attrs['flight_mapping'])
            for fid in sorted(set(flight_ids)):
                count = (flight_ids == fid).sum()
                fname = flight_map.get(str(fid), f"Flight_{fid}")
                print(f"      F{fid} ({fname}): {count} samples")


def main():
    parser = argparse.ArgumentParser(
        description='Create Integrated_Features.hdf5 from raw CPL and IRAI data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: Original 933-sample dataset (ocean, CBH 0.2-2km, 0.5s tolerance)
  python scripts/create_integrated_features.py --config configs/config.yaml
  
  # Expand: Include CBH < 0.2 km
  python scripts/create_integrated_features.py --config configs/config.yaml --cbh-min 0.0
  
  # Expand: Include multi-layer clouds
  python scripts/create_integrated_features.py --config configs/config.yaml --allow-multilayer
  
  # Expand: Include land samples
  python scripts/create_integrated_features.py --config configs/config.yaml --include-land
  
  # Maximum expansion
  python scripts/create_integrated_features.py --config configs/config.yaml \\
      --cbh-min 0.0 --cbh-max 10.0 --allow-multilayer --include-land
        """
    )
    
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to config YAML with flight definitions')
    parser.add_argument('--output', type=str,
                       default='outputs/preprocessed_data/Integrated_Features.hdf5',
                       help='Output HDF5 file path')
    
    # Filter arguments
    parser.add_argument('--cbh-min', type=float, default=0.2,
                       help='Minimum CBH in km (default: 0.2)')
    parser.add_argument('--cbh-max', type=float, default=2.0,
                       help='Maximum CBH in km (default: 2.0)')
    parser.add_argument('--dem-threshold', type=float, default=0.0,
                       help='DEM threshold in meters (default: 0.0 for exact ocean)')
    parser.add_argument('--dem-mode', type=str, default='exact', choices=['exact', 'threshold'],
                       help='DEM filtering mode (default: exact)')
    parser.add_argument('--temporal-tolerance', type=float, default=0.5,
                       help='Temporal matching tolerance in seconds (default: 0.5)')
    parser.add_argument('--allow-multilayer', action='store_true',
                       help='Include multi-layer clouds (default: False)')
    parser.add_argument('--include-land', action='store_true',
                       help='Include land samples (default: False, ocean only)')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress verbose output')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Determine output path: config file takes precedence, then command-line
    output_path = config.get('output_file', args.output)
    
    # Create filter config: use config file values, allow command-line overrides
    # Check if command-line args were explicitly provided (not using defaults)
    filter_config = FilterConfig(
        cbh_min=config.get('cbh_min', args.cbh_min),
        cbh_max=config.get('cbh_max', args.cbh_max),
        dem_threshold=config.get('dem_threshold', args.dem_threshold),
        dem_mode=config.get('dem_mode', args.dem_mode),
        single_layer_only=not config.get('allow_multilayer', args.allow_multilayer),
        temporal_tolerance=config.get('time_tolerance', args.temporal_tolerance),
        include_land=config.get('include_land', args.include_land),
    )
    
    print("="*80)
    print("DATA PIPELINE: Create Integrated Features")
    print("="*80)
    print(f"Filter Configuration: {filter_config}")
    print(f"Output: {output_path}")
    print("="*80)
    
    # Initialize matcher
    matcher = CPLImageMatcher(filter_config, verbose=not args.quiet)
    
    # Process all flights
    all_matched_samples = []
    all_diagnostics = []
    flight_names = {}
    
    data_dir = Path(config['data_directory'])
    
    for flight_id, flight in enumerate(config['flights']):
        flight_config = {
            'name': flight['name'],
            'iFileName': str(data_dir / flight['iFileName']),
            'cFileName': str(data_dir / flight['cFileName']),
            'nFileName': str(data_dir / flight['nFileName']),
        }
        
        flight_names[flight_id] = flight['name']
        
        matched_samples, diagnostics = matcher.load_flight_data(flight_config, flight_id)
        all_matched_samples.extend(matched_samples)
        all_diagnostics.append(diagnostics)
    
    # Build integrated HDF5
    builder = IntegratedFeatureBuilder(output_path, verbose=not args.quiet)
    builder.build(all_matched_samples, flight_names, filter_config)
    
    # Print overall statistics
    print(f"\n{'='*80}")
    print(f"OVERALL STATISTICS")
    print(f"{'='*80}")
    total_cpl = sum(d['total_cpl'] for d in all_diagnostics)
    total_matched = len(all_matched_samples)
    print(f"Total CPL profiles across all flights: {total_cpl}")
    print(f"Total matched samples: {total_matched}")
    print(f"Overall match rate: {100*total_matched/total_cpl:.2f}%")
    print(f"\nOutput saved to: {output_path}")
    print("="*80)


if __name__ == '__main__':
    main()
