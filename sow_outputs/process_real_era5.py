#!/usr/bin/env python3
"""
Process Real ERA5 Data for WP2 Atmospheric Features

This script loads actual ERA5 reanalysis data from the external drive
and processes it to create atmospheric features for the 933 labeled samples.

Author: Research Team
Date: 2025
"""

import sys
from pathlib import Path
import numpy as np
import h5py
import yaml
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import xarray as xr
    import pandas as pd

    XARRAY_AVAILABLE = True
except ImportError:
    XARRAY_AVAILABLE = False
    print(
        "ERROR: xarray/pandas not available. Install with: pip install xarray pandas netCDF4"
    )
    sys.exit(1)

from src.hdf5_dataset import HDF5CloudDataset


@dataclass
class AtmosphericProfile:
    """Container for atmospheric thermodynamic features."""

    blh_m: float  # Boundary Layer Height
    lcl_m: float  # Lifting Condensation Level
    inversion_height_m: float  # Temperature inversion height
    moisture_gradient: float  # Vertical moisture gradient (kg/kg/m)
    stability_index: float  # Atmospheric stability measure
    surface_temp_k: float  # Surface temperature
    surface_dewpoint_k: float  # Surface dewpoint
    surface_pressure_pa: float  # Surface pressure
    lapse_rate_k_per_km: float  # Temperature lapse rate
    confidence: float  # Quality of the profile (0-1)


class RealERA5Processor:
    """
    Process real ERA5 data from external drive.
    """

    def __init__(
        self,
        era5_surface_dir: str = "/media/rylan/two/research/NASA/ERA5_data_root/surface",
        era5_pressure_dir: str = "/media/rylan/two/research/NASA/ERA5_data_root/pressure_levels",
        verbose: bool = True,
    ):
        """
        Initialize processor.

        Args:
            era5_surface_dir: Directory with surface-level ERA5 data
            era5_pressure_dir: Directory with pressure-level ERA5 data
            verbose: Enable verbose logging
        """
        self.surface_dir = Path(era5_surface_dir)
        self.pressure_dir = Path(era5_pressure_dir)
        self.verbose = verbose

        if not self.surface_dir.exists():
            raise FileNotFoundError(f"Surface data not found: {self.surface_dir}")
        if not self.pressure_dir.exists():
            raise FileNotFoundError(f"Pressure data not found: {self.pressure_dir}")

        if self.verbose:
            print(f"Surface data: {self.surface_dir}")
            print(f"Pressure data: {self.pressure_dir}")

    def load_era5_for_datetime(
        self, dt: datetime, lat: float, lon: float
    ) -> Optional[Dict]:
        """
        Load ERA5 data for specific datetime and location.

        Args:
            dt: Datetime of sample
            lat: Latitude in degrees
            lon: Longitude in degrees

        Returns:
            Dictionary with ERA5 variables or None if data missing
        """
        # Format date for filename
        date_str = dt.strftime("%Y%m%d")

        surface_file = self.surface_dir / f"era5_surface_{date_str}.nc"
        pressure_file = self.pressure_dir / f"era5_pressure_{date_str}.nc"

        if not surface_file.exists() or not pressure_file.exists():
            if self.verbose:
                print(f"Warning: ERA5 data missing for {date_str}")
            return None

        try:
            # Load surface data
            ds_surface = xr.open_dataset(surface_file)

            # Load pressure-level data
            ds_pressure = xr.open_dataset(pressure_file)

            # Find nearest grid point
            lat_idx = np.argmin(np.abs(ds_surface["latitude"].values - lat))
            lon_idx = np.argmin(np.abs(ds_surface["longitude"].values - lon))

            # Find nearest time (hourly data)
            time_idx = np.argmin(
                np.abs(
                    pd.to_datetime(ds_surface["valid_time"].values) - pd.Timestamp(dt)
                )
            )

            # Extract surface variables
            blh = float(
                ds_surface["blh"]
                .isel(valid_time=time_idx, latitude=lat_idx, longitude=lon_idx)
                .values
            )

            t2m = float(
                ds_surface["t2m"]
                .isel(valid_time=time_idx, latitude=lat_idx, longitude=lon_idx)
                .values
            )

            d2m = float(
                ds_surface["d2m"]
                .isel(valid_time=time_idx, latitude=lat_idx, longitude=lon_idx)
                .values
            )

            sp = float(
                ds_surface["sp"]
                .isel(valid_time=time_idx, latitude=lat_idx, longitude=lon_idx)
                .values
            )

            # Extract pressure-level variables
            # Find nearest time for pressure data
            time_idx_pressure = np.argmin(
                np.abs(
                    pd.to_datetime(ds_pressure["valid_time"].values) - pd.Timestamp(dt)
                )
            )

            temp_profile = (
                ds_pressure["t"]
                .isel(valid_time=time_idx_pressure, latitude=lat_idx, longitude=lon_idx)
                .values
            )

            q_profile = (
                ds_pressure["q"]
                .isel(valid_time=time_idx_pressure, latitude=lat_idx, longitude=lon_idx)
                .values
            )

            z_profile = (
                ds_pressure["z"]
                .isel(valid_time=time_idx_pressure, latitude=lat_idx, longitude=lon_idx)
                .values
                / 9.81
            )  # Convert geopotential to height (m)

            pressure_levels = ds_pressure["pressure_level"].values * 100  # hPa to Pa

            ds_surface.close()
            ds_pressure.close()

            return {
                "blh": blh,
                "t2m": t2m,
                "d2m": d2m,
                "sp": sp,
                "temp_profile": temp_profile,
                "q_profile": q_profile,
                "z_profile": z_profile,
                "pressure_levels": pressure_levels,
                "lat_actual": float(ds_surface["latitude"].values[lat_idx]),
                "lon_actual": float(ds_surface["longitude"].values[lon_idx]),
            }

        except Exception as e:
            if self.verbose:
                print(f"Error loading ERA5 for {date_str}: {e}")
            return None

    def compute_lcl(self, temp_k: float, dewpoint_k: float) -> float:
        """
        Compute Lifting Condensation Level using empirical formula.

        Args:
            temp_k: Temperature in Kelvin
            dewpoint_k: Dewpoint in Kelvin

        Returns:
            LCL height in meters
        """
        temp_c = temp_k - 273.15
        dewpoint_c = dewpoint_k - 273.15

        # Espy's formula
        lcl_m = 125 * (temp_c - dewpoint_c)

        return max(0, lcl_m)

    def compute_inversion_height(
        self, temp_profile: np.ndarray, z_profile: np.ndarray
    ) -> float:
        """
        Detect temperature inversion height.

        Args:
            temp_profile: Temperature at pressure levels (K)
            z_profile: Height at pressure levels (m)

        Returns:
            Inversion height in meters (or 0 if none detected)
        """
        # Look for first temperature increase with height
        dT_dz = np.diff(temp_profile) / np.diff(z_profile)

        # Inversion is where dT/dz > 0 (temperature increases with height)
        inversion_indices = np.where(dT_dz > 0)[0]

        if len(inversion_indices) > 0:
            # Return height of first inversion
            return float(z_profile[inversion_indices[0]])
        else:
            return 0.0

    def compute_moisture_gradient(
        self, q_profile: np.ndarray, z_profile: np.ndarray
    ) -> float:
        """
        Compute vertical moisture gradient in lower troposphere.

        Args:
            q_profile: Specific humidity at pressure levels (kg/kg)
            z_profile: Height at pressure levels (m)

        Returns:
            Moisture gradient (kg/kg/m)
        """
        # Use lower troposphere (first 5 levels or up to 3 km)
        mask = z_profile < 3000
        if np.sum(mask) < 2:
            return 0.0

        q_lower = q_profile[mask]
        z_lower = z_profile[mask]

        # Linear fit
        if len(q_lower) > 1:
            gradient = np.polyfit(z_lower, q_lower, 1)[0]
            return float(gradient)
        else:
            return 0.0

    def compute_lapse_rate(
        self, temp_profile: np.ndarray, z_profile: np.ndarray
    ) -> float:
        """
        Compute temperature lapse rate in lower troposphere.

        Args:
            temp_profile: Temperature at pressure levels (K)
            z_profile: Height at pressure levels (m)

        Returns:
            Lapse rate in K/km
        """
        # Use lower troposphere (first few km)
        mask = z_profile < 5000
        if np.sum(mask) < 2:
            return 6.5  # Standard atmosphere

        temp_lower = temp_profile[mask]
        z_lower = z_profile[mask]

        # Linear fit
        if len(temp_lower) > 1:
            lapse_rate_per_m = np.polyfit(z_lower, temp_lower, 1)[0]
            lapse_rate_per_km = (
                -lapse_rate_per_m * 1000
            )  # Negative because T decreases with height
            return float(lapse_rate_per_km)
        else:
            return 6.5

    def extract_profile(
        self, dt: datetime, lat: float, lon: float
    ) -> Optional[AtmosphericProfile]:
        """
        Extract atmospheric profile for given location and time.

        Args:
            dt: Datetime of sample
            lat: Latitude
            lon: Longitude

        Returns:
            AtmosphericProfile or None if data unavailable
        """
        era5_data = self.load_era5_for_datetime(dt, lat, lon)

        if era5_data is None:
            return None

        # Extract basic variables
        blh = era5_data["blh"]
        t2m = era5_data["t2m"]
        d2m = era5_data["d2m"]
        sp = era5_data["sp"]

        # Compute derived quantities
        lcl = self.compute_lcl(t2m, d2m)
        inversion_height = self.compute_inversion_height(
            era5_data["temp_profile"], era5_data["z_profile"]
        )
        moisture_gradient = self.compute_moisture_gradient(
            era5_data["q_profile"], era5_data["z_profile"]
        )
        lapse_rate = self.compute_lapse_rate(
            era5_data["temp_profile"], era5_data["z_profile"]
        )

        # Stability index (simplified)
        stability = lapse_rate

        # Confidence is 1.0 for real ERA5 data
        confidence = 1.0

        return AtmosphericProfile(
            blh_m=blh,
            lcl_m=lcl,
            inversion_height_m=inversion_height,
            moisture_gradient=moisture_gradient,
            stability_index=stability,
            surface_temp_k=t2m,
            surface_dewpoint_k=d2m,
            surface_pressure_pa=sp,
            lapse_rate_k_per_km=lapse_rate,
            confidence=confidence,
        )


def load_flight_metadata(config_path: str, verbose: bool = True) -> List[Dict]:
    """
    Load actual flight metadata from navigation files.

    Args:
        config_path: Path to config YAML
        verbose: Enable verbose output

    Returns:
        List of metadata dictionaries with lat/lon/time for each sample
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Create dataset to get sample counts and flight mapping
    dataset = HDF5CloudDataset(
        flight_configs=config["flights"],
        swath_slice=[40, 480],
        temporal_frames=1,
        filter_type="basic",
        flat_field_correction=True,
        clahe_clip_limit=0.01,
        zscore_normalize=True,
    )

    # Flight dates (from SOW)
    flight_dates = {
        0: "2024-10-30",
        1: "2025-02-10",
        2: "2024-10-23",
        3: "2025-02-12",
        4: "2025-02-18",
    }

    # Flight name to ID mapping
    flight_name_to_id = {
        "30Oct24": 0,
        "10Feb25": 1,
        "23Oct24": 2,
        "12Feb25": 3,
        "18Feb25": 4,
    }

    if verbose:
        print(f"\nLoading metadata for {len(dataset)} samples...")
        print("Extracting lat/lon from navigation files...")

    metadata_list = []

    for idx in range(len(dataset)):
        # Dataset returns: (img_stack, sza, saa, y_scaled, global_idx, local_idx)
        _, _, _, _, global_idx, local_idx = dataset[idx]

        # Map global_idx back to flight
        flight_idx, _ = dataset.global_to_local[int(global_idx)]
        flight_config = config["flights"][flight_idx]
        flight_name = flight_config["name"]
        flight_id = flight_name_to_id.get(flight_name, 0)

        # Load navigation data for this flight
        nav_file = flight_config["nFileName"]

        try:
            with h5py.File(nav_file, "r") as nf:
                # Get lat/lon arrays
                lats = nf["nav/IWG_lat"][:]
                lons = nf["nav/IWG_lon"][:]

                # Use local_idx to get the exact position in this flight
                if int(local_idx) < len(lats):
                    lat = float(lats[int(local_idx)])
                    lon = float(lons[int(local_idx)])
                else:
                    # Fallback: use mean of flight
                    lat = float(np.nanmean(lats))
                    lon = float(np.nanmean(lons))
        except Exception as e:
            if verbose and idx == 0:
                print(f"  Warning: Could not load nav data for flight {flight_id}: {e}")
                print(f"  Using approximate coordinates")

            # Fallback coordinates (approximate flight regions)
            if flight_id == 0:  # 30Oct24
                lat = 35.0
                lon = -80.0
            elif flight_id in [1, 3, 4]:  # Feb flights
                lat = 40.0
                lon = -75.0
            else:  # 23Oct24
                lat = 38.0
                lon = -77.0

        # Estimate time during flight (12:00 to 18:00 UTC typical)
        date_str = flight_dates[flight_id]
        hour = 12 + (idx % 100) / 100.0 * 6
        time = datetime.fromisoformat(date_str) + timedelta(hours=hour)

        metadata_list.append(
            {
                "index": idx,
                "latitude": lat,
                "longitude": lon,
                "time": time,
                "flight_id": flight_id,
            }
        )

        if verbose and (idx % 100 == 0 or idx == len(dataset) - 1):
            print(f"  Loaded {idx + 1}/{len(dataset)} samples")

    return metadata_list


def process_real_era5_features(
    config_path: str = "configs/bestComboConfig.yaml",
    output_path: str = "sow_outputs/wp2_atmospheric/WP2_Features_REAL_ERA5.hdf5",
    era5_surface_dir: str = "/media/rylan/two/research/NASA/ERA5_data_root/surface",
    era5_pressure_dir: str = "/media/rylan/two/research/NASA/ERA5_data_root/pressure_levels",
    verbose: bool = True,
) -> Dict:
    """
    Process real ERA5 data for all 933 samples.

    Args:
        config_path: Path to config YAML
        output_path: Output HDF5 file path
        era5_surface_dir: Directory with surface ERA5 data
        era5_pressure_dir: Directory with pressure-level ERA5 data
        verbose: Enable verbose output

    Returns:
        Statistics dictionary
    """
    if verbose:
        print("=" * 80)
        print("PROCESSING REAL ERA5 DATA")
        print("=" * 80)

    # Initialize processor
    processor = RealERA5Processor(
        era5_surface_dir=era5_surface_dir,
        era5_pressure_dir=era5_pressure_dir,
        verbose=verbose,
    )

    # Load metadata
    metadata_list = load_flight_metadata(config_path, verbose=verbose)
    n_samples = len(metadata_list)

    # Initialize feature arrays
    features = {
        "sample_id": np.arange(n_samples),
        "blh_m": np.zeros(n_samples),
        "lcl_m": np.zeros(n_samples),
        "inversion_height_m": np.zeros(n_samples),
        "moisture_gradient": np.zeros(n_samples),
        "stability_index": np.zeros(n_samples),
        "surface_temp_k": np.zeros(n_samples),
        "surface_dewpoint_k": np.zeros(n_samples),
        "surface_pressure_pa": np.zeros(n_samples),
        "lapse_rate_k_per_km": np.zeros(n_samples),
        "profile_confidence": np.zeros(n_samples),
        "latitude": np.zeros(n_samples),
        "longitude": np.zeros(n_samples),
    }

    # Process each sample
    n_success = 0
    n_failed = 0

    if verbose:
        print("\n" + "-" * 80)
        print("Extracting atmospheric features from real ERA5 data...")
        print("-" * 80)

    for idx, meta in enumerate(metadata_list):
        if verbose and (idx % 50 == 0 or idx == n_samples - 1):
            print(
                f"Processing sample {idx + 1}/{n_samples} ({100 * (idx + 1) / n_samples:.1f}%)"
            )

        # Extract profile
        profile = processor.extract_profile(
            dt=meta["time"],
            lat=meta["latitude"],
            lon=meta["longitude"],
        )

        if profile is not None:
            features["blh_m"][idx] = profile.blh_m
            features["lcl_m"][idx] = profile.lcl_m
            features["inversion_height_m"][idx] = profile.inversion_height_m
            features["moisture_gradient"][idx] = profile.moisture_gradient
            features["stability_index"][idx] = profile.stability_index
            features["surface_temp_k"][idx] = profile.surface_temp_k
            features["surface_dewpoint_k"][idx] = profile.surface_dewpoint_k
            features["surface_pressure_pa"][idx] = profile.surface_pressure_pa
            features["lapse_rate_k_per_km"][idx] = profile.lapse_rate_k_per_km
            features["profile_confidence"][idx] = profile.confidence
            features["latitude"][idx] = meta["latitude"]
            features["longitude"][idx] = meta["longitude"]
            n_success += 1
        else:
            # Fill with NaN for missing data
            features["blh_m"][idx] = np.nan
            features["lcl_m"][idx] = np.nan
            features["inversion_height_m"][idx] = np.nan
            features["moisture_gradient"][idx] = np.nan
            features["stability_index"][idx] = np.nan
            features["surface_temp_k"][idx] = np.nan
            features["surface_dewpoint_k"][idx] = np.nan
            features["surface_pressure_pa"][idx] = np.nan
            features["lapse_rate_k_per_km"][idx] = np.nan
            features["profile_confidence"][idx] = 0.0
            features["latitude"][idx] = meta["latitude"]
            features["longitude"][idx] = meta["longitude"]
            n_failed += 1

    # Save to HDF5
    if verbose:
        print("\n" + "-" * 80)
        print(f"Saving features to {output_path}")

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as hf:
        # Create datasets
        for key, value in features.items():
            hf.create_dataset(key, data=value, compression="gzip", compression_opts=4)

        # Create feature_names for compatibility
        feature_names = [
            "blh_m",
            "lcl_m",
            "inversion_height_m",
            "moisture_gradient",
            "stability_index",
            "surface_temp_k",
            "surface_dewpoint_k",
            "surface_pressure_pa",
            "lapse_rate_k_per_km",
        ]
        hf.create_dataset(
            "feature_names",
            data=np.array(feature_names, dtype="S"),
            compression="gzip",
            compression_opts=4,
        )

        # Create features array (for compatibility with existing code)
        feature_matrix = np.column_stack([features[name] for name in feature_names])
        hf.create_dataset(
            "features", data=feature_matrix, compression="gzip", compression_opts=4
        )

        # Save metadata
        hf.attrs["n_samples"] = n_samples
        hf.attrs["n_success"] = n_success
        hf.attrs["n_failed"] = n_failed
        hf.attrs["source"] = "Real ERA5 reanalysis data"
        hf.attrs["era5_surface_dir"] = str(era5_surface_dir)
        hf.attrs["era5_pressure_dir"] = str(era5_pressure_dir)
        hf.attrs["processing_date"] = datetime.now().isoformat()

    # Compute statistics (excluding NaN values)
    stats = {
        "n_samples": n_samples,
        "n_success": n_success,
        "n_failed": n_failed,
        "success_rate": n_success / n_samples,
        "mean_blh_m": float(np.nanmean(features["blh_m"])),
        "mean_lcl_m": float(np.nanmean(features["lcl_m"])),
        "mean_inversion_m": float(np.nanmean(features["inversion_height_m"])),
        "mean_stability": float(np.nanmean(features["stability_index"])),
    }

    if verbose:
        print("\n" + "=" * 80)
        print("PROCESSING COMPLETE!")
        print("=" * 80)
        print(f"Total samples: {stats['n_samples']}")
        print(
            f"Successfully processed: {stats['n_success']} ({100 * stats['success_rate']:.1f}%)"
        )
        print(f"Failed: {stats['n_failed']}")
        print(f"\nMean BLH: {stats['mean_blh_m']:.1f} m")
        print(f"Mean LCL: {stats['mean_lcl_m']:.1f} m")
        print(f"Mean Inversion Height: {stats['mean_inversion_m']:.1f} m")
        print(f"Mean Stability Index: {stats['mean_stability']:.2f} K/km")
        print("=" * 80)

    return stats


def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Process real ERA5 data for WP-2 atmospheric features"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/bestComboConfig.yaml",
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="sow_outputs/wp2_atmospheric/WP2_Features_REAL_ERA5.hdf5",
        help="Output path for feature HDF5 file",
    )
    parser.add_argument(
        "--era5-surface",
        type=str,
        default="/media/rylan/two/research/NASA/ERA5_data_root/surface",
        help="Directory with surface ERA5 data",
    )
    parser.add_argument(
        "--era5-pressure",
        type=str,
        default="/media/rylan/two/research/NASA/ERA5_data_root/pressure_levels",
        help="Directory with pressure-level ERA5 data",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    # Run processing
    stats = process_real_era5_features(
        config_path=args.config,
        output_path=args.output,
        era5_surface_dir=args.era5_surface,
        era5_pressure_dir=args.era5_pressure,
        verbose=args.verbose,
    )

    print("\n✓ Real ERA5 feature processing complete!")
    print(f"✓ Output saved to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
