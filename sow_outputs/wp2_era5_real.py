#!/usr/bin/env python3
"""
Work Package 2: REAL ERA5 Atmospheric Feature Engineering for CBH Retrieval

This module implements COMPLETE atmospheric thermodynamic feature extraction from ERA5
reanalysis data as specified in the SOW-AGENT-CBH-WP-001 document.

This is the PRODUCTION version that ACTUALLY downloads ERA5 data from CDS.

Key Features:
- Real ERA5 reanalysis data acquisition via CDS API
- Navigation file parsing to extract lat/lon/time for all 933 samples
- Thermodynamic variable derivation (BLH, LCL, Inversion Height, etc.)
- Spatio-temporal interpolation to align with 933 labeled samples
- Handling of spatial resolution mismatch (25 km ERA5 vs 200 m imagery)

ERA5 Data Storage: /media/rylan/two/research/NASA/ERA5_data_root/
- surface/ - Surface variables (BLH, 2m temp, dewpoint, etc.)
- pressure_levels/ - Pressure level variables (temperature, humidity profiles)
- processed/ - Derived thermodynamic variables

Author: Autonomous Agent (Production Version)
Date: 2025-06-04
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
import json
import time
import os

warnings.filterwarnings("ignore")

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Check dependencies
try:
    import cdsapi

    CDS_AVAILABLE = True
except ImportError:
    CDS_AVAILABLE = False
    print("ERROR: cdsapi not available. Install with: pip install cdsapi")
    print("Also set up ~/.cdsapirc with your CDS API credentials")

try:
    import xarray as xr
    import pandas as pd

    XARRAY_AVAILABLE = True
except ImportError:
    XARRAY_AVAILABLE = False
    print(
        "ERROR: xarray/pandas not available. Install with: pip install xarray netCDF4"
    )

try:
    from src.hdf5_dataset import HDF5CloudDataset
    from src.cplCompareSub import cplTimeConvert
except ImportError as e:
    print(f"Warning: Could not import project modules: {e}")


# Constants
ERA5_DATA_ROOT = "/media/rylan/two/research/NASA/ERA5_data_root"
SURFACE_DIR = os.path.join(ERA5_DATA_ROOT, "surface")
PRESSURE_DIR = os.path.join(ERA5_DATA_ROOT, "pressure_levels")
PROCESSED_DIR = os.path.join(ERA5_DATA_ROOT, "processed")


@dataclass
class SampleMetadata:
    """Metadata for a single sample."""

    sample_idx: int
    flight_name: str
    flight_idx: int
    local_idx: int
    latitude: float
    longitude: float
    timestamp: float  # Unix timestamp
    datetime: datetime
    sza: float
    saa: float
    cbh: float


class NavigationParser:
    """
    Parse navigation files to extract lat/lon/time for all samples.
    """

    def __init__(self, config_path: str, verbose: bool = True):
        """
        Initialize navigation parser.

        Args:
            config_path: Path to configuration YAML
            verbose: Enable verbose logging
        """
        self.verbose = verbose
        self.config_path = config_path

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        if self.verbose:
            print("=" * 80)
            print("Navigation Parser: Extracting lat/lon/time for all 933 samples")
            print("=" * 80)

    def parse_all_samples(self) -> List[SampleMetadata]:
        """
        Parse navigation data for all 933 labeled samples.

        Returns:
            List of SampleMetadata objects
        """
        if self.verbose:
            print("\nLoading dataset to identify labeled samples...")

        # Load dataset
        dataset = HDF5CloudDataset(
            flight_configs=self.config["flights"],
            indices=None,
            augment=False,
            swath_slice=self.config.get("swath_slice", [40, 480]),
        )

        if self.verbose:
            print(f"Dataset loaded: {len(dataset)} labeled samples")

        metadata_list = []

        # Process each sample
        for idx in range(len(dataset)):
            if self.verbose and idx % 100 == 0:
                print(f"Processing sample {idx}/{len(dataset)}...")

            # Get sample info
            _, sza_tensor, saa_tensor, cbh_tensor, global_idx, local_idx = dataset[idx]

            # Extract scalar values
            sza = float(
                sza_tensor.item() if hasattr(sza_tensor, "item") else sza_tensor
            )
            saa = float(
                saa_tensor.item() if hasattr(saa_tensor, "item") else saa_tensor
            )
            cbh = float(
                cbh_tensor.item() if hasattr(cbh_tensor, "item") else cbh_tensor
            )

            # Get flight info using global_to_local mapping
            global_sample_idx = int(dataset.indices[idx])
            flight_idx, local_sample_idx = dataset.global_to_local[global_sample_idx]
            flight_info = dataset.flight_data[flight_idx]
            flight_name = flight_info["name"]

            # Load navigation data for this specific sample
            nav_file = self.config["flights"][flight_idx]["nFileName"]

            try:
                with h5py.File(nav_file, "r") as nf:
                    # Get latitude and longitude
                    lat = float(nf["nav/IWG_lat"][local_sample_idx])
                    lon = float(nf["nav/IWG_lon"][local_sample_idx])

                # Get time from the times_all array already stored in flight_info
                timestamp = float(flight_info["times_all"][local_sample_idx])
                dt = datetime.utcfromtimestamp(timestamp)

                # Create metadata object
                meta = SampleMetadata(
                    sample_idx=idx,
                    flight_name=flight_name,
                    flight_idx=flight_idx,
                    local_idx=local_sample_idx,
                    latitude=lat,
                    longitude=lon,
                    timestamp=timestamp,
                    datetime=dt,
                    sza=sza,
                    saa=saa,
                    cbh=cbh,
                )

                metadata_list.append(meta)

            except Exception as e:
                print(f"Warning: Could not parse navigation for sample {idx}: {e}")
                import traceback

                traceback.print_exc()
                # Create placeholder with NaN values
                meta = SampleMetadata(
                    sample_idx=idx,
                    flight_name=flight_name,
                    flight_idx=flight_idx,
                    local_idx=local_sample_idx,
                    latitude=np.nan,
                    longitude=np.nan,
                    timestamp=np.nan,
                    datetime=datetime(2024, 1, 1),
                    sza=sza,
                    saa=saa,
                    cbh=cbh,
                )
                metadata_list.append(meta)

        if self.verbose:
            print(f"\nParsed {len(metadata_list)} samples")
            valid_count = sum(1 for m in metadata_list if not np.isnan(m.latitude))
            print(f"Valid navigation data: {valid_count}/{len(metadata_list)}")

        return metadata_list

    def get_spatial_temporal_bounds(self, metadata_list: List[SampleMetadata]) -> Dict:
        """
        Calculate spatial and temporal bounds for ERA5 download.

        Args:
            metadata_list: List of sample metadata

        Returns:
            Dictionary with bounds information
        """
        # Filter out NaN values
        valid_meta = [m for m in metadata_list if not np.isnan(m.latitude)]

        if not valid_meta:
            raise ValueError("No valid navigation data found!")

        lats = [m.latitude for m in valid_meta]
        lons = [m.longitude for m in valid_meta]
        times = [m.datetime for m in valid_meta]

        # Calculate bounds with buffer
        lat_buffer = 2.0  # degrees
        lon_buffer = 2.0  # degrees
        time_buffer = timedelta(hours=6)

        bounds = {
            "lat_min": min(lats) - lat_buffer,
            "lat_max": max(lats) + lat_buffer,
            "lon_min": min(lons) - lon_buffer,
            "lon_max": max(lons) + lon_buffer,
            "time_min": min(times) - time_buffer,
            "time_max": max(times) + time_buffer,
            "n_samples": len(valid_meta),
        }

        if self.verbose:
            print("\nSpatial/Temporal Bounds for ERA5 Download:")
            print(f"  Latitude:  [{bounds['lat_min']:.2f}, {bounds['lat_max']:.2f}]")
            print(f"  Longitude: [{bounds['lon_min']:.2f}, {bounds['lon_max']:.2f}]")
            print(f"  Time:      {bounds['time_min']} to {bounds['time_max']}")
            print(f"  Duration:  {(bounds['time_max'] - bounds['time_min']).days} days")

        return bounds


class ERA5Downloader:
    """
    Download ERA5 data from CDS.
    """

    def __init__(self, output_dir: str = ERA5_DATA_ROOT, verbose: bool = True):
        """
        Initialize ERA5 downloader.

        Args:
            output_dir: Root directory for ERA5 data
            verbose: Enable verbose logging
        """
        self.output_dir = Path(output_dir)
        self.surface_dir = self.output_dir / "surface"
        self.pressure_dir = self.output_dir / "pressure_levels"
        self.verbose = verbose

        # Create directories
        self.surface_dir.mkdir(parents=True, exist_ok=True)
        self.pressure_dir.mkdir(parents=True, exist_ok=True)

        if not CDS_AVAILABLE:
            raise RuntimeError("cdsapi not available. Install: pip install cdsapi")

        self.client = cdsapi.Client()

        if self.verbose:
            print("=" * 80)
            print("ERA5 Downloader: Acquiring reanalysis data from CDS")
            print("=" * 80)
            print(f"Surface data:       {self.surface_dir}")
            print(f"Pressure level data: {self.pressure_dir}")

    def download_surface_data(self, bounds: Dict, force: bool = False) -> List[Path]:
        """
        Download ERA5 surface-level data.

        Variables:
        - Boundary Layer Height (BLH)
        - 2m temperature (t2m)
        - 2m dewpoint temperature (d2m)
        - Surface pressure (sp)
        - Total column water vapor (tcwv)

        Args:
            bounds: Spatial/temporal bounds dictionary
            force: Force re-download even if files exist

        Returns:
            List of downloaded file paths
        """
        if self.verbose:
            print("\n" + "=" * 80)
            print("Downloading ERA5 Surface Data")
            print("=" * 80)

        # Generate list of days to download
        start_date = bounds["time_min"].date()
        end_date = bounds["time_max"].date()

        date_list = []
        current_date = start_date
        while current_date <= end_date:
            date_list.append(current_date)
            current_date += timedelta(days=1)

        if self.verbose:
            print(f"Downloading {len(date_list)} days of surface data...")
            print(f"Date range: {start_date} to {end_date}")

        downloaded_files = []

        for date in date_list:
            output_file = (
                self.surface_dir / f"era5_surface_{date.strftime('%Y%m%d')}.nc"
            )

            if output_file.exists() and not force:
                if self.verbose:
                    print(f"  [SKIP] {output_file.name} (already exists)")
                downloaded_files.append(output_file)
                continue

            if self.verbose:
                print(f"  [DOWNLOAD] {date} -> {output_file.name}")

            try:
                self.client.retrieve(
                    "reanalysis-era5-single-levels",
                    {
                        "product_type": "reanalysis",
                        "variable": [
                            "boundary_layer_height",
                            "2m_temperature",
                            "2m_dewpoint_temperature",
                            "surface_pressure",
                            "total_column_water_vapour",
                        ],
                        "year": date.strftime("%Y"),
                        "month": date.strftime("%m"),
                        "day": date.strftime("%d"),
                        "time": [
                            "00:00",
                            "01:00",
                            "02:00",
                            "03:00",
                            "04:00",
                            "05:00",
                            "06:00",
                            "07:00",
                            "08:00",
                            "09:00",
                            "10:00",
                            "11:00",
                            "12:00",
                            "13:00",
                            "14:00",
                            "15:00",
                            "16:00",
                            "17:00",
                            "18:00",
                            "19:00",
                            "20:00",
                            "21:00",
                            "22:00",
                            "23:00",
                        ],
                        "area": [
                            bounds["lat_max"],
                            bounds["lon_min"],
                            bounds["lat_min"],
                            bounds["lon_max"],
                        ],
                        "format": "netcdf",
                    },
                    str(output_file),
                )

                downloaded_files.append(output_file)

                if self.verbose:
                    file_size_mb = output_file.stat().st_size / (1024 * 1024)
                    print(f"    -> Downloaded {file_size_mb:.2f} MB")

                # Brief pause to avoid overwhelming the server
                time.sleep(1)

            except Exception as e:
                print(f"    ERROR downloading {date}: {e}")
                continue

        if self.verbose:
            total_size_mb = sum(f.stat().st_size for f in downloaded_files) / (
                1024 * 1024
            )
            print(f"\nSurface data download complete:")
            print(f"  Files: {len(downloaded_files)}")
            print(f"  Total size: {total_size_mb:.2f} MB")

        return downloaded_files

    def download_pressure_level_data(
        self, bounds: Dict, force: bool = False
    ) -> List[Path]:
        """
        Download ERA5 pressure-level data for atmospheric profiles.

        Variables:
        - Temperature (t)
        - Specific humidity (q)
        - Geopotential (z)

        Levels: 1000, 975, 950, 925, 900, 850, 800, 750, 700, 650, 600, 550, 500 hPa

        Args:
            bounds: Spatial/temporal bounds dictionary
            force: Force re-download even if files exist

        Returns:
            List of downloaded file paths
        """
        if self.verbose:
            print("\n" + "=" * 80)
            print("Downloading ERA5 Pressure Level Data")
            print("=" * 80)

        # Generate list of days to download
        start_date = bounds["time_min"].date()
        end_date = bounds["time_max"].date()

        date_list = []
        current_date = start_date
        while current_date <= end_date:
            date_list.append(current_date)
            current_date += timedelta(days=1)

        if self.verbose:
            print(f"Downloading {len(date_list)} days of pressure level data...")
            print(f"Date range: {start_date} to {end_date}")
            print("This may take several hours - pressure level data is large!")

        downloaded_files = []

        for date in date_list:
            output_file = (
                self.pressure_dir / f"era5_pressure_{date.strftime('%Y%m%d')}.nc"
            )

            if output_file.exists() and not force:
                if self.verbose:
                    print(f"  [SKIP] {output_file.name} (already exists)")
                downloaded_files.append(output_file)
                continue

            if self.verbose:
                print(f"  [DOWNLOAD] {date} -> {output_file.name}")

            try:
                self.client.retrieve(
                    "reanalysis-era5-pressure-levels",
                    {
                        "product_type": "reanalysis",
                        "variable": [
                            "temperature",
                            "specific_humidity",
                            "geopotential",
                        ],
                        "pressure_level": [
                            "500",
                            "550",
                            "600",
                            "650",
                            "700",
                            "750",
                            "800",
                            "850",
                            "900",
                            "925",
                            "950",
                            "975",
                            "1000",
                        ],
                        "year": date.strftime("%Y"),
                        "month": date.strftime("%m"),
                        "day": date.strftime("%d"),
                        "time": [
                            "00:00",
                            "01:00",
                            "02:00",
                            "03:00",
                            "04:00",
                            "05:00",
                            "06:00",
                            "07:00",
                            "08:00",
                            "09:00",
                            "10:00",
                            "11:00",
                            "12:00",
                            "13:00",
                            "14:00",
                            "15:00",
                            "16:00",
                            "17:00",
                            "18:00",
                            "19:00",
                            "20:00",
                            "21:00",
                            "22:00",
                            "23:00",
                        ],
                        "area": [
                            bounds["lat_max"],
                            bounds["lon_min"],
                            bounds["lat_min"],
                            bounds["lon_max"],
                        ],
                        "format": "netcdf",
                    },
                    str(output_file),
                )

                downloaded_files.append(output_file)

                if self.verbose:
                    file_size_mb = output_file.stat().st_size / (1024 * 1024)
                    print(f"    -> Downloaded {file_size_mb:.2f} MB")

                # Brief pause to avoid overwhelming the server
                time.sleep(1)

            except Exception as e:
                print(f"    ERROR downloading {date}: {e}")
                continue

        if self.verbose:
            total_size_mb = sum(f.stat().st_size for f in downloaded_files) / (
                1024 * 1024
            )
            print(f"\nPressure level data download complete:")
            print(f"  Files: {len(downloaded_files)}")
            print(f"  Total size: {total_size_mb:.2f} MB")

        return downloaded_files


class AtmosphericFeatureExtractor:
    """
    Extract and derive atmospheric features from ERA5 data.
    """

    def __init__(self, era5_dir: str = ERA5_DATA_ROOT, verbose: bool = True):
        """
        Initialize feature extractor.

        Args:
            era5_dir: Root directory with ERA5 data
            verbose: Enable verbose logging
        """
        self.era5_dir = Path(era5_dir)
        self.surface_dir = self.era5_dir / "surface"
        self.pressure_dir = self.era5_dir / "pressure_levels"
        self.verbose = verbose

        if not XARRAY_AVAILABLE:
            raise RuntimeError(
                "xarray not available. Install: pip install xarray netCDF4"
            )

        if self.verbose:
            print("=" * 80)
            print("Atmospheric Feature Extractor: Deriving thermodynamic variables")
            print("=" * 80)

    def compute_lcl(self, t2m: float, d2m: float) -> float:
        """
        Compute Lifting Condensation Level using parcel theory.

        Formula: LCL (km) = (T - Td) / 8 K/km
        Where T is 2m temperature and Td is 2m dewpoint

        Args:
            t2m: 2-meter temperature (K)
            d2m: 2-meter dewpoint temperature (K)

        Returns:
            LCL height in kilometers
        """
        # Convert to Celsius
        t_celsius = t2m - 273.15
        td_celsius = d2m - 273.15

        # LCL formula
        lcl_km = (t_celsius - td_celsius) / 8.0

        return max(0.0, lcl_km)  # LCL cannot be negative

    def find_inversion_height(
        self, temperature_profile: np.ndarray, height_profile: np.ndarray
    ) -> float:
        """
        Find temperature inversion height from vertical profile.

        Identifies the altitude of strongest temperature gradient.

        Args:
            temperature_profile: Temperature at each level (K)
            height_profile: Geopotential height at each level (m)

        Returns:
            Inversion height in kilometers
        """
        if len(temperature_profile) < 2:
            return np.nan

        # Compute temperature gradient (dT/dz)
        dt_dz = np.gradient(temperature_profile, height_profile)

        # Find strongest positive gradient (inversion)
        max_gradient_idx = np.argmax(dt_dz)

        inversion_height_m = height_profile[max_gradient_idx]
        inversion_height_km = inversion_height_m / 1000.0

        return inversion_height_km

    def compute_stability_index(
        self, temperature_profile: np.ndarray, height_profile: np.ndarray
    ) -> float:
        """
        Compute atmospheric stability index.

        Uses the mean lapse rate in the lower troposphere.

        Args:
            temperature_profile: Temperature at each level (K)
            height_profile: Geopotential height at each level (m)

        Returns:
            Lapse rate in K/km (negative = stable, positive = unstable)
        """
        if len(temperature_profile) < 2:
            return np.nan

        # Compute mean lapse rate
        dT = temperature_profile[-1] - temperature_profile[0]
        dz = (height_profile[-1] - height_profile[0]) / 1000.0  # Convert to km

        lapse_rate = dT / dz if dz > 0 else np.nan

        return lapse_rate

    def extract_features_for_sample(self, metadata: SampleMetadata) -> Dict[str, float]:
        """
        Extract atmospheric features for a single sample.

        Args:
            metadata: Sample metadata with lat/lon/time

        Returns:
            Dictionary of atmospheric features
        """
        if np.isnan(metadata.latitude) or np.isnan(metadata.longitude):
            # Return NaN features
            return {
                "blh": np.nan,
                "lcl": np.nan,
                "inversion_height": np.nan,
                "moisture_gradient": np.nan,
                "stability_index": np.nan,
                "t2m": np.nan,
                "d2m": np.nan,
                "sp": np.nan,
                "tcwv": np.nan,
            }

        # Find corresponding ERA5 file
        date_str = metadata.datetime.strftime("%Y%m%d")
        surface_file = self.surface_dir / f"era5_surface_{date_str}.nc"
        pressure_file = self.pressure_dir / f"era5_pressure_{date_str}.nc"

        if not surface_file.exists():
            print(f"Warning: Surface file not found for {date_str}")
            return self._nan_features()

        try:
            # Load surface data
            ds_surface = xr.open_dataset(surface_file)

            # Interpolate to sample location and time
            sample_time = pd.Timestamp(metadata.datetime)

            # Nearest neighbor interpolation for now
            # (Could use more sophisticated methods)
            ds_interp = ds_surface.sel(
                latitude=metadata.latitude,
                longitude=metadata.longitude,
                valid_time=sample_time,
                method="nearest",
            )

            # Extract surface variables
            blh = float(ds_interp["blh"].values) / 1000.0  # Convert to km
            t2m = float(ds_interp["t2m"].values)
            d2m = float(ds_interp["d2m"].values)
            sp = float(ds_interp["sp"].values)
            tcwv = float(ds_interp["tcwv"].values)

            # Compute LCL
            lcl = self.compute_lcl(t2m, d2m)

            # Load pressure level data if available
            if pressure_file.exists():
                ds_pressure = xr.open_dataset(pressure_file)

                ds_pressure_interp = ds_pressure.sel(
                    latitude=metadata.latitude,
                    longitude=metadata.longitude,
                    valid_time=sample_time,
                    method="nearest",
                )

                # Extract vertical profiles
                temp_profile = ds_pressure_interp["t"].values
                geopotential = (
                    ds_pressure_interp["z"].values / 9.81
                )  # Convert to height (m)
                humidity_profile = ds_pressure_interp["q"].values

                # Compute derived quantities
                inversion_height = self.find_inversion_height(
                    temp_profile, geopotential
                )
                stability_index = self.compute_stability_index(
                    temp_profile, geopotential
                )

                # Moisture gradient
                moisture_gradient = np.gradient(humidity_profile, geopotential).mean()

            else:
                inversion_height = np.nan
                stability_index = np.nan
                moisture_gradient = np.nan

            ds_surface.close()
            if pressure_file.exists():
                ds_pressure.close()

            return {
                "blh": blh,
                "lcl": lcl,
                "inversion_height": inversion_height,
                "moisture_gradient": moisture_gradient,
                "stability_index": stability_index,
                "t2m": t2m,
                "d2m": d2m,
                "sp": sp,
                "tcwv": tcwv,
            }

        except Exception as e:
            print(
                f"Warning: Could not extract features for sample {metadata.sample_idx}: {e}"
            )
            return self._nan_features()

    def _nan_features(self) -> Dict[str, float]:
        """Return dictionary of NaN features."""
        return {
            "blh": np.nan,
            "lcl": np.nan,
            "inversion_height": np.nan,
            "moisture_gradient": np.nan,
            "stability_index": np.nan,
            "t2m": np.nan,
            "d2m": np.nan,
            "sp": np.nan,
            "tcwv": np.nan,
        }

    def extract_all_features(self, metadata_list: List[SampleMetadata]) -> np.ndarray:
        """
        Extract atmospheric features for all samples.

        Args:
            metadata_list: List of sample metadata

        Returns:
            Feature array (n_samples, n_features)
        """
        if self.verbose:
            print(
                f"\nExtracting atmospheric features for {len(metadata_list)} samples..."
            )

        feature_names = [
            "blh",
            "lcl",
            "inversion_height",
            "moisture_gradient",
            "stability_index",
            "t2m",
            "d2m",
            "sp",
            "tcwv",
        ]

        n_samples = len(metadata_list)
        n_features = len(feature_names)

        features = np.zeros((n_samples, n_features))

        for i, meta in enumerate(metadata_list):
            if self.verbose and i % 100 == 0:
                print(f"  Processing sample {i}/{n_samples}...")

            feat_dict = self.extract_features_for_sample(meta)

            for j, name in enumerate(feature_names):
                features[i, j] = feat_dict[name]

        if self.verbose:
            print(f"\nFeature extraction complete!")
            print(f"Feature array shape: {features.shape}")

            # Report statistics
            for j, name in enumerate(feature_names):
                valid_count = np.sum(~np.isnan(features[:, j]))
                mean_val = np.nanmean(features[:, j])
                std_val = np.nanstd(features[:, j])
                print(
                    f"  {name:20s}: {valid_count:4d}/{n_samples} valid, "
                    f"mean={mean_val:8.3f}, std={std_val:8.3f}"
                )

        return features


def save_features_hdf5(
    output_path: str,
    metadata_list: List[SampleMetadata],
    features: np.ndarray,
    verbose: bool = True,
):
    """
    Save atmospheric features to HDF5 file.

    Args:
        output_path: Path to output HDF5 file
        metadata_list: List of sample metadata
        features: Feature array (n_samples, n_features)
        verbose: Enable verbose logging
    """
    if verbose:
        print(f"\nSaving features to {output_path}...")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as f:
        # Save features
        f.create_dataset("features", data=features, compression="gzip")

        # Save feature names
        feature_names = [
            "blh",
            "lcl",
            "inversion_height",
            "moisture_gradient",
            "stability_index",
            "t2m",
            "d2m",
            "sp",
            "tcwv",
        ]
        f.create_dataset("feature_names", data=np.array(feature_names, dtype="S"))

        # Save metadata
        sample_indices = np.array([m.sample_idx for m in metadata_list])
        latitudes = np.array([m.latitude for m in metadata_list])
        longitudes = np.array([m.longitude for m in metadata_list])
        timestamps = np.array([m.timestamp for m in metadata_list])

        f.create_dataset("sample_indices", data=sample_indices)
        f.create_dataset("latitudes", data=latitudes)
        f.create_dataset("longitudes", data=longitudes)
        f.create_dataset("timestamps", data=timestamps)

        # Save summary statistics
        f.attrs["n_samples"] = len(metadata_list)
        f.attrs["n_features"] = features.shape[1]
        f.attrs["creation_date"] = datetime.now().isoformat()
        f.attrs["era5_data_dir"] = ERA5_DATA_ROOT

    if verbose:
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"Features saved: {file_size_mb:.2f} MB")


def main():
    """Main execution pipeline."""
    import argparse

    parser = argparse.ArgumentParser(
        description="WP-2: Real ERA5 Atmospheric Feature Engineering"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/bestComboConfig.yaml",
        help="Path to configuration YAML",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="sow_outputs/wp2_atmospheric/WP2_Features.hdf5",
        help="Path to output HDF5 file",
    )
    parser.add_argument(
        "--era5-dir",
        type=str,
        default=ERA5_DATA_ROOT,
        help="Root directory for ERA5 data",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip ERA5 download (use existing data)",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download even if files exist",
    )
    parser.add_argument(
        "--surface-only",
        action="store_true",
        help="Download only surface data (faster, fewer features)",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    print("=" * 80)
    print("WORK PACKAGE 2: REAL ERA5 ATMOSPHERIC FEATURE ENGINEERING")
    print("=" * 80)
    print(f"Configuration:  {args.config}")
    print(f"Output:         {args.output}")
    print(f"ERA5 data root: {args.era5_dir}")
    print(f"Skip download:  {args.skip_download}")
    print(f"Force download: {args.force_download}")
    print(f"Surface only:   {args.surface_only}")
    print("=" * 80)

    # Check dependencies
    if not CDS_AVAILABLE:
        print("\nERROR: cdsapi not installed!")
        print("Install with: pip install cdsapi")
        print("Set up ~/.cdsapirc with your CDS API credentials")
        print("Register at: https://cds.climate.copernicus.eu/user/register")
        return 1

    if not XARRAY_AVAILABLE:
        print("\nERROR: xarray not installed!")
        print("Install with: pip install xarray netCDF4")
        return 1

    # Step 1: Parse navigation data
    print("\n" + "=" * 80)
    print("STEP 1: Parse Navigation Data")
    print("=" * 80)

    nav_parser = NavigationParser(args.config, verbose=args.verbose)
    metadata_list = nav_parser.parse_all_samples()

    # Calculate spatial/temporal bounds
    bounds = nav_parser.get_spatial_temporal_bounds(metadata_list)

    # Save metadata for reference
    metadata_file = Path(args.era5_dir) / "processed" / "sample_metadata.json"
    metadata_file.parent.mkdir(parents=True, exist_ok=True)

    with open(metadata_file, "w") as f:
        metadata_dict = {
            "n_samples": len(metadata_list),
            "bounds": {k: str(v) for k, v in bounds.items()},
            "samples": [
                {
                    "idx": m.sample_idx,
                    "flight": m.flight_name,
                    "lat": float(m.latitude) if not np.isnan(m.latitude) else None,
                    "lon": float(m.longitude) if not np.isnan(m.longitude) else None,
                    "time": m.datetime.isoformat()
                    if not np.isnan(m.timestamp)
                    else None,
                }
                for m in metadata_list[:10]  # Save first 10 as examples
            ],
        }
        json.dump(metadata_dict, f, indent=2)

    print(f"Metadata saved to: {metadata_file}")

    # Step 2: Download ERA5 data
    if not args.skip_download:
        print("\n" + "=" * 80)
        print("STEP 2: Download ERA5 Data")
        print("=" * 80)
        print("\nWARNING: This will download substantial data (~30-70 GB)")
        print("This may take several hours depending on CDS queue times.")
        print(f"Data will be saved to: {args.era5_dir}")

        downloader = ERA5Downloader(output_dir=args.era5_dir, verbose=args.verbose)

        # Download surface data
        surface_files = downloader.download_surface_data(
            bounds, force=args.force_download
        )

        # Download pressure level data (unless surface-only mode)
        if not args.surface_only:
            pressure_files = downloader.download_pressure_level_data(
                bounds, force=args.force_download
            )
        else:
            print("\nSkipping pressure level data (--surface-only mode)")
            print(
                "Warning: Some features (inversion_height, stability_index) will be NaN"
            )
    else:
        print("\n" + "=" * 80)
        print("STEP 2: Download ERA5 Data [SKIPPED]")
        print("=" * 80)
        print("Using existing ERA5 data")

    # Step 3: Extract atmospheric features
    print("\n" + "=" * 80)
    print("STEP 3: Extract Atmospheric Features")
    print("=" * 80)

    extractor = AtmosphericFeatureExtractor(
        era5_dir=args.era5_dir, verbose=args.verbose
    )
    features = extractor.extract_all_features(metadata_list)

    # Step 4: Save features
    print("\n" + "=" * 80)
    print("STEP 4: Save Features")
    print("=" * 80)

    save_features_hdf5(args.output, metadata_list, features, verbose=args.verbose)

    # Final summary
    print("\n" + "=" * 80)
    print("WP-2 COMPLETE: Real ERA5 Atmospheric Features")
    print("=" * 80)
    print(f"Output file:    {args.output}")
    print(f"ERA5 data:      {args.era5_dir}")
    print(f"Samples:        {len(metadata_list)}")
    print(f"Features:       {features.shape[1]}")
    print("\nNext step: Run WP-3 (Physical Baseline Validation)")
    print("  python sow_outputs/wp3_physical_baseline.py --wp2-features", args.output)
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
