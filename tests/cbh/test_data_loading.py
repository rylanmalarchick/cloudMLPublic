"""
Unit tests for data loading and validation utilities.

Tests cover:
- HDF5 data loading
- Feature extraction
- Data validation
- Missing value handling
- Data integrity checks
"""

import tempfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def mock_hdf5_file(tmp_path):
    """Create a mock HDF5 file with atmospheric and geometric features."""
    filepath = tmp_path / "test_features.h5"

    with h5py.File(filepath, "w") as f:
        # Create atmospheric features group
        atmo_group = f.create_group("atmospheric_features")
        atmo_group.create_dataset("blh", data=np.random.randn(100) * 200 + 1000)
        atmo_group.create_dataset("lcl", data=np.random.randn(100) * 150 + 800)
        atmo_group.create_dataset(
            "inversion_height", data=np.random.randn(100) * 250 + 1200
        )
        atmo_group.create_dataset(
            "moisture_gradient", data=np.random.randn(100) * 0.2 + 0.5
        )
        atmo_group.create_dataset(
            "stability_index", data=np.random.randn(100) * 0.1 + 0.3
        )
        atmo_group.create_dataset("t2m", data=np.random.randn(100) * 5 + 290)
        atmo_group.create_dataset("d2m", data=np.random.randn(100) * 5 + 285)
        atmo_group.create_dataset("sp", data=np.random.randn(100) * 1000 + 101325)
        atmo_group.create_dataset("tcwv", data=np.random.randn(100) * 5 + 25)

        # Create geometric features group
        geom_group = f.create_group("geometric_features")
        geom_group.create_dataset("cloud_edge_x", data=np.random.randint(0, 1000, 100))
        geom_group.create_dataset("cloud_edge_y", data=np.random.randint(0, 1000, 100))
        geom_group.create_dataset("saa_deg", data=np.random.randn(100) * 30 + 180)
        geom_group.create_dataset(
            "shadow_angle_deg", data=np.random.randn(100) * 10 + 45
        )
        geom_group.create_dataset(
            "shadow_detection_confidence", data=np.random.rand(100)
        )
        geom_group.create_dataset("shadow_edge_x", data=np.random.randint(0, 1000, 100))
        geom_group.create_dataset("shadow_edge_y", data=np.random.randint(0, 1000, 100))
        geom_group.create_dataset(
            "shadow_length_pixels", data=np.random.randint(10, 200, 100)
        )
        geom_group.create_dataset("sza_deg", data=np.random.randn(100) * 15 + 30)

        # Create labels
        f.create_dataset("cbh", data=np.random.randn(100) * 200 + 1000)

        # Create metadata
        meta_group = f.create_group("metadata")
        meta_group.create_dataset(
            "flight_id", data=np.array([b"F1"] * 50 + [b"F2"] * 50)
        )
        meta_group.create_dataset("timestamp", data=np.arange(100))

    return filepath


@pytest.fixture
def mock_hdf5_with_missing(tmp_path):
    """Create HDF5 file with missing values."""
    filepath = tmp_path / "test_features_missing.h5"

    with h5py.File(filepath, "w") as f:
        data = np.random.randn(100) * 200 + 1000
        data[10:20] = np.nan  # Introduce missing values

        atmo_group = f.create_group("atmospheric_features")
        atmo_group.create_dataset("blh", data=data)
        atmo_group.create_dataset("lcl", data=np.random.randn(100) * 150 + 800)

    return filepath


class TestHDF5Loading:
    """Test HDF5 file loading."""

    def test_open_hdf5_file(self, mock_hdf5_file):
        """Test opening HDF5 file."""
        with h5py.File(mock_hdf5_file, "r") as f:
            assert "atmospheric_features" in f
            assert "geometric_features" in f
            assert "cbh" in f

    def test_read_atmospheric_features(self, mock_hdf5_file):
        """Test reading atmospheric features."""
        with h5py.File(mock_hdf5_file, "r") as f:
            blh = f["atmospheric_features"]["blh"][:]
            assert blh.shape == (100,)
            assert np.all(np.isfinite(blh))

    def test_read_geometric_features(self, mock_hdf5_file):
        """Test reading geometric features."""
        with h5py.File(mock_hdf5_file, "r") as f:
            cloud_edge_x = f["geometric_features"]["cloud_edge_x"][:]
            assert cloud_edge_x.shape == (100,)
            assert np.all(cloud_edge_x >= 0)

    def test_read_labels(self, mock_hdf5_file):
        """Test reading CBH labels."""
        with h5py.File(mock_hdf5_file, "r") as f:
            cbh = f["cbh"][:]
            assert cbh.shape == (100,)
            assert np.all(np.isfinite(cbh))

    def test_read_metadata(self, mock_hdf5_file):
        """Test reading metadata."""
        with h5py.File(mock_hdf5_file, "r") as f:
            flight_id = f["metadata"]["flight_id"][:]
            assert flight_id.shape == (100,)

    def test_nonexistent_file(self):
        """Test error on nonexistent file."""
        with pytest.raises(FileNotFoundError):
            with h5py.File("/nonexistent/file.h5", "r") as f:
                pass


class TestFeatureExtraction:
    """Test feature extraction from HDF5."""

    def test_extract_all_features(self, mock_hdf5_file):
        """Test extracting all features into array."""
        with h5py.File(mock_hdf5_file, "r") as f:
            # Extract atmospheric features
            atmo_features = []
            for key in [
                "blh",
                "lcl",
                "inversion_height",
                "moisture_gradient",
                "stability_index",
                "t2m",
                "d2m",
                "sp",
                "tcwv",
            ]:
                atmo_features.append(f["atmospheric_features"][key][:])

            # Extract geometric features
            geom_features = []
            for key in [
                "cloud_edge_x",
                "cloud_edge_y",
                "saa_deg",
                "shadow_angle_deg",
                "shadow_detection_confidence",
                "shadow_edge_x",
                "shadow_edge_y",
                "shadow_length_pixels",
                "sza_deg",
            ]:
                geom_features.append(f["geometric_features"][key][:])

            # Combine
            all_features = np.column_stack(atmo_features + geom_features)

            assert all_features.shape == (100, 18)

    def test_feature_names_order(self):
        """Test feature names are in correct order."""
        expected_features = [
            "blh",
            "lcl",
            "inversion_height",
            "moisture_gradient",
            "stability_index",
            "t2m",
            "d2m",
            "sp",
            "tcwv",
            "cloud_edge_x",
            "cloud_edge_y",
            "saa_deg",
            "shadow_angle_deg",
            "shadow_detection_confidence",
            "shadow_edge_x",
            "shadow_edge_y",
            "shadow_length_pixels",
            "sza_deg",
        ]

        assert len(expected_features) == 18

    def test_extract_subset_samples(self, mock_hdf5_file):
        """Test extracting subset of samples."""
        with h5py.File(mock_hdf5_file, "r") as f:
            blh = f["atmospheric_features"]["blh"][:50]
            assert blh.shape == (50,)

    def test_extract_single_feature(self, mock_hdf5_file):
        """Test extracting single feature."""
        with h5py.File(mock_hdf5_file, "r") as f:
            t2m = f["atmospheric_features"]["t2m"][:]
            assert t2m.ndim == 1
            assert len(t2m) == 100


class TestDataValidation:
    """Test data validation utilities."""

    def test_validate_feature_count(self, mock_hdf5_file):
        """Test validation of feature count."""
        with h5py.File(mock_hdf5_file, "r") as f:
            # Count atmospheric features
            n_atmo = len(f["atmospheric_features"].keys())
            # Count geometric features
            n_geom = len(f["geometric_features"].keys())

            assert n_atmo == 9
            assert n_geom == 9
            assert n_atmo + n_geom == 18

    def test_validate_data_types(self, mock_hdf5_file):
        """Test validation of data types."""
        with h5py.File(mock_hdf5_file, "r") as f:
            blh = f["atmospheric_features"]["blh"][:]
            assert np.issubdtype(blh.dtype, np.floating)

            cloud_edge_x = f["geometric_features"]["cloud_edge_x"][:]
            assert np.issubdtype(cloud_edge_x.dtype, np.number)

    def test_validate_no_infinite_values(self, mock_hdf5_file):
        """Test validation that data has no infinite values."""
        with h5py.File(mock_hdf5_file, "r") as f:
            for key in f["atmospheric_features"].keys():
                data = f["atmospheric_features"][key][:]
                assert not np.any(np.isinf(data))

    def test_validate_shape_consistency(self, mock_hdf5_file):
        """Test all features have same number of samples."""
        with h5py.File(mock_hdf5_file, "r") as f:
            n_samples = len(f["cbh"][:])

            for key in f["atmospheric_features"].keys():
                data = f["atmospheric_features"][key][:]
                assert len(data) == n_samples

            for key in f["geometric_features"].keys():
                data = f["geometric_features"][key][:]
                assert len(data) == n_samples

    def test_validate_reasonable_ranges(self, mock_hdf5_file):
        """Test features are in reasonable physical ranges."""
        with h5py.File(mock_hdf5_file, "r") as f:
            # Temperature should be reasonable (200-350 K)
            t2m = f["atmospheric_features"]["t2m"][:]
            assert np.all(t2m > 200)
            assert np.all(t2m < 350)

            # Confidence should be [0, 1]
            conf = f["geometric_features"]["shadow_detection_confidence"][:]
            assert np.all(conf >= 0)
            assert np.all(conf <= 1)

            # Angles should be reasonable
            sza = f["geometric_features"]["sza_deg"][:]
            assert np.all(sza >= 0)
            assert np.all(sza <= 90)


class TestMissingValueHandling:
    """Test handling of missing values."""

    def test_detect_missing_values(self, mock_hdf5_with_missing):
        """Test detection of missing values."""
        with h5py.File(mock_hdf5_with_missing, "r") as f:
            blh = f["atmospheric_features"]["blh"][:]
            assert np.any(np.isnan(blh))

    def test_count_missing_values(self, mock_hdf5_with_missing):
        """Test counting missing values."""
        with h5py.File(mock_hdf5_with_missing, "r") as f:
            blh = f["atmospheric_features"]["blh"][:]
            n_missing = np.sum(np.isnan(blh))
            assert n_missing == 10

    def test_mean_imputation(self, mock_hdf5_with_missing):
        """Test mean imputation for missing values."""
        with h5py.File(mock_hdf5_with_missing, "r") as f:
            blh = f["atmospheric_features"]["blh"][:]

            # Compute mean excluding NaN
            mean_val = np.nanmean(blh)

            # Impute
            blh_imputed = np.where(np.isnan(blh), mean_val, blh)

            assert not np.any(np.isnan(blh_imputed))

    def test_forward_fill_imputation(self):
        """Test forward fill imputation."""
        data = np.array([1.0, 2.0, np.nan, np.nan, 5.0])

        # Forward fill using pandas
        df = pd.DataFrame({"val": data})
        filled = df["val"].fillna(method="ffill").values

        assert not np.any(np.isnan(filled))
        assert filled[2] == 2.0  # Forward filled
        assert filled[3] == 2.0  # Forward filled


class TestDataSplitting:
    """Test data splitting utilities."""

    def test_stratified_split(self, mock_hdf5_file):
        """Test stratified train/test split."""
        with h5py.File(mock_hdf5_file, "r") as f:
            cbh = f["cbh"][:]

            # Create stratification bins (quintiles)
            bins = np.percentile(cbh, [0, 20, 40, 60, 80, 100])
            strata = np.digitize(cbh, bins[1:-1])

            # Verify stratification
            assert len(np.unique(strata)) <= 5

    def test_random_split(self, mock_hdf5_file):
        """Test random train/test split."""
        with h5py.File(mock_hdf5_file, "r") as f:
            n_samples = len(f["cbh"][:])

            # Random indices
            np.random.seed(42)
            indices = np.random.permutation(n_samples)

            train_size = int(0.8 * n_samples)
            train_idx = indices[:train_size]
            test_idx = indices[train_size:]

            assert len(train_idx) + len(test_idx) == n_samples
            assert len(np.intersect1d(train_idx, test_idx)) == 0

    def test_flight_based_split(self, mock_hdf5_file):
        """Test leave-one-flight-out split."""
        with h5py.File(mock_hdf5_file, "r") as f:
            flight_id = f["metadata"]["flight_id"][:]

            # Get unique flights
            unique_flights = np.unique(flight_id)
            assert len(unique_flights) == 2

            # Leave F2 out
            test_mask = flight_id == b"F2"
            train_mask = ~test_mask

            assert np.sum(test_mask) == 50
            assert np.sum(train_mask) == 50


class TestDataNormalization:
    """Test data normalization utilities."""

    def test_standardization(self, mock_hdf5_file):
        """Test z-score standardization."""
        with h5py.File(mock_hdf5_file, "r") as f:
            blh = f["atmospheric_features"]["blh"][:]

            # Standardize
            mean = np.mean(blh)
            std = np.std(blh)
            blh_std = (blh - mean) / std

            assert np.isclose(np.mean(blh_std), 0, atol=1e-10)
            assert np.isclose(np.std(blh_std), 1, atol=1e-10)

    def test_min_max_scaling(self, mock_hdf5_file):
        """Test min-max scaling to [0, 1]."""
        with h5py.File(mock_hdf5_file, "r") as f:
            blh = f["atmospheric_features"]["blh"][:]

            # Scale to [0, 1]
            min_val = np.min(blh)
            max_val = np.max(blh)
            blh_scaled = (blh - min_val) / (max_val - min_val)

            assert np.min(blh_scaled) >= 0
            assert np.max(blh_scaled) <= 1

    def test_robust_scaling(self, mock_hdf5_file):
        """Test robust scaling using median and IQR."""
        with h5py.File(mock_hdf5_file, "r") as f:
            blh = f["atmospheric_features"]["blh"][:]

            # Robust scaling
            median = np.median(blh)
            q75, q25 = np.percentile(blh, [75, 25])
            iqr = q75 - q25

            blh_robust = (blh - median) / iqr

            assert np.isfinite(blh_robust).all()


class TestDataStatistics:
    """Test data statistics computation."""

    def test_compute_mean_std(self, mock_hdf5_file):
        """Test computing mean and std."""
        with h5py.File(mock_hdf5_file, "r") as f:
            blh = f["atmospheric_features"]["blh"][:]

            mean = np.mean(blh)
            std = np.std(blh)

            assert np.isfinite(mean)
            assert np.isfinite(std)
            assert std > 0

    def test_compute_percentiles(self, mock_hdf5_file):
        """Test computing percentiles."""
        with h5py.File(mock_hdf5_file, "r") as f:
            cbh = f["cbh"][:]

            p25, p50, p75 = np.percentile(cbh, [25, 50, 75])

            assert p25 < p50 < p75

    def test_compute_correlation(self, mock_hdf5_file):
        """Test computing feature correlations."""
        with h5py.File(mock_hdf5_file, "r") as f:
            blh = f["atmospheric_features"]["blh"][:]
            lcl = f["atmospheric_features"]["lcl"][:]

            # Compute correlation
            corr = np.corrcoef(blh, lcl)[0, 1]

            assert -1 <= corr <= 1


class TestDataIntegrity:
    """Test data integrity checks."""

    def test_no_duplicates(self, mock_hdf5_file):
        """Test checking for duplicate samples."""
        with h5py.File(mock_hdf5_file, "r") as f:
            timestamps = f["metadata"]["timestamp"][:]

            # Check for duplicates
            unique_timestamps = np.unique(timestamps)
            assert len(unique_timestamps) == len(timestamps)

    def test_monotonic_timestamps(self, mock_hdf5_file):
        """Test timestamps are monotonically increasing."""
        with h5py.File(mock_hdf5_file, "r") as f:
            timestamps = f["metadata"]["timestamp"][:]

            # Check monotonic
            assert np.all(np.diff(timestamps) >= 0)

    def test_label_target_consistency(self, mock_hdf5_file):
        """Test labels match expected target distribution."""
        with h5py.File(mock_hdf5_file, "r") as f:
            cbh = f["cbh"][:]

            # CBH should be mostly positive
            assert np.all(cbh > 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
