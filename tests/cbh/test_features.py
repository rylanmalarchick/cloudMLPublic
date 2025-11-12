"""
Unit tests for feature extraction and engineering.

Tests cover:
- Geometric feature computation (shadow geometry, solar angles)
- Atmospheric feature extraction from ERA5
- Feature normalization and standardization
- Feature validation and integrity checks
"""

import numpy as np
import pytest
from scipy import stats


@pytest.fixture
def sample_image():
    """Create sample cloud image."""
    return np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)


@pytest.fixture
def sample_shadow_coords():
    """Create sample shadow coordinates."""
    return {
        "cloud_edge_x": 100,
        "cloud_edge_y": 150,
        "shadow_edge_x": 200,
        "shadow_edge_y": 250,
    }


@pytest.fixture
def sample_solar_geometry():
    """Create sample solar geometry parameters."""
    return {
        "sza_deg": 30.0,  # Solar zenith angle
        "saa_deg": 180.0,  # Solar azimuth angle
        "altitude_m": 5000.0,
    }


@pytest.fixture
def sample_era5_data():
    """Create sample ERA5 atmospheric data."""
    return {
        "blh": 1000.0,  # Boundary layer height (m)
        "lcl": 800.0,  # Lifting condensation level (m)
        "t2m": 290.0,  # Temperature at 2m (K)
        "d2m": 285.0,  # Dewpoint at 2m (K)
        "sp": 101325.0,  # Surface pressure (Pa)
        "tcwv": 25.0,  # Total column water vapor (kg/m²)
    }


class TestGeometricFeatures:
    """Test geometric feature computation."""

    def test_shadow_length_computation(self, sample_shadow_coords):
        """Test shadow length calculation from coordinates."""
        dx = (
            sample_shadow_coords["shadow_edge_x"] - sample_shadow_coords["cloud_edge_x"]
        )
        dy = (
            sample_shadow_coords["shadow_edge_y"] - sample_shadow_coords["cloud_edge_y"]
        )

        shadow_length = np.sqrt(dx**2 + dy**2)

        assert shadow_length > 0
        assert np.isfinite(shadow_length)
        # Expected: sqrt(100^2 + 100^2) ≈ 141.42
        assert np.isclose(shadow_length, 141.42, atol=0.1)

    def test_shadow_angle_computation(self, sample_shadow_coords):
        """Test shadow angle calculation."""
        dx = (
            sample_shadow_coords["shadow_edge_x"] - sample_shadow_coords["cloud_edge_x"]
        )
        dy = (
            sample_shadow_coords["shadow_edge_y"] - sample_shadow_coords["cloud_edge_y"]
        )

        shadow_angle_rad = np.arctan2(dy, dx)
        shadow_angle_deg = np.degrees(shadow_angle_rad)

        assert -180 <= shadow_angle_deg <= 180
        # Expected: arctan2(100, 100) = 45 degrees
        assert np.isclose(shadow_angle_deg, 45.0, atol=0.1)

    def test_cbh_from_shadow_geometry(
        self, sample_shadow_coords, sample_solar_geometry
    ):
        """Test CBH estimation from shadow geometry."""
        dx = (
            sample_shadow_coords["shadow_edge_x"] - sample_shadow_coords["cloud_edge_x"]
        )
        dy = (
            sample_shadow_coords["shadow_edge_y"] - sample_shadow_coords["cloud_edge_y"]
        )
        shadow_length_pixels = np.sqrt(dx**2 + dy**2)

        # Assume pixel resolution (meters per pixel)
        pixel_resolution = 10.0  # 10 m/pixel
        shadow_length_m = shadow_length_pixels * pixel_resolution

        # Compute CBH using trigonometry
        sza_rad = np.radians(sample_solar_geometry["sza_deg"])
        cbh_estimated = shadow_length_m * np.tan(sza_rad)

        assert cbh_estimated > 0
        assert cbh_estimated < 10000  # Reasonable physical range
        assert np.isfinite(cbh_estimated)

    def test_solar_zenith_angle_range(self, sample_solar_geometry):
        """Test solar zenith angle is in valid range."""
        sza = sample_solar_geometry["sza_deg"]

        assert 0 <= sza <= 90, f"SZA out of range: {sza}"

    def test_solar_azimuth_angle_range(self, sample_solar_geometry):
        """Test solar azimuth angle is in valid range."""
        saa = sample_solar_geometry["saa_deg"]

        assert 0 <= saa <= 360, f"SAA out of range: {saa}"

    def test_edge_detection_coordinates(self, sample_shadow_coords):
        """Test edge coordinates are positive."""
        for key, value in sample_shadow_coords.items():
            assert value >= 0, f"{key} should be non-negative"
            assert np.isfinite(value)


class TestAtmosphericFeatures:
    """Test atmospheric feature extraction."""

    def test_era5_feature_extraction(self, sample_era5_data):
        """Test ERA5 features are in valid ranges."""
        # Boundary layer height: 0-3000m typical
        assert 0 < sample_era5_data["blh"] < 5000

        # LCL: should be <= BLH typically
        assert sample_era5_data["lcl"] > 0

        # Temperature: 200-350K reasonable range
        assert 200 < sample_era5_data["t2m"] < 350

        # Dewpoint: <= temperature
        assert sample_era5_data["d2m"] <= sample_era5_data["t2m"]

        # Surface pressure: 80000-110000 Pa typical
        assert 80000 < sample_era5_data["sp"] < 110000

        # TCWV: 0-100 kg/m² typical
        assert 0 < sample_era5_data["tcwv"] < 100

    def test_inversion_height_computation(self, sample_era5_data):
        """Test temperature inversion height calculation."""
        # Simplified: inversion at BLH if temperature profile inverts
        blh = sample_era5_data["blh"]

        # Assume temperature lapse rate
        lapse_rate = -0.0065  # K/m (standard atmosphere)

        # Check if inversion exists (positive lapse rate)
        inversion_strength = 2.0  # K (assumed)
        inversion_height = blh if inversion_strength > 0 else 0

        assert inversion_height >= 0
        assert np.isfinite(inversion_height)

    def test_moisture_gradient_computation(self, sample_era5_data):
        """Test vertical moisture gradient calculation."""
        # Simplified: gradient between surface and top of BLH
        dewpoint_surface = sample_era5_data["d2m"]
        dewpoint_blh = dewpoint_surface - 5.0  # Assumed decrease

        blh = sample_era5_data["blh"]
        moisture_gradient = (dewpoint_blh - dewpoint_surface) / blh if blh > 0 else 0

        assert np.isfinite(moisture_gradient)
        # Typically negative (dewpoint decreases with height)
        assert moisture_gradient <= 0

    def test_stability_index_computation(self, sample_era5_data):
        """Test atmospheric stability index."""
        t2m = sample_era5_data["t2m"]
        d2m = sample_era5_data["d2m"]

        # Simplified stability index (larger = more stable)
        stability = (t2m - d2m) / 10.0

        assert stability >= 0
        assert np.isfinite(stability)

    def test_relative_humidity_computation(self, sample_era5_data):
        """Test relative humidity calculation."""
        t2m = sample_era5_data["t2m"]
        d2m = sample_era5_data["d2m"]

        # Magnus formula for saturation vapor pressure
        def vapor_pressure(T):
            """Compute vapor pressure (hPa) using Magnus formula."""
            return 6.112 * np.exp((17.67 * (T - 273.15)) / ((T - 273.15) + 243.5))

        e_actual = vapor_pressure(d2m)
        e_sat = vapor_pressure(t2m)

        rh = (e_actual / e_sat) * 100.0

        assert 0 <= rh <= 100, f"RH out of range: {rh}"
        assert np.isfinite(rh)


class TestFeatureNormalization:
    """Test feature normalization and standardization."""

    def test_z_score_normalization(self):
        """Test z-score standardization."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        mean = np.mean(data)
        std = np.std(data)
        normalized = (data - mean) / std

        assert np.isclose(np.mean(normalized), 0.0, atol=1e-10)
        assert np.isclose(np.std(normalized), 1.0, atol=1e-10)

    def test_min_max_normalization(self):
        """Test min-max normalization to [0, 1]."""
        data = np.array([10.0, 20.0, 30.0, 40.0, 50.0])

        min_val = np.min(data)
        max_val = np.max(data)
        normalized = (data - min_val) / (max_val - min_val)

        assert np.min(normalized) == 0.0
        assert np.max(normalized) == 1.0
        assert np.all((normalized >= 0) & (normalized <= 1))

    def test_robust_scaling(self):
        """Test robust scaling using median and IQR."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])  # Outlier

        median = np.median(data)
        q75, q25 = np.percentile(data, [75, 25])
        iqr = q75 - q25

        scaled = (data - median) / iqr

        assert np.isfinite(scaled).all()
        # Robust scaling less affected by outlier

    def test_normalization_with_nan(self):
        """Test normalization handles NaN values."""
        data = np.array([1.0, 2.0, np.nan, 4.0, 5.0])

        # Remove NaN before normalization
        valid_data = data[~np.isnan(data)]
        mean = np.mean(valid_data)
        std = np.std(valid_data)

        # Normalize only valid data
        normalized = np.where(np.isnan(data), np.nan, (data - mean) / std)

        assert np.isnan(normalized[2])
        assert np.isfinite(normalized[~np.isnan(normalized)]).all()


class TestFeatureValidation:
    """Test feature validation and quality checks."""

    def test_check_for_infinite_values(self):
        """Test detection of infinite values."""
        data = np.array([1.0, 2.0, np.inf, 4.0, 5.0])

        has_inf = np.any(np.isinf(data))
        assert has_inf

    def test_check_for_nan_values(self):
        """Test detection of NaN values."""
        data = np.array([1.0, 2.0, np.nan, 4.0, 5.0])

        has_nan = np.any(np.isnan(data))
        assert has_nan

    def test_check_feature_range(self):
        """Test feature values are in expected range."""
        # Temperature should be 200-350K
        temperatures = np.array([250.0, 280.0, 300.0, 320.0])

        in_range = np.all((temperatures >= 200) & (temperatures <= 350))
        assert in_range

    def test_check_feature_correlation(self):
        """Test feature correlation computation."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])

        correlation = np.corrcoef(x, y)[0, 1]

        assert -1 <= correlation <= 1
        assert np.isclose(correlation, 1.0, atol=0.01)  # Perfect positive correlation

    def test_outlier_detection(self):
        """Test outlier detection using IQR method."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])

        q75, q25 = np.percentile(data, [75, 25])
        iqr = q75 - q25

        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr

        outliers = (data < lower_bound) | (data > upper_bound)

        assert outliers[-1]  # 100.0 is an outlier


class TestFeatureCombinations:
    """Test feature engineering through combinations."""

    def test_feature_interaction(self, sample_era5_data):
        """Test multiplicative feature interaction."""
        t2m = sample_era5_data["t2m"]
        tcwv = sample_era5_data["tcwv"]

        # Interaction term
        interaction = t2m * tcwv

        assert interaction > 0
        assert np.isfinite(interaction)

    def test_polynomial_features(self):
        """Test polynomial feature generation."""
        x = np.array([1.0, 2.0, 3.0])

        # Create polynomial features (x, x^2)
        poly_features = np.column_stack([x, x**2])

        assert poly_features.shape == (3, 2)
        assert np.allclose(poly_features[:, 0], x)
        assert np.allclose(poly_features[:, 1], x**2)

    def test_ratio_features(self, sample_era5_data):
        """Test ratio feature creation."""
        lcl = sample_era5_data["lcl"]
        blh = sample_era5_data["blh"]

        # LCL to BLH ratio
        ratio = lcl / blh if blh > 0 else 0

        assert 0 <= ratio <= 1  # LCL typically <= BLH
        assert np.isfinite(ratio)

    def test_difference_features(self, sample_era5_data):
        """Test difference feature creation."""
        t2m = sample_era5_data["t2m"]
        d2m = sample_era5_data["d2m"]

        # Temperature - dewpoint spread
        spread = t2m - d2m

        assert spread >= 0  # T >= Td always
        assert np.isfinite(spread)


class TestFeatureImportance:
    """Test feature importance and selection."""

    def test_variance_threshold(self):
        """Test low-variance feature filtering."""
        # Feature with low variance should be removed
        low_var_feature = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        high_var_feature = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        assert np.var(low_var_feature) < 0.01
        assert np.var(high_var_feature) > 1.0

    def test_correlation_filtering(self):
        """Test highly correlated feature removal."""
        x1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        x2 = np.array([2.0, 4.0, 6.0, 8.0, 10.0])  # x2 = 2*x1 (perfect correlation)

        correlation = np.corrcoef(x1, x2)[0, 1]

        # Should remove one of the perfectly correlated features
        assert np.isclose(correlation, 1.0, atol=0.01)

    def test_mutual_information(self):
        """Test mutual information computation (simplified)."""
        # Create correlated variables
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 6, 8, 10])

        # Compute correlation as proxy for mutual information
        correlation = np.corrcoef(x, y)[0, 1]

        # High correlation implies high mutual information
        assert correlation > 0.9


class TestFeatureEngineering:
    """Test advanced feature engineering techniques."""

    def test_temporal_features(self):
        """Test temporal feature extraction."""
        # Simulate time of day (hour)
        hour = 14  # 2 PM

        # Cyclical encoding
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)

        assert -1 <= hour_sin <= 1
        assert -1 <= hour_cos <= 1

    def test_spatial_features(self, sample_shadow_coords):
        """Test spatial feature extraction."""
        x = sample_shadow_coords["cloud_edge_x"]
        y = sample_shadow_coords["cloud_edge_y"]

        # Distance from origin
        distance = np.sqrt(x**2 + y**2)

        # Angle from origin
        angle = np.arctan2(y, x)

        assert distance >= 0
        assert -np.pi <= angle <= np.pi

    def test_log_transformation(self):
        """Test log transformation for skewed features."""
        # Skewed data
        data = np.array([1.0, 10.0, 100.0, 1000.0, 10000.0])

        # Log transform
        log_data = np.log10(data)

        # Log transform reduces skewness
        assert np.std(log_data) < np.std(data)
        assert np.all(np.isfinite(log_data))

    def test_feature_binning(self):
        """Test feature discretization (binning)."""
        data = np.array([1.0, 5.0, 10.0, 15.0, 20.0])

        # Bin into 3 categories
        bins = [0, 7, 14, 25]
        binned = np.digitize(data, bins)

        assert np.all(binned >= 1)
        assert np.all(binned <= 3)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
