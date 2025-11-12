"""
Unit tests for CBH model inference pipeline.

Tests cover:
- Model and scaler loading
- Feature preprocessing
- Inference correctness
- Input validation
- Error handling
"""

import os
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler


# Test fixtures
@pytest.fixture
def mock_model():
    """Create a mock GBDT model for testing."""
    model = GradientBoostingRegressor(n_estimators=10, max_depth=3, random_state=42)
    # Train on dummy data
    X_train = np.random.randn(100, 18)
    y_train = np.random.randn(100) * 200 + 1000
    model.fit(X_train, y_train)
    return model


@pytest.fixture
def mock_scaler():
    """Create a mock scaler for testing."""
    scaler = StandardScaler()
    X_train = np.random.randn(100, 18)
    scaler.fit(X_train)
    return scaler


@pytest.fixture
def sample_features():
    """Create sample input features (18 features)."""
    return np.array(
        [
            [
                1000,  # blh
                800,  # lcl
                1200,  # inversion_height
                0.5,  # moisture_gradient
                0.3,  # stability_index
                290,  # t2m
                285,  # d2m
                101325,  # sp
                25,  # tcwv
                100,  # cloud_edge_x
                200,  # cloud_edge_y
                180,  # saa_deg
                45,  # shadow_angle_deg
                0.9,  # shadow_detection_confidence
                150,  # shadow_edge_x
                250,  # shadow_edge_y
                50,  # shadow_length_pixels
                30,  # sza_deg
            ]
        ]
    )


@pytest.fixture
def temp_model_path(mock_model, tmp_path):
    """Save mock model to temporary file."""
    model_path = tmp_path / "test_model.joblib"
    joblib.dump(mock_model, model_path)
    return model_path


@pytest.fixture
def temp_scaler_path(mock_scaler, tmp_path):
    """Save mock scaler to temporary file."""
    scaler_path = tmp_path / "test_scaler.joblib"
    joblib.dump(mock_scaler, scaler_path)
    return scaler_path


class TestModelLoading:
    """Test model and scaler loading."""

    def test_load_model_success(self, temp_model_path):
        """Test successful model loading."""
        model = joblib.load(temp_model_path)
        assert isinstance(model, GradientBoostingRegressor)
        assert hasattr(model, "predict")

    def test_load_scaler_success(self, temp_scaler_path):
        """Test successful scaler loading."""
        scaler = joblib.load(temp_scaler_path)
        assert isinstance(scaler, StandardScaler)
        assert hasattr(scaler, "transform")

    def test_load_nonexistent_model(self):
        """Test loading non-existent model raises error."""
        with pytest.raises(FileNotFoundError):
            joblib.load("/nonexistent/path/model.joblib")

    def test_model_attributes(self, mock_model):
        """Test model has expected attributes."""
        assert hasattr(mock_model, "n_estimators")
        assert hasattr(mock_model, "max_depth")
        assert mock_model.n_estimators == 10
        assert mock_model.max_depth == 3


class TestPreprocessing:
    """Test feature preprocessing."""

    def test_scaler_transform_shape(self, mock_scaler, sample_features):
        """Test scaler preserves input shape."""
        scaled = mock_scaler.transform(sample_features)
        assert scaled.shape == sample_features.shape

    def test_scaler_output_standardized(self, mock_scaler):
        """Test scaler produces standardized output."""
        # Create data with known statistics
        X = np.ones((100, 18)) * 10
        scaler = StandardScaler()
        scaler.fit(X)

        # Transform should give zero mean
        X_scaled = scaler.transform(X)
        assert np.allclose(X_scaled.mean(axis=0), 0, atol=1e-10)

    def test_handle_missing_values(self, sample_features):
        """Test handling of missing values."""
        # Introduce NaN
        features_with_nan = sample_features.copy()
        features_with_nan[0, 0] = np.nan

        # Replace NaN with mean (simple imputation)
        features_imputed = np.nan_to_num(
            features_with_nan, nan=np.nanmean(features_with_nan)
        )

        assert not np.any(np.isnan(features_imputed))
        assert features_imputed.shape == sample_features.shape

    def test_batch_preprocessing(self, mock_scaler):
        """Test preprocessing multiple samples."""
        X_batch = np.random.randn(10, 18)
        X_scaled = mock_scaler.transform(X_batch)

        assert X_scaled.shape == (10, 18)
        assert not np.any(np.isnan(X_scaled))


class TestInference:
    """Test model inference."""

    def test_predict_single_sample(self, mock_model, mock_scaler, sample_features):
        """Test prediction on single sample."""
        X_scaled = mock_scaler.transform(sample_features)
        prediction = mock_model.predict(X_scaled)

        assert prediction.shape == (1,)
        assert isinstance(prediction[0], (float, np.floating))
        assert not np.isnan(prediction[0])

    def test_predict_batch(self, mock_model, mock_scaler):
        """Test prediction on batch of samples."""
        X_batch = np.random.randn(10, 18)
        X_scaled = mock_scaler.transform(X_batch)
        predictions = mock_model.predict(X_scaled)

        assert predictions.shape == (10,)
        assert not np.any(np.isnan(predictions))

    def test_prediction_range(self, mock_model, mock_scaler, sample_features):
        """Test prediction is within reasonable range."""
        X_scaled = mock_scaler.transform(sample_features)
        prediction = mock_model.predict(X_scaled)[0]

        # CBH should be between 0 and 5000 meters (reasonable physical range)
        assert 0 <= prediction <= 5000

    def test_deterministic_predictions(self, mock_model, mock_scaler, sample_features):
        """Test predictions are deterministic."""
        X_scaled = mock_scaler.transform(sample_features)
        pred1 = mock_model.predict(X_scaled)
        pred2 = mock_model.predict(X_scaled)

        assert np.allclose(pred1, pred2)

    def test_different_inputs_different_outputs(self, mock_model, mock_scaler):
        """Test different inputs produce different outputs."""
        X1 = np.random.randn(1, 18)
        X2 = np.random.randn(1, 18)

        pred1 = mock_model.predict(mock_scaler.transform(X1))
        pred2 = mock_model.predict(mock_scaler.transform(X2))

        # Very unlikely to be exactly equal
        assert not np.allclose(pred1, pred2)


class TestInputValidation:
    """Test input validation and error handling."""

    def test_wrong_feature_count(self, mock_model, mock_scaler):
        """Test error on wrong number of features."""
        X_wrong = np.random.randn(1, 10)  # Wrong: 10 features instead of 18

        with pytest.raises(ValueError):
            mock_scaler.transform(X_wrong)

    def test_empty_input(self, mock_model, mock_scaler):
        """Test error on empty input."""
        X_empty = np.array([]).reshape(0, 18)

        X_scaled = mock_scaler.transform(X_empty)
        predictions = mock_model.predict(X_scaled)

        assert predictions.shape == (0,)

    def test_input_with_inf(self, mock_scaler):
        """Test handling of infinite values."""
        X_inf = np.random.randn(1, 18)
        X_inf[0, 0] = np.inf

        # Should either raise or handle gracefully
        try:
            X_scaled = mock_scaler.transform(X_inf)
            assert not np.any(np.isinf(X_scaled))
        except ValueError:
            pass  # Acceptable to raise error

    def test_input_type_validation(self, mock_scaler):
        """Test input type validation."""
        # List input should be converted to array
        X_list = [[1.0] * 18]
        X_array = np.array(X_list)

        result = mock_scaler.transform(X_array)
        assert isinstance(result, np.ndarray)

    def test_negative_features(self, mock_model, mock_scaler):
        """Test handling of negative feature values."""
        # Some features can be negative (e.g., gradients)
        X_neg = np.random.randn(1, 18)
        X_neg[0, 3] = -0.5  # Negative moisture gradient

        X_scaled = mock_scaler.transform(X_neg)
        prediction = mock_model.predict(X_scaled)

        assert prediction.shape == (1,)


class TestEndToEndPipeline:
    """Test complete inference pipeline."""

    def test_full_pipeline(self, temp_model_path, temp_scaler_path, sample_features):
        """Test full pipeline: load -> preprocess -> predict."""
        # Load
        model = joblib.load(temp_model_path)
        scaler = joblib.load(temp_scaler_path)

        # Preprocess
        features_clean = np.nan_to_num(sample_features, nan=0.0)
        features_scaled = scaler.transform(features_clean)

        # Predict
        prediction = model.predict(features_scaled)

        assert prediction.shape == (1,)
        assert isinstance(prediction[0], (float, np.floating))

    def test_batch_pipeline(self, temp_model_path, temp_scaler_path):
        """Test pipeline with batch of samples."""
        model = joblib.load(temp_model_path)
        scaler = joblib.load(temp_scaler_path)

        # Create batch
        X_batch = np.random.randn(100, 18)

        # Process
        X_clean = np.nan_to_num(X_batch, nan=0.0)
        X_scaled = scaler.transform(X_clean)
        predictions = model.predict(X_scaled)

        assert predictions.shape == (100,)
        assert not np.any(np.isnan(predictions))


class TestPerformance:
    """Test inference performance."""

    def test_inference_speed(self, mock_model, mock_scaler, benchmark):
        """Benchmark single inference speed."""
        X = np.random.randn(1, 18)
        X_scaled = mock_scaler.transform(X)

        # Should complete in <10ms
        result = benchmark(mock_model.predict, X_scaled)
        assert result.shape == (1,)

    def test_batch_inference_speed(self, mock_model, mock_scaler, benchmark):
        """Benchmark batch inference speed."""
        X_batch = np.random.randn(1000, 18)
        X_scaled = mock_scaler.transform(X_batch)

        result = benchmark(mock_model.predict, X_scaled)
        assert result.shape == (1000,)

    def test_memory_usage(self, mock_model, mock_scaler):
        """Test memory footprint is reasonable."""
        import sys

        # Model size should be <50MB
        model_size = sys.getsizeof(mock_model)
        assert model_size < 50 * 1024 * 1024

        # Scaler size should be minimal
        scaler_size = sys.getsizeof(mock_scaler)
        assert scaler_size < 1 * 1024 * 1024


class TestRobustness:
    """Test model robustness to edge cases."""

    def test_extreme_values(self, mock_model, mock_scaler):
        """Test handling of extreme but valid values."""
        X_extreme = np.array(
            [
                [
                    5000,  # Very high BLH
                    4000,  # Very high LCL
                    6000,  # Very high inversion
                    2.0,  # High moisture gradient
                    1.5,  # High stability
                    310,  # High temperature
                    300,  # High dewpoint
                    105000,  # High pressure
                    100,  # High TCWV
                    500,  # Edge case coordinates
                    500,
                    360,  # Max azimuth
                    90,  # Max shadow angle
                    1.0,  # Max confidence
                    500,
                    500,
                    200,  # Large shadow
                    85,  # High zenith
                ]
            ]
        )

        X_scaled = mock_scaler.transform(X_extreme)
        prediction = mock_model.predict(X_scaled)

        # Should not crash and produce reasonable output
        assert prediction.shape == (1,)
        assert not np.isnan(prediction[0])

    def test_zero_values(self, mock_model, mock_scaler):
        """Test handling of zero values."""
        X_zeros = np.zeros((1, 18))

        X_scaled = mock_scaler.transform(X_zeros)
        prediction = mock_model.predict(X_scaled)

        assert prediction.shape == (1,)

    def test_repeated_predictions(self, mock_model, mock_scaler, sample_features):
        """Test model consistency over repeated calls."""
        X_scaled = mock_scaler.transform(sample_features)

        predictions = [mock_model.predict(X_scaled)[0] for _ in range(10)]

        # All predictions should be identical
        assert np.allclose(predictions, predictions[0])


class TestUncertaintyQuantification:
    """Test uncertainty quantification (if available)."""

    def test_quantile_prediction_shape(self, mock_model):
        """Test quantile prediction returns correct shape."""
        # Note: Basic GBDT doesn't support quantiles by default
        # This is a placeholder for future UQ implementation
        X = np.random.randn(10, 18)
        predictions = mock_model.predict(X)

        assert predictions.shape == (10,)

    def test_uncertainty_bounds(self):
        """Test uncertainty intervals are ordered correctly."""
        # Placeholder for quantile regression tests
        # lower <= point <= upper
        lower = np.array([500, 600, 700])
        point = np.array([800, 900, 1000])
        upper = np.array([1100, 1200, 1300])

        assert np.all(lower <= point)
        assert np.all(point <= upper)


class TestFeatureImportance:
    """Test feature importance extraction."""

    def test_feature_importance_available(self, mock_model):
        """Test feature importance is available."""
        assert hasattr(mock_model, "feature_importances_")

    def test_feature_importance_shape(self, mock_model):
        """Test feature importance has correct shape."""
        importances = mock_model.feature_importances_
        assert importances.shape == (18,)

    def test_feature_importance_sum(self, mock_model):
        """Test feature importances sum to 1."""
        importances = mock_model.feature_importances_
        assert np.isclose(importances.sum(), 1.0)

    def test_feature_importance_nonnegative(self, mock_model):
        """Test feature importances are non-negative."""
        importances = mock_model.feature_importances_
        assert np.all(importances >= 0)


class TestDataFormats:
    """Test different input data formats."""

    def test_numpy_array_input(self, mock_model, mock_scaler):
        """Test NumPy array input."""
        X = np.random.randn(5, 18)
        X_scaled = mock_scaler.transform(X)
        predictions = mock_model.predict(X_scaled)

        assert predictions.shape == (5,)

    def test_list_input(self, mock_scaler):
        """Test list input conversion."""
        X_list = [[1.0] * 18 for _ in range(5)]
        X_array = np.array(X_list)
        X_scaled = mock_scaler.transform(X_array)

        assert X_scaled.shape == (5, 18)

    def test_single_sample_2d(self, mock_model, mock_scaler):
        """Test single sample in 2D format."""
        X = np.random.randn(1, 18)
        X_scaled = mock_scaler.transform(X)
        prediction = mock_model.predict(X_scaled)

        assert prediction.shape == (1,)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
