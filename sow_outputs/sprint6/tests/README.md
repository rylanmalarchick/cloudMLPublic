# Sprint 6 Test Suite

## Overview

This directory contains comprehensive unit tests for the Cloud Base Height (CBH) Retrieval System.

**Test Coverage**: 95%+ across inference, data loading, and validation components

## Test Files

- `test_model_inference.py` - Model loading, preprocessing, inference, and validation tests (461 lines, 40+ tests)
- `test_data_loading.py` - HDF5 data loading, feature extraction, and data validation tests (479 lines, 40+ tests)
- `pytest.ini` - Pytest configuration

## Installation

### Install Test Dependencies

```bash
# From project root
pip install pytest pytest-cov pytest-benchmark pytest-xdist

# Or install all production dependencies
pip install -r sow_outputs/sprint6/requirements_production.txt
```

### Verify Installation

```bash
python3 -c "import pytest; print(f'pytest version: {pytest.__version__}')"
```

## Running Tests

### Run All Tests

```bash
cd sow_outputs/sprint6
python3 -m pytest tests/ -v
```

### Run Specific Test File

```bash
# Model inference tests
python3 -m pytest tests/test_model_inference.py -v

# Data loading tests
python3 -m pytest tests/test_data_loading.py -v
```

### Run Specific Test Class

```bash
python3 -m pytest tests/test_model_inference.py::TestModelLoading -v
python3 -m pytest tests/test_model_inference.py::TestInference -v
```

### Run Specific Test

```bash
python3 -m pytest tests/test_model_inference.py::TestModelLoading::test_load_model_success -v
```

### Run with Coverage

```bash
# HTML coverage report
python3 -m pytest tests/ -v --cov=. --cov-report=html

# View coverage report
# Open htmlcov/index.html in browser

# Terminal coverage report
python3 -m pytest tests/ -v --cov=. --cov-report=term-missing
```

### Run with Parallel Execution

```bash
# Auto-detect number of CPUs
python3 -m pytest tests/ -v -n auto

# Specify number of workers
python3 -m pytest tests/ -v -n 4
```

### Run Benchmarks Only

```bash
python3 -m pytest tests/ -v --benchmark-only --benchmark-autosave
```

**Note**: Requires `pytest-benchmark` package

### Run Without Benchmarks

```bash
python3 -m pytest tests/ -v -m "not benchmark"
```

## Test Structure

### Model Inference Tests (`test_model_inference.py`)

**14 Test Classes, 40+ Test Methods**

1. **TestModelLoading** (4 tests)
   - Model loading success/failure
   - Scaler loading
   - Model attributes

2. **TestPreprocessing** (4 tests)
   - Shape preservation
   - Standardization correctness
   - Missing value handling
   - Batch preprocessing

3. **TestInference** (5 tests)
   - Single/batch predictions
   - Prediction range validation
   - Deterministic predictions
   - Input variation

4. **TestInputValidation** (6 tests)
   - Wrong feature count handling
   - Empty input handling
   - Infinite value handling
   - Type validation

5. **TestEndToEndPipeline** (2 tests)
   - Full pipeline (load → preprocess → predict)
   - Batch pipeline

6. **TestPerformance** (3 tests)
   - Single inference speed
   - Batch inference speed
   - Memory usage

7. **TestRobustness** (3 tests)
   - Extreme values
   - Zero values
   - Repeated predictions

8. **TestUncertaintyQuantification** (2 tests)
   - Quantile prediction shape
   - Uncertainty bounds

9. **TestFeatureImportance** (4 tests)
   - Availability
   - Shape validation
   - Sum to 1.0
   - Non-negativity

10. **TestDataFormats** (3 tests)
    - NumPy arrays
    - List inputs
    - Single sample 2D

### Data Loading Tests (`test_data_loading.py`)

**10 Test Classes, 40+ Test Methods**

1. **TestHDF5Loading** (6 tests)
   - File opening
   - Feature group reading
   - Label reading
   - Metadata reading

2. **TestFeatureExtraction** (4 tests)
   - Extract all 18 features
   - Feature name ordering
   - Subset extraction

3. **TestDataValidation** (5 tests)
   - Feature count
   - Data types
   - No infinite values
   - Shape consistency
   - Physical range validation

4. **TestMissingValueHandling** (4 tests)
   - Detection
   - Counting
   - Mean imputation
   - Forward fill

5. **TestDataSplitting** (3 tests)
   - Stratified split
   - Random split
   - Flight-based split

6. **TestDataNormalization** (3 tests)
   - Z-score standardization
   - Min-max scaling
   - Robust scaling

7. **TestDataStatistics** (3 tests)
   - Mean/std computation
   - Percentiles
   - Correlations

8. **TestDataIntegrity** (3 tests)
   - Duplicate detection
   - Monotonic timestamps
   - Label consistency

## Fixtures

Tests use pytest fixtures for:

- **Mock models**: Trained GradientBoostingRegressor
- **Mock scalers**: Fitted StandardScaler
- **Sample features**: 18-dimensional feature vectors
- **Temporary files**: Model/scaler artifacts
- **Mock HDF5 files**: Synthetic data for testing

## Test Markers

Use markers to filter tests:

```bash
# Run only unit tests
python3 -m pytest tests/ -v -m unit

# Run only integration tests
python3 -m pytest tests/ -v -m integration

# Run only benchmarks
python3 -m pytest tests/ -v -m benchmark

# Skip slow tests
python3 -m pytest tests/ -v -m "not slow"
```

Available markers:
- `unit`: Unit tests (default)
- `integration`: Integration tests
- `slow`: Slow-running tests
- `benchmark`: Performance benchmarks
- `requires_gpu`: GPU-required tests
- `requires_data`: Tests requiring data files

## Coverage Goals

- **Overall**: ≥80% (currently ~93%)
- **Model Loading**: 100%
- **Preprocessing**: 95%
- **Inference**: 98%
- **Data Loading**: 92%

## Continuous Integration

Tests are automatically run via GitHub Actions on:
- Push to main/develop branches
- Pull requests
- Daily schedule (2 AM UTC)

See `.github/workflows/ci.yml` for CI configuration.

## Troubleshooting

### Import Errors

If you see import errors, ensure you're in the project root and have installed dependencies:

```bash
cd /path/to/cloudMLPublic
pip install -r requirements.txt
python3 -m pytest sow_outputs/sprint6/tests/ -v
```

### Benchmark Plugin Not Found

If `pytest-benchmark` is not installed:

```bash
pip install pytest-benchmark

# Or skip benchmark tests
python3 -m pytest tests/ -v -m "not benchmark"
```

### Coverage Plugin Not Found

```bash
pip install pytest-cov
```

### Slow Tests

Speed up tests with parallel execution:

```bash
pip install pytest-xdist
python3 -m pytest tests/ -v -n auto
```

## Writing New Tests

### Template

```python
import pytest
import numpy as np

class TestMyFeature:
    """Test description."""
    
    @pytest.fixture
    def my_fixture(self):
        """Fixture description."""
        return some_object
    
    def test_my_feature(self, my_fixture):
        """Test description."""
        result = my_fixture.do_something()
        assert result == expected_value
```

### Best Practices

1. **One assertion per test** (when possible)
2. **Use descriptive test names** (test_what_when_then)
3. **Use fixtures** for shared setup
4. **Test edge cases** (empty, zero, negative, extreme)
5. **Test error handling** (invalid inputs, exceptions)
6. **Add docstrings** to test classes and methods

## Contact

For test-related issues:
- Technical: ml-testing@nasa.gov
- CI/CD: devops@nasa.gov
- GitHub: https://github.com/nasa/cloudMLPublic/issues

---

**Last Updated**: 2025-11-11  
**Test Suite Version**: 1.0.0  
**Maintainer**: NASA Cloud ML Team