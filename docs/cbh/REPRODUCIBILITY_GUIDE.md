# Reproducibility Guide: Cloud Base Height Retrieval System

**Version**: 1.0.0  
**Last Updated**: 2025-11-11  
**Author**: NASA Cloud ML Team

---

## Table of Contents

1. [Overview](#overview)
2. [Environment Setup](#environment-setup)
3. [Data Preparation](#data-preparation)
4. [Model Training](#model-training)
5. [Validation & Testing](#validation--testing)
6. [Reproducing Results](#reproducing-results)
7. [Common Issues](#common-issues)
8. [Verification Checklist](#verification-checklist)

---

## Overview

This guide provides step-by-step instructions to exactly reproduce all results from Sprint 6 Phase 1-4, including:

- Cross-validation results (R² = 0.744 ± 0.037)
- Production model training
- Uncertainty quantification
- Ensemble experiments
- Domain adaptation tests
- Visualization suite

### Key Reproducibility Guarantees

- **Random Seed**: Fixed at 42 for all experiments
- **Environment**: Pinned dependencies in `requirements_production.txt`
- **Data**: Fixed dataset (Integrated_Features.hdf5, 933 samples)
- **Hyperparameters**: Documented in configuration files
- **Validation Strategy**: Stratified 5-fold cross-validation

---

## Environment Setup

### Step 1: Clone Repository

```bash
git clone https://github.com/nasa/cloudMLPublic.git
cd cloudMLPublic
```

### Step 2: Create Python Environment

**Option A: venv (recommended)**

```bash
python3.12 -m venv venv_sprint6
source venv_sprint6/bin/activate  # Linux/macOS
# OR
venv_sprint6\Scripts\activate  # Windows
```

**Option B: conda**

```bash
conda create -n cbh_sprint6 python=3.12
conda activate cbh_sprint6
```

### Step 3: Install Dependencies

```bash
# Install exact versions from pinned requirements
pip install --upgrade pip
pip install -r sow_outputs/sprint6/requirements_production.txt

# Verify installation
python -c "import sklearn; print(f'scikit-learn: {sklearn.__version__}')"
python -c "import numpy; print(f'numpy: {numpy.__version__}')"
python -c "import pandas; print(f'pandas: {pandas.__version__}')"
```

**Expected output:**
```
scikit-learn: 1.7.0
numpy: 2.2.6
pandas: 2.3.0
```

### Step 4: Verify Environment

```bash
python sow_outputs/sprint6/tests/test_environment.py
```

Create `test_environment.py`:

```python
import sys
import sklearn
import numpy as np
import pandas as pd
import h5py
import joblib

print("Environment Check:")
print(f"Python: {sys.version}")
print(f"scikit-learn: {sklearn.__version__}")
print(f"numpy: {np.__version__}")
print(f"pandas: {pd.__version__}")
print(f"h5py: {h5py.__version__}")
print(f"joblib: {joblib.__version__}")

# Check versions match
assert sklearn.__version__ == "1.7.0", f"Wrong sklearn version: {sklearn.__version__}"
assert np.__version__ == "2.2.6", f"Wrong numpy version: {np.__version__}"
assert pd.__version__ == "2.3.0", f"Wrong pandas version: {pd.__version__}"

print("\n Environment verified successfully!")
```

---

## Data Preparation

### Step 1: Locate Data File

```bash
# Data should be at:
ls -lh sow_outputs/integrated_features/Integrated_Features.hdf5
```

**Expected**: File size ~10-50 MB, HDF5 format

### Step 2: Verify Data Integrity

```python
import h5py
import numpy as np

# Load and verify
with h5py.File('sow_outputs/integrated_features/Integrated_Features.hdf5', 'r') as f:
    print("HDF5 Groups:", list(f.keys()))
    
    # Check structure
    assert 'atmospheric_features' in f
    assert 'geometric_features' in f
    assert 'cbh' in f
    
    # Check sample count
    cbh = f['cbh'][:]
    print(f"Number of samples: {len(cbh)}")
    assert len(cbh) == 933, f"Expected 933 samples, got {len(cbh)}"
    
    # Check feature count
    n_atmo = len(f['atmospheric_features'].keys())
    n_geom = len(f['geometric_features'].keys())
    print(f"Atmospheric features: {n_atmo}")
    print(f"Geometric features: {n_geom}")
    assert n_atmo == 9
    assert n_geom == 9
    
    print(" Data verified successfully!")
```

### Step 3: Compute Data Statistics (for verification)

```bash
python sow_outputs/sprint6/analysis/compute_data_stats.py
```

**Expected statistics:**
- CBH mean: ~1000-1200 meters
- CBH std: ~200-300 meters
- No missing values in labels
- All features have 933 samples

---

## Model Training

### Reproduce Cross-Validation Results

**Exact command to reproduce CV results:**

```bash
cd sow_outputs/sprint6
python validation/cross_validate_tabular.py \
  --data-path ../../sow_outputs/integrated_features/Integrated_Features.hdf5 \
  --n-folds 5 \
  --random-seed 42 \
  --output-path reports/validation_report_tabular.json
```

**Expected results:**
```
Mean R²: 0.744 ± 0.037
Mean MAE: 117.4 ± 7.4 meters
Mean RMSE: 187.3 ± 15.3 meters
```

**Key configuration:**
```python
{
    "model_type": "GradientBoostingRegressor",
    "n_estimators": 200,
    "max_depth": 5,
    "learning_rate": 0.1,
    "loss": "squared_error",
    "random_state": 42,
    "validation_strategy": "stratified_5fold"
}
```

### Reproduce Production Model Training

```bash
python training/train_production_model.py \
  --data-path ../../sow_outputs/integrated_features/Integrated_Features.hdf5 \
  --output-dir checkpoints \
  --random-seed 42
```

**Outputs:**
- `checkpoints/production_model.joblib` (model)
- `checkpoints/production_scaler.joblib` (scaler)
- `checkpoints/production_config.json` (config)

**Verification:**

```python
import joblib
import numpy as np

# Load model
model = joblib.load('checkpoints/production_model.joblib')
scaler = joblib.load('checkpoints/production_scaler.joblib')

# Check attributes
assert model.n_estimators == 200
assert model.max_depth == 5
assert model.learning_rate == 0.1

# Test inference
X_test = np.random.randn(10, 18)
X_scaled = scaler.transform(X_test)
predictions = model.predict(X_scaled)

assert predictions.shape == (10,)
print(" Production model verified!")
```

---

## Validation & Testing

### 1. Uncertainty Quantification

```bash
python validation/uncertainty_quantification.py \
  --data-path ../../sow_outputs/integrated_features/Integrated_Features.hdf5 \
  --n-folds 5 \
  --confidence-level 0.90 \
  --random-seed 42 \
  --output-path reports/uncertainty_quantification_report.json
```

**Expected results:**
```
Mean coverage: 77.1% (target: 90%)
Mean interval width: 533.4 ± 20.8 meters
Uncertainty-error correlation: 0.485
Calibration status: Poorly calibrated
```

### 2. Ensemble Experiments

```bash
python ensemble/ensemble_tabular_image.py \
  --data-path ../../sow_outputs/integrated_features/Integrated_Features.hdf5 \
  --image-path ../../data_ssl/images/train.h5 \
  --n-folds 5 \
  --random-seed 42 \
  --output-path reports/ensemble_results.json
```

**Expected results:**
```
GBDT (Tabular): R² = 0.727 ± 0.112
CNN (Image): R² = 0.351 ± 0.075
Weighted Avg: R² = 0.739 ± 0.096 (best)
Optimal weights: [0.88, 0.12] (GBDT, CNN)
```

### 3. Domain Adaptation (Flight F4)

```bash
python domain_adaptation/few_shot_f4_tabular.py \
  --data-path ../../sow_outputs/integrated_features/Integrated_Features.hdf5 \
  --test-flight F4 \
  --n-shots 10 \
  --n-trials 5 \
  --random-seed 42 \
  --output-path reports/domain_adaptation_f4.json
```

**Expected results:**
```
Baseline (LOO): R² = -0.98
10-shot: R² = -0.22 ± 0.18 (best)
```

### 4. Run All Tests

```bash
cd sow_outputs/sprint6
pytest tests/ -v --tb=short
```

**Expected**: All tests pass (or specific known failures documented)

---

## Reproducing Results

### Complete Reproduction Pipeline

Run this script to reproduce all major results:

```bash
#!/bin/bash
# reproduce_all.sh

set -e  # Exit on error

echo "=== Sprint 6 Reproducibility Pipeline ==="
echo ""

# 1. Cross-validation
echo "[1/6] Running cross-validation..."
python validation/cross_validate_tabular.py \
  --data-path ../../sow_outputs/integrated_features/Integrated_Features.hdf5 \
  --n-folds 5 \
  --random-seed 42 \
  --output-path reports/validation_report_tabular_repro.json

# 2. Production model training
echo "[2/6] Training production model..."
python training/train_production_model.py \
  --data-path ../../sow_outputs/integrated_features/Integrated_Features.hdf5 \
  --output-dir checkpoints_repro \
  --random-seed 42

# 3. Uncertainty quantification
echo "[3/6] Running uncertainty quantification..."
python validation/uncertainty_quantification.py \
  --data-path ../../sow_outputs/integrated_features/Integrated_Features.hdf5 \
  --n-folds 5 \
  --confidence-level 0.90 \
  --random-seed 42 \
  --output-path reports/uq_report_repro.json

# 4. Ensemble experiments
echo "[4/6] Running ensemble experiments..."
python ensemble/ensemble_tabular_image.py \
  --data-path ../../sow_outputs/integrated_features/Integrated_Features.hdf5 \
  --image-path ../../data_ssl/images/train.h5 \
  --n-folds 5 \
  --random-seed 42 \
  --output-path reports/ensemble_results_repro.json

# 5. Domain adaptation
echo "[5/6] Running domain adaptation..."
python domain_adaptation/few_shot_f4_tabular.py \
  --data-path ../../sow_outputs/integrated_features/Integrated_Features.hdf5 \
  --test-flight F4 \
  --n-shots 10 \
  --n-trials 5 \
  --random-seed 42 \
  --output-path reports/domain_adaptation_repro.json

# 6. Generate visualizations
echo "[6/6] Generating visualizations..."
python visualization/generate_all_figures.py \
  --reports-dir reports \
  --output-dir figures/reproduced

echo ""
echo "=== Reproduction Complete ==="
echo "Results saved to reports/*_repro.json"
echo "Figures saved to figures/reproduced/"
```

**Run:**

```bash
chmod +x reproduce_all.sh
./reproduce_all.sh
```

### Verify Reproduced Results

```python
import json
import numpy as np

# Load original and reproduced results
with open('reports/validation_report_tabular.json') as f:
    original = json.load(f)

with open('reports/validation_report_tabular_repro.json') as f:
    reproduced = json.load(f)

# Compare metrics
orig_r2 = original['mean_metrics']['r2']
repro_r2 = reproduced['mean_metrics']['r2']

print(f"Original R²: {orig_r2:.6f}")
print(f"Reproduced R²: {repro_r2:.6f}")
print(f"Difference: {abs(orig_r2 - repro_r2):.6f}")

# Should be identical (or within floating-point precision)
assert np.isclose(orig_r2, repro_r2, atol=1e-6), "R² mismatch!"

print(" Results successfully reproduced!")
```

---

## Common Issues

### Issue 1: Different Results Despite Same Random Seed

**Cause**: Operating system or hardware differences can affect random number generation

**Solution**:
```python
# Add this at the start of all scripts
import numpy as np
import random
import os

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
set_seed(42)
```

### Issue 2: Version Mismatch Errors

**Cause**: Different package versions

**Solution**:
```bash
# Uninstall all packages
pip freeze | xargs pip uninstall -y

# Reinstall from pinned requirements
pip install -r sow_outputs/sprint6/requirements_production.txt
```

### Issue 3: HDF5 File Format Errors

**Cause**: Different h5py versions

**Solution**:
```bash
# Ensure h5py version matches
pip install h5py==3.14.0

# Verify data can be read
python -c "import h5py; f = h5py.File('sow_outputs/integrated_features/Integrated_Features.hdf5', 'r'); print(list(f.keys()))"
```

### Issue 4: Out of Memory

**Cause**: Insufficient RAM for large batch processing

**Solution**:
```python
# Reduce batch size in config
config['batch_size'] = 100  # Instead of 1000

# Or process in chunks
for i in range(0, len(X), batch_size):
    batch = X[i:i+batch_size]
    # Process batch
```

### Issue 5: Missing Data Files

**Cause**: Data not downloaded or in wrong location

**Solution**:
```bash
# Check data location
ls -lh sow_outputs/integrated_features/Integrated_Features.hdf5

# If missing, check original location or re-generate
# (Contact data team for access)
```

---

## Verification Checklist

Use this checklist to verify complete reproducibility:

### Environment Verification

- [ ] Python version: 3.12
- [ ] scikit-learn version: 1.7.0
- [ ] numpy version: 2.2.6
- [ ] pandas version: 2.3.0
- [ ] All tests pass: `pytest tests/ -v`

### Data Verification

- [ ] Data file exists: `Integrated_Features.hdf5`
- [ ] Sample count: 933
- [ ] Feature count: 18
- [ ] No missing labels
- [ ] MD5 checksum matches (if available)

### Results Verification

- [ ] Cross-validation R²: 0.744 ± 0.037 (tolerance: ±0.001)
- [ ] Cross-validation MAE: 117.4 ± 7.4 m (tolerance: ±1.0 m)
- [ ] Production model trains successfully
- [ ] Model artifacts saved correctly
- [ ] UQ coverage: ~77% (tolerance: ±2%)
- [ ] Ensemble weighted R²: ~0.739 (tolerance: ±0.005)
- [ ] F4 few-shot (10): R² ≈ -0.22 (tolerance: ±0.05)

### Output Verification

- [ ] All JSON reports generated
- [ ] All figures generated (PNG + PDF)
- [ ] Model card exists
- [ ] Deployment guide exists
- [ ] No errors in logs

### Reproducibility Score

Count checked items:
- **20/20**: Perfect reproducibility 
- **18-19/20**: Excellent (minor variations acceptable)
- **15-17/20**: Good (investigate missing items)
- **<15/20**: Issues present (contact support)

---

## Docker-based Reproducibility (Guaranteed)

For absolute reproducibility, use Docker:

```dockerfile
# Dockerfile.reproduce
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc g++ git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY sow_outputs/sprint6/requirements_production.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements_production.txt

# Copy code and data
COPY sow_outputs/ /app/sow_outputs/
COPY data_ssl/ /app/data_ssl/

# Set random seed environment variable
ENV PYTHONHASHSEED=42

# Run reproduction pipeline
CMD ["bash", "/app/sow_outputs/sprint6/reproduce_all.sh"]
```

**Build and run:**

```bash
docker build -t cbh-reproduce:1.0.0 -f Dockerfile.reproduce .
docker run -v $(pwd)/results:/app/results cbh-reproduce:1.0.0
```

**Results will be in `./results/` directory**

---

## Contact & Support

For reproducibility issues:

- **Technical Issues**: ml-reproducibility@nasa.gov
- **Data Access**: data-team@nasa.gov
- **GitHub Issues**: https://github.com/nasa/cloudMLPublic/issues
- **Documentation**: See `DEPLOYMENT_GUIDE.md` and `MODEL_CARD.md`

---

## Citation

If you reproduce these results, please cite:

```bibtex
@techreport{nasa_cbh_sprint6_2025,
  title={Cloud Base Height Retrieval System - Sprint 6},
  author={NASA Cloud ML Team},
  institution={NASA},
  year={2025},
  type={Technical Report},
  note={Version 1.0.0}
}
```

---

**Document Version**: 1.0.0  
**Last Updated**: 2025-11-11  
**Maintainer**: NASA Cloud ML Team