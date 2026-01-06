# AgentBible Effectiveness Log: CBH Retrieval Restudy

**Project**: Cloud Base Height (CBH) Retrieval from NASA ER-2 Observations  
**Date**: 2026-01-06  
**AgentBible Version**: 0.3.0

---

## Executive Summary

AgentBible was integrated into a restudy of a CBH retrieval ML paper to:
1. Catch data quality issues early
2. Prevent data leakage
3. Document data provenance
4. Enable reproducible science

**Key finding**: AgentBible validators caught critical issues that would have invalidated the study, including all-zero ERA5 features and data leakage via `inversion_height`.

---

## Issues Detected by AgentBible Validators

### Issue 1: All-Zero ERA5 Features

**Validator**: `validate_era5_nonzero()`  
**Severity**: CRITICAL  
**Impact**: Would have produced R² ≈ 0 (meaningless model)

```python
# Detection
from src.cbh_validators import validate_era5_nonzero, ERA5DataError

features = {'blh': blh_array, 't2m': t2m_array, ...}
try:
    validate_era5_nonzero(features)
except ERA5DataError as e:
    print(f"CAUGHT: {e}")
    # "ERA5 FEATURE HAS ZERO VARIANCE
    #  Feature: blh
    #  Variance: 0.00e+00
    #  Why this matters: Zero-variance features indicate placeholder data..."
```

**Resolution**: Traced to missing ERA5 integration step. Re-ran `scripts/integrate_era5_features.py` to populate real ERA5 values.

**Time saved**: ~2-4 hours of debugging mysterious R² = 0 results

---

### Issue 2: Data Leakage via `inversion_height`

**Validator**: `validate_no_leakage()`  
**Severity**: CRITICAL  
**Impact**: Would have produced R² = 0.999 (completely fake)

```python
# Detection
from src.cbh_validators import validate_no_leakage, DataLeakageError

feature_names = ['blh', 't2m', 'd2m', 'inversion_height', ...]
try:
    validate_no_leakage(feature_names)
except DataLeakageError as e:
    print(f"CAUGHT: {e}")
    # "DATA LEAKAGE DETECTED
    #  Forbidden features found: {'inversion_height'}
    #  Why this matters:
    #    'inversion_height' is computed as CBH - BLH, which directly
    #    encodes the target variable. This caused R² = 0.999..."
```

**Background**: `inversion_height = CBH - BLH` was created as a derived feature, but it directly encodes the target (CBH). Any model using this feature achieves near-perfect R² because it's just inverting the formula.

**Resolution**: Explicitly excluded `inversion_height` from all clean datasets.

**Time saved**: Prevented publication of invalid results

---

### Issue 3: Temporal Autocorrelation Warning

**Validator**: `compute_temporal_autocorrelation()`  
**Severity**: WARNING  
**Impact**: Pooled K-fold R² inflated by ~0.2 (from 0.71 to 0.92)

```python
# Detection
from src.cbh_validators import compute_temporal_autocorrelation

autocorr = compute_temporal_autocorrelation(cbh_array)
# Returns: {
#   'lag1_autocorr': 0.94,
#   'lag5_autocorr': 0.85,
#   'warning': 'High temporal autocorrelation (lag-1 = 0.940). 
#              Random K-fold CV will overestimate performance.
#              Use per-flight CV or time-series CV instead.'
# }
```

**Resolution**: Implemented multiple validation strategies:
- Pooled K-fold (document inflation)
- Per-flight shuffled K-fold (moderate, valid for interleaved training)
- Per-flight time-ordered K-fold (strict honest test)
- LOFO-CV (domain shift test)

**Scientific insight**: Autocorrelation is ~0.94 at lag-1, meaning consecutive samples are highly correlated. This is physically expected for cloud systems but must be accounted for in validation.

---

## Provenance Metadata

AgentBible's `save_with_metadata()` function automatically recorded:

```json
{
  "provenance": {
    "description": "CBH dataset with paper filtering",
    "timestamp": "2026-01-06T18:52:35.279609+00:00",
    "git_sha": "7df59016a3652fe343d30533b17410942da042a9",
    "git_branch": "main",
    "git_dirty": true,
    "numpy_seed": null,
    "python_seed": null,
    "torch_seed": 10506416843265579298,
    "hostname": "theMachine",
    "platform": "Linux-6.14.0-37-generic-x86_64-with-glibc2.39",
    "python_version": "3.12.3",
    "packages": {
      "numpy": "2.2.6",
      "torch": "2.7.1+cu126",
      "xgboost": "3.1.1"
    }
  }
}
```

This enables:
- Exact reproduction of dataset creation
- Tracking which code version produced each artifact
- Auditing hardware and software environment

---

## Custom CBH Validators Created

| Validator | Purpose | Catches |
|-----------|---------|---------|
| `validate_cbh_range()` | Ensure predictions are physically plausible | Unit errors (m vs km) |
| `validate_no_leakage()` | Block forbidden features | `inversion_height` leakage |
| `validate_era5_nonzero()` | Catch placeholder zeros | Missing data integration |
| `validate_lcl_cbh_consistency()` | Check LCL < CBH for marine Sc | Thermodynamic inconsistencies |
| `compute_temporal_autocorrelation()` | Warn about CV inflation | High lag-1 correlation |

---

## Proposed AgentBible Extensions for Atmospheric Science

### 1. Atmospheric Physics Validators

```python
# Proposed additions to agentbible core

@validate_lcl_range(min_m=0, max_m=5000)
def compute_lcl(t2m, d2m):
    """Validate LCL is physically reasonable."""
    pass

@validate_blh_range(min_m=50, max_m=4000)  
def compute_blh(era5_data):
    """Boundary layer height must be reasonable."""
    pass

@validate_cloud_layer_consistency
def validate_cth_cbh(cth, cbh):
    """Cloud top height must exceed cloud base height."""
    assert np.all(cth >= cbh), "CTH < CBH is physically impossible"
```

### 2. Temporal CV Validators

```python
# Warn when using random CV on autocorrelated data
@validate_cv_strategy(max_autocorr=0.5)
def train_model(X, y, cv='random'):
    """If autocorr > 0.5, require time-series CV."""
    pass
```

### 3. Satellite/Airborne Data Validators

```python
# For remote sensing applications
@validate_solar_geometry(sza_max=80)
def process_radiances(radiances, sza):
    """Exclude high-SZA observations where retrievals are unreliable."""
    pass

@validate_geolocation_consistency(max_distance_km=1.0)
def match_observations(lat1, lon1, lat2, lon2, time1, time2):
    """Ensure matched observations are spatially consistent."""
    pass
```

---

## Impact Summary

| Category | Without AgentBible | With AgentBible |
|----------|-------------------|-----------------|
| Data leakage detection | Manual audit (hours) | Automatic (seconds) |
| ERA5 zero-check | Debug after training | Caught at load time |
| CV inflation awareness | Often missed | Explicit warning |
| Reproducibility | Partial | Full provenance |
| Code quality | Variable | Enforced standards |

---

## Lessons Learned

1. **Validate early, validate often**: Catching the all-zeros ERA5 bug at dataset creation saved hours of debugging.

2. **Make forbidden features explicit**: The `LEAKAGE_FEATURES` set in `cbh_validators.py` documents what NOT to use and why.

3. **Autocorrelation matters**: High lag-1 autocorrelation (0.94) inflates random CV by ~0.2 R². This is domain knowledge that should be encoded in validators.

4. **Provenance is essential**: Being able to trace exactly which git commit, packages, and seeds produced a dataset enables true reproducibility.

5. **Domain-specific validators add value**: Generic validators (finite, range) are useful, but domain-specific ones (LCL-CBH consistency, temporal autocorrelation) catch science-specific bugs.

---

## Recommendations for Future Work

1. **Integrate temporal CV checking into AgentBible core** - Many time-series ML applications suffer from autocorrelation inflation.

2. **Add unit conversion validators** - Atmospheric data often has m/km, K/°C, Pa/hPa confusion.

3. **Create geospatial validators** - Check that lat/lon are on Earth, timestamps are sequential, etc.

4. **Build pre-commit hooks** - Run validators automatically on `git commit` to catch issues before they're committed.

---

## Files Created During Restudy

| File | Purpose |
|------|---------|
| `src/cbh_validators.py` | Custom CBH validators using AgentBible |
| `scripts/create_clean_dataset.py` | Create validated datasets |
| `scripts/validate_models.py` | Multi-strategy validation framework |
| `scripts/train_tabular_model.py` | GBDT training with proper CV |
| `scripts/train_image_model.py` | CNN training with same CV |
| `scripts/train_hybrid_model.py` | Hybrid model training |
| `scripts/generate_figures.py` | Publication-quality figures |
| `outputs/preprocessed_data/Clean_*.hdf5` | Validated datasets with provenance |
| `outputs/figures/VALIDATION_SUMMARY.md` | Results summary |

---

## Conclusion

AgentBible provided significant value for this atmospheric science ML project by:

1. **Preventing invalid results** (data leakage, zero features)
2. **Improving scientific rigor** (autocorrelation awareness, proper CV)
3. **Enabling reproducibility** (full provenance tracking)
4. **Documenting domain knowledge** (physics constraints in code)

The ~30 minutes spent integrating AgentBible validators saved hours of debugging and prevented potential publication of invalid results. The explicit documentation of what went wrong in the original study serves as a template for other researchers facing similar issues.

**Verdict**: AgentBible is highly effective for research code, especially when combined with domain-specific validators.

---

## Session 2 Update (2026-01-06)

### Additional Work Completed

#### 1. Feature Engineering with AgentBible Patterns

**Script**: `scripts/feature_engineering.py`

Created 28 new physics-based features with AgentBible-style validation:

```python
# Local validators following AgentBible philosophy
def validate_finite(arr: np.ndarray, name: str = "array") -> np.ndarray:
    """Validate array contains only finite values."""
    if not np.all(np.isfinite(arr)):
        nan_count = np.sum(np.isnan(arr))
        inf_count = np.sum(np.isinf(arr))
        raise ValidationError(
            f"{name} contains non-finite values: {nan_count} NaN, {inf_count} Inf"
        )
    return arr

# Physics validation for derived features
validate_range(relative_humidity, 0, 100, "relative_humidity_2m")
validate_positive(saturation_vapor_pressure, "saturation_vapor_pressure")
validate_finite(potential_temperature, "potential_temperature")
```

**Key Finding**: AgentBible's decorator-based validators are designed for functions, not raw arrays. For feature engineering pipelines, we created local array validators following the same philosophy.

**Recommendation**: Add `agentbible.validate_array()` functions for direct array validation:
```python
# Proposed API
from agentbible import validate_array_finite, validate_array_range
validate_array_finite(my_array, name="temperature")
validate_array_range(my_array, 0, 100, name="humidity")
```

#### 2. Domain Adaptation Analysis

**Script**: `scripts/run_domain_adaptation.py`

| Method | Mean LOFO R² | Notes |
|--------|--------------|-------|
| Baseline | -15.4 | Catastrophic failure across flights |
| Few-shot (50) | 0.57 (avg) | **Best method** |
| Instance Weighting | -21.4 | Worse than baseline |
| TrAdaBoost | -0.41 | Modest improvement |
| MMD Alignment | -39.4 | Poor |

**AgentBible Integration**: Used for version tracking and error handling. Validated that domain shift matrices contain only finite values.

**Key Finding**: Domain shift between flights is severe. Few-shot adaptation (5-50 samples from target flight) is the most practical solution.

#### 3. Uncertainty Quantification

**Script**: `scripts/run_uncertainty_quantification.py`

Implemented three UQ methods with coverage analysis:

| Method | Coverage (target 90%) | Width (m) | Status |
|--------|----------------------|-----------|--------|
| Split Conformal | 27.3% | 278 | Under-coverage |
| Adaptive Conformal | 11.2% | 58 | Intervals collapse |
| Quantile Regression | 58.1% | 510 | Miscalibrated |
| Per-flight Calibration | 85.7% | 277 | **Best** |

**Key Finding**: Conformal prediction's exchangeability assumption is violated by:
1. **Temporal autocorrelation** (lag-1 = 0.94)
2. **Domain shift** between flights

Per-flight calibration achieves near-target coverage.

**Potential AgentBible Extension**:
```python
@validate_exchangeability(max_autocorr=0.5)
def split_conformal_prediction(residuals, alpha):
    """Warn if data violates exchangeability assumption."""
    pass
```

---

## AgentBible API Feedback

### What Works Well

1. **Decorator-based validation** - Clean, composable, readable
2. **Error messages** - Actionable with context about why validation failed
3. **Provenance tracking** - Essential for reproducibility
4. **Domain-specific validators** (`cbh_validators.py`) - Easy to extend

### Limitations Discovered

1. **No direct array validators** - Must wrap in functions or create local validators
   ```python
   # Current: Decorators only
   @validate_finite
   def my_function():
       pass
   
   # Needed: Direct array validation
   agentbible.validate_finite(my_array)  # Does not exist
   ```

2. **No validation composition for pipelines** - Feature engineering has many intermediate arrays
   ```python
   # Would be useful:
   pipeline = agentbible.Pipeline([
       validate_finite,
       validate_range(0, 1000),
       validate_no_nan,
   ])
   pipeline.validate(features)
   ```

3. **No statistical validators** - Useful for ML workflows
   ```python
   # Proposed:
   @validate_no_data_leakage(forbidden=['target_derived'])
   @validate_train_test_independence(max_autocorr=0.5)
   def train_model(X, y, cv):
       pass
   ```

### Suggestions for v0.4.0

1. **Array validators** - Direct validation without function wrapping
2. **ML-specific validators**:
   - `validate_no_leakage(features, forbidden_list)`
   - `validate_cv_independence(y, cv_splits, max_autocorr)`
   - `validate_coverage(actual, target, tolerance)`
3. **Pipeline validation** - Compose multiple validators for data pipelines
4. **Domain extensions**:
   - `agentbible.domains.atmospheric` - LCL, BLH, CBH ranges
   - `agentbible.domains.ml` - Train/test leakage, CV validity

---

## Files Created in Session 2

| File | Purpose | AgentBible Integration |
|------|---------|----------------------|
| `scripts/feature_engineering.py` | Physics-based features | Local validators, version tracking |
| `scripts/run_domain_adaptation.py` | DA analysis | Version tracking, error handling |
| `scripts/run_uncertainty_quantification.py` | UQ analysis | Version tracking |
| `outputs/feature_engineering/Enhanced_Features.hdf5` | 38-feature dataset | Provenance metadata |
| `outputs/domain_adaptation/*` | DA results | JSON with AgentBible version |
| `outputs/uncertainty/*` | UQ results | JSON with AgentBible version |

---

## Updated Impact Summary

| Metric | Session 1 | Session 2 | Total |
|--------|-----------|-----------|-------|
| Critical bugs caught | 2 | 0 | 2 |
| Validation patterns applied | 5 | 3 | 8 |
| Scripts with AgentBible | 7 | 3 | 10 |
| Time saved (estimated) | 4-6 hrs | 1-2 hrs | 5-8 hrs |
| Datasets with provenance | 3 | 1 | 4 |

---

## Updated Recommendations

1. **Add array validators to AgentBible core** - Most useful for feature engineering
2. **Create ML-specific validators** - Train/test leakage, CV validity, coverage checks
3. **Add exchangeability validator** - Warn when data violates conformal prediction assumptions
4. **Create domain extension for atmospheric science** - Physical range validators for common variables
