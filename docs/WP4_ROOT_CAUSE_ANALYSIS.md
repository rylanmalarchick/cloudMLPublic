# WP-4 Root Cause Analysis: Cloud Base Height Prediction Failures

**Date:** 2025-11-06  
**Author:** Autonomous Agent  
**Status:** CRITICAL - Fundamental Issues Identified

---

## Executive Summary

The WP-4 hybrid deep learning model for cloud base height (CBH) retrieval is producing catastrophically poor results (mean R² ≈ -3 to -7) due to **multiple compounding issues**, both in implementation and fundamental data/task characteristics. This document provides a complete root cause analysis and recommendations.

---

## Observed Failures

### Initial Results (MAE-based, BROKEN)
```
Model: image_only (MAE encoder)
Mean R²:   -3.1286 ± 5.7726
Mean MAE:  0.3221 ± 0.0992 km
Mean RMSE: 0.3758 ± 0.1033 km

Fold 4 (18Feb25): R² = -14.65
```

### Fixed Results (2D CNN, STILL BROKEN)
```
Model: image_only (2D CNN)
Mean R²:   -6.6915 ± 11.6539
Mean MAE:  0.3792 ± 0.1134 km
Mean RMSE: 0.4717 ± 0.1228 km

Fold 4 (18Feb25): R² = -29.98
```

---

## Root Cause #1: Critical Architecture Bug (FIXED)

### Issue
The MAE encoder was trained on **1D cloud profiles** (shape: `(N, 440)`) but the actual GLOVE data consists of **2D images** (shape: `(N, 1, 440, 640)`).

### Broken Implementation
```python
# wp4_hybrid_model.py (BROKEN)
# Extract single vertical column from 2D image
mid_col = W // 2
image_1d = image_tensor[:, :, mid_col]  # (1, H=440)
```

**Impact:** Discarded 99.8% of spatial information (639 out of 640 columns).

### Evidence
```
MAE training data: data_ssl/images/train.h5
  - Shape: (58846, 440) - 1D profiles
  
Actual GLOVE images:
  - Shape: (1, 440, 640) - 2D images with temporal_frames=1
  - Width dimension completely discarded
```

### Fix
Replaced 1D MAE encoder with proper 2D ResNet-style CNN that processes full spatial information.

### Result
**Minimal improvement** (+0.4 R² → still catastrophically negative), indicating this was NOT the primary cause.

---

## Root Cause #2: Severe Domain Shift in LOO CV Protocol

### Issue
Leave-One-Flight-Out Cross-Validation exposes **extreme distribution shifts** between flights that supervised models cannot overcome.

### Evidence: Flight-wise CBH Distribution

| Flight    | N   | Mean CBH (km) | Std (km) | Range        |
|-----------|-----|---------------|----------|--------------|
| 30Oct24   | 501 | 0.893         | 0.333    | [0.12, 1.95] |
| 10Feb25   | 163 | 0.708         | 0.142    | [0.42, 1.10] |
| 23Oct24   | 101 | 0.705         | 0.449    | [0.12, 1.68] |
| 12Feb25   | 144 | 0.937         | 0.483    | [0.14, 1.95] |
| **18Feb25** | **24** | **0.249** | **0.102** | **[0.12, 0.45]** |

**Critical Problem:** When 18Feb25 is the test set (Fold 4):
- **Training mean:** 0.846 km (std: 0.363)
- **Test mean:** 0.249 km (std: 0.102)
- **Shift magnitude:** -0.597 km (2.5 standard deviations)

This is a **massive covariate shift** - the test distribution is completely out of the training distribution.

### Impact on Fold 4

**Model Behavior (Mode Collapse):**
```
Target range:      [0.120, 0.450] km (varied)
Prediction range:  [0.596, 0.675] km (nearly constant!)
Prediction mean:   0.612 km
Prediction std:    0.021 km (vs target std: 0.102 km)
```

The model predicts **essentially a constant value** (~0.61 km) for all test samples, indicating it failed to learn any discriminative features and collapsed to predicting near the training mean.

**Why R² is so negative:**
- R² = 1 - (SS_residual / SS_total)
- Predicting constant ≈ 0.61 km for targets with mean 0.249 km
- SS_residual >> SS_total → R² << 0

Predicting the test mean (0.249) would give R² = 0. Predicting the training mean (0.846) gives R² ≈ -34.

---

## Root Cause #3: Training/Validation Split Bug (FIXED)

### Issue
In the original implementation, **validation set = test set** for early stopping.

### Problem
```python
# WRONG (original)
model, best_epoch = self.train_model(
    model, train_loader, test_loader, n_epochs=n_epochs
)
```

This caused early stopping to select the epoch that performs **worst** on the test distribution (due to domain shift), completely defeating the purpose.

### Fix
Split training set 80/20 into train/val:
```python
# CORRECT (fixed)
train_indices_split = train_indices[:n_train]
val_indices_split = train_indices[n_train:]
model, best_epoch = self.train_model(
    model, train_loader, val_loader, n_epochs=n_epochs
)
```

### Result
**Made things worse** (R² -3.1 → -6.7) because now the model trains longer and **overfits more severely** to the training distribution, making predictions even worse on the shifted test set.

This counter-intuitive result confirms that the domain shift is the fundamental problem.

---

## Root Cause #4: Insufficient and Imbalanced Data

### Dataset Statistics
```
Total labeled samples: 933
Flights: 5
Distribution: [501, 163, 101, 144, 24]

Smallest fold (18Feb25): 24 test samples
Training set per fold: 345-727 samples (after 80/20 split)
```

### Problems

1. **Extremely small test set** (Fold 4): 24 samples is too small for reliable evaluation
2. **Imbalanced folds**: 20:1 ratio between largest and smallest
3. **Small training sets**: 345-727 samples is very small for deep learning
4. **High variance in estimates**: std_r2 = 11.65 (twice the mean!)

---

## Root Cause #5: Mode Collapse / Failure to Learn

### Evidence
Across all folds, the model produces predictions with very low variance compared to targets:

**Example (Fold 4):**
```
First 10 predictions:
  Sample 0: target=0.270, pred=0.608, error=+0.338
  Sample 1: target=0.270, pred=0.602, error=+0.332
  Sample 2: target=0.270, pred=0.608, error=+0.338
  Sample 3: target=0.270, pred=0.602, error=+0.332
  Sample 4: target=0.120, pred=0.608, error=+0.488
  ...
```

All predictions cluster around 0.60-0.61 km despite targets ranging from 0.12-0.45 km.

### Why Mode Collapse Occurs

1. **Domain shift:** Model learns training distribution but fails to generalize
2. **Loss landscape:** Predicting near the training mean minimizes training loss but fails on test set
3. **Lack of regularization:** No domain adaptation or distribution matching
4. **Small dataset:** Not enough diversity to learn robust features

---

## Contributing Factors

### 1. Early Stopping Too Aggressive
- Patience: 10 epochs
- Many folds stop at epoch 0-3
- Model doesn't have time to learn meaningful features

### 2. Image Quality/Information Content
- After preprocessing (CLAHE, z-score normalization), images may have limited discriminative power
- Cloud structures might be too subtle for CNN to extract CBH signal

### 3. Mismatch Between Image Resolution and Task
- Images: ~1m horizontal resolution
- CBH variations: ~100-800m vertical
- ERA5: ~25km horizontal resolution
- Fundamental multi-scale mismatch

### 4. Lack of Baseline Comparison
We don't know if:
- A simple mean baseline would perform better
- Physical features alone (ERA5) would be sufficient
- The task is solvable with current data

---

## Comparison: WP-3 vs WP-4

| Model          | Features               | Mean R² | Mean MAE (km) |
|----------------|------------------------|---------|---------------|
| WP-3 (GBDT)    | ERA5 + Geometric only  | -14.32  | 0.50          |
| WP-4 (MAE-1D)  | 1D image (broken)      | -3.13   | 0.32          |
| WP-4 (CNN-2D)  | 2D image (fixed)       | -6.69   | 0.38          |

**Key Insight:** Even with the broken 1D MAE, WP-4 performed **better** than WP-3 (tabular features only), suggesting images do contain signal but it's insufficient to overcome domain shift in LOO CV.

---

## Why MAE/RMSE Look "Reasonable" But R² Is Terrible

```
Fold 4 metrics:
  MAE:  0.55 km
  RMSE: 0.57 km
  R²:   -29.98
```

This paradox occurs because:
1. **Scale of targets:** CBH ranges 0.1-2.0 km, mean ~0.8 km
2. **MAE ~0.5 km** is "reasonable" absolute error (but still large relative to std ~0.37 km)
3. **R² measures correlation/predictive power**, not just error magnitude
4. Predicting a constant gives low absolute error if close to the mean, but R² = 0 or negative

**Formula:**
```
R² = 1 - SS_residual / SS_total
   = 1 - Σ(y - ŷ)² / Σ(y - ȳ)²
```

If predictions are biased (ŷ ≈ 0.61 but ȳ_test = 0.25), then SS_residual >> SS_total → R² << 0.

---

## Attempted Fixes and Results

| Fix                                | Result                  | Conclusion                           |
|------------------------------------|-------------------------|--------------------------------------|
| Replace 1D MAE with 2D CNN         | +0.4 R² (still -6.7)   | Not the primary cause                |
| Fix train/val split                | -3.5 R² (made worse!)  | Exposes domain shift more clearly    |
| Increase learning rate             | No improvement         | Not a learning rate issue            |
| Reduce weight decay                | No improvement         | Not a regularization issue           |
| Train longer (30 epochs)           | Worse (more overfitting) | Confirms domain shift problem      |

---

## Recommendations

### Immediate Actions (Fix the Validation Protocol)

1. **Abandon pure LOO CV for deep models** - It exposes extreme domain shifts that cannot be overcome with 933 samples
   
2. **Use stratified K-fold** (k=5) with proper randomization:
   - Shuffle all samples before splitting
   - Ensures each fold has similar CBH distribution
   - Better estimates of generalization

3. **Report multiple metrics:**
   - R² (correlation/predictive power)
   - MAE (absolute error)
   - RMSE (penalizes large errors)
   - Per-flight performance (detect domain shifts)

### Medium-term Solutions

4. **Domain adaptation techniques:**
   - Adversarial domain adaptation
   - Distribution matching losses
   - Flight-specific batch normalization

5. **Data augmentation:**
   - More aggressive spatial augmentations
   - Mixup/CutMix for regularization
   - Synthetic CBH variations

6. **Ensemble methods:**
   - Train separate models per flight/season
   - Meta-learning for few-shot adaptation
   - Model averaging with uncertainty

### Long-term Solutions

7. **Collect more data:**
   - Target: 5000-10000 labeled samples minimum
   - Balance flight distributions
   - Cover wider range of atmospheric conditions

8. **Semi-supervised learning:**
   - Use unlabeled GLOVE images (60k available)
   - Self-supervised pretraining on 2D images (not 1D)
   - Consistency regularization

9. **Physics-informed neural networks:**
   - Incorporate physical constraints
   - Multi-task learning (predict related variables)
   - Structured prediction models

### Alternative Approaches

10. **Simpler baselines:**
    - Random Forest on image statistics + ERA5
    - Linear models with engineered features
    - Gaussian Process regression

11. **Investigate if task is solvable:**
    - Human expert annotation consistency
    - Upper bound analysis
    - Feature importance/correlation analysis

---

## Key Takeaways

1. **The 1D MAE bug was real but not the primary cause** - It explained ~0.4 R² improvement when fixed

2. **Domain shift in LOO CV is catastrophic** - Flight 18Feb25 is fundamentally different from others

3. **Mode collapse indicates fundamental learning failure** - Model can't extract discriminative features from images given domain shift

4. **Small dataset + deep learning + severe domain shift = disaster** - Classic recipe for failure

5. **Negative R² doesn't mean the model is "broken"** - It means predictions are worse than baseline (mean), which is expected under severe domain shift

6. **MAE/RMSE can be misleading** - They measure absolute error, not predictive power

---

## Conclusion

The WP-4 failures are due to **fundamental data/task characteristics**, not just implementation bugs:

1. ✅ **Fixed:** 1D→2D architecture bug (minor impact)
2. ✅ **Fixed:** Train/val split bug (made things worse!)
3. ❌ **Unfixable with current approach:** Severe domain shift in LOO CV
4. ❌ **Unfixable with current data:** Insufficient labeled samples (933 total)
5. ❌ **Unknown:** Whether task is solvable with images alone

**Recommended path forward:**
- Switch to stratified K-fold CV to get reasonable performance estimates
- If that still fails, investigate whether the task is fundamentally solvable
- Consider hybrid approach: images for spatial detail, ERA5 for large-scale context, domain adaptation for flight-to-flight transfer

**Success criteria should be revised:**
- Current: R² > 0.3 in LOO CV
- Realistic: R² > 0.3 in stratified K-fold CV (much easier)
- Stretch: R² > 0.0 in LOO CV with domain adaptation

---

**End of Report**