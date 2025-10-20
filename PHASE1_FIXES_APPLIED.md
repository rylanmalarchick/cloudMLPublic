# Phase 1 Fixes Applied - Variance Collapse Resolution

**Date:** October 20, 2024  
**Status:** ✅ COMPLETE - All critical fixes implemented and ready for testing  
**Branch:** main

---

## Executive Summary

After analyzing three training runs that all exhibited variance collapse (predictions clustered around constant values), we identified the root cause and implemented comprehensive fixes.

**The Problem:**
- Run 1: R²=-0.045, variance ratio=40% (best, but still poor)
- Run 2: R²=-0.023, variance ratio=3.9% (near-total collapse)
- Run 3: R²=-0.203, variance ratio=3.4% (worst - catastrophic collapse)

**The Root Cause:**
Loss function didn't penalize variance collapse, allowing model to achieve low validation loss (~0.51-0.53) while predicting near-constant values.

**The Solution:**
Five critical fixes implemented across training pipeline, model initialization, and configuration.

---

## Fixes Implemented

### 1. Variance-Preserving Loss Function ✅

**File:** `src/train_model.py`

**Change:**
```python
# Added variance-preserving term to loss
pred_var = y_pred.var()
target_var = y_true.var()
variance_loss = (1.0 - pred_var / (target_var + 1e-8)) ** 2
loss = base_loss + variance_lambda * variance_loss
```

**Impact:**
- Explicitly penalizes models that collapse to constant predictions
- Forces model to maintain prediction diversity
- Configurable lambda weight (default: 0.5)

---

### 2. R²-Based Early Stopping ✅

**File:** `src/train_model.py`

**Change:**
```python
# Early stopping now tracks R² instead of validation loss
if val_r2 > best_r2 + config["early_stopping_min_delta"]:
    best_r2 = val_r2
    save_model()
```

**Impact:**
- Validation loss was misleading (0.51-0.53 for all runs, but R² varied 4.5x)
- Now optimizes the metric we actually care about (R²)
- Emergency stop if variance_ratio < 5% (catches collapse early)

---

### 3. Output Layer Initialization Reverted ✅

**File:** `src/pytorchmodel.py`

**Change:**
```python
# REVERTED from std=0.1 back to std=0.5 (like Run 1)
nn.init.normal_(self.output.weight, 0, 0.5)
nn.init.constant_(self.output.bias, 0.0)
```

**Impact:**
- Run 1 (std≈0.5): 40% variance ratio
- Run 2-3 (std=0.01-0.1): 3-4% variance ratio
- Larger initial weights = larger initial gradients = easier to maintain variance

---

### 4. Reduced Temporal Frames ✅

**File:** `configs/colab_optimized_full_tuned.yaml`

**Change:**
```yaml
# Reduced from 7 to 3 frames
temporal_frames: 3
```

**Impact:**
- 7 frames caused excessive smoothing/over-averaging
- Start with 3 frames (less aggressive)
- Can test 5 frames later in ablation study

---

### 5. Disabled Overweighting ✅

**File:** `configs/colab_optimized_full_tuned.yaml`

**Change:**
```yaml
# Disabled overweighting
overweight_factor: 1.0  # was 1.5
```

**Impact:**
- Overweighting may have encouraged conservative predictions (avoid risky high values)
- Start with balanced weighting
- Can re-enable in ablation study if needed

---

### 6. Enhanced Monitoring ✅

**File:** `src/train_model.py`

**Added metrics logged every epoch:**
- R² score
- Variance ratio (pred_std / target_std)
- Prediction std dev
- Prediction range
- Emergency stop if variance_ratio < 5%

**Impact:**
- Catch collapse early (don't waste hours on doomed training)
- Track metrics that actually matter
- Better understanding of model behavior during training

---

## Files Modified

### Core Training Pipeline
1. `src/train_model.py` - Variance loss, R² tracking, emergency stop
2. `src/pytorchmodel.py` - Output initialization reverted to std=0.5

### Configuration
3. `configs/colab_optimized_full_tuned.yaml` - temporal_frames=3, overweight=1.0, variance_lambda=0.5

### Documentation
4. `colab_training.ipynb` - Updated header with Phase 1 status
5. `NEXT_STEPS.md` - Comprehensive action plan
6. `trainingrun/third/THIRD_RUN_ANALYSIS.md` - Detailed analysis
7. `trainingrun/compare_runs.py` - Comparison script
8. `PHASE1_FIXES_APPLIED.md` - This document

---

## New Configuration Parameters

### In Config YAML:
```yaml
variance_lambda: 0.5          # Weight for variance-preserving loss term
temporal_frames: 3            # Reduced from 7
overweight_factor: 1.0        # Disabled (was 1.5)
early_stopping_min_delta: 0.0005  # For R² improvement
```

### In Training Loop:
- Tracks: val_r2, variance_ratio, pred_std, pred_range
- Emergency stop: variance_ratio < 0.05
- Early stopping criterion: R² (not val_loss)

---

## Expected Results

### Minimum Viable Success
- R² > 0.0 (better than mean baseline)
- Variance ratio > 40% (beat Run 1)
- Prediction spread > 0.5
- No near-constant predictions

### Good Performance
- R² > 0.3
- Variance ratio > 70%
- Prediction spread > 1.0
- RMSE < 0.35

### Excellent Performance
- R² > 0.5
- Variance ratio > 85%
- RMSE < 0.30
- Clear linear correlation in scatter plots

---

## How to Use

### 1. Pull Latest Code
```bash
cd /content/repo
!git pull origin main
```

### 2. Run Training (Option A-Tuned)
Use the updated config with all Phase 1 fixes:
```python
from src.pipeline import run_pipeline
config_path = "/content/repo/configs/colab_optimized_full_tuned.yaml"
run_pipeline(config_path)
```

### 3. Monitor Training
Watch for these metrics in output:
```
Epoch X | Train Loss: X.XX | Val Loss: X.XX | 
R²: 0.XXX | Var Ratio: XX.X% | Pred Std: X.XX
```

**Good signs:**
- R² increasing over epochs
- Variance ratio staying above 40%
- Pred std not collapsing to near-zero

**Bad signs (should NOT see):**
- R² stuck around -0.2 to -0.02
- Variance ratio dropping below 10%
- Emergency stop triggered (variance_ratio < 5%)

---

## What Was NOT Fixed (Phase 5 - Future)

### Self-Supervised Pretraining
**Status:** Still disabled

**Bugs identified:**
1. Uses only first frame (`images[:,0,:,:]`) instead of all frames
2. Early stopping in pretraining is monotonic, not patience-based
3. Pretraining validation loss diverged (increased instead of decreased)

**Fix required in:** `src/pretraining.py`

**Timeline:** Phase 5 (after Phase 3 validates that variance-preserving loss works)

---

## Testing Strategy

### Phase 3: Validation Experiment (NEXT)
Run training with all Phase 1 fixes and verify:
- [ ] Training completes without emergency stop
- [ ] R² > 0.0 by final epoch
- [ ] Variance ratio > 40%
- [ ] Prediction spread > 0.5
- [ ] No near-constant predictions

**If successful:** Proceed to Phase 4 (ablation study)  
**If failed:** Debug with diagnostics in NEXT_STEPS.md

### Phase 4: Ablation Study
Systematically test:
- temporal_frames ∈ {3, 5, 7}
- variance_lambda ∈ {0.25, 0.5, 1.0}
- overweight_factor ∈ {1.0, 1.5}

---

## Rollback Instructions

If Phase 1 fixes cause unexpected issues:

### Revert to Run 1 Config
```bash
git checkout <commit_before_phase1>
```

### Or manually revert specific changes:
```python
# In pytorchmodel.py
nn.init.normal_(self.output.weight, 0, 0.1)  # Was 0.5 in Phase 1

# In colab_optimized_full_tuned.yaml
temporal_frames: 7
overweight_factor: 1.5
# Remove: variance_lambda: 0.5

# In train_model.py
# Remove variance_loss term
# Revert early stopping to track val_loss
```

---

## Git Commit Message

```
feat: Phase 1 fixes for variance collapse (Runs 1-3)

PROBLEM:
- All three runs showed variance collapse (predictions near-constant)
- Run 1 (broken SSL): R²=-0.045, var_ratio=40% (BEST)
- Run 2 (std=0.01): R²=-0.023, var_ratio=3.9% (collapse)
- Run 3 (std=0.1): R²=-0.203, var_ratio=3.4% (catastrophic)
- Root cause: Loss function didn't penalize variance collapse

FIXES APPLIED:
1. Variance-preserving loss (lambda=0.5) - prevents collapse
2. R²-based early stopping - val_loss was misleading
3. Output init reverted to std=0.5 - smaller weights made it worse
4. temporal_frames: 7→3 - reduce over-averaging
5. overweight_factor: 1.5→1.0 - disable conservative weighting
6. Enhanced monitoring - track R², var_ratio, emergency stop

FILES MODIFIED:
- src/train_model.py - variance loss + R² tracking
- src/pytorchmodel.py - init std=0.5
- configs/colab_optimized_full_tuned.yaml - frames=3, overweight=1.0
- colab_training.ipynb - updated header
- Added: NEXT_STEPS.md, PHASE1_FIXES_APPLIED.md, analysis docs

EXPECTED RESULTS:
- R² > 0.0 (minimum)
- Variance ratio > 40%
- Prediction spread > 0.5

See NEXT_STEPS.md for full action plan.
```

---

## References

- **Third run analysis:** `trainingrun/third/THIRD_RUN_ANALYSIS.md`
- **Comparison script:** `trainingrun/compare_runs.py`
- **Action plan:** `NEXT_STEPS.md`
- **Previous analysis:** `trainingrun/first/BUG_INVESTIGATION_REPORT.md`

---

## Contact

**Status:** Ready for Phase 3 validation experiment  
**Next action:** Run training with Phase 1 config and verify results  
**Expected runtime:** 2-3 hours on Colab T4

---

*Last updated: 2024-10-20*  
*All fixes committed to main branch*