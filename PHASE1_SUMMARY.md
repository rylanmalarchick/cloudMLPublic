# Phase 1 Fixes - Quick Summary

**Date:** October 20, 2024  
**Status:** ✅ DEPLOYED - All fixes committed and pushed to main  
**Commit:** 734fe13

---

## The Problem

Three training runs all showed **severe variance collapse** (model predicting near-constant values):

| Run | Config | R² | Variance Ratio | Status |
|-----|--------|----|--------------|---------| 
| 1 | Broken SSL, std=0.5 | -0.045 | 40% | 🟡 Best (but still poor) |
| 2 | No SSL, std=0.01 | -0.023 | 3.9% | 🔴 Near-total collapse |
| 3 | No SSL, std=0.1 | -0.203 | 3.4% | 🔴 Catastrophic collapse |

**Root Cause:** Loss function didn't penalize variance collapse. Model learned to predict ~0.72 for everything → low MAE, terrible R².

---

## The Solution (5 Critical Fixes)

### 1. ✅ Variance-Preserving Loss
```python
variance_loss = (1.0 - pred_var / target_var) ** 2
loss = base_loss + 0.5 * variance_loss
```
**Why:** Explicitly penalizes constant predictions

### 2. ✅ R²-Based Early Stopping
**Before:** Stopped on validation loss (misleading - all runs had ~0.51-0.53)  
**After:** Stops on R² improvement (actual quality metric)

### 3. ✅ Output Init Reverted
**Before:** std=0.1 (Runs 2-3)  
**After:** std=0.5 (like Run 1 - the best performer)  
**Why:** Smaller weights → smaller gradients → harder to maintain variance

### 4. ✅ Reduced Temporal Frames
**Before:** 7 frames (over-averaging)  
**After:** 3 frames (less smoothing)

### 5. ✅ Disabled Overweighting
**Before:** overweight_factor=1.5  
**After:** overweight_factor=1.0  
**Why:** May have encouraged conservative predictions

---

## Files Changed

### Core Code
- `src/train_model.py` - Variance loss, R² tracking, emergency stop
- `src/pytorchmodel.py` - Init std=0.5 (both models)

### Config
- `configs/colab_optimized_full_tuned.yaml` - temporal_frames=3, overweight=1.0, variance_lambda=0.5

### Notebook
- `colab_training.ipynb` - Updated header with Phase 1 status

### Documentation
- `NEXT_STEPS.md` - Full action plan (388 lines)
- `PHASE1_FIXES_APPLIED.md` - Detailed fix documentation (347 lines)
- `trainingrun/third/THIRD_RUN_ANALYSIS.md` - Run 3 analysis (361 lines)
- `trainingrun/compare_runs.py` - Comparison tool (384 lines)

---

## Expected Results

| Metric | Minimum | Good | Excellent |
|--------|---------|------|-----------|
| **R²** | > 0.0 | > 0.3 | > 0.5 |
| **Variance Ratio** | > 40% | > 70% | > 85% |
| **Pred Spread** | > 0.5 | > 1.0 | > 1.4 |
| **RMSE** | - | < 0.35 | < 0.30 |

---

## What to Watch During Training

### ✅ Good Signs
- R² increasing over epochs (should reach positive values)
- Variance ratio staying above 40%
- Pred std not collapsing near zero
- Training completes without emergency stop

### 🔴 Bad Signs (Emergency - Stop and Debug)
- R² stuck between -0.2 and 0.0
- Variance ratio dropping below 10%
- "Severe variance collapse detected" message
- Emergency stop triggered

---

## How to Run (Updated)

```python
# In Colab, after pulling latest code:
%cd /content/repo
!git pull origin main

# Run with Phase 1 fixes
from src.pipeline import run_pipeline
config_path = "/content/repo/configs/colab_optimized_full_tuned.yaml"
run_pipeline(config_path)
```

**Monitor output for:**
```
Epoch X | R²: 0.XXX | Var Ratio: XX.X% | Pred Std: X.XX
```

---

## Next Steps

1. **Phase 3 (NOW):** Run validation experiment with Phase 1 config
   - Expected: 2-3 hours on Colab T4
   - Verify: R² > 0.0, var_ratio > 40%

2. **Phase 4:** Ablation study (temporal_frames, variance_lambda)

3. **Phase 5:** Fix SSL pretraining bugs, re-enable if beneficial

---

## Quick Reference - New Parameters

```yaml
# In configs/colab_optimized_full_tuned.yaml
temporal_frames: 3        # was 7
overweight_factor: 1.0    # was 1.5
variance_lambda: 0.5      # NEW - variance loss weight
```

**New metrics logged:**
- `r2` - R² score (primary early stopping metric)
- `variance_ratio` - pred_std / target_std
- `pred_std` - prediction standard deviation
- `pred_range` - max(pred) - min(pred)

---

## Key Insights from Analysis

1. **Val loss is NOT a good metric** for this problem
   - All runs: val_loss ≈ 0.51-0.53
   - But R² varied from -0.045 to -0.203 (4.5x difference!)

2. **Initialization matters HUGELY**
   - Run 1 (std=0.5): 40% variance ratio ✅
   - Run 2-3 (std=0.01-0.1): 3-4% variance ratio ❌
   - Counterintuitive: "fixing" init made it worse!

3. **The "broken" SSL was actually helpful**
   - Run 1 with broken SSL: BEST performer
   - Runs 2-3 without SSL: WORST performers
   - May have acted as beneficial noise/regularization

4. **Over-regularization kills variance**
   - 7 frames + overweighting + dropout → collapse
   - Need to preserve diversity, not just minimize loss

---

## Documentation

- **Full analysis:** `trainingrun/third/THIRD_RUN_ANALYSIS.md`
- **Action plan:** `NEXT_STEPS.md`
- **Implementation details:** `PHASE1_FIXES_APPLIED.md`
- **Comparison tool:** `trainingrun/compare_runs.py`

---

**Status:** Ready for Phase 3 validation  
**Last updated:** 2024-10-20  
**Commit:** 734fe13