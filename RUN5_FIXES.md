# Run 5 Fixes - Stronger Variance Loss

**Date:** October 20, 2024  
**Status:** Ready to deploy  
**Previous:** Run 4 with lambda=0.5 → variance collapsed 46%→8%

---

## The Problem (Run 4)

Run 4 applied Phase 1 fixes but **variance_lambda=0.5 was too weak**:

- Started training: 46% variance ratio ✅
- Best epoch (37): 8% variance ratio ❌ (collapsed!)
- Final result: 22% variance ratio (partial recovery)
- R² = -0.0655 (still negative)

**Key insight:** Variance loss gradient was ~2-3x weaker than base loss gradient, so model learned to sacrifice variance for lower loss.

---

## The Solution (Run 5)

### 1. Increase Variance Lambda
```yaml
variance_lambda: 2.0  # Was 0.5 (4x increase)
```

**Why:** Makes variance preservation as important as base loss minimization.

### 2. Minimum Variance Ratio Check
```yaml
min_variance_ratio: 0.35  # Don't save models with <35% variance
```

**Why:** Prevents early stopping from picking collapsed models (like Run 4's epoch 37 with 8% variance).

---

## Changes Applied

### Config File: `configs/colab_optimized_full_tuned.yaml`
```yaml
variance_lambda: 2.0          # 0.5 → 2.0
min_variance_ratio: 0.35      # NEW parameter
```

### Training Code: `src/train_model.py`
```python
# Early stopping now checks variance ratio
if val_r2 > best_r2 and variance_ratio >= min_variance_ratio:
    save_model()  # Only save if variance is acceptable
else:
    print("R² improved but variance too low, not saving")
```

---

## Expected Results

### During Training
- Variance ratio should stay **above 40%** throughout (no collapse)
- May see higher initial losses (variance penalty is stronger)
- Should converge to better R² as variance is maintained

### Final Metrics
**Minimum success:**
- Variance ratio > 40% (match/beat Run 1)
- R² > -0.04 (beat Run 1's -0.0457)
- Prediction spread > 0.5

**Good performance:**
- Variance ratio > 60%
- R² > 0.0 (beat mean baseline)
- Prediction spread > 1.0

---

## How to Run

```python
# In Colab
%cd /content/repo
!git pull origin main

# Run with stronger variance loss
from src.pipeline import run_pipeline
config_path = "/content/repo/configs/colab_optimized_full_tuned.yaml"
run_pipeline(config_path)
```

---

## What to Watch

### ✅ Good Signs
- Variance ratio stays 40-60% throughout training
- R² increases (becomes less negative, then positive)
- No "variance too low" warnings during early stopping
- Prediction spread > 0.5

### 🔴 Bad Signs
- Variance ratio drops below 30%
- Frequent "variance too low" warnings → increase lambda to 5.0
- R² stays below -0.1
- Emergency stop triggered

---

## If This Doesn't Work

### Plan B: Increase Lambda More
```yaml
variance_lambda: 5.0  # Even stronger penalty
```

### Plan C: Revert to Run 1 Config
```yaml
temporal_frames: 7        # Run 1's value
overweight_factor: 1.5    # Run 1's value
variance_lambda: 2.0      # Add variance loss to Run 1
```

Test if Run 1's success was due to temporal_frames=7 or other factors.

---

## Comparison to Previous Runs

| Run | Lambda | Min Var Check | Final Var Ratio | R² |
|-----|--------|--------------|-----------------|-----|
| 1 | N/A | N/A | 40.0% | -0.0457 ✅ |
| 2 | N/A | N/A | 3.9% | -0.0226 |
| 3 | N/A | N/A | 3.4% | -0.2034 |
| 4 | 0.5 | ❌ | 22.3% | -0.0655 |
| 5 | 2.0 ✅ | ✅ 35% | ? | ? |

**Goal:** Beat Run 1's 40% variance ratio and -0.0457 R²

---

## Key Files Modified

- `configs/colab_optimized_full_tuned.yaml` - variance_lambda=2.0, min_variance_ratio=0.35
- `src/train_model.py` - Variance check in early stopping
- `trainingrun/fourth/RUN4_ANALYSIS.md` - Detailed analysis

---

## Timeline

- **Run duration:** 2-3 hours on Colab T4
- **If successful:** Proceed to ablation study (temporal_frames 3 vs 5 vs 7)
- **If failed:** Try lambda=5.0 or Plan C (Run 1 config)

---

**Status:** Ready to run  
**Last commit:** [to be added after push]

---

## Quick Reference

**Pull latest code:**
```bash
git pull origin main
```

**Key parameters changed:**
- `variance_lambda: 0.5 → 2.0` (4x stronger)
- `min_variance_ratio: 0.35` (new check)

**Success criteria:**
- Variance ratio > 40%
- R² > -0.04
- No variance warnings

**If variance still collapses:** Increase lambda to 5.0