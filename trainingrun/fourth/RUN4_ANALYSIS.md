# Run 4 Analysis - Variance Loss Too Weak

**Date:** October 20, 2024  
**Config:** Phase 1 fixes with variance_lambda=0.5  
**Status:** ❌ PARTIAL IMPROVEMENT - Variance loss helped but was too weak

---

## Executive Summary

Run 4 applied Phase 1 fixes (variance-preserving loss, R²-based early stopping, temporal_frames=3, etc.) but **still resulted in negative R²**.

**Critical Finding:** Variance ratio **collapsed during training** from 46% → 6% → 22%, despite the variance-preserving loss term. This means **lambda=0.5 is too weak** to counteract the base loss's pressure toward constant predictions.

---

## Final Results

```
Loss:  1.2601
MAE:   0.3457
RMSE:  0.4983
R²:    -0.0655
```

### Prediction Distribution
- **Samples:** 144
- **True:** mean=0.937, std=0.4827, range=[0.150, 1.950]
- **Pred:** mean=0.808, std=0.1076, range=[0.621, 0.925]
- **Variance ratio:** 22.3% (final test set)
- **Prediction spread:** 0.304

**Better than Run 3** (catastrophic collapse) but **still much worse than Run 1** (40% variance ratio).

---

## Training Progression

### Variance Ratio Over Time

| Epoch | Variance Ratio | R² | Val Loss | Status |
|-------|----------------|----|---------:|---------|
| 1 | 46.2% | -4.38 | 2.39 | Good variance! |
| 10 | 20.4% | -0.86 | 1.15 | Collapsing... |
| 20 | 29.6% | -1.83 | 1.66 | Recovery attempt |
| 30 | 25.7% | -0.10 | 0.98 | Stabilizing |
| **37** | **8.1%** | **-0.05** | **0.95** | **Best epoch (COLLAPSED!)** |
| 47 | 14.1% | -0.07 | 1.14 | Final (partial recovery) |

### Key Observations

1. **Started strong:** Epoch 1 had 46% variance ratio
2. **Rapid collapse:** By epoch 10, dropped to 20%
3. **Best validation loss = worst variance:** Epoch 37 (best val_loss) had only 8% variance ratio
4. **Early stopping picked wrong model:** Stopped at epoch 37 when variance was at minimum
5. **Later epochs were better:** Epochs 40+ had higher variance (14-20%) but worse val_loss

**Conclusion:** The model learned that sacrificing variance → lower val_loss, despite the variance penalty.

---

## Comparison to Previous Runs

| Run | Config | R² | Var Ratio | Pred Spread |
|-----|--------|----|-----------:|------------:|
| 1 | Broken SSL, std=0.5 | -0.0457 ✅ | 40.0% ✅ | 0.572 ✅ |
| 2 | std=0.01, no var loss | -0.0226 | 3.9% | 0.058 |
| 3 | std=0.1, no var loss | -0.2034 | 3.4% | 0.068 |
| 4 | Phase 1, lambda=0.5 | -0.0655 | 22.3% | 0.304 |

### Progress Assessment

**Improvement from Runs 2-3:**
- ✅ Variance ratio improved: 3-4% → 22% (5-7x better!)
- ✅ Predictions more diverse: spread 0.06 → 0.30 (5x better!)
- ❌ R² still negative (though better than Run 3)

**Still worse than Run 1:**
- ❌ Variance ratio: 22% vs 40% (nearly half)
- ❌ R²: -0.0655 vs -0.0457
- ❌ Prediction spread: 0.30 vs 0.57

**Conclusion:** Phase 1 fixes helped prevent catastrophic collapse, but variance_lambda=0.5 is insufficient.

---

## Root Cause Analysis

### Why Variance Loss Failed to Prevent Collapse

The variance-preserving loss is:
```python
loss = base_loss + lambda * (1 - pred_var/target_var)²
```

At **lambda=0.5** with typical values:
- `base_loss` ≈ 0.95 (Huber loss)
- `variance_loss` ≈ (1 - 0.08)² = 0.85 at 8% variance ratio
- `total_variance_penalty` ≈ 0.5 × 0.85 = 0.43

**Problem:** The base loss gradient is ~2-3x stronger than the variance gradient!
- Model learns: "Reduce base_loss even if it costs variance"
- Base loss is minimized by predicting near-mean values
- Variance penalty isn't strong enough to overcome this

### Why Early Stopping Failed

Early stopping picked **epoch 37** (best R² during training):
- R² = -0.0513
- Variance ratio = 8.1% (VERY LOW!)

But **later epochs were better** (e.g., epoch 47):
- R² = -0.0748 (slightly worse)
- Variance ratio = 14.1% (75% better!)

**Problem:** R²-based early stopping still picked a collapsed model because R² can be "less negative" with low variance if predictions are biased toward the right mean.

---

## Why Run 1 Was Still Best

Run 1 (with "broken" SSL and no variance loss) achieved:
- 40% variance ratio
- R² = -0.0457 (best of all runs)

**Possible reasons:**
1. **Random initialization diversity:** Broken SSL may have added beneficial noise
2. **No explicit variance penalty:** Model didn't learn to "game" the loss
3. **Different training dynamics:** Early stopping on val_loss may have accidentally preserved variance
4. **Larger temporal_frames (7):** More averaging, but also more information to extract variance from

---

## Recommended Fixes

### 🔴 CRITICAL: Increase Variance Lambda

**Current:** `variance_lambda: 0.5`  
**Problem:** Too weak, base loss dominates  
**Recommendation:**

**Option A (Conservative):** `variance_lambda: 2.0` (4x increase)
```yaml
variance_lambda: 2.0  # Make variance loss as important as base loss
```

**Option B (Aggressive):** `variance_lambda: 5.0` (10x increase)
```yaml
variance_lambda: 5.0  # Strongly penalize variance collapse
```

**Option C (Adaptive):** Start high, decrease over time
```python
# Start at 5.0, decay to 1.0 over training
variance_lambda = 5.0 * (0.8 ** epoch) + 1.0
```

**Recommended:** Start with Option A (lambda=2.0). If variance still collapses, try Option B.

---

### 🟡 HIGH PRIORITY: Change Early Stopping Criterion

**Current:** Stops on best R²  
**Problem:** Picks models with low variance ratio

**Fix:** Composite metric that weights variance more heavily
```python
# Stop on best composite score
composite = r2 + 2.0 * (variance_ratio - 0.7)  # Target 70% variance ratio
# Or minimum variance threshold
if variance_ratio < 0.4:  # Reject any model with <40% variance
    continue  # Don't consider this epoch for early stopping
```

---

### 🟡 EXPERIMENT: Try Different Loss Functions

**Current:** Huber loss + variance penalty

**Alternative 1:** Quantile loss (explicitly models distribution)
```python
loss = quantile_loss(y_pred, y_true, quantiles=[0.1, 0.5, 0.9])
```

**Alternative 2:** Distribution matching loss
```python
loss = mse_loss + kl_divergence(pred_distribution, target_distribution)
```

**Alternative 3:** Return to Run 1 config as baseline
- temporal_frames: 7 (not 3)
- Add variance_lambda: 2.0
- Keep std=0.5 initialization
- Test if "broken SSL" was actually helpful

---

### 🟢 DIAGNOSTIC: Monitor Variance Gradient Magnitude

Add logging to track if variance loss is strong enough:
```python
# During training
base_loss_grad = torch.autograd.grad(base_loss, model.output.weight, retain_graph=True)[0].norm()
var_loss_grad = torch.autograd.grad(variance_loss, model.output.weight, retain_graph=True)[0].norm()

print(f"Gradient ratio: base={base_loss_grad:.4f}, var={var_loss_grad:.4f}, ratio={var_loss_grad/base_loss_grad:.2f}")
```

Target: variance gradient should be 50-100% of base gradient magnitude.

---

## Immediate Next Steps

### Experiment 5A: Increase Lambda
```yaml
# configs/colab_optimized_full_tuned.yaml
variance_lambda: 2.0  # Increased from 0.5
temporal_frames: 3
overweight_factor: 1.0
```

**Expected results:**
- Variance ratio should stay above 40% throughout training
- R² might initially be worse, but should improve as variance is maintained
- Prediction spread > 0.5

**If this works:** Proceed to ablation study  
**If variance still collapses:** Try lambda=5.0

---

### Experiment 5B: Revert to Run 1 Config + Variance Loss
```yaml
# Test if Run 1's config with variance loss is optimal
temporal_frames: 7  # Revert to Run 1's value
variance_lambda: 2.0  # Add variance loss
overweight_factor: 1.5  # Run 1's value
# Everything else same as Run 1
```

**Purpose:** Determine if Run 1's success was due to:
- temporal_frames=7
- Lucky initialization
- Training dynamics
- Or something else

---

### Experiment 5C: Variance-Weighted Early Stopping
```python
# Modify early stopping to require minimum variance
if variance_ratio > 0.4 and r2 > best_r2:
    # Only save if variance is acceptable
    save_model()
```

**Purpose:** Prevent selecting collapsed models even if R² is "best"

---

## Success Criteria (Updated)

### Minimum Viable (Run 5)
- Variance ratio > 40% (match Run 1)
- Variance ratio stays above 35% throughout training (no collapse)
- R² > -0.04 (beat Run 1)
- Prediction spread > 0.5

### Good (Target)
- Variance ratio > 60%
- R² > 0.0 (better than mean baseline)
- Prediction spread > 1.0

### Excellent (Stretch Goal)
- Variance ratio > 80%
- R² > 0.3
- Clear linear correlation in scatter plots

---

## Lessons Learned

1. **Variance loss works, but needs to be stronger**
   - lambda=0.5 prevented catastrophic collapse (3% → 22%)
   - But wasn't strong enough to reach Run 1's 40%

2. **Early stopping on R² can still pick collapsed models**
   - Need to also check variance ratio
   - Composite metric or hard variance threshold required

3. **Training dynamics matter**
   - Model learns to minimize loss by sacrificing variance
   - Need to make variance preservation equally rewarding

4. **Run 1 remains the gold standard**
   - 40% variance ratio with default setup
   - Need to understand why and replicate it

---

## Files & Artifacts

**Logs:**
- Training: `logs/csv/final_overweighted_tier1_tuned_20251020_204931.csv`
- Predictions: `logs/csv/predictions_final_overweighted_tier1_tuned_20251020_204931.csv`
- Metrics: `logs/csv/metrics_final_overweighted_tier1_tuned_20251020_204931.json`

**Key Finding:** Variance ratio collapsed from 46% → 8% during training, showing lambda=0.5 is too weak.

---

## Next Action

**Recommended:** Run Experiment 5A with `variance_lambda: 2.0`

**Alternative:** If you want faster iteration, run Experiment 5B to test if Run 1's config + variance loss is the sweet spot.

**Timeline:** ~2-3 hours per experiment on Colab T4

---

**Status:** Waiting for decision on next experiment  
**Last updated:** 2024-10-20