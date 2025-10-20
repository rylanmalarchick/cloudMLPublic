# Third Training Run Analysis
**Date:** October 19-20, 2024  
**Configuration:** Tier 1 (no SSL pretraining), std=0.1/bias=0.0 initialization, overweight_factor=1.5  
**Status:** ❌ **CRITICAL FAILURE - SEVERE VARIANCE COLLAPSE**

---

## Executive Summary

The third training run **performed worse than both previous runs**, exhibiting the most severe variance collapse yet observed:

- **R² = -0.2034** (worst of all three runs; model is 20% worse than predicting the mean)
- **Variance ratio = 3.4%** (predictions have only 3.4% of true variance)
- **Prediction spread = 0.068** (true spread = 1.8; 96% compression)
- All predictions clustered in narrow range [0.681, 0.749]

**The initialization "fix" (std=0.1, bias=0.0) made the problem WORSE, not better.**

---

## Final Test Metrics

```
Loss:  1.3475
MAE:   0.3138
MSE:   0.2804
RMSE:  0.5296
R²:    -0.2034
MAPE:  27.2%

Samples: 144
```

### Prediction Distribution

| Metric | True Values | Predicted Values | Ratio |
|--------|-------------|------------------|-------|
| Mean | 0.937 | 0.718 | -23.4% bias |
| Std Dev | 0.4827 | 0.0166 | **3.4%** |
| Range | [0.15, 1.95] | [0.681, 0.749] | 3.8% |
| Spread | 1.800 | 0.068 | 3.8% |

**Critical Finding:** The model is predicting nearly constant values around 0.72, with 96% of the variance removed.

---

## Comparison Across All Three Runs

| Metric | Run 1 (SSL broken) | Run 2 (std=0.01) | Run 3 (std=0.1) | Trend |
|--------|-------------------|------------------|-----------------|-------|
| **R²** | -0.0457 | -0.0226 | **-0.2034** | ⚠️ WORSE |
| **MAE** | 0.3965 | 0.3493 | **0.3138** | ✅ BETTER |
| **RMSE** | 0.4936 | 0.4882 | **0.5296** | ⚠️ WORSE |
| **Pred Std** | 0.1930 | 0.0190 | **0.0166** | ⚠️ WORSE |
| **Pred Spread** | 0.572 | 0.058 | **0.068** | ⚠️ WORSE |
| **Var Ratio** | 40.0% | 3.9% | **3.4%** | ⚠️ WORSE |

### Key Observations

1. **Run 1 was actually the best** despite having broken SSL pretraining
   - Had 40% variance ratio (still poor but 10x better than Run 3)
   - R² closest to zero (-0.0457 vs -0.2034)
   - Predictions showed some diversity (spread = 0.572)

2. **Run 2 and Run 3 both collapsed** after initialization changes
   - Both have ~4% variance ratio (near-total collapse)
   - Predictions are essentially constant
   - MAE improved but R² got much worse (model learned to predict biased constant)

3. **Contradictory metrics explained:**
   - MAE improved because model learned to predict values closer to the dense cluster around 0.6-0.7
   - R² and RMSE worsened because model lost ability to predict high values (1.0-2.0 range)
   - Model effectively "gave up" on variance and optimized for absolute error on common values

---

## Training Progression Analysis

### Third Run Training Dynamics

```
Epochs: 29 (early stopped)

First epoch:  train_loss=0.7269, val_loss=0.9866
Best epoch:   19, val_loss=0.5323
Last epoch:   train_loss=0.6275, val_loss=0.5837

Early stopped 10 epochs after best validation loss
```

### Comparison of Training

| Run | Initial Val Loss | Best Val Loss | Best Epoch | Total Epochs |
|-----|------------------|---------------|------------|--------------|
| Run 1 | 1.9769 | 0.5266 | 17 | 27 |
| Run 2 | 0.6949 | 0.5136 | 21 | 31 |
| Run 3 | 0.9866 | 0.5323 | 19 | 29 |

**All three runs achieved similar validation loss (~0.51-0.53) but produced vastly different prediction distributions.**

This indicates **the loss function does not adequately penalize variance collapse**. The model can achieve low validation loss while predicting near-constant values.

---

## Root Cause Analysis

### What Went Wrong in Run 3

1. **Output layer initialization (std=0.1, bias=0.0) was still too conservative**
   - Small initial weights (std=0.1) meant small initial gradients
   - Zero bias meant model started at 0.0, required learning to reach target range
   - Training likely got stuck in local minimum predicting ~0.72

2. **Loss function encourages variance collapse**
   - MSE/MAE losses don't explicitly penalize lack of variance
   - Predicting the median/mode is a valid strategy to minimize MAE
   - Model learned: "predict 0.72 for everything" → low MAE, terrible R²

3. **No variance preservation mechanism**
   - No regularization to maintain prediction diversity
   - No variance penalty in loss
   - Early stopping based on validation loss, not R²

4. **Overweight factor may be counterproductive**
   - overweight_factor=1.5 emphasizes high optical depth samples
   - But model may have learned to avoid high predictions (risky, high error)
   - Safer strategy: predict middle values for everything

### Why Run 1 Was Better

Run 1 had:
- **Larger initial weights** (default initialization, likely std ≈ 0.5)
- **Broken SSL pretraining** that actually helped by providing random but diverse features
- **Higher initial diversity** that training didn't fully collapse

The "broken" pretraining may have acted as beneficial noise/regularization!

---

## Critical Issues Identified

### Issue #1: Loss Function Does Not Reflect Model Quality
- **Validation loss ≈ 0.51-0.53** for all three runs
- But R² varies from -0.045 to -0.203 (4.5x difference!)
- **Early stopping uses validation loss, which doesn't detect variance collapse**

### Issue #2: Initialization Changes Made Problem Worse
- Attempt to "fix" initialization by reducing std and bias backfired
- Run 1's default initialization (larger std) was actually better
- **Smaller initial weights → smaller initial variance → harder to recover variance**

### Issue #3: Architecture or Training Dynamics Favor Collapse
- Multi-scale temporal attention with 7 frames
- Dropout, weight decay, overweighting
- All these regularizations push toward conservative predictions
- **No countermeasure to preserve variance**

### Issue #4: Self-Supervised Pretraining Implementation
- We disabled SSL in Runs 2 and 3 due to bugs
- But Run 1 (with broken SSL) performed best
- **Need to properly implement SSL or determine if it's beneficial**

---

## Prediction Analysis

### Prediction Histogram (Run 3)

```
Range [0.681-0.688]:   9 samples
Range [0.688-0.694]:   7 samples
Range [0.694-0.701]:  11 samples
Range [0.701-0.708]:  15 samples
Range [0.708-0.715]:  20 samples  ← peak
Range [0.715-0.722]:  14 samples
Range [0.722-0.729]:   8 samples
Range [0.729-0.735]:  44 samples  ← major peak
Range [0.735-0.742]:  12 samples
Range [0.742-0.749]:   4 samples
```

**86% of predictions fall in range [0.68, 0.74]** — essentially constant predictions.

### True Value Distribution

True values span [0.15, 1.95] with mean 0.94, but model ignores this and predicts ~0.72 for everything.

For samples with true optical depth > 1.2, the model systematically underpredicts by 0.5-1.2 units!

---

## Recommendations (Prioritized)

### 🔴 CRITICAL - Must Fix Immediately

#### 1. Fix Loss Function and Training Objective
**Problem:** Validation loss doesn't reflect model quality; doesn't penalize variance collapse.

**Solutions:**
```python
# Option A: Add variance preservation term
loss = mse_loss + lambda_var * (1.0 - pred_var / target_var)**2

# Option B: Use composite loss with R² penalty
loss = mse_loss - alpha * r2_score

# Option C: Add explicit variance regularization
loss = mse_loss + beta * torch.abs(pred.std() - target.std())

# Option D: Use quantile loss or distribution matching
```

**Recommended:** Start with Option A (variance term) with lambda_var=0.5.

#### 2. Change Early Stopping Criterion
**Current:** Stop on validation loss plateau  
**Problem:** Validation loss can be low while R² is terrible

**Solution:**
- Track R² on validation set during training
- Early stop when R² stops improving (or starts degrading)
- OR use composite metric: `metric = val_loss - 2.0 * r2`

#### 3. Revert to Run 1 Initialization or Larger Weights
**Problem:** Small initial weights (std=0.1) caused worse collapse than Run 1 (std≈0.5)

**Solutions:**
- **Option A:** Revert to default PyTorch initialization (Xavier/Kaiming)
- **Option B:** Use larger std for output layer (std=0.3 or 0.5)
- **Option C:** Initialize output layer with Xavier normal + bias=0.5

**Recommended:** Try Option A first (default init), it worked best in Run 1.

### 🟡 HIGH PRIORITY - Important to Address

#### 4. Reduce Temporal Smoothing
**Problem:** 7 frames with multi-scale attention may be over-averaging

**Solution:**
- Try `temporal_frames=3` or `temporal_frames=5`
- OR keep 7 frames but add skip connections
- OR use temporal dropout (randomly drop some frames during training)

#### 5. Adjust or Remove Overweight Factor
**Current:** overweight_factor=1.5 emphasizes high optical depth samples

**Problem:** May encourage model to avoid high predictions (risky)

**Solution:**
- Try overweight_factor=1.0 (no overweighting)
- OR use inverse weighting (emphasize low-sample regions)
- OR use focal loss approach

#### 6. Add Prediction Diversity Regularization
```python
# During training, add to loss:
diversity_loss = -torch.std(predictions)  # Penalize low std
# OR
diversity_loss = -torch.var(predictions)  # Penalize low variance
```

### 🟢 MEDIUM PRIORITY - Experiments to Try

#### 7. Run Ablation Study
Test configurations systematically:
```
Baseline:     temporal_frames=3, no pretraining, default init, overweight=1.0, variance loss
Exp 1:        temporal_frames=5, ...
Exp 2:        temporal_frames=7, ...
Exp 3:        ... + fixed SSL pretraining
Exp 4:        ... + overweight=1.5
```

#### 8. Fix and Re-enable Self-Supervised Pretraining
**Issues to fix:**
- Use multiple frames (not just first frame) for reconstruction
- Fix early stopping in pretraining (patience-based, not monotonic)
- Ensure pretraining converges properly

**Then test:** SSL vs no-SSL with proper training setup

#### 9. Diagnostic Monitoring During Training
Add logging every N steps:
- Prediction std dev on validation batch
- R² on validation batch
- Histogram of predictions
- Output layer weight norms

Stop training if prediction std drops below threshold (e.g., < 0.05).

---

## Immediate Next Steps

### Step 1: Quick Experiment (Recommended)
**Goal:** Prove that variance-preserving loss helps

**Changes:**
1. Revert to default initialization (remove custom init)
2. Add variance term to loss: `loss = mse + 0.5 * (1 - pred_var/target_var)**2`
3. Set overweight_factor=1.0
4. Set temporal_frames=5 (reduce over-smoothing)
5. Track R² during training, stop if it degrades

**Expected result:** R² > 0.0, variance ratio > 40%

### Step 2: If Step 1 Works
- Run full training with best config
- Try temporal_frames=3 and 7 for comparison
- Re-enable SSL pretraining (after fixing bugs)

### Step 3: If Step 1 Fails
- Consider architectural changes
- Try simpler baseline model
- Re-evaluate if this task is learnable with current data

---

## Files and Artifacts

### Logs
- Training CSV: `logs/csv/final_overweighted_tier1_tuned_20251020_021138.csv`
- Predictions: `logs/csv/predictions_final_overweighted_tier1_tuned_20251020_021138.csv`
- Metrics: `logs/csv/metrics_final_overweighted_tier1_tuned_20251020_021138.json`

### Plots
- Correlation: `plots/.../plots_correlation/correlation_enhanced.png`
- Flight path: `plots/.../plots_path/flight_path_highlighted.png`
- Positions: `plots/.../plots_positions/combined_positions_absolute_*.png`

### Export
- Full archive: `results_export.zip`

---

## Conclusion

**The third run definitively demonstrates that the current training setup is fundamentally flawed:**

1. ❌ Loss function doesn't capture model quality
2. ❌ Initialization changes made collapse worse
3. ❌ No mechanism to preserve prediction variance
4. ❌ Early stopping criterion is inadequate

**The path forward requires:**
- Fixing the loss function to penalize variance collapse
- Reverting to larger initial weights (Run 1 style)
- Changing early stopping to track R² or composite metric
- Reducing over-regularization (fewer frames, less overweighting)

**Run 1 was the best configuration so far.** The "fixes" in Runs 2 and 3 made the problem catastrophically worse. We should:
1. Return to Run 1's initialization approach
2. Add variance preservation
3. Fix validation monitoring
4. Then iterate from there

The model is capable of learning (Run 1 showed 40% variance ratio), but the current training regime crushes that capacity.

---

**Status:** Awaiting decision on next experiment configuration.