# Next Steps for CloudML Training

**Date:** October 20, 2024  
**Status:** 🔴 CRITICAL - Training pipeline has fundamental issues that must be fixed before proceeding  
**Current Best:** Run 1 (R² = -0.045, Variance Ratio = 40%) - ironically, the "broken" version

---

## Executive Summary

After three training runs, we have identified **the root cause of failure:**

1. **Loss function doesn't penalize variance collapse** - Model achieves low validation loss (~0.51-0.53) while predicting near-constant values
2. **Initialization "fixes" made it worse** - Runs 2-3 with smaller weights have 3-4% variance ratio vs Run 1's 40%
3. **Early stopping criterion is wrong** - Stops based on validation loss, which doesn't correlate with R²
4. **Over-regularization** - 7 temporal frames + overweighting + dropout → conservative predictions

### The Paradox

- Run 1 (with broken SSL pretraining) = **BEST** performer (40% variance ratio)
- Run 2-3 (our "fixes" with smaller init) = **CATASTROPHIC** collapse (3-4% variance ratio)

**Conclusion:** We "fixed" the wrong things and made the core problem worse.

---

## Comparison of All Three Runs

| Metric | Run 1 (Broken SSL) | Run 2 (std=0.01) | Run 3 (std=0.1) | Winner |
|--------|-------------------|------------------|-----------------|---------|
| R² | -0.0457 | -0.0226 | **-0.2034** | Run 2 (least negative) |
| MAE | 0.3965 | 0.3493 | **0.3138** | Run 3 |
| Variance Ratio | **40.0%** | 3.9% | 3.4% | **Run 1** |
| Pred Spread | **0.572** | 0.058 | 0.068 | **Run 1** |
| Correlation | **0.143** | 0.054 | 0.067 | **Run 1** |

**Key Insight:** All runs achieved similar validation loss (~0.51-0.53), but Run 1 is 10x better at preserving variance.

---

## Immediate Action Plan

### Phase 1: Fix the Core Issues (REQUIRED BEFORE ANY FURTHER TRAINING)

#### 1. Implement Variance-Preserving Loss Function

**File:** `src/train_model.py`

**Change:**
```python
# Current (WRONG - only uses MSE/MAE)
loss = criterion(predictions, targets)

# New (CORRECT - penalizes variance collapse)
mse_loss = criterion(predictions, targets)
pred_var = predictions.var()
target_var = targets.var()
variance_loss = (1.0 - pred_var / (target_var + 1e-8)) ** 2
loss = mse_loss + 0.5 * variance_loss  # lambda=0.5 is tunable
```

**Rationale:** Without this, model will always collapse to near-constant predictions to minimize MAE.

#### 2. Change Early Stopping to Track R²

**File:** `src/train_model.py` or `src/pipeline.py`

**Change:**
```python
# Current (WRONG - validation loss doesn't reflect quality)
early_stopping_metric = val_loss

# New (CORRECT - R² actually measures prediction quality)
r2 = calculate_r2(predictions, targets)
early_stopping_metric = -r2  # Negative because we minimize
# OR use composite: val_loss - 2.0 * r2
```

**Rationale:** Validation loss of 0.51-0.53 can mean R² of -0.045 OR -0.203. We need to optimize what we care about.

#### 3. Revert Output Layer Initialization

**File:** `src/pytorchmodel.py`

**Change:**
```python
# Current (WRONG - too small, causes collapse)
torch.nn.init.normal_(self.output.weight, mean=0.0, std=0.1)
torch.nn.init.constant_(self.output.bias, 0.0)

# New (CORRECT - use default or larger std)
# Option A: Just remove custom init (use PyTorch defaults)
# Option B: Explicitly use larger std
torch.nn.init.normal_(self.output.weight, mean=0.0, std=0.5)
torch.nn.init.constant_(self.output.bias, 0.0)
```

**Rationale:** Run 1's larger initial weights (std≈0.5) achieved 40% variance ratio. Smaller weights = smaller gradients = harder to recover variance.

#### 4. Add Prediction Diversity Monitoring

**File:** `src/train_model.py`

**Add to training loop:**
```python
# After each validation epoch
pred_std = predictions.std().item()
pred_range = predictions.max().item() - predictions.min().item()
target_std = targets.std().item()
variance_ratio = pred_std / target_std

# Log these metrics
wandb.log({
    'val/pred_std': pred_std,
    'val/pred_range': pred_range,
    'val/variance_ratio': variance_ratio,
})

# Emergency stop if collapse detected
if variance_ratio < 0.05:
    print(f"WARNING: Severe variance collapse detected (ratio={variance_ratio:.3f}). Stopping.")
    break
```

**Rationale:** Catch collapse early instead of wasting hours on doomed training.

---

### Phase 2: Reduce Over-Regularization

#### 5. Reduce Temporal Frames

**File:** `configs/colab_optimized_full_tuned.yaml`

**Change:**
```yaml
# Current
temporal_frames: 7  # Too much smoothing

# New
temporal_frames: 3  # Start conservative, test 5 later
```

**Rationale:** 7 consecutive frames may be over-averaging and destroying high-frequency variance signals.

#### 6. Disable Overweighting

**File:** `configs/colab_optimized_full_tuned.yaml`

**Change:**
```yaml
# Current
overweight_factor: 1.5  # May encourage conservative predictions

# New
overweight_factor: 1.0  # No overweighting
```

**Rationale:** Overweighting high optical depths may paradoxically make model avoid high predictions (too risky).

---

### Phase 3: Run Validation Experiment

**Goal:** Prove that fixes work before investing in long training.

**Configuration:**
```yaml
experiment_name: "tier1_variance_fix_test"
temporal_frames: 3
overweight_factor: 1.0
pretraining:
  enabled: false  # Keep disabled for now
training:
  max_epochs: 50
  early_stopping_patience: 10
  # New: track R² and variance ratio
```

**Code changes:**
- Variance-preserving loss (lambda=0.5)
- Early stop on R² instead of val_loss
- Default output initialization (or std=0.5)
- Log variance ratio every epoch

**Expected Results:**
- R² > 0.0 (at minimum, better than mean)
- Variance ratio > 40% (better than Run 1)
- Prediction spread > 0.5
- Training should show increasing variance ratio over epochs

**If this succeeds:** Proceed to Phase 4  
**If this fails:** Re-evaluate architecture or data quality

---

### Phase 4: Systematic Ablation Study

Once Phase 3 validates the fixes, run controlled experiments:

| Experiment | temporal_frames | variance_loss | overweight | SSL | Expected R² |
|------------|----------------|---------------|------------|-----|-------------|
| Baseline | 3 | ✅ | 1.0 | ❌ | > 0.0 |
| Exp 1 | 5 | ✅ | 1.0 | ❌ | > 0.1 |
| Exp 2 | 7 | ✅ | 1.0 | ❌ | > 0.0 |
| Exp 3 | 5 | ✅ | 1.5 | ❌ | > 0.1 |
| Exp 4 | 5 | ✅ | 1.0 | ✅ (fixed) | > 0.2 |

**Purpose:** Find optimal hyperparameters with working loss function.

---

### Phase 5: Fix Self-Supervised Pretraining (Optional)

**Current bugs:**
1. Pretraining uses only first frame (`images[:,0,:,:]`) instead of all frames
2. Early stopping in pretraining is monotonic, not patience-based
3. Pretraining validation loss increased, suggesting it diverged

**Fixes needed in `src/pretraining.py`:**

```python
# Bug 1: Use multiple frames
# Current (WRONG)
reconstructed = decoder(encoded)  # Only first frame

# Fixed (CORRECT)
# Option A: Reconstruct all frames with frame-wise loss
for i in range(temporal_frames):
    frame_i = images[:, i, :, :]
    encoded_i = encoder(frame_i)
    recon_i = decoder(encoded_i)
    loss += reconstruction_loss(recon_i, frame_i)

# Option B: Use middle frame or random frame
frame_idx = random.randint(0, temporal_frames - 1)
frame = images[:, frame_idx, :, :]
```

**After fixing:**
- Run pretraining standalone and verify it converges (val loss decreases)
- Then run Exp 4 from Phase 4 with SSL enabled
- Compare SSL vs no-SSL performance

---

## Files That Need Changes

### Critical (Phase 1 - Required)
1. `src/train_model.py` - Add variance loss, change early stopping
2. `src/pytorchmodel.py` - Revert output initialization
3. `configs/colab_optimized_full_tuned.yaml` - Reduce temporal_frames, set overweight=1.0

### Important (Phase 2-3)
4. `src/pipeline.py` - Update metrics tracking and logging
5. `colab_training.ipynb` - Update instructions and expected results

### Optional (Phase 5)
6. `src/pretraining.py` - Fix multi-frame reconstruction
7. `configs/pretraining_config.yaml` - Adjust if needed

---

## What NOT To Do

❌ **Don't run more experiments without fixing the loss function**  
   → You'll just waste compute reproducing the same variance collapse

❌ **Don't make initialization even smaller (std < 0.1)**  
   → This made Run 3 the worst of all runs

❌ **Don't increase temporal_frames beyond 7**  
   → Already too much smoothing; need to reduce, not increase

❌ **Don't rely on validation loss as primary metric**  
   → It's decorrelated from R²; use R² or composite metric

❌ **Don't re-enable SSL pretraining until bugs are fixed**  
   → Current implementation is broken and untested

---

## Success Criteria

### Minimum Viable Performance
- R² > 0.3
- Variance ratio > 70%
- Prediction spread > 1.0
- RMSE < 0.35
- MAE < 0.25

### Good Performance
- R² > 0.5
- Variance ratio > 85%
- Prediction spread > 1.4
- RMSE < 0.30
- Strong correlation visible in scatter plot

### Excellent Performance
- R² > 0.7
- Variance ratio > 95%
- RMSE < 0.25
- Clear linear correlation in predictions

---

## Timeline Estimate

| Phase | Tasks | Time | Blocker |
|-------|-------|------|---------|
| Phase 1 | Code changes (4 files) | 2-3 hours | None |
| Phase 2 | Config changes | 30 min | Phase 1 |
| Phase 3 | Test run (50 epochs) | 2-3 hours | Phase 1+2 |
| Phase 4 | Ablation (5 runs) | 10-15 hours | Phase 3 success |
| Phase 5 | Fix SSL + test | 4-6 hours | Phase 4 |

**Total:** ~20-30 hours to completion, assuming Phase 3 validates approach.

---

## Risk Mitigation

### If Phase 3 Still Shows Collapse

**Possible causes:**
1. Variance loss lambda too small → try 1.0 or 2.0
2. Still over-regularized → reduce dropout, remove weight decay
3. Architecture fundamentally incompatible → try simpler baseline model
4. Data quality issue → inspect input images for artifacts

**Diagnostic steps:**
1. Plot prediction distribution at epochs 1, 5, 10, 20
2. Log output layer weight norms (should increase, not decrease)
3. Compute gradient norms for output layer (should be non-trivial)
4. Try training without temporal attention (single-frame baseline)

### If Phase 3 Succeeds But Phase 4 Plateaus

- R² stuck around 0.3-0.4
- Can't break through to 0.5+

**Possible improvements:**
- Increase model capacity (more attention heads, larger hidden dim)
- Better data augmentation
- Ensemble multiple models
- Incorporate additional features (terrain, weather, etc.)

---

## Questions for User

Before proceeding with Phase 1 implementation:

1. **Priority:** Do you want me to implement Phase 1 changes now, or would you prefer to review the analysis first?

2. **Compute availability:** How much Colab time do you have available? Phase 3 test is ~3 hours, Phase 4 ablation is ~15 hours.

3. **Validation set:** Is the current train/val/test split appropriate, or should we adjust?

4. **Success definition:** Are the success criteria (R² > 0.3 minimum, > 0.5 good) aligned with your project goals?

5. **SSL pretraining:** Should we fix and test SSL in Phase 5, or abandon it and focus on supervised-only training?

---

## References

- **Previous analysis:** `trainingrun/first/BUG_INVESTIGATION_REPORT.md`
- **Current analysis:** `trainingrun/third/THIRD_RUN_ANALYSIS.md`
- **Comparison script:** `trainingrun/compare_runs.py`
- **Original changes:** `CHANGES_2024_10_19.md`

---

## Contact & Next Actions

**Current status:** Awaiting decision on Phase 1 implementation.

**Recommended immediate action:**  
Implement Phase 1 changes (variance loss + early stopping + init fix) and run Phase 3 validation experiment.

**Expected next update:**  
Phase 3 results with metrics showing whether variance collapse is resolved.

---

*Last updated: 2024-10-20*  
*Status: Ready to implement fixes*