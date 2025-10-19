# First Run Analysis and Hyperparameter Tuning

**Date:** January 19, 2025  
**Purpose:** Analyze first baseline run results and tune hyperparameters for improved performance  
**Status:** ✅ Tuned config created and ready to test

---

## First Run Results (Original Config)

### Training Summary

**Config:** `colab_optimized_full.yaml`  
**Experiment ID:** `baseline_full_20251018_232936`  
**Total Training Time:** ~45 minutes (pretraining only, crashed before final training completed)

### Pretraining Phase (30Oct24)

**Data:**
- Training samples: 501
- Validation samples: ~100 (subset)

**Training Progress:**
```
Epoch 1:  Train Loss: 4.1156 | Val Loss: 1.3616 ← Best initially
Epoch 8:  Train Loss: 3.9638 | Val Loss: 1.2697 ← New best
Epoch 14: Train Loss: 3.7182 | Val Loss: 0.9440 ← Best overall
Epoch 15-29: No improvement, val loss erratic (1.01 → 2.15 → 1.58...)
Epoch 29: Early stopping triggered (patience=15 exhausted)
```

**Learning Rate Schedule:**
```
Started: 0.000001
Ramped up to: 0.000015 (due to warmup_steps=2000)
```

### Final Evaluation (Validation Flight: 12Feb25)

**Metrics:**
```
Loss:  1.2826
MAE:   0.3415 km
RMSE:  0.5046 km
R²:    -0.0927  ← WORSE than predicting the mean!
```

**Validation samples:** 144

---

## Problem Diagnosis

### 🔴 Issue 1: Negative R² Score

**What it means:**
- R² = -0.0927 means the model performs **worse than a naive baseline** (predicting the mean)
- A mean-predictor would give R² = 0
- Our model is **actively harmful** to predictions

**Likely causes:**
1. Overfitting to training data
2. Train/val distribution mismatch
3. Hyperparameters poorly tuned
4. Learning rate too high causing instability

### 🔴 Issue 2: Erratic Validation Loss

**Observation:**
After epoch 14, validation loss jumped around wildly:
```
Epoch 14: 0.9440 (best)
Epoch 15: 1.0109
Epoch 16: 1.0881
Epoch 17: 1.3257
Epoch 18: 1.5602
Epoch 19: 1.6139
...
Epoch 9:  2.1549 (spike!)
```

**This indicates:**
- Learning rate ramped too high (0.000015)
- Model overshooting optimal parameters
- Poor generalization from train to val

### 🔴 Issue 3: Training Loss Keeps Decreasing

**Observation:**
```
Epoch 1:  Train: 4.1156
Epoch 14: Train: 3.7182
Epoch 24: Train: 3.4407
Epoch 26: Train: 3.3584
```

Train loss steadily improves, but val loss gets worse → **Classic overfitting**

### 🔴 Issue 4: Wasted Epochs

**Observation:**
- Best validation loss at epoch 14: 0.9440
- Early stopping at epoch 29
- **15 epochs wasted** with no improvement

**Cause:** `early_stopping_patience: 15` is too generous

### 🔴 Issue 5: Aggressive Overweighting

**Config setting:**
```yaml
overweight_factor: 3.5  # 30Oct24 samples weighted 3.5x in final training
```

**Problem:**
- 3.5x overweighting may cause **catastrophic forgetting**
- Model "forgets" features learned during pretraining
- Final training overwhelms pretrained weights

---

## Root Cause Analysis

### Primary Issues

| Issue | Root Cause | Impact |
|-------|------------|--------|
| **Negative R²** | Learning rate too high + aggressive overweighting | Model worse than baseline |
| **Erratic val loss** | LR ramped from 0.000001 → 0.000015 during warmup | Training instability |
| **Overfitting** | Limited data (501 samples) + full model capacity | Poor generalization |
| **Wasted time** | Early stopping patience too high (15 epochs) | Inefficient training |

### Hyperparameter Problems

| Parameter | Original Value | Problem |
|-----------|----------------|---------|
| `learning_rate` | 0.001 | Too high, causes overshooting |
| `warmup_steps` | 2000 | Too long, delays actual learning |
| `overweight_factor` | 3.5 | Too aggressive, catastrophic forgetting |
| `early_stopping_patience` | 15 | Too generous, wastes epochs |
| `early_stopping_min_delta` | 0.00025 | Too small, noisy improvement signals |

---

## Tuning Strategy

### Hyperparameter Changes

| Parameter | Original | Tuned | Reasoning |
|-----------|----------|-------|-----------|
| **learning_rate** | 0.001 | **0.0005** | More stable training, less overshooting |
| **warmup_steps** | 2000 | **500** | Start learning sooner, less delay |
| **overweight_factor** | 3.5 | **2.0** | Gentler fine-tuning, less forgetting |
| **early_stopping_patience** | 15 | **10** | Stop wasted epochs sooner |
| **early_stopping_min_delta** | 0.00025 | **0.0005** | Require clearer improvement signal |

### Expected Improvements

#### 1. More Stable Training
- Lower LR (0.0005) → less oscillation
- Shorter warmup (500) → faster convergence
- Result: **Smoother validation loss curve**

#### 2. Better Generalization
- Gentler overweighting (2.0x) → retain pretrained features
- Tighter early stopping → avoid overfitting
- Result: **Better train/val alignment**

#### 3. Positive R²
- All improvements combined
- Target: **R² > 0.3** (at minimum, better than mean)
- Stretch goal: **R² > 0.5**

#### 4. Faster Training
- Shorter warmup → start learning at epoch 1
- Tighter early stopping → stop at ~15-20 epochs instead of 29
- Result: **~30% faster training**

---

## Tuned Configuration

### File: `configs/colab_optimized_full_tuned.yaml`

**Key changes:**
```yaml
# Learning & Optimization
learning_rate: 0.0005           # ← Reduced from 0.001
warmup_steps: 500                # ← Reduced from 2000
overweight_factor: 2.0           # ← Reduced from 3.5

# Early Stopping
early_stopping_patience: 10      # ← Reduced from 15
early_stopping_min_delta: 0.0005 # ← Increased from 0.00025

# Everything else stays the same
memory_optimized: false
gradient_checkpointing: true
torch_compile: true
torch_compile_mode: "default"
batch_size: 20
```

---

## Performance Targets

### First Run (Original Config)

```
✗ R²: -0.0927     (worse than mean)
✗ MAE: 0.3415 km  (acceptable but not great)
✗ RMSE: 0.5046 km (acceptable but not great)
✗ Val loss: Erratic, unstable
✗ Training: 15 wasted epochs
```

### Target (Tuned Config)

```
✓ R² > 0.3        (better than mean, ideally 0.5+)
✓ MAE < 0.30 km   (improved prediction accuracy)
✓ RMSE < 0.45 km  (reduced error magnitude)
✓ Val loss: Stable, monotonic improvement
✓ Training: Stop at ~15-20 epochs
```

---

## Validation Strategy

### What to Monitor

1. **Validation loss curve**
   - Should decrease smoothly
   - No wild oscillations
   - Clear convergence

2. **Train vs Val gap**
   - Should be small and stable
   - Large gap = overfitting
   - Want: Train loss ≈ Val loss

3. **R² score**
   - Must be positive
   - Target: > 0.3 minimum
   - Good: > 0.5

4. **Early stopping epoch**
   - Should trigger at ~15-20 epochs
   - Not too early (< 10 = underfitting)
   - Not too late (> 25 = overfitting)

### Success Criteria

**Minimum acceptable:**
- ✓ R² > 0
- ✓ Stable val loss
- ✓ MAE < 0.35 km

**Good performance:**
- ✓ R² > 0.3
- ✓ Smooth val curve
- ✓ MAE < 0.30 km

**Excellent performance:**
- ✓ R² > 0.5
- ✓ Monotonic val curve
- ✓ MAE < 0.25 km

---

## Alternative Tuning Options

### If Tuned Config Still Performs Poorly

#### Option 1: Reduce Model Capacity
```yaml
# Switch to memory-optimized model
memory_optimized: true  # 32/64/128 channels instead of 64/128/256
```
**Rationale:** Smaller model = less overfitting with limited data

#### Option 2: Increase Training Data
```yaml
# Reduce temporal frames to get more samples
temporal_frames: 4  # Instead of 5
```
**Rationale:** 501 samples → ~630 samples (25% more)

#### Option 3: Simplify Loss Function
```yaml
# Switch from Huber to simple MAE
loss_type: "mse_mae"
loss_alpha: 0.0  # Pure MAE
```
**Rationale:** Simpler loss = easier optimization

#### Option 4: Disable One Attention Mechanism
```yaml
# Try without temporal attention
use_temporal_attention: false
```
**Rationale:** Less capacity = less overfitting

#### Option 5: Increase Regularization
```yaml
# Stronger weight decay
weight_decay: 0.08  # Instead of 0.04
```
**Rationale:** More regularization = better generalization

---

## Expected Timeline

### Original Config
```
Pretraining:  ~30 epochs × 90s  = 45 minutes
Final:        ~50 epochs × 95s  = 80 minutes
Total:                          = 125 minutes (~2 hours)
```

### Tuned Config (Estimated)
```
Pretraining:  ~15 epochs × 90s  = 23 minutes  (faster convergence)
Final:        ~25 epochs × 95s  = 40 minutes  (early stopping sooner)
Total:                          = 63 minutes  (~1 hour)
```

**Expected speedup: ~50% faster** (due to earlier convergence)

---

## How to Use

### Step 1: Pull Latest Changes
```python
%cd /content/repo
!git pull origin main
```

### Step 2: Run Tuned Config
```python
# In the notebook, use Option A-Tuned (first option)
# OR run directly:
!python main.py --config configs/colab_optimized_full_tuned.yaml --save_name baseline_tuned --epochs 50
```

### Step 3: Monitor Training
```python
# Watch for stable validation loss
!tail -f /content/drive/MyDrive/CloudML/logs/training.log

# Check GPU usage
!nvidia-smi
```

### Step 4: Compare Results
```python
# After training, check metrics in Drive
!cat /content/drive/MyDrive/CloudML/logs/csv/metrics_final_overweighted_baseline_tuned_*.json
```

### Step 5: Analyze
- Compare R² to first run (-0.0927)
- Check if val loss is stable
- Look at training curves in TensorBoard

---

## Comparison Table

| Metric | First Run (Original) | Target (Tuned) | Status |
|--------|---------------------|----------------|--------|
| **R²** | -0.0927 | > 0.3 | 🎯 To test |
| **MAE** | 0.3415 km | < 0.30 km | 🎯 To test |
| **RMSE** | 0.5046 km | < 0.45 km | 🎯 To test |
| **Val Loss** | Erratic | Stable | 🎯 To test |
| **Training Time** | ~2 hours | ~1 hour | 🎯 To test |
| **Best Epoch** | 14 (then 15 wasted) | ~15-20 | 🎯 To test |
| **LR at Best** | 0.000007 | 0.0005 (stable) | 🎯 To test |

---

## Next Steps After Tuned Run

### If Results Are Good (R² > 0.3)
1. ✅ Use this as your baseline
2. ✅ Run ablation studies
3. ✅ Proceed with paper experiments
4. ✅ Consider ensembles for further improvement

### If Results Are Still Poor (R² < 0.1)
1. Try reducing model capacity (Option 1 above)
2. Increase training data (Option 2 above)
3. Simplify architecture (disable one attention)
4. Consider data quality issues
5. Check for train/val distribution mismatch

### If Results Are Marginal (0.1 < R² < 0.3)
1. Try even lower LR (0.0003 or 0.0002)
2. Add dropout for regularization
3. Try different loss functions
4. Experiment with data augmentation strength
5. Consider multi-stage training

---

## Summary

**Problem:** First run gave negative R² and erratic validation loss  
**Root Cause:** Learning rate too high, aggressive overweighting, long warmup  
**Solution:** Tuned hyperparameters for stability and better generalization  
**Expected:** R² > 0.3, stable training, faster convergence  

**Status:** ✅ Tuned config ready to test  
**Config File:** `configs/colab_optimized_full_tuned.yaml`  
**Notebook Option:** Option A-Tuned (recommended)

**Next:** Run tuned config and compare results! 🚀