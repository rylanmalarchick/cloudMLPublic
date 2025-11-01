# PHASE 3: FINE-TUNING FOR CBH REGRESSION

**Status:** ✅ Ready to Run  
**Date:** 2025-01-31  
**Prerequisites:** ✅ Phase 1 Complete, ✅ Phase 2 Complete

---

## Overview

This phase fine-tunes the pre-trained MAE encoder for Cloud Base Height (CBH) regression using the labeled CPL-aligned dataset (~933 samples).

### Approach: Two-Stage Fine-Tuning

**Stage 1: Train Head Only (Frozen Encoder)**
- Freeze pre-trained encoder weights
- Train regression head from scratch
- Duration: ~30 epochs
- Purpose: Initialize head without corrupting encoder

**Stage 2: Fine-Tune End-to-End (Unfrozen Encoder)**
- Unfreeze encoder weights
- Fine-tune entire model together
- Duration: ~50 epochs
- Purpose: Adapt encoder to regression task

### Target Performance

**Classical Baseline (GradientBoosting):**
- R² = 0.7464
- MAE = 0.1265 km
- RMSE = 0.1929 km

**Goal:** Beat or match classical baseline using deep learning with SSL pre-training.

---

## Quick Start

```bash
# 1. Verify Phase 2 is complete
ls -lh outputs/mae_pretrain/mae_encoder_pretrained.pth

# 2. Run fine-tuning
./scripts/run_phase3_finetune.sh

# 3. Monitor progress (in another terminal)
tensorboard --logdir outputs/cbh_finetune/logs/

# 4. View results
xdg-open outputs/cbh_finetune/plots/test_results.png
```

**Estimated Time:**
- GTX 1070 Ti: ~1-2 hours (both stages combined)
- RTX 3090: ~30-45 minutes

---

## Configuration

### Config File: `configs/ssl_finetune_cbh.yaml`

Key settings you may want to adjust:

```yaml
# Data split
data:
  train_ratio: 0.7      # 70% for training
  val_ratio: 0.15       # 15% for validation
  test_ratio: 0.15      # 15% for final test

# Model architecture
model:
  head:
    hidden_dims: [256, 128]  # MLP layers
    dropout: 0.3             # Regularization
    activation: "gelu"       # Activation function

# Training - Stage 1 (freeze encoder)
training:
  stage1:
    epochs: 30
    batch_size: 32
    optimizer:
      lr: 1.0e-3           # Higher LR for head
      weight_decay: 0.01

  # Training - Stage 2 (finetune all)
  stage2:
    epochs: 50
    batch_size: 32
    optimizer:
      lr: 1.0e-4           # Lower LR for fine-tuning
      weight_decay: 0.01
```

### Hyperparameter Tuning Suggestions

If results are suboptimal, try:

1. **Increase regularization** (if overfitting):
   - `head.dropout: 0.5`
   - `weight_decay: 0.05`

2. **Adjust learning rates**:
   - Stage 1 LR: `5e-4` to `2e-3`
   - Stage 2 LR: `5e-5` to `2e-4`

3. **Longer training**:
   - `stage1.epochs: 50`
   - `stage2.epochs: 100`

4. **Different head architecture**:
   - Deeper: `[512, 256, 128]`
   - Wider: `[384, 384]`

---

## Dataset

### Source
CPL-aligned samples from 5 flights (same as original experiments):
- 10Feb25, 30Oct24, 23Oct24, 18Feb25, 12Feb25

### Filtering
- Filter type: `basic` (cloud base heights in valid range)
- CPL alignment: Time-matched IRAI images to CPL cloud base observations
- Quality checks: Remove NaN/invalid heights

### Split Strategy
- **Train (70%):** Used for both Stage 1 and Stage 2 training
- **Validation (15%):** Early stopping and hyperparameter selection
- **Test (15%):** Final evaluation (never seen during training)

### Data Augmentation
**Training:** Enabled (same as SSL pre-training)
- Color jitter (brightness, contrast, saturation)
- Random erasing (simulate missing data)
- Horizontal flips

**Validation/Test:** Disabled (raw data only)

---

## Model Architecture

### Full Pipeline

```
Input Image (440 pixels)
         ↓
[Pre-trained MAE Encoder]
  - Patch embedding (16 pixels/patch)
  - Transformer blocks (6 layers)
  - Output: 192-dim embedding
         ↓
[Regression Head]
  - Linear(192 + angles → 256)
  - BatchNorm + GELU + Dropout
  - Linear(256 → 128)
  - BatchNorm + GELU + Dropout
  - Linear(128 → 1)
         ↓
Predicted CBH (km)
```

### Parameters

**Total:** ~2.3M parameters
- **Encoder:** ~2.0M (pre-trained)
- **Head:** ~0.3M (trained from scratch)

**Stage 1:** Only ~0.3M trainable (encoder frozen)
**Stage 2:** All ~2.3M trainable (encoder unfrozen)

---

## Training Details

### Stage 1: Freeze Encoder (Epochs 0-30)

**Strategy:**
- Encoder weights frozen (no gradients)
- Only regression head trained
- Higher learning rate (1e-3)

**Why?**
- Allows head to learn from scratch without corrupting pre-trained encoder
- Head starts random; encoder starts well-initialized
- Prevents "catastrophic forgetting" of SSL representations

**Expected Behavior:**
- Fast initial improvement (head learning from strong encoder)
- Val R² should reach ~0.4-0.6 by end of Stage 1

### Stage 2: Fine-Tune All (Epochs 30-80)

**Strategy:**
- Encoder weights unfrozen (gradients flow everywhere)
- Lower learning rate (1e-4)
- Fine-tune encoder + head together

**Why?**
- Adapt encoder representations to regression task
- Refine features for CBH prediction specifically
- Encoder adjusts while preserving SSL knowledge

**Expected Behavior:**
- Gradual improvement beyond Stage 1 performance
- Smaller learning rate prevents encoder collapse
- Val R² should reach target (~0.65-0.75+)

### Loss Function

**Default:** MSE (Mean Squared Error)

Alternatives in config:
- `mae`: Mean Absolute Error (L1 loss)
- `huber`: Huber loss (robust to outliers)
- `smooth_l1`: Smooth L1 loss

### Early Stopping

**Stage 1:** Patience = 15 epochs
**Stage 2:** Patience = 20 epochs

Stops training if validation loss doesn't improve for N epochs.

---

## Evaluation Metrics

### Primary Metrics

1. **R² (Coefficient of Determination)**
   - Measures explained variance
   - Range: (-∞, 1], higher is better
   - 1.0 = perfect predictions
   - 0.0 = predicts mean only
   - < 0 = worse than mean

2. **MAE (Mean Absolute Error)**
   - Average |predicted - true| in km
   - Lower is better
   - Baseline: 0.1265 km

3. **RMSE (Root Mean Squared Error)**
   - Sqrt of MSE, penalizes large errors
   - Lower is better
   - Baseline: 0.1929 km

### Success Thresholds

| Threshold | R² Range | Interpretation |
|-----------|----------|----------------|
| 🎉 **Excellent** | R² ≥ 0.75 | Beat classical baseline! Ready for publication |
| ✅ **Good** | 0.60 ≤ R² < 0.75 | Competitive performance, promising approach |
| 👍 **Acceptable** | 0.40 ≤ R² < 0.60 | Proof of concept works, needs refinement |
| ⚠️ **Below Target** | R² < 0.40 | Re-evaluate approach or hyperparameters |

---

## Output Files

### Directory Structure

```
outputs/cbh_finetune/
├── checkpoints/
│   ├── stage1_freeze_best.pth      # Best Stage 1 checkpoint
│   ├── stage1_freeze_final.pth     # Final Stage 1 checkpoint
│   ├── stage2_finetune_best.pth    # Best Stage 2 checkpoint
│   ├── stage2_finetune_final.pth   # Final Stage 2 checkpoint
│   └── final_model.pth             # Best overall model
├── logs/
│   └── YYYYMMDD_HHMMSS/            # TensorBoard logs
│       ├── events.out.tfevents.*
│       └── ...
├── plots/
│   └── test_results.png            # Scatter + residual plots
└── predictions/
    └── (optional) test_predictions.npz
```

### Checkpoint Contents

Each `.pth` file contains:
```python
{
    'epoch': int,                    # Epoch number
    'model_state_dict': dict,        # Model weights
    'optimizer_state_dict': dict,    # Optimizer state
    'val_loss': float,               # Validation loss
    'val_metrics': {                 # Validation metrics
        'r2': float,
        'mae': float,
        'rmse': float,
        'mse': float,
    }
}
```

### Loading a Checkpoint

```python
import torch
from scripts.finetune_cbh import CBHModel
from src.mae_model import MAEEncoder

# Create model
encoder = MAEEncoder(...)
model = CBHModel(encoder, head_config, num_angles)

# Load checkpoint
checkpoint = torch.load('outputs/cbh_finetune/checkpoints/final_model.pth')
model.load_state_dict(checkpoint)
model.eval()

# Make predictions
with torch.no_grad():
    pred = model(images, angles)
```

---

## Monitoring Training

### TensorBoard

**Start TensorBoard:**
```bash
tensorboard --logdir outputs/cbh_finetune/logs/
```

**Open browser:** http://localhost:6006

**Key Plots to Watch:**

1. **Train/Val Loss**
   - Should decrease steadily
   - Gap indicates overfitting

2. **Train/Val R²**
   - Should increase toward baseline
   - Stage 1 → Stage 2 transition visible

3. **Learning Rate**
   - Stage 1: 1e-3 → cosine decay
   - Stage 2: 1e-4 → cosine decay

4. **Metrics Comparison**
   - MAE, RMSE trends
   - Per-stage improvements

### Console Output

**Example output:**
```
Epoch   1/30 | Train Loss: 0.012345 | Val Loss: 0.013456 | Val R²: 0.4523 | LR: 1.00e-03
    💾 Saved best checkpoint (val_loss: 0.013456, R²: 0.4523)
Epoch   2/30 | Train Loss: 0.011234 | Val Loss: 0.012345 | Val R²: 0.5012 | LR: 9.98e-04
    💾 Saved best checkpoint (val_loss: 0.012345, R²: 0.5012)
...
```

---

## Results Visualization

### Test Results Plot

**File:** `outputs/cbh_finetune/plots/test_results.png`

**Left panel:** Scatter plot (Predicted vs True)
- Points should cluster along diagonal
- R² score shown in title

**Right panel:** Residual histogram
- Should be centered at 0
- Narrow spread indicates low error
- MAE shown in title

### Interpreting Results

**Good signs:**
- Tight scatter around diagonal
- Symmetric residual distribution
- R² close to or above 0.7464

**Warning signs:**
- Systematic bias (residuals not centered)
- High scatter (low R²)
- Outliers far from diagonal

---

## Troubleshooting

### Issue: Low R² (< 0.40)

**Possible causes:**
1. Pre-training didn't converge well
2. Head architecture too simple/complex
3. Overfitting (small dataset)
4. Learning rate too high/low

**Solutions:**
- Check Phase 2 MAE loss (should be < 0.01)
- Try different head architectures
- Increase dropout / weight decay
- Adjust learning rates
- Train longer (more epochs)

### Issue: Overfitting (train R² >> val R²)

**Symptoms:**
- Training R² = 0.9, Validation R² = 0.3
- Val loss increases while train loss decreases

**Solutions:**
- Increase dropout: `0.3 → 0.5`
- Increase weight decay: `0.01 → 0.05`
- Reduce head size: `[256, 128] → [128, 64]`
- More augmentation (Phase 1 config)
- Early stopping (reduce patience)

### Issue: Underfitting (both train/val R² low)

**Symptoms:**
- Training R² = 0.3, Validation R² = 0.3
- Loss plateaus early

**Solutions:**
- Increase model capacity: `[256, 128] → [512, 256, 128]`
- Decrease dropout: `0.3 → 0.1`
- Decrease weight decay: `0.01 → 0.001`
- Increase learning rate
- Train longer

### Issue: NaN Loss

**Causes:**
- Learning rate too high
- Gradient explosion
- Bad data (NaN/Inf in inputs)

**Solutions:**
- Reduce learning rate: `1e-3 → 1e-4`
- Enable gradient clipping: `grad_clip: 1.0`
- Check data quality (Phase 1 verification)
- Use mixed precision: `use_amp: true`

### Issue: Out of Memory (OOM)

**Solutions:**
- Reduce batch size: `32 → 16`
- Reduce num_workers: `4 → 2`
- Disable persistent_workers
- Use gradient accumulation

---

## Advanced Experiments (After Phase 3)

Once you achieve good performance (R² ≥ 0.60), consider:

### 1. Encoder Representation Analysis
- Extract embeddings from encoder
- Visualize with t-SNE/UMAP
- Cluster by CBH ranges
- Understand what encoder learned

### 2. Per-Flight Evaluation
- Evaluate each flight separately
- Identify challenging conditions
- Flight-specific fine-tuning

### 3. Multi-Task Learning
- Predict CBH + cloud type simultaneously
- Shared encoder, multiple heads
- Leverage CPL cloud classification

### 4. Temporal Modeling
- Use sequence of frames (not just 3)
- Add LSTM/Transformer on top of encoder
- Predict CBH trajectory

### 5. Uncertainty Estimation
- Add dropout at inference (MC Dropout)
- Predict confidence intervals
- Identify unreliable predictions

### 6. Hyperparameter Optimization
- Grid/random search over:
  - Learning rates (1e-5 to 1e-2)
  - Dropout (0.0 to 0.5)
  - Head architectures
  - Batch sizes
- Use Optuna or Ray Tune

---

## Expected Performance

### Conservative Estimate
- **R² ≈ 0.60-0.70**
- MAE ≈ 0.12-0.15 km
- RMSE ≈ 0.16-0.20 km

**Rationale:**
- SSL provides good initialization
- Small labeled dataset limits fine-tuning
- Classical methods have feature engineering advantage

### Optimistic Estimate
- **R² ≈ 0.70-0.80**
- MAE ≈ 0.10-0.12 km
- RMSE ≈ 0.14-0.16 km

**Conditions:**
- Pre-training converged well (MAE loss < 0.01)
- Good hyperparameter tuning
- Encoder learned generalizable features

### If Performance is Lower (R² < 0.50)

**Don't give up!** This is still valuable research:

1. **Document the attempt:**
   - SSL approach for CBH regression
   - Comparison to classical baseline
   - Lessons learned

2. **Analyze failures:**
   - What did encoder learn?
   - Why didn't it transfer?
   - Dataset size limitations?

3. **Write a "negative results" section:**
   - Valuable for research community
   - Shows what doesn't work
   - Guides future approaches

---

## Research Output

### If Successful (R² ≥ 0.60)

**Paper sections:**
1. **Introduction:** SSL for limited-label remote sensing
2. **Methods:** MAE pre-training + fine-tuning pipeline
3. **Results:** Beat/match classical baseline with DL
4. **Discussion:** Encoder representations, transfer learning
5. **Conclusion:** SSL enables DL in small-data regimes

### If Unsuccessful (R² < 0.60)

**Paper sections:**
1. **Introduction:** Limitations of SSL for CBH
2. **Methods:** Thorough SSL pipeline description
3. **Results:** Classical methods outperform SSL
4. **Discussion:** Why SSL failed, dataset challenges
5. **Conclusion:** When to use classical vs DL methods

**Both outcomes are publishable!**

---

## Comparison to Baseline

### Classical ML (GradientBoosting)
**Advantages:**
- No pre-training needed
- Works well with small data
- Interpretable feature importance
- Fast training

**Disadvantages:**
- Requires manual feature engineering
- No representation learning
- Limited to tabular features

### Deep Learning (MAE + Fine-tuning)
**Advantages:**
- Learns representations automatically
- Leverages unlabeled data (62k images)
- Scalable to more data
- Can handle raw images

**Disadvantages:**
- Requires pre-training (12-24 hours)
- More complex pipeline
- Needs careful tuning
- Black-box representations

---

## Next Steps After Phase 3

### If R² ≥ 0.75 (Excellent)
✅ **Ready for publication!**
- Write paper: "Self-Supervised Learning for Cloud Base Height Estimation"
- Submit to remote sensing conference/journal
- Explore advanced experiments (multi-task, temporal)

### If 0.60 ≤ R² < 0.75 (Good)
👍 **Promising results!**
- Refine approach (hyperparameter tuning, more pre-training)
- Try different SSL methods (SimCLR, DINO)
- Write paper emphasizing SSL methodology

### If 0.40 ≤ R² < 0.60 (Acceptable)
🤔 **Proof of concept works**
- Analyze what encoder learned (embeddings)
- Identify gaps vs classical methods
- Document as "promising but needs work"

### If R² < 0.40 (Below Target)
🔄 **Re-evaluate**
- Check Phase 2 pre-training quality
- Try longer pre-training (200 epochs)
- Consider hybrid approach (SSL features → classical ML)
- Document negative results (still valuable!)

---

## Summary

**Phase 3 Goal:** Fine-tune pre-trained encoder for CBH regression

**Input:** 
- Pre-trained MAE encoder (`mae_encoder_pretrained.pth`)
- ~933 labeled CPL-aligned samples

**Output:**
- Fine-tuned CBH model
- Performance vs classical baseline
- Insights on SSL effectiveness

**Success Metric:** R² ≥ 0.60 (ideally ≥ 0.75)

**Time Investment:** 1-2 hours training + analysis

**Value:**
- Completes 3-phase SSL pipeline
- Demonstrates transfer learning for CBH
- Provides publishable results (success or failure)

---

## Commands Reference

```bash
# Run fine-tuning
./scripts/run_phase3_finetune.sh

# Custom config
./scripts/run_phase3_finetune.sh configs/custom_finetune.yaml

# Monitor training
tensorboard --logdir outputs/cbh_finetune/logs/

# View results
xdg-open outputs/cbh_finetune/plots/test_results.png

# Check GPU usage (during training)
watch -n 1 nvidia-smi
```

---

**Ready to run Phase 3!** 🚀

See `PHASE3_READY.md` for final checklist and next steps.