# Phase 2: MAE Self-Supervised Pre-Training Guide

## Overview

Phase 2 trains a **Masked Autoencoder (MAE)** on all ~60k unlabeled images extracted in Phase 1. The MAE learns robust visual representations of cloud structure without requiring labels.

## Background

### Why Self-Supervised Learning?

Based on Phase 1-2 diagnostic findings:
- ✅ **Section 1:** Task is learnable (GradientBoosting R² = 0.7464)
- ❌ **Section 2:** Supervised DL failed with ~933 labels (all negative R²)
- 💡 **Solution:** Pre-train on ALL ~60k images via SSL, then fine-tune

### How MAE Works

1. **Randomly mask** 75% of image patches
2. **Encode** visible (unmasked) patches with transformer
3. **Decode** to reconstruct masked patches
4. **Learn** representations by minimizing reconstruction error

**Key insight:** Reconstruction forces the model to learn meaningful visual features without labels!

---

## Quick Start

### Option 1: Automated (Recommended)

```bash
./scripts/run_phase2_pretrain.sh
```

### Option 2: Manual Execution

```bash
# Activate environment
source venv/bin/activate

# Run pre-training
python scripts/pretrain_mae.py \
    --config configs/ssl_pretrain_mae.yaml \
    --model-size small \
    --epochs 100

# Monitor training (in separate terminal)
tensorboard --logdir outputs/mae_pretrain/logs/
```

---

## Files Created in Phase 2

### New Scripts
- `scripts/pretrain_mae.py` - MAE training script (570 lines)
- `scripts/run_phase2_pretrain.sh` - Automated runner (executable)

### New Source Code
- `src/ssl_dataset.py` - Dataset loader for Phase 1 HDF5 files (312 lines)
- `src/mae_model.py` - MAE architecture implementation (497 lines)

### Configuration
- `configs/ssl_pretrain_mae.yaml` - Pre-training configuration

### Documentation
- `PHASE2_PRETRAIN_GUIDE.md` - This file

---

## Expected Output

After successful training, you'll have:

```
outputs/mae_pretrain/
├── mae_encoder_pretrained.pth      # 🎯 Encoder weights for Phase 3
├── checkpoints/
│   ├── best.pth                    # Best checkpoint (lowest val loss)
│   ├── latest.pth                  # Latest checkpoint
│   ├── epoch_020.pth               # Periodic checkpoints
│   ├── epoch_040.pth
│   └── ...
├── plots/
│   ├── reconstruction_epoch_001.png
│   ├── reconstruction_epoch_005.png
│   └── ...                         # Visual progress over training
└── logs/
    └── YYYYMMDD_HHMMSS/           # TensorBoard logs
```

### Key Output Files

| File | Purpose | Size |
|------|---------|------|
| `mae_encoder_pretrained.pth` | **Pre-trained encoder for Phase 3** | ~5-10 MB |
| `checkpoints/best.pth` | Full model checkpoint (best val loss) | ~10-20 MB |
| `plots/reconstruction_*.png` | Reconstruction visualizations | ~500 KB each |

---

## Training Parameters

### Model Sizes

**"small" (recommended for GTX 1070 Ti):**
- Parameters: ~1.5M
- Embedding dim: 192
- Encoder depth: 4 layers
- Batch size: 64-128
- Training time: ~12-24 hours

**"tiny" (faster, for testing):**
- Parameters: ~0.5M
- Embedding dim: 128
- Encoder depth: 3 layers
- Batch size: 128-256
- Training time: ~8-12 hours

### Configuration Options

Edit `configs/ssl_pretrain_mae.yaml`:

```yaml
# Training duration
epochs: 100               # Total epochs (50-200 typical)

# Model architecture
model_size: "small"       # "tiny" or "small"
mask_ratio: 0.75          # 75% masking (0.5-0.9 typical)

# Optimization
batch_size: 64            # Reduce if OOM
learning_rate: 1.0e-3     # 1e-3 recommended for MAE
weight_decay: 0.05

# Scheduler
scheduler: "cosine"       # Cosine annealing

# Early stopping
early_stopping_patience: 20
```

---

## Hardware Requirements

### GPU (Recommended)

- **GTX 1070 Ti (8GB):** ✅ Works well with model_size="small", batch_size=64
- **RTX 20XX/30XX:** ✅ Can use larger batch sizes (128+)
- **CPU only:** ⚠️ Very slow, not recommended (100x slower)

### Memory & Storage

- **GPU VRAM:** 4-8 GB required
- **System RAM:** 8+ GB recommended
- **Disk space:** 5-10 GB for outputs

### Training Time

| Hardware | Model Size | Batch Size | 100 Epochs |
|----------|------------|------------|------------|
| GTX 1070 Ti | small | 64 | ~16-24 hours |
| RTX 3080 | small | 128 | ~8-12 hours |
| CPU | small | 32 | ~10-20 days |

---

## Monitoring Training

### TensorBoard (Real-Time)

```bash
# In separate terminal
tensorboard --logdir outputs/mae_pretrain/logs/

# Open browser to http://localhost:6006
```

**Metrics to watch:**
- `train/epoch_loss` - Should decrease steadily
- `val/loss` - Should decrease and stabilize
- `train/learning_rate` - Should decay (if using scheduler)
- `reconstructions` - Visual quality should improve

### Console Output

```
Epoch   1/100 | Train Loss: 0.045123 | Val Loss: 0.042876 | LR: 1.00e-03 | Time: 145.2s
Epoch   2/100 | Train Loss: 0.038456 | Val Loss: 0.039234 | LR: 9.98e-04 | Time: 143.8s
...
    💾 Saved best checkpoint (val_loss: 0.029123)
...
Epoch 100/100 | Train Loss: 0.015234 | Val Loss: 0.018765 | LR: 1.00e-06 | Time: 141.3s

================================================================================
TRAINING COMPLETE
================================================================================
Best validation loss: 0.015123

💾 Saved encoder weights: outputs/mae_pretrain/mae_encoder_pretrained.pth
```

### Reconstruction Visualizations

Check `outputs/mae_pretrain/plots/` for visual progress:
- Early epochs: Poor reconstructions
- Mid epochs: Structure emerges
- Late epochs: High-quality reconstructions

Good training = reconstructions look like originals!

---

## Success Criteria

Phase 2 is successful if:

✅ **Training converges:** Loss decreases over epochs  
✅ **Validation stable:** Val loss doesn't explode or oscillate wildly  
✅ **Reconstructions improve:** Visual quality increases over training  
✅ **Encoder saved:** `mae_encoder_pretrained.pth` exists  
✅ **No crashes:** Training completes without OOM or errors  

**Target metrics:**
- Final train loss: 0.01-0.03 (lower is better)
- Final val loss: 0.015-0.04 (should be close to train loss)
- Best val loss typically at epoch 60-90

---

## Troubleshooting

### Out of Memory (OOM)

**Error:** `RuntimeError: CUDA out of memory`

**Solutions:**
1. Reduce `batch_size` in config (64 → 32 → 16)
2. Use `model_size: "tiny"` instead of "small"
3. Reduce `num_workers` (4 → 2 → 0)
4. Close other applications using GPU

### Training Loss Not Decreasing

**Symptoms:** Loss stays constant or increases

**Solutions:**
1. Check data normalization (images should be in [0, 1])
2. Reduce learning rate (1e-3 → 5e-4 → 1e-4)
3. Verify data loaded correctly (check one batch manually)
4. Try different mask ratio (0.75 → 0.5 or 0.9)

### Val Loss Much Higher Than Train Loss

**Symptoms:** Large train/val gap (overfitting)

**Solutions:**
1. Enable data augmentation (`augment: true`)
2. Increase weight decay (0.05 → 0.1)
3. Use lower mask ratio (0.75 → 0.5)
4. Stop training earlier (use early stopping)

### Training Too Slow

**Symptoms:** <1 it/s, estimated time >5 days

**Solutions:**
1. Verify GPU is being used (`nvidia-smi` in terminal)
2. Increase `num_workers` (4 → 8)
3. Use `model_size: "tiny"`
4. Reduce epochs (100 → 50)

### Reconstructions Don't Improve

**Symptoms:** Reconstructions stay blurry after many epochs

**Solutions:**
1. Check mask ratio (should be 0.5-0.9)
2. Verify targets are correct (check visualization)
3. Increase model capacity (tiny → small)
4. Train longer (100 → 150 epochs)

---

## Advanced Usage

### Resume Training

```bash
python scripts/pretrain_mae.py \
    --config configs/ssl_pretrain_mae.yaml \
    --resume outputs/mae_pretrain/checkpoints/latest.pth
```

### Custom Architecture

Edit `configs/ssl_pretrain_mae.yaml`:

```yaml
model_size: "custom"
patch_size: 20       # Larger = fewer patches = faster
embed_dim: 256       # Larger = more capacity
depth: 6             # Deeper = more capacity
num_heads: 4         # More heads = better attention
mask_ratio: 0.85     # Higher = harder task
```

### Hyperparameter Sweep

```bash
# Try different mask ratios
for mask_ratio in 0.5 0.75 0.9; do
    python scripts/pretrain_mae.py \
        --config configs/ssl_pretrain_mae.yaml \
        --epochs 50 \
        # Add custom mask_ratio override if needed
done
```

---

## Understanding the Output

### Encoder Weights

The most important output: `mae_encoder_pretrained.pth`

**What it contains:**
- Patch embedding weights
- Transformer encoder weights (4 layers)
- Positional embeddings
- Layer norms

**What it does NOT contain:**
- Decoder (not needed for downstream tasks)
- Classification/regression head (added in Phase 3)

**How to use in Phase 3:**
```python
from src.mae_model import build_mae_small

# Load pre-trained encoder
model = build_mae_small()
model.encoder.load_state_dict(torch.load('mae_encoder_pretrained.pth'))

# Add regression head for CBH
regression_head = nn.Linear(192, 1)
```

### Checkpoints vs Encoder

**Full checkpoint (`best.pth`):**
- Entire MAE model (encoder + decoder)
- Optimizer state
- Scheduler state
- Training epoch
- Config

**Encoder only (`mae_encoder_pretrained.pth`):**
- Just encoder weights
- Much smaller file
- Ready for fine-tuning

---

## Next Steps

After successful Phase 2:

1. ✅ **Verify encoder saved:** Check for `mae_encoder_pretrained.pth`
2. ✅ **Review training curves:** Open TensorBoard, check convergence
3. ✅ **Inspect reconstructions:** Check visual quality in `plots/`
4. ➡️ **Proceed to Phase 3:** Fine-tune encoder for CBH prediction

### Phase 3 Preview

```bash
# Next: Fine-tune pre-trained encoder on labeled CBH data
./scripts/run_phase3_finetune.sh
```

Phase 3 will:
- Load pre-trained encoder from Phase 2
- Add regression head for CBH prediction
- Fine-tune on ~933 CPL-labeled samples
- Compare to classical baseline (R² = 0.7464)

---

## Technical Details

### MAE Architecture

**Encoder (what we keep for Phase 3):**
```
PatchEmbed (Conv1d, patch_size=16)
  ↓
Positional Embedding
  ↓
Transformer Blocks × 4
  ↓
LayerNorm
  ↓
CLS Token Output (192-dim)
```

**Decoder (discarded after pre-training):**
```
Linear Projection (192 → 96)
  ↓
Add Mask Tokens
  ↓
Positional Embedding
  ↓
Transformer Blocks × 2
  ↓
Prediction Head (96 → patch_size pixels)
```

### Loss Function

MAE uses **Mean Squared Error (MSE)** on masked patches only:

```python
loss = ((pred - target) ** 2 * mask).sum() / mask.sum()
```

Where:
- `pred`: Reconstructed pixel values
- `target`: True pixel values (from original image)
- `mask`: Binary mask (1 = masked, 0 = visible)

**Why only masked patches?**
- More efficient (don't waste compute on easy visible patches)
- Forces model to learn from context (can't just copy input)
- Follows original MAE paper methodology

### Data Augmentation

For SSL, we use conservative augmentations:
- Horizontal flip (50%)
- Vertical flip (20%)
- Random crop & resize (80-100% scale)
- Brightness/contrast jitter (±30%)

**Why conservative?**
- Cloud structure is meaningful (can't distort too much)
- SSL benefits from multiple views of same data
- Heavy augmentation may hurt reconstruction quality

---

## FAQ

**Q: How long should I train?**  
A: 100 epochs is a good default. Early stopping (patience=20) will stop if no improvement. Typical best epoch: 60-90.

**Q: What if I don't have a GPU?**  
A: CPU training is possible but very slow (10-20 days). Consider using Google Colab (free GPU) or cloud services.

**Q: Can I use the decoder for anything?**  
A: The decoder is MAE-specific and not useful for downstream tasks. We only keep the encoder.

**Q: What if val loss is higher than train loss?**  
A: Small gap (<0.01) is normal. Large gap suggests overfitting—enable augmentation or reduce model size.

**Q: Should I use "tiny" or "small"?**  
A: "small" for best performance, "tiny" for faster testing. Both work fine for Phase 3.

**Q: How do I know if training is working?**  
A: Loss should decrease, reconstructions should improve visually. If stuck after 20 epochs, something is wrong.

**Q: Can I modify the architecture?**  
A: Yes, but keep it reasonable. Increasing depth/width helps capacity but slows training and uses more memory.

**Q: What's a good validation loss?**  
A: 0.015-0.04 is typical for well-trained MAE on normalized cloud images. Lower is better.

---

## Performance Benchmarks

### Training Speed (GTX 1070 Ti)

| Model | Batch Size | Time/Epoch | 100 Epochs |
|-------|------------|------------|------------|
| tiny | 128 | ~5 min | ~8 hours |
| small | 64 | ~10 min | ~17 hours |
| small | 128 | ~7 min | ~12 hours |

### Memory Usage

| Model | Batch Size | GPU Memory |
|-------|------------|------------|
| tiny | 128 | ~3 GB |
| small | 64 | ~5 GB |
| small | 128 | ~7 GB |

---

## Summary

Phase 2 Status: **✅ IMPLEMENTED - READY TO RUN**

**Created:**
- ✅ `scripts/pretrain_mae.py` (570 lines)
- ✅ `scripts/run_phase2_pretrain.sh` (90 lines, executable)
- ✅ `src/ssl_dataset.py` (312 lines)
- ✅ `src/mae_model.py` (497 lines)
- ✅ `configs/ssl_pretrain_mae.yaml` (83 lines)
- ✅ `PHASE2_PRETRAIN_GUIDE.md` (this file)

**Next Action:**
```bash
./scripts/run_phase2_pretrain.sh
```

**Expected Result:** Pre-trained encoder saved to `outputs/mae_pretrain/mae_encoder_pretrained.pth`

**Time Estimate:** 12-24 hours (GTX 1070 Ti, model_size="small")

---

*Phase 2 complete! Ready for execution.* 🚀