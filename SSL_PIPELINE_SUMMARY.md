# SSL Pipeline for Cloud Base Height (CBH) Estimation - Summary

**Last Updated:** November 1, 2024  
**Pipeline Status:** ✅ Complete and validated  
**Test Performance:** R² = 0.37, MAE = 0.22 km, RMSE = 0.30 km

---

## Overview

This document summarizes the complete Self-Supervised Learning (SSL) pipeline for Cloud Base Height estimation using Masked Autoencoder (MAE) pre-training followed by supervised fine-tuning.

### Pipeline Architecture

```
Phase 1: Data Extraction
    └─> Extract unlabeled images from HDF5 files
    └─> 61,946 total images (58,846 train, 3,100 val)

Phase 2: MAE Pre-training (Self-Supervised)
    └─> Train encoder on unlabeled data
    └─> 100 epochs, best val loss: 0.0093
    └─> Saved: outputs/mae_pretrain/mae_encoder_pretrained.pth

Phase 3: Fine-tuning (Supervised)
    └─> Two-stage fine-tuning on CPL-labeled data (933 samples)
    └─> Stage 1: Freeze encoder, train head only
    └─> Stage 2: Unfreeze encoder, fine-tune end-to-end
    └─> Final test R²: 0.37
```

---

## Quick Start

### Prerequisites
```bash
# Activate virtual environment
source .venv/bin/activate

# Verify GPU
nvidia-smi
```

### Run Complete Pipeline

```bash
# Phase 1: Extract images for SSL pre-training
./scripts/run_phase1.sh

# Phase 2: MAE self-supervised pre-training
./scripts/run_phase2_pretrain.sh

# Phase 3: Fine-tune for CBH regression
./scripts/run_phase3_finetune.sh
```

### Monitor Training

```bash
# View TensorBoard logs
tensorboard --logdir outputs/cbh_finetune/logs/

# View results plot
xdg-open outputs/cbh_finetune/plots/test_results.png
```

---

## Configuration Files

### Phase 1: Data Extraction
- **Config:** `configs/ssl_extract.yaml`
- **Key settings:**
  - Temporal frames: 1
  - Swath slice: [40, 480] → 440 pixels
  - Train/val split: 95/5

### Phase 2: MAE Pre-training
- **Config:** `configs/ssl_pretrain_mae.yaml`
- **Model:** MAE "small"
  - embed_dim: 192
  - depth: 4 transformer blocks
  - num_heads: 3
  - patch_size: 16
- **Training:**
  - Epochs: 100
  - Batch size: 256
  - Mask ratio: 0.75
  - Optimizer: AdamW (lr=1e-3, wd=0.05)

### Phase 3: Fine-tuning
- **Config:** `configs/ssl_finetune_cbh.yaml`
- **Dataset:** 933 CPL-labeled samples (5 flights)
  - Train: 653 samples (70%)
  - Val: 139 samples (15%)
  - Test: 141 samples (15%)
- **Two-stage training:**
  
  **Stage 1 - Freeze Encoder:**
  - Epochs: 30
  - Batch size: 512
  - LR: 1e-3
  - Trainable params: 265,985 (head only)
  - Result: Typically fails (negative R²)
  
  **Stage 2 - Fine-tune All:**
  - Epochs: 50
  - Batch size: 384
  - LR: 1e-4
  - Trainable params: 2,054,465 (full model)
  - Result: R² peaks ~0.32, test R² = 0.37

---

## Model Architecture

### Encoder (MAE - Pre-trained)
```
Input: (batch, 1, 440) pixels
  ↓
PatchEmbed: 440 → 27 patches of size 16
  ↓
ViT Encoder: 4 transformer blocks (embed_dim=192, 3 heads)
  ↓
Output: (batch, 28, 192)  # 27 patches + 1 CLS token
```

### Regression Head (Fine-tuned)
```
CLS token: (batch, 192)
  ↓ (concat angles if enabled)
Input: (batch, 194)  # 192 + 2 angles (SZA, SAA)
  ↓
MLP: [512, 256, 128] with BatchNorm, GELU, Dropout(0.3)
  ↓
Output: (batch, 1)  # CBH prediction in km
```

**Total Parameters:** 2,054,465
- Encoder: 1,788,480 (pre-trained, frozen in Stage 1)
- Head: 265,985 (trained from scratch)

---

## Performance Results

### Current Performance (Nov 1, 2024)

| Metric | SSL Model | Classical Baseline | Ratio |
|--------|-----------|-------------------|-------|
| **Test R²** | 0.3665 | 0.7464 | 0.49x |
| **Test MAE** | 0.2211 km | 0.1265 km | 1.75x |
| **Test RMSE** | 0.2993 km | 0.1929 km | 1.55x |

**Classical Baseline:** GradientBoosting on hand-crafted features

### Key Findings

✅ **SSL pipeline works:** Model learns and generalizes (R² > 0)

✅ **Better than supervised-from-scratch:** Previous supervised attempts failed completely

⚠️ **Underperforms classical ML:** GradientBoosting achieves 2x better R²

🔍 **Stage 1 consistently fails:** Frozen encoder doesn't provide good features alone

✅ **Stage 2 recovers:** Unfreezing encoder enables learning → R² 0.32-0.37

### Training Behavior Pattern

**Observed across all runs:**
1. Stage 1 (frozen encoder): Val R² stays negative, no learning
2. Stage 2 (unfrozen): Val R² climbs from negative → 0.32 peak
3. Test performance: R² = 0.36-0.48 range

**Interpretation:** Pre-trained encoder features alone are insufficient; encoder must adapt during fine-tuning for this specific task.

---

## Hardware Requirements

### Tested Configuration
- **GPU:** NVIDIA GeForce GTX 1070 Ti (8GB VRAM)
- **Memory usage:**
  - Phase 2 pre-training: ~3-4 GB
  - Phase 3 fine-tuning: ~1.8-2 GB (small labeled dataset)
- **Training time:**
  - Phase 2: ~2-3 hours (100 epochs)
  - Phase 3: ~20-30 minutes (30 + 50 epochs)

### Notes
- Low memory usage in Phase 3 is expected (only 653 training samples)
- Batch size 512 processes almost entire dataset in one batch
- AMP (mixed precision) enabled for efficiency

---

## File Structure

```
cloudMLPublic/
├── configs/
│   ├── ssl_extract.yaml          # Phase 1: Data extraction
│   ├── ssl_pretrain_mae.yaml     # Phase 2: MAE pre-training
│   └── ssl_finetune_cbh.yaml     # Phase 3: Fine-tuning
├── scripts/
│   ├── extract_all_images.py     # Phase 1 extraction script
│   ├── verify_extraction.py      # Phase 1 verification
│   ├── pretrain_mae.py           # Phase 2 pre-training script
│   ├── finetune_cbh.py          # Phase 3 fine-tuning script
│   ├── run_phase1.sh            # Phase 1 runner
│   ├── run_phase2_pretrain.sh   # Phase 2 runner
│   └── run_phase3_finetune.sh   # Phase 3 runner
├── src/
│   ├── mae_model.py             # MAE architecture
│   ├── ssl_dataset.py           # HDF5 dataset loaders
│   └── pytorchmodel.py          # Base models and losses
├── data_ssl/                     # Phase 1 outputs (gitignored)
│   └── images/
│       ├── train.h5             # 58,846 unlabeled images
│       ├── val.h5               # 3,100 unlabeled images
│       └── extraction_stats.yaml
└── outputs/                      # Training outputs (gitignored)
    ├── mae_pretrain/
    │   ├── mae_encoder_pretrained.pth  # Pre-trained encoder
    │   └── logs/                       # TensorBoard logs
    └── cbh_finetune/
        ├── checkpoints/
        │   ├── final_model.pth
        │   ├── stage1_freeze_best.pth
        │   └── stage2_finetune_best.pth
        ├── plots/
        │   └── test_results.png
        └── logs/                       # TensorBoard logs
```

---

## Next Steps / Future Work

### Performance Improvement Options

1. **Longer Pre-training**
   - Increase MAE epochs from 100 → 200+
   - May improve encoder quality

2. **Larger Model**
   - Try MAE "base" (embed_dim=384, depth=8)
   - Requires re-running Phase 2

3. **Temporal Modeling**
   - Use 3-frame sequences instead of single frames
   - Capture temporal dynamics

4. **Hybrid Approach**
   - Extract SSL embeddings (CLS tokens)
   - Train GradientBoosting on embeddings
   - Combine neural + classical strengths

5. **Data Augmentation**
   - Add augmentations during fine-tuning
   - Geometric, intensity, noise

### Research Questions

- Why does frozen encoder fail (Stage 1)?
- What features does the encoder learn during pre-training?
- Can embeddings improve classical models?
- How does performance vary per flight/condition?

### Analysis Tasks

- [ ] t-SNE/UMAP visualization of embeddings
- [ ] Per-flight performance breakdown
- [ ] Residual analysis (where does model fail?)
- [ ] Feature importance from encoder attention

---

## Troubleshooting

### Common Issues

**1. Negative R² in early epochs**
- Expected behavior during Stage 1
- Model should recover in Stage 2
- If persistent, reduce learning rate

**2. OOM (Out of Memory)**
- Reduce batch size in config
- Enable AMP: `use_amp: true`
- Use smaller model size

**3. Pre-trained weights not found**
- Run Phase 2 first
- Check path: `outputs/mae_pretrain/mae_encoder_pretrained.pth`

**4. Config path errors**
- Ensure data paths match your system
- Update `data.flights[].iFileName` paths in config

### Debugging

```bash
# Check GPU
nvidia-smi

# Verify data extraction
python scripts/verify_extraction.py

# Test single batch
# Edit config: debug.enabled: true, debug.max_samples: 100
./scripts/run_phase3_finetune.sh
```

---

## References

### Papers
- He et al., "Masked Autoencoders Are Scalable Vision Learners" (2021)
- Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" (2020)

### Documentation
- Phase 1: `PHASE1_EXTRACTION_GUIDE.md`
- Phase 2: `PHASE2_PRETRAIN_GUIDE.md`
- Phase 3: `PHASE3_FINETUNE_GUIDE.md`
- Full status: `PROJECT_STATUS.md`

---

## Changelog

### Nov 1, 2024
- Validated complete pipeline end-to-end
- Achieved test R² = 0.3665 (consistent with previous runs)
- Fixed autocast deprecation warning
- Cleaned up TensorBoard logs (kept latest run only)
- Documented stable performance baseline

### Oct 31, 2024
- Completed Phase 3 implementation
- Two-stage fine-tuning strategy
- Initial results: R² = 0.48

### Oct 30, 2024
- Completed Phase 2 MAE pre-training
- 100 epochs, val loss = 0.0093

### Oct 29, 2024
- Completed Phase 1 data extraction
- 61,946 images extracted and verified