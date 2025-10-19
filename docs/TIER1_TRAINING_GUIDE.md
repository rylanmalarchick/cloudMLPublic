# TIER 1 Training Guide - CloudML Model Improvements

**Date**: 2025-01-28  
**Status**: Ready to Run  
**Expected Improvement**: +15-25% R² (from -0.09 to 0.15-0.25)

---

## Overview

This guide walks you through running the **TIER 1 improvements** for the CloudML model in Google Colab. These improvements are based on validated literature insights and include:

1. ✅ **Multi-scale temporal attention** - Captures cross-view relationships at different scales
2. ✅ **Self-supervised pre-training** - Encoder learns spatial features before supervised training  
3. ✅ **Increased temporal frames** - 5→7 frames for better spatial coverage (if data permits)

---

## Pre-Flight Checklist

Before starting, ensure:

- [ ] Google Drive mounted at `/content/drive/MyDrive/`
- [ ] CloudML data in `/content/drive/MyDrive/CloudML/data/`
- [ ] GPU enabled (Runtime → Change runtime type → T4 GPU)
- [ ] Latest code pulled from repo
- [ ] Config file updated: `configs/colab_optimized_full_tuned.yaml`

---

## Step 1: Mount Drive & Setup Environment

```python
# Cell 1: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Cell 2: Clone/Update Repository
import os
os.chdir('/content')

# If first time:
# !git clone https://github.com/YOUR_USERNAME/cloudMLPublic.git repo

# If already cloned:
os.chdir('/content/repo')
!git pull origin main

# Cell 3: Install Dependencies
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install -q tensorboard tqdm pyyaml h5py matplotlib scikit-learn scipy

# Verify GPU
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
```

---

## Step 2: Verify Data & Config

```python
# Cell 4: Verify Data Files Exist
import os

data_dir = "/content/drive/MyDrive/CloudML/data/"
required_files = [
    "10Feb25/GLOVE2025_IRAI_L1B_Rev-_20250210.h5",
    "30Oct24/WHYMSIE2024_IRAI_L1B_Rev-_20241030.h5",
    "04Nov24/WHYMSIE2024_IRAI_L1B_Rev-_20241104.h5",
    "23Oct24/WHYMSIE2024_IRAI_L1B_Rev-_20241023.h5",
    "18Feb25/GLOVE2025_IRAI_L1B_Rev-_20250218.h5",
    "12Feb25/GLOVE2025_IRAI_L1B_Rev-_20250212.h5",
]

print("Checking data files...")
for f in required_files:
    path = os.path.join(data_dir, f)
    exists = os.path.exists(path)
    status = "✓" if exists else "✗"
    print(f"{status} {f}")

# Cell 5: Load and Verify Config
import yaml

config_path = "/content/repo/configs/colab_optimized_full_tuned.yaml"

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

print("=" * 70)
print("TIER 1 CONFIGURATION")
print("=" * 70)
print(f"Temporal frames: {config['temporal_frames']}")
print(f"Use multi-scale temporal: {config.get('use_multiscale_temporal', False)}")
print(f"Attention heads: {config.get('attention_heads', 4)}")
print(f"Pre-training enabled: {config.get('pretraining', {}).get('enabled', False)}")
print(f"Pre-training epochs: {config.get('pretraining', {}).get('epochs', 0)}")
print(f"Learning rate: {config['learning_rate']}")
print(f"Warmup steps: {config['warmup_steps']}")
print(f"Overweight factor: {config['overweight_factor']}")
print("=" * 70)

# Verify TIER 1 settings are enabled
assert config['temporal_frames'] == 7, "ERROR: temporal_frames should be 7"
assert config.get('use_multiscale_temporal', False) == True, "ERROR: use_multiscale_temporal not enabled"
assert config.get('pretraining', {}).get('enabled', False) == True, "ERROR: pretraining not enabled"
print("✓ All TIER 1 settings verified!")
```

---

## Step 3: Create Output Directories

```python
# Cell 6: Create Output Directories
import os

dirs = [
    "/content/drive/MyDrive/CloudML/models/trained/",
    "/content/drive/MyDrive/CloudML/models/pretrained/",
    "/content/drive/MyDrive/CloudML/logs/csv/",
    "/content/drive/MyDrive/CloudML/logs/tensorboard/",
    "/content/drive/MyDrive/CloudML/plots/",
]

for d in dirs:
    os.makedirs(d, exist_ok=True)
    print(f"✓ {d}")
```

---

## Step 4: Run TIER 1 Training

```python
# Cell 7: Run Training with TIER 1 Improvements
import os
os.chdir('/content/repo')

# Run training
!python main.py \
    --config configs/colab_optimized_full_tuned.yaml \
    --save_name tier1_baseline \
    --epochs 50

# This will:
# 1. Load data from all 6 flights
# 2. Run self-supervised pre-training for 20 epochs (Phase 1)
# 3. Save pre-trained encoder checkpoint
# 4. Run supervised training for 50 epochs (Phase 2)
# 5. Apply multi-scale temporal attention
# 6. Use 7 temporal frames (increased from 5)
# 7. Save final model, logs, and plots to Drive
```

---

## Step 5: Monitor Training Progress

### Option A: Real-time Console Output

Watch the console output for:
- ✅ **Pre-training phase**: Reconstruction loss should decrease (target: < 0.01)
- ✅ **Supervised training**: Training loss decreasing smoothly
- ✅ **Validation metrics**: R² should improve over epochs
- ⚠️ **Memory usage**: Monitor for OOM errors

### Option B: TensorBoard (Recommended)

```python
# Cell 8: Launch TensorBoard
%load_ext tensorboard
%tensorboard --logdir /content/drive/MyDrive/CloudML/logs/tensorboard/

# OR view in separate tab:
# !tensorboard --logdir /content/drive/MyDrive/CloudML/logs/tensorboard/ --host 0.0.0.0 --port 6006
```

Watch for:
- **train_loss**: Should decrease smoothly
- **val_loss**: Should decrease and stabilize
- **val_r2**: Should increase (target: > 0.15)
- **val_mae**: Should decrease (target: < 0.30 km)

### Option C: Check Files

```python
# Cell 9: Verify Training Progress
import os
import glob

# Check logs
csv_logs = glob.glob("/content/drive/MyDrive/CloudML/logs/csv/*.csv")
tb_logs = glob.glob("/content/drive/MyDrive/CloudML/logs/tensorboard/*/events*")

print(f"CSV logs: {len(csv_logs)} files")
print(f"TensorBoard logs: {len(tb_logs)} files")

# Check models
pretrained = glob.glob("/content/drive/MyDrive/CloudML/models/pretrained/*.pth")
trained = glob.glob("/content/drive/MyDrive/CloudML/models/trained/*.pth")

print(f"Pre-trained checkpoints: {len(pretrained)}")
print(f"Trained models: {len(trained)}")

# Check plots
plots = glob.glob("/content/drive/MyDrive/CloudML/plots/*.png")
print(f"Plots generated: {len(plots)}")
```

---

## Step 6: Evaluate Results

```python
# Cell 10: Load and Analyze Results
import pandas as pd
import matplotlib.pyplot as plt

# Load CSV log
log_file = max(glob.glob("/content/drive/MyDrive/CloudML/logs/csv/*.csv"), 
               key=os.path.getctime)
df = pd.read_csv(log_file)

print("=" * 70)
print("TRAINING SUMMARY")
print("=" * 70)
print(f"Total epochs: {len(df)}")
print(f"Best val_loss: {df['val_loss'].min():.4f} at epoch {df['val_loss'].idxmin()}")
print(f"Final val_R²: {df['val_r2'].iloc[-1]:.4f}")
print(f"Final val_MAE: {df['val_mae'].iloc[-1]:.4f} km")
print(f"Final val_RMSE: {df['val_rmse'].iloc[-1]:.4f} km")
print("=" * 70)

# Plot training curves
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].plot(df['epoch'], df['train_loss'], label='Train')
axes[0, 0].plot(df['epoch'], df['val_loss'], label='Val')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Loss Curves')
axes[0, 0].legend()
axes[0, 0].grid(True)

axes[0, 1].plot(df['epoch'], df['val_r2'])
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('R²')
axes[0, 1].set_title('Validation R²')
axes[0, 1].grid(True)
axes[0, 1].axhline(y=0, color='r', linestyle='--', label='Baseline')
axes[0, 1].legend()

axes[1, 0].plot(df['epoch'], df['val_mae'])
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('MAE (km)')
axes[1, 0].set_title('Validation MAE')
axes[1, 0].grid(True)

axes[1, 1].plot(df['epoch'], df['val_rmse'])
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('RMSE (km)')
axes[1, 1].set_title('Validation RMSE')
axes[1, 1].grid(True)

plt.tight_layout()
plt.savefig('/content/drive/MyDrive/CloudML/plots/tier1_training_curves.png', dpi=150)
plt.show()
```

---

## Step 7: Compare to Baseline

```python
# Cell 11: Compare TIER 1 vs Baseline
baseline_results = {
    'R²': -0.0927,
    'MAE': 0.3415,
    'RMSE': 0.5046,
}

tier1_results = {
    'R²': df['val_r2'].iloc[-1],
    'MAE': df['val_mae'].iloc[-1],
    'RMSE': df['val_rmse'].iloc[-1],
}

print("=" * 70)
print("TIER 1 vs BASELINE COMPARISON")
print("=" * 70)
print(f"{'Metric':<10} {'Baseline':<12} {'TIER 1':<12} {'Improvement':<15}")
print("-" * 70)

for metric in ['R²', 'MAE', 'RMSE']:
    base = baseline_results[metric]
    tier1 = tier1_results[metric]
    
    if metric == 'R²':
        improvement = tier1 - base
        print(f"{metric:<10} {base:>11.4f} {tier1:>11.4f} {improvement:>+14.4f}")
    else:
        improvement = ((base - tier1) / base) * 100
        print(f"{metric:<10} {base:>11.4f} {tier1:>11.4f} {improvement:>13.1f}%")

print("=" * 70)

# Success criteria
print("\nSUCCESS CRITERIA:")
print(f"✓ R² > 0.15: {'PASS' if tier1_results['R²'] > 0.15 else 'FAIL'}")
print(f"✓ MAE < 0.30 km: {'PASS' if tier1_results['MAE'] < 0.30 else 'FAIL'}")
print(f"✓ RMSE < 0.45 km: {'PASS' if tier1_results['RMSE'] < 0.45 else 'FAIL'}")
```

---

## Step 8: View Final Predictions

```python
# Cell 12: View Prediction Scatter Plots
from PIL import Image

plot_files = sorted(glob.glob("/content/drive/MyDrive/CloudML/plots/*predictions*.png"))

if plot_files:
    latest_plot = plot_files[-1]
    print(f"Displaying: {os.path.basename(latest_plot)}")
    img = Image.open(latest_plot)
    plt.figure(figsize=(12, 8))
    plt.imshow(img)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
else:
    print("No prediction plots found")
```

---

## Troubleshooting

### Issue: Out of Memory (OOM)

**Solution 1**: Reduce batch size
```yaml
# In config:
batch_size: 16  # Reduce from 20
```

**Solution 2**: Use memory-optimized model
```yaml
# In config:
memory_optimized: true  # Reduces channels 64/128/256 → 32/64/128
```

**Solution 3**: Reduce temporal frames
```yaml
# In config:
temporal_frames: 5  # Reduce from 7
```

### Issue: temporal_frames=7 causes "Index out of range"

**Cause**: Not all samples have 7 frames available

**Solution**: Fallback to 5 frames
```yaml
temporal_frames: 5
```

### Issue: Pre-training takes too long

**Solution**: Reduce pre-training epochs
```yaml
pretraining:
  enabled: true
  epochs: 10  # Reduce from 20
```

### Issue: Validation loss not improving

**Check**:
1. Is training loss decreasing? (Yes → overfitting; No → learning rate issue)
2. Is pre-training reconstruction loss low? (Target: < 0.01)
3. Are attention mechanisms working? (Check TensorBoard histograms)

**Solutions**:
- Reduce learning rate: `learning_rate: 0.0003`
- Increase regularization: `weight_decay: 0.06`
- Add more dropout: edit `cnn_layers` dropout values

### Issue: R² still negative

**Possible causes**:
1. Data quality issues (check for NaN/inf in inputs)
2. Target scaling issues (check y_scaler)
3. Model capacity too small (try full model, not memory_optimized)
4. Need more pre-training epochs

**Debug**:
```python
# Check data distributions
import h5py
f = h5py.File('/content/drive/MyDrive/CloudML/data/30Oct24/WHYMSIE2024_IRAI_L1B_Rev-_20241030.h5', 'r')
print(f.keys())
print(f['IRAI'].shape)
print(f['IRAI'][0, :, 100, 100])  # Sample pixel values
```

---

## Expected Output Files

After successful training, you should have:

```
/content/drive/MyDrive/CloudML/
├── models/
│   ├── pretrained/
│   │   └── pretrained_encoder_best.pth
│   └── trained/
│       ├── tier1_baseline_30Oct24.pth
│       ├── tier1_baseline_unified_model.pth
│       └── tier1_baseline_final_model.pth
├── logs/
│   ├── csv/
│   │   └── training_log_tier1_baseline_YYYYMMDD_HHMMSS.csv
│   └── tensorboard/
│       └── tier1_baseline_YYYYMMDD_HHMMSS/
│           └── events.out.tfevents.*
└── plots/
    ├── tier1_baseline_predictions_30Oct24.png
    ├── tier1_baseline_predictions_unified.png
    ├── tier1_baseline_predictions_multi_flight.png
    └── tier1_training_curves.png
```

---

## Next Steps

### If R² > 0.15 (Success!) 🎉
1. Document results in `TIER1_RESULTS.md`
2. Proceed to **TIER 2** implementations:
   - Location-variant spatial attention
   - Three-phase training strategy
   - Additional spectral channels (if available)

### If R² < 0.15 (Needs Work) 🔧
1. Run ablation study to identify which component helps most:
   - Disable pre-training: `pretraining: enabled: false`
   - Disable multi-scale: `use_multiscale_temporal: false`
   - Use baseline attention: (revert to original config)
2. Analyze failure modes:
   - Check which flights perform worst
   - Visualize attention maps
   - Compare to LiDAR ground truth
3. Try alternative approaches:
   - Simpler model architecture
   - Different loss functions
   - Hand-crafted features

---

## Quick Reference: Config Options

```yaml
# Essential TIER 1 settings:
temporal_frames: 7
use_multiscale_temporal: true
attention_heads: 4

pretraining:
  enabled: true
  epochs: 20
  learning_rate: 0.0001

# Training hyperparameters:
learning_rate: 0.0005
warmup_steps: 500
overweight_factor: 2.0
early_stopping_patience: 10

# Performance settings:
batch_size: 20
gradient_checkpointing: true
torch_compile: true
torch_compile_mode: "default"
memory_optimized: false
```

---

## Estimated Runtime

| Phase | Time | GPU Memory |
|-------|------|------------|
| Pre-training (20 epochs) | ~30-45 min | ~8-10 GB |
| Supervised training (50 epochs) | ~2-3 hours | ~10-12 GB |
| Evaluation & plotting | ~5-10 min | ~2-4 GB |
| **Total** | **~3-4 hours** | **Peak: ~12 GB** |

---

## Support

If you encounter issues:
1. Check the console output for error messages
2. Verify all files are in Drive (not ephemeral `/content/`)
3. Check GPU memory: `!nvidia-smi`
4. Review TensorBoard for training anomalies
5. Compare config against `colab_optimized_full_tuned.yaml`

---

**Good luck with your TIER 1 training run!** 🚀

*Last updated: 2025-01-28*