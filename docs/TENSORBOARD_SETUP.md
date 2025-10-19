# TensorBoard Setup and Troubleshooting Guide

**Date:** January 19, 2025  
**Purpose:** Fix "No dashboards are active" errors and set up TensorBoard correctly  
**Status:** ✅ All paths fixed and verification cell added

---

## Quick Start

### Step 1: Verify Logs Exist

Run the verification cell in the notebook (right before the TensorBoard cell):

```python
# This will show you if TensorBoard logs were created
tb_dir = "/content/drive/MyDrive/CloudML/logs/tensorboard/"
runs = os.listdir(tb_dir)
print(f"Found {len(runs)} run(s): {runs}")
```

**Expected output:**
```
✓ Found 2 TensorBoard run(s):
  - pretrain_30Oct24_baseline_tuned_20251019_012345: 1 event file(s)
  - final_overweighted_baseline_tuned_20251019_012345: 1 event file(s)

✓ Ready to launch TensorBoard!
```

### Step 2: Launch TensorBoard

```python
%load_ext tensorboard
%tensorboard --logdir /content/drive/MyDrive/CloudML/logs/tensorboard/
```

You should see training curves appear in the notebook!

---

## Problem: "No dashboards are active for the current data set"

### What This Means

TensorBoard can't find any event files (`.tfevents`) in the specified directory.

### Common Causes & Fixes

#### ❌ Cause 1: Files Not Saving to Google Drive

**Before the fix:**
- Files saved to `/content/repo/logs/` (local Colab storage - ephemeral)
- Lost when runtime disconnects

**After the fix (commit 3066a57):**
- Files save to `/content/drive/MyDrive/CloudML/logs/tensorboard/`
- Persistent across runtime disconnects

**How to verify:**
```python
!ls -lh /content/drive/MyDrive/CloudML/logs/tensorboard/
```

**Expected:** Should see directories like `pretrain_30Oct24_baseline_full_20251019_003612/`

**If empty:** Config paths not set correctly - check your config file has:
```yaml
log_dir: "/content/drive/MyDrive/CloudML/logs/"
```

---

#### ❌ Cause 2: Wrong TensorBoard Path

**Wrong:**
```python
%tensorboard --logdir /content/drive/MyDrive/CloudML/logs/  # Missing /tensorboard/
```

**Correct:**
```python
%tensorboard --logdir /content/drive/MyDrive/CloudML/logs/tensorboard/  # ✓
```

**Directory structure:**
```
/content/drive/MyDrive/CloudML/logs/
├── csv/                              # CSV metrics files
│   ├── metrics_final_*.json
│   └── predictions_final_*.csv
└── tensorboard/                      # TensorBoard event files ← Point here!
    ├── pretrain_30Oct24_*/
    │   └── events.out.tfevents.*
    └── final_overweighted_*/
        └── events.out.tfevents.*
```

---

#### ❌ Cause 3: No Training Completed Yet

TensorBoard needs at least one training epoch to have completed.

**Solution:** Wait for training to start, then run the TensorBoard cell.

**Minimum requirements:**
- At least 1 epoch of training completed
- Training loop has called `writer.add_scalar()` at least once
- Event file has been flushed to disk

---

#### ❌ Cause 4: Google Drive Not Mounted

**Check if mounted:**
```python
import os
if os.path.exists('/content/drive/MyDrive'):
    print("✓ Google Drive is mounted")
else:
    print("✗ Google Drive is NOT mounted")
```

**Fix:**
```python
from google.colab import drive
drive.mount('/content/drive')
```

---

## File Path History

### Before Fixes (Issues)

| Component | Old Path | Problem |
|-----------|----------|---------|
| TensorBoard logs | `/content/repo/logs/tensorboard/` | Lost on disconnect |
| CSV logs | `/content/repo/logs/csv/` | Hardcoded, not in Drive |
| Models | `/content/repo/models/trained/` | Hardcoded, not in Drive |
| Plots | `/content/repo/plots/` | Hardcoded, not in Drive |

### After Fixes (Commit 3066a57 + d226329)

| Component | New Path | Status |
|-----------|----------|--------|
| TensorBoard logs | `/content/drive/MyDrive/CloudML/logs/tensorboard/` | ✅ Persistent |
| CSV logs | `/content/drive/MyDrive/CloudML/logs/csv/` | ✅ Persistent |
| Models | `/content/drive/MyDrive/CloudML/models/trained/` | ✅ Persistent |
| Plots | `/content/drive/MyDrive/CloudML/plots/` | ✅ Persistent |

---

## How TensorBoard Logging Works

### Code Flow

1. **Config specifies base log directory:**
   ```yaml
   log_dir: "/content/drive/MyDrive/CloudML/logs/"
   ```

2. **Pipeline appends `/tensorboard/` subdirectory:**
   ```python
   # In src/pipeline.py
   pretrain_log_dir = os.path.join(log_dir, "tensorboard")
   # Result: /content/drive/MyDrive/CloudML/logs/tensorboard/
   ```

3. **train_model creates SummaryWriter:**
   ```python
   # In src/train_model.py
   writer = SummaryWriter(os.path.join(log_dir, config["save_name"]))
   # Result: /content/drive/MyDrive/CloudML/logs/tensorboard/pretrain_30Oct24_baseline_full_20251019_003612/
   ```

4. **Writer logs metrics each epoch:**
   ```python
   writer.add_scalar("Loss/train", avg_train_loss, epoch)
   writer.add_scalar("Loss/val", avg_val_loss, epoch)
   writer.add_scalar("Learning_rate", current_lr, epoch)
   ```

5. **Event files written to disk:**
   ```
   /content/drive/MyDrive/CloudML/logs/tensorboard/pretrain_30Oct24_baseline_full_20251019_003612/
   └── events.out.tfevents.1760830182.abcd1234.12345.0
   ```

---

## Verification Checklist

Run these commands to verify everything is set up correctly:

### 1. Check Config Paths
```python
import yaml
with open('/content/repo/configs/colab_optimized_full_tuned.yaml') as f:
    config = yaml.safe_load(f)
    print(f"log_dir: {config['log_dir']}")
    print(f"models_directory: {config['models_directory']}")
    print(f"output_directory: {config['output_directory']}")
```

**Expected:**
```
log_dir: /content/drive/MyDrive/CloudML/logs/
models_directory: /content/drive/MyDrive/CloudML/models/trained/
output_directory: /content/drive/MyDrive/CloudML/plots/
```

### 2. Check Google Drive Structure
```python
!tree -L 3 /content/drive/MyDrive/CloudML/ 2>/dev/null || \
find /content/drive/MyDrive/CloudML/ -maxdepth 3 -type d | sort
```

**Expected:**
```
/content/drive/MyDrive/CloudML/
├── data/
├── logs/
│   ├── csv/
│   └── tensorboard/
├── models/
│   └── trained/
└── plots/
```

### 3. Check TensorBoard Logs
```python
tb_dir = "/content/drive/MyDrive/CloudML/logs/tensorboard/"
if os.path.exists(tb_dir):
    for run in os.listdir(tb_dir):
        run_path = os.path.join(tb_dir, run)
        files = os.listdir(run_path)
        event_files = [f for f in files if 'events.out.tfevents' in f]
        print(f"Run: {run}")
        print(f"  Event files: {len(event_files)}")
        if event_files:
            print(f"  Latest: {event_files[-1][:50]}...")
```

### 4. Launch TensorBoard
```python
%load_ext tensorboard
%tensorboard --logdir /content/drive/MyDrive/CloudML/logs/tensorboard/
```

**Success:** You see graphs of training/validation loss, learning rate, etc.

**Failure:** "No dashboards are active" → Go back to troubleshooting section

---

## What You Should See in TensorBoard

### Scalars Tab

**Loss metrics:**
- `Loss/train` - Training loss per epoch
- `Loss/val` - Validation loss per epoch

**Learning rate:**
- `Learning_rate` - LR schedule over epochs

**Example curves:**
```
Training Loss: Steadily decreasing from ~4.0 → ~3.0
Validation Loss: Decreasing from ~1.3 → ~0.9 (should be stable, not erratic)
Learning Rate: Warmup from 0.000001 → 0.0005 over 500 steps, then stable
```

### Multiple Runs

If you've run multiple experiments, you'll see them overlaid:
- `pretrain_30Oct24_baseline_full_*` - First run (original config)
- `pretrain_30Oct24_baseline_tuned_*` - Second run (tuned config)

**Useful for comparison:**
- Which run converged faster?
- Which had more stable validation loss?
- Which achieved lower final loss?

---

## Advanced Usage

### Compare Specific Runs

```python
# Point to specific experiment directories
%tensorboard --logdir_spec \
  run1:/content/drive/MyDrive/CloudML/logs/tensorboard/pretrain_30Oct24_baseline_full_20251018_232936,\
  run2:/content/drive/MyDrive/CloudML/logs/tensorboard/pretrain_30Oct24_baseline_tuned_20251019_012345
```

### Filter by Regex

```python
# Only show runs matching pattern
%tensorboard --logdir /content/drive/MyDrive/CloudML/logs/tensorboard/ \
  --path_prefix pretrain
```

### Refresh TensorBoard

If training is ongoing and you want to see updated curves:

```python
# TensorBoard auto-refreshes every 30 seconds
# Or manually click the refresh icon in the TensorBoard UI
```

---

## Troubleshooting Common Issues

### Issue: "Port already in use"

**Error:**
```
ERROR: TensorBoard could not bind to port 6006, it was already in use
```

**Solution:**
```python
# Kill existing TensorBoard instances
!pkill -f tensorboard

# Or use a different port
%tensorboard --logdir /content/drive/MyDrive/CloudML/logs/tensorboard/ --port 6007
```

---

### Issue: "Permission denied"

**Error:**
```
PermissionError: [Errno 13] Permission denied: '/content/drive/MyDrive/CloudML/logs/tensorboard/'
```

**Solution:**
```python
# Remount Google Drive
from google.colab import drive
drive.flush_and_unmount()
drive.mount('/content/drive')
```

---

### Issue: Curves stop updating

**Problem:** TensorBoard shows data but stops updating during training

**Cause:** Event file buffering

**Solution:**
- Wait 30 seconds (auto-refresh interval)
- Click refresh icon in TensorBoard UI
- Check that training is still running (didn't crash)

---

### Issue: Multiple TensorBoard instances

**Problem:** Launched TensorBoard multiple times, seeing duplicates

**Solution:**
```python
# Kill all TensorBoard processes
!pkill -f tensorboard

# Launch fresh instance
%load_ext tensorboard
%tensorboard --logdir /content/drive/MyDrive/CloudML/logs/tensorboard/
```

---

## Expected TensorBoard Output

### Good Training Run
```
✓ Smooth curves (not jagged)
✓ Training loss decreasing steadily
✓ Validation loss decreasing (may plateau)
✓ Small gap between train and val loss
✓ Learning rate follows expected schedule
```

### Poor Training Run (Red Flags)
```
✗ Validation loss oscillating wildly
✗ Large gap between train and val (overfitting)
✗ Loss not decreasing (learning rate too low)
✗ Loss exploding (learning rate too high)
✗ NaN values (numerical instability)
```

---

## Summary

### File Saving Fixes (Commits 3066a57, d226329)

| Fix | Description |
|-----|-------------|
| ✅ Config paths | Added `log_dir`, `models_directory`, `output_directory` to all configs |
| ✅ Pipeline code | Updated `src/pipeline.py` to use config paths instead of hardcoded |
| ✅ CSV logging | Fixed `src/train_model.py` to use config log_dir |
| ✅ Notebook paths | Updated all print statements and TensorBoard cell |
| ✅ Verification cell | Added cell to check if logs exist before launching TensorBoard |

### TensorBoard Usage

```python
# 1. Run verification cell (check logs exist)
# 2. Launch TensorBoard
%tensorboard --logdir /content/drive/MyDrive/CloudML/logs/tensorboard/

# 3. View training curves in notebook
# 4. Compare multiple runs
# 5. Export data if needed
```

### When to Use TensorBoard

- ✅ **During training:** Monitor progress in real-time
- ✅ **After training:** Compare runs and hyperparameters
- ✅ **Debugging:** Identify learning rate issues, overfitting, etc.
- ✅ **Paper figures:** Export high-quality plots

---

**Status:** ✅ TensorBoard fully functional with Google Drive persistence  
**Next:** Run tuned config and monitor curves in TensorBoard! 📊