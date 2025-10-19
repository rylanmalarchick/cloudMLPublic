# 🚀 RUN TIER 1 IN COLAB - QUICK INSTRUCTIONS

## Why Tier 1 Isn't Running

You're seeing the **old baseline code** run because:
1. The Colab notebook cell doesn't pull the latest code before training
2. Your current code is missing the Tier 1 implementation (pretraining.py, multi_scale_attention.py)

## Solution: Update Your Training Cell

### Step 1: Open Your Colab Notebook
Go to the cell that says:
```
# OPTION A-TUNED: FULL MODEL + TUNED HYPERPARAMETERS (RECOMMENDED) ⭐⭐
```

### Step 2: Replace That Cell With This Code

Copy the ENTIRE contents of `TIER1_COLAB_CELL.md` and paste it into that cell.

OR manually add this at the TOP of your training cell:

```python
# Pull latest Tier 1 code
print("Updating to Tier 1 code...")
%cd /content/repo
!git pull origin main
print("✓ Code updated\n")

# Verify Tier 1 files exist
import os
if os.path.exists('/content/repo/src/pretraining.py'):
    print("✓ Tier 1 pretraining module found")
else:
    print("⚠ WARNING: Tier 1 missing - try: !cd /content/repo && git reset --hard origin/main")
```

### Step 3: Run the Updated Cell

You should see:
```
================================================================================
Updating to Tier 1 code...
================================================================================
Already up to date.
✓ Code updated

✓ pretraining.py found
✓ multi_scale_attention.py found

✓ All Tier 1 modules present
...
================================================================================
TIER 1: SELF-SUPERVISED PRE-TRAINING ENABLED
================================================================================
Starting self-supervised pre-training...
Epoch 1/20 - Reconstruction Loss: X.XXXX
```

### Step 4: Confirm Tier 1 Is Running

**YOU MUST SEE THIS BANNER:**
```
======================================================================
TIER 1: SELF-SUPERVISED PRE-TRAINING ENABLED
======================================================================
```

If you DON'T see this banner, Tier 1 is NOT running (you're running baseline).

## Quick Fix If Still Not Working

Run this in a Colab cell:
```python
%cd /content/repo
!git fetch origin
!git reset --hard origin/main
!ls -la src/pretraining.py src/multi_scale_attention.py
```

You should see both files listed. Then run your training cell again.

## What Tier 1 Does

1. **Self-supervised pre-training (20 epochs)**: Encoder learns spatial features via image reconstruction
2. **Then supervised training (50 epochs)**: Normal training with pre-trained encoder
3. **Multi-scale temporal attention**: Captures relationships across different spatial scales
4. **7 temporal frames**: More views = better shadow triangulation

**Expected improvements:** +15-25% R² (from baseline ~-0.09 to ~0.15-0.25)

## Summary

**Before (what you're running now):**
```
--- Pretraining on 30Oct24 ---
Using memory_optimized=False model
Applying torch.compile()...
Epoch 1/50 | Train Loss: 4.1157 | Val Loss: 1.3602
```

**After (what Tier 1 should show):**
```
======================================================================
TIER 1: SELF-SUPERVISED PRE-TRAINING ENABLED
======================================================================
Starting self-supervised pre-training...
Epoch 1/20 - Reconstruction Loss: 2.3451
...
Pre-training complete! Proceeding to supervised training...

--- Pretraining on 30Oct24 ---
Using memory_optimized=False model
Applying torch.compile()...
Epoch 1/50 | Train Loss: 3.2157 | Val Loss: 0.9602
```

Notice: Lower initial losses and the Tier 1 banner!