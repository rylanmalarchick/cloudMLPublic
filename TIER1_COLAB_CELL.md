# ============================================================================
# TIER 1 TRAINING CELL - COPY THIS INTO YOUR COLAB NOTEBOOK
# ============================================================================
# Replace the "Option A-Tuned" cell with this code block
# This ensures you pull the latest Tier 1 implementation before training

# STEP 1: Pull latest code with Tier 1 implementation
print("=" * 80)
print("UPDATING TO TIER 1 CODE...")
print("=" * 80)
%cd /content/repo
!git pull origin main
print("✓ Code updated\n")

# STEP 2: Verify Tier 1 modules exist
import os
tier1_files = [
    '/content/repo/src/pretraining.py',
    '/content/repo/src/multi_scale_attention.py'
]

all_present = True
for f in tier1_files:
    if os.path.exists(f):
        print(f"✓ {os.path.basename(f)} found")
    else:
        print(f"✗ {os.path.basename(f)} MISSING")
        all_present = False

if not all_present:
    print("\n⚠ WARNING: Some Tier 1 files missing - you may be running baseline only")
    print("Try running: !cd /content/repo && git fetch && git reset --hard origin/main")
else:
    print("\n✓ All Tier 1 modules present")

# STEP 3: Start Tier 1 training
import datetime

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
experiment_name = f"tier1_tuned_{timestamp}"

print("\n" + "=" * 80)
print("TIER 1 TRAINING: FULL MODEL + TUNED HYPERPARAMETERS + LITERATURE IMPROVEMENTS")
print("=" * 80)
print(f"Experiment ID: {experiment_name}")
print(f"Config: colab_optimized_full_tuned.yaml")
print(f"Model: 64/128/256 channels (FULL)")
print(f"\nTIER 1 FEATURES ENABLED:")
print(f"  ✅ Self-supervised pre-training (20 epochs reconstruction)")
print(f"  ✅ Multi-scale temporal attention (4 heads)")
print(f"  ✅ Increased temporal frames (7 frames)")
print(f"  ✅ Expected R² improvement: +15-25%")
print(f"\nTuned Hyperparameters:")
print(f"  - Learning Rate: 0.0005 (reduced from 0.001)")
print(f"  - Warmup Steps: 500 (reduced from 2000)")
print(f"  - Overweight Factor: 2.0 (reduced from 3.5)")
print(f"  - Early Stopping Patience: 10 (reduced from 15)")
print(f"\nExpected Runtime: 3-4 hours (includes 20 epochs self-supervised pre-training)")
print(f"Expected GPU Usage: ~11-13GB (batch_size=20, 7 frames)")
print(f"Target: R² > 0.15, MAE < 0.30 km")
print("=" * 80)
print("\nTraining started... Monitor GPU with: !nvidia-smi")
print("\n⭐ WATCH FOR THIS BANNER (confirms Tier 1 is running):")
print("   ======================================================================")
print("   TIER 1: SELF-SUPERVISED PRE-TRAINING ENABLED")
print("   ======================================================================")
print()

%cd /content/repo
!python main.py \
    --config configs/colab_optimized_full_tuned.yaml \
    --save_name {experiment_name} \
    --epochs 50

print("\n" + "=" * 80)
print("TIER 1 TRAINING COMPLETE!")
print("=" * 80)
print(f"Model saved to: /content/drive/MyDrive/CloudML/models/trained/{experiment_name}.pth")
print(f"Pre-trained encoder: /content/drive/MyDrive/CloudML/models/pretrained/")
print(f"Results saved to: /content/drive/MyDrive/CloudML/plots/")
print(f"Logs saved to: /content/drive/MyDrive/CloudML/logs/")
print(f"\nCompare with baseline results to see Tier 1 improvements!")
print("\nCheck TensorBoard:")
print("  %load_ext tensorboard")
print("  %tensorboard --logdir /content/drive/MyDrive/CloudML/logs/tensorboard/")

# ============================================================================
# WHAT YOU SHOULD SEE IF TIER 1 IS WORKING:
# ============================================================================
# 1. Git pull shows "Already up to date" or lists updated files
# 2. Both pretraining.py and multi_scale_attention.py are found
# 3. Training output shows:
#    ======================================================================
#    TIER 1: SELF-SUPERVISED PRE-TRAINING ENABLED
#    ======================================================================
#    Starting self-supervised pre-training...
#    Epoch 1/20 - Reconstruction Loss: X.XXXX
#    ...
#    Pre-training complete! Proceeding to supervised training...
#
# 4. Then normal training proceeds with "--- Pretraining on 30Oct24 ---"
#
# IF YOU DON'T SEE THE TIER 1 BANNER:
# - You're running old code (baseline only)
# - Run this cell again to re-pull the latest code
# ============================================================================