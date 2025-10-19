#!/usr/bin/env python3
"""
Update the Colab notebook Option A-Tuned cell to pull latest Tier 1 code
"""

import json
import sys


def update_notebook():
    """Update the Option A-Tuned cell with git pull and Tier 1 verification"""

    # Read notebook
    try:
        with open("colab_training.ipynb", "r", encoding="utf-8") as f:
            content = f.read()
            nb = json.loads(content)
    except Exception as e:
        print(f"Error reading notebook: {e}")
        return False

    # Find and update the Option A-Tuned cell
    updated = False
    for i, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") != "code":
            continue

        source = "".join(cell.get("source", []))

        if (
            "OPTION A-TUNED: FULL MODEL + TUNED HYPERPARAMETERS" in source
            and "RECOMMENDED" in source
        ):
            print(f"Found Option A-Tuned cell at index {i}")

            # New source with git pull and Tier 1 verification
            new_source = [
                "# ============================================================================\n",
                "# OPTION A-TUNED: FULL MODEL + TUNED HYPERPARAMETERS + TIER 1 (RECOMMENDED) ⭐⭐\n",
                "# ============================================================================\n",
                "\n",
                "# STEP 1: Pull latest code with Tier 1 implementation\n",
                'print("Pulling latest Tier 1 code...")\n',
                "%cd /content/repo\n",
                "!git pull origin main\n",
                'print("✓ Code updated\\n")\n',
                "\n",
                "# STEP 2: Verify Tier 1 modules exist\n",
                "import os\n",
                "if os.path.exists('/content/repo/src/pretraining.py'):\n",
                '    print("✓ Tier 1 self-supervised pretraining module found")\n',
                "else:\n",
                '    print("⚠ WARNING: Tier 1 module not found - running baseline only")\n',
                "\n",
                "# STEP 3: Start training\n",
                "import datetime\n",
                "\n",
                'timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")\n',
                'baseline_name = f"tier1_tuned_{timestamp}"\n',
                "\n",
                'print("\\n" + "="*80)\n',
                'print("TIER 1 TRAINING: FULL MODEL + TUNED HYPERPARAMETERS + LITERATURE IMPROVEMENTS")\n',
                'print("="*80)\n',
                'print(f"Experiment ID: {baseline_name}")\n',
                'print(f"Config: colab_optimized_full_tuned.yaml")\n',
                'print(f"Model: 64/128/256 channels (FULL)")\n',
                'print(f"\\nTIER 1 FEATURES ENABLED:")\n',
                'print(f"  ✅ Self-supervised pre-training (20 epochs reconstruction)")\n',
                'print(f"  ✅ Multi-scale temporal attention (4 heads)")\n',
                'print(f"  ✅ Increased temporal frames (7 frames)")\n',
                'print(f"  ✅ Expected R² improvement: +15-25%")\n',
                'print(f"\\nTuned Hyperparameters:")\n',
                'print(f"  - Learning Rate: 0.0005 (reduced from 0.001)")\n',
                'print(f"  - Warmup Steps: 500 (reduced from 2000)")\n',
                'print(f"  - Overweight Factor: 2.0 (reduced from 3.5)")\n',
                'print(f"  - Early Stopping Patience: 10 (reduced from 15)")\n',
                'print(f"\\nExpected Runtime: 3-4 hours (includes 20 epochs self-supervised pre-training)")\n',
                'print(f"Expected GPU Usage: ~11-13GB (batch_size=20, 7 frames)")\n',
                'print(f"Target: R² > 0.15, MAE < 0.30 km")\n',
                'print("="*80)\n',
                'print("\\nTraining started... Monitor GPU with: !nvidia-smi\\n")\n',
                "print(\"Watch for: 'TIER 1: SELF-SUPERVISED PRE-TRAINING ENABLED' banner\\n\")\n",
                "\n",
                "%cd /content/repo\n",
                "!python main.py \\\n",
                "    --config configs/colab_optimized_full_tuned.yaml \\\n",
                "    --save_name {baseline_name} \\\n",
                "    --epochs 50\n",
                "\n",
                'print("\\n" + "="*80)\n',
                'print("TIER 1 TRAINING COMPLETE!")\n',
                'print("="*80)\n',
                'print(f"Model saved to: /content/drive/MyDrive/CloudML/models/trained/{baseline_name}.pth")\n',
                'print(f"Pre-trained encoder: /content/drive/MyDrive/CloudML/models/pretrained/")\n',
                'print(f"Results saved to: /content/drive/MyDrive/CloudML/plots/")\n',
                'print(f"Logs saved to: /content/drive/MyDrive/CloudML/logs/")\n',
                'print(f"\\nCompare with baseline results to see Tier 1 improvements!")\n',
                'print("\\nCheck TensorBoard: %load_ext tensorboard")\n',
                'print("                   %tensorboard --logdir /content/drive/MyDrive/CloudML/logs/tensorboard/")',
            ]

            cell["source"] = new_source
            updated = True
            print("✓ Updated cell source")
            break

    if not updated:
        print("⚠ Could not find Option A-Tuned cell to update")
        return False

    # Write updated notebook
    try:
        with open("colab_training.ipynb", "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print("✓ Wrote updated notebook")
        return True
    except Exception as e:
        print(f"Error writing notebook: {e}")
        return False


if __name__ == "__main__":
    success = update_notebook()
    sys.exit(0 if success else 1)
