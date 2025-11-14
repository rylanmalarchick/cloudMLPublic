#!/bin/bash
#
# Week 1, Task 1.1: Spatial Feature Extraction Experiments
# ==========================================================
#
# This script runs all three spatial MAE variants with LOO cross-validation
# to compare against the baseline CLS token approach.
#
# Expected outputs:
# - outputs/spatial_mae/pooling_<timestamp>/loo_results.json
# - outputs/spatial_mae/cnn_<timestamp>/loo_results.json
# - outputs/spatial_mae/attention_<timestamp>/loo_results.json
#
# Author: Cloud ML Research Team
# Date: 2024

set -e  # Exit on error

echo "========================================="
echo "Week 1, Task 1.1: Spatial MAE Training"
echo "========================================="
echo ""

# Configuration
CONFIG="configs/ssl_finetune_cbh.yaml"
ENCODER="outputs/mae_pretrain/mae_encoder_pretrained.pth"
EPOCHS=50
DEVICE="cuda"

# Check if encoder exists
if [ ! -f "$ENCODER" ]; then
    echo "WARNING: Pretrained encoder not found at $ENCODER"
    echo "Training will proceed with random initialization."
    echo "To use a pretrained encoder, run MAE pretraining first:"
    echo "  python scripts/pretrain_mae.py --config configs/ssl_pretrain_mae.yaml"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if config exists
if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Config file not found at $CONFIG"
    exit 1
fi

echo "Configuration:"
echo "  Config:  $CONFIG"
echo "  Encoder: $ENCODER"
echo "  Epochs:  $EPOCHS"
echo "  Device:  $DEVICE"
echo ""

# Create output directory
mkdir -p outputs/spatial_mae
mkdir -p logs

# Timestamp for this run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="logs/week1_task1_$TIMESTAMP"
mkdir -p "$LOG_DIR"

echo "Logs will be saved to: $LOG_DIR"
echo ""

# Function to run training for a variant
run_variant() {
    VARIANT=$1
    echo ""
    echo "========================================="
    echo "Training Variant: $VARIANT"
    echo "========================================="

    LOG_FILE="$LOG_DIR/${VARIANT}.log"

    ./venv/bin/python scripts/train_spatial_mae.py \
        --variant "$VARIANT" \
        --config "$CONFIG" \
        --encoder "$ENCODER" \
        --epochs "$EPOCHS" \
        --device "$DEVICE" \
        2>&1 | tee "$LOG_FILE"

    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo " $VARIANT training completed successfully"
    else
        echo " $VARIANT training failed (see $LOG_FILE)"
        exit 1
    fi
}

# Run all three variants sequentially
echo "Starting training runs..."
echo ""

run_variant "pooling"
run_variant "cnn"
run_variant "attention"

echo ""
echo "========================================="
echo "All Training Runs Complete!"
echo "========================================="
echo ""

# Collect and compare results
echo "Collecting results..."

# Find the latest results for each variant
POOLING_RESULTS=$(ls -t outputs/spatial_mae/pooling_*/loo_results.json 2>/dev/null | head -1)
CNN_RESULTS=$(ls -t outputs/spatial_mae/cnn_*/loo_results.json 2>/dev/null | head -1)
ATTENTION_RESULTS=$(ls -t outputs/spatial_mae/attention_*/loo_results.json 2>/dev/null | head -1)

# Create comparison summary
SUMMARY_FILE="$LOG_DIR/summary.txt"

{
    echo "========================================="
    echo "Week 1, Task 1.1: Results Summary"
    echo "========================================="
    echo ""
    echo "Timestamp: $TIMESTAMP"
    echo ""

    if [ -f "$POOLING_RESULTS" ]; then
        echo "Variant A: Spatial Pooling + MLP"
        echo "--------------------------------"
        ./venv/bin/python << EOF
import json
with open('$POOLING_RESULTS', 'r') as f:
    data = json.load(f)
    agg = data['aggregate']
    print(f"  Mean R²:   {agg['mean_r2']:.4f} ± {agg['std_r2']:.4f}")
    print(f"  Mean MAE:  {agg['mean_mae']:.1f} ± {agg['std_mae']:.1f} m")
    print(f"  Mean RMSE: {agg['mean_rmse']:.1f} ± {agg['std_rmse']:.1f} m")
    print(f"  Results:   $POOLING_RESULTS")
EOF
        echo ""
    fi

    if [ -f "$CNN_RESULTS" ]; then
        echo "Variant B: CNN Spatial Head"
        echo "--------------------------------"
        ./venv/bin/python << EOF
import json
with open('$CNN_RESULTS', 'r') as f:
    data = json.load(f)
    agg = data['aggregate']
    print(f"  Mean R²:   {agg['mean_r2']:.4f} ± {agg['std_r2']:.4f}")
    print(f"  Mean MAE:  {agg['mean_mae']:.1f} ± {agg['std_mae']:.1f} m")
    print(f"  Mean RMSE: {agg['mean_rmse']:.1f} ± {agg['std_rmse']:.1f} m")
    print(f"  Results:   $CNN_RESULTS")
EOF
        echo ""
    fi

    if [ -f "$ATTENTION_RESULTS" ]; then
        echo "Variant C: Attention Pooling"
        echo "--------------------------------"
        ./venv/bin/python << EOF
import json
with open('$ATTENTION_RESULTS', 'r') as f:
    data = json.load(f)
    agg = data['aggregate']
    print(f"  Mean R²:   {agg['mean_r2']:.4f} ± {agg['std_r2']:.4f}")
    print(f"  Mean MAE:  {agg['mean_mae']:.1f} ± {agg['std_mae']:.1f} m")
    print(f"  Mean RMSE: {agg['mean_rmse']:.1f} ± {agg['std_rmse']:.1f} m")
    print(f"  Results:   $ATTENTION_RESULTS")
EOF
        echo ""
    fi

    echo "========================================="
    echo "Baseline Comparison"
    echo "========================================="
    echo ""
    echo "Previous Results (from diagnostic analysis):"
    echo "  CLS Token + GBDT: R² = 0.49-0.51, MAE = 173-188 m"
    echo "  Angles-only GBDT: R² = 0.70-0.71, MAE = 120-123 m"
    echo "  LOO CV (angles):  R² = -4.46, MAE = 348 m"
    echo ""
    echo "Goal: Spatial features should achieve R² > 0 in LOO CV"
    echo "      (better than mean prediction baseline)"
    echo ""

} | tee "$SUMMARY_FILE"

echo "Summary saved to: $SUMMARY_FILE"
echo ""

# Generate comparison plot if Python is available
echo "Generating comparison plots..."
./venv/bin/python << 'PLOTSCRIPT'
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

# Find latest results
def find_latest_results():
    results = {}
    variants = ['pooling', 'cnn', 'attention']

    for variant in variants:
        pattern = f"outputs/spatial_mae/{variant}_*/loo_results.json"
        import glob
        files = sorted(glob.glob(pattern), reverse=True)
        if files:
            results[variant] = files[0]

    return results

try:
    results_files = find_latest_results()

    if not results_files:
        print("No results found to plot.")
        sys.exit(0)

    # Load data
    data = {}
    for variant, filepath in results_files.items():
        with open(filepath, 'r') as f:
            data[variant] = json.load(f)

    # Create comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    variants = list(data.keys())
    variant_labels = {
        'pooling': 'Pooling + MLP',
        'cnn': 'CNN Head',
        'attention': 'Attention Pool'
    }

    # Plot 1: Mean R² comparison
    ax = axes[0]
    mean_r2 = [data[v]['aggregate']['mean_r2'] for v in variants]
    std_r2 = [data[v]['aggregate']['std_r2'] for v in variants]
    x_pos = np.arange(len(variants))

    bars = ax.bar(x_pos, mean_r2, yerr=std_r2, capsize=5, alpha=0.7)
    ax.axhline(0, color='red', linestyle='--', linewidth=2, label='Baseline (mean)')
    ax.axhline(0.5, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Target (R²=0.5)')
    ax.set_xlabel('Variant')
    ax.set_ylabel('Mean R² (LOO CV)')
    ax.set_title('R² Comparison Across Variants')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([variant_labels[v] for v in variants])
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Color bars based on performance
    for i, bar in enumerate(bars):
        if mean_r2[i] > 0.3:
            bar.set_color('green')
        elif mean_r2[i] > 0:
            bar.set_color('orange')
        else:
            bar.set_color('red')

    # Plot 2: Mean MAE comparison
    ax = axes[1]
    mean_mae = [data[v]['aggregate']['mean_mae'] for v in variants]
    std_mae = [data[v]['aggregate']['std_mae'] for v in variants]

    bars = ax.bar(x_pos, mean_mae, yerr=std_mae, capsize=5, alpha=0.7, color='orange')
    ax.axhline(200, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Baseline (~200m)')
    ax.axhline(100, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Target (<100m)')
    ax.set_xlabel('Variant')
    ax.set_ylabel('Mean MAE (m)')
    ax.set_title('MAE Comparison Across Variants')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([variant_labels[v] for v in variants])
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Per-fold R² breakdown (first variant)
    ax = axes[2]
    first_variant = variants[0]
    folds_data = data[first_variant]['folds']
    flights = [f['test_flight'] for f in folds_data]
    fold_r2 = [f['r2'] for f in folds_data]

    bars = ax.bar(range(len(flights)), fold_r2, alpha=0.7)
    ax.axhline(0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Test Flight')
    ax.set_ylabel('R²')
    ax.set_title(f'Per-Fold R² ({variant_labels[first_variant]})')
    ax.set_xticks(range(len(flights)))
    ax.set_xticklabels(flights, rotation=45, ha='right')
    ax.grid(True, alpha=0.3)

    # Color code bars
    for i, bar in enumerate(bars):
        if fold_r2[i] > 0.3:
            bar.set_color('green')
        elif fold_r2[i] > 0:
            bar.set_color('orange')
        else:
            bar.set_color('red')

    plt.tight_layout()

    # Save plot
    import os
    timestamp = os.path.basename(Path(results_files[variants[0]]).parent).split('_')[-1]
    plot_path = f"outputs/spatial_mae/comparison_{timestamp}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Comparison plot saved to: {plot_path}")

except Exception as e:
    print(f"Error generating plots: {e}")
    import traceback
    traceback.print_exc()

PLOTSCRIPT

echo ""
echo "========================================="
echo "Week 1, Task 1.1 Complete!"
echo "========================================="
echo ""
echo "Next Steps:"
echo "  1. Review results in $LOG_DIR/"
echo "  2. Analyze which variant performs best"
echo "  3. If R² > 0: Proceed to Task 1.2 (Physical Priors)"
echo "  4. If R² < 0: Investigate failure modes, consider alternative approaches"
echo ""
echo "To view summary:"
echo "  cat $SUMMARY_FILE"
echo ""
