#!/bin/bash

# ============================================================================
# STRATEGY 1: HYBRID MAE + GRADIENTBOOSTING
# ============================================================================
# This script runs the hybrid approach combining:
# - Pre-trained MAE encoder (for deep feature extraction)
# - GradientBoosting regressor (proven classical ML)
#
# Expected: R² potentially > 0.75 (beating classical baseline)
# ============================================================================

set -e  # Exit on error

echo "=========================================="
echo "STRATEGY 1: HYBRID MAE + GBDT"
echo "=========================================="
echo ""

# Check if encoder exists
ENCODER_PATH="outputs/mae_pretrain/mae_encoder_pretrained.pth"
if [ ! -f "$ENCODER_PATH" ]; then
    echo "❌ ERROR: Pre-trained encoder not found at $ENCODER_PATH"
    echo ""
    echo "Please run Phase 2 pre-training first:"
    echo "  ./scripts/run_phase2_pretrain.sh"
    echo ""
    exit 1
fi

echo "✓ Found pre-trained encoder: $ENCODER_PATH"
echo ""

# Activate virtual environment
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
elif [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
else
    echo "⚠ Warning: No virtual environment found"
fi

# Configuration
CONFIG="configs/ssl_finetune_cbh.yaml"

echo "Configuration:"
echo "  Config file: $CONFIG"
echo "  Encoder: $ENCODER_PATH"
echo ""

# Check GPU
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits | awk -F', ' '{print $1 ", " $2 " MiB, " $3 " MiB"}'
    echo ""
fi

# Run hybrid pipeline
echo "Starting hybrid MAE + GBDT pipeline..."
echo ""
echo "This will:"
echo "  1. Load pre-trained MAE encoder"
echo "  2. Extract embeddings (192-dim) from all labeled samples"
echo "  3. Combine embeddings with solar angles (SZA, SAA)"
echo "  4. Train GradientBoosting regressor with hyperparameter tuning"
echo "  5. Evaluate and compare to baselines"
echo ""
echo "Expected runtime: ~10-15 minutes"
echo ""

# Run with hyperparameter tuning
python scripts/hybrid_mae_gbdt.py \
    --config $CONFIG \
    --encoder $ENCODER_PATH \
    --device cuda

echo ""
echo "=========================================="
echo "HYBRID PIPELINE COMPLETE!"
echo "=========================================="
echo ""
echo "✅ Results saved to: outputs/hybrid_mae_gbdt/"
echo ""
echo "View results:"
echo "  - Metrics: outputs/hybrid_mae_gbdt/*/metrics.json"
echo "  - Plots: outputs/hybrid_mae_gbdt/*/hybrid_results.png"
echo ""
echo "Next steps:"
echo "  - If R² > 0.75: SUCCESS! Publish results"
echo "  - If R² 0.4-0.75: Try Strategy 2 (optimize SSL pipeline)"
echo "  - If R² < 0.4: Debug and analyze failure modes"
echo ""
echo "=========================================="
echo "STRATEGY 1 COMPLETE! 🚀"
echo "=========================================="
