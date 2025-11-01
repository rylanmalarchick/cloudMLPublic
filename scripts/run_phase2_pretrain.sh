#!/bin/bash
# Phase 2: MAE Self-Supervised Pre-Training
# This script runs MAE pre-training on extracted images from Phase 1

set -e  # Exit on error

echo "=========================================="
echo "PHASE 2: MAE SELF-SUPERVISED PRE-TRAINING"
echo "=========================================="
echo ""

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Configuration
CONFIG_FILE="${1:-configs/ssl_pretrain_mae.yaml}"
MODEL_SIZE="${2:-small}"

echo "Configuration:"
echo "  Config file: $CONFIG_FILE"
echo "  Model size: $MODEL_SIZE"
echo ""

# Check if config exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    echo "Usage: $0 [config_file] [model_size]"
    exit 1
fi

# Check if Phase 1 data exists
if [ ! -f "data_ssl/images/train.h5" ]; then
    echo "ERROR: Phase 1 data not found!"
    echo "Please run Phase 1 first: ./scripts/run_phase1.sh"
    exit 1
fi

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    echo ""
else
    echo "WARNING: nvidia-smi not found. Training will use CPU (very slow!)"
    echo ""
fi

# Run pre-training
echo "Starting MAE pre-training..."
echo ""

python scripts/pretrain_mae.py \
    --config "$CONFIG_FILE" \
    --model-size "$MODEL_SIZE"

# Check if training was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "PRE-TRAINING COMPLETE!"
    echo "=========================================="
    echo ""

    # Check for encoder weights
    if [ -f "outputs/mae_pretrain/mae_encoder_pretrained.pth" ]; then
        echo "✅ Encoder weights saved successfully"
        echo ""
        echo "Files created:"
        echo "  - outputs/mae_pretrain/mae_encoder_pretrained.pth (encoder weights)"
        echo "  - outputs/mae_pretrain/checkpoints/best.pth (best checkpoint)"
        echo "  - outputs/mae_pretrain/checkpoints/latest.pth (latest checkpoint)"
        echo "  - outputs/mae_pretrain/plots/ (reconstruction visualizations)"
        echo "  - outputs/mae_pretrain/logs/ (TensorBoard logs)"
        echo ""
        echo "View training curves with:"
        echo "  tensorboard --logdir outputs/mae_pretrain/logs/"
        echo ""
        echo "Next step: Phase 3 (Fine-tuning for CBH)"
        echo "  ./scripts/run_phase3_finetune.sh"
    else
        echo "⚠️  Warning: Encoder weights not found at expected location"
    fi
else
    echo ""
    echo "❌ Pre-training failed! Please check errors above."
    exit 1
fi
