#!/bin/bash
#
# Run Leave-One-Out per-flight cross-validation
#
# This script validates the hybrid MAE+GBDT model using LOO CV,
# training on 4 flights and testing on the 5th, rotating through all flights.
#

set -e  # Exit on error

# Activate virtual environment
source venv/bin/activate

# Default paths
CONFIG="configs/ssl_finetune_cbh.yaml"
ENCODER="outputs/mae_pretrain/mae_encoder_pretrained.pth"
DEVICE="cuda"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --encoder)
            ENCODER="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--config CONFIG] [--encoder ENCODER] [--device DEVICE]"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "LOO Per-Flight Cross-Validation"
echo "=========================================="
echo "Config: $CONFIG"
echo "Encoder: $ENCODER"
echo "Device: $DEVICE"
echo ""

# Run LOO validation
python scripts/validate_hybrid_loo.py \
    --config "$CONFIG" \
    --encoder "$ENCODER" \
    --device "$DEVICE"

echo ""
echo "=========================================="
echo "LOO validation complete!"
echo "Check outputs/loo_validation/ for results"
echo "=========================================="
