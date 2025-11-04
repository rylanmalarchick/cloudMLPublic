#!/bin/bash
#
# Run Physical Sanity Checks and Results Analysis
#

set -e
source venv/bin/activate

CONFIG="configs/ssl_finetune_cbh.yaml"
ENCODER="outputs/mae_pretrain/mae_encoder_pretrained.pth"
DEVICE="cuda"

while [[ $# -gt 0 ]]; do
    case $1 in
        --config) CONFIG="$2"; shift 2 ;;
        --encoder) ENCODER="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "=========================================="
echo "Results Analysis & Sanity Checks"
echo "=========================================="

python scripts/analyze_results.py \
    --config "$CONFIG" \
    --encoder "$ENCODER" \
    --device "$DEVICE"

echo ""
echo "Complete! Check outputs/results_analysis/"
