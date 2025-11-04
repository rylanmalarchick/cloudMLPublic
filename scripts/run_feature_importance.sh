#!/bin/bash

echo "=========================================="
echo "Feature Importance Analysis"
echo "=========================================="
echo "Config: configs/ssl_finetune_cbh.yaml"
echo "Encoder: outputs/mae_pretrain/mae_encoder_pretrained.pth"
echo ""

# Check if encoder exists
if [ ! -f "outputs/mae_pretrain/mae_encoder_pretrained.pth" ]; then
    echo "ERROR: Pre-trained encoder not found!"
    echo "Expected: outputs/mae_pretrain/mae_encoder_pretrained.pth"
    echo ""
    echo "Please run MAE pretraining first:"
    echo "  ./scripts/run_mae_pretrain.sh"
    exit 1
fi

# Run feature importance analysis
./venv/bin/python scripts/analyze_feature_importance.py \
    --config configs/ssl_finetune_cbh.yaml \
    --encoder outputs/mae_pretrain/mae_encoder_pretrained.pth

echo ""
echo "=========================================="
echo "Feature importance analysis complete!"
echo "Check outputs/feature_importance/ for:"
echo "  - feature_importance_full.png"
echo "  - angle_importance_comparison.png"
echo "  - top_mae_dimensions.png"
echo "  - importance_method_comparison.png"
echo "  - feature_importance_analysis.json"
echo "=========================================="
