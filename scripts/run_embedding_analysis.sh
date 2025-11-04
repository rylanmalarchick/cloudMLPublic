#!/bin/bash

echo "=========================================="
echo "MAE Embedding Analysis"
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

# Run embedding analysis
./venv/bin/python scripts/visualize_embeddings.py \
    --config configs/ssl_finetune_cbh.yaml \
    --encoder outputs/mae_pretrain/mae_encoder_pretrained.pth

echo ""
echo "=========================================="
echo "Embedding analysis complete!"
echo "Check outputs/embedding_analysis/ for:"
echo "  - embeddings_pca.png"
echo "  - embeddings_tsne.png"
echo "  - embeddings_umap.png (if UMAP installed)"
echo "  - correlation_heatmap.png"
echo "  - top_10_cbh_dimensions.png"
echo "  - embedding_analysis.json"
echo "=========================================="
