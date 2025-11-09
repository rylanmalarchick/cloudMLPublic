#!/bin/bash
# WP-4 Full Training Script

source venv/bin/activate

echo "=========================================="
echo "WP-4 HYBRID MODEL TRAINING"
echo "=========================================="
echo "Start time: $(date)"
echo ""

# Run all three model variants
for mode in image_only concat attention; do
    echo "=========================================="
    echo "Training: $mode"
    echo "=========================================="
    
    python -u sow_outputs/wp4_hybrid_model.py \
        --mode $mode \
        --epochs 50 \
        --batch-size 32 \
        > sow_outputs/wp4_hybrid/training_${mode}.log 2>&1
    
    echo "✓ Completed: $mode at $(date)"
    echo ""
done

echo "=========================================="
echo "ALL TRAINING COMPLETE"
echo "End time: $(date)"
echo "=========================================="
