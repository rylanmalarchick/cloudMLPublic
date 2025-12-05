#!/bin/bash
# Run MoCo pretraining on labeled data + linear probe evaluation
# 
# This script:
#   1. Trains MoCo on 2,321 labeled samples (matching eval distribution)
#   2. Runs linear probe evaluation on the trained model
#   3. Compares against SimCLR and random baseline
#
# Expected runtime: ~20-30 minutes on GPU (similar to SimCLR)
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 nohup bash scripts/cbh/run_moco_labeled.sh > moco_labeled.log 2>&1 &

set -e

echo "=========================================="
echo "MoCo Pretraining on Labeled Data"
echo "=========================================="
echo "Start time: $(date)"
echo ""

# Activate virtual environment
cd /home/rylan/Documents/research/NASA/programDirectory/cloudMLPublic
source venv/bin/activate

# Check GPU
echo "GPU Status:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# Verify CUDA is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
echo ""

# Step 1: Run MoCo pretraining on labeled data
echo "=========================================="
echo "Step 1: MoCo Pretraining (labeled only)"
echo "=========================================="
python experiments/paper2/moco_pretrain_labeled.py

# Find the latest run directory
LATEST_RUN=$(ls -td outputs/paper2_moco_labeled/run_* | head -1)
echo ""
echo "Latest run directory: $LATEST_RUN"

# Step 2: Run linear probe evaluation
echo ""
echo "=========================================="
echo "Step 2: Linear Probe Evaluation (MoCo)"
echo "=========================================="
python experiments/paper2/linear_probe.py --checkpoint "$LATEST_RUN/best_model.pt" --model moco

# Summary
echo ""
echo "=========================================="
echo "COMPLETE"
echo "=========================================="
echo "End time: $(date)"
echo ""
echo "Results saved to:"
echo "  - Pretrained model: $LATEST_RUN/"
echo "  - Linear probe results: outputs/paper2_simclr/linear_probe/"
echo ""
echo "To view results:"
echo "  cat $LATEST_RUN/training.log"
echo "  cat outputs/paper2_simclr/linear_probe/eval_*/results.json"
echo ""
echo "Compare with SimCLR results to determine if MoCo helps cross-flight generalization."
