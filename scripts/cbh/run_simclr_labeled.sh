#!/bin/bash
# Run SimCLR pretraining on labeled data + linear probe evaluation
# 
# This script:
#   1. Trains SimCLR on 2,321 labeled samples (matching eval distribution)
#   2. Runs linear probe evaluation on the trained model
#   3. Compares against baseline
#
# Expected runtime: ~20-30 minutes on GPU
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 nohup bash scripts/cbh/run_simclr_labeled.sh > simclr_labeled.log 2>&1 &

set -e

echo "=========================================="
echo "SimCLR Pretraining on Labeled Data"
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

# Step 1: Run SimCLR pretraining on labeled data
echo "=========================================="
echo "Step 1: SimCLR Pretraining (labeled only)"
echo "=========================================="
python experiments/paper2/simclr_pretrain_labeled.py

# Find the latest run directory
LATEST_RUN=$(ls -td outputs/paper2_simclr_labeled/run_* | head -1)
echo ""
echo "Latest run directory: $LATEST_RUN"

# Step 2: Run linear probe evaluation
echo ""
echo "=========================================="
echo "Step 2: Linear Probe Evaluation"
echo "=========================================="
python experiments/paper2/linear_probe.py --checkpoint "$LATEST_RUN/best_model.pt"

# Step 3: Also run random baseline for comparison
echo ""
echo "=========================================="
echo "Step 3: Random Baseline (for comparison)"
echo "=========================================="
python experiments/paper2/linear_probe.py --random-baseline

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
