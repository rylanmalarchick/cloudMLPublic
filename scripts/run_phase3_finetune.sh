#!/bin/bash
# ================================================================================
# PHASE 3: FINE-TUNE PRE-TRAINED ENCODER FOR CBH REGRESSION
# ================================================================================
# This script runs two-stage fine-tuning:
#   Stage 1: Freeze encoder, train regression head only
#   Stage 2: Unfreeze encoder, fine-tune end-to-end
#
# Target: Beat classical baseline (GradientBoosting R² = 0.7464)
# ================================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "=========================================="
echo "PHASE 3: FINE-TUNING FOR CBH REGRESSION"
echo "=========================================="
echo ""

# ================================================================================
# CONFIGURATION
# ================================================================================

# Default config file
CONFIG_FILE="${1:-configs/ssl_finetune_cbh.yaml}"

# Check if config exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}Error: Config file not found: $CONFIG_FILE${NC}"
    echo "Usage: $0 [config_file]"
    echo "Example: $0 configs/ssl_finetune_cbh.yaml"
    exit 1
fi

# Check if pre-trained encoder exists
ENCODER_WEIGHTS="outputs/mae_pretrain/mae_encoder_pretrained.pth"
if [ ! -f "$ENCODER_WEIGHTS" ]; then
    echo -e "${RED}Error: Pre-trained encoder not found: $ENCODER_WEIGHTS${NC}"
    echo ""
    echo "Please run Phase 2 (MAE pre-training) first:"
    echo "  ./scripts/run_phase2_pretrain.sh"
    exit 1
fi

echo -e "${GREEN}✓ Found pre-trained encoder: $ENCODER_WEIGHTS${NC}"
echo ""

# ================================================================================
# ACTIVATE VIRTUAL ENVIRONMENT
# ================================================================================

if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
elif [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
else
    echo -e "${YELLOW}Warning: No virtual environment found${NC}"
    echo "Consider creating one with: python -m venv venv"
fi

# ================================================================================
# SYSTEM INFO
# ================================================================================

echo "Configuration:"
echo "  Config file: $CONFIG_FILE"
echo ""

# Check GPU
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits | \
        awk -F', ' '{printf "%s, %d MiB, %d MiB\n", $1, $2, $3}'
    echo ""
else
    echo -e "${YELLOW}Warning: nvidia-smi not found. Using CPU.${NC}"
    echo ""
fi

# ================================================================================
# RUN FINE-TUNING
# ================================================================================

echo "Starting fine-tuning..."
echo ""

python scripts/finetune_cbh.py --config "$CONFIG_FILE"

# ================================================================================
# CHECK RESULTS
# ================================================================================

echo ""
echo "=========================================="
echo "FINE-TUNING COMPLETE!"
echo "=========================================="
echo ""

# Check if output directory exists
OUTPUT_DIR="outputs/cbh_finetune"
if [ -d "$OUTPUT_DIR" ]; then
    echo -e "${GREEN}✅ Fine-tuning completed successfully${NC}"
    echo ""
    echo "Files created:"

    # List key output files
    if [ -d "$OUTPUT_DIR/checkpoints" ]; then
        echo "  - Checkpoints:"
        ls -lh "$OUTPUT_DIR/checkpoints" | tail -n +2 | awk '{printf "      %s (%s)\n", $9, $5}'
    fi

    if [ -d "$OUTPUT_DIR/plots" ]; then
        echo "  - Plots:"
        ls -1 "$OUTPUT_DIR/plots" | sed 's/^/      /'
    fi

    if [ -d "$OUTPUT_DIR/logs" ]; then
        echo "  - TensorBoard logs:"
        ls -1 "$OUTPUT_DIR/logs" | sed 's/^/      /'
    fi

    echo ""
    echo "View training curves with:"
    echo "  tensorboard --logdir $OUTPUT_DIR/logs/"
    echo ""

    # Check test results
    if [ -f "$OUTPUT_DIR/plots/test_results.png" ]; then
        echo "View test results plot:"
        echo "  xdg-open $OUTPUT_DIR/plots/test_results.png"
        echo ""
    fi

    # Display baseline comparison
    echo "Baseline Comparison:"
    echo "  Classical baseline (GradientBoosting):"
    echo "    R² = 0.7464, MAE = 0.1265 km, RMSE = 0.1929 km"
    echo ""
    echo "  Your model performance:"
    echo "    Check the final test set evaluation above ☝️"
    echo ""

    # Success thresholds
    echo "Performance Thresholds:"
    echo "  🎉 Excellent: R² >= 0.75"
    echo "  ✅ Good:      R² >= 0.60"
    echo "  👍 Acceptable: R² >= 0.40"
    echo "  ⚠️  Below:     R² <  0.40"
    echo ""

else
    echo -e "${RED}✗ Output directory not found: $OUTPUT_DIR${NC}"
    echo "Fine-tuning may have failed. Check error messages above."
    exit 1
fi

# ================================================================================
# NEXT STEPS
# ================================================================================

echo "Next steps:"
echo ""
echo "1. Review training curves:"
echo "   tensorboard --logdir $OUTPUT_DIR/logs/"
echo ""
echo "2. Analyze test results:"
echo "   - Check scatter plot: $OUTPUT_DIR/plots/test_results.png"
echo "   - Review residual distribution"
echo "   - Compare R² to baseline (0.7464)"
echo ""
echo "3. If performance is good (R² >= 0.60):"
echo "   - Consider advanced experiments (multi-task, temporal)"
echo "   - Explore encoder representations"
echo "   - Try different augmentation strategies"
echo ""
echo "4. If performance needs improvement (R² < 0.60):"
echo "   - Try longer pre-training (Phase 2)"
echo "   - Adjust fine-tuning hyperparameters"
echo "   - Experiment with model size (tiny vs small)"
echo "   - Check data quality and CPL alignment"
echo ""
echo "5. Write up results:"
echo "   - Document SSL approach effectiveness"
echo "   - Compare to supervised baseline"
echo "   - Discuss learned representations"
echo ""

echo "=========================================="
echo "PHASE 3 COMPLETE! 🚀"
echo "=========================================="
