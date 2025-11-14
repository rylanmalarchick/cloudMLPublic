#!/bin/bash
# Phase 1: Data Extraction for SSL Pre-Training
# This script extracts all IR images from flight HDF5 files

set -e  # Exit on error

echo "=========================================="
echo "PHASE 1: DATA EXTRACTION FOR SSL"
echo "=========================================="
echo ""

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Configuration
CONFIG_FILE="${1:-configs/ssl_extract.yaml}"
OUTPUT_DIR="${2:-data_ssl/images}"

echo "Configuration:"
echo "  Config file: $CONFIG_FILE"
echo "  Output directory: $OUTPUT_DIR"
echo ""

# Check if config exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    echo "Usage: $0 [config_file] [output_dir]"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Run extraction
echo "Starting extraction..."
echo ""
python scripts/extract_all_images.py \
    --config "$CONFIG_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --format hdf5

# Check if extraction was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "EXTRACTION COMPLETE!"
    echo "=========================================="
    echo ""

    # Run verification
    echo "Running verification..."
    python scripts/verify_extraction.py \
        --data-dir "$OUTPUT_DIR" \
        --format hdf5 \
        --plot-samples

    if [ $? -eq 0 ]; then
        echo ""
        echo " Phase 1 complete! Ready for Phase 2 (SSL pre-training)"
        echo ""
        echo "Files created:"
        ls -lh "$OUTPUT_DIR"
        echo ""
        echo "Next step:"
        echo "  ./scripts/run_phase2_pretrain.sh"
    else
        echo ""
        echo "  Verification completed with warnings. Please review output above."
    fi
else
    echo ""
    echo " Extraction failed! Please check errors above."
    exit 1
fi
