#!/bin/bash
#
# Visualize Angle-CBH Correlations
#

set -e
source venv/bin/activate

CONFIG="configs/ssl_finetune_cbh.yaml"

while [[ $# -gt 0 ]]; do
    case $1 in
        --config) CONFIG="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "=========================================="
echo "Angle-CBH Correlation Analysis"
echo "=========================================="

python scripts/visualize_angle_cbh.py --config "$CONFIG"

echo ""
echo "Complete! Check outputs/angle_cbh_analysis/"
