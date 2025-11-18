#!/bin/bash
# Run All Preprint Revision Analyses
# Execute from project root directory

set -e  # Exit on error

echo "========================================="
echo "Preprint Revision Analysis Pipeline"
echo "========================================="
echo ""

# Create output directories
echo "[1/4] Creating output directories..."
mkdir -p outputs/vision_baselines/{figures,reports}
mkdir -p outputs/domain_analysis/{figures,reports}
mkdir -p outputs/physics_validation/{figures,reports}
echo "✓ Directories created"
echo ""

# Run vision baselines
echo "[2/4] Running vision baseline ablation study..."
echo "This will train 6 models (ResNet-18, EfficientNet-B0 with ablations)"
echo "Estimated time: 30-60 minutes (GPU recommended)"
python src/cbh_retrieval/vision_baselines.py
echo "✓ Vision baselines complete"
echo ""

# Run domain shift analysis
echo "[3/4] Running domain shift and LOFO-CV analysis..."
echo "Estimated time: 5-10 minutes"
python src/cbh_retrieval/domain_shift_analysis.py
echo "✓ Domain shift analysis complete"
echo ""

# Run physics validation
echo "[4/4] Running physics-based validation..."
echo "Estimated time: 3-5 minutes"
python src/cbh_retrieval/physics_validation.py
echo "✓ Physics validation complete"
echo ""

echo "========================================="
echo "All Analyses Complete!"
echo "========================================="
echo ""
echo "Results saved to:"
echo "  - outputs/vision_baselines/"
echo "  - outputs/domain_analysis/"
echo "  - outputs/physics_validation/"
echo ""
echo "Next steps:"
echo "  1. Review generated figures and reports"
echo "  2. Integrate LaTeX tables into preprint"
echo "  3. Update results section with new metrics"
echo "  4. Complete remaining tasks (see REVISION_PROGRESS_REPORT.md)"
echo ""
