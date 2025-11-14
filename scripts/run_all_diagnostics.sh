#!/bin/bash

echo "================================================================================"
echo "COMPLETE DIAGNOSTIC SUITE FOR MAE+GBDT CBH PREDICTION"
echo "================================================================================"
echo ""
echo "This script runs all diagnostic analyses to understand model performance:"
echo "  1. Embedding visualization (t-SNE, UMAP, correlations)"
echo "  2. Feature importance analysis (GBDT feature weights)"
echo "  3. Ablation studies (stratified splits)"
echo "  4. Leave-one-out cross-validation (gold standard)"
echo ""
echo "Estimated runtime: 15-30 minutes"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Aborted."
    exit 1
fi

# Check if encoder exists
if [ ! -f "outputs/mae_pretrain/mae_encoder_pretrained.pth" ]; then
    echo ""
    echo "ERROR: Pre-trained encoder not found!"
    echo "Expected: outputs/mae_pretrain/mae_encoder_pretrained.pth"
    echo ""
    echo "Please run MAE pretraining first:"
    echo "  ./scripts/run_mae_pretrain.sh"
    exit 1
fi

# Create timestamp for this diagnostic run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DIAG_DIR="outputs/diagnostics_${TIMESTAMP}"
mkdir -p "$DIAG_DIR"

echo ""
echo "================================================================================"
echo "Diagnostic run: $TIMESTAMP"
echo "All results will be saved to: $DIAG_DIR"
echo "================================================================================"
echo ""

# Log file
LOG_FILE="$DIAG_DIR/diagnostic_log.txt"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "Started at: $(date)"
echo ""

# ============================================================================
# 1. Embedding Visualization
# ============================================================================
echo ""
echo "================================================================================"
echo "[1/4] EMBEDDING VISUALIZATION"
echo "================================================================================"
echo "Analyzing what MAE embeddings encode..."
echo ""

./venv/bin/python scripts/visualize_embeddings.py \
    --config configs/ssl_finetune_cbh.yaml \
    --encoder outputs/mae_pretrain/mae_encoder_pretrained.pth \
    --output "$DIAG_DIR/embedding_analysis"

if [ $? -ne 0 ]; then
    echo "ERROR: Embedding visualization failed!"
    exit 1
fi

echo ""
echo " Embedding visualization complete"
echo ""

# ============================================================================
# 2. Feature Importance Analysis
# ============================================================================
echo ""
echo "================================================================================"
echo "[2/4] FEATURE IMPORTANCE ANALYSIS"
echo "================================================================================"
echo "Analyzing which features GBDT relies on..."
echo ""

./venv/bin/python scripts/analyze_feature_importance.py \
    --config configs/ssl_finetune_cbh.yaml \
    --encoder outputs/mae_pretrain/mae_encoder_pretrained.pth \
    --output "$DIAG_DIR/feature_importance"

if [ $? -ne 0 ]; then
    echo "ERROR: Feature importance analysis failed!"
    exit 1
fi

echo ""
echo " Feature importance analysis complete"
echo ""

# ============================================================================
# 3. Ablation Studies (Stratified Splits)
# ============================================================================
echo ""
echo "================================================================================"
echo "[3/4] ABLATION STUDIES"
echo "================================================================================"
echo "Running comprehensive ablation tests..."
echo ""

./venv/bin/python scripts/ablation_studies.py \
    --config configs/ssl_finetune_cbh.yaml \
    --encoder outputs/mae_pretrain/mae_encoder_pretrained.pth \
    --output "$DIAG_DIR/ablation_studies"

if [ $? -ne 0 ]; then
    echo "ERROR: Ablation studies failed!"
    exit 1
fi

echo ""
echo " Ablation studies complete"
echo ""

# ============================================================================
# 4. Leave-One-Out Cross-Validation
# ============================================================================
echo ""
echo "================================================================================"
echo "[4/4] LEAVE-ONE-OUT CROSS-VALIDATION"
echo "================================================================================"
echo "Testing cross-flight generalization (gold standard)..."
echo ""

./venv/bin/python scripts/validate_hybrid_loo.py \
    --config configs/ssl_finetune_cbh.yaml \
    --encoder outputs/mae_pretrain/mae_encoder_pretrained.pth \
    --output "$DIAG_DIR/loo_validation"

if [ $? -ne 0 ]; then
    echo "ERROR: LOO cross-validation failed!"
    exit 1
fi

echo ""
echo " LOO cross-validation complete"
echo ""

# ============================================================================
# Generate Summary Report
# ============================================================================
echo ""
echo "================================================================================"
echo "GENERATING SUMMARY REPORT"
echo "================================================================================"
echo ""

# Create summary document
SUMMARY_FILE="$DIAG_DIR/DIAGNOSTIC_SUMMARY.md"

cat > "$SUMMARY_FILE" << 'EOF'
# Diagnostic Suite Summary

**Run Date:**
EOF

echo "$(date)" >> "$SUMMARY_FILE"

cat >> "$SUMMARY_FILE" << 'EOF'

---

## Overview

This directory contains comprehensive diagnostic analyses of the MAE+GBDT CBH prediction model.

## Results Organization

```
diagnostics_TIMESTAMP/
 DIAGNOSTIC_SUMMARY.md (this file)
 diagnostic_log.txt (full console output)
 embedding_analysis/
    embeddings_pca.png
    embeddings_tsne.png
    embeddings_umap.png (if available)
    correlation_heatmap.png
    top_10_cbh_dimensions.png
    embedding_analysis.json
 feature_importance/
    feature_importance_full.png
    angle_importance_comparison.png
    top_mae_dimensions.png
    importance_method_comparison.png
    feature_importance_analysis.json
 ablation_studies/
    ablation_results.png
    ablation_summary.csv
    ablation_results.json
 loo_validation/
     loo_validation_results.png
     fold_results.json
     aggregated_metrics.json
     summary_table.csv
```

---

## Quick Findings

### 1. Embedding Analysis

**Key Questions:**
- What do MAE embeddings encode?
- Do any dimensions correlate with CBH?
- Are embeddings clustered by flight, CBH, or angles?

**Check:**
- `embedding_analysis/correlation_heatmap.png` - Per-dimension correlations
- `embedding_analysis/embeddings_tsne.png` - Visual clustering patterns
- `embedding_analysis/embedding_analysis.json` - Correlation statistics

### 2. Feature Importance

**Key Questions:**
- Which features does GBDT rely on most?
- Do angles dominate over MAE dimensions?
- Are MAE embeddings actually used?

**Check:**
- `feature_importance/feature_importance_full.png` - Top features
- `feature_importance/angle_importance_comparison.png` - Angle vs MAE
- `feature_importance/feature_importance_analysis.json` - Aggregate stats

### 3. Ablation Studies

**Key Questions:**
- How do different feature combinations perform?
- Does MAE help or hurt performance?
- What's the best baseline?

**Check:**
- `ablation_studies/ablation_results.png` - Performance comparison
- `ablation_studies/ablation_summary.csv` - Detailed metrics
- `ablation_studies/ablation_results.json` - Raw results

### 4. LOO Cross-Validation

**Key Questions:**
- Does the model generalize across flights?
- What's the true cross-flight performance?
- Which flights are hardest to predict?

**Check:**
- `loo_validation/loo_validation_results.png` - Per-flight results
- `loo_validation/aggregated_metrics.json` - Overall statistics
- `loo_validation/summary_table.csv` - Detailed breakdown

---

## Interpretation Guide

### If MAE Embeddings Help:
- Feature importance: MAE dimensions have high importance
- Embedding analysis: Some dimensions correlate with CBH
- Ablation: MAE+Angles > Angles_only
- LOO CV: Positive R²

### If MAE Embeddings Don't Help (Current Status):
- Feature importance: Angles dominate importance
- Embedding analysis: Weak/no correlation with CBH
- Ablation: Angles_only ≥ MAE+Angles
- LOO CV: Negative R² (no generalization)

### If There's Overfitting:
- Ablation: Good stratified split performance
- LOO CV: Poor cross-flight performance
- Embedding analysis: Clusters by flight, not CBH

---

## Next Steps Based on Results

### If embeddings are useful:
1. Optimize fusion strategy (try MLP, attention, etc.)
2. Fine-tune hyperparameters
3. Consider end-to-end training

### If embeddings are not useful:
1. Investigate why (embedding visualization)
2. Try different pretraining strategies
3. Consider supervised alternatives
4. Publish failure analysis

### If no cross-flight generalization:
1. Analyze per-flight differences
2. Try domain adaptation techniques
3. Consider per-flight calibration
4. Investigate physical relationships

---

## Documentation

For full context, see:
- `EXECUTIVE_SUMMARY.md` - Overall findings and recommendations
- `docs/STRATIFIED_RESULTS_ANALYSIS.md` - Detailed results analysis
- `docs/STRATIFIED_SPLITTING.md` - Methodology documentation
- `QUICK_REFERENCE.md` - Quick command reference

---

**End of Diagnostic Summary**
EOF

echo " Summary report created: $SUMMARY_FILE"
echo ""

# ============================================================================
# Final Summary
# ============================================================================
echo ""
echo "================================================================================"
echo "DIAGNOSTIC SUITE COMPLETE"
echo "================================================================================"
echo ""
echo "Completed at: $(date)"
echo ""
echo "All results saved to: $DIAG_DIR"
echo ""
echo "Key files to check:"
echo "   $DIAG_DIR/DIAGNOSTIC_SUMMARY.md (overview)"
echo "   $DIAG_DIR/embedding_analysis/ (what MAE learned)"
echo "   $DIAG_DIR/feature_importance/ (what GBDT uses)"
echo "   $DIAG_DIR/ablation_studies/ (feature comparisons)"
echo "   $DIAG_DIR/loo_validation/ (cross-flight test)"
echo ""
echo "Next steps:"
echo "  1. Review DIAGNOSTIC_SUMMARY.md for overview"
echo "  2. Check embedding visualizations for clustering patterns"
echo "  3. Examine feature importance to see if MAE is used"
echo "  4. Compare ablation results (MAE vs angles)"
echo "  5. Review LOO CV for true generalization performance"
echo ""
echo "================================================================================"
