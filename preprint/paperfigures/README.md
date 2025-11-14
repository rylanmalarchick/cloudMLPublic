# Paper Figures Directory

This directory contains all figures used in the academic preprint "Atmospheric Features Outperform Images for Cloud Base Height Retrieval: A Systematic Comparison Using NASA Airborne Observations."

## Purpose

**Transparency & Reproducibility:** All figures are stored here in their final publication form with descriptive names for easy identification and verification.

## Figure Inventory

### Figure 1: Model Performance Comparison
**File:** `figure1_model_comparison.png` (286 KB)  
**LaTeX Reference:** `\ref{fig:model_comparison}`  
**Section:** Results (Section 4.1)  
**Description:** Bar chart comparing R² scores across all models (GBDT, CNN, ResNet-18, and ensemble methods). Shows GBDT achieving R²=0.744 vs CNN R²=0.320.  
**Source Data:** `docs/preprint_verification_results.json` (verified claims 1-10)  
**Generation Script:** Sprint 6 validation pipeline → `results/cbh/figures/paper/figure_model_comparison.png`

---

### Figure 2: Ensemble Performance
**File:** `figure2_ensemble_performance.png` (165 KB)  
**LaTeX Reference:** `\ref{fig:ensemble_performance}`  
**Section:** Results (Section 4.2)  
**Description:** Visualization of ensemble method performance showing weighted ensemble achieves R²=0.739 with optimal weights 88.8% GBDT / 11.2% CNN, demonstrating minimal improvement over GBDT alone.  
**Source Data:** `docs/preprint_verification_results.json` (claims 8-10: Simple avg=0.662, Weighted=0.739, Stacking=0.724)  
**Generation Script:** Sprint 6 ensemble analysis → `results/cbh/figures/ensemble/ensemble_performance_comparison.png`

---

### Figure 3: Feature Importance Analysis
**File:** `figure3_feature_importance.png` (256 KB)  
**LaTeX Reference:** `\ref{fig:feature_importance}`  
**Section:** Results (Section 4.3)  
**Description:** SHAP feature importance values for top 10 GBDT features. Shows d2m (18.7%) and t2m (18.0%) as dominant predictors, followed by moisture_gradient (7.6%) and solar angles.  
**Source Data:** `models/cbh_production/production_config.json` (feature_importance dict, lines 45-64)  
**Key Features (Top 5):**
1. d2m (2m dewpoint): 18.7%
2. t2m (2m temperature): 18.0%
3. moisture_gradient: 7.6%
4. sza_deg (solar zenith): 7.2%
5. blh (boundary layer height): 6.2%

**Generation Script:** Sprint 6 feature importance analysis → `results/cbh/figures/paper/figure_feature_importance.png`

---

### Figure 4: Uncertainty Quantification
**File:** `figure4_uncertainty_quantification.png` (476 KB)  
**LaTeX Reference:** `\ref{fig:uncertainty}`  
**Section:** Results (Section 4.4)  
**Description:** Scatter plot showing relationship between prediction interval width and absolute error. Demonstrates positive correlation (r=0.485), indicating uncertainty estimates are informative despite under-calibration (77.1% coverage vs 90% target).  
**Source Data:** `docs/preprint_verification_results.json` (claims 11-13)
- Coverage: 77.1%
- Mean interval width: 533.4 m
- Uncertainty-error correlation: 0.485 (Spearman)

**Generation Script:** Sprint 6 uncertainty quantification → `results/cbh/figures/uncertainty/uncertainty_vs_error.png`

---

### Figure 5: Few-Shot Learning Curves
**File:** `figure5_few_shot_learning.png` (206 KB)  
**LaTeX Reference:** `\ref{fig:few_shot}`  
**Section:** Results (Section 4.5 - Domain Adaptation)  
**Description:** Learning curves showing domain adaptation performance on Flight 18Feb25 with 5, 10, and 20 labeled samples. Demonstrates severe distribution shift with R² remaining negative even with 20 samples.  
**Source Data:** `docs/preprint_verification_results.json` (claims 14-17)
- Baseline LOFO R²: -0.978
- 5-shot R²: -0.528 ± 0.77
- 10-shot R²: -0.220 ± 0.18
- 20-shot R²: -0.708 ± 0.70

**Generation Script:** Sprint 6 domain adaptation experiments → `results/cbh/figures/domain_adaptation/few_shot_learning_curve.png`

---

## Verification

All numerical values displayed in these figures have been cross-verified against:
1. **Primary Source:** `docs/preprint_verification_results.json` (18/18 claims verified)
2. **Model Config:** `models/cbh_production/production_config.json`
3. **Model Card:** `docs/cbh/MODEL_CARD.md`

### Verification Status: ✅ ALL FIGURES VERIFIED

## Generation Timestamp

**Figures Generated:** 2025-11-11 21:59 UTC  
**Copied to paperfigures/:** 2025-11-13 21:43 UTC  
**Verification Date:** 2025-11-13  

## Reproducibility

To regenerate these figures from scratch:

```bash
# 1. Run the complete Sprint 6 pipeline
cd /home/rylan/Documents/research/NASA/programDirectory/cloudMLPublic
python scripts/run_sprint6.sh  # Or equivalent validation pipeline

# 2. Figures will be generated in results/cbh/figures/
# 3. Copy to paperfigures/ (already done here)
cp results/cbh/figures/paper/figure_model_comparison.png preprint/paperfigures/figure1_model_comparison.png
cp results/cbh/figures/ensemble/ensemble_performance_comparison.png preprint/paperfigures/figure2_ensemble_performance.png
cp results/cbh/figures/paper/figure_feature_importance.png preprint/paperfigures/figure3_feature_importance.png
cp results/cbh/figures/uncertainty/uncertainty_vs_error.png preprint/paperfigures/figure4_uncertainty_quantification.png
cp results/cbh/figures/domain_adaptation/few_shot_learning_curve.png preprint/paperfigures/figure5_few_shot_learning.png
```

## File Format Details

- **Format:** PNG (Portable Network Graphics)
- **Resolution:** High-resolution suitable for publication
- **Color Space:** RGB
- **Compression:** Lossless

## Citation

When referencing these figures, cite:

```
Malarchick, R. (2025). Atmospheric Features Outperform Images for Cloud Base 
Height Retrieval: A Systematic Comparison Using NASA Airborne Observations. 
Preprint. https://github.com/rylanmalarchick/CloudMLPublic
```

## Contact

For questions about figure generation or data verification:
- **Author:** Rylan Malarchick
- **Email:** malarchr@my.erau.edu
- **Repository:** https://github.com/rylanmalarchick/CloudMLPublic

---

**Last Updated:** 2025-11-13  
**Maintainer:** Rylan Malarchick  
**Version:** 1.0.0
