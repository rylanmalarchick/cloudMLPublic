# Paper Figures Manifest

**Document:** Atmospheric Features Outperform Images for Cloud Base Height Retrieval  
**Author:** Rylan Malarchick  
**Date Created:** 2025-11-13  
**Version:** 1.0.0

## Manifest Purpose

This manifest provides a complete audit trail for all figures used in the preprint, ensuring full transparency and reproducibility.

## File Checksums (SHA256)

```
# Generated: 2025-11-13 21:43 UTC
SHA256 (figure1_model_comparison.png) = 25123c59169862f0bf2d27a1aac7922f27245786c60bcf4072d2c494c5ab798f
SHA256 (figure2_ensemble_performance.png) = 39a34dc9e85d3ad97cf87435355e56f69404aa0d0b15b13c8b63585757cf0098
SHA256 (figure3_feature_importance.png) = 4eac668fcd3eb58af840d1e7d26e240993ad98a22aa9426e8629a4031e6e03dc
SHA256 (figure4_uncertainty_quantification.png) = 6092e98665ea0fde6a7f11dfbd58076562623b864c5b4079296548055059ed5d
SHA256 (figure5_few_shot_learning.png) = f996a1c25df31b8f42e8cd4f29c9ec8eebbc37114649b6ea88c0882181fc4ae4
```

## Figure Mapping

| Paper Ref | Figure Number | Filename | Original Source Path | Section |
|-----------|---------------|----------|---------------------|---------|
| `\ref{fig:model_comparison}` | Figure 1 | `figure1_model_comparison.png` | `results/cbh/figures/paper/figure_model_comparison.png` | 4.1 |
| `\ref{fig:ensemble_performance}` | Figure 2 | `figure2_ensemble_performance.png` | `results/cbh/figures/ensemble/ensemble_performance_comparison.png` | 4.2 |
| `\ref{fig:feature_importance}` | Figure 3 | `figure3_feature_importance.png` | `results/cbh/figures/paper/figure_feature_importance.png` | 4.3 |
| `\ref{fig:uncertainty}` | Figure 4 | `figure4_uncertainty_quantification.png` | `results/cbh/figures/uncertainty/uncertainty_vs_error.png` | 4.4 |
| `\ref{fig:few_shot}` | Figure 5 | `figure5_few_shot_learning.png` | `results/cbh/figures/domain_adaptation/few_shot_learning_curve.png` | 4.5 |

## Data Provenance

### Figure 1: Model Performance Comparison
**Data Sources:**
- GBDT metrics: `sow_outputs/sprint6/reports/validation_report_tabular.json`
- CNN metrics: `sow_outputs/sprint6/reports/validation_report_images.json`
- Ensemble metrics: `sow_outputs/sprint6/reports/ensemble_results.json`

**Verified Against:** `docs/preprint_verification_results.json` (claims 1-10)

**Key Values:**
- GBDT R²: 0.744 ± 0.037 ✓
- CNN R²: 0.320 ± 0.152 ✓
- Weighted Ensemble R²: 0.739 ± 0.096 ✓

---

### Figure 2: Ensemble Performance
**Data Sources:**
- `sow_outputs/sprint6/reports/ensemble_results.json`
- Ensemble strategies: simple_avg, weighted_avg, stacking

**Verified Against:** `docs/preprint_verification_results.json` (claims 8-10)

**Key Values:**
- Simple Average: R²=0.662 ✓
- Weighted Ensemble: R²=0.739, weights=[0.888, 0.112] ✓
- Stacking: R²=0.724 ✓

---

### Figure 3: Feature Importance
**Data Sources:**
- `models/cbh_production/production_config.json` (feature_importance dict)
- SHAP values from production GBDT model

**Verified Against:** Production model configuration (lines 45-64)

**Key Values (Top 5):**
1. d2m: 18.7% ✓
2. t2m: 18.0% ✓
3. moisture_gradient: 7.6% ✓
4. sza_deg: 7.2% ✓
5. blh: 6.2% ✓

---

### Figure 4: Uncertainty Quantification
**Data Sources:**
- `sow_outputs/sprint6/reports/uncertainty_quantification_report.json`
- Quantile regression results (5th, 95th percentiles)

**Verified Against:** `docs/preprint_verification_results.json` (claims 11-13)

**Key Values:**
- Coverage: 77.1% ✓
- Mean interval width: 533.4 m ✓
- Correlation: 0.485 ✓

---

### Figure 5: Few-Shot Learning
**Data Sources:**
- `sow_outputs/sprint6/reports/domain_adaptation_f4_report.json`
- Few-shot experiments: k ∈ {5, 10, 20} samples

**Verified Against:** `docs/preprint_verification_results.json` (claims 14-17)

**Key Values:**
- Baseline LOFO: R²=-0.978 ✓
- 5-shot: R²=-0.528 ± 0.77 ✓
- 10-shot: R²=-0.220 ± 0.18 ✓
- 20-shot: R²=-0.708 ± 0.70 ✓

---

## LaTeX Integration

All figures are referenced in `cloudml_academic_preprint.tex` using:

```latex
\includegraphics[width=0.8\textwidth]{paperfigures/figure<N>_<name>.png}
```

**Compilation Verified:** ✓ PDF successfully compiled on 2025-11-13  
**Figure Pages:** 10-14 (in 22-page document)

## Quality Assurance Checklist

- [x] All 5 figures exist in paperfigures/ directory
- [x] All figures load correctly in LaTeX compilation
- [x] All numerical values cross-verified against source data
- [x] README.md documentation complete
- [x] MANIFEST.md audit trail complete
- [x] File sizes reasonable (165 KB - 476 KB per figure)
- [x] PNG format with lossless compression
- [x] Original source paths documented
- [x] Generation timestamps recorded

## Reproducibility Statement

These figures can be **fully reproduced** by:

1. Running the Sprint 6 validation pipeline from the repository
2. Verification against `docs/preprint_verification_results.json` (18/18 claims pass)
3. Cross-checking with production model config and validation reports

**Reproducibility Level:** ✅ **FULL** - All data sources, scripts, and verification available

---

**Manifest Version:** 1.0.0  
**Last Updated:** 2025-11-13  
**Maintained By:** Rylan Malarchick
