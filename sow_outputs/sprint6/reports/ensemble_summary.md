# Task 2.1: Ensemble Methods - Results Summary

**Generated:** 2025-11-11 20:50:46

## Executive Summary

Target not achieved: Weighted averaging ensemble achieves R² = 0.7391, below the target of 0.74. Improvement over baseline: 1.70%. Further improvements needed through better image feature extraction or alternative fusion strategies.

## Baseline Models Performance

### GBDT (Tabular Features)
- **R²:** 0.7267 ± 0.1121
- **MAE:** 0.119 ± 0.016 km

### CNN (Image Features)
- **R²:** 0.3512 ± 0.0747
- **MAE:** 0.237 ± 0.017 km

## Ensemble Strategies Performance

### 1. Simple Averaging
- **R²:** 0.6616 ± 0.0728
- **MAE:** 0.161 ± 0.014 km
- **Improvement over best baseline:** -0.0650 (-8.95%)

### 2. Weighted Averaging ⭐ BEST
- **R²:** 0.7391 ± 0.0956
- **MAE:** 0.122 ± 0.020 km
- **Improvement over best baseline:** 0.0124 (+1.70%)
- **Optimal Weights:**
  - GBDT: 0.888
  - CNN: 0.112

### 3. Stacking (Ridge Meta-Learner)
- **R²:** 0.7245 ± 0.1148
- **MAE:** 0.118 ± 0.016 km
- **Improvement over best baseline:** -0.0022 (-0.30%)
- **Meta-learner:** Ridge Regression

## Best Ensemble Recommendation

**Strategy:** Weighted Averaging
- **Target R² ≥ 0.74:** ❌ NOT ACHIEVED
- **Final R²:** 0.7391 ± 0.0956
- **Final MAE:** 0.122 km

## Validation Protocol

- **Method:** Stratified K-Fold Cross-Validation
- **Number of Folds:** 5
- **Random Seed:** 42

## Key Insights

1. **Tabular features dominate**: The optimal weights heavily favor GBDT (88.8%), indicating atmospheric/geometric features are more predictive than raw images.

2. **Image model contribution**: While CNN alone performs poorly (R² = 0.3512), it provides complementary information that improves ensemble performance.

3. **Ensemble benefit**: Weighted averaging provides a +1.70% improvement over the GBDT baseline.

4. **Production recommendation**: Use weighted averaging ensemble with learned optimal weights for production deployment.

## Visualizations

All ensemble visualizations are saved in:
`sow_outputs/sprint6/figures/ensemble/`

- `ensemble_performance_comparison.png/pdf` - Overall performance comparison
- `per_fold_performance.png/pdf` - Performance across CV folds
- `ensemble_prediction_scatter.png/pdf` - Prediction quality scatter plots
- `ensemble_error_distributions.png/pdf` - Error distribution analysis
- `ensemble_weight_distribution.png/pdf` - Optimal weight analysis
- `ensemble_improvement_analysis.png/pdf` - Improvement over baseline

---
*Task 2.1 Complete - Ensemble Methods Analysis*
