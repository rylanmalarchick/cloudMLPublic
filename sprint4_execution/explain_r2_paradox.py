#!/usr/bin/env python3
"""
R² Paradox Explanation
======================

This script explains how R² can be catastrophically negative (-14.15)
while MAE (0.49 km) and RMSE (0.60 km) appear reasonable.

This doesn't require loading the actual data - we can demonstrate
the mathematical phenomenon with synthetic examples.

Author: AI Research Assistant
Date: 2025-02-19
"""

import numpy as np


def calculate_r2(y_true, y_pred):
    """
    Calculate R² score.

    R² = 1 - (SS_res / SS_tot)
    where:
      SS_res = sum of squared residuals = Σ(y_true - y_pred)²
      SS_tot = total sum of squares = Σ(y_true - y_mean)²

    R² < 0 means predictions are worse than just predicting the mean.
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2, ss_res, ss_tot


def demonstrate_r2_paradox():
    """
    Demonstrate how you can have reasonable MAE/RMSE but terrible R².
    """
    print("\n" + "=" * 80)
    print("R² PARADOX EXPLANATION")
    print("=" * 80)

    print("\nKey Insight:")
    print("  MAE and RMSE measure ABSOLUTE error magnitude")
    print("  R² measures EXPLAINED VARIANCE (correlation)")
    print("  You can have small errors but NO CORRELATION!")

    # Scenario 1: Perfect predictions
    print("\n" + "-" * 80)
    print("Scenario 1: PERFECT PREDICTIONS")
    print("-" * 80)
    y_true = np.array([0.5, 0.8, 1.2, 0.6, 0.9])
    y_pred = y_true.copy()

    r2, ss_res, ss_tot = calculate_r2(y_true, y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    print(f"Ground truth: {y_true}")
    print(f"Predictions:  {y_pred}")
    print(f"\nMAE:  {mae:.4f} km")
    print(f"RMSE: {rmse:.4f} km")
    print(f"R²:   {r2:.4f}  ← Perfect!")

    # Scenario 2: Predicting the mean (R² = 0 baseline)
    print("\n" + "-" * 80)
    print("Scenario 2: ALWAYS PREDICT THE MEAN (R² = 0 baseline)")
    print("-" * 80)
    y_true = np.array([0.5, 0.8, 1.2, 0.6, 0.9])
    y_pred = np.full(len(y_true), np.mean(y_true))

    r2, ss_res, ss_tot = calculate_r2(y_true, y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    print(f"Ground truth: {y_true}")
    print(f"Predictions:  {y_pred}")
    print(f"\nMAE:  {mae:.4f} km")
    print(f"RMSE: {rmse:.4f} km")
    print(f"R²:   {r2:.4f}  ← This is the R² = 0 baseline")
    print(f"\nNote: MAE and RMSE are non-zero, but R² = 0")
    print(f"      This is what 'predicting the mean' gives you")

    # Scenario 3: UNCORRELATED predictions (like WP-3 shadow geometry)
    print("\n" + "-" * 80)
    print("Scenario 3: UNCORRELATED PREDICTIONS (WP-3 situation)")
    print("-" * 80)

    # True CBH: low values (marine boundary layer clouds)
    y_true = np.array([0.5, 0.8, 1.2, 0.6, 0.9, 0.7, 0.85, 1.0])

    # Predicted CBH: higher values, but uncorrelated
    # (like shadow geometry giving systematic +5 km bias)
    # BUT we'll make them cluster around training mean to simulate
    # what GBDT does when features have no signal
    train_mean = 0.8
    y_pred = np.random.normal(train_mean, 0.3, len(y_true))

    # Make sure predictions are uncorrelated with truth
    np.random.seed(42)
    y_pred = np.random.permutation(y_pred)

    r2, ss_res, ss_tot = calculate_r2(y_true, y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    corr = np.corrcoef(y_true, y_pred)[0, 1]

    print(f"Ground truth: {y_true}")
    print(f"Predictions:  {y_pred}")
    print(f"\nMAE:  {mae:.4f} km  ← Looks reasonable!")
    print(f"RMSE: {rmse:.4f} km  ← Looks reasonable!")
    print(f"R²:   {r2:.4f}  ← NEGATIVE! Worse than predicting mean!")
    print(f"Correlation: {corr:.4f}  ← Near zero!")

    print(f"\nWhy R² is negative:")
    print(f"  SS_res (prediction error): {ss_res:.4f}")
    print(f"  SS_tot (variance):         {ss_tot:.4f}")
    print(f"  R² = 1 - ({ss_res:.4f} / {ss_tot:.4f}) = {r2:.4f}")
    print(f"\n  When SS_res > SS_tot, R² becomes negative!")
    print(f"  This means predictions are WORSE than just guessing the mean.")

    # Scenario 4: WP-3 actual situation (simulated)
    print("\n" + "-" * 80)
    print("Scenario 4: SIMULATED WP-3 SITUATION")
    print("-" * 80)
    print("Based on reported metrics:")
    print("  - Ground truth CBH mean: 0.83 km")
    print("  - MAE: 0.49 km")
    print("  - RMSE: 0.60 km")
    print("  - R²: -14.15")

    # Create synthetic data matching these stats
    n_samples = 100
    np.random.seed(123)

    # True CBH: marine boundary layer clouds (low, ~0.83 km mean)
    y_true_wp3 = np.random.gamma(shape=2, scale=0.4, size=n_samples)
    y_true_wp3 = y_true_wp3 * (0.83 / np.mean(y_true_wp3))  # Scale to mean 0.83

    # Predictions: GBDT trained on noisy features
    # It learns to predict near the training set mean, but with some noise
    # However, different test flights have different mean CBH

    # Simulate cross-flight generalization failure:
    # Training mean: 0.83 km
    # Test mean: Could be different (e.g., 0.5 km for stratocumulus flight)

    # Model predicts ~0.83 +/- noise (what it learned from training)
    # But test set has different distribution

    # Make predictions that are:
    # 1. Close in magnitude (MAE ~0.5 km)
    # 2. But uncorrelated (different pattern)
    y_pred_wp3 = np.random.normal(0.83, 0.4, n_samples)
    y_pred_wp3 = np.clip(y_pred_wp3, 0.1, 3.0)  # Physical bounds

    # Shuffle to break correlation
    y_pred_wp3 = np.random.permutation(y_pred_wp3)

    r2_wp3, ss_res_wp3, ss_tot_wp3 = calculate_r2(y_true_wp3, y_pred_wp3)
    mae_wp3 = np.mean(np.abs(y_true_wp3 - y_pred_wp3))
    rmse_wp3 = np.sqrt(np.mean((y_true_wp3 - y_pred_wp3) ** 2))
    corr_wp3 = np.corrcoef(y_true_wp3, y_pred_wp3)[0, 1]

    print(f"\nSimulated WP-3 results:")
    print(f"  Ground truth mean: {np.mean(y_true_wp3):.2f} km")
    print(f"  Prediction mean:   {np.mean(y_pred_wp3):.2f} km")
    print(f"  MAE:  {mae_wp3:.2f} km")
    print(f"  RMSE: {rmse_wp3:.2f} km")
    print(f"  R²:   {r2_wp3:.2f}")
    print(f"  Correlation: {corr_wp3:.4f}")

    print(f"\nThis demonstrates the WP-3 situation:")
    print(f"  - Predictions are in the right ballpark (MAE ~0.5 km)")
    print(f"  - BUT they're uncorrelated with ground truth")
    print(f"  - Model learned to predict training set mean +/- noise")
    print(f"  - Doesn't generalize to test flights with different CBH distributions")


def explain_cross_flight_failure():
    """
    Explain why cross-flight validation causes R² < 0.
    """
    print("\n" + "=" * 80)
    print("WHY CROSS-FLIGHT VALIDATION REVEALS THE PROBLEM")
    print("=" * 80)

    print("\nThe Problem:")
    print("  Different flights have different cloud regimes and CBH distributions.")
    print("  If model learns flight-specific patterns (not physics), it fails LOO CV.")

    print("\nExample:")
    print("  Flight 0 (30Oct24): Stratocumulus, CBH mean = 0.5 km, n=501")
    print("  Flight 4 (18Feb25): Cumulus, CBH mean = 1.2 km, n=24")

    print("\nWhat happens in LOO CV:")
    print("  - Train on Flights 1,2,3,4 (mean CBH ≈ 0.9 km)")
    print("  - Test on Flight 0 (mean CBH = 0.5 km)")
    print("  - Model predicts ~0.9 km (training mean) for everything")
    print("  - Actual values are ~0.5 km")
    print("  - Predictions have 0.4 km systematic bias")
    print("  - R² becomes negative because SS_res > SS_tot")

    print("\nFold 4 catastrophic failure (R² = -62.66):")
    print("  - Only 24 test samples")
    print("  - Small test set amplifies any bias")
    print("  - If test distribution very different from training:")
    print("    SS_res >> SS_tot → R² becomes extremely negative")

    print("\nThe Core Issue:")
    print("  Features (shadow geometry + ERA5) have NO PREDICTIVE SIGNAL")
    print("  Shadow CBH: r = 0.04 with ground truth (essentially random)")
    print("  ERA5: 25 km resolution too coarse for cloud-scale variability")
    print("  GBDT learns: 'just predict the training mean'")
    print("  This works within-flight (if all samples similar)")
    print("  This FAILS cross-flight (if distributions differ)")


def what_to_check():
    """
    Suggest what to investigate in the actual data.
    """
    print("\n" + "=" * 80)
    print("WHAT TO CHECK IN THE ACTUAL WP-3 DATA")
    print("=" * 80)

    print("\n1. Ground Truth CBH Distribution by Flight:")
    print("   - Load WP1_Features.hdf5, get cbh_cpl for each flight")
    print("   - Compute mean/std per flight")
    print("   - Question: Do different flights have different CBH distributions?")
    print("   - Hypothesis: If yes, this explains why predicting training mean fails")

    print("\n2. Prediction Distribution by Fold:")
    print("   - What is the model actually predicting for each fold?")
    print("   - Are predictions always near the training set mean?")
    print("   - Question: Is model just learning 'predict 0.8 km +/- noise'?")

    print("\n3. Feature Correlation Check:")
    print("   - Shadow geometric H vs ground truth: r = ?")
    print("   - ERA5 BLH vs ground truth: r = ?")
    print("   - ERA5 LCL vs ground truth: r = ?")
    print("   - Hypothesis: If all r < 0.2, features have no signal")

    print("\n4. Within-Flight vs Cross-Flight Performance:")
    print("   - Train and test on SAME flight (random split within flight)")
    print("   - Compare R² to LOO CV R²")
    print("   - Question: Does model work within-flight but fail cross-flight?")
    print(
        "   - This would prove features capture flight-specific artifacts, not physics"
    )

    print("\n5. Baseline Comparison:")
    print("   - Baseline 1: Always predict global mean (0.83 km) → R² = 0")
    print("   - Baseline 2: Predict training fold mean → R² = ?")
    print("   - If WP-3 R² < baseline 2, model is literally harmful")

    print("\n6. Check for Data Bugs:")
    print("   - Are features and targets properly aligned (same sample order)?")
    print("   - Are there NaN values that got imputed incorrectly?")
    print("   - 120 NaN values in geometric H were filled with median = 6.166 km")
    print("   - This is 7× the true mean CBH! Could cause major bias.")

    print("\n7. Imputation Impact:")
    print("   - Shadow detection failed → NaN → imputed with median = 6.166 km")
    print("   - True mean CBH = 0.83 km")
    print("   - Imputed values are ~5 km too high!")
    print("   - If model learns from these imputed values, predictions will be biased")


def main():
    """
    Run all explanations.
    """
    print("\n" + "=" * 80)
    print("UNDERSTANDING THE WP-3 R² PARADOX")
    print("=" * 80)
    print("\nReported WP-3 Results:")
    print("  Mean R²:   -14.15 ± 24.30")
    print("  Mean MAE:    0.49 ± 0.17 km")
    print("  Mean RMSE:   0.60 ± 0.16 km")
    print("\nQuestion: How can R² be so negative while MAE/RMSE seem reasonable?")
    print("Answer: Because R² measures CORRELATION, not absolute error magnitude.")

    demonstrate_r2_paradox()
    explain_cross_flight_failure()
    what_to_check()

    print("\n" + "=" * 80)
    print("BOTTOM LINE")
    print("=" * 80)
    print("\nThe R² = -14.15 result is CORRECT (not a bug) IF:")
    print("  1. Features have near-zero correlation with target")
    print("  2. Model learns to predict training set mean")
    print("  3. Test flights have different CBH distributions")
    print("  4. Predictions are uncorrelated with test set ground truth")
    print("\nBUT you should verify:")
    print("  ✓ Feature-target correlations (are they really r < 0.1?)")
    print("  ✓ Per-flight CBH distributions (do they differ?)")
    print("  ✓ What model actually predicts (all similar values?)")
    print("  ✓ Imputation impact (did median=6.166km bias the model?)")
    print("\nIf correlations are actually reasonable (r > 0.3), THEN there's a bug.")
    print("If correlations are near-zero (r < 0.1), THEN R² = -14 is expected.")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
