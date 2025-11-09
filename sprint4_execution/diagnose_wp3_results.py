#!/usr/bin/env python3
"""
WP-3 Results Diagnostic
========================

Deep dive into WP-3 results to understand why R² is negative
but MAE/RMSE seem reasonable.

Investigates:
1. What features are actually being used?
2. What are the predictions vs ground truth?
3. Is R² calculation correct?
4. What's the variance structure of the data?

Author: AI Research Assistant
Date: 2025-02-19
"""

import json
import numpy as np
import h5py
from pathlib import Path
from scipy.stats import pearsonr

# Configuration
WP1_PATH = "sow_outputs/wp1_geometric/WP1_Features.hdf5"
WP2_PATH = "sow_outputs/wp2_atmospheric/WP2_Features.hdf5"
WP3_REPORT = "sow_outputs/wp3_baseline/WP3_Report.json"


def load_wp3_report():
    """Load WP3 report JSON."""
    with open(WP3_REPORT, "r") as f:
        return json.load(f)


def inspect_features():
    """Inspect the actual feature values."""
    print("\n" + "=" * 80)
    print("FEATURE INSPECTION")
    print("=" * 80)

    # WP1 Geometric Features
    print("\n--- WP1 Geometric Features ---")
    with h5py.File(WP1_PATH, "r") as f:
        print(f"Datasets: {list(f.keys())}")

        if "derived_geometric_H" in f:
            geo_h = f["derived_geometric_H"][:]
            print(f"\nderived_geometric_H (shadow-based CBH):")
            print(f"  Shape: {geo_h.shape}")
            print(f"  Range: [{np.nanmin(geo_h):.2f}, {np.nanmax(geo_h):.2f}] m")
            print(
                f"  Mean: {np.nanmean(geo_h):.2f} m ({np.nanmean(geo_h) / 1000:.2f} km)"
            )
            print(f"  Median: {np.nanmedian(geo_h):.2f} m")
            print(f"  Std: {np.nanstd(geo_h):.2f} m")
            print(f"  NaN count: {np.sum(np.isnan(geo_h))}/{len(geo_h)}")

        if "shadow_length_pixels" in f:
            shadow_len = f["shadow_length_pixels"][:]
            print(f"\nshadow_length_pixels:")
            print(
                f"  Range: [{np.nanmin(shadow_len):.2f}, {np.nanmax(shadow_len):.2f}]"
            )
            print(f"  Mean: {np.nanmean(shadow_len):.2f}")

        if "shadow_detection_confidence" in f:
            confidence = f["shadow_detection_confidence"][:]
            print(f"\nshadow_detection_confidence:")
            print(
                f"  Range: [{np.nanmin(confidence):.4f}, {np.nanmax(confidence):.4f}]"
            )
            print(f"  Mean: {np.nanmean(confidence):.4f}")

        if "cbh_cpl" in f:
            cbh_true = f["cbh_cpl"][:]
            print(f"\nGround Truth CBH (cbh_cpl):")
            print(f"  Range: [{np.min(cbh_true):.2f}, {np.max(cbh_true):.2f}] m")
            print(
                f"  Mean: {np.mean(cbh_true):.2f} m ({np.mean(cbh_true) / 1000:.2f} km)"
            )
            print(f"  Median: {np.median(cbh_true):.2f} m")
            print(f"  Std: {np.std(cbh_true):.2f} m")

    # WP2 Atmospheric Features
    print("\n--- WP2 Atmospheric Features ---")
    with h5py.File(WP2_PATH, "r") as f:
        print(f"Datasets: {list(f.keys())}")

        if "features" in f:
            atm_features = f["features"][:]
            print(f"\nAtmospheric features shape: {atm_features.shape}")
            print(f"Features: {atm_features.shape[1]}")

            if "feature_names" in f:
                names = f["feature_names"][:]
                names = [n.decode() if isinstance(n, bytes) else n for n in names]
                print(f"Feature names: {names}")

                for i, name in enumerate(names):
                    feat = atm_features[:, i]
                    print(f"\n{name}:")
                    print(f"  Range: [{np.nanmin(feat):.2f}, {np.nanmax(feat):.2f}]")
                    print(f"  Mean: {np.nanmean(feat):.2f}")
                    print(f"  NaN count: {np.sum(np.isnan(feat))}/{len(feat)}")


def analyze_r2_calculation():
    """Manually verify R² calculation."""
    print("\n" + "=" * 80)
    print("R² CALCULATION ANALYSIS")
    print("=" * 80)

    report = load_wp3_report()

    print(f"\nReported metrics from WP3:")
    print(f"  Mean R²: {report['aggregate_metrics']['mean_r2']:.4f}")
    print(f"  Mean MAE: {report['aggregate_metrics']['mean_mae_km']:.4f} km")
    print(f"  Mean RMSE: {report['aggregate_metrics']['mean_rmse_km']:.4f} km")

    print(f"\nPer-fold breakdown:")
    for fold in report["folds"]:
        fid = fold["fold_id"]
        flight = fold["test_flight"]
        r2 = fold["r2"]
        mae = fold["mae_km"]
        rmse = fold["rmse_km"]
        n_test = fold["n_test"]

        print(f"\nFold {fid} ({flight}, n={n_test}):")
        print(f"  R²: {r2:.4f}")
        print(f"  MAE: {mae:.4f} km")
        print(f"  RMSE: {rmse:.4f} km")

        # R² = 1 - (SS_res / SS_tot)
        # If R² is negative, SS_res > SS_tot
        # This means variance of residuals > variance of targets

        # From RMSE, we can get SS_res (approximately)
        # SS_res = n_test * RMSE²
        ss_res_approx = n_test * (rmse**2)

        # If R² = 1 - SS_res/SS_tot, then:
        # SS_tot = SS_res / (1 - R²)
        if r2 != 1:
            ss_tot_approx = ss_res_approx / (1 - r2)
            var_y_approx = ss_tot_approx / n_test
            std_y_approx = np.sqrt(var_y_approx)

            print(f"  Implied target variance: {var_y_approx:.4f} km²")
            print(f"  Implied target std: {std_y_approx:.4f} km")

            if r2 < 0:
                print(f"  ⚠️ Negative R² means: SS_res > SS_tot")
                print(f"     Model predictions worse than predicting the mean!")


def compare_features_to_target():
    """Compare individual features to target CBH."""
    print("\n" + "=" * 80)
    print("FEATURE vs TARGET CORRELATIONS")
    print("=" * 80)

    # Load ground truth
    with h5py.File(WP1_PATH, "r") as f:
        cbh_true = f["cbh_cpl"][:] / 1000  # Convert to km

    print(f"\nGround truth CBH statistics:")
    print(f"  Range: [{np.min(cbh_true):.3f}, {np.max(cbh_true):.3f}] km")
    print(f"  Mean: {np.mean(cbh_true):.3f} km")
    print(f"  Std: {np.std(cbh_true):.3f} km")

    # Check geometric feature correlation
    with h5py.File(WP1_PATH, "r") as f:
        if "derived_geometric_H" in f:
            geo_h = f["derived_geometric_H"][:] / 1000  # Convert to km

            # Remove NaN for correlation
            valid = ~np.isnan(geo_h)
            if np.sum(valid) > 10:
                r, p = pearsonr(geo_h[valid], cbh_true[valid])
                bias = np.mean(geo_h[valid] - cbh_true[valid])
                mae = np.mean(np.abs(geo_h[valid] - cbh_true[valid]))

                print(f"\nGeometric H (shadow-based CBH):")
                print(f"  Valid samples: {np.sum(valid)}/{len(geo_h)}")
                print(f"  Correlation with true CBH: r = {r:.4f} (p={p:.4e})")
                print(f"  Bias: {bias:.3f} km")
                print(f"  MAE: {mae:.3f} km")

                if abs(r) < 0.1:
                    print(f"  ⚠️ WARNING: Essentially uncorrelated!")
                if abs(bias) > 1.0:
                    print(f"  ⚠️ WARNING: Large systematic bias!")

    # Check atmospheric features
    with h5py.File(WP2_PATH, "r") as f:
        if "features" in f and "feature_names" in f:
            atm_features = f["features"][:]
            names = f["feature_names"][:]
            names = [n.decode() if isinstance(n, bytes) else n for n in names]

            print(f"\nAtmospheric features:")
            for i, name in enumerate(names):
                feat = atm_features[:, i] / 1000  # Convert to km if height

                # Remove NaN
                valid = ~np.isnan(feat)
                if np.sum(valid) > 10:
                    r, p = pearsonr(feat[valid], cbh_true[valid])
                    print(f"  {name}: r = {r:.4f} (p={p:.4e})")


def check_within_vs_across_flight():
    """Check if problem is cross-flight generalization."""
    print("\n" + "=" * 80)
    print("WITHIN-FLIGHT vs CROSS-FLIGHT PERFORMANCE")
    print("=" * 80)

    # This would require re-running training on each flight individually
    # For now, just note what the LOO CV results tell us

    report = load_wp3_report()

    print("\nLOO CV tests CROSS-FLIGHT generalization:")
    print("  Train on 4 flights, test on 1 held-out flight")
    print("  Negative R² means model doesn't generalize across flights")
    print("\nPossible explanations:")
    print("  1. Features are flight-specific (e.g., solar angle varies)")
    print("  2. Target distribution varies by flight")
    print("  3. Features contain no generalizable signal")

    # Check target distribution by flight
    print("\nTarget CBH distribution by fold:")
    for fold in report["folds"]:
        fid = fold["fold_id"]
        flight = fold["test_flight"]
        n_test = fold["n_test"]
        rmse = fold["rmse_km"]

        # If we assume predictions are around the training set mean,
        # and RMSE is high, the test set might have very different mean

        print(f"  Fold {fid} ({flight}): n={n_test}, RMSE={rmse:.3f} km")


def main():
    """Run all diagnostics."""
    print("\n" + "=" * 80)
    print("WP-3 RESULTS DIAGNOSTIC")
    print("=" * 80)

    print("\nObjective: Understand why R² is negative but MAE/RMSE seem reasonable")

    # 1. Inspect features
    inspect_features()

    # 2. Analyze R² calculation
    analyze_r2_calculation()

    # 3. Compare features to target
    compare_features_to_target()

    # 4. Check cross-flight performance
    check_within_vs_across_flight()

    print("\n" + "=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80)

    print("\n🔍 Key Questions to Answer:")
    print("  1. Is the shadow-based CBH actually being used as a feature?")
    print("  2. What's its correlation with true CBH? (Expected: r ≈ 0.04)")
    print("  3. Do ERA5 features have any predictive power?")
    print("  4. What's the target variance within each test flight?")
    print("  5. Are predictions just predicting the training set mean?")

    print("\n💡 Hypothesis:")
    print("  - If geometric H has r ≈ 0 and large bias, it's just noise")
    print("  - If ERA5 features have r < 0.3, they lack signal")
    print("  - Model learns training set mean, fails on different test flights")
    print("  - Negative R² = predictions worse than just predicting test mean")


if __name__ == "__main__":
    main()
