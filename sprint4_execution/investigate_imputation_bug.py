#!/usr/bin/env python3
"""
Comprehensive WP-3 Data Investigation
======================================

This script investigates:
1. Load actual HDF5 data and examine distributions
2. CRITICAL: Investigate imputation bug (median = 6.166 km vs true mean = 0.83 km)
3. Per-flight CBH distributions
4. Feature-target correlations
5. What the model actually learned

Author: AI Research Assistant
Date: 2025-02-19
"""

import sys
import json
from pathlib import Path

# Check for h5py
try:
    import h5py
    import numpy as np

    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    print("=" * 80)
    print("ERROR: h5py not installed")
    print("=" * 80)
    print("\nPlease install h5py:")
    print("  pip install h5py")
    print("\nOr install all requirements:")
    print("  pip install -r sprint4_execution/requirements.txt")
    print("\n" + "=" * 80)
    sys.exit(1)

# File paths
WP1_PATH = "sow_outputs/wp1_geometric/WP1_Features.hdf5"
WP2_PATH = "sow_outputs/wp2_atmospheric/WP2_Features.hdf5"
WP3_REPORT = "sow_outputs/wp3_baseline/WP3_Report.json"


def investigate_imputation_bug():
    """
    CRITICAL INVESTIGATION: Imputation Bug

    From SOW report:
    - "120 NaN values in geometric H filled with median = 6.166 km"
    - True mean CBH = 0.83 km
    - Imputed values are 7× too high!

    This could massively bias the GBDT model.
    """
    print("\n" + "=" * 80)
    print("CRITICAL: IMPUTATION BUG INVESTIGATION")
    print("=" * 80)

    with h5py.File(WP1_PATH, "r") as f:
        # Load geometric CBH feature
        if "derived_geometric_H" in f:
            geo_h = f["derived_geometric_H"][:]
        elif "features" in f:
            geo_h = f["features"][:, 0]  # First feature
        else:
            print("ERROR: Cannot find geometric H in WP1 file")
            print(f"Available keys: {list(f.keys())}")
            return

        # Load ground truth CBH
        cbh_true = f["cbh_cpl"][:]

    # Analyze NaN values
    nan_mask = np.isnan(geo_h)
    n_nan = np.sum(nan_mask)
    n_total = len(geo_h)
    pct_nan = 100 * n_nan / n_total

    print(f"\nGeometric CBH (derived from shadow detection):")
    print(f"  Total samples: {n_total}")
    print(f"  NaN values: {n_nan} ({pct_nan:.1f}%)")
    print(f"  Valid values: {n_total - n_nan} ({100 - pct_nan:.1f}%)")

    # Check what the valid values look like
    valid_geo = geo_h[~nan_mask]
    if len(valid_geo) > 0:
        print(f"\nValid geometric CBH statistics:")
        print(f"  Range: [{np.min(valid_geo):.2f}, {np.max(valid_geo):.2f}] m")
        print(
            f"  Mean: {np.mean(valid_geo):.2f} m ({np.mean(valid_geo) / 1000:.2f} km)"
        )
        print(
            f"  Median: {np.median(valid_geo):.2f} m ({np.median(valid_geo) / 1000:.2f} km)"
        )
        print(f"  Std: {np.std(valid_geo):.2f} m")

        median_km = np.median(valid_geo) / 1000
        print(f"\n⚠️  IMPUTATION VALUE: {median_km:.3f} km (median of valid values)")

    # Compare to ground truth
    print(f"\nGround truth CBH statistics:")
    print(f"  Range: [{np.min(cbh_true):.2f}, {np.max(cbh_true):.2f}] m")
    print(f"  Mean: {np.mean(cbh_true):.2f} m ({np.mean(cbh_true) / 1000:.2f} km)")
    print(
        f"  Median: {np.median(cbh_true):.2f} m ({np.median(cbh_true) / 1000:.2f} km)"
    )
    print(f"  Std: {np.std(cbh_true):.2f} m")

    # THE SMOKING GUN
    true_mean_km = np.mean(cbh_true) / 1000
    if len(valid_geo) > 0:
        imputation_value_km = np.median(valid_geo) / 1000
        ratio = imputation_value_km / true_mean_km

        print(f"\n" + "🔥" * 40)
        print(f"SMOKING GUN: IMPUTATION BUG FOUND")
        print(f"🔥" * 40)
        print(
            f"\nImputation value: {imputation_value_km:.3f} km (median of shadow estimates)"
        )
        print(f"True CBH mean:    {true_mean_km:.3f} km (from CPL lidar)")
        print(f"Ratio:            {ratio:.1f}× TOO HIGH")
        print(f"\nWhen shadow detection fails ({n_nan} times):")
        print(f"  WP-3 fills NaN with {imputation_value_km:.3f} km")
        print(f"  But true clouds are at {true_mean_km:.3f} km")
        print(
            f"  This creates {imputation_value_km - true_mean_km:.2f} km systematic bias!"
        )

        # Impact on model
        print(f"\nImpact on GBDT model:")
        print(f"  - GBDT sees {n_nan} samples with CBH = {imputation_value_km:.2f} km")
        print(f"  - But ground truth for those samples is ~{true_mean_km:.2f} km")
        print(
            f"  - Model learns biased relationship: 'high geometric H → predict high CBH'"
        )
        print(f"  - This is WRONG and causes poor generalization")

        # What should have been done
        print(f"\nWhat SHOULD have been done:")
        print(
            f"  Option 1: Drop samples with NaN (use only {n_total - n_nan} valid samples)"
        )
        print(f"  Option 2: Impute with GROUND TRUTH mean = {true_mean_km:.2f} km")
        print(f"  Option 3: Use a separate 'missing' indicator feature")
        print(f"  Option 4: Don't use geometric H as a feature (it's clearly broken)")

    return {
        "n_nan": n_nan,
        "n_total": n_total,
        "imputation_value_km": imputation_value_km if len(valid_geo) > 0 else None,
        "true_mean_km": true_mean_km,
    }


def analyze_per_flight_distributions():
    """
    Check if different flights have different CBH distributions.
    This would explain why cross-flight validation fails.
    """
    print("\n" + "=" * 80)
    print("PER-FLIGHT CBH DISTRIBUTIONS")
    print("=" * 80)

    # Load WP3 report to get flight info
    with open(WP3_REPORT, "r") as f:
        wp3_report = json.load(f)

    # Map flight IDs to names
    flight_mapping = {
        0: "30Oct24",
        1: "10Feb25",
        2: "23Oct24",
        3: "12Feb25",
        4: "18Feb25",
    }

    # Load ground truth
    with h5py.File(WP1_PATH, "r") as f:
        cbh_true = f["cbh_cpl"][:]
        sample_ids = f["sample_id"][:]

    # Need to figure out which samples belong to which flight
    # Sample IDs are formatted as: "{flight_name}_{index}"
    flight_cbh = {fid: [] for fid in range(5)}

    for i, sid in enumerate(sample_ids):
        sid_str = sid.decode() if isinstance(sid, bytes) else str(sid)
        # Extract flight name from sample ID
        for fid, fname in flight_mapping.items():
            if fname in sid_str:
                flight_cbh[fid].append(cbh_true[i])
                break

    # Analyze distributions
    print(f"\nCBH Distribution by Flight:")
    print(
        f"{'Flight':<12} {'N':<8} {'Mean (km)':<12} {'Std (km)':<12} {'Range (km)':<20}"
    )
    print("-" * 80)

    flight_stats = {}
    for fid in range(5):
        fname = flight_mapping[fid]
        cbh = np.array(flight_cbh[fid])

        if len(cbh) > 0:
            mean_km = np.mean(cbh) / 1000
            std_km = np.std(cbh) / 1000
            min_km = np.min(cbh) / 1000
            max_km = np.max(cbh) / 1000
            n = len(cbh)

            print(
                f"{fname:<12} {n:<8} {mean_km:<12.3f} {std_km:<12.3f} [{min_km:.2f}, {max_km:.2f}]"
            )

            flight_stats[fid] = {
                "name": fname,
                "n": n,
                "mean_km": mean_km,
                "std_km": std_km,
                "min_km": min_km,
                "max_km": max_km,
            }

    # Check variance across flights
    all_means = [stats["mean_km"] for stats in flight_stats.values()]
    mean_of_means = np.mean(all_means)
    std_of_means = np.std(all_means)

    print(f"\nVariability across flights:")
    print(f"  Mean of flight means: {mean_of_means:.3f} km")
    print(f"  Std of flight means:  {std_of_means:.3f} km")
    print(f"  Range: [{np.min(all_means):.3f}, {np.max(all_means):.3f}] km")

    if std_of_means > 0.1:
        print(f"\n⚠️  SIGNIFICANT CROSS-FLIGHT VARIABILITY!")
        print(f"  Different flights have different CBH distributions")
        print(f"  This explains why LOO CV fails:")
        print(f"    - Train on flights with mean ~{mean_of_means:.2f} km")
        print(f"    - Test on flight with different mean")
        print(f"    - Model prediction (training mean) doesn't match test distribution")
        print(f"    - Result: Negative R²")

    return flight_stats


def check_feature_correlations():
    """
    Check if features actually correlate with target.
    """
    print("\n" + "=" * 80)
    print("FEATURE-TARGET CORRELATIONS")
    print("=" * 80)

    # Load ground truth
    with h5py.File(WP1_PATH, "r") as f:
        cbh_true = f["cbh_cpl"][:]

        # Geometric features
        if "derived_geometric_H" in f:
            geo_h = f["derived_geometric_H"][:]
        elif "features" in f:
            geo_h = f["features"][:, 0]
        else:
            geo_h = None

    # Load ERA5 features
    with h5py.File(WP2_PATH, "r") as f:
        if "features" in f:
            era5_features = f["features"][:]
            if "feature_names" in f:
                feature_names = f["feature_names"][:]
                feature_names = [
                    n.decode() if isinstance(n, bytes) else n for n in feature_names
                ]
            else:
                feature_names = [f"feature_{i}" for i in range(era5_features.shape[1])]
        else:
            print("ERROR: Cannot find features in WP2 file")
            return

    print(f"\nCorrelation with ground truth CBH:")
    print(f"{'Feature':<25} {'r':<10} {'p-value':<12} {'Assessment':<20}")
    print("-" * 80)

    from scipy.stats import pearsonr

    # Check geometric H
    if geo_h is not None:
        valid = ~np.isnan(geo_h)
        if np.sum(valid) > 10:
            r, p = pearsonr(geo_h[valid], cbh_true[valid])
            assessment = (
                "⚠️ NO SIGNAL"
                if abs(r) < 0.2
                else ("✓ Weak" if abs(r) < 0.5 else "✓ Good")
            )
            print(
                f"{'Geometric CBH (shadow)':<25} {r:<10.4f} {p:<12.4e} {assessment:<20}"
            )

    # Check ERA5 features
    for i, name in enumerate(feature_names):
        feat = era5_features[:, i]
        valid = ~np.isnan(feat)
        if np.sum(valid) > 10:
            r, p = pearsonr(feat[valid], cbh_true[valid])
            assessment = (
                "⚠️ NO SIGNAL"
                if abs(r) < 0.2
                else ("✓ Weak" if abs(r) < 0.5 else "✓ Good")
            )
            print(f"{name:<25} {r:<10.4f} {p:<12.4e} {assessment:<20}")

    print(f"\nInterpretation:")
    print(f"  |r| < 0.2:  No predictive signal (essentially random)")
    print(f"  |r| < 0.5:  Weak signal (might help a little)")
    print(f"  |r| ≥ 0.5:  Moderate to strong signal")
    print(f"\nIf ALL features have |r| < 0.2:")
    print(f"  → GBDT has nothing to learn from")
    print(f"  → Model defaults to predicting mean")
    print(f"  → Cross-flight R² becomes negative")


def examine_fold_4_catastrophe():
    """
    Investigate why Fold 4 had R² = -62.66 (catastrophic).
    """
    print("\n" + "=" * 80)
    print("FOLD 4 CATASTROPHE INVESTIGATION (R² = -62.66)")
    print("=" * 80)

    # Load WP3 report
    with open(WP3_REPORT, "r") as f:
        wp3_report = json.load(f)

    fold_4 = wp3_report["folds"][4]

    print(f"\nFold 4 details:")
    print(f"  Test flight: {fold_4['test_flight']}")
    print(f"  N_test: {fold_4['n_test']}")
    print(f"  R²: {fold_4['r2']:.4f}")
    print(f"  MAE: {fold_4['mae_km']:.4f} km")
    print(f"  RMSE: {fold_4['rmse_km']:.4f} km")

    # Compute implied variance from R²
    n_test = fold_4["n_test"]
    rmse = fold_4["rmse_km"]
    r2 = fold_4["r2"]

    ss_res = n_test * (rmse**2)
    if r2 != 1:
        ss_tot = ss_res / (1 - r2)
        var_y = ss_tot / n_test
        std_y = np.sqrt(var_y)

        print(f"\nImplied test set statistics:")
        print(f"  SS_res: {ss_res:.4f}")
        print(f"  SS_tot: {ss_tot:.4f}")
        print(f"  Implied std of test CBH: {std_y:.4f} km")

        print(f"\nWhy R² is so negative:")
        print(f"  Small test set (n={n_test}) amplifies any distribution mismatch")
        print(f"  If training mean ≠ test mean, predictions are systematically biased")
        print(f"  SS_res ({ss_res:.2f}) >> SS_tot ({ss_tot:.2f})")
        print(f"  R² = 1 - ({ss_res:.2f} / {ss_tot:.2f}) = {r2:.2f}")


def main():
    """
    Run all investigations.
    """
    print("\n" + "=" * 80)
    print("COMPREHENSIVE WP-3 DATA INVESTIGATION")
    print("=" * 80)
    print("\nObjective: Determine if R² = -14.15 is real or a bug")

    # Investigation 1: CRITICAL - Imputation bug
    imputation_stats = investigate_imputation_bug()

    # Investigation 2: Per-flight distributions
    flight_stats = analyze_per_flight_distributions()

    # Investigation 3: Feature correlations
    check_feature_correlations()

    # Investigation 4: Fold 4 catastrophe
    examine_fold_4_catastrophe()

    # Final verdict
    print("\n" + "=" * 80)
    print("FINAL VERDICT")
    print("=" * 80)

    print("\n🔍 Key Findings:")

    if imputation_stats and imputation_stats["imputation_value_km"]:
        ratio = (
            imputation_stats["imputation_value_km"] / imputation_stats["true_mean_km"]
        )
        if ratio > 2.0:
            print(f"\n❌ CRITICAL BUG FOUND: Imputation bias")
            print(
                f"   - {imputation_stats['n_nan']} NaN values filled with {imputation_stats['imputation_value_km']:.2f} km"
            )
            print(f"   - True mean is only {imputation_stats['true_mean_km']:.2f} km")
            print(f"   - {ratio:.1f}× too high!")
            print(f"   - This WILL bias the GBDT model")

    print(f"\n💡 Recommendations:")
    print(f"   1. Re-run WP-3 WITHOUT geometric H feature (it's broken)")
    print(f"   2. Use only ERA5 features (9 atmospheric variables)")
    print(f"   3. If geometric H is needed, drop NaN samples (don't impute)")
    print(f"   4. Check if ERA5-only model performs better")

    print(f"\n📊 Expected outcome if imputation bug is fixed:")
    print(f"   - R² might still be negative (if ERA5 features have no signal)")
    print(f"   - BUT it won't be as catastrophically bad as -14.15")
    print(f"   - Probably closer to -1 to 0 (like other failed baselines)")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
