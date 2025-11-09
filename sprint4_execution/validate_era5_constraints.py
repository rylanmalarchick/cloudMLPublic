#!/usr/bin/env python3
"""
Validate ERA5 Physical Constraints
===================================

This script validates whether ERA5-derived atmospheric features (BLH, LCL)
satisfy expected physical constraints relative to CPL ground truth CBH.

Expected constraint: BLH > CBH (boundary layer should contain clouds)

Author: AI Research Assistant
Date: 2025-02-19
Sprint: 4 (Negative Results Analysis)
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr

# Configuration
WP2_FEATURES_PATH = "sow_outputs/wp2_atmospheric/WP2_Features.hdf5"
WP1_FEATURES_PATH = "sow_outputs/wp1_geometric/WP1_Features.hdf5"
OUTPUT_DIR = "sprint4_execution/figures"
OUTPUT_FILE = "era5_constraint_validation.png"


def load_data():
    """Load ERA5 atmospheric features and CPL ground truth CBH."""
    print("Loading WP-2 ERA5 features...")
    with h5py.File(WP2_FEATURES_PATH, "r") as f:
        print(f"  Available datasets: {list(f.keys())}")

        # ERA5 features structure: [n_samples, n_features]
        # Features order (per WP-2 spec): blh, lcl, inversion_height, moisture_gradient,
        #                                  stability_index, t2m, d2m, sp, tcwv
        features = f["features"][:]
        feature_names = f["feature_names"][:] if "feature_names" in f else None

        blh = features[:, 0]  # Boundary layer height (m)
        lcl = features[:, 1]  # Lifting condensation level (m)

        print(f"  Loaded {len(features)} samples with {features.shape[1]} features")
        print(f"  BLH range: {np.min(blh):.0f} - {np.max(blh):.0f} m")
        print(f"  LCL range: {np.min(lcl):.0f} - {np.max(lcl):.0f} m")

    print("\nLoading CPL ground truth CBH...")
    with h5py.File(WP1_FEATURES_PATH, "r") as f:
        print(f"  Available datasets: {list(f.keys())}")

        # CPL CBH is the ground truth target
        cbh_cpl = f["cbh_cpl"][:]

        print(f"  Loaded {len(cbh_cpl)} CPL CBH samples")
        print(f"  CBH range: {np.min(cbh_cpl):.0f} - {np.max(cbh_cpl):.0f} m")
        print(f"  CBH mean: {np.mean(cbh_cpl):.0f} m")
        print(f"  CBH median: {np.median(cbh_cpl):.0f} m")

    # Validate array lengths match
    assert len(blh) == len(lcl) == len(cbh_cpl), (
        f"Array length mismatch: BLH={len(blh)}, LCL={len(lcl)}, CBH={len(cbh_cpl)}"
    )

    return blh, lcl, cbh_cpl


def compute_violations(blh, lcl, cbh_cpl):
    """Compute physical constraint violations."""
    # BLH should be >= CBH (boundary layer contains clouds)
    blh_violations = cbh_cpl > blh
    n_blh_violations = np.sum(blh_violations)
    pct_blh_violations = 100 * n_blh_violations / len(blh)

    # LCL should be approximately equal to CBH for well-mixed convective clouds
    # But LCL can be below or above CBH depending on cloud type
    # For stratiform: CBH often near inversion, not LCL
    # For cumulus: CBH approximately equals LCL
    lcl_violations = cbh_cpl > lcl
    n_lcl_violations = np.sum(lcl_violations)
    pct_lcl_violations = 100 * n_lcl_violations / len(lcl)

    print("\n" + "=" * 60)
    print("PHYSICAL CONSTRAINT VALIDATION RESULTS")
    print("=" * 60)
    print(f"\nBLH > CBH constraint (expected: BLH should contain clouds):")
    print(f"  Violations: {n_blh_violations}/{len(blh)} ({pct_blh_violations:.1f}%)")
    print(f"  Cases where CBH > BLH: {pct_blh_violations:.1f}%")
    if pct_blh_violations > 10:
        print(f"  ⚠️  WARNING: High violation rate suggests BLH is unreliable")

    print(f"\nLCL vs CBH comparison:")
    print(f"  Cases where CBH > LCL: {pct_lcl_violations:.1f}%")
    print(f"  Mean LCL-CBH difference: {np.mean(lcl - cbh_cpl):.0f} m")
    print(f"  Median LCL-CBH difference: {np.median(lcl - cbh_cpl):.0f} m")

    # Compute correlations
    r_blh, p_blh = pearsonr(blh, cbh_cpl)
    r_lcl, p_lcl = pearsonr(lcl, cbh_cpl)

    print(f"\nCorrelations with CPL CBH:")
    print(f"  BLH: r = {r_blh:.4f} (p = {p_blh:.4e})")
    print(f"  LCL: r = {r_lcl:.4f} (p = {p_lcl:.4e})")

    if abs(r_blh) < 0.3 and abs(r_lcl) < 0.3:
        print(
            f"  ⚠️  WARNING: Weak correlations suggest ERA5 features lack predictive power"
        )

    print("=" * 60 + "\n")

    return {
        "blh_violations": n_blh_violations,
        "pct_blh_violations": pct_blh_violations,
        "lcl_violations": n_lcl_violations,
        "pct_lcl_violations": pct_lcl_violations,
        "r_blh": r_blh,
        "p_blh": p_blh,
        "r_lcl": r_lcl,
        "p_lcl": p_lcl,
    }


def create_validation_figure(blh, lcl, cbh_cpl, stats):
    """Create comprehensive validation figure."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Top-left: BLH vs CBH scatter
    ax = axes[0, 0]
    violations = cbh_cpl > blh
    ax.scatter(
        blh[~violations] / 1000,
        cbh_cpl[~violations] / 1000,
        alpha=0.5,
        s=15,
        c="blue",
        label="Valid (BLH > CBH)",
    )
    ax.scatter(
        blh[violations] / 1000,
        cbh_cpl[violations] / 1000,
        alpha=0.7,
        s=15,
        c="red",
        label="Violation (CBH > BLH)",
    )
    ax.plot([0, 10], [0, 10], "k--", linewidth=2, label="1:1 line")
    ax.set_xlabel("ERA5 Boundary Layer Height (km)", fontsize=12)
    ax.set_ylabel("CPL Cloud Base Height (km)", fontsize=12)
    ax.set_title(
        f"BLH vs CBH (Violations: {stats['pct_blh_violations']:.1f}%)\n"
        + f"r = {stats['r_blh']:.4f}",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max(10, np.percentile(blh / 1000, 99)))
    ax.set_ylim(0, max(4, np.percentile(cbh_cpl / 1000, 99)))

    # Top-right: LCL vs CBH scatter
    ax = axes[0, 1]
    violations_lcl = cbh_cpl > lcl
    ax.scatter(
        lcl[~violations_lcl] / 1000,
        cbh_cpl[~violations_lcl] / 1000,
        alpha=0.5,
        s=15,
        c="green",
        label="CBH ≤ LCL",
    )
    ax.scatter(
        lcl[violations_lcl] / 1000,
        cbh_cpl[violations_lcl] / 1000,
        alpha=0.7,
        s=15,
        c="orange",
        label="CBH > LCL",
    )
    ax.plot([0, 10], [0, 10], "k--", linewidth=2, label="1:1 line")
    ax.set_xlabel("Computed Lifting Condensation Level (km)", fontsize=12)
    ax.set_ylabel("CPL Cloud Base Height (km)", fontsize=12)
    ax.set_title(
        f"LCL vs CBH (CBH > LCL: {stats['pct_lcl_violations']:.1f}%)\n"
        + f"r = {stats['r_lcl']:.4f}",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max(4, np.percentile(lcl / 1000, 99)))
    ax.set_ylim(0, max(4, np.percentile(cbh_cpl / 1000, 99)))

    # Bottom-left: BLH-CBH difference histogram
    ax = axes[1, 0]
    diff_blh = (blh - cbh_cpl) / 1000  # km
    ax.hist(diff_blh, bins=50, alpha=0.7, edgecolor="black", color="steelblue")
    ax.axvline(
        x=0,
        color="red",
        linestyle="--",
        linewidth=2.5,
        label="Zero (constraint boundary)",
    )
    ax.axvline(
        x=np.mean(diff_blh),
        color="orange",
        linestyle="-",
        linewidth=2,
        label=f"Mean: {np.mean(diff_blh):.2f} km",
    )
    ax.axvline(
        x=np.median(diff_blh),
        color="purple",
        linestyle=":",
        linewidth=2,
        label=f"Median: {np.median(diff_blh):.2f} km",
    )
    ax.set_xlabel("BLH - CBH (km)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title(
        "Distribution of BLH-CBH Difference\n(Negative = Constraint Violation)",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # Shade violation region
    xlim = ax.get_xlim()
    ax.axvspan(xlim[0], 0, alpha=0.2, color="red", label="Violation region")
    ax.set_xlim(xlim)

    # Bottom-right: Summary statistics text
    ax = axes[1, 1]
    ax.axis("off")

    summary_text = f"""
CONSTRAINT VALIDATION SUMMARY
{"=" * 45}

Dataset: {len(cbh_cpl)} samples from 5 flights

BLH Statistics:
  Range:  {np.min(blh) / 1000:.2f} - {np.max(blh) / 1000:.2f} km
  Mean:   {np.mean(blh) / 1000:.2f} km
  Median: {np.median(blh) / 1000:.2f} km

CBH Statistics:
  Range:  {np.min(cbh_cpl) / 1000:.2f} - {np.max(cbh_cpl) / 1000:.2f} km
  Mean:   {np.mean(cbh_cpl) / 1000:.2f} km
  Median: {np.median(cbh_cpl) / 1000:.2f} km

Constraint: BLH > CBH
  Violations: {stats["blh_violations"]}/{len(cbh_cpl)} ({stats["pct_blh_violations"]:.1f}%)

Correlation with CPL CBH:
  BLH: r = {stats["r_blh"]:.4f} (p = {stats["p_blh"]:.2e})
  LCL: r = {stats["r_lcl"]:.4f} (p = {stats["p_lcl"]:.2e})

INTERPRETATION:
{"─" * 45}
"""

    # Add interpretation based on results
    if stats["pct_blh_violations"] > 20:
        summary_text += "\n⚠️  HIGH VIOLATION RATE\n"
        summary_text += "BLH frequently below CBH suggests:\n"
        summary_text += "• ERA5 BLH parameterization unreliable\n"
        summary_text += "• 25 km spatial resolution too coarse\n"
        summary_text += "• Constraint cannot guide ML model\n"
    else:
        summary_text += "\n✓  Low violation rate\n"
        summary_text += "BLH constraint mostly satisfied\n"

    if abs(stats["r_blh"]) < 0.3 and abs(stats["r_lcl"]) < 0.3:
        summary_text += "\n⚠️  WEAK CORRELATIONS\n"
        summary_text += "ERA5 features have little predictive\n"
        summary_text += "power for cloud base height at this\n"
        summary_text += "spatial scale (25 km grid)\n"

    ax.text(
        0.05,
        0.95,
        summary_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.6),
    )

    plt.suptitle(
        "ERA5 Physical Constraint Validation for CBH Retrieval",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.99])

    # Save figure
    output_path = Path(OUTPUT_DIR) / OUTPUT_FILE
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved figure: {output_path}")

    return fig


def main():
    """Main execution function."""
    print("\n" + "=" * 60)
    print("ERA5 PHYSICAL CONSTRAINT VALIDATION")
    print("=" * 60 + "\n")

    # Load data
    blh, lcl, cbh_cpl = load_data()

    # Compute violations
    stats = compute_violations(blh, lcl, cbh_cpl)

    # Create figure
    fig = create_validation_figure(blh, lcl, cbh_cpl, stats)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nKey Findings:")
    print(f"  • BLH constraint violations: {stats['pct_blh_violations']:.1f}%")
    print(f"  • BLH-CBH correlation: r = {stats['r_blh']:.4f}")
    print(f"  • LCL-CBH correlation: r = {stats['r_lcl']:.4f}")

    if stats["pct_blh_violations"] > 10 or abs(stats["r_blh"]) < 0.3:
        print(f"\n⚠️  WARNING: ERA5 features may not be suitable for CBH prediction")
        print(
            f"   Consider higher-resolution reanalysis (HRRR: 3km) or alternative approach"
        )

    plt.show()


if __name__ == "__main__":
    main()
