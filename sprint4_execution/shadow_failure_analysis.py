#!/usr/bin/env python3
"""
Shadow Geometry Failure Analysis
=================================

This script analyzes why shadow-based geometric CBH estimation failed,
including scatter plots, residual analysis, and Bland-Altman plots.

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
WP1_FEATURES_PATH = "sow_outputs/wp1_geometric/WP1_Features.hdf5"
OUTPUT_DIR = "sprint4_execution/figures"
OUTPUT_FILE = "shadow_geometry_failure.png"


def load_data():
    """Load WP-1 geometric features and CPL ground truth."""
    print("Loading WP-1 geometric features...")
    with h5py.File(WP1_FEATURES_PATH, "r") as f:
        print(f"  Available datasets: {list(f.keys())}")

        # Load features and ground truth
        features = f["features"][:]
        cbh_cpl = f["cbh_cpl"][:]

        # Feature structure: [shadow_cbh, shadow_length, confidence, ...]
        # Extract shadow-based CBH (first column)
        cbh_shadow = features[:, 0]

        # Confidence score (if available)
        if features.shape[1] > 2:
            confidence = features[:, 2]
        else:
            confidence = np.ones(len(cbh_shadow))

        print(f"  Loaded {len(cbh_shadow)} samples")
        print(
            f"  Shadow CBH range: {np.nanmin(cbh_shadow):.0f} - {np.nanmax(cbh_shadow):.0f} m"
        )
        print(f"  CPL CBH range: {np.min(cbh_cpl):.0f} - {np.max(cbh_cpl):.0f} m")

    return cbh_shadow, cbh_cpl, confidence


def filter_valid_samples(cbh_shadow, cbh_cpl, confidence):
    """Filter to valid shadow estimates."""
    # Valid criteria:
    # 1. Not NaN
    # 2. Positive CBH
    # 3. Reasonable range (0-10 km)
    valid_mask = (~np.isnan(cbh_shadow)) & (cbh_shadow > 0) & (cbh_shadow < 10000)

    cbh_shadow_valid = cbh_shadow[valid_mask]
    cbh_cpl_valid = cbh_cpl[valid_mask]
    confidence_valid = confidence[valid_mask]

    n_total = len(cbh_shadow)
    n_valid = len(cbh_shadow_valid)
    pct_valid = 100 * n_valid / n_total

    print(f"\nFiltering to valid samples:")
    print(f"  Total samples: {n_total}")
    print(f"  Valid samples: {n_valid} ({pct_valid:.1f}%)")
    print(f"  Removed: {n_total - n_valid} (NaN, negative, or out-of-range)")

    return cbh_shadow_valid, cbh_cpl_valid, confidence_valid


def compute_metrics(cbh_shadow, cbh_cpl):
    """Compute error metrics."""
    mae = np.mean(np.abs(cbh_shadow - cbh_cpl))
    rmse = np.sqrt(np.mean((cbh_shadow - cbh_cpl) ** 2))
    bias = np.mean(cbh_shadow - cbh_cpl)
    r, p = pearsonr(cbh_shadow, cbh_cpl)

    print("\n" + "=" * 60)
    print("SHADOW GEOMETRY PERFORMANCE METRICS")
    print("=" * 60)
    print(f"  MAE:  {mae:.2f} m ({mae / 1000:.2f} km)")
    print(f"  RMSE: {rmse:.2f} m ({rmse / 1000:.2f} km)")
    print(f"  Bias: {bias:.2f} m ({bias / 1000:.2f} km)")
    print(f"  Correlation: r = {r:.4f} (p = {p:.4e})")
    print(f"\nShadow CBH Statistics:")
    print(f"  Mean: {np.mean(cbh_shadow):.0f} m ({np.mean(cbh_shadow) / 1000:.2f} km)")
    print(f"  Std:  {np.std(cbh_shadow):.0f} m ({np.std(cbh_shadow) / 1000:.2f} km)")
    print(f"\nCPL CBH Statistics:")
    print(f"  Mean: {np.mean(cbh_cpl):.0f} m ({np.mean(cbh_cpl) / 1000:.2f} km)")
    print(f"  Std:  {np.std(cbh_cpl):.0f} m ({np.std(cbh_cpl) / 1000:.2f} km)")
    print("=" * 60 + "\n")

    if abs(r) < 0.1:
        print(
            "⚠️  WARNING: Correlation near zero - shadow CBH essentially uncorrelated with truth"
        )
    if abs(bias) > 1000:
        print(
            "⚠️  WARNING: Large systematic bias - shadow method has fundamental offset"
        )

    return {"mae": mae, "rmse": rmse, "bias": bias, "r": r, "p": p}


def create_failure_figure(cbh_shadow, cbh_cpl, confidence, metrics):
    """Create comprehensive failure analysis figure."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Top-left: Overall scatter with confidence coloring
    ax = axes[0, 0]
    sc = ax.scatter(
        cbh_cpl / 1000,
        cbh_shadow / 1000,
        c=confidence,
        cmap="viridis",
        alpha=0.6,
        s=20,
        vmin=0,
        vmax=1,
    )
    ax.plot(
        [0, 10], [0, 10], "r--", linewidth=2.5, label="1:1 line (perfect agreement)"
    )
    ax.set_xlabel("CPL Ground Truth CBH (km)", fontsize=13)
    ax.set_ylabel("Shadow-derived CBH (km)", fontsize=13)
    ax.set_title(
        f"Shadow Geometry Failure\n"
        f"r = {metrics['r']:.4f}, Bias = {metrics['bias'] / 1000:.2f} km, MAE = {metrics['mae'] / 1000:.2f} km",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Set equal aspect and reasonable limits
    max_cbh = max(
        np.percentile(cbh_cpl / 1000, 99), np.percentile(cbh_shadow / 1000, 99)
    )
    ax.set_xlim(0, min(max_cbh + 1, 10))
    ax.set_ylim(0, min(max_cbh + 1, 10))

    # Add colorbar
    cbar = plt.colorbar(sc, ax=ax, label="Detection Confidence")
    cbar.set_label("Detection Confidence", fontsize=11)

    # Top-right: Residual plot
    ax = axes[0, 1]
    residuals = (cbh_shadow - cbh_cpl) / 1000  # km
    ax.scatter(cbh_cpl / 1000, residuals, alpha=0.5, s=15, c="steelblue")
    ax.axhline(y=0, color="red", linestyle="--", linewidth=2.5, label="Zero error")
    ax.axhline(
        y=np.mean(residuals),
        color="orange",
        linestyle="-",
        linewidth=2.5,
        label=f"Mean bias: {np.mean(residuals):.2f} km",
    )
    ax.set_xlabel("CPL Ground Truth CBH (km)", fontsize=13)
    ax.set_ylabel("Residual (Shadow - CPL) (km)", fontsize=13)
    ax.set_title("Systematic Bias in Shadow Estimates", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max(4, np.percentile(cbh_cpl / 1000, 99)))

    # Bottom-left: Histogram of residuals
    ax = axes[1, 0]
    ax.hist(residuals, bins=50, alpha=0.7, edgecolor="black", color="coral")
    ax.axvline(x=0, color="red", linestyle="--", linewidth=2.5, label="Zero error")
    ax.axvline(
        x=np.mean(residuals),
        color="orange",
        linestyle="-",
        linewidth=2.5,
        label=f"Mean: {np.mean(residuals):.2f} km",
    )
    ax.axvline(
        x=np.median(residuals),
        color="purple",
        linestyle=":",
        linewidth=2,
        label=f"Median: {np.median(residuals):.2f} km",
    )
    ax.set_xlabel("Residual (Shadow - CPL) (km)", fontsize=13)
    ax.set_ylabel("Frequency", fontsize=13)
    ax.set_title("Distribution of Errors", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    # Bottom-right: Bland-Altman plot
    ax = axes[1, 1]
    mean_cbh = (cbh_shadow + cbh_cpl) / 2000  # km
    diff_cbh = (cbh_shadow - cbh_cpl) / 1000  # km
    ax.scatter(mean_cbh, diff_cbh, alpha=0.5, s=15, c="steelblue")
    ax.axhline(y=0, color="red", linestyle="--", linewidth=2.5, label="Zero difference")
    ax.axhline(
        y=np.mean(diff_cbh),
        color="orange",
        linestyle="-",
        linewidth=2.5,
        label=f"Mean: {np.mean(diff_cbh):.2f} km",
    )

    # Add ±1.96 SD lines (95% limits of agreement)
    sd = np.std(diff_cbh)
    ax.axhline(
        y=np.mean(diff_cbh) + 1.96 * sd,
        color="gray",
        linestyle=":",
        linewidth=2,
        label=f"±1.96 SD: [{np.mean(diff_cbh) - 1.96 * sd:.2f}, {np.mean(diff_cbh) + 1.96 * sd:.2f}] km",
    )
    ax.axhline(
        y=np.mean(diff_cbh) - 1.96 * sd, color="gray", linestyle=":", linewidth=2
    )

    ax.set_xlabel("Mean CBH (km)", fontsize=13)
    ax.set_ylabel("Difference (Shadow - CPL) (km)", fontsize=13)
    ax.set_title("Bland-Altman Plot", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)

    plt.suptitle(
        "Shadow-Based Geometric CBH Estimation: Comprehensive Failure Analysis",
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
    print("SHADOW GEOMETRY FAILURE ANALYSIS")
    print("=" * 60 + "\n")

    # Load data
    cbh_shadow, cbh_cpl, confidence = load_data()

    # Filter to valid samples
    cbh_shadow_valid, cbh_cpl_valid, confidence_valid = filter_valid_samples(
        cbh_shadow, cbh_cpl, confidence
    )

    # Compute metrics
    metrics = compute_metrics(cbh_shadow_valid, cbh_cpl_valid)

    # Create figure
    fig = create_failure_figure(
        cbh_shadow_valid, cbh_cpl_valid, confidence_valid, metrics
    )

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nKey Findings:")
    print(f"  • Correlation with ground truth: r = {metrics['r']:.4f}")
    print(f"  • Systematic bias: {metrics['bias'] / 1000:.2f} km")
    print(f"  • Mean absolute error: {metrics['mae'] / 1000:.2f} km")
    print(f"\nConclusion:")
    if abs(metrics["r"]) < 0.1:
        print("  Shadow-based CBH is UNCORRELATED with ground truth.")
    if abs(metrics["bias"]) > 3000:
        print("  Shadow-based CBH has SEVERE systematic bias (>3 km).")
    print("  Shadow geometry from nadir imagery over ocean is INFEASIBLE.")

    plt.show()


if __name__ == "__main__":
    main()
