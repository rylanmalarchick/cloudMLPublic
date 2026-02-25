#!/usr/bin/env python3
"""
Generate Paper 2 figures from v2 rerun results.

Reads LOFO predictions from paper2_all_results_v2.json and CPL flight data
to generate:
  Fig 2: CBH distribution comparison (Oct 23 vs Feb 10, ocean-only filtered)
  Fig 3: LOFO scatter plots (Oct 23 vs Feb 10, with correct n and R²)

Also regenerates supplementary figures:
  - Domain adaptation summary bar chart
  - K-S divergence heatmap
  - Few-shot learning curve

Author: Rylan Malarchick
Date: February 2026 (audit reconciliation, iteration 2)
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
RESULTS_JSON = PROJECT_DIR / "results" / "paper2_rerun_v2" / "paper2_all_results_v2.json"
OUTPUT_DIR = SCRIPT_DIR.parent / "paperfigures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load results
with open(RESULTS_JSON) as f:
    RESULTS = json.load(f)

FLIGHT_LABELS = {
    "23Oct24": "Oct 23, 2024 (WHySMIE)",
    "30Oct24": "Oct 30, 2024 (WHySMIE)",
    "04Nov24": "Nov 4, 2024 (WHySMIE)",
    "10Feb25": "Feb 10, 2025 (GLOVE)",
    "12Feb25": "Feb 12, 2025 (GLOVE)",
    "18Feb25": "Feb 18, 2025 (GLOVE)",
}


def fig2_cbh_distribution_comparison():
    """
    CBH distribution comparison for Oct 23 and Feb 10 (ocean-only, ≤2 km).
    Uses v2 LOFO predictions which contain y_true for each flight.
    
    Paper caption should say:
      Oct 23: n=857 ocean-only, mean=138 m
      Feb 10: n=608 ocean-only, mean=380 m
    """
    print("\n[Fig 2] CBH distribution comparison...")
    
    preds = RESULTS["lofo_baseline"]["predictions"]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for flight_key, color, label_short in [
        ("23Oct24", "#2196F3", "Oct 23 (WHySMIE)"),
        ("10Feb25", "#F44336", "Feb 10 (GLOVE)"),
    ]:
        y_true_km = np.array(preds[flight_key]["y_true_km"])
        y_true_m = y_true_km * 1000
        n = len(y_true_m)
        mean_m = np.mean(y_true_m)
        
        ax.hist(y_true_m, bins=50, alpha=0.55, color=color, edgecolor="white",
                label=f"{label_short} (n={n}, mean={mean_m:.0f} m)",
                density=True)
    
    ax.set_xlabel("Cloud Base Height (m)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("CBH Distribution: WHySMIE vs GLOVE (Ocean-Only, BL Clouds ≤ 2 km)",
                 fontsize=13)
    ax.set_xlim(0, 2000)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    out = OUTPUT_DIR / "paper2_fig2_cbh_distribution_comparison.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")
    
    # Print stats for caption verification
    for fk in ["23Oct24", "10Feb25"]:
        yt = np.array(preds[fk]["y_true_km"]) * 1000
        print(f"  {fk}: n={len(yt)}, mean={np.mean(yt):.0f} m, "
              f"std={np.std(yt):.0f} m, range={np.min(yt):.0f}-{np.max(yt):.0f} m")


def fig3_scatter_comparison():
    """
    LOFO scatter plots for Oct 23 and Feb 10.
    Uses v2 LOFO predictions (y_true, y_pred per flight).
    
    Paper caption should say:
      Oct 23: R² = -8.99, n=857
      Feb 10: R² = -1.97, n=608
    """
    print("\n[Fig 3] LOFO scatter comparison...")
    
    preds = RESULTS["lofo_baseline"]["predictions"]
    lofo = RESULTS["lofo_baseline"]["per_flight"]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for ax, flight_key, label_short in zip(
        axes,
        ["23Oct24", "10Feb25"],
        ["Oct 23 (WHySMIE)", "Feb 10 (GLOVE)"],
    ):
        y_true = np.array(preds[flight_key]["y_true_km"]) * 1000  # to meters
        y_pred = np.array(preds[flight_key]["y_pred_km"]) * 1000
        r2 = lofo[flight_key]["r2"]
        mae_m = lofo[flight_key]["mae_m"]
        n = lofo[flight_key]["n_test"]
        
        ax.scatter(y_true, y_pred, alpha=0.4, s=15, c="#2196F3", edgecolors="none")
        
        # 1:1 line
        all_vals = np.concatenate([y_true, y_pred])
        lims = [max(0, all_vals.min() - 50), all_vals.max() + 50]
        ax.plot(lims, lims, "k--", linewidth=1.5, alpha=0.7, label="1:1 line")
        
        # Mean baseline (horizontal line at mean of y_true)
        mean_true = np.mean(y_true)
        ax.axhline(mean_true, color="red", linestyle=":", linewidth=1,
                    alpha=0.6, label=f"Mean baseline ({mean_true:.0f} m)")
        
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect("equal")
        ax.set_xlabel("CPL Cloud Base Height (m)", fontsize=11)
        ax.set_ylabel("LOFO Predicted CBH (m)", fontsize=11)
        ax.set_title(f"{label_short}\n(Leave-One-Flight-Out)", fontsize=12)
        
        stats_text = (f"n = {n}\n"
                      f"R² = {r2:.2f}\n"
                      f"MAE = {mae_m:.0f} m")
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=11,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, alpha=0.3)
    
    fig.suptitle("Domain Shift: LOFO Cross-Validation Failure", fontsize=14, y=1.02)
    fig.tight_layout()
    out = OUTPUT_DIR / "paper2_fig3_scatter_comparison.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def fig_adaptation_summary():
    """Bar chart of domain adaptation method comparison."""
    print("\n[Supp] Adaptation summary bar chart...")
    
    methods = [
        ("LOFO baseline", RESULTS["lofo_baseline"]["mean_r2"]),
        ("IW (KNN)", RESULTS["instance_weighting"]["mean_r2"]["knn"]),
        ("IW (density)", RESULTS["instance_weighting"]["mean_r2"]["density"]),
        ("MMD alignment", RESULTS["mmd_alignment"]["mean_r2"]),
        ("Feature selection", RESULTS["feature_selection"]["mean_r2"]),
        ("TrAdaBoost", RESULTS["tradaboost"]["mean_r2"]),
        ("Few-shot (50)", RESULTS["few_shot"]["mean_by_shots"]["50"]),
    ]
    
    labels = [m[0] for m in methods]
    values = [m[1] for m in methods]
    colors = ["#9E9E9E" if v < 0 else "#4CAF50" for v in values]
    colors[0] = "#F44336"  # baseline in red
    colors[-1] = "#2196F3"  # best in blue
    
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(range(len(labels)), values, color=colors, edgecolor="white", alpha=0.85)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("Mean R² (LOFO)", fontsize=12)
    ax.set_title("Domain Adaptation Method Comparison", fontsize=13)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.grid(True, alpha=0.3, axis="x")
    
    # Value labels
    for i, (bar, val) in enumerate(zip(bars, values)):
        x_pos = val + 0.1 if val >= 0 else val - 0.1
        ha = "left" if val >= 0 else "right"
        ax.text(x_pos, i, f"{val:.2f}", va="center", ha=ha, fontsize=10)
    
    fig.tight_layout()
    out = OUTPUT_DIR / "paper2_fig_adaptation_summary.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def fig_fewshot_curve():
    """Few-shot learning curve: R² vs number of shots."""
    print("\n[Supp] Few-shot learning curve...")
    
    shots = [0, 5, 10, 20, 50]
    means = [RESULTS["lofo_baseline"]["mean_r2"]]
    means += [RESULTS["few_shot"]["mean_by_shots"][str(s)] for s in shots[1:]]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(shots, means, "o-", color="#2196F3", linewidth=2, markersize=8)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Number of Target Samples", fontsize=12)
    ax.set_ylabel("Mean R² (LOFO)", fontsize=12)
    ax.set_title("Few-Shot Adaptation: Recovery from Domain Shift", fontsize=13)
    ax.set_xticks(shots)
    ax.set_xticklabels(["0\n(baseline)", "5", "10", "20", "50"])
    ax.grid(True, alpha=0.3)
    
    for s, m in zip(shots, means):
        ax.annotate(f"{m:.2f}", (s, m), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=10)
    
    fig.tight_layout()
    out = OUTPUT_DIR / "paper2_fig_fewshot_curve.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def fig_ks_divergence():
    """K-S divergence bar chart for Oct 23 vs Feb 10."""
    print("\n[Supp] K-S divergence chart...")
    
    ks_data = RESULTS["ks_divergence"]
    # Top 15
    features = [d["feature"] for d in ks_data[:15]]
    ks_vals = [d["ks_stat"] for d in ks_data[:15]]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#F44336" if v >= 0.95 else "#FF9800" if v >= 0.8 else "#4CAF50"
              for v in ks_vals]
    ax.barh(range(len(features)), ks_vals, color=colors, edgecolor="white", alpha=0.85)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features, fontsize=10)
    ax.set_xlabel("K-S Statistic", fontsize=12)
    ax.set_title("Feature Distribution Shift: Oct 23 (WHySMIE) vs Feb 10 (GLOVE)", fontsize=13)
    ax.set_xlim(0, 1.05)
    ax.axvline(0.95, color="red", linestyle="--", linewidth=0.8, alpha=0.5,
               label="Near-total shift (0.95)")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="x")
    ax.invert_yaxis()
    
    fig.tight_layout()
    out = OUTPUT_DIR / "paper2_fig_ks_divergence.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def fig_feature_importance():
    """Feature importance comparison: base-5 vs full-34."""
    print("\n[Supp] Feature importance comparison...")
    
    base_imp = RESULTS["feature_importance"]["base_5"]
    full_imp = RESULTS["feature_importance"]["full_34"]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Base 5
    feats = list(base_imp.keys())
    vals = list(base_imp.values())
    ax1.barh(range(len(feats)), vals, color="#2196F3", edgecolor="white", alpha=0.85)
    ax1.set_yticks(range(len(feats)))
    ax1.set_yticklabels(feats, fontsize=11)
    ax1.set_xlabel("Importance (%)", fontsize=12)
    ax1.set_title("Base Model (5 ERA5 Features)", fontsize=13)
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.3, axis="x")
    
    # Full 34 (top 10)
    top_feats = list(full_imp.keys())[:10]
    top_vals = [full_imp[f] for f in top_feats]
    ax2.barh(range(len(top_feats)), top_vals, color="#FF9800", edgecolor="white", alpha=0.85)
    ax2.set_yticks(range(len(top_feats)))
    ax2.set_yticklabels(top_feats, fontsize=11)
    ax2.set_xlabel("Importance (%)", fontsize=12)
    ax2.set_title("Enhanced Model (34 Features, Top 10)", fontsize=13)
    ax2.invert_yaxis()
    ax2.grid(True, alpha=0.3, axis="x")
    
    fig.tight_layout()
    out = OUTPUT_DIR / "paper2_fig_feature_importance.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def main():
    print("=" * 60)
    print("Paper 2 Figure Generator (v2 Audit Reconciliation)")
    print(f"Source: {RESULTS_JSON}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Metadata: {RESULTS['metadata']['version']}, "
          f"{RESULTS['metadata']['total_samples']} samples, "
          f"{RESULTS['metadata']['n_features']} features")
    print("=" * 60)
    
    # Core paper figures
    fig2_cbh_distribution_comparison()
    fig3_scatter_comparison()
    
    # Supplementary figures
    fig_adaptation_summary()
    fig_fewshot_curve()
    fig_ks_divergence()
    fig_feature_importance()
    
    print("\n" + "=" * 60)
    print("All Paper 2 figures generated!")
    print("=" * 60)


if __name__ == "__main__":
    main()
