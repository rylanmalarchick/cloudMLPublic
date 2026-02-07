#!/usr/bin/env python3
"""
Generate Paper 1 figures from CNN vision baseline results.

Reads JSON result files from vision_baselines.py and generates:
  1. CBH distribution histogram
  2. Model comparison bar chart (R², MAE)
  3. Scatter plot: CNN predictions vs CPL ground truth (best model)
  4. Fold performance variability
  5. Residual analysis (residuals vs predicted, residual histogram)

Usage:
  python generate_paper1_cnn_figures.py [--results-dir PATH] [--output-dir PATH]
"""

import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent.parent / "outputs" / "vision_baselines" / "reports"
OUTPUT_DIR = Path(__file__).parent.parent / "paperfigures"

MODEL_CONFIGS = [
    ("resnet18_scratch_noaugment", "ResNet-18 (scratch)"),
    ("resnet18_pretrained_noaugment", "ResNet-18 (pretrained)"),
    ("resnet18_pretrained_augment", "ResNet-18 (pretrained+aug)"),
    ("efficientnet_b0_scratch_noaugment", "EfficientNet-B0 (scratch)"),
    ("efficientnet_b0_pretrained_noaugment", "EfficientNet-B0 (pretrained)"),
    ("efficientnet_b0_pretrained_augment", "EfficientNet-B0 (pretrained+aug)"),
]

COLORS = {
    "resnet18": "#2196F3",
    "efficientnet_b0": "#FF9800",
}


def load_results(results_dir: Path) -> dict:
    """Load all result JSON files."""
    results = {}
    for config_key, label in MODEL_CONFIGS:
        path = results_dir / f"{config_key}_results.json"
        if path.exists():
            with open(path) as f:
                results[config_key] = json.load(f)
            results[config_key]["label"] = label
        else:
            print(f"Warning: Missing {path.name}")
    return results


def get_best_model(results: dict) -> str:
    """Return config key of the model with highest mean R²."""
    best_key = None
    best_r2 = -np.inf
    for key, data in results.items():
        r2_vals, _, _ = collect_fold_metrics(data)
        mean_r2 = np.mean(r2_vals)
        if mean_r2 > best_r2:
            best_r2 = mean_r2
            best_key = key
    return best_key


def collect_fold_metrics(data: dict):
    """Extract per-fold metrics from a result dict."""
    folds = data.get("fold_results", data.get("folds", []))
    r2_vals, mae_vals, rmse_vals = [], [], []
    for f in folds:
        vm = f.get("val_metrics", {})
        if vm:
            r2_vals.append(vm.get("r2", 0))
            mae_vals.append(vm.get("mae_m", 0))
            rmse_vals.append(vm.get("rmse_m", 0))
        else:
            r2_vals.append(f.get("val_r2", f.get("r2", 0)))
            mae_vals.append(f.get("val_mae_m", f.get("mae_m", 0)))
            rmse_vals.append(f.get("val_rmse_m", f.get("rmse_m", 0)))
    return np.array(r2_vals), np.array(mae_vals), np.array(rmse_vals)


def collect_predictions(data: dict):
    """Extract all y_true, y_pred from fold results. Values in km."""
    folds = data.get("fold_results", data.get("folds", []))
    y_true_all, y_pred_all = [], []
    for f in folds:
        yt = f.get("y_true", [])
        yp = f.get("y_pred", [])
        if yt and yp:
            y_true_all.extend(yt)
            y_pred_all.extend(yp)
    return np.array(y_true_all), np.array(y_pred_all)


def fig1_cbh_distribution(results: dict, output_dir: Path):
    """CBH distribution histogram from the best model's y_true values."""
    best_key = get_best_model(results)
    y_true, _ = collect_predictions(results[best_key])

    # Convert to meters if in km
    if np.max(y_true) < 10:
        y_true = y_true * 1000

    # Each sample appears exactly once across folds as a val sample,
    # so y_true across all folds = all 380 samples (no dedup needed)
    cbh = y_true

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(cbh, bins=25, color="#2196F3", edgecolor="white", alpha=0.85)
    ax.axvline(np.mean(cbh), color="red", linestyle="--", linewidth=1.5,
               label=f"Mean = {np.mean(cbh):.0f} m")
    ax.set_xlabel("Cloud Base Height (m)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("CBH Distribution in Training Dataset", fontsize=13)
    ax.legend(fontsize=11)
    stats_text = (f"n = {len(cbh)}\n"
                  f"Mean = {np.mean(cbh):.0f} m\n"
                  f"Std = {np.std(cbh):.0f} m\n"
                  f"Range: {np.min(cbh):.0f}--{np.max(cbh):.0f} m")
    ax.text(0.97, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = output_dir / "paper1_fig_cbh_distribution.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def fig2_model_comparison(results: dict, output_dir: Path):
    """Bar chart comparing R² and MAE across all 6 configs."""
    labels, r2_means, r2_stds, mae_means, mae_stds = [], [], [], [], []
    bar_colors = []

    for config_key, label in MODEL_CONFIGS:
        if config_key not in results:
            continue
        r2, mae, _ = collect_fold_metrics(results[config_key])
        labels.append(label)
        r2_means.append(np.mean(r2))
        r2_stds.append(np.std(r2))
        mae_means.append(np.mean(mae))
        mae_stds.append(np.std(mae))
        arch = "resnet18" if "resnet" in config_key else "efficientnet_b0"
        bar_colors.append(COLORS[arch])

    x = np.arange(len(labels))
    width = 0.6

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # R² plot
    bars1 = ax1.bar(x, r2_means, width, yerr=r2_stds, color=bar_colors,
                    edgecolor="white", capsize=4, alpha=0.85)
    ax1.axhline(0, color="gray", linestyle="-", linewidth=0.8)
    ax1.set_ylabel("R$^2$", fontsize=12)
    ax1.set_title("Model Comparison: R$^2$", fontsize=13)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax1.grid(True, alpha=0.3, axis="y")

    # MAE plot
    bars2 = ax2.bar(x, mae_means, width, yerr=mae_stds, color=bar_colors,
                    edgecolor="white", capsize=4, alpha=0.85)
    ax2.set_ylabel("MAE (m)", fontsize=12)
    ax2.set_title("Model Comparison: MAE", fontsize=13)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax2.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    out = output_dir / "paper1_fig_model_comparison.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def fig3_scatter(results: dict, output_dir: Path):
    """Scatter plot of best model predictions vs CPL truth."""
    best_key = get_best_model(results)
    y_true, y_pred = collect_predictions(results[best_key])

    # Convert to meters if in km
    if np.max(y_true) < 10:
        y_true = y_true * 1000
        y_pred = y_pred * 1000

    # Use mean-of-folds metrics to match paper table (not pooled)
    r2_vals, mae_vals, rmse_vals = collect_fold_metrics(results[best_key])
    r2 = np.mean(r2_vals)
    mae = np.mean(mae_vals)
    rmse = np.mean(rmse_vals)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_true, y_pred, alpha=0.5, s=30, c="#2196F3", edgecolors="none")

    lims = [min(y_true.min(), y_pred.min()) - 30,
            max(y_true.max(), y_pred.max()) + 30]
    ax.plot(lims, lims, "k--", linewidth=1.5, alpha=0.7, label="1:1 line")

    z = np.polyfit(y_true, y_pred, 1)
    x_fit = np.linspace(lims[0], lims[1], 100)
    ax.plot(x_fit, np.polyval(z, x_fit), "r-", linewidth=1.5, alpha=0.7,
            label=f"Best fit (slope={z[0]:.2f})")

    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")
    ax.set_xlabel("CPL Cloud Base Height (m)", fontsize=12)
    ax.set_ylabel("CNN Predicted CBH (m)", fontsize=12)
    ax.set_title(f"{results[best_key]['label']} vs CPL Ground Truth", fontsize=13)

    stats_text = (f"n = {len(y_true)}\n"
                  f"R$^2$ = {r2:.3f}\n"
                  f"MAE = {mae:.1f} m\n"
                  f"RMSE = {rmse:.1f} m")
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = output_dir / "paper1_fig_scatter_cnn_vs_cpl.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def fig4_fold_performance(results: dict, output_dir: Path):
    """Per-fold R² and MAE for the best model."""
    best_key = get_best_model(results)
    r2_vals, mae_vals, _ = collect_fold_metrics(results[best_key])
    n_folds = len(r2_vals)
    folds = np.arange(1, n_folds + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.bar(folds, r2_vals, color="#2196F3", edgecolor="white", alpha=0.85)
    ax1.axhline(np.mean(r2_vals), color="red", linestyle="--", linewidth=1.5,
                label=f"Mean = {np.mean(r2_vals):.3f}")
    ax1.set_xlabel("Fold", fontsize=12)
    ax1.set_ylabel("R$^2$", fontsize=12)
    ax1.set_title(f"{results[best_key]['label']}: R$^2$ by Fold", fontsize=13)
    ax1.set_xticks(folds)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis="y")

    ax2.bar(folds, mae_vals, color="#FF9800", edgecolor="white", alpha=0.85)
    ax2.axhline(np.mean(mae_vals), color="red", linestyle="--", linewidth=1.5,
                label=f"Mean = {np.mean(mae_vals):.1f} m")
    ax2.set_xlabel("Fold", fontsize=12)
    ax2.set_ylabel("MAE (m)", fontsize=12)
    ax2.set_title(f"{results[best_key]['label']}: MAE by Fold", fontsize=13)
    ax2.set_xticks(folds)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    out = output_dir / "paper1_fig_fold_performance.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def fig5_residual_analysis(results: dict, output_dir: Path):
    """Residual analysis: residuals vs predicted, and residual histogram."""
    best_key = get_best_model(results)
    y_true, y_pred = collect_predictions(results[best_key])

    if np.max(y_true) < 10:
        y_true = y_true * 1000
        y_pred = y_pred * 1000

    residuals = y_pred - y_true

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Residuals vs predicted
    ax1.scatter(y_pred, residuals, alpha=0.5, s=30, c="#2196F3", edgecolors="none")
    ax1.axhline(0, color="red", linestyle="--", linewidth=1.5)
    ax1.set_xlabel("Predicted CBH (m)", fontsize=12)
    ax1.set_ylabel("Residual (m)", fontsize=12)
    ax1.set_title("Residuals vs. Predicted", fontsize=13)
    ax1.grid(True, alpha=0.3)

    # Residual histogram
    ax2.hist(residuals, bins=25, color="#FF9800", edgecolor="white", alpha=0.85)
    ax2.axvline(0, color="red", linestyle="--", linewidth=1.5)
    ax2.axvline(np.mean(residuals), color="blue", linestyle="--", linewidth=1.5,
                label=f"Mean = {np.mean(residuals):.1f} m")
    ax2.set_xlabel("Residual (m)", fontsize=12)
    ax2.set_ylabel("Count", fontsize=12)
    ax2.set_title("Residual Distribution", fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    out = output_dir / "paper1_fig_residual_analysis.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def main():
    parser = argparse.ArgumentParser(description="Generate Paper 1 CNN figures")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Paper 1 CNN Figure Generator")
    print(f"Results: {args.results_dir}")
    print(f"Output:  {args.output_dir}")
    print("=" * 60)

    results = load_results(args.results_dir)
    if not results:
        print("ERROR: No result files found!")
        return

    best_key = get_best_model(results)
    print(f"\nBest model: {results[best_key]['label']}")
    r2, mae, _ = collect_fold_metrics(results[best_key])
    print(f"  R² = {np.mean(r2):.4f} +/- {np.std(r2):.4f}")
    print(f"  MAE = {np.mean(mae):.1f} +/- {np.std(mae):.1f} m")

    print("\nGenerating figures...")
    fig1_cbh_distribution(results, args.output_dir)
    fig2_model_comparison(results, args.output_dir)
    fig3_scatter(results, args.output_dir)
    fig4_fold_performance(results, args.output_dir)
    fig5_residual_analysis(results, args.output_dir)

    print("\nAll Paper 1 figures generated!")


if __name__ == "__main__":
    main()
