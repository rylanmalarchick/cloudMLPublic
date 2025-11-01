#!/usr/bin/env python3
"""
SECTION 2.3: PLOT PREDICTION DISTRIBUTIONS
===========================================

This script generates visualization of prediction distributions from all
Section 2 variance_lambda experiments to support the qualitative analysis
required in Section 2.3 of the research program.

For each experiment, it creates:
  1. Histogram comparing predictions vs. true values
  2. Variance ratio over training epochs
  3. R² over training epochs

These plots enable visual assessment of:
  - How well prediction distribution matches target distribution
  - Training stability
  - Effectiveness of variance preservation

Usage:
    python scripts/plot_section2_distributions.py

Output:
    - diagnostics/results/section2_distributions.png (grid of histograms)
    - diagnostics/results/section2_training_curves.png (R² and variance ratio)
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re
import glob
import pickle

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (16, 10)
plt.rcParams["font.size"] = 10


def load_predictions(config_name):
    """
    Load saved predictions from a training run.

    Returns:
        tuple: (predictions, true_values) or (None, None) if not found
    """
    # Common output directories
    output_dirs = [
        "diagnostics/results",
        "/content/drive/MyDrive/CloudML/plots",
        "plots",
        "output",
    ]

    for output_dir in output_dirs:
        if not Path(output_dir).exists():
            continue

        # Look for pickle files with predictions
        patterns = [
            f"{output_dir}/*{config_name}*predictions.pkl",
            f"{output_dir}/*{config_name}*.pkl",
            f"{output_dir}/predictions*.pkl",
        ]

        for pattern in patterns:
            matches = glob.glob(pattern)
            if matches:
                # Load most recent
                latest = max(matches, key=lambda p: Path(p).stat().st_mtime)
                try:
                    with open(latest, "rb") as f:
                        data = pickle.load(f)
                        if isinstance(data, dict):
                            return data.get("predictions"), data.get("targets")
                        elif isinstance(data, tuple):
                            return data
                except Exception as e:
                    print(f"  Warning: Could not load {latest}: {e}")

    return None, None


def parse_training_history(log_path):
    """
    Parse training log to extract R² and variance ratio over epochs.

    Returns:
        dict with keys: epochs, r2_values, variance_ratios
    """
    history = {"epochs": [], "r2_values": [], "variance_ratios": []}

    if not Path(log_path).exists():
        return history

    with open(log_path, "r") as f:
        lines = f.readlines()

    current_epoch = 0

    for line in lines:
        # Detect epoch number
        epoch_match = re.search(r"Epoch\s+(\d+)", line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))

        # Extract validation R²
        if "Validation R²" in line or "Val R²" in line or "val_r2" in line.lower():
            r2_match = re.search(r"[-]?\d+\.\d+", line)
            if r2_match:
                val = float(r2_match.group())
                if not np.isnan(val) and not np.isinf(val):
                    if current_epoch not in history["epochs"]:
                        history["epochs"].append(current_epoch)
                        history["r2_values"].append(val)
                        history["variance_ratios"].append(None)  # Placeholder

        # Extract variance ratio
        if "variance_ratio" in line.lower() or "Variance Ratio" in line:
            var_match = re.search(r"\d+\.\d+", line)
            if var_match:
                val = float(var_match.group())
                # Update most recent epoch
                if (
                    history["variance_ratios"]
                    and history["variance_ratios"][-1] is None
                ):
                    history["variance_ratios"][-1] = val

    return history


def find_latest_log(config_name):
    """Find the most recent log file for a given config."""
    log_dirs = ["logs", "/content/drive/MyDrive/CloudML/logs", "diagnostics/results"]

    for log_dir in log_dirs:
        if Path(log_dir).exists():
            pattern = f"{log_dir}/*{config_name}*.log"
            matches = glob.glob(pattern)
            if matches:
                return max(matches, key=lambda p: Path(p).stat().st_mtime)

    return None


def plot_distribution_grid(experiments_data, output_path):
    """
    Create a grid of histograms comparing predictions vs. true values.
    """
    n_experiments = len(experiments_data)
    n_cols = 3
    n_rows = (n_experiments + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
    axes = axes.flatten() if n_experiments > 1 else [axes]

    for idx, exp_data in enumerate(experiments_data):
        ax = axes[idx]
        name = exp_data["name"]
        predictions = exp_data["predictions"]
        targets = exp_data["targets"]

        if predictions is None or targets is None:
            ax.text(
                0.5,
                0.5,
                f"{name}\n[No data available]",
                ha="center",
                va="center",
                fontsize=12,
            )
            ax.set_title(name, fontsize=14, fontweight="bold")
            ax.axis("off")
            continue

        # Plot histograms
        bins = np.linspace(0, 2, 30)  # CBH range 0-2 km
        ax.hist(
            targets,
            bins=bins,
            alpha=0.6,
            label="True CBH",
            color="blue",
            edgecolor="black",
        )
        ax.hist(
            predictions,
            bins=bins,
            alpha=0.6,
            label="Predictions",
            color="red",
            edgecolor="black",
        )

        # Add statistics
        pred_mean = np.mean(predictions)
        pred_std = np.std(predictions)
        true_mean = np.mean(targets)
        true_std = np.std(targets)
        variance_ratio = (pred_std / true_std) * 100 if true_std > 0 else 0

        stats_text = f"μ_pred={pred_mean:.3f}, σ_pred={pred_std:.3f}\n"
        stats_text += f"μ_true={true_mean:.3f}, σ_true={true_std:.3f}\n"
        stats_text += f"Variance Ratio={variance_ratio:.1f}%"

        ax.text(
            0.95,
            0.95,
            stats_text,
            transform=ax.transAxes,
            fontsize=9,
            va="top",
            ha="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        ax.set_xlabel("Cloud Base Height (km)", fontsize=11)
        ax.set_ylabel("Frequency", fontsize=11)
        ax.set_title(name, fontsize=14, fontweight="bold")
        ax.legend(loc="upper left", fontsize=10)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_experiments, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved distribution grid to: {output_path}")
    plt.close()


def plot_training_curves(experiments_data, output_path):
    """
    Plot R² and variance ratio over training epochs for all experiments.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    colors = plt.cm.viridis(np.linspace(0, 1, len(experiments_data)))

    for idx, exp_data in enumerate(experiments_data):
        name = exp_data["name"]
        history = exp_data["history"]

        if not history["epochs"]:
            continue

        # Plot R²
        ax1.plot(
            history["epochs"],
            history["r2_values"],
            marker="o",
            label=name,
            color=colors[idx],
            linewidth=2,
        )

        # Plot variance ratio
        if any(v is not None for v in history["variance_ratios"]):
            valid_ratios = [
                (e, r)
                for e, r in zip(history["epochs"], history["variance_ratios"])
                if r is not None
            ]
            if valid_ratios:
                epochs, ratios = zip(*valid_ratios)
                ax2.plot(
                    epochs,
                    ratios,
                    marker="s",
                    label=name,
                    color=colors[idx],
                    linewidth=2,
                )

    # Format R² plot
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Validation R²", fontsize=12)
    ax1.set_title("Validation R² Over Training", fontsize=14, fontweight="bold")
    ax1.axhline(y=0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax1.legend(loc="best", fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Format variance ratio plot
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Variance Ratio (%)", fontsize=12)
    ax2.set_title(
        "Prediction Variance Ratio Over Training", fontsize=14, fontweight="bold"
    )
    ax2.axhline(
        y=100,
        color="green",
        linestyle="--",
        linewidth=1,
        alpha=0.5,
        label="Target (100%)",
    )
    ax2.legend(loc="best", fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved training curves to: {output_path}")
    plt.close()


def main():
    print("=" * 70)
    print("SECTION 2.3: PLOTTING PREDICTION DISTRIBUTIONS")
    print("=" * 70)
    print()

    # Define experiments
    experiments = [
        {"name": "Baseline (λ=0.0)", "config": "section2_baseline_collapse"},
        {"name": "λ=0.5", "config": "section2_lambda_0.5"},
        {"name": "λ=1.0", "config": "section2_lambda_1.0"},
        {"name": "λ=2.0", "config": "section2_lambda_2.0"},
        {"name": "λ=5.0", "config": "section2_lambda_5.0"},
        {"name": "λ=10.0", "config": "section2_lambda_10.0"},
    ]

    # Collect data
    experiments_data = []

    for exp in experiments:
        print(f"Processing {exp['name']}...")

        # Load predictions
        predictions, targets = load_predictions(exp["config"])

        # Load training history
        log_file = find_latest_log(exp["config"])
        if log_file:
            print(f"  Found log: {log_file}")
            history = parse_training_history(log_file)
        else:
            print(f"  WARNING: No log file found")
            history = {"epochs": [], "r2_values": [], "variance_ratios": []}

        experiments_data.append(
            {
                "name": exp["name"],
                "predictions": predictions,
                "targets": targets,
                "history": history,
            }
        )

    # Create output directory
    output_dir = Path("diagnostics/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate plots
    print("\nGenerating visualizations...")

    # Distribution grid
    dist_path = output_dir / "section2_distributions.png"
    plot_distribution_grid(experiments_data, dist_path)

    # Training curves
    curves_path = output_dir / "section2_training_curves.png"
    plot_training_curves(experiments_data, curves_path)

    print("\n" + "=" * 70)
    print("✅ VISUALIZATION COMPLETE")
    print("=" * 70)
    print(f"\nGenerated files:")
    print(f"  1. {dist_path}")
    print(f"  2. {curves_path}")
    print("\nThese visualizations support Section 2.3 analysis:")
    print("  - Distribution matching: Do predictions match target spread?")
    print("  - Training stability: Are curves smooth or erratic?")
    print("  - Variance preservation: Does variance ratio approach 100%?")
    print("\nUse these plots to select optimal variance_lambda.")
    print("=" * 70)


if __name__ == "__main__":
    main()
