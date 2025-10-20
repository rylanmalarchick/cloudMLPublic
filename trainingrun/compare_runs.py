#!/usr/bin/env python3
"""
Compare All Three Training Runs
================================

Comprehensive comparison of the three Tier 1 training runs to understand
the progression from broken SSL to severe variance collapse.
"""

import csv
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

# Run configurations
RUNS = [
    {
        "name": "Run 1: Broken SSL",
        "desc": "Original run with broken SSL pretraining (std≈0.5 init)",
        "pred_path": "first/logs/csv/predictions_final_overweighted_tier1_final.csv",
        "metrics_path": "first/logs/csv/metrics_final_overweighted_tier1_final.json",
        "train_path": "first/logs/csv/final_overweighted_tier1_final.csv",
    },
    {
        "name": "Run 2: No SSL (std=0.01)",
        "desc": "No pretraining, output init std=0.01/bias=0.9",
        "pred_path": "second/logs/csv/predictions_final_overweighted_tier1_tuned_20251019_210420.csv",
        "metrics_path": "second/logs/csv/metrics_final_overweighted_tier1_tuned_20251019_210420.json",
        "train_path": "second/logs/csv/final_overweighted_tier1_tuned_20251019_210420.csv",
    },
    {
        "name": "Run 3: No SSL (std=0.1)",
        "desc": "No pretraining, output init std=0.1/bias=0.0",
        "pred_path": "third/logs/csv/predictions_final_overweighted_tier1_tuned_20251020_021138.csv",
        "metrics_path": "third/logs/csv/metrics_final_overweighted_tier1_tuned_20251020_021138.json",
        "train_path": "third/logs/csv/final_overweighted_tier1_tuned_20251020_021138.csv",
    },
]


def load_predictions(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load predictions from CSV."""
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        data = list(reader)

    y_true = np.array([float(r["y_true"]) for r in data])
    y_pred = np.array([float(r["y_pred"]) for r in data])

    return y_true, y_pred


def load_metrics(path: str) -> Dict:
    """Load metrics from JSON."""
    with open(path, "r") as f:
        return json.load(f)


def load_training(path: str) -> List[Dict]:
    """Load training history from CSV."""
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        return list(reader)


def compute_additional_stats(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Compute additional statistics not in metrics file."""
    stats = {
        "true_mean": float(np.mean(y_true)),
        "true_std": float(np.std(y_true)),
        "true_min": float(np.min(y_true)),
        "true_max": float(np.max(y_true)),
        "pred_mean": float(np.mean(y_pred)),
        "pred_std": float(np.std(y_pred)),
        "pred_min": float(np.min(y_pred)),
        "pred_max": float(np.max(y_pred)),
        "variance_ratio": float(np.std(y_pred) / np.std(y_true)),
        "spread_true": float(np.max(y_true) - np.min(y_true)),
        "spread_pred": float(np.max(y_pred) - np.min(y_pred)),
        "spread_ratio": float(
            (np.max(y_pred) - np.min(y_pred)) / (np.max(y_true) - np.min(y_true))
        ),
        "bias": float(np.mean(y_pred) - np.mean(y_true)),
    }

    # Compute correlation
    correlation = np.corrcoef(y_true, y_pred)[0, 1]
    stats["correlation"] = float(correlation)

    return stats


def analyze_run(base_dir: Path, run_config: Dict) -> Dict:
    """Analyze a single run."""
    pred_path = base_dir / run_config["pred_path"]
    metrics_path = base_dir / run_config["metrics_path"]
    train_path = base_dir / run_config["train_path"]

    # Load data
    y_true, y_pred = load_predictions(pred_path)
    metrics = load_metrics(metrics_path)
    training = load_training(train_path)

    # Compute additional stats
    stats = compute_additional_stats(y_true, y_pred)

    # Training stats
    val_losses = [float(row["val_loss"]) for row in training]
    train_losses = [float(row["train_loss"]) for row in training]

    best_val_idx = np.argmin(val_losses)
    training_stats = {
        "num_epochs": len(training),
        "initial_val_loss": val_losses[0],
        "best_val_loss": val_losses[best_val_idx],
        "best_epoch": best_val_idx + 1,
        "final_val_loss": val_losses[-1],
        "final_train_loss": train_losses[-1],
        "epochs_after_best": len(training) - best_val_idx - 1,
    }

    return {
        "name": run_config["name"],
        "desc": run_config["desc"],
        "metrics": metrics,
        "stats": stats,
        "training": training_stats,
        "predictions": (y_true, y_pred),
    }


def print_comparison_table(results: List[Dict]):
    """Print comparison table."""
    print("\n" + "=" * 100)
    print("COMPREHENSIVE COMPARISON OF ALL THREE RUNS")
    print("=" * 100)

    # Main metrics table
    print("\n" + "─" * 100)
    print("PERFORMANCE METRICS")
    print("─" * 100)
    print(f"{'Metric':<20} {'Run 1':<20} {'Run 2':<20} {'Run 3':<20} {'Best':<20}")
    print("─" * 100)

    metrics_to_compare = [
        ("R² Score", "r2", "higher", "{:.4f}"),
        ("MAE", "mae", "lower", "{:.4f}"),
        ("RMSE", "rmse", "lower", "{:.4f}"),
        ("MSE", "mse", "lower", "{:.4f}"),
        ("MAPE (%)", "mape", "lower", "{:.1f}"),
    ]

    for metric_name, key, better, fmt in metrics_to_compare:
        values = [r["metrics"][key] for r in results]
        formatted = [fmt.format(v) for v in values]

        if better == "higher":
            best_idx = np.argmax(values)
        else:
            best_idx = np.argmin(values)

        best_name = results[best_idx]["name"].split(":")[0]

        print(
            f"{metric_name:<20} {formatted[0]:<20} {formatted[1]:<20} {formatted[2]:<20} {best_name:<20}"
        )

    # Variance analysis
    print("\n" + "─" * 100)
    print("VARIANCE & DISTRIBUTION ANALYSIS")
    print("─" * 100)
    print(f"{'Metric':<20} {'Run 1':<20} {'Run 2':<20} {'Run 3':<20} {'Best':<20}")
    print("─" * 100)

    variance_metrics = [
        ("Pred Std Dev", "pred_std", "higher", "{:.4f}"),
        ("Variance Ratio", "variance_ratio", "higher", "{:.1%}"),
        ("Pred Spread", "spread_pred", "higher", "{:.4f}"),
        ("Spread Ratio", "spread_ratio", "higher", "{:.1%}"),
        ("Correlation", "correlation", "higher", "{:.4f}"),
        ("Bias", "bias", "abs_lower", "{:+.4f}"),
    ]

    for metric_name, key, better, fmt in variance_metrics:
        values = [r["stats"][key] for r in results]
        formatted = [fmt.format(v) for v in values]

        if better == "higher":
            best_idx = np.argmax(values)
        elif better == "abs_lower":
            best_idx = np.argmin(np.abs(values))
        else:
            best_idx = np.argmin(values)

        best_name = results[best_idx]["name"].split(":")[0]

        print(
            f"{metric_name:<20} {formatted[0]:<20} {formatted[1]:<20} {formatted[2]:<20} {best_name:<20}"
        )

    # Training dynamics
    print("\n" + "─" * 100)
    print("TRAINING DYNAMICS")
    print("─" * 100)
    print(f"{'Metric':<20} {'Run 1':<20} {'Run 2':<20} {'Run 3':<20}")
    print("─" * 100)

    train_metrics = [
        ("Total Epochs", "num_epochs", "{:d}"),
        ("Best Epoch", "best_epoch", "{:d}"),
        ("Initial Val Loss", "initial_val_loss", "{:.4f}"),
        ("Best Val Loss", "best_val_loss", "{:.4f}"),
        ("Final Val Loss", "final_val_loss", "{:.4f}"),
        ("Final Train Loss", "final_train_loss", "{:.4f}"),
        ("Epochs After Best", "epochs_after_best", "{:d}"),
    ]

    for metric_name, key, fmt in train_metrics:
        values = [r["training"][key] for r in results]
        formatted = [fmt.format(v) for v in values]
        print(
            f"{metric_name:<20} {formatted[0]:<20} {formatted[1]:<20} {formatted[2]:<20}"
        )


def print_variance_analysis(results: List[Dict]):
    """Print detailed variance collapse analysis."""
    print("\n" + "=" * 100)
    print("VARIANCE COLLAPSE ANALYSIS")
    print("=" * 100)

    for i, result in enumerate(results, 1):
        print(f"\n{result['name']}")
        print("─" * 100)
        stats = result["stats"]

        print(
            f"  True values:    mean={stats['true_mean']:.3f}, std={stats['true_std']:.4f}"
        )
        print(
            f"                  range=[{stats['true_min']:.3f}, {stats['true_max']:.3f}], spread={stats['spread_true']:.3f}"
        )
        print(
            f"  Predictions:    mean={stats['pred_mean']:.3f}, std={stats['pred_std']:.4f}"
        )
        print(
            f"                  range=[{stats['pred_min']:.3f}, {stats['pred_max']:.3f}], spread={stats['spread_pred']:.3f}"
        )
        print(f"  Variance ratio: {stats['variance_ratio']:.1%}")
        print(f"  Spread ratio:   {stats['spread_ratio']:.1%}")
        print(f"  Bias:           {stats['bias']:+.4f}")
        print(f"  Correlation:    {stats['correlation']:.4f}")

        # Severity assessment
        var_ratio = stats["variance_ratio"]
        if var_ratio < 0.05:
            severity = "🔴 CRITICAL - Near-total collapse"
        elif var_ratio < 0.20:
            severity = "🟠 SEVERE - Massive collapse"
        elif var_ratio < 0.50:
            severity = "🟡 HIGH - Significant collapse"
        elif var_ratio < 0.80:
            severity = "🟢 MODERATE - Some collapse"
        else:
            severity = "✅ GOOD - Minimal collapse"

        print(f"  Assessment:     {severity}")


def print_key_findings(results: List[Dict]):
    """Print key findings and recommendations."""
    print("\n" + "=" * 100)
    print("KEY FINDINGS")
    print("=" * 100)

    # Sort by R²
    sorted_by_r2 = sorted(results, key=lambda x: x["metrics"]["r2"], reverse=True)
    sorted_by_var = sorted(
        results, key=lambda x: x["stats"]["variance_ratio"], reverse=True
    )

    print("\n🏆 Best R² Score:")
    best_r2 = sorted_by_r2[0]
    print(f"   {best_r2['name']} → R² = {best_r2['metrics']['r2']:.4f}")
    print(f"   {best_r2['desc']}")

    print("\n🏆 Best Variance Ratio:")
    best_var = sorted_by_var[0]
    print(
        f"   {best_var['name']} → Variance Ratio = {best_var['stats']['variance_ratio']:.1%}"
    )
    print(f"   {best_var['desc']}")

    print("\n📉 Worst Performer:")
    worst = sorted_by_r2[-1]
    print(f"   {worst['name']} → R² = {worst['metrics']['r2']:.4f}")
    print(f"   Variance ratio: {worst['stats']['variance_ratio']:.1%}")

    print("\n🔍 Critical Observations:")
    print("   1. All three runs achieved similar validation loss (~0.51-0.53)")
    print("   2. But R² varies from -0.045 to -0.203 (4.5x difference!)")
    print("   3. Run 1 (broken SSL) has 10x better variance ratio than Runs 2-3")
    print("   4. Initialization 'fixes' in Runs 2-3 made collapse WORSE")
    print("   5. Validation loss is NOT a good metric for this problem")

    print("\n⚠️  Main Problem:")
    print("   Loss function (MSE/MAE) doesn't penalize variance collapse.")
    print("   Model can achieve low loss by predicting constant values.")

    print("\n✅ What Actually Worked:")
    print("   - Run 1's larger initial weights (default init, std≈0.5)")
    print("   - (Ironically) broken SSL pretraining may have added beneficial noise")

    print("\n❌ What Made It Worse:")
    print("   - Smaller initial weights (std=0.01 or 0.1)")
    print("   - Removing SSL without addressing underlying collapse issue")


def print_recommendations():
    """Print actionable recommendations."""
    print("\n" + "=" * 100)
    print("RECOMMENDATIONS")
    print("=" * 100)

    print("\n🔴 CRITICAL PRIORITY:")
    print("   1. Add variance-preserving term to loss function:")
    print("      loss = mse + 0.5 * (1 - pred_var/target_var)**2")
    print("   2. Change early stopping to track R² instead of validation loss")
    print("   3. Revert to Run 1 initialization (default/larger weights)")

    print("\n🟡 HIGH PRIORITY:")
    print("   4. Reduce temporal_frames from 7 to 3 or 5")
    print("   5. Set overweight_factor=1.0 (no overweighting)")
    print("   6. Add prediction diversity monitoring during training")

    print("\n🟢 EXPERIMENTS TO RUN:")
    print("   7. Baseline: Run 1 config + variance loss")
    print("   8. Ablation: temporal_frames ∈ {3, 5, 7}")
    print("   9. Fix SSL pretraining properly, then compare SSL vs no-SSL")

    print("\n📊 What Success Looks Like:")
    print("   - R² > 0.3 (minimum acceptable)")
    print("   - Variance ratio > 70%")
    print("   - Prediction spread > 1.0")
    print("   - RMSE < 0.35")


def main():
    """Main comparison routine."""
    base_dir = Path(__file__).parent

    print("=" * 100)
    print("TRAINING RUNS COMPARISON")
    print("=" * 100)
    print("\nAnalyzing three Tier 1 training runs...")

    # Analyze all runs
    results = []
    for run_config in RUNS:
        try:
            result = analyze_run(base_dir, run_config)
            results.append(result)
            print(f"✓ Loaded {result['name']}")
        except Exception as e:
            print(f"✗ Failed to load {run_config['name']}: {e}")

    if len(results) < 3:
        print("\n⚠️  Warning: Not all runs could be loaded!")
        return

    # Print all comparisons
    print_comparison_table(results)
    print_variance_analysis(results)
    print_key_findings(results)
    print_recommendations()

    print("\n" + "=" * 100)
    print("Analysis complete. See THIRD_RUN_ANALYSIS.md for detailed report.")
    print("=" * 100 + "\n")


if __name__ == "__main__":
    main()
