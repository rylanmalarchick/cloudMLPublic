#!/usr/bin/env python3
"""
Compare GPU-Optimized Run vs Baseline Run

This script compares the results from two Phase 3 fine-tuning runs:
1. Baseline run (smaller batches, no AMP)
2. GPU-optimized run (larger batches, AMP enabled)

Usage:
    python scripts/compare_runs.py

Output:
    - Comparison table printed to console
    - Optional: Save comparison plot
"""

import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def load_results(checkpoint_path):
    """
    Load results from a checkpoint file.

    Returns dict with metrics if found, else None.
    """
    import torch

    if not Path(checkpoint_path).exists():
        return None

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        return checkpoint.get("val_metrics", {})
    except Exception as e:
        print(f"Error loading {checkpoint_path}: {e}")
        return None


def compare_runs(
    baseline_dir="outputs/cbh_finetune_baseline", optimized_dir="outputs/cbh_finetune"
):
    """
    Compare baseline and GPU-optimized runs.
    """

    print("=" * 80)
    print("PHASE 3 RUN COMPARISON: Baseline vs GPU-Optimized")
    print("=" * 80)
    print()

    # Define what we're comparing
    runs = {
        "Baseline": {
            "dir": baseline_dir,
            "batch_sizes": "128 / 32 (Stage 1 / Stage 2)",
            "amp": "Disabled",
            "gpu_usage": "~2GB",
        },
        "GPU-Optimized": {
            "dir": optimized_dir,
            "batch_sizes": "256 / 128 (Stage 1 / Stage 2)",
            "amp": "Enabled",
            "gpu_usage": "~6-7GB",
        },
    }

    # Try to load actual results from checkpoints
    results = {}
    for name, info in runs.items():
        checkpoint_path = Path(info["dir"]) / "checkpoints" / "final_model.pth"
        metrics = load_results(checkpoint_path)
        results[name] = metrics

    # Print configuration comparison
    print("CONFIGURATION COMPARISON")
    print("-" * 80)
    print(f"{'Setting':<20} {'Baseline':<30} {'GPU-Optimized':<30}")
    print("-" * 80)
    print(
        f"{'Batch Sizes':<20} {runs['Baseline']['batch_sizes']:<30} {runs['GPU-Optimized']['batch_sizes']:<30}"
    )
    print(
        f"{'Mixed Precision':<20} {runs['Baseline']['amp']:<30} {runs['GPU-Optimized']['amp']:<30}"
    )
    print(
        f"{'GPU Memory':<20} {runs['Baseline']['gpu_usage']:<30} {runs['GPU-Optimized']['gpu_usage']:<30}"
    )
    print()

    # Print performance comparison
    print("PERFORMANCE COMPARISON")
    print("-" * 80)

    # Classical baseline for reference
    classical_baseline = {"r2": 0.7464, "mae": 0.1265, "rmse": 0.1929}

    # Manual baseline results (from your first run)
    baseline_results = {"r2": 0.4791, "mae": 0.2126, "rmse": 0.2714}

    print(
        f"{'Metric':<15} {'Classical ML':<15} {'Baseline Run':<15} {'GPU-Optimized':<15} {'Improvement':<15}"
    )
    print("-" * 80)

    metrics_to_compare = ["r2", "mae", "rmse"]
    metric_names = {"r2": "R²", "mae": "MAE (km)", "rmse": "RMSE (km)"}

    for metric in metrics_to_compare:
        classical = classical_baseline.get(metric, 0)
        baseline = baseline_results.get(metric, 0)

        # Try to get optimized results from checkpoint
        if results.get("GPU-Optimized"):
            optimized = results["GPU-Optimized"].get(metric, 0)
        else:
            optimized = 0  # Will be updated when run completes

        # Calculate improvement
        if metric == "r2":
            improvement = optimized - baseline if optimized > 0 else "TBD"
        else:
            improvement = baseline - optimized if optimized > 0 else "TBD"

        # Format values
        classical_str = f"{classical:.4f}"
        baseline_str = f"{baseline:.4f}"
        optimized_str = f"{optimized:.4f}" if optimized > 0 else "TBD"
        improvement_str = (
            f"{improvement:+.4f}" if isinstance(improvement, float) else improvement
        )

        print(
            f"{metric_names[metric]:<15} {classical_str:<15} {baseline_str:<15} {optimized_str:<15} {improvement_str:<15}"
        )

    print()
    print("-" * 80)
    print()

    # Performance interpretation
    print("INTERPRETATION")
    print("-" * 80)

    if results.get("GPU-Optimized"):
        opt_r2 = results["GPU-Optimized"].get("r2", 0)
        base_r2 = baseline_results["r2"]

        if opt_r2 > 0:
            improvement = opt_r2 - base_r2
            improvement_pct = (improvement / base_r2) * 100

            print(f"R² Change: {improvement:+.4f} ({improvement_pct:+.1f}%)")
            print()

            if opt_r2 >= 0.60:
                print("🎉 EXCELLENT! Reached 'GOOD' threshold (R² ≥ 0.60)")
                print("   → GPU optimization made a significant difference!")
                print("   → This is publishable performance")
            elif improvement > 0.02:
                print("✅ MEANINGFUL IMPROVEMENT!")
                print("   → Larger batches + AMP helped convergence")
                print("   → Still behind classical ML, but progress made")
            elif improvement > 0:
                print("👍 MODEST IMPROVEMENT")
                print("   → Small gains from better optimization")
                print("   → Main limiting factor is likely dataset size")
            else:
                print("➖ NO IMPROVEMENT")
                print("   → Batch size / AMP didn't help for this task")
                print(
                    "   → Performance bottleneck is elsewhere (data, architecture, etc.)"
                )
        else:
            print("⏳ GPU-Optimized run not completed yet.")
            print("   Run ./scripts/run_phase3_finetune.sh to generate results.")
    else:
        print("⏳ GPU-Optimized run not found.")
        print("   Run ./scripts/run_phase3_finetune.sh to compare results.")

    print()
    print("=" * 80)

    return results


def main():
    parser = argparse.ArgumentParser(description="Compare Phase 3 runs")
    parser.add_argument(
        "--baseline-dir",
        default="outputs/cbh_finetune_baseline",
        help="Directory with baseline run results",
    )
    parser.add_argument(
        "--optimized-dir",
        default="outputs/cbh_finetune",
        help="Directory with GPU-optimized run results",
    )

    args = parser.parse_args()

    results = compare_runs(args.baseline_dir, args.optimized_dir)


if __name__ == "__main__":
    main()
