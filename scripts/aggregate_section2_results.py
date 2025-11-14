#!/usr/bin/env python3
"""
SECTION 2.3: AGGREGATE VARIANCE_LAMBDA EXPERIMENT RESULTS
===========================================================

This script aggregates the results from all Section 2 experiments
(baseline + hyperparameter sweep) and generates Table 2 for the research program.

It extracts key metrics from each training run:
  - Final Validation R²
  - Final Variance Ratio (%)
  - Average Base Loss (Huber loss component)
  - Average Variance Loss (variance penalty component)
  - Training Stability (stable, minor instability, exploded)

Usage:
    python scripts/aggregate_section2_results.py

Output:
    - diagnostics/results/section2_table2.csv
    - diagnostics/results/section2_summary.json
    - diagnostics/results/section2_table2.txt (formatted for paper)
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import re
import glob


def parse_log_file(log_path):
    """
    Parse a training log file to extract final metrics.

    Returns:
        dict with keys: final_r2, final_variance_ratio, avg_base_loss,
                       avg_variance_loss, stability
    """
    results = {
        "final_r2": None,
        "final_variance_ratio": None,
        "avg_base_loss": None,
        "avg_variance_loss": None,
        "stability": "Unknown",
    }

    if not Path(log_path).exists():
        return results

    with open(log_path, "r") as f:
        lines = f.readlines()

    # Extract metrics from the last few epochs
    r2_values = []
    variance_ratios = []
    base_losses = []
    variance_losses = []

    has_nan = False
    has_explosion = False

    for line in lines:
        # Look for validation R²
        if "Validation R²" in line or "Val R²" in line or "val_r2" in line.lower():
            match = re.search(r"[-]?\d+\.\d+", line)
            if match:
                val = float(match.group())
                r2_values.append(val)
                if np.isnan(val) or np.isinf(val):
                    has_nan = True

        # Look for variance ratio
        if "variance_ratio" in line.lower() or "Variance Ratio" in line:
            match = re.search(r"\d+\.\d+", line)
            if match:
                val = float(match.group())
                variance_ratios.append(val)

        # Look for base loss
        if "base_loss" in line.lower() or "Base Loss" in line:
            match = re.search(r"\d+\.\d+", line)
            if match:
                val = float(match.group())
                base_losses.append(val)
                if val > 100:  # Arbitrary threshold for explosion
                    has_explosion = True

        # Look for variance loss
        if "variance_loss" in line.lower() or "Variance Loss" in line:
            match = re.search(r"\d+\.\d+", line)
            if match:
                val = float(match.group())
                variance_losses.append(val)
                if val > 100:  # Arbitrary threshold for explosion
                    has_explosion = True

        # Check for NaN or explosion indicators
        if "nan" in line.lower() or "inf" in line.lower():
            has_nan = True

    # Calculate final and average values
    if r2_values:
        results["final_r2"] = r2_values[-1]

    if variance_ratios:
        results["final_variance_ratio"] = variance_ratios[-1]

    if base_losses:
        # Average over last 5 epochs or all if fewer
        results["avg_base_loss"] = np.mean(base_losses[-5:])

    if variance_losses:
        results["avg_variance_loss"] = np.mean(variance_losses[-5:])

    # Determine stability
    if has_nan or has_explosion:
        results["stability"] = "Exploded"
    elif (
        results["final_variance_ratio"] is not None
        and results["final_variance_ratio"] < 20
    ):
        results["stability"] = "Stable (Collapsed)"
    elif results["final_r2"] is not None and results["final_r2"] > 0:
        results["stability"] = "Stable"
    else:
        results["stability"] = "Minor Instability"

    return results


def find_latest_log(config_name):
    """
    Find the most recent log file for a given config.
    """
    # Common log locations
    log_dirs = ["logs", "/content/drive/MyDrive/CloudML/logs", "diagnostics/results"]

    for log_dir in log_dirs:
        if Path(log_dir).exists():
            # Look for logs matching the config name
            pattern = f"{log_dir}/*{config_name}*.log"
            matches = glob.glob(pattern)
            if matches:
                # Return most recent
                return max(matches, key=lambda p: Path(p).stat().st_mtime)

            # Also try without config name prefix
            pattern = f"{log_dir}/*.log"
            matches = glob.glob(pattern)
            if matches:
                # Return most recent
                return max(matches, key=lambda p: Path(p).stat().st_mtime)

    return None


def main():
    print("=" * 70)
    print("SECTION 2.3: AGGREGATING VARIANCE_LAMBDA EXPERIMENT RESULTS")
    print("=" * 70)
    print()

    # Define experiment configurations
    experiments = [
        {
            "name": "Baseline (λ=0.0)",
            "lambda": 0.0,
            "config": "section2_baseline_collapse",
        },
        {"name": "λ=0.5", "lambda": 0.5, "config": "section2_lambda_0.5"},
        {"name": "λ=1.0", "lambda": 1.0, "config": "section2_lambda_1.0"},
        {"name": "λ=2.0", "lambda": 2.0, "config": "section2_lambda_2.0"},
        {"name": "λ=5.0", "lambda": 5.0, "config": "section2_lambda_5.0"},
        {"name": "λ=10.0", "lambda": 10.0, "config": "section2_lambda_10.0"},
    ]

    # Collect results
    results_data = []

    for exp in experiments:
        print(f"Processing {exp['name']}...")

        # Try to find the log file
        log_file = find_latest_log(exp["config"])

        if log_file:
            print(f"  Found log: {log_file}")
            metrics = parse_log_file(log_file)
        else:
            print(f"  WARNING: No log file found for {exp['config']}")
            metrics = {
                "final_r2": "[pending]",
                "final_variance_ratio": "[pending]",
                "avg_base_loss": "[pending]",
                "avg_variance_loss": "[pending]",
                "stability": "[pending]",
            }

        results_data.append(
            {
                "variance_lambda": exp["lambda"],
                "experiment": exp["name"],
                "final_val_r2": metrics["final_r2"],
                "final_variance_ratio": metrics["final_variance_ratio"],
                "avg_base_loss": metrics["avg_base_loss"],
                "avg_variance_loss": metrics["avg_variance_loss"],
                "stability": metrics["stability"],
            }
        )

    # Create DataFrame
    df = pd.DataFrame(results_data)

    # Create output directory
    output_dir = Path("diagnostics/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save CSV
    csv_path = output_dir / "section2_table2.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV to: {csv_path}")

    # Save JSON summary
    summary = {
        "section": "Section 2: Model Collapse Investigation",
        "experiments": results_data,
        "best_lambda": None,
        "best_r2": None,
        "recommendation": None,
    }

    # Find best performing lambda (excluding baseline)
    valid_results = [
        r
        for r in results_data
        if r["final_val_r2"] not in [None, "[pending]"] and r["variance_lambda"] > 0
    ]
    if valid_results:
        best = max(valid_results, key=lambda x: x["final_val_r2"])
        summary["best_lambda"] = best["variance_lambda"]
        summary["best_r2"] = best["final_val_r2"]
        summary["recommendation"] = (
            f"Use variance_lambda = {best['variance_lambda']} for all subsequent experiments (Section 3 onwards)"
        )

    json_path = output_dir / "section2_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved JSON summary to: {json_path}")

    # Create formatted table for paper
    table_path = output_dir / "section2_table2.txt"
    with open(table_path, "w") as f:
        f.write("=" * 100 + "\n")
        f.write(
            "TABLE 2: Impact of variance_lambda on Model Stability and Performance\n"
        )
        f.write("=" * 100 + "\n\n")
        f.write(df.to_string(index=False))
        f.write("\n\n")

        if summary["best_lambda"] is not None:
            f.write("=" * 100 + "\n")
            f.write("RECOMMENDATION\n")
            f.write("=" * 100 + "\n")
            f.write(f"Best variance_lambda: {summary['best_lambda']}\n")
            f.write(f"Best validation R²: {summary['best_r2']:.4f}\n")
            f.write(f"\n{summary['recommendation']}\n")

        f.write("\n")

    print(f"Saved formatted table to: {table_path}")

    # Display results
    print("\n" + "=" * 70)
    print("TABLE 2: VARIANCE_LAMBDA HYPERPARAMETER SWEEP RESULTS")
    print("=" * 70)
    print(df.to_string(index=False))
    print("\n" + "=" * 70)

    if summary["best_lambda"] is not None:
        print("\n RECOMMENDATION")
        print("=" * 70)
        print(f"Best variance_lambda: {summary['best_lambda']}")
        print(f"Best validation R²: {summary['best_r2']:.4f}")
        print(f"\n{summary['recommendation']}")
        print("=" * 70)
    else:
        print("\n  WARNING: No completed experiments found.")
        print("   Run the Section 2 experiments first:")
        print("   bash scripts/run_section2_experiments.sh")
        print("=" * 70)

    print("\n Section 2.3 aggregation complete!")
    print("\nNext steps:")
    print("  1. Review the results above")
    print("  2. Run: python scripts/plot_section2_distributions.py")
    print("  3. Select optimal variance_lambda")
    print("  4. Proceed to Section 3: Architectural Ablation Study")
    print()


if __name__ == "__main__":
    main()
