#!/usr/bin/env python3
"""
Simple Diagnostic Analysis for Tier 1 Training Results
======================================================

Analyzes the disappointing Tier 1 results (R² = -0.0457)
without requiring pandas dependency.
"""

import json
import csv
from pathlib import Path

# Set up paths
BASE_DIR = Path(__file__).parent
LOGS_DIR = BASE_DIR / "logs" / "csv"


def load_metrics():
    """Load metrics JSON."""
    with open(LOGS_DIR / "metrics_final_overweighted_tier1_final.json", "r") as f:
        return json.load(f)


def load_csv(filename):
    """Load CSV file into list of dicts."""
    data = []
    with open(LOGS_DIR / filename, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data


def analyze_predictions():
    """Analyze predictions distribution."""
    predictions = load_csv("predictions_final_overweighted_tier1_final.csv")

    y_true = [float(p["y_true"]) for p in predictions]
    y_pred = [float(p["y_pred"]) for p in predictions]

    print("\n" + "=" * 70)
    print("PREDICTION DISTRIBUTION ANALYSIS")
    print("=" * 70)

    print(f"\nTrue Values Statistics:")
    print(f"  Range: [{min(y_true):.3f}, {max(y_true):.3f}]")
    print(f"  Spread: {max(y_true) - min(y_true):.3f}")
    print(f"  Mean: {sum(y_true) / len(y_true):.3f}")

    print(f"\nPredicted Values Statistics:")
    print(f"  Range: [{min(y_pred):.3f}, {max(y_pred):.3f}]")
    print(f"  Spread: {max(y_pred) - min(y_pred):.3f}")
    print(f"  Mean: {sum(y_pred) / len(y_pred):.3f}")

    # Calculate variance
    true_mean = sum(y_true) / len(y_true)
    pred_mean = sum(y_pred) / len(y_pred)

    true_var = sum((x - true_mean) ** 2 for x in y_true) / len(y_true)
    pred_var = sum((x - pred_mean) ** 2 for x in y_pred) / len(y_pred)

    true_std = true_var**0.5
    pred_std = pred_var**0.5

    print(f"\nVariance Analysis:")
    print(f"  True Std Dev: {true_std:.4f}")
    print(f"  Pred Std Dev: {pred_std:.4f}")
    print(f"  Variance Ratio: {pred_std / true_std:.1%}")

    if pred_std / true_std < 0.5:
        print(f"  ⚠️  CRITICAL: Severe variance collapse!")

    # Bias by range
    print(f"\nBias Analysis by True Value Range:")
    ranges = [
        ("Very Low (<0.4)", 0.0, 0.4),
        ("Low (0.4-0.6)", 0.4, 0.6),
        ("Medium (0.6-0.8)", 0.6, 0.8),
        ("High (0.8-1.0)", 0.8, 1.0),
        ("Very High (>1.0)", 1.0, 3.0),
    ]

    for label, low, high in ranges:
        subset_true = [y_true[i] for i in range(len(y_true)) if low <= y_true[i] < high]
        subset_pred = [y_pred[i] for i in range(len(y_pred)) if low <= y_true[i] < high]

        if len(subset_true) > 0:
            bias = sum(
                subset_pred[i] - subset_true[i] for i in range(len(subset_true))
            ) / len(subset_true)
            mean_true = sum(subset_true) / len(subset_true)
            mean_pred = sum(subset_pred) / len(subset_pred)
            print(
                f"  {label}: n={len(subset_true)}, bias={bias:+.3f}, "
                f"true_mean={mean_true:.3f}, pred_mean={mean_pred:.3f}"
            )

    return pred_std / true_std


def analyze_training():
    """Analyze training progression."""
    train_data = load_csv("final_overweighted_tier1_final.csv")
    pretrain_data = load_csv("pretrain_30Oct24_tier1_tuned_20251019_164826.csv")

    print("\n" + "=" * 70)
    print("TRAINING PROGRESSION ANALYSIS")
    print("=" * 70)

    print(f"\nPretraining ({len(pretrain_data)} epochs):")
    print(f"  Initial train loss: {float(pretrain_data[0]['train_loss']):.4f}")
    print(f"  Final train loss: {float(pretrain_data[-1]['train_loss']):.4f}")

    pretrain_val_losses = [float(row["val_loss"]) for row in pretrain_data]
    min_val_loss = min(pretrain_val_losses)
    min_val_epoch = pretrain_val_losses.index(min_val_loss) + 1
    print(f"  Best val loss: {min_val_loss:.4f} (epoch {min_val_epoch})")
    print(f"  Final val loss: {float(pretrain_data[-1]['val_loss']):.4f}")

    print(f"\nFinal Training ({len(train_data)} epochs - early stopped):")
    print(f"  Initial train loss: {float(train_data[0]['train_loss']):.4f}")
    print(f"  Final train loss: {float(train_data[-1]['train_loss']):.4f}")

    val_losses = [float(row["val_loss"]) for row in train_data]
    min_val_loss = min(val_losses)
    min_val_epoch = val_losses.index(min_val_loss) + 1
    print(f"  Best val loss: {min_val_loss:.4f} (epoch {min_val_epoch})")
    print(f"  Final val loss: {float(train_data[-1]['val_loss']):.4f}")

    # Check overfitting
    best_idx = val_losses.index(min(val_losses))
    gap_at_best = float(train_data[best_idx]["train_loss"]) - float(
        train_data[best_idx]["val_loss"]
    )
    final_gap = float(train_data[-1]["train_loss"]) - float(train_data[-1]["val_loss"])

    print(f"\nTrain-Val Gap:")
    print(f"  At best epoch: {gap_at_best:+.4f}")
    print(f"  At final epoch: {final_gap:+.4f}")

    if final_gap > 2.0:
        print(f"  ⚠️  Large positive gap suggests overfitting")


def print_diagnosis(metrics, variance_ratio):
    """Print diagnosis of issues."""
    print("\n" + "=" * 70)
    print("DIAGNOSIS & CRITICAL ISSUES")
    print("=" * 70)

    print(f"\n[CRITICAL] Negative R² Score")
    print(f"  R² = {metrics['r2']:.4f}")
    print(f"  This means the model performs WORSE than just predicting the mean!")
    print(f"  Likely causes:")
    print(f"    • Model failed to learn meaningful patterns")
    print(f"    • Test distribution differs from train/val")
    print(f"    • Architecture/hyperparameters fundamentally mismatched")

    print(f"\n[CRITICAL] Severe Variance Collapse")
    print(f"  Predictions have only {variance_ratio:.1%} of true variance")
    print(f"  Model is regressing toward the mean, not learning variations")
    print(f"  Likely causes:")
    print(f"    • 7-frame temporal smoothing is over-averaging")
    print(f"    • Pretrained encoder may be learning wrong features")
    print(f"    • Loss function doesn't penalize variance collapse")
    print(f"    • Over-regularization")

    print(f"\n[HIGH] Extreme MAPE")
    print(f"  MAPE = {metrics['mape']:.1f}%")
    print(f"  Model has massive relative errors, especially for low values")

    pred_range = metrics["prediction_range"][1] - metrics["prediction_range"][0]
    true_range = metrics["true_range"][1] - metrics["true_range"][0]
    print(f"\n[HIGH] Narrow Prediction Range")
    print(f"  Predictions: {pred_range:.3f} vs True: {true_range:.3f}")
    print(f"  Model is playing it safe, predicting middle values only")


def print_recommendations():
    """Print recommended fixes."""
    print("\n" + "=" * 70)
    print("RECOMMENDED FIXES (Priority Order)")
    print("=" * 70)

    print(f"\n[Priority 1] Reduce Temporal Smoothing")
    print(f"  → Change temporal_frames from 7 to 3 or 4")
    print(f"  → 7 frames may be over-averaging and destroying variation signals")
    print(f"  → Could also try temporal_frames=5 with stride=2")

    print(f"\n[Priority 2] Fix Loss Function")
    print(f"  → Add variance preservation term to loss")
    print(f"  → Try Huber loss instead of MSE")
    print(f"  → Reduce overweight_factor from 2.0 to 1.5")
    print(f"  → Consider variance-weighted MSE")

    print(f"\n[Priority 3] Reevaluate Pretraining")
    print(f"  → Run WITHOUT pretraining as baseline comparison")
    print(f"  → Pretraining val loss was high (~1.48) - may not have converged")
    print(f"  → Try unfreezing encoder earlier")
    print(f"  → Consider different pretraining objective")

    print(f"\n[Priority 4] Tune Multi-Scale Attention")
    print(f"  → Verify attention isn't collapsing to uniform weights")
    print(f"  → Try 8 attention heads instead of 4")
    print(f"  → Add attention visualization")
    print(f"  → Consider separate spatial/temporal attention")

    print(f"\n[Priority 5] Hyperparameter Adjustments")
    print(f"  → Increase learning_rate from 0.0005 to 0.001")
    print(f"  → Try different warmup schedules")
    print(f"  → Reduce early_stopping_patience")

    print(f"\n[Priority 6] Run Diagnostic Experiments")
    print(f"  → Baseline (Option 1) for comparison")
    print(f"  → Ablation: Tier 1 WITHOUT pretraining")
    print(f"  → Ablation: Different temporal_frames (3, 5, 7)")
    print(f"  → Monitor prediction distribution during training")


def print_summary():
    """Print executive summary."""
    print("\n" + "=" * 70)
    print("EXECUTIVE SUMMARY")
    print("=" * 70)

    print("""
Current Status: ❌ TIER 1 SEVERELY UNDERPERFORMING

Key Findings:
  1. Negative R² (-0.0457) = worse than predicting the mean
  2. Severe variance collapse (~30% of true variance)
  3. Systematic overprediction for low optical depths
  4. Training completed but model failed to learn real patterns

Root Cause (Most Likely):
  The combination of:
    • 7-frame temporal smoothing (over-averaging)
    • Pretrained encoder (may have learned wrong features)
    • Multi-scale attention (needs tuning)
  ...is causing the model to over-regularize and predict
  conservative near-mean values.

Immediate Action Items:
  1. Run baseline (Option 1) to establish comparison point
  2. Try Tier 1 with temporal_frames=3 (less smoothing)
  3. Run Tier 1 WITHOUT pretraining to isolate impact
  4. Add variance-preserving loss term

Expected Improvement:
  With fixes, we should achieve:
    • R² > 0.3 (minimum acceptable)
    • Prediction variance > 70% of true variance
    • RMSE < 0.35 (vs current 0.49)
""")


def main():
    """Main analysis routine."""
    print("=" * 70)
    print("TIER 1 RESULTS DIAGNOSTIC ANALYSIS")
    print("=" * 70)

    # Load data
    metrics = load_metrics()

    # Print basic metrics
    print(f"\nFinal Test Metrics:")
    print(f"  Loss: {metrics['loss']:.4f}")
    print(f"  MAE: {metrics['mae']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  R²: {metrics['r2']:.4f} ❌")
    print(f"  MAPE: {metrics['mape']:.1f}%")

    # Run analyses
    variance_ratio = analyze_predictions()
    analyze_training()
    print_diagnosis(metrics, variance_ratio)
    print_recommendations()
    print_summary()

    print("\n" + "=" * 70)
    print("Analysis complete. See recommendations above.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
