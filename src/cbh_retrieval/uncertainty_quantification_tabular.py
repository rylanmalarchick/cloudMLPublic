#!/usr/bin/env python3
"""
Sprint 6 - Phase 1, Task 1.2: Uncertainty Quantification (Tabular Adaptation)

This script performs uncertainty quantification on the GBDT baseline model
using quantile regression to estimate prediction intervals.

Implements stratified 5-fold CV with quantile regression (10th, 50th, 90th percentiles)
to provide calibrated uncertainty estimates.

Author: Sprint 6 Agent
Date: 2025
"""

import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

print("=" * 80)
print("Sprint 6 - Phase 1, Task 1.2: Uncertainty Quantification (Tabular)")
print("=" * 80)
print(f"Project Root: {PROJECT_ROOT}")

# Paths
INTEGRATED_FEATURES = (
    PROJECT_ROOT / "outputs/preprocessed_data/Integrated_Features.hdf5"
)
OUTPUT_DIR = PROJECT_ROOT / "."
FIGURES_DIR = OUTPUT_DIR / "figures/uncertainty"
REPORTS_DIR = OUTPUT_DIR / "reports"

# Create directories
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

print(f"Integrated Features: {INTEGRATED_FEATURES}")
print(f"Output Directory: {OUTPUT_DIR}")


class UncertaintyQuantifier:
    """Uncertainty quantification using quantile regression."""

    def __init__(self, n_folds=5, random_seed=42, confidence_level=0.90):
        self.n_folds = n_folds
        self.random_seed = random_seed
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level

        # Quantiles for prediction intervals
        self.lower_quantile = self.alpha / 2
        self.upper_quantile = 1 - (self.alpha / 2)

        self.results = {
            "folds": [],
            "aggregated_metrics": {},
            "metadata": {
                "model_type": "GradientBoostingQuantileRegressor",
                "n_folds": n_folds,
                "random_seed": random_seed,
                "confidence_level": confidence_level,
                "lower_quantile": self.lower_quantile,
                "upper_quantile": self.upper_quantile,
                "timestamp": datetime.now().isoformat(),
            },
        }

    def load_data(self, hdf5_path):
        """Load tabular features and labels from integrated features file."""
        print("\n" + "=" * 80)
        print("Loading Dataset")
        print("=" * 80)

        with h5py.File(hdf5_path, "r") as f:
            # Load labels
            cbh_km = f["metadata/cbh_km"][:]
            flight_ids = f["metadata/flight_id"][:]

            # Load atmospheric features
            era5_features = f["atmospheric_features/era5_features"][:]
            era5_feature_names = [
                name.decode("utf-8") if isinstance(name, bytes) else name
                for name in f["atmospheric_features/era5_feature_names"][:]
            ]

            # Load geometric features
            geometric_features = {}
            for key in f["geometric_features"].keys():
                if key != "derived_geometric_H":
                    data = f[f"geometric_features/{key}"][:]
                    if data.ndim == 1:
                        geometric_features[key] = data

        print(f"✓ Loaded {len(cbh_km)} samples")
        print(f"✓ CBH range: [{cbh_km.min():.3f}, {cbh_km.max():.3f}] km")

        # Combine all features
        feature_list = [era5_features]
        feature_names = era5_feature_names.copy()

        for name, values in geometric_features.items():
            feature_list.append(values.reshape(-1, 1))
            feature_names.append(name)

        X = np.hstack(feature_list)
        y = cbh_km

        print(f"✓ Total feature matrix shape: {X.shape}")

        return X, y, flight_ids

    def create_stratified_bins(self, y, n_bins=10):
        """Create stratified bins for CBH values."""
        bins = np.percentile(y, np.linspace(0, 100, n_bins + 1))
        bins = np.unique(bins)
        bin_indices = np.digitize(y, bins) - 1
        bin_indices = np.clip(bin_indices, 0, len(bins) - 2)
        return bin_indices

    def train_quantile_models(self, X_train, y_train, fold_idx):
        """Train three quantile regression models (lower, median, upper)."""
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        models = {}

        # Lower bound (e.g., 5th percentile for 90% CI)
        print(f"  Training lower quantile model ({self.lower_quantile:.2f})...")
        models["lower"] = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            min_samples_split=5,
            min_samples_leaf=2,
            subsample=0.8,
            loss="quantile",
            alpha=self.lower_quantile,
            random_state=self.random_seed + fold_idx,
            verbose=0,
        )
        models["lower"].fit(X_train_scaled, y_train)

        # Median (50th percentile)
        print(f"  Training median model (0.50)...")
        models["median"] = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            min_samples_split=5,
            min_samples_leaf=2,
            subsample=0.8,
            loss="quantile",
            alpha=0.5,
            random_state=self.random_seed + fold_idx,
            verbose=0,
        )
        models["median"].fit(X_train_scaled, y_train)

        # Upper bound (e.g., 95th percentile for 90% CI)
        print(f"  Training upper quantile model ({self.upper_quantile:.2f})...")
        models["upper"] = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            min_samples_split=5,
            min_samples_leaf=2,
            subsample=0.8,
            loss="quantile",
            alpha=self.upper_quantile,
            random_state=self.random_seed + fold_idx,
            verbose=0,
        )
        models["upper"].fit(X_train_scaled, y_train)

        return models, scaler

    def predict_with_uncertainty(self, models, scaler, X):
        """Generate predictions with uncertainty bounds."""
        X_scaled = scaler.transform(X)

        pred_lower = models["lower"].predict(X_scaled)
        pred_median = models["median"].predict(X_scaled)
        pred_upper = models["upper"].predict(X_scaled)

        # Ensure proper ordering
        pred_lower = np.minimum(pred_lower, pred_median)
        pred_upper = np.maximum(pred_upper, pred_median)

        # Uncertainty as interval width
        uncertainty = pred_upper - pred_lower

        return pred_median, pred_lower, pred_upper, uncertainty

    def evaluate_fold(self, X_train, y_train, X_val, y_val, fold_idx):
        """Train and evaluate uncertainty quantification for one fold."""
        print(f"\n{'=' * 80}")
        print(f"Fold {fold_idx + 1}/{self.n_folds}")
        print(f"{'=' * 80}")
        print(f"Train samples: {len(y_train)}, Val samples: {len(y_val)}")

        # Train quantile models
        models, scaler = self.train_quantile_models(X_train, y_train, fold_idx)

        # Predict on validation set
        pred_median, pred_lower, pred_upper, uncertainty = (
            self.predict_with_uncertainty(models, scaler, X_val)
        )

        # Compute metrics
        metrics = self._compute_uq_metrics(
            y_val, pred_median, pred_lower, pred_upper, uncertainty
        )

        # Store results
        fold_result = {
            "fold": fold_idx + 1,
            "metrics": metrics,
            "y_true": y_val.tolist(),
            "y_pred": pred_median.tolist(),
            "y_lower": pred_lower.tolist(),
            "y_upper": pred_upper.tolist(),
            "uncertainty": uncertainty.tolist(),
        }

        return fold_result

    def _compute_uq_metrics(self, y_true, y_pred, y_lower, y_upper, uncertainty):
        """Compute uncertainty quantification metrics."""
        # Standard metrics
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        # Coverage: fraction of true values within prediction interval
        coverage = np.mean((y_true >= y_lower) & (y_true <= y_upper))

        # Average interval width
        avg_interval_width = np.mean(y_upper - y_lower)

        # Calibration: correlation between uncertainty and absolute error
        abs_error = np.abs(y_true - y_pred)
        if np.std(uncertainty) > 0 and np.std(abs_error) > 0:
            uncertainty_correlation = np.corrcoef(uncertainty, abs_error)[0, 1]
        else:
            uncertainty_correlation = 0.0

        # Sharpness: mean uncertainty
        mean_uncertainty = np.mean(uncertainty)

        # Print metrics
        print(f"\nUncertainty Quantification Metrics:")
        print(f"  R² = {r2:.4f}")
        print(f"  MAE = {mae * 1000:.2f} m")
        print(
            f"  Coverage ({self.confidence_level * 100:.0f}% CI) = {coverage:.4f} (target: {self.confidence_level:.2f})"
        )
        print(f"  Avg Interval Width = {avg_interval_width * 1000:.2f} m")
        print(f"  Mean Uncertainty = {mean_uncertainty * 1000:.2f} m")
        print(f"  Uncertainty-Error Correlation = {uncertainty_correlation:.4f}")

        return {
            "r2": float(r2),
            "mae_km": float(mae),
            "mae_m": float(mae * 1000),
            "coverage": float(coverage),
            "coverage_target": float(self.confidence_level),
            "avg_interval_width_km": float(avg_interval_width),
            "avg_interval_width_m": float(avg_interval_width * 1000),
            "mean_uncertainty_km": float(mean_uncertainty),
            "mean_uncertainty_m": float(mean_uncertainty * 1000),
            "uncertainty_error_correlation": float(uncertainty_correlation),
        }

    def run_validation(self, X, y):
        """Run stratified 5-fold cross-validation with UQ."""
        print("\n" + "=" * 80)
        print("Running Stratified 5-Fold CV with Uncertainty Quantification")
        print("=" * 80)

        # Create stratified bins
        stratified_bins = self.create_stratified_bins(y, n_bins=10)

        # Initialize k-fold
        skf = StratifiedKFold(
            n_splits=self.n_folds, shuffle=True, random_state=self.random_seed
        )

        # Run folds
        fold_results = []
        all_y_true = []
        all_y_pred = []
        all_y_lower = []
        all_y_upper = []
        all_uncertainty = []

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, stratified_bins)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            fold_result = self.evaluate_fold(X_train, y_train, X_val, y_val, fold_idx)

            fold_results.append(fold_result)
            all_y_true.extend(fold_result["y_true"])
            all_y_pred.extend(fold_result["y_pred"])
            all_y_lower.extend(fold_result["y_lower"])
            all_y_upper.extend(fold_result["y_upper"])
            all_uncertainty.extend(fold_result["uncertainty"])

        # Store results
        self.results["folds"] = fold_results
        self.results["aggregated_predictions"] = {
            "y_true": all_y_true,
            "y_pred": all_y_pred,
            "y_lower": all_y_lower,
            "y_upper": all_y_upper,
            "uncertainty": all_uncertainty,
        }

        # Aggregate metrics
        self._aggregate_metrics(fold_results)

        return self.results

    def _aggregate_metrics(self, fold_results):
        """Aggregate metrics across folds."""
        print("\n" + "=" * 80)
        print("Aggregated UQ Results Across All Folds")
        print("=" * 80)

        # Extract metrics
        coverages = [f["metrics"]["coverage"] for f in fold_results]
        interval_widths = [f["metrics"]["avg_interval_width_m"] for f in fold_results]
        mean_uncertainties = [f["metrics"]["mean_uncertainty_m"] for f in fold_results]
        correlations = [
            f["metrics"]["uncertainty_error_correlation"] for f in fold_results
        ]
        r2_vals = [f["metrics"]["r2"] for f in fold_results]
        mae_vals = [f["metrics"]["mae_m"] for f in fold_results]

        # Compute means and stds
        mean_coverage = np.mean(coverages)
        std_coverage = np.std(coverages)
        mean_interval = np.mean(interval_widths)
        std_interval = np.std(interval_widths)
        mean_uncertainty = np.mean(mean_uncertainties)
        std_uncertainty = np.std(mean_uncertainties)
        mean_correlation = np.mean(correlations)
        std_correlation = np.std(correlations)
        mean_r2 = np.mean(r2_vals)
        mean_mae = np.mean(mae_vals)

        print(
            f"Mean Coverage = {mean_coverage:.4f} ± {std_coverage:.4f} (target: {self.confidence_level:.2f})"
        )
        print(f"Mean Interval Width = {mean_interval:.2f} ± {std_interval:.2f} m")
        print(f"Mean Uncertainty = {mean_uncertainty:.2f} ± {std_uncertainty:.2f} m")
        print(
            f"Mean Uncertainty-Error Correlation = {mean_correlation:.4f} ± {std_correlation:.4f}"
        )
        print(f"Mean R² = {mean_r2:.4f}")
        print(f"Mean MAE = {mean_mae:.2f} m")

        # Calibration assessment
        coverage_gap = abs(mean_coverage - self.confidence_level)
        if coverage_gap < 0.05:
            calibration_status = "Well-calibrated"
        elif coverage_gap < 0.10:
            calibration_status = "Moderately calibrated"
        else:
            calibration_status = "Poorly calibrated"

        print(f"\nCalibration Status: {calibration_status}")
        print(f"  Coverage gap from target: {coverage_gap:.4f}")

        self.results["aggregated_metrics"] = {
            "mean_coverage": float(mean_coverage),
            "std_coverage": float(std_coverage),
            "coverage_target": float(self.confidence_level),
            "coverage_gap": float(coverage_gap),
            "calibration_status": calibration_status,
            "mean_interval_width_m": float(mean_interval),
            "std_interval_width_m": float(std_interval),
            "mean_uncertainty_m": float(mean_uncertainty),
            "std_uncertainty_m": float(std_uncertainty),
            "mean_uncertainty_error_correlation": float(mean_correlation),
            "std_uncertainty_error_correlation": float(std_correlation),
            "mean_r2": float(mean_r2),
            "mean_mae_m": float(mean_mae),
        }

    def generate_visualizations(self, output_dir):
        """Generate UQ visualizations."""
        print("\n" + "=" * 80)
        print("Generating Uncertainty Quantification Visualizations")
        print("=" * 80)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Set style
        sns.set_style("whitegrid")
        plt.rcParams["figure.dpi"] = 300

        # 1. Predictions with uncertainty bands
        self._plot_predictions_with_uncertainty(output_dir)

        # 2. Calibration plot
        self._plot_calibration(output_dir)

        # 3. Uncertainty vs error scatter
        self._plot_uncertainty_vs_error(output_dir)

        # 4. Coverage and interval width across folds
        self._plot_fold_uq_metrics(output_dir)

        print(f"✓ Visualizations saved to {output_dir}")

    def _plot_predictions_with_uncertainty(self, output_dir):
        """Plot predictions with uncertainty bands."""
        y_true = np.array(self.results["aggregated_predictions"]["y_true"])
        y_pred = np.array(self.results["aggregated_predictions"]["y_pred"])
        y_lower = np.array(self.results["aggregated_predictions"]["y_lower"])
        y_upper = np.array(self.results["aggregated_predictions"]["y_upper"])

        # Sort by true values for better visualization
        sorted_indices = np.argsort(y_true)
        y_true_sorted = y_true[sorted_indices]
        y_pred_sorted = y_pred[sorted_indices]
        y_lower_sorted = y_lower[sorted_indices]
        y_upper_sorted = y_upper[sorted_indices]

        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot predictions
        x = np.arange(len(y_true_sorted))
        ax.scatter(x, y_true_sorted, s=10, alpha=0.6, label="True CBH", color="black")
        ax.scatter(
            x, y_pred_sorted, s=10, alpha=0.6, label="Predicted CBH", color="blue"
        )

        # Plot uncertainty bands
        ax.fill_between(
            x,
            y_lower_sorted,
            y_upper_sorted,
            alpha=0.3,
            color="skyblue",
            label=f"{self.confidence_level * 100:.0f}% Prediction Interval",
        )

        ax.set_xlabel("Sample Index (sorted by true CBH)", fontsize=12)
        ax.set_ylabel("CBH (km)", fontsize=12)
        ax.set_title(
            f"Predictions with {self.confidence_level * 100:.0f}% Uncertainty Intervals",
            fontsize=14,
            fontweight="bold",
        )
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            output_dir / "predictions_with_uncertainty.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
        print("  ✓ predictions_with_uncertainty.png")

    def _plot_calibration(self, output_dir):
        """Plot calibration curve."""
        y_true = np.array(self.results["aggregated_predictions"]["y_true"])
        y_pred = np.array(self.results["aggregated_predictions"]["y_pred"])
        y_lower = np.array(self.results["aggregated_predictions"]["y_lower"])
        y_upper = np.array(self.results["aggregated_predictions"]["y_upper"])

        # Compute coverage at different confidence levels
        confidence_levels = np.linspace(0.1, 1.0, 10)
        empirical_coverages = []

        for conf in confidence_levels:
            alpha = 1 - conf
            # Approximate quantiles from the interval
            lower_adj = y_pred - (y_pred - y_lower) * (conf / self.confidence_level)
            upper_adj = y_pred + (y_upper - y_pred) * (conf / self.confidence_level)
            coverage = np.mean((y_true >= lower_adj) & (y_true <= upper_adj))
            empirical_coverages.append(coverage)

        fig, ax = plt.subplots(figsize=(8, 8))

        # Plot calibration curve
        ax.plot(
            confidence_levels,
            empirical_coverages,
            "o-",
            linewidth=2,
            markersize=8,
            label="Empirical Coverage",
        )
        ax.plot([0, 1], [0, 1], "r--", linewidth=2, label="Perfect Calibration")

        # Highlight target confidence level
        target_coverage = self.results["aggregated_metrics"]["mean_coverage"]
        ax.scatter(
            [self.confidence_level],
            [target_coverage],
            s=200,
            color="orange",
            edgecolors="black",
            linewidths=2,
            zorder=5,
            label=f"Target: {self.confidence_level:.0%}",
        )

        ax.set_xlabel("Expected Coverage (Confidence Level)", fontsize=12)
        ax.set_ylabel("Empirical Coverage", fontsize=12)
        ax.set_title("Calibration Curve", fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

        plt.tight_layout()
        plt.savefig(output_dir / "calibration_curve.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("  ✓ calibration_curve.png")

    def _plot_uncertainty_vs_error(self, output_dir):
        """Plot uncertainty vs absolute error."""
        y_true = np.array(self.results["aggregated_predictions"]["y_true"])
        y_pred = np.array(self.results["aggregated_predictions"]["y_pred"])
        uncertainty = np.array(self.results["aggregated_predictions"]["uncertainty"])

        abs_error = np.abs(y_true - y_pred) * 1000  # meters
        uncertainty_m = uncertainty * 1000  # meters

        fig, ax = plt.subplots(figsize=(10, 8))

        # Scatter plot
        ax.scatter(uncertainty_m, abs_error, alpha=0.5, s=20)

        # Correlation
        correlation = self.results["aggregated_metrics"][
            "mean_uncertainty_error_correlation"
        ]

        ax.set_xlabel("Predicted Uncertainty (m)", fontsize=12)
        ax.set_ylabel("Absolute Error (m)", fontsize=12)
        ax.set_title("Uncertainty vs Absolute Error", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)

        # Add correlation text
        textstr = f"Correlation: {correlation:.3f}"
        ax.text(
            0.05,
            0.95,
            textstr,
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        plt.tight_layout()
        plt.savefig(
            output_dir / "uncertainty_vs_error.png", dpi=300, bbox_inches="tight"
        )
        plt.close()
        print("  ✓ uncertainty_vs_error.png")

    def _plot_fold_uq_metrics(self, output_dir):
        """Plot UQ metrics across folds."""
        folds = [f["fold"] for f in self.results["folds"]]
        coverages = [f["metrics"]["coverage"] for f in self.results["folds"]]
        interval_widths = [
            f["metrics"]["avg_interval_width_m"] for f in self.results["folds"]
        ]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Coverage across folds
        axes[0].bar(folds, coverages, color="steelblue", edgecolor="black")
        axes[0].axhline(
            self.confidence_level,
            color="r",
            linestyle="--",
            linewidth=2,
            label="Target",
        )
        axes[0].axhline(
            np.mean(coverages), color="orange", linestyle="-", linewidth=2, label="Mean"
        )
        axes[0].set_xlabel("Fold", fontsize=12)
        axes[0].set_ylabel("Coverage", fontsize=12)
        axes[0].set_title(
            f"Coverage Across Folds (Target: {self.confidence_level:.0%})",
            fontsize=14,
            fontweight="bold",
        )
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis="y")
        axes[0].set_ylim([0, 1])

        # Interval width across folds
        axes[1].bar(folds, interval_widths, color="coral", edgecolor="black")
        axes[1].axhline(
            np.mean(interval_widths),
            color="orange",
            linestyle="-",
            linewidth=2,
            label="Mean",
        )
        axes[1].set_xlabel("Fold", fontsize=12)
        axes[1].set_ylabel("Avg Interval Width (m)", fontsize=12)
        axes[1].set_title(
            "Prediction Interval Width Across Folds", fontsize=14, fontweight="bold"
        )
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        plt.savefig(output_dir / "fold_uq_metrics.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("  ✓ fold_uq_metrics.png")

    def save_report(self, output_path):
        """Save UQ report as JSON."""
        output_path = Path(output_path)
        with open(output_path, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"\n✓ UQ report saved to {output_path}")


def main():
    """Main execution."""
    # Initialize quantifier
    quantifier = UncertaintyQuantifier(n_folds=5, random_seed=42, confidence_level=0.90)

    # Load data
    X, y, flight_ids = quantifier.load_data(INTEGRATED_FEATURES)

    # Run validation with UQ
    results = quantifier.run_validation(X, y)

    # Generate visualizations
    quantifier.generate_visualizations(FIGURES_DIR)

    # Save report
    quantifier.save_report(REPORTS_DIR / "uncertainty_quantification_report.json")

    print("\n" + "=" * 80)
    print("Uncertainty Quantification Complete!")
    print("=" * 80)
    agg = results["aggregated_metrics"]
    print(
        f"Mean Coverage = {agg['mean_coverage']:.4f} (target: {agg['coverage_target']:.2f})"
    )
    print(f"Calibration Status: {agg['calibration_status']}")
    print(f"Mean Interval Width = {agg['mean_interval_width_m']:.2f} m")
    print(
        f"Uncertainty-Error Correlation = {agg['mean_uncertainty_error_correlation']:.4f}"
    )
    print("\nOutputs:")
    print(f"  - Report: {REPORTS_DIR / 'uncertainty_quantification_report.json'}")
    print(f"  - Figures: {FIGURES_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
