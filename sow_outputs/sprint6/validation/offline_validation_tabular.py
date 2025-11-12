#!/usr/bin/env python3
"""
Sprint 6 - Phase 1, Task 1.1: Offline Validation (Tabular Features Adaptation)

This script performs stratified 5-fold cross-validation on the integrated tabular features
(geometric + atmospheric) using a Gradient Boosting Decision Tree (GBDT) baseline model.

This is an adaptation for when raw image data is not available, using only the
tabular features from Integrated_Features.hdf5.

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
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

print("=" * 80)
print("Sprint 6 - Phase 1, Task 1.1: Offline Validation (Tabular Features)")
print("=" * 80)
print(f"Project Root: {PROJECT_ROOT}")

# Paths
INTEGRATED_FEATURES = (
    PROJECT_ROOT / "sow_outputs/integrated_features/Integrated_Features.hdf5"
)
OUTPUT_DIR = PROJECT_ROOT / "sow_outputs/sprint6"
VALIDATION_DIR = OUTPUT_DIR / "validation"
FIGURES_DIR = OUTPUT_DIR / "figures/validation"
REPORTS_DIR = OUTPUT_DIR / "reports"

# Create directories
VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

print(f"Integrated Features: {INTEGRATED_FEATURES}")
print(f"Output Directory: {OUTPUT_DIR}")
print(f"✓ Validation output directory: {VALIDATION_DIR}")


class TabularValidationAnalyzer:
    """Performs stratified 5-fold CV validation on tabular features."""

    def __init__(self, n_folds=5, random_seed=42):
        self.n_folds = n_folds
        self.random_seed = random_seed
        self.results = {
            "folds": [],
            "mean_metrics": {},
            "std_metrics": {},
            "metadata": {
                "model_type": "GradientBoostingRegressor",
                "n_folds": n_folds,
                "random_seed": random_seed,
                "validation_strategy": "stratified_5fold",
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
            sample_ids = f["metadata/sample_id"][:]

            # Load atmospheric features
            era5_features = f["atmospheric_features/era5_features"][:]
            era5_feature_names = [
                name.decode("utf-8") if isinstance(name, bytes) else name
                for name in f["atmospheric_features/era5_feature_names"][:]
            ]

            # Load geometric features
            geometric_features = {}
            for key in f["geometric_features"].keys():
                if (
                    key != "derived_geometric_H"
                ):  # Skip this as it might be derived from CBH
                    data = f[f"geometric_features/{key}"][:]
                    if data.ndim == 1:
                        geometric_features[key] = data

            # Get flight mapping
            flight_mapping = json.loads(f.attrs["flight_mapping"])

        print(f"✓ Loaded {len(cbh_km)} samples")
        print(f"✓ CBH range: [{cbh_km.min():.3f}, {cbh_km.max():.3f}] km")
        print(
            f"✓ Atmospheric features: {len(era5_feature_names)} ({', '.join(era5_feature_names)})"
        )
        print(
            f"✓ Geometric features: {len(geometric_features)} ({', '.join(geometric_features.keys())})"
        )
        print(f"✓ Flight mapping: {flight_mapping}")

        # Combine all features
        feature_list = [era5_features]
        feature_names = era5_feature_names.copy()

        for name, values in geometric_features.items():
            feature_list.append(values.reshape(-1, 1))
            feature_names.append(name)

        X = np.hstack(feature_list)
        y = cbh_km

        print(f"\n✓ Total feature matrix shape: {X.shape}")
        print(f"✓ Feature names ({len(feature_names)}): {feature_names}")

        # Create metadata dataframe
        metadata = pd.DataFrame(
            {
                "flight_id": flight_ids,
                "sample_id": sample_ids,
                "cbh_km": cbh_km,
            }
        )

        return X, y, feature_names, metadata, flight_mapping

    def create_stratified_bins(self, y, n_bins=10):
        """Create stratified bins for CBH values."""
        bins = np.percentile(y, np.linspace(0, 100, n_bins + 1))
        bins = np.unique(bins)  # Remove duplicates
        bin_indices = np.digitize(y, bins) - 1
        bin_indices = np.clip(bin_indices, 0, len(bins) - 2)
        return bin_indices

    def train_and_evaluate_fold(self, X_train, y_train, X_val, y_val, fold_idx):
        """Train and evaluate on a single fold."""
        print(f"\n{'=' * 80}")
        print(f"Fold {fold_idx + 1}/{self.n_folds}")
        print(f"{'=' * 80}")
        print(f"Train samples: {len(y_train)}, Val samples: {len(y_val)}")
        print(f"Train CBH range: [{y_train.min():.3f}, {y_train.max():.3f}] km")
        print(f"Val CBH range: [{y_val.min():.3f}, {y_val.max():.3f}] km")

        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # Train GBDT model
        print("\nTraining GBDT model...")
        model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            min_samples_split=5,
            min_samples_leaf=2,
            subsample=0.8,
            random_state=self.random_seed + fold_idx,
            verbose=0,
        )

        model.fit(X_train_scaled, y_train)

        # Predictions
        train_pred = model.predict(X_train_scaled)
        val_pred = model.predict(X_val_scaled)

        # Metrics
        train_metrics = self._compute_metrics(y_train, train_pred, "Train")
        val_metrics = self._compute_metrics(y_val, val_pred, "Validation")

        # Feature importance
        feature_importance = model.feature_importances_

        return {
            "fold": fold_idx + 1,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "train_size": len(y_train),
            "val_size": len(y_val),
            "y_true": y_val.tolist(),
            "y_pred": val_pred.tolist(),
            "feature_importance": feature_importance.tolist(),
        }

    def _compute_metrics(self, y_true, y_pred, prefix=""):
        """Compute regression metrics."""
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)

        # Convert to meters for reporting
        mae_m = mae * 1000
        rmse_m = rmse * 1000

        print(f"{prefix} Metrics:")
        print(f"  R² = {r2:.4f}")
        print(f"  MAE = {mae_m:.2f} m ({mae:.4f} km)")
        print(f"  RMSE = {rmse_m:.2f} m ({rmse:.4f} km)")

        return {
            "r2": float(r2),
            "mae_km": float(mae),
            "rmse_km": float(rmse),
            "mae_m": float(mae_m),
            "rmse_m": float(rmse_m),
        }

    def run_validation(self, X, y, feature_names):
        """Run stratified 5-fold cross-validation."""
        print("\n" + "=" * 80)
        print("Running Stratified 5-Fold Cross-Validation")
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

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, stratified_bins)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            fold_result = self.train_and_evaluate_fold(
                X_train, y_train, X_val, y_val, fold_idx
            )

            fold_results.append(fold_result)
            all_y_true.extend(fold_result["y_true"])
            all_y_pred.extend(fold_result["y_pred"])

        # Aggregate results
        self.results["folds"] = fold_results
        self._aggregate_metrics(fold_results, feature_names)

        # Store aggregated predictions
        self.results["aggregated_predictions"] = {
            "y_true": all_y_true,
            "y_pred": all_y_pred,
        }

        return self.results

    def _aggregate_metrics(self, fold_results, feature_names):
        """Aggregate metrics across folds."""
        print("\n" + "=" * 80)
        print("Aggregated Results Across All Folds")
        print("=" * 80)

        # Extract validation metrics
        val_r2 = [f["val_metrics"]["r2"] for f in fold_results]
        val_mae_m = [f["val_metrics"]["mae_m"] for f in fold_results]
        val_rmse_m = [f["val_metrics"]["rmse_m"] for f in fold_results]

        mean_r2 = np.mean(val_r2)
        std_r2 = np.std(val_r2)
        mean_mae_m = np.mean(val_mae_m)
        std_mae_m = np.std(val_mae_m)
        mean_rmse_m = np.mean(val_rmse_m)
        std_rmse_m = np.std(val_rmse_m)

        print(f"Mean R² = {mean_r2:.4f} ± {std_r2:.4f}")
        print(f"Mean MAE = {mean_mae_m:.2f} ± {std_mae_m:.2f} m")
        print(f"Mean RMSE = {mean_rmse_m:.2f} ± {std_rmse_m:.2f} m")

        self.results["mean_metrics"] = {
            "r2": float(mean_r2),
            "mae_m": float(mean_mae_m),
            "rmse_m": float(mean_rmse_m),
        }

        self.results["std_metrics"] = {
            "r2": float(std_r2),
            "mae_m": float(std_mae_m),
            "rmse_m": float(std_rmse_m),
        }

        # Aggregate feature importance
        feature_importances = np.array([f["feature_importance"] for f in fold_results])
        mean_importance = np.mean(feature_importances, axis=0)
        std_importance = np.std(feature_importances, axis=0)

        # Sort by importance
        sorted_indices = np.argsort(mean_importance)[::-1]

        print("\nTop 10 Most Important Features:")
        for i, idx in enumerate(sorted_indices[:10]):
            print(
                f"  {i + 1}. {feature_names[idx]}: {mean_importance[idx]:.4f} ± {std_importance[idx]:.4f}"
            )

        self.results["feature_importance"] = {
            "feature_names": feature_names,
            "mean_importance": mean_importance.tolist(),
            "std_importance": std_importance.tolist(),
        }

    def generate_visualizations(self, output_dir):
        """Generate validation visualizations."""
        print("\n" + "=" * 80)
        print("Generating Visualizations")
        print("=" * 80)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Set style
        sns.set_style("whitegrid")
        plt.rcParams["figure.dpi"] = 300

        # 1. Scatter plot: Predicted vs True
        self._plot_predictions_scatter(output_dir)

        # 2. Residuals plot
        self._plot_residuals(output_dir)

        # 3. Per-fold metrics
        self._plot_fold_metrics(output_dir)

        # 4. Feature importance
        self._plot_feature_importance(output_dir)

        print(f"✓ Visualizations saved to {output_dir}")

    def _plot_predictions_scatter(self, output_dir):
        """Plot predicted vs true values."""
        y_true = np.array(self.results["aggregated_predictions"]["y_true"])
        y_pred = np.array(self.results["aggregated_predictions"]["y_pred"])

        fig, ax = plt.subplots(figsize=(8, 8))

        # Scatter plot
        ax.scatter(y_true, y_pred, alpha=0.5, s=20)

        # Perfect prediction line
        lim = [y_true.min(), y_true.max()]
        ax.plot(lim, lim, "r--", lw=2, label="Perfect Prediction")

        # Labels and formatting
        ax.set_xlabel("True CBH (km)", fontsize=12)
        ax.set_ylabel("Predicted CBH (km)", fontsize=12)
        ax.set_title(
            "Predicted vs True CBH (5-Fold CV)", fontsize=14, fontweight="bold"
        )
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add metrics text
        r2 = self.results["mean_metrics"]["r2"]
        mae = self.results["mean_metrics"]["mae_m"]
        textstr = f"R² = {r2:.3f}\nMAE = {mae:.1f} m"
        ax.text(
            0.05,
            0.95,
            textstr,
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        plt.tight_layout()
        plt.savefig(
            output_dir / "predictions_scatter.png", dpi=300, bbox_inches="tight"
        )
        plt.close()
        print("  ✓ predictions_scatter.png")

    def _plot_residuals(self, output_dir):
        """Plot residuals distribution."""
        y_true = np.array(self.results["aggregated_predictions"]["y_true"])
        y_pred = np.array(self.results["aggregated_predictions"]["y_pred"])
        residuals = (y_pred - y_true) * 1000  # Convert to meters

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram
        axes[0].hist(residuals, bins=50, edgecolor="black", alpha=0.7)
        axes[0].axvline(0, color="r", linestyle="--", lw=2, label="Zero Error")
        axes[0].set_xlabel("Residual (m)", fontsize=12)
        axes[0].set_ylabel("Frequency", fontsize=12)
        axes[0].set_title("Distribution of Residuals", fontsize=14, fontweight="bold")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Residuals vs predictions
        axes[1].scatter(y_pred, residuals, alpha=0.5, s=20)
        axes[1].axhline(0, color="r", linestyle="--", lw=2, label="Zero Error")
        axes[1].set_xlabel("Predicted CBH (km)", fontsize=12)
        axes[1].set_ylabel("Residual (m)", fontsize=12)
        axes[1].set_title("Residuals vs Predicted", fontsize=14, fontweight="bold")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "residuals.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("  ✓ residuals.png")

    def _plot_fold_metrics(self, output_dir):
        """Plot metrics across folds."""
        folds = [f["fold"] for f in self.results["folds"]]
        r2_vals = [f["val_metrics"]["r2"] for f in self.results["folds"]]
        mae_vals = [f["val_metrics"]["mae_m"] for f in self.results["folds"]]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # R² across folds
        axes[0].bar(folds, r2_vals, color="steelblue", edgecolor="black")
        axes[0].axhline(np.mean(r2_vals), color="r", linestyle="--", lw=2, label="Mean")
        axes[0].set_xlabel("Fold", fontsize=12)
        axes[0].set_ylabel("R²", fontsize=12)
        axes[0].set_title("R² Across Folds", fontsize=14, fontweight="bold")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis="y")

        # MAE across folds
        axes[1].bar(folds, mae_vals, color="coral", edgecolor="black")
        axes[1].axhline(
            np.mean(mae_vals), color="r", linestyle="--", lw=2, label="Mean"
        )
        axes[1].set_xlabel("Fold", fontsize=12)
        axes[1].set_ylabel("MAE (m)", fontsize=12)
        axes[1].set_title("MAE Across Folds", fontsize=14, fontweight="bold")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        plt.savefig(output_dir / "fold_metrics.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("  ✓ fold_metrics.png")

    def _plot_feature_importance(self, output_dir):
        """Plot feature importance."""
        feature_names = self.results["feature_importance"]["feature_names"]
        mean_importance = np.array(
            self.results["feature_importance"]["mean_importance"]
        )
        std_importance = np.array(self.results["feature_importance"]["std_importance"])

        # Sort by importance
        sorted_indices = np.argsort(mean_importance)[::-1][:15]  # Top 15

        sorted_names = [feature_names[i] for i in sorted_indices]
        sorted_mean = mean_importance[sorted_indices]
        sorted_std = std_importance[sorted_indices]

        fig, ax = plt.subplots(figsize=(10, 8))

        y_pos = np.arange(len(sorted_names))
        ax.barh(
            y_pos,
            sorted_mean,
            xerr=sorted_std,
            align="center",
            color="steelblue",
            edgecolor="black",
            capsize=5,
        )
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_names, fontsize=10)
        ax.invert_yaxis()
        ax.set_xlabel("Feature Importance", fontsize=12)
        ax.set_title(
            "Top 15 Feature Importances (Mean ± Std)", fontsize=14, fontweight="bold"
        )
        ax.grid(True, alpha=0.3, axis="x")

        plt.tight_layout()
        plt.savefig(output_dir / "feature_importance.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("  ✓ feature_importance.png")

    def save_report(self, output_path):
        """Save validation report as JSON."""
        output_path = Path(output_path)
        with open(output_path, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"\n✓ Validation report saved to {output_path}")


def main():
    """Main execution."""
    # Initialize analyzer
    analyzer = TabularValidationAnalyzer(n_folds=5, random_seed=42)

    # Load data
    X, y, feature_names, metadata, flight_mapping = analyzer.load_data(
        INTEGRATED_FEATURES
    )

    # Run validation
    results = analyzer.run_validation(X, y, feature_names)

    # Generate visualizations
    analyzer.generate_visualizations(FIGURES_DIR)

    # Save report
    analyzer.save_report(REPORTS_DIR / "validation_report_tabular.json")

    print("\n" + "=" * 80)
    print("Validation Complete!")
    print("=" * 80)
    print(
        f"Mean R² = {results['mean_metrics']['r2']:.4f} ± {results['std_metrics']['r2']:.4f}"
    )
    print(
        f"Mean MAE = {results['mean_metrics']['mae_m']:.1f} ± {results['std_metrics']['mae_m']:.1f} m"
    )
    print(
        f"Mean RMSE = {results['mean_metrics']['rmse_m']:.1f} ± {results['std_metrics']['rmse_m']:.1f} m"
    )
    print("\nOutputs:")
    print(f"  - Report: {REPORTS_DIR / 'validation_report_tabular.json'}")
    print(f"  - Figures: {FIGURES_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
