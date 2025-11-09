#!/usr/bin/env python3
"""
WP-3 Physical Baseline Validation (ERA5-ONLY VERSION)

This script re-runs WP-3 baseline validation excluding geometric features
to test whether the catastrophic R² = -14.15 was caused by:
1. Imputation bug (median=6.166 km vs true mean=0.83 km)
2. Poor geometric feature quality (r ≈ 0.04)

Expected outcome:
- If R² improves significantly (e.g., > -1): geometric features were the problem
- If R² remains negative: ERA5 features lack predictive power at cloud scale

Author: Sprint 4 Quick Fix
Date: 2025
"""

import h5py
import numpy as np
import xgboost as xgb
import yaml
from pathlib import Path
from typing import List, Tuple
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import json

# Import dataset utilities
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.hdf5_dataset import HDF5CloudDataset


@dataclass
class FoldMetrics:
    """Metrics for a single LOO CV fold."""

    fold_id: int
    test_flight: str
    n_train: int
    n_test: int
    r2: float
    mae_km: float
    rmse_km: float
    predictions: np.ndarray
    targets: np.ndarray


class ERA5OnlyValidator:
    """
    ERA5-only baseline validator.

    This version excludes all geometric features from WP-1.
    """

    def __init__(
        self,
        wp2_features_path: str,
        config_path: str,
        output_dir: str = "sow_outputs/wp3_era5_only",
        verbose: bool = True,
    ):
        """
        Initialize the validator.

        Args:
            wp2_features_path: Path to WP2_Features.hdf5
            config_path: Path to config YAML
            output_dir: Directory for outputs
            verbose: Enable verbose logging
        """
        self.wp2_path = Path(wp2_features_path)
        self.config_path = Path(config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose

        # Flight mapping (as per SOW Section 2.2)
        self.flight_mapping = {
            0: "30Oct24",  # F_0: n=501
            1: "10Feb25",  # F_1: n=191
            2: "23Oct24",  # F_2: n=105
            3: "12Feb25",  # F_3: n=92
            4: "18Feb25",  # F_4: n=44
        }

        # Load config
        with open(self.config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Flight name to ID mapping
        self.flight_name_to_id = {}
        for i, flight in enumerate(self.config["flights"]):
            name = flight["name"]
            # Map to standard names
            if name in ["30Oct24", "10Feb25", "23Oct24", "12Feb25", "18Feb25"]:
                for fid, fname in self.flight_mapping.items():
                    if fname == name:
                        self.flight_name_to_id[name] = fid
                        break

    def load_features(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load WP2 features ONLY (no geometric features).

        Returns:
            X: Feature matrix (n_samples, n_features) - ERA5 only
            y: Target CBH in km (n_samples,)
            flight_ids: Flight ID for each sample (n_samples,)
            feature_names: List of feature names
        """
        if self.verbose:
            print("\n" + "=" * 80)
            print("Loading Features (ERA5-ONLY)")
            print("=" * 80)

        # Load WP2 features
        if not self.wp2_path.exists():
            raise FileNotFoundError(f"WP2 features not found: {self.wp2_path}")

        with h5py.File(self.wp2_path, "r") as f:
            if self.verbose:
                print(f"\nWP2 Features: {self.wp2_path}")
                print(f"  Keys: {list(f.keys())}")

            # WP2 file has 'features' array and 'feature_names' array
            if "features" in f and "feature_names" in f:
                wp2_features = f["features"][:]
                feature_names_raw = f["feature_names"][:]
                atmospheric_names = [
                    f"atm_{name.decode() if isinstance(name, bytes) else name}"
                    for name in feature_names_raw
                ]
            else:
                # Fallback: try individual keys
                atmospheric_features = []
                atmospheric_names = []

                for key in [
                    "blh",
                    "lcl",
                    "inversion_height",
                    "moisture_gradient",
                    "stability_index",
                    "t2m",
                    "d2m",
                    "sp",
                    "tcwv",
                ]:
                    if key in f:
                        atmospheric_features.append(f[key][:])
                        atmospheric_names.append(f"atm_{key}")

                wp2_features = (
                    np.column_stack(atmospheric_features)
                    if atmospheric_features
                    else np.zeros((len(f[list(f.keys())[0]]), 0))
                )

            if self.verbose:
                print(f"  Loaded {len(atmospheric_names)} atmospheric features")
                print(f"  Features: {atmospheric_names}")

        # Load dataset to get flight IDs and true CBH
        if self.verbose:
            print(f"\nLoading dataset from config...")

        dataset = HDF5CloudDataset(
            flight_configs=self.config["flights"],
            indices=None,
            augment=False,
            swath_slice=self.config.get("swath_slice", [40, 480]),
        )

        # Get true CBH values
        y = dataset.get_unscaled_y()  # km

        # Get flight IDs for each sample
        flight_ids = np.zeros(len(dataset), dtype=int)
        for i in range(len(dataset)):
            _, _, _, _, global_idx, local_idx = dataset[i]
            flight_idx, _ = dataset.global_to_local[int(global_idx)]

            # Map flight_idx to standard flight ID
            flight_name = dataset.flight_data[flight_idx]["name"]
            if flight_name in self.flight_name_to_id:
                flight_ids[i] = self.flight_name_to_id[flight_name]
            else:
                # Try to map by position
                flight_ids[i] = flight_idx

        # Use only ERA5 features
        X = wp2_features
        feature_names = atmospheric_names

        if self.verbose:
            print(f"\n" + "-" * 80)
            print(f"Feature Matrix (ERA5-ONLY):")
            print(f"  Samples: {X.shape[0]}")
            print(f"  Features: {X.shape[1]}")
            print(f"  Feature names: {feature_names}")
            print(f"  Flight distribution:")
            for fid in sorted(np.unique(flight_ids)):
                count = np.sum(flight_ids == fid)
                fname = self.flight_mapping.get(fid, f"Unknown_{fid}")
                print(f"    F{fid} ({fname}): {count} samples")
            print(f"  Target (CBH) range: [{np.min(y):.3f}, {np.max(y):.3f}] km")
            print(f"  Target mean: {np.mean(y):.3f} km")
            print(f"  Target std: {np.std(y):.3f} km")
            print("=" * 80)

        return X, y, flight_ids, feature_names

    def train_gbdt(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray = None,
        y_test: np.ndarray = None,
    ) -> xgb.XGBRegressor:
        """
        Train GBDT model.

        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Validation features (optional, for early stopping)
            y_test: Validation targets (optional, for early stopping)

        Returns:
            Trained XGBoost model
        """
        # XGBoost hyperparameters (consistent with prior experiments)
        params = {
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
            "gamma": 0.1,
            "random_state": 42,
            "n_jobs": -1,
            "tree_method": "hist",
        }

        model = xgb.XGBRegressor(**params)

        # Train with early stopping if validation set provided
        if X_test is not None and y_test is not None:
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_test, y_test)],
                early_stopping_rounds=50,
                verbose=False,
            )
        else:
            model.fit(X_train, y_train)

        return model

    def evaluate_fold(
        self, model: xgb.XGBRegressor, X_test: np.ndarray, y_test: np.ndarray
    ) -> Tuple[float, float, float]:
        """Evaluate model on test set."""
        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        return r2, mae, rmse

    def run_loo_cv(self) -> List[FoldMetrics]:
        """
        Execute Leave-One-Flight-Out Cross-Validation.

        This is the mandated validation protocol (SOW Section 2.1).

        Returns:
            List of FoldMetrics for each fold
        """
        if self.verbose:
            print("\n" + "=" * 80)
            print("Leave-One-Flight-Out Cross-Validation (ERA5-ONLY)")
            print("=" * 80)

        # Load all features
        X, y, flight_ids, feature_names = self.load_features()

        # Check for NaNs in ERA5 features
        if np.any(np.isnan(X)):
            print("\nWARNING: NaN values detected in ERA5 features!")
            for j in range(X.shape[1]):
                col = X[:, j]
                if np.any(np.isnan(col)):
                    n_nan = np.sum(np.isnan(col))
                    print(f"  Feature {j} ({feature_names[j]}): {n_nan} NaNs")

            # Impute with median if needed
            X_clean = X.copy()
            for j in range(X.shape[1]):
                col = X[:, j]
                if np.any(np.isnan(col)):
                    median_val = np.nanmedian(col)
                    X_clean[np.isnan(col), j] = median_val
                    if self.verbose:
                        n_nan = np.sum(np.isnan(col))
                        print(
                            f"  Imputed {n_nan} NaN values in feature {j} ({feature_names[j]}) with median={median_val:.3f}"
                        )
        else:
            if self.verbose:
                print("\n✓ No NaN values detected in ERA5 features")
            X_clean = X

        # Results storage
        fold_results = []

        # Iterate through 5 folds (one per flight)
        for test_flight_id in range(5):
            test_flight_name = self.flight_mapping[test_flight_id]

            if self.verbose:
                print(f"\n" + "-" * 80)
                print(
                    f"Fold {test_flight_id}: Test on {test_flight_name} (F{test_flight_id})"
                )
                print("-" * 80)

            # Split data
            test_mask = flight_ids == test_flight_id
            train_mask = ~test_mask

            X_train = X_clean[train_mask]
            y_train = y[train_mask]
            X_test = X_clean[test_mask]
            y_test = y[test_mask]

            n_train = len(y_train)
            n_test = len(y_test)

            if self.verbose:
                print(f"  Training samples: {n_train}")
                print(f"  Test samples: {n_test}")
                print(
                    f"  Test CBH range: [{np.min(y_test):.3f}, {np.max(y_test):.3f}] km"
                )
                print(f"  Test CBH mean: {np.mean(y_test):.3f} km")

            # Check if we have enough data
            if n_test == 0:
                if self.verbose:
                    print(
                        f"  WARNING: No test samples for flight {test_flight_name}. Skipping."
                    )
                continue

            if n_train < 10:
                if self.verbose:
                    print(
                        f"  WARNING: Insufficient training samples ({n_train}). Skipping."
                    )
                continue

            # Scale features (fit on train, apply to test)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train model
            if self.verbose:
                print(f"  Training GBDT...")

            model = self.train_gbdt(X_train_scaled, y_train)

            # Evaluate
            r2, mae, rmse = self.evaluate_fold(model, X_test_scaled, y_test)

            # Get predictions for diagnostics
            y_pred = model.predict(X_test_scaled)

            if self.verbose:
                print(f"  Results:")
                print(f"    R² = {r2:.4f}")
                print(f"    MAE = {mae:.4f} km")
                print(f"    RMSE = {rmse:.4f} km")
                print(f"  Prediction stats:")
                print(f"    Mean prediction: {np.mean(y_pred):.3f} km")
                print(f"    Std prediction: {np.std(y_pred):.3f} km")
                print(
                    f"    Baseline R² (predict mean): {1 - np.var(y_test - np.mean(y_train)) / np.var(y_test):.4f}"
                )

            # Store results
            fold_metrics = FoldMetrics(
                fold_id=test_flight_id,
                test_flight=test_flight_name,
                n_train=n_train,
                n_test=n_test,
                r2=r2,
                mae_km=mae,
                rmse_km=rmse,
                predictions=y_pred,
                targets=y_test,
            )
            fold_results.append(fold_metrics)

        return fold_results

    def aggregate_results(self, fold_results: List[FoldMetrics]) -> dict:
        """Aggregate results across all folds."""
        r2_scores = [f.r2 for f in fold_results]
        mae_scores = [f.mae_km for f in fold_results]
        rmse_scores = [f.rmse_km for f in fold_results]

        aggregate = {
            "mean_r2": np.mean(r2_scores),
            "std_r2": np.std(r2_scores),
            "mean_mae_km": np.mean(mae_scores),
            "std_mae_km": np.std(mae_scores),
            "mean_rmse_km": np.mean(rmse_scores),
            "std_rmse_km": np.std(rmse_scores),
            "n_folds": len(fold_results),
        }

        return aggregate

    def generate_report(self, fold_results: List[FoldMetrics]) -> dict:
        """Generate final report."""
        # Per-fold results
        fold_reports = []
        for fold in fold_results:
            fold_reports.append(
                {
                    "fold_id": fold.fold_id,
                    "test_flight": fold.test_flight,
                    "n_train": fold.n_train,
                    "n_test": fold.n_test,
                    "r2": float(fold.r2),
                    "mae_km": float(fold.mae_km),
                    "rmse_km": float(fold.rmse_km),
                }
            )

        # Aggregate
        aggregate = self.aggregate_results(fold_results)

        report = {
            "experiment": "WP-3 ERA5-Only Baseline",
            "description": "Physical baseline using only ERA5 atmospheric features (no geometric features)",
            "validation_protocol": "Leave-One-Flight-Out Cross-Validation",
            "features": "ERA5 atmospheric variables only (9 features)",
            "model": "XGBoost GBDT",
            "fold_results": fold_reports,
            "aggregate_metrics": aggregate,
        }

        return report

    def save_report(self, report: dict):
        """Save report to JSON."""
        output_file = self.output_dir / "WP3_ERA5_Only_Report.json"
        with open(output_file, "w") as f:
            json.dump(report, f, indent=2)

        if self.verbose:
            print(f"\nReport saved to: {output_file}")

    def print_summary(self, report: dict):
        """Print summary to console."""
        print("\n" + "=" * 80)
        print("FINAL RESULTS (ERA5-ONLY)")
        print("=" * 80)

        print(f"\nExperiment: {report['experiment']}")
        print(f"Description: {report['description']}")
        print(f"Features: {report['features']}")

        print(f"\nPer-Fold Results:")
        print(
            f"{'Fold':<6} {'Flight':<12} {'n_train':<8} {'n_test':<8} {'R²':<10} {'MAE (km)':<12} {'RMSE (km)':<12}"
        )
        print("-" * 80)
        for fold in report["fold_results"]:
            print(
                f"{fold['fold_id']:<6} {fold['test_flight']:<12} {fold['n_train']:<8} {fold['n_test']:<8} "
                f"{fold['r2']:<10.4f} {fold['mae_km']:<12.4f} {fold['rmse_km']:<12.4f}"
            )

        agg = report["aggregate_metrics"]
        print(f"\nAggregate Metrics ({agg['n_folds']} folds):")
        print(f"  Mean R²:        {agg['mean_r2']:.4f} ± {agg['std_r2']:.4f}")
        print(
            f"  Mean MAE:       {agg['mean_mae_km']:.4f} ± {agg['std_mae_km']:.4f} km"
        )
        print(
            f"  Mean RMSE:      {agg['mean_rmse_km']:.4f} ± {agg['std_rmse_km']:.4f} km"
        )

        print("\n" + "=" * 80)
        print("INTERPRETATION")
        print("=" * 80)

        if agg["mean_r2"] > -1.0:
            print(
                "✓ R² significantly improved compared to WP-3 with geometric features (R² = -14.15)"
            )
            print("  → Geometric features were the primary cause of poor performance")
            print("  → ERA5 features provide weak but not catastrophic signal")
        elif agg["mean_r2"] > 0:
            print("✓ R² is positive! ERA5 features have predictive power")
            print("  → Geometric features were poisoning the model")
        else:
            print("✗ R² still negative")
            print(
                "  → ERA5 features at 25 km resolution lack predictive power for cloud-base height"
            )
            print(
                "  → Proceed with negative-results paper or test finer reanalysis (HRRR 3 km)"
            )

        print("=" * 80)

    def run(self):
        """Execute full validation pipeline."""
        print("\n" + "=" * 80)
        print("WP-3 ERA5-ONLY BASELINE VALIDATION")
        print("=" * 80)
        print("\nObjective: Test if removing geometric features improves R²")
        print("Previous WP-3 result (with geometric): R² = -14.15 ± 24.30")
        print("=" * 80)

        # Run LOO CV
        fold_results = self.run_loo_cv()

        # Generate report
        report = self.generate_report(fold_results)

        # Save report
        self.save_report(report)

        # Print summary
        self.print_summary(report)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="WP-3 ERA5-Only Baseline Validation (Quick Fix)"
    )
    parser.add_argument(
        "--wp2-features",
        type=str,
        default="sow_outputs/wp2_atmospheric/WP2_Features.hdf5",
        help="Path to WP2 features file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/bestComboConfig.yaml",
        help="Path to config YAML",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="sow_outputs/wp3_era5_only",
        help="Output directory",
    )
    parser.add_argument(
        "--verbose", action="store_true", default=True, help="Enable verbose output"
    )

    args = parser.parse_args()

    # Run validation
    validator = ERA5OnlyValidator(
        wp2_features_path=args.wp2_features,
        config_path=args.config,
        output_dir=args.output_dir,
        verbose=args.verbose,
    )

    validator.run()


if __name__ == "__main__":
    main()
