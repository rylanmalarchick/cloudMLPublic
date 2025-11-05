#!/usr/bin/env python3
"""
Work Package 3: Physical Baseline Model Validation

This module implements the critical GO/NO-GO gate for the physics-constrained
CBH retrieval hypothesis. It trains a GBDT model using ONLY physical features
(geometric + atmospheric) with strict Leave-One-Flight-Out Cross-Validation.

This is the control experiment to validate whether physical features alone
can provide generalizable CBH predictions, where all previous approaches
(angles-only, MAE embeddings, etc.) have catastrophically failed.

Author: Autonomous Agent
Date: 2025
SOW: SOW-AGENT-CBH-WP-001 Section 5
"""

import sys
from pathlib import Path
import numpy as np
import h5py
import yaml
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# ML imports
try:
    import xgboost as xgb
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("ERROR: XGBoost not available. Install with: pip install xgboost")

from src.hdf5_dataset import HDF5CloudDataset


@dataclass
class FoldMetrics:
    """Container for per-fold evaluation metrics."""

    fold_id: int
    test_flight: str
    n_train: int
    n_test: int
    r2: float
    mae_km: float
    rmse_km: float
    predictions: np.ndarray = None
    targets: np.ndarray = None


class PhysicalBaselineValidator:
    """
    Validates the physical features baseline model using LOO CV.

    This is the critical GO/NO-GO gate: if mean R² ≤ 0, the core
    hypothesis is rejected and the project must pivot.
    """

    def __init__(
        self,
        wp1_features_path: str,
        wp2_features_path: str,
        config_path: str,
        output_dir: str = "sow_outputs/wp3_baseline",
        verbose: bool = True,
    ):
        """
        Initialize the validator.

        Args:
            wp1_features_path: Path to WP1_Features.hdf5
            wp2_features_path: Path to WP2_Features.hdf5
            config_path: Path to config YAML
            output_dir: Directory for outputs
            verbose: Enable verbose logging
        """
        self.wp1_path = Path(wp1_features_path)
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
        Load and merge WP1 and WP2 features.

        Returns:
            X: Combined feature matrix (n_samples, n_features)
            y: Target CBH in km (n_samples,)
            flight_ids: Flight ID for each sample (n_samples,)
            feature_names: List of feature names
        """
        if self.verbose:
            print("\n" + "=" * 80)
            print("Loading Features")
            print("=" * 80)

        # Load WP1 features
        if not self.wp1_path.exists():
            raise FileNotFoundError(f"WP1 features not found: {self.wp1_path}")

        with h5py.File(self.wp1_path, "r") as f:
            if self.verbose:
                print(f"\nWP1 Features: {self.wp1_path}")
                print(f"  Keys: {list(f.keys())}")

            n_samples = len(f["sample_id"])

            # Select geometric features (exclude metadata)
            geometric_features = []
            geometric_names = []

            for key in [
                "derived_geometric_H",
                "shadow_length_pixels",
                "shadow_detection_confidence",
            ]:
                if key in f:
                    geometric_features.append(f[key][:])
                    geometric_names.append(f"geo_{key}")

            wp1_features = (
                np.column_stack(geometric_features)
                if geometric_features
                else np.zeros((n_samples, 0))
            )

            if self.verbose:
                print(f"  Loaded {len(geometric_names)} geometric features")
                print(f"  Features: {geometric_names}")

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
                    else np.zeros((n_samples, 0))
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

        # Combine features
        X = np.hstack([wp1_features, wp2_features])
        feature_names = geometric_names + atmospheric_names

        if self.verbose:
            print(f"\n" + "-" * 80)
            print(f"Combined Feature Matrix:")
            print(f"  Samples: {X.shape[0]}")
            print(f"  Features: {X.shape[1]}")
            print(f"  Feature names: {feature_names}")
            print(f"  Flight distribution:")
            for fid in sorted(np.unique(flight_ids)):
                count = np.sum(flight_ids == fid)
                fname = self.flight_mapping.get(fid, f"Unknown_{fid}")
                print(f"    F{fid} ({fname}): {count} samples")
            print(f"  Target (CBH) range: [{np.min(y):.3f}, {np.max(y):.3f}] km")
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
        """
        Evaluate model on test set.

        Args:
            model: Trained model
            X_test: Test features
            y_test: Test targets (km)

        Returns:
            r2, mae_km, rmse_km
        """
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
            print("Leave-One-Flight-Out Cross-Validation")
            print("=" * 80)

        # Load all features
        X, y, flight_ids, feature_names = self.load_features()

        # Handle missing values
        # Replace NaN with median (common for tree-based models)
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

    def aggregate_results(self, fold_results: List[FoldMetrics]) -> Dict:
        """
        Aggregate results across folds.

        Args:
            fold_results: List of per-fold metrics

        Returns:
            Dictionary with aggregated metrics
        """
        if len(fold_results) == 0:
            raise ValueError("No fold results to aggregate")

        r2_values = [f.r2 for f in fold_results]
        mae_values = [f.mae_km for f in fold_results]
        rmse_values = [f.rmse_km for f in fold_results]

        aggregated = {
            "n_folds": len(fold_results),
            "mean_r2": float(np.mean(r2_values)),
            "std_r2": float(np.std(r2_values)),
            "mean_mae_km": float(np.mean(mae_values)),
            "std_mae_km": float(np.std(mae_values)),
            "mean_rmse_km": float(np.mean(rmse_values)),
            "std_rmse_km": float(np.std(rmse_values)),
            "per_fold_r2": {f.test_flight: f.r2 for f in fold_results},
            "per_fold_mae": {f.test_flight: f.mae_km for f in fold_results},
            "per_fold_rmse": {f.test_flight: f.rmse_km for f in fold_results},
        }

        return aggregated

    def generate_report(self, fold_results: List[FoldMetrics]) -> Dict:
        """
        Generate WP3 validation report.

        Args:
            fold_results: List of per-fold metrics

        Returns:
            Complete report dictionary
        """
        aggregated = self.aggregate_results(fold_results)

        # Determine pass/fail status
        mean_r2 = aggregated["mean_r2"]
        pass_threshold = 0.0
        status = "PASS" if mean_r2 > pass_threshold else "FAIL"

        # Build full report
        report = {
            "model": "Physical_Baseline_GBDT",
            "features": ["geometric", "atmospheric"],
            "n_samples": sum(f.n_train + f.n_test for f in fold_results),
            "n_folds": len(fold_results),
            "timestamp": datetime.now().isoformat(),
            "folds": [
                {
                    "fold_id": f.fold_id,
                    "test_flight": f.test_flight,
                    "n_train": f.n_train,
                    "n_test": f.n_test,
                    "r2": f.r2,
                    "mae_km": f.mae_km,
                    "rmse_km": f.rmse_km,
                }
                for f in fold_results
            ],
            "aggregate_metrics": {
                "mean_r2": aggregated["mean_r2"],
                "std_r2": aggregated["std_r2"],
                "mean_mae_km": aggregated["mean_mae_km"],
                "std_mae_km": aggregated["std_mae_km"],
                "mean_rmse_km": aggregated["mean_rmse_km"],
                "std_rmse_km": aggregated["std_rmse_km"],
            },
            "pass_threshold": pass_threshold,
            "status": status,
            "hypothesis_validation": {
                "question": "Can physical features (geometric + atmospheric) generalize across flights?",
                "result": "VALIDATED" if status == "PASS" else "REJECTED",
                "evidence": f"Mean LOO CV R² = {mean_r2:.4f} (threshold: {pass_threshold})",
            },
        }

        return report

    def save_report(self, report: Dict, filename: str = "WP3_Report.json"):
        """
        Save report to JSON file.

        Args:
            report: Report dictionary
            filename: Output filename
        """
        output_path = self.output_dir / filename

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        if self.verbose:
            print(f"\nReport saved to: {output_path}")

    def print_summary(self, report: Dict):
        """
        Print executive summary of results.

        Args:
            report: Report dictionary
        """
        print("\n" + "=" * 80)
        print("WP-3: PHYSICAL BASELINE VALIDATION - EXECUTIVE SUMMARY")
        print("=" * 80)

        agg = report["aggregate_metrics"]
        status = report["status"]

        print(f"\nModel: {report['model']}")
        print(f"Features: {', '.join(report['features'])}")
        print(f"Total Samples: {report['n_samples']}")
        print(f"CV Folds: {report['n_folds']}")

        print(f"\n{'AGGREGATE RESULTS (LOO CV)':^80}")
        print("-" * 80)
        print(f"  Mean R²:       {agg['mean_r2']:>8.4f} ± {agg['std_r2']:.4f}")
        print(
            f"  Mean MAE:      {agg['mean_mae_km']:>8.4f} ± {agg['std_mae_km']:.4f} km"
        )
        print(
            f"  Mean RMSE:     {agg['mean_rmse_km']:>8.4f} ± {agg['std_rmse_km']:.4f} km"
        )
        print("-" * 80)

        print(f"\n{'PER-FOLD RESULTS':^80}")
        print("-" * 80)
        print(
            f"{'Fold':<6} {'Flight':<10} {'N_test':<8} {'R²':<10} {'MAE (km)':<12} {'RMSE (km)':<12}"
        )
        print("-" * 80)
        for fold in report["folds"]:
            print(
                f"{fold['fold_id']:<6} {fold['test_flight']:<10} {fold['n_test']:<8} "
                f"{fold['r2']:<10.4f} {fold['mae_km']:<12.4f} {fold['rmse_km']:<12.4f}"
            )
        print("-" * 80)

        print(f"\n{'GO/NO-GO DECISION':^80}")
        print("=" * 80)
        print(f"  Threshold: Mean R² > {report['pass_threshold']}")
        print(f"  Achieved:  Mean R² = {agg['mean_r2']:.4f}")

        if status == "PASS":
            print(f"\n  ✓ STATUS: {status}")
            print(f"  ✓ HYPOTHESIS VALIDATED")
            print(f"  ✓ Physical features provide generalizable signal")
            print(f"  ✓ Proceed to WP-4 (Hybrid Model Integration)")
        else:
            print(f"\n  ✗ STATUS: {status}")
            print(f"  ✗ HYPOTHESIS REJECTED")
            print(f"  ✗ Physical features do not generalize across flights")
            print(f"  ✗ Project requires new approach - HALT at WP-3")

        print("=" * 80)

    def run(self) -> Dict:
        """
        Execute complete WP-3 validation pipeline.

        Returns:
            Final report dictionary
        """
        if not XGBOOST_AVAILABLE:
            raise RuntimeError(
                "XGBoost is required for WP-3. Install with: pip install xgboost"
            )

        if self.verbose:
            print("\n" + "=" * 80)
            print("WORK PACKAGE 3: PHYSICAL BASELINE MODEL VALIDATION")
            print("=" * 80)
            print(f"\nSOW: SOW-AGENT-CBH-WP-001 Section 5")
            print(f"Purpose: GO/NO-GO gate for physics-constrained hypothesis")
            print(f"Success Criterion: Mean LOO CV R² > 0")
            print("=" * 80)

        # Run LOO CV
        fold_results = self.run_loo_cv()

        # Generate report
        report = self.generate_report(fold_results)

        # Save report
        self.save_report(report)

        # Print summary
        self.print_summary(report)

        # Critical decision point
        if report["status"] == "FAIL":
            if self.verbose:
                print("\n" + "!" * 80)
                print("CRITICAL: WP-3 FAILED - HYPOTHESIS REJECTED")
                print("!" * 80)
                print("\nThe physical features baseline does not achieve cross-flight")
                print("generalization (R² ≤ 0). This invalidates the core hypothesis.")
                print("\nProject should HALT at WP-3 and not proceed to WP-4.")
                print("A new research direction is required.")
                print("!" * 80)

        return report


def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="WP-3: Physical Baseline Model Validation (GO/NO-GO Gate)"
    )
    parser.add_argument(
        "--wp1-features",
        type=str,
        default="sow_outputs/wp1_geometric/WP1_Features.hdf5",
        help="Path to WP1 geometric features",
    )
    parser.add_argument(
        "--wp2-features",
        type=str,
        default="sow_outputs/wp2_atmospheric/WP2_Features.hdf5",
        help="Path to WP2 atmospheric features",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/bestComboConfig.yaml",
        help="Path to configuration YAML",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="sow_outputs/wp3_baseline",
        help="Output directory",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    # Create validator
    validator = PhysicalBaselineValidator(
        wp1_features_path=args.wp1_features,
        wp2_features_path=args.wp2_features,
        config_path=args.config,
        output_dir=args.output_dir,
        verbose=args.verbose,
    )

    # Run validation
    report = validator.run()

    # Exit with appropriate code
    if report["status"] == "FAIL":
        return 1  # Indicate failure
    else:
        return 0  # Success


if __name__ == "__main__":
    sys.exit(main())
