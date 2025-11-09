#!/usr/bin/env python3
"""
Work Package 3: Physical Baseline Model - K-Fold Cross-Validation

This module implements the physical baseline using stratified K-Fold CV
to provide a fair comparison with the WP-4 hybrid models.

Unlike the LOO CV version (which tests flight-to-flight generalization),
this version uses stratified K-Fold to match the WP-4 validation protocol.

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
    from sklearn.model_selection import StratifiedKFold

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print(
        "ERROR: XGBoost not available. Install with: pip install xgboost scikit-learn"
    )

from src.hdf5_dataset import HDF5CloudDataset


@dataclass
class FoldMetrics:
    """Container for per-fold evaluation metrics."""

    fold_id: int
    n_train: int
    n_test: int
    r2: float
    mae_km: float
    rmse_km: float
    predictions: np.ndarray = None
    targets: np.ndarray = None


class PhysicalBaselineKFold:
    """
    Physical baseline model using stratified K-Fold CV.

    Features:
    - WP1: Geometric features (shadow-derived CBH, solar angles, etc.)
    - WP2: Atmospheric features (ERA5 reanalysis)

    Model: Gradient Boosted Decision Trees (XGBoost)
    Validation: Stratified 5-Fold CV (matches WP-4 protocol)
    """

    def __init__(
        self,
        wp1_features_path: str,
        wp2_features_path: str,
        config_path: str,
        output_dir: str = "sow_outputs/wp3_kfold",
        n_folds: int = 5,
        random_state: int = 42,
        verbose: bool = True,
    ):
        """
        Initialize the validator.

        Args:
            wp1_features_path: Path to WP1_Features.hdf5
            wp2_features_path: Path to WP2_Features.hdf5
            config_path: Path to config YAML
            output_dir: Directory for outputs
            n_folds: Number of folds for K-Fold CV
            random_state: Random seed for reproducibility
            verbose: Enable verbose logging
        """
        self.wp1_path = Path(wp1_features_path)
        self.wp2_path = Path(wp2_features_path)
        self.config_path = Path(config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.n_folds = n_folds
        self.random_state = random_state
        self.verbose = verbose

        # Load config
        with open(self.config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Flight mapping (as per SOW Section 2.2)
        self.flight_mapping = {
            0: "30Oct24",  # F_0: n=501
            1: "10Feb25",  # F_1: n=191
            2: "23Oct24",  # F_2: n=105
            3: "12Feb25",  # F_3: n=92
            4: "18Feb25",  # F_4: n=44
        }

        # Flight name to ID mapping
        self.flight_name_to_id = {}
        for fid, fname in self.flight_mapping.items():
            self.flight_name_to_id[fname] = fid

        # XGBoost hyperparameters (tuned for CBH regression)
        self.xgb_params = {
            "objective": "reg:squarederror",
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_estimators": 100,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
            "gamma": 0.1,
            "reg_alpha": 0.01,
            "reg_lambda": 1.0,
            "random_state": self.random_state,
            "n_jobs": -1,
            "verbosity": 0,
        }

    def load_features(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
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

        # Load geometric features
        if not self.wp1_path.exists():
            raise FileNotFoundError(f"WP1 features not found: {self.wp1_path}")

        with h5py.File(self.wp1_path, "r") as f:
            if self.verbose:
                print(f"\nWP1 Features: {self.wp1_path}")
                print(f"  Keys: {list(f.keys())}")

            geo_cbh = f["derived_geometric_H"][:]
            geo_length = f["shadow_length_pixels"][:]
            geo_conf = f["shadow_detection_confidence"][:]

            geometric_features = np.column_stack([geo_cbh, geo_length, geo_conf])
            geo_names = [
                "derived_geometric_H",
                "shadow_length_pixels",
                "shadow_detection_confidence",
            ]

            # Load targets
            y = f["true_cbh_km"][:]

            # Get sample IDs and extract flight IDs
            sample_ids = f["sample_id"][:]

            if self.verbose:
                print(f"  Geometric features: {geometric_features.shape}")
                print(f"  Targets: {y.shape}")

        # Load ERA5 features
        if not self.wp2_path.exists():
            raise FileNotFoundError(f"WP2 features not found: {self.wp2_path}")

        with h5py.File(self.wp2_path, "r") as f:
            if self.verbose:
                print(f"\nWP2 Features: {self.wp2_path}")
                print(f"  Keys: {list(f.keys())}")

            era5_features = f["features"][:]
            era5_names = [
                n.decode() if isinstance(n, bytes) else n for n in f["feature_names"][:]
            ]

            if self.verbose:
                print(f"  ERA5 features: {era5_features.shape}")

        # Create flight IDs from sample order (matches dataset creation order)
        # Based on known distribution: F0=501, F1=191, F2=105, F3=92, F4=44
        n_samples = len(y)
        flight_sizes = [501, 191, 105, 92, 44]
        flight_ids = np.zeros(n_samples, dtype=int)
        idx = 0
        for fid, size in enumerate(flight_sizes[:5]):  # Only 5 flights
            flight_ids[idx : idx + size] = fid
            idx += size

        # Combine features
        X = np.concatenate([geometric_features, era5_features], axis=1)
        feature_names = geo_names + era5_names

        if self.verbose:
            print(f"\nCombined Features:")
            print(f"  Shape: {X.shape}")
            print(f"  Total samples: {n_samples}")
            print(f"  Target (CBH) range: [{y.min():.3f}, {y.max():.3f}] km")
            print(f"  Target mean: {y.mean():.3f} km")
            print(f"  Flight distribution:")
            for fid in sorted(np.unique(flight_ids)):
                count = np.sum(flight_ids == fid)
                fname = self.flight_mapping.get(fid, f"Unknown_{fid}")
                print(f"    F{fid} ({fname}): {count} samples")
            print(f"  NaN counts:")
            for j, name in enumerate(feature_names):
                n_nan = np.sum(np.isnan(X[:, j]))
                if n_nan > 0:
                    print(
                        f"    {name}: {n_nan} / {len(X)} ({100 * n_nan / len(X):.1f}%)"
                    )

        return X, y, flight_ids, feature_names

    def train_gbdt(self, X_train: np.ndarray, y_train: np.ndarray) -> xgb.XGBRegressor:
        """
        Train Gradient Boosted Decision Tree model.

        Args:
            X_train: Training features
            y_train: Training targets

        Returns:
            Trained XGBoost model
        """
        model = xgb.XGBRegressor(**self.xgb_params)
        model.fit(X_train, y_train, verbose=False)
        return model

    def evaluate_fold(
        self, model: xgb.XGBRegressor, X_test: np.ndarray, y_test: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        Evaluate model on test set.

        Args:
            model: Trained model
            X_test: Test features
            y_test: Test targets

        Returns:
            r2, mae_km, rmse_km
        """
        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        return r2, mae, rmse

    def run_kfold_cv(self) -> List[FoldMetrics]:
        """
        Execute Stratified K-Fold Cross-Validation.

        This matches the WP-4 validation protocol for fair comparison.

        Returns:
            List of FoldMetrics for each fold
        """
        if self.verbose:
            print("\n" + "=" * 80)
            print(f"Stratified {self.n_folds}-Fold Cross-Validation")
            print("=" * 80)

        # Load all features
        X, y, flight_ids, feature_names = self.load_features()

        # Handle missing values (median imputation for tree-based models)
        X_clean = X.copy()
        for j in range(X.shape[1]):
            col = X[:, j]
            if np.any(np.isnan(col)):
                median_val = np.nanmedian(col)
                X_clean[np.isnan(col), j] = median_val
                if self.verbose:
                    n_nan = np.sum(np.isnan(col))
                    print(
                        f"  Imputed {n_nan} NaN in {feature_names[j]} with median={median_val:.3f}"
                    )

        # Create stratification bins (stratify on CBH quantiles)
        n_bins = 10
        cbh_bins = np.percentile(y, np.linspace(0, 100, n_bins + 1))
        strata = np.digitize(y, cbh_bins[1:-1])

        if self.verbose:
            print(f"\nStratification:")
            print(f"  CBH bins: {n_bins}")
            print(f"  Bin edges: {cbh_bins}")
            unique, counts = np.unique(strata, return_counts=True)
            print(f"  Stratum counts: {dict(zip(unique, counts))}")

        # K-Fold CV
        skf = StratifiedKFold(
            n_splits=self.n_folds, shuffle=True, random_state=self.random_state
        )

        fold_results = []

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_clean, strata)):
            if self.verbose:
                print(f"\n" + "-" * 80)
                print(f"Fold {fold_idx + 1}/{self.n_folds}")
                print("-" * 80)

            # Split data
            X_train, X_test = X_clean[train_idx], X_clean[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            n_train = len(y_train)
            n_test = len(y_test)

            if self.verbose:
                print(f"  Training samples: {n_train}")
                print(f"  Test samples: {n_test}")
                print(f"  Train CBH: {y_train.mean():.3f} ± {y_train.std():.3f} km")
                print(f"  Test CBH:  {y_test.mean():.3f} ± {y_test.std():.3f} km")

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

            # Get predictions
            y_pred = model.predict(X_test_scaled)

            if self.verbose:
                print(f"  Results:")
                print(f"    R²   = {r2:.4f}")
                print(f"    MAE  = {mae:.4f} km")
                print(f"    RMSE = {rmse:.4f} km")

            # Store results
            fold_metrics = FoldMetrics(
                fold_id=fold_idx,
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
            Dictionary of aggregated statistics
        """
        r2_values = [f.r2 for f in fold_results]
        mae_values = [f.mae_km for f in fold_results]
        rmse_values = [f.rmse_km for f in fold_results]

        aggregate = {
            "n_folds": len(fold_results),
            "mean_r2": float(np.mean(r2_values)),
            "std_r2": float(np.std(r2_values)),
            "mean_mae_km": float(np.mean(mae_values)),
            "std_mae_km": float(np.std(mae_values)),
            "mean_rmse_km": float(np.mean(rmse_values)),
            "std_rmse_km": float(np.std(rmse_values)),
            "per_fold": [
                {
                    "fold": f.fold_id,
                    "n_train": f.n_train,
                    "n_test": f.n_test,
                    "r2": float(f.r2),
                    "mae_km": float(f.mae_km),
                    "rmse_km": float(f.rmse_km),
                }
                for f in fold_results
            ],
        }

        return aggregate

    def generate_report(self, fold_results: List[FoldMetrics]) -> Dict:
        """
        Generate comprehensive report.

        Args:
            fold_results: List of per-fold metrics

        Returns:
            Complete report dictionary
        """
        aggregate = self.aggregate_results(fold_results)

        report = {
            "model_name": "physical_baseline",
            "description": "GBDT baseline (geometric + atmospheric features)",
            "validation": "Stratified K-Fold Cross-Validation",
            "n_folds": self.n_folds,
            "random_state": self.random_state,
            "hyperparameters": self.xgb_params,
            "results": aggregate,
            "timestamp": datetime.now().isoformat(),
        }

        return report

    def save_report(self, report: Dict, filename: str = "WP3_Report_kfold.json"):
        """Save report to JSON file."""
        output_path = self.output_dir / filename
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        if self.verbose:
            print(f"\n✓ Report saved to: {output_path}")

    def print_summary(self, report: Dict):
        """Print formatted summary."""
        print("\n" + "=" * 80)
        print("WP-3 PHYSICAL BASELINE RESULTS (K-FOLD CV)")
        print("=" * 80)

        results = report["results"]

        print(f"\nModel: {report['model_name']}")
        print(f"Description: {report['description']}")
        print(f"Validation: {report['validation']}")
        print(f"Number of folds: {results['n_folds']}")

        print("\nPer-Fold Performance:")
        print(
            f"{'Fold':<6} {'n_train':<8} {'n_test':<8} {'R²':<10} {'MAE (km)':<10} {'RMSE (km)':<10}"
        )
        print("-" * 80)

        for fold in results["per_fold"]:
            print(
                f"{fold['fold']:<6} {fold['n_train']:<8} {fold['n_test']:<8} "
                f"{fold['r2']:<10.4f} {fold['mae_km']:<10.4f} {fold['rmse_km']:<10.4f}"
            )

        print("\nAggregate Performance:")
        print(f"  Mean R²:    {results['mean_r2']:.4f} ± {results['std_r2']:.4f}")
        print(
            f"  Mean MAE:   {results['mean_mae_km']:.4f} ± {results['std_mae_km']:.4f} km"
        )
        print(
            f"  Mean RMSE:  {results['mean_rmse_km']:.4f} ± {results['std_rmse_km']:.4f} km"
        )

        # Verdict
        print("\n" + "=" * 80)
        if results["mean_r2"] > 0.3:
            print("✅ GOOD - Physical features alone show strong performance")
        elif results["mean_r2"] > 0.0:
            print("⚠️  WEAK - Physical features have limited predictive power")
        else:
            print("❌ FAIL - Physical features do not generalize")
        print("=" * 80)

    def run(self):
        """Execute complete K-Fold CV workflow."""
        if not XGBOOST_AVAILABLE:
            raise RuntimeError("XGBoost is required. Install with: pip install xgboost")

        print("\n" + "=" * 80)
        print("WP-3: PHYSICAL BASELINE MODEL - K-FOLD CV")
        print("=" * 80)
        print(f"Output directory: {self.output_dir}")
        print(f"Number of folds: {self.n_folds}")
        print(f"Random state: {self.random_state}")

        # Run K-Fold CV
        fold_results = self.run_kfold_cv()

        # Generate report
        report = self.generate_report(fold_results)

        # Save report
        self.save_report(report)

        # Print summary
        self.print_summary(report)

        return report


def main():
    """Main entry point."""
    # Paths (adjust as needed)
    project_root = Path(__file__).parent.parent
    wp1_path = project_root / "sow_outputs" / "wp1_geometric" / "WP1_Features.hdf5"
    wp2_path = project_root / "sow_outputs" / "wp2_atmospheric" / "WP2_Features.hdf5"
    config_path = project_root / "configs" / "config.yaml"

    # Check paths
    if not wp1_path.exists():
        print(f"ERROR: WP1 features not found: {wp1_path}")
        return 1

    if not wp2_path.exists():
        print(f"ERROR: WP2 features not found: {wp2_path}")
        return 1

    if not config_path.exists():
        print(f"ERROR: Config not found: {config_path}")
        return 1

    # Run validator
    validator = PhysicalBaselineKFold(
        wp1_features_path=str(wp1_path),
        wp2_features_path=str(wp2_path),
        config_path=str(config_path),
        output_dir="sow_outputs/wp3_kfold",
        n_folds=5,
        random_state=42,
        verbose=True,
    )

    report = validator.run()

    return 0


if __name__ == "__main__":
    sys.exit(main())
