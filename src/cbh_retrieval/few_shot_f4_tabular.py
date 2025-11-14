#!/usr/bin/env python3
"""
Sprint 6 - Phase 2, Task 2.2: Domain Adaptation for Flight F4 (Tabular GBDT)

This script implements few-shot domain adaptation to mitigate the
Leave-One-Flight-Out (LOO) failure on Flight F4.

Key Challenge:
Flight F4 exhibits extreme domain shift, causing model failure in LOO validation.

Solution:
Few-shot fine-tuning with 5, 10, and 20 samples from F4 to adapt the GBDT model
to the F4 domain while preserving general knowledge.

Approach:
1. Train baseline GBDT on F1, F2, F3 (excluding all F4)
2. For each few-shot size (5, 10, 20):
   - Randomly sample N samples from F4
   - Fine-tune (or retrain with augmented data) GBDT
   - Evaluate on remaining F4 samples
3. Compare to baseline LOO performance

Deliverables:
- Few-shot adapted models
- Domain adaptation results report (JSON + Markdown)
- Learning curves (R² vs. number of F4 samples)
- Visualizations

Author: Sprint 6 Execution Agent
Date: 2025-01-11
"""

import json
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import h5py
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# Set style
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")


class DomainAdaptationF4Tabular:
    """
    Performs few-shot domain adaptation for Flight F4 using tabular GBDT.
    """

    def __init__(
        self,
        integrated_features_path: str,
        output_dir: str,
        random_seed: int = 42,
    ):
        self.integrated_features_path = Path(integrated_features_path)
        self.output_dir = Path(output_dir)
        self.random_seed = random_seed
        np.random.seed(random_seed)

        # Create output directories
        self.models_dir = self.output_dir / "models" / "domain_adapted"
        self.reports_dir = self.output_dir / "reports"
        self.figures_dir = self.output_dir / "figures" / "domain_adaptation"

        for dir_path in [self.models_dir, self.reports_dir, self.figures_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        print(f" Domain adaptation initialized")
        print(f" Output directory: {self.output_dir}")
        print(f" Random seed: {self.random_seed}")

    def load_data(self) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """Load tabular features and labels."""
        print(f"\n{'=' * 80}")
        print("Loading Tabular Dataset")
        print(f"{'=' * 80}")

        with h5py.File(self.integrated_features_path, "r") as h5f:
            # Load metadata
            cbh = h5f["metadata/cbh_km"][:]
            flight_ids = h5f["metadata/flight_id"][:]

            # Load ERA5 features
            era5_features = h5f["atmospheric_features/era5_features"][:]
            era5_feature_names = h5f["atmospheric_features/era5_feature_names"][:]
            era5_feature_names = [
                name.decode() if isinstance(name, bytes) else name
                for name in era5_feature_names
            ]

            # Load geometric features
            geometric_group = h5f["geometric_features"]
            geometric_dict = {}
            for key in geometric_group.keys():
                geometric_dict[key] = geometric_group[key][:]

            # Combine features
            features_dict = {}

            # Add ERA5 features
            for i, name in enumerate(era5_feature_names):
                features_dict[name] = era5_features[:, i]

            # Add geometric features
            features_dict.update(geometric_dict)

            features_df = pd.DataFrame(features_dict)

        print(f" Total samples: {len(features_df)}")
        print(f" Number of features: {len(features_df.columns)}")
        print(f" CBH range: [{cbh.min():.3f}, {cbh.max():.3f}] km")
        print(f" Unique flights: {np.unique(flight_ids)}")

        # Print per-flight statistics
        print(f"\nPer-Flight Statistics:")
        for flight_id in np.unique(flight_ids):
            flight_mask = flight_ids == flight_id
            flight_cbh = cbh[flight_mask]
            print(
                f"  Flight {flight_id}: N={np.sum(flight_mask)}, "
                f"Mean CBH={flight_cbh.mean():.3f} km ± {flight_cbh.std():.3f} km"
            )

        return features_df, cbh, flight_ids

    def get_baseline_loo_performance(
        self, features_df: pd.DataFrame, cbh: np.ndarray, flight_ids: np.ndarray
    ) -> Dict[str, float]:
        """
        Get baseline Leave-One-Out performance for F4.
        Train on F1, F2, F3 and evaluate on F4.
        """
        print(f"\n{'=' * 80}")
        print("Computing Baseline LOO Performance for F4")
        print(f"{'=' * 80}")

        # Identify F4
        target_flight = 4
        train_mask = flight_ids != target_flight
        test_mask = flight_ids == target_flight

        X_train = features_df[train_mask].values
        y_train = cbh[train_mask]
        X_test = features_df[test_mask].values
        y_test = cbh[test_mask]

        print(f" Training samples (F1, F2, F3): {np.sum(train_mask)}")
        print(f" Test samples (F4): {np.sum(test_mask)}")

        # Impute missing values
        imputer = SimpleImputer(strategy="mean")
        X_train_imputed = imputer.fit_transform(X_train)
        X_test_imputed = imputer.transform(X_test)

        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imputed)
        X_test_scaled = scaler.transform(X_test_imputed)

        # Train GBDT
        model = GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=7,
            subsample=0.8,
            max_features=0.8,
            random_state=self.random_seed,
        )
        model.fit(X_train_scaled, y_train)

        # Predict on F4
        y_pred = model.predict(X_test_scaled)

        # Compute metrics
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        baseline_metrics = {
            "r2": float(r2),
            "mae_km": float(mae),
            "rmse_km": float(rmse),
            "mae_m": float(mae * 1000),
            "rmse_m": float(rmse * 1000),
            "n_train": int(np.sum(train_mask)),
            "n_test": int(np.sum(test_mask)),
        }

        print(f"\nBaseline LOO Performance (F4):")
        print(f"  R² = {r2:.4f}")
        print(f"  MAE = {mae:.4f} km ({mae * 1000:.1f} m)")
        print(f"  RMSE = {rmse:.4f} km ({rmse * 1000:.1f} m)")

        return baseline_metrics

    def few_shot_experiment(
        self,
        features_df: pd.DataFrame,
        cbh: np.ndarray,
        flight_ids: np.ndarray,
        n_few_shot: int,
        n_trials: int = 10,
    ) -> Dict[str, Any]:
        """
        Run few-shot adaptation experiment.

        Args:
            features_df: Feature dataframe
            cbh: CBH labels
            flight_ids: Flight identifiers
            n_few_shot: Number of F4 samples to use for adaptation
            n_trials: Number of random trials to average over

        Returns:
            Dictionary with trial results and aggregated metrics
        """
        print(f"\n{'=' * 80}")
        print(f"Few-Shot Experiment: N = {n_few_shot} samples from F4")
        print(f"{'=' * 80}")

        target_flight = 4
        train_mask = flight_ids != target_flight
        f4_mask = flight_ids == target_flight

        X_base = features_df[train_mask].values
        y_base = cbh[train_mask]

        X_f4 = features_df[f4_mask].values
        y_f4 = cbh[f4_mask]
        f4_indices = np.where(f4_mask)[0]

        results_trials = []

        for trial in range(n_trials):
            # Randomly sample n_few_shot from F4
            np.random.seed(self.random_seed + trial)
            sampled_indices = np.random.choice(
                len(f4_indices), size=n_few_shot, replace=False
            )
            few_shot_indices = f4_indices[sampled_indices]

            # Create training set: base + few-shot F4
            X_train = np.vstack([X_base, features_df.iloc[few_shot_indices].values])
            y_train = np.concatenate([y_base, cbh[few_shot_indices]])

            # Test set: remaining F4 samples
            test_mask = np.ones(len(X_f4), dtype=bool)
            test_mask[sampled_indices] = False
            X_test = X_f4[test_mask]
            y_test = y_f4[test_mask]

            # Impute missing values
            imputer = SimpleImputer(strategy="mean")
            X_train_imputed = imputer.fit_transform(X_train)
            X_test_imputed = imputer.transform(X_test)

            # Standardize
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_imputed)
            X_test_scaled = scaler.transform(X_test_imputed)

            # Train adapted model
            model = GradientBoostingRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=7,
                subsample=0.8,
                max_features=0.8,
                random_state=self.random_seed + trial,
            )
            model.fit(X_train_scaled, y_train)

            # Predict
            y_pred = model.predict(X_test_scaled)

            # Compute metrics
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            trial_result = {
                "trial": trial,
                "r2": float(r2),
                "mae_km": float(mae),
                "rmse_km": float(rmse),
                "n_train": len(y_train),
                "n_test": len(y_test),
                "few_shot_indices": few_shot_indices.tolist(),
            }
            results_trials.append(trial_result)

            print(f"  Trial {trial + 1}/{n_trials}: R² = {r2:.4f}, MAE = {mae:.4f} km")

        # Aggregate results
        r2_values = [t["r2"] for t in results_trials]
        mae_values = [t["mae_km"] for t in results_trials]
        rmse_values = [t["rmse_km"] for t in results_trials]

        aggregated = {
            "n_few_shot": n_few_shot,
            "n_trials": n_trials,
            "mean_r2": float(np.mean(r2_values)),
            "std_r2": float(np.std(r2_values)),
            "mean_mae_km": float(np.mean(mae_values)),
            "std_mae_km": float(np.std(mae_values)),
            "mean_rmse_km": float(np.mean(rmse_values)),
            "std_rmse_km": float(np.std(rmse_values)),
            "trials": results_trials,
        }

        print(f"\nAggregated Results ({n_few_shot} samples):")
        print(f"  Mean R² = {aggregated['mean_r2']:.4f} ± {aggregated['std_r2']:.4f}")
        print(
            f"  Mean MAE = {aggregated['mean_mae_km']:.4f} ± {aggregated['std_mae_km']:.4f} km"
        )

        return aggregated

    def run_all_experiments(
        self, features_df: pd.DataFrame, cbh: np.ndarray, flight_ids: np.ndarray
    ) -> Dict[str, Any]:
        """Run all few-shot experiments (5, 10, 20 samples)."""
        print(f"\n{'=' * 80}")
        print("Running All Few-Shot Experiments")
        print(f"{'=' * 80}")

        # Get baseline
        baseline = self.get_baseline_loo_performance(features_df, cbh, flight_ids)

        # Run few-shot experiments
        few_shot_sizes = [5, 10, 20]
        few_shot_results = {}

        for n_samples in few_shot_sizes:
            result = self.few_shot_experiment(
                features_df, cbh, flight_ids, n_samples, n_trials=10
            )
            few_shot_results[f"{n_samples}_samples"] = result

        # Compile full report
        report = {
            "task": "2.2_domain_adaptation_f4",
            "timestamp": datetime.now().isoformat(),
            "target_flight": "F4",
            "baseline_loo_r2": baseline["r2"],
            "baseline_loo_mae_km": baseline["mae_km"],
            "domain_shift_description": (
                "Flight F4 exhibits significantly lower mean CBH compared to other flights, "
                "causing catastrophic failure in leave-one-out validation. Few-shot adaptation "
                "uses a small number of F4 samples to adapt the model to the F4 domain."
            ),
            "few_shot_experiments": {
                "5_samples": {
                    "r2": few_shot_results["5_samples"]["mean_r2"],
                    "std_r2": few_shot_results["5_samples"]["std_r2"],
                    "mae_km": few_shot_results["5_samples"]["mean_mae_km"],
                    "std_mae_km": few_shot_results["5_samples"]["std_mae_km"],
                    "improvement_over_baseline": float(
                        few_shot_results["5_samples"]["mean_r2"] - baseline["r2"]
                    ),
                },
                "10_samples": {
                    "r2": few_shot_results["10_samples"]["mean_r2"],
                    "std_r2": few_shot_results["10_samples"]["std_r2"],
                    "mae_km": few_shot_results["10_samples"]["mean_mae_km"],
                    "std_mae_km": few_shot_results["10_samples"]["std_mae_km"],
                    "improvement_over_baseline": float(
                        few_shot_results["10_samples"]["mean_r2"] - baseline["r2"]
                    ),
                },
                "20_samples": {
                    "r2": few_shot_results["20_samples"]["mean_r2"],
                    "std_r2": few_shot_results["20_samples"]["std_r2"],
                    "mae_km": few_shot_results["20_samples"]["mean_mae_km"],
                    "std_mae_km": few_shot_results["20_samples"]["std_mae_km"],
                    "improvement_over_baseline": float(
                        few_shot_results["20_samples"]["mean_r2"] - baseline["r2"]
                    ),
                },
            },
            "conclusion": self._generate_conclusion(baseline["r2"], few_shot_results),
            "baseline_metrics": baseline,
            "detailed_results": few_shot_results,
        }

        return report

    def _generate_conclusion(
        self, baseline_r2: float, few_shot_results: Dict[str, Any]
    ) -> str:
        """Generate conclusion text."""
        r2_5 = few_shot_results["5_samples"]["mean_r2"]
        r2_10 = few_shot_results["10_samples"]["mean_r2"]
        r2_20 = few_shot_results["20_samples"]["mean_r2"]

        improvement_5 = r2_5 - baseline_r2
        improvement_10 = r2_10 - baseline_r2
        improvement_20 = r2_20 - baseline_r2

        if improvement_20 > 0.1:
            conclusion = (
                f" Few-shot adaptation successfully mitigates F4 domain shift. "
                f"With 20 samples, R² improves from {baseline_r2:.4f} to {r2_20:.4f} "
                f"(+{improvement_20:.4f}), demonstrating effective domain adaptation. "
                f"Even 5 samples provide meaningful improvement (+{improvement_5:.4f}). "
                f"Recommendation: Use 10-20 labeled F4 samples for production deployment "
                f"to handle F4-like domains."
            )
        elif improvement_20 > 0:
            conclusion = (
                f"Few-shot adaptation provides modest improvement for F4. "
                f"R² improves from {baseline_r2:.4f} to {r2_20:.4f} (+{improvement_20:.4f}) "
                f"with 20 samples. Further investigation into F4-specific features or "
                f"alternative adaptation strategies may be needed."
            )
        else:
            conclusion = (
                f"Few-shot adaptation shows limited effectiveness for F4. "
                f"Baseline R² = {baseline_r2:.4f}, best few-shot R² = {r2_20:.4f}. "
                f"F4 may require more sophisticated domain adaptation techniques or "
                f"deeper investigation into the root causes of domain shift."
            )

        return conclusion

    def create_visualizations(self, report: Dict[str, Any]):
        """Create comprehensive domain adaptation visualizations."""
        print(f"\n{'=' * 80}")
        print("Creating Visualizations")
        print(f"{'=' * 80}")

        # 1. Learning curve: R² vs. number of few-shot samples
        self._plot_learning_curve(report)

        # 2. Performance comparison
        self._plot_performance_comparison(report)

        # 3. Improvement over baseline
        self._plot_improvement_analysis(report)

        # 4. Detailed trial results
        self._plot_trial_results(report)

        print(f" All visualizations saved to {self.figures_dir}")

    def _plot_learning_curve(self, report: Dict[str, Any]):
        """Plot learning curve: R² vs. number of few-shot samples."""
        baseline_r2 = report["baseline_loo_r2"]
        experiments = report["few_shot_experiments"]

        n_samples = [0, 5, 10, 20]
        r2_means = [
            baseline_r2,
            experiments["5_samples"]["r2"],
            experiments["10_samples"]["r2"],
            experiments["20_samples"]["r2"],
        ]
        r2_stds = [
            0,  # Baseline has no variance (single model)
            experiments["5_samples"]["std_r2"],
            experiments["10_samples"]["std_r2"],
            experiments["20_samples"]["std_r2"],
        ]

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.errorbar(
            n_samples,
            r2_means,
            yerr=r2_stds,
            marker="o",
            markersize=10,
            linewidth=2.5,
            capsize=5,
            capthick=2,
            color="#2ecc71",
            label="Few-Shot Adapted Model",
        )
        ax.axhline(
            y=baseline_r2,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Baseline LOO (R² = {baseline_r2:.4f})",
            alpha=0.7,
        )

        ax.set_xlabel(
            "Number of F4 Samples Used for Adaptation", fontsize=12, fontweight="bold"
        )
        ax.set_ylabel("R² Score on F4 Test Set", fontsize=12, fontweight="bold")
        ax.set_title(
            "Few-Shot Learning Curve: Domain Adaptation for F4",
            fontsize=14,
            fontweight="bold",
        )
        ax.legend(fontsize=11, loc="best")
        ax.grid(True, alpha=0.3)
        ax.set_xticks(n_samples)

        # Add value labels
        for x, y, std in zip(n_samples, r2_means, r2_stds):
            if x == 0:
                label = f"{y:.4f}"
            else:
                label = f"{y:.4f}\n±{std:.4f}"
            ax.text(
                x,
                y + std + 0.02,
                label,
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=9,
            )

        plt.tight_layout()
        plt.savefig(
            self.figures_dir / "few_shot_learning_curve.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.savefig(
            self.figures_dir / "few_shot_learning_curve.pdf", bbox_inches="tight"
        )
        plt.close()
        print(" Created learning curve plot")

    def _plot_performance_comparison(self, report: Dict[str, Any]):
        """Plot performance comparison across all few-shot sizes."""
        baseline = report["baseline_metrics"]
        experiments = report["few_shot_experiments"]

        models = ["Baseline\n(0 samples)", "5 samples", "10 samples", "20 samples"]
        r2_means = [
            baseline["r2"],
            experiments["5_samples"]["r2"],
            experiments["10_samples"]["r2"],
            experiments["20_samples"]["r2"],
        ]
        r2_stds = [
            0,
            experiments["5_samples"]["std_r2"],
            experiments["10_samples"]["std_r2"],
            experiments["20_samples"]["std_r2"],
        ]
        mae_means = [
            baseline["mae_km"],
            experiments["5_samples"]["mae_km"],
            experiments["10_samples"]["mae_km"],
            experiments["20_samples"]["mae_km"],
        ]
        mae_stds = [
            0,
            experiments["5_samples"]["std_mae_km"],
            experiments["10_samples"]["std_mae_km"],
            experiments["20_samples"]["std_mae_km"],
        ]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        colors = ["#e74c3c", "#3498db", "#f39c12", "#2ecc71"]

        # R² comparison
        bars1 = axes[0].bar(
            models,
            r2_means,
            yerr=r2_stds,
            capsize=5,
            color=colors,
            alpha=0.8,
            edgecolor="black",
        )
        axes[0].set_ylabel("R² Score", fontsize=12, fontweight="bold")
        axes[0].set_title(
            "R² Performance on F4 Test Set", fontsize=14, fontweight="bold"
        )
        axes[0].grid(axis="y", alpha=0.3)

        # Add value labels
        for bar, mean, std in zip(bars1, r2_means, r2_stds):
            height = bar.get_height()
            if std > 0:
                label = f"{mean:.4f}\n±{std:.4f}"
            else:
                label = f"{mean:.4f}"
            axes[0].text(
                bar.get_x() + bar.get_width() / 2.0,
                height + std + 0.01,
                label,
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=9,
            )

        # MAE comparison
        bars2 = axes[1].bar(
            models,
            mae_means,
            yerr=mae_stds,
            capsize=5,
            color=colors,
            alpha=0.8,
            edgecolor="black",
        )
        axes[1].set_ylabel("MAE (km)", fontsize=12, fontweight="bold")
        axes[1].set_title(
            "MAE Performance on F4 Test Set", fontsize=14, fontweight="bold"
        )
        axes[1].grid(axis="y", alpha=0.3)

        # Add value labels
        for bar, mean, std in zip(bars2, mae_means, mae_stds):
            height = bar.get_height()
            if std > 0:
                label = f"{mean:.3f}\n±{std:.3f}"
            else:
                label = f"{mean:.3f}"
            axes[1].text(
                bar.get_x() + bar.get_width() / 2.0,
                height + std + 0.005,
                label,
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=9,
            )

        plt.tight_layout()
        plt.savefig(
            self.figures_dir / "few_shot_performance_comparison.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.savefig(
            self.figures_dir / "few_shot_performance_comparison.pdf",
            bbox_inches="tight",
        )
        plt.close()
        print(" Created performance comparison plot")

    def _plot_improvement_analysis(self, report: Dict[str, Any]):
        """Plot improvement over baseline."""
        baseline_r2 = report["baseline_loo_r2"]
        experiments = report["few_shot_experiments"]

        models = ["5 samples", "10 samples", "20 samples"]
        improvements = [
            experiments["5_samples"]["improvement_over_baseline"],
            experiments["10_samples"]["improvement_over_baseline"],
            experiments["20_samples"]["improvement_over_baseline"],
        ]

        fig, ax = plt.subplots(figsize=(10, 6))

        colors = ["#3498db", "#f39c12", "#2ecc71"]
        bars = ax.barh(models, improvements, color=colors, alpha=0.8, edgecolor="black")

        ax.axvline(x=0, color="black", linestyle="-", linewidth=2)
        ax.set_xlabel("Improvement in R² over Baseline", fontsize=12, fontweight="bold")
        ax.set_title(
            f"Few-Shot Improvement over Baseline (R² = {baseline_r2:.4f})",
            fontsize=14,
            fontweight="bold",
        )
        ax.grid(axis="x", alpha=0.3)

        # Add value labels
        for bar, imp in zip(bars, improvements):
            width = bar.get_width()
            label_x = width + (0.01 if width > 0 else -0.01)
            ax.text(
                label_x,
                bar.get_y() + bar.get_height() / 2,
                f"{imp:+.4f}",
                ha="left" if width > 0 else "right",
                va="center",
                fontweight="bold",
                fontsize=11,
            )

        plt.tight_layout()
        plt.savefig(
            self.figures_dir / "few_shot_improvement.png", dpi=300, bbox_inches="tight"
        )
        plt.savefig(self.figures_dir / "few_shot_improvement.pdf", bbox_inches="tight")
        plt.close()
        print(" Created improvement analysis plot")

    def _plot_trial_results(self, report: Dict[str, Any]):
        """Plot individual trial results for all few-shot sizes."""
        detailed = report["detailed_results"]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        for ax, (label, key) in zip(
            axes,
            [
                ("5 Samples", "5_samples"),
                ("10 Samples", "10_samples"),
                ("20 Samples", "20_samples"),
            ],
        ):
            trials = detailed[key]["trials"]
            r2_values = [t["r2"] for t in trials]
            trial_numbers = [t["trial"] + 1 for t in trials]

            ax.scatter(
                trial_numbers,
                r2_values,
                s=100,
                alpha=0.7,
                edgecolors="black",
                linewidth=1.5,
            )
            ax.axhline(
                y=np.mean(r2_values),
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Mean = {np.mean(r2_values):.4f}",
            )
            ax.axhline(
                y=report["baseline_loo_r2"],
                color="orange",
                linestyle=":",
                linewidth=2,
                label=f"Baseline = {report['baseline_loo_r2']:.4f}",
            )

            ax.set_xlabel("Trial Number", fontsize=11, fontweight="bold")
            ax.set_ylabel("R² Score", fontsize=11, fontweight="bold")
            ax.set_title(f"{label} from F4", fontsize=12, fontweight="bold")
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_xticks(trial_numbers)

        plt.tight_layout()
        plt.savefig(
            self.figures_dir / "few_shot_trial_results.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.savefig(
            self.figures_dir / "few_shot_trial_results.pdf", bbox_inches="tight"
        )
        plt.close()
        print(" Created trial results plot")

    def save_report(self, report: Dict[str, Any]):
        """Save domain adaptation report."""
        # Save JSON
        json_path = self.reports_dir / "domain_adaptation_f4_report.json"
        with open(json_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n Saved JSON report to {json_path}")

        # Save Markdown
        md_path = self.reports_dir / "domain_adaptation_f4_summary.md"
        self._generate_markdown_report(report, md_path)
        print(f" Saved Markdown summary to {md_path}")

    def _generate_markdown_report(self, report: Dict[str, Any], output_path: Path):
        """Generate markdown summary report."""
        exp = report["few_shot_experiments"]

        md = f"""# Task 2.2: Domain Adaptation for Flight F4 - Results Summary

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Executive Summary

{report["conclusion"]}

## Problem Statement

{report["domain_shift_description"]}

## Baseline Performance (LOO on F4)

Training on F1, F2, F3 and testing on F4 (zero-shot):

- **R²:** {report["baseline_loo_r2"]:.4f}
- **MAE:** {report["baseline_loo_mae_km"]:.4f} km ({report["baseline_metrics"]["mae_m"]:.1f} m)
- **RMSE:** {report["baseline_metrics"]["rmse_km"]:.4f} km ({report["baseline_metrics"]["rmse_m"]:.1f} m)

## Few-Shot Adaptation Results

### 5 Samples from F4

- **R²:** {exp["5_samples"]["r2"]:.4f} ± {exp["5_samples"]["std_r2"]:.4f}
- **MAE:** {exp["5_samples"]["mae_km"]:.4f} ± {exp["5_samples"]["std_mae_km"]:.4f} km
- **Improvement over baseline:** {exp["5_samples"]["improvement_over_baseline"]:+.4f} ({exp["5_samples"]["improvement_over_baseline"] / report["baseline_loo_r2"] * 100:+.2f}%)

### 10 Samples from F4

- **R²:** {exp["10_samples"]["r2"]:.4f} ± {exp["10_samples"]["std_r2"]:.4f}
- **MAE:** {exp["10_samples"]["mae_km"]:.4f} ± {exp["10_samples"]["std_mae_km"]:.4f} km
- **Improvement over baseline:** {exp["10_samples"]["improvement_over_baseline"]:+.4f} ({exp["10_samples"]["improvement_over_baseline"] / report["baseline_loo_r2"] * 100:+.2f}%)

### 20 Samples from F4

- **R²:** {exp["20_samples"]["r2"]:.4f} ± {exp["20_samples"]["std_r2"]:.4f}
- **MAE:** {exp["20_samples"]["mae_km"]:.4f} ± {exp["20_samples"]["std_mae_km"]:.4f} km
- **Improvement over baseline:** {exp["20_samples"]["improvement_over_baseline"]:+.4f} ({exp["20_samples"]["improvement_over_baseline"] / report["baseline_loo_r2"] * 100:+.2f}%)

## Key Findings

1. **Few-shot effectiveness**: {" Effective" if exp["20_samples"]["improvement_over_baseline"] > 0.1 else " Modest improvement" if exp["20_samples"]["improvement_over_baseline"] > 0 else " Limited improvement"}

2. **Sample efficiency**: {"Even 5 samples provide meaningful improvement" if exp["5_samples"]["improvement_over_baseline"] > 0.05 else "10+ samples needed for noticeable improvement"}

3. **Diminishing returns**: {"Significant gains from 5→10 samples" if (exp["10_samples"]["r2"] - exp["5_samples"]["r2"]) > 0.05 else "Marginal gains beyond 10 samples"}

4. **Production recommendation**: {"Use 10-20 labeled F4 samples for robust F4 performance" if exp["20_samples"]["r2"] > 0.5 else "Further investigation into F4 domain shift needed"}

## Experimental Protocol

- **Method:** Few-shot domain adaptation with GBDT
- **Base training:** F1, F2, F3 (excluding all F4)
- **Adaptation:** Add N samples from F4, retrain model
- **Evaluation:** Test on remaining F4 samples
- **Trials:** 10 random trials per few-shot size
- **Random seed:** {report["baseline_metrics"].get("random_seed", 42)}

## Visualizations

All domain adaptation visualizations are saved in:
`./figures/domain_adaptation/`

- `few_shot_learning_curve.png/pdf` - R² vs. number of F4 samples
- `few_shot_performance_comparison.png/pdf` - Performance comparison
- `few_shot_improvement.png/pdf` - Improvement over baseline
- `few_shot_trial_results.png/pdf` - Individual trial results

---
*Task 2.2 Complete - Domain Adaptation for Flight F4*
"""

        with open(output_path, "w") as f:
            f.write(md)


def main():
    """Main execution."""
    print("=" * 80)
    print("TASK 2.2: DOMAIN ADAPTATION FOR FLIGHT F4 (TABULAR GBDT)")
    print("=" * 80)

    # Paths
    project_root = Path(__file__).parent.parent.parent.parent
    integrated_features_path = (
        project_root
        / "sow_outputs"
        / "integrated_features"
        / "Integrated_Features.hdf5"
    )
    output_dir = project_root / "sow_outputs" / "sprint6"

    print(f"\nProject root: {project_root}")
    print(f"Integrated features: {integrated_features_path}")
    print(f"Output directory: {output_dir}")

    # Initialize adapter
    adapter = DomainAdaptationF4Tabular(
        integrated_features_path=str(integrated_features_path),
        output_dir=str(output_dir),
        random_seed=42,
    )

    # Load data
    features_df, cbh, flight_ids = adapter.load_data()

    # Run all experiments
    report = adapter.run_all_experiments(features_df, cbh, flight_ids)

    # Create visualizations
    adapter.create_visualizations(report)

    # Save report
    adapter.save_report(report)

    # Print summary
    print("\n" + "=" * 80)
    print("DOMAIN ADAPTATION COMPLETE")
    print("=" * 80)
    print(f"\n Baseline R² (F4): {report['baseline_loo_r2']:.4f}")
    print(
        f" Best Few-Shot R² (20 samples): {report['few_shot_experiments']['20_samples']['r2']:.4f}"
    )
    print(
        f" Improvement: {report['few_shot_experiments']['20_samples']['improvement_over_baseline']:+.4f}"
    )
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
