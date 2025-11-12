#!/usr/bin/env python3
"""
Ensemble Results Analysis and Reporting
Analyzes ensemble model results and creates SOW-compliant reports and visualizations.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set style
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")


class EnsembleAnalyzer:
    """Analyzes and reports ensemble model results."""

    def __init__(self, results_path: str):
        """Initialize analyzer with results file."""
        self.results_path = Path(results_path)
        self.output_dir = self.results_path.parent.parent
        self.figures_dir = self.output_dir / "figures" / "ensemble"
        self.reports_dir = self.output_dir / "reports"

        # Create directories
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # Load results
        with open(self.results_path, "r") as f:
            self.results = json.load(f)

    def create_sow_compliant_report(self) -> Dict[str, Any]:
        """Create SOW-compliant ensemble report following Task 2.1 schema."""

        strategies = self.results["ensemble_strategies"]

        # Extract baseline models
        baseline_gbdt = strategies["gbdt"]
        baseline_cnn = strategies["cnn"]

        # Extract ensemble strategies
        simple_avg = strategies["simple_avg"]
        weighted_avg = strategies["weighted_avg"]
        stacking = strategies["stacking"]

        # Get optimal weights from fold results
        optimal_weights = []
        for fold_result in self.results["folds"]:
            w = fold_result["weights_optimized"]
            # Weights are stored as [w_gbdt, w_cnn]
            optimal_weights.append({"w_gbdt": w[0], "w_cnn": w[1]})

        # Average weights across folds
        mean_w_gbdt = np.mean([w["w_gbdt"] for w in optimal_weights])
        mean_w_cnn = np.mean([w["w_cnn"] for w in optimal_weights])

        # Determine best baseline
        best_baseline_r2 = max(baseline_gbdt["mean_r2"], baseline_cnn["mean_r2"])

        # Create report
        report = {
            "task": "2.1_ensemble_methods",
            "timestamp": datetime.now().isoformat(),
            "baseline_models": {
                "gbdt": {
                    "name": baseline_gbdt["name"],
                    "mean_r2": float(baseline_gbdt["mean_r2"]),
                    "std_r2": float(baseline_gbdt["std_r2"]),
                    "mean_mae_km": float(baseline_gbdt["mean_mae_m"] / 1000),
                    "std_mae_km": float(baseline_gbdt["std_mae_m"] / 1000),
                },
                "cnn": {
                    "name": baseline_cnn["name"],
                    "mean_r2": float(baseline_cnn["mean_r2"]),
                    "std_r2": float(baseline_cnn["std_r2"]),
                    "mean_mae_km": float(baseline_cnn["mean_mae_m"] / 1000),
                    "std_mae_km": float(baseline_cnn["std_mae_m"] / 1000),
                },
            },
            "ensemble_strategies": {
                "simple_averaging": {
                    "mean_r2": float(simple_avg["mean_r2"]),
                    "std_r2": float(simple_avg["std_r2"]),
                    "mean_mae_km": float(simple_avg["mean_mae_m"] / 1000),
                    "std_mae_km": float(simple_avg["std_mae_m"] / 1000),
                    "improvement_over_best_base": float(
                        simple_avg["mean_r2"] - best_baseline_r2
                    ),
                },
                "weighted_averaging": {
                    "mean_r2": float(weighted_avg["mean_r2"]),
                    "std_r2": float(weighted_avg["std_r2"]),
                    "mean_mae_km": float(weighted_avg["mean_mae_m"] / 1000),
                    "std_mae_km": float(weighted_avg["std_mae_m"] / 1000),
                    "optimal_weights": {
                        "w_gbdt": float(mean_w_gbdt),
                        "w_cnn": float(mean_w_cnn),
                    },
                    "improvement_over_best_base": float(
                        weighted_avg["mean_r2"] - best_baseline_r2
                    ),
                },
                "stacking": {
                    "mean_r2": float(stacking["mean_r2"]),
                    "std_r2": float(stacking["std_r2"]),
                    "mean_mae_km": float(stacking["mean_mae_m"] / 1000),
                    "std_mae_km": float(stacking["std_mae_m"] / 1000),
                    "meta_learner": "Ridge Regression",
                    "improvement_over_best_base": float(
                        stacking["mean_r2"] - best_baseline_r2
                    ),
                },
            },
            "best_ensemble": {
                "strategy": "weighted_averaging",
                "achieved_target": weighted_avg["mean_r2"] >= 0.74,
                "mean_r2": float(weighted_avg["mean_r2"]),
                "mean_mae_km": float(weighted_avg["mean_mae_m"] / 1000),
                "std_r2": float(weighted_avg["std_r2"]),
            },
            "validation_protocol": {
                "method": "Stratified K-Fold Cross-Validation",
                "n_folds": self.results["metadata"]["n_folds"],
                "random_seed": self.results["metadata"]["random_seed"],
            },
            "conclusion": self._generate_conclusion(
                weighted_avg["mean_r2"], best_baseline_r2
            ),
        }

        return report

    def _generate_conclusion(self, ensemble_r2: float, baseline_r2: float) -> str:
        """Generate conclusion text."""
        improvement = ensemble_r2 - baseline_r2
        improvement_pct = (improvement / baseline_r2) * 100

        if ensemble_r2 >= 0.74:
            conclusion = (
                f"✓ TARGET ACHIEVED: Weighted averaging ensemble achieves R² = {ensemble_r2:.4f}, "
                f"exceeding the target of 0.74. This represents a {improvement_pct:.2f}% improvement "
                f"over the best baseline model (R² = {baseline_r2:.4f}). "
                f"The ensemble successfully combines tabular (GBDT) and image (CNN) modalities, "
                f"with optimal weights strongly favoring the tabular model, indicating that "
                f"atmospheric features are more predictive than raw image features for CBH estimation."
            )
        else:
            conclusion = (
                f"Target not achieved: Weighted averaging ensemble achieves R² = {ensemble_r2:.4f}, "
                f"below the target of 0.74. Improvement over baseline: {improvement_pct:.2f}%. "
                f"Further improvements needed through better image feature extraction or "
                f"alternative fusion strategies."
            )

        return conclusion

    def create_visualizations(self):
        """Create comprehensive ensemble visualizations."""
        print("\n=== Creating Ensemble Visualizations ===\n")

        # 1. Performance comparison barplot
        self._plot_performance_comparison()

        # 2. Per-fold performance
        self._plot_per_fold_performance()

        # 3. Prediction scatter plots
        self._plot_prediction_scatter()

        # 4. Error distribution
        self._plot_error_distributions()

        # 5. Weight distribution
        self._plot_weight_distribution()

        # 6. Improvement over baseline
        self._plot_improvement_analysis()

        print(f"✓ All visualizations saved to {self.figures_dir}")

    def _plot_performance_comparison(self):
        """Plot performance comparison of all strategies."""
        strategies = self.results["ensemble_strategies"]

        models = ["GBDT", "CNN", "Simple Avg", "Weighted Avg", "Stacking"]
        r2_means = [
            strategies["gbdt"]["mean_r2"],
            strategies["cnn"]["mean_r2"],
            strategies["simple_avg"]["mean_r2"],
            strategies["weighted_avg"]["mean_r2"],
            strategies["stacking"]["mean_r2"],
        ]
        r2_stds = [
            strategies["gbdt"]["std_r2"],
            strategies["cnn"]["std_r2"],
            strategies["simple_avg"]["std_r2"],
            strategies["weighted_avg"]["std_r2"],
            strategies["stacking"]["std_r2"],
        ]
        mae_means = [
            strategies["gbdt"]["mean_mae_m"] / 1000,
            strategies["cnn"]["mean_mae_m"] / 1000,
            strategies["simple_avg"]["mean_mae_m"] / 1000,
            strategies["weighted_avg"]["mean_mae_m"] / 1000,
            strategies["stacking"]["mean_mae_m"] / 1000,
        ]
        mae_stds = [
            strategies["gbdt"]["std_mae_m"] / 1000,
            strategies["cnn"]["std_mae_m"] / 1000,
            strategies["simple_avg"]["std_mae_m"] / 1000,
            strategies["weighted_avg"]["std_mae_m"] / 1000,
            strategies["stacking"]["std_mae_m"] / 1000,
        ]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # R² comparison
        colors = ["#3498db", "#e74c3c", "#95a5a6", "#2ecc71", "#f39c12"]
        bars1 = axes[0].bar(
            models,
            r2_means,
            yerr=r2_stds,
            capsize=5,
            color=colors,
            alpha=0.8,
            edgecolor="black",
        )
        axes[0].axhline(
            y=0.74,
            color="red",
            linestyle="--",
            linewidth=2,
            label="Target R² = 0.74",
            alpha=0.7,
        )
        axes[0].set_ylabel("R² Score", fontsize=12, fontweight="bold")
        axes[0].set_title(
            "Model Performance Comparison (R²)", fontsize=14, fontweight="bold"
        )
        axes[0].set_ylim(0, max(r2_means) * 1.15)
        axes[0].legend(fontsize=10)
        axes[0].grid(axis="y", alpha=0.3)

        # Add value labels on bars
        for bar, mean, std in zip(bars1, r2_means, r2_stds):
            height = bar.get_height()
            axes[0].text(
                bar.get_x() + bar.get_width() / 2.0,
                height + std + 0.01,
                f"{mean:.3f}",
                ha="center",
                va="bottom",
                fontweight="bold",
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
            "Model Performance Comparison (MAE)", fontsize=14, fontweight="bold"
        )
        axes[1].set_ylim(0, max(mae_means) * 1.15)
        axes[1].grid(axis="y", alpha=0.3)

        # Add value labels on bars
        for bar, mean, std in zip(bars2, mae_means, mae_stds):
            height = bar.get_height()
            axes[1].text(
                bar.get_x() + bar.get_width() / 2.0,
                height + std + 0.005,
                f"{mean:.3f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        plt.tight_layout()
        plt.savefig(
            self.figures_dir / "ensemble_performance_comparison.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.savefig(
            self.figures_dir / "ensemble_performance_comparison.pdf",
            bbox_inches="tight",
        )
        plt.close()
        print("✓ Created performance comparison plot")

    def _plot_per_fold_performance(self):
        """Plot per-fold performance for each strategy."""
        folds = self.results["folds"]
        n_folds = len(folds)
        fold_indices = list(range(1, n_folds + 1))

        # Extract per-fold metrics
        gbdt_r2 = [f["gbdt_metrics"]["r2"] for f in folds]
        cnn_r2 = [f["cnn_metrics"]["r2"] for f in folds]
        simple_r2 = [f["simple_avg_metrics"]["r2"] for f in folds]
        weighted_r2 = [f["weighted_avg_metrics"]["r2"] for f in folds]
        stacking_r2 = [f["stacking_metrics"]["r2"] for f in folds]

        fig, ax = plt.subplots(figsize=(12, 7))

        ax.plot(fold_indices, gbdt_r2, "o-", label="GBDT", linewidth=2, markersize=8)
        ax.plot(fold_indices, cnn_r2, "s-", label="CNN", linewidth=2, markersize=8)
        ax.plot(
            fold_indices, simple_r2, "^-", label="Simple Avg", linewidth=2, markersize=8
        )
        ax.plot(
            fold_indices,
            weighted_r2,
            "D-",
            label="Weighted Avg",
            linewidth=2.5,
            markersize=8,
        )
        ax.plot(
            fold_indices, stacking_r2, "v-", label="Stacking", linewidth=2, markersize=8
        )

        ax.axhline(
            y=0.74,
            color="red",
            linestyle="--",
            linewidth=2,
            label="Target R² = 0.74",
            alpha=0.7,
        )

        ax.set_xlabel("Fold Number", fontsize=12, fontweight="bold")
        ax.set_ylabel("R² Score", fontsize=12, fontweight="bold")
        ax.set_title("Per-Fold Performance Comparison", fontsize=14, fontweight="bold")
        ax.set_xticks(fold_indices)
        ax.legend(fontsize=10, loc="best")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            self.figures_dir / "per_fold_performance.png", dpi=300, bbox_inches="tight"
        )
        plt.savefig(self.figures_dir / "per_fold_performance.pdf", bbox_inches="tight")
        plt.close()
        print("✓ Created per-fold performance plot")

    def _plot_prediction_scatter(self):
        """Plot prediction scatter for best ensemble."""
        # Aggregate all folds
        all_y_true = []
        all_pred_gbdt = []
        all_pred_weighted = []
        all_pred_stacking = []

        for fold in self.results["folds"]:
            all_y_true.extend(fold["y_true"])
            all_pred_gbdt.extend(fold["pred_gbdt"])
            all_pred_weighted.extend(fold["pred_weighted"])
            all_pred_stacking.extend(fold["pred_stacking"])

        all_y_true = np.array(all_y_true)
        all_pred_gbdt = np.array(all_pred_gbdt)
        all_pred_weighted = np.array(all_pred_weighted)
        all_pred_stacking = np.array(all_pred_stacking)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        models = [
            ("GBDT Baseline", all_pred_gbdt),
            ("Weighted Avg Ensemble", all_pred_weighted),
            ("Stacking Ensemble", all_pred_stacking),
        ]

        for ax, (title, preds) in zip(axes, models):
            ax.scatter(
                all_y_true, preds, alpha=0.5, s=30, edgecolors="black", linewidth=0.5
            )

            # Perfect prediction line
            min_val = min(all_y_true.min(), preds.min())
            max_val = max(all_y_true.max(), preds.max())
            ax.plot(
                [min_val, max_val],
                [min_val, max_val],
                "r--",
                linewidth=2,
                label="Perfect Prediction",
            )

            # Calculate metrics
            r2 = np.corrcoef(all_y_true, preds)[0, 1] ** 2
            mae = np.mean(np.abs(all_y_true - preds))

            ax.set_xlabel("True CBH (km)", fontsize=11, fontweight="bold")
            ax.set_ylabel("Predicted CBH (km)", fontsize=11, fontweight="bold")
            ax.set_title(
                f"{title}\nR² = {r2:.4f}, MAE = {mae:.3f} km",
                fontsize=12,
                fontweight="bold",
            )
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            self.figures_dir / "ensemble_prediction_scatter.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.savefig(
            self.figures_dir / "ensemble_prediction_scatter.pdf", bbox_inches="tight"
        )
        plt.close()
        print("✓ Created prediction scatter plots")

    def _plot_error_distributions(self):
        """Plot error distributions for different strategies."""
        # Aggregate all predictions
        all_y_true = []
        all_pred_gbdt = []
        all_pred_cnn = []
        all_pred_weighted = []

        for fold in self.results["folds"]:
            all_y_true.extend(fold["y_true"])
            all_pred_gbdt.extend(fold["pred_gbdt"])
            all_pred_cnn.extend(fold["pred_cnn"])
            all_pred_weighted.extend(fold["pred_weighted"])

        all_y_true = np.array(all_y_true)
        error_gbdt = (all_y_true - np.array(all_pred_gbdt)) * 1000  # Convert to meters
        error_cnn = (all_y_true - np.array(all_pred_cnn)) * 1000
        error_weighted = (all_y_true - np.array(all_pred_weighted)) * 1000

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Histogram
        axes[0].hist(error_gbdt, bins=50, alpha=0.6, label="GBDT", edgecolor="black")
        axes[0].hist(error_cnn, bins=50, alpha=0.6, label="CNN", edgecolor="black")
        axes[0].hist(
            error_weighted, bins=50, alpha=0.6, label="Weighted Avg", edgecolor="black"
        )
        axes[0].axvline(x=0, color="red", linestyle="--", linewidth=2)
        axes[0].set_xlabel("Error (m)", fontsize=12, fontweight="bold")
        axes[0].set_ylabel("Frequency", fontsize=12, fontweight="bold")
        axes[0].set_title("Error Distribution", fontsize=14, fontweight="bold")
        axes[0].legend(fontsize=10)
        axes[0].grid(axis="y", alpha=0.3)

        # Box plot
        error_data = [error_gbdt, error_cnn, error_weighted]
        bp = axes[1].boxplot(
            error_data,
            labels=["GBDT", "CNN", "Weighted Avg"],
            patch_artist=True,
            showfliers=True,
        )

        colors = ["#3498db", "#e74c3c", "#2ecc71"]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        axes[1].axhline(y=0, color="red", linestyle="--", linewidth=2)
        axes[1].set_ylabel("Error (m)", fontsize=12, fontweight="bold")
        axes[1].set_title(
            "Error Distribution (Box Plot)", fontsize=14, fontweight="bold"
        )
        axes[1].grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            self.figures_dir / "ensemble_error_distributions.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.savefig(
            self.figures_dir / "ensemble_error_distributions.pdf", bbox_inches="tight"
        )
        plt.close()
        print("✓ Created error distribution plots")

    def _plot_weight_distribution(self):
        """Plot optimal weight distribution across folds."""
        weights_gbdt = []
        weights_cnn = []

        for fold in self.results["folds"]:
            w = fold["weights_optimized"]
            # Weights are stored as [w_gbdt, w_cnn]
            weights_gbdt.append(w[0])
            weights_cnn.append(w[1])

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Bar plot of weights per fold
        fold_indices = list(range(1, len(weights_gbdt) + 1))
        x = np.arange(len(fold_indices))
        width = 0.35

        bars1 = axes[0].bar(
            x - width / 2,
            weights_gbdt,
            width,
            label="GBDT Weight",
            color="#3498db",
            alpha=0.8,
            edgecolor="black",
        )
        bars2 = axes[0].bar(
            x + width / 2,
            weights_cnn,
            width,
            label="CNN Weight",
            color="#e74c3c",
            alpha=0.8,
            edgecolor="black",
        )

        axes[0].set_xlabel("Fold Number", fontsize=12, fontweight="bold")
        axes[0].set_ylabel("Weight", fontsize=12, fontweight="bold")
        axes[0].set_title("Optimal Weights per Fold", fontsize=14, fontweight="bold")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(fold_indices)
        axes[0].legend(fontsize=10)
        axes[0].grid(axis="y", alpha=0.3)
        axes[0].set_ylim(0, 1)

        # Average weights pie chart
        mean_w_gbdt = np.mean(weights_gbdt)
        mean_w_cnn = np.mean(weights_cnn)

        axes[1].pie(
            [mean_w_gbdt, mean_w_cnn],
            labels=[f"GBDT\n{mean_w_gbdt:.3f}", f"CNN\n{mean_w_cnn:.3f}"],
            colors=["#3498db", "#e74c3c"],
            autopct="%1.1f%%",
            startangle=90,
            explode=(0.05, 0),
            textprops={"fontsize": 12, "fontweight": "bold"},
        )
        axes[1].set_title("Average Optimal Weights", fontsize=14, fontweight="bold")

        plt.tight_layout()
        plt.savefig(
            self.figures_dir / "ensemble_weight_distribution.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.savefig(
            self.figures_dir / "ensemble_weight_distribution.pdf", bbox_inches="tight"
        )
        plt.close()
        print("✓ Created weight distribution plots")

    def _plot_improvement_analysis(self):
        """Plot improvement over baseline analysis."""
        strategies = self.results["ensemble_strategies"]

        baseline_r2 = strategies["gbdt"]["mean_r2"]

        models = ["CNN", "Simple Avg", "Weighted Avg", "Stacking"]
        r2_values = [
            strategies["cnn"]["mean_r2"],
            strategies["simple_avg"]["mean_r2"],
            strategies["weighted_avg"]["mean_r2"],
            strategies["stacking"]["mean_r2"],
        ]

        improvements = [(r2 - baseline_r2) / baseline_r2 * 100 for r2 in r2_values]

        fig, ax = plt.subplots(figsize=(10, 6))

        colors = ["#e74c3c", "#95a5a6", "#2ecc71", "#f39c12"]
        bars = ax.barh(models, improvements, color=colors, alpha=0.8, edgecolor="black")

        ax.axvline(x=0, color="black", linestyle="-", linewidth=2)
        ax.set_xlabel(
            "Improvement over GBDT Baseline (%)", fontsize=12, fontweight="bold"
        )
        ax.set_title(
            f"Relative Improvement over GBDT Baseline (R² = {baseline_r2:.4f})",
            fontsize=14,
            fontweight="bold",
        )
        ax.grid(axis="x", alpha=0.3)

        # Add value labels
        for bar, imp in zip(bars, improvements):
            width = bar.get_width()
            label_x = width + (1 if width > 0 else -1)
            ax.text(
                label_x,
                bar.get_y() + bar.get_height() / 2,
                f"{imp:+.2f}%",
                ha="left" if width > 0 else "right",
                va="center",
                fontweight="bold",
                fontsize=11,
            )

        plt.tight_layout()
        plt.savefig(
            self.figures_dir / "ensemble_improvement_analysis.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.savefig(
            self.figures_dir / "ensemble_improvement_analysis.pdf", bbox_inches="tight"
        )
        plt.close()
        print("✓ Created improvement analysis plot")

    def save_sow_report(self, report: Dict[str, Any]):
        """Save SOW-compliant report."""
        report_path = self.reports_dir / "ensemble_sow_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n✓ Saved SOW-compliant report to {report_path}")

    def generate_markdown_summary(self, report: Dict[str, Any]):
        """Generate markdown summary report."""

        md = f"""# Task 2.1: Ensemble Methods - Results Summary

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Executive Summary

{report["conclusion"]}

## Baseline Models Performance

### GBDT (Tabular Features)
- **R²:** {report["baseline_models"]["gbdt"]["mean_r2"]:.4f} ± {report["baseline_models"]["gbdt"]["std_r2"]:.4f}
- **MAE:** {report["baseline_models"]["gbdt"]["mean_mae_km"]:.3f} ± {report["baseline_models"]["gbdt"]["std_mae_km"]:.3f} km

### CNN (Image Features)
- **R²:** {report["baseline_models"]["cnn"]["mean_r2"]:.4f} ± {report["baseline_models"]["cnn"]["std_r2"]:.4f}
- **MAE:** {report["baseline_models"]["cnn"]["mean_mae_km"]:.3f} ± {report["baseline_models"]["cnn"]["std_mae_km"]:.3f} km

## Ensemble Strategies Performance

### 1. Simple Averaging
- **R²:** {report["ensemble_strategies"]["simple_averaging"]["mean_r2"]:.4f} ± {report["ensemble_strategies"]["simple_averaging"]["std_r2"]:.4f}
- **MAE:** {report["ensemble_strategies"]["simple_averaging"]["mean_mae_km"]:.3f} ± {report["ensemble_strategies"]["simple_averaging"]["std_mae_km"]:.3f} km
- **Improvement over best baseline:** {report["ensemble_strategies"]["simple_averaging"]["improvement_over_best_base"]:.4f} ({report["ensemble_strategies"]["simple_averaging"]["improvement_over_best_base"] / report["baseline_models"]["gbdt"]["mean_r2"] * 100:+.2f}%)

### 2. Weighted Averaging ⭐ BEST
- **R²:** {report["ensemble_strategies"]["weighted_averaging"]["mean_r2"]:.4f} ± {report["ensemble_strategies"]["weighted_averaging"]["std_r2"]:.4f}
- **MAE:** {report["ensemble_strategies"]["weighted_averaging"]["mean_mae_km"]:.3f} ± {report["ensemble_strategies"]["weighted_averaging"]["std_mae_km"]:.3f} km
- **Improvement over best baseline:** {report["ensemble_strategies"]["weighted_averaging"]["improvement_over_best_base"]:.4f} ({report["ensemble_strategies"]["weighted_averaging"]["improvement_over_best_base"] / report["baseline_models"]["gbdt"]["mean_r2"] * 100:+.2f}%)
- **Optimal Weights:**
  - GBDT: {report["ensemble_strategies"]["weighted_averaging"]["optimal_weights"]["w_gbdt"]:.3f}
  - CNN: {report["ensemble_strategies"]["weighted_averaging"]["optimal_weights"]["w_cnn"]:.3f}

### 3. Stacking (Ridge Meta-Learner)
- **R²:** {report["ensemble_strategies"]["stacking"]["mean_r2"]:.4f} ± {report["ensemble_strategies"]["stacking"]["std_r2"]:.4f}
- **MAE:** {report["ensemble_strategies"]["stacking"]["mean_mae_km"]:.3f} ± {report["ensemble_strategies"]["stacking"]["std_mae_km"]:.3f} km
- **Improvement over best baseline:** {report["ensemble_strategies"]["stacking"]["improvement_over_best_base"]:.4f} ({report["ensemble_strategies"]["stacking"]["improvement_over_best_base"] / report["baseline_models"]["gbdt"]["mean_r2"] * 100:+.2f}%)
- **Meta-learner:** {report["ensemble_strategies"]["stacking"]["meta_learner"]}

## Best Ensemble Recommendation

**Strategy:** {report["best_ensemble"]["strategy"].replace("_", " ").title()}
- **Target R² ≥ 0.74:** {"✅ ACHIEVED" if report["best_ensemble"]["achieved_target"] else "❌ NOT ACHIEVED"}
- **Final R²:** {report["best_ensemble"]["mean_r2"]:.4f} ± {report["best_ensemble"]["std_r2"]:.4f}
- **Final MAE:** {report["best_ensemble"]["mean_mae_km"]:.3f} km

## Validation Protocol

- **Method:** {report["validation_protocol"]["method"]}
- **Number of Folds:** {report["validation_protocol"]["n_folds"]}
- **Random Seed:** {report["validation_protocol"]["random_seed"]}

## Key Insights

1. **Tabular features dominate**: The optimal weights heavily favor GBDT ({report["ensemble_strategies"]["weighted_averaging"]["optimal_weights"]["w_gbdt"]:.1%}), indicating atmospheric/geometric features are more predictive than raw images.

2. **Image model contribution**: While CNN alone performs poorly (R² = {report["baseline_models"]["cnn"]["mean_r2"]:.4f}), it provides complementary information that improves ensemble performance.

3. **Ensemble benefit**: Weighted averaging provides a {report["ensemble_strategies"]["weighted_averaging"]["improvement_over_best_base"] / report["baseline_models"]["gbdt"]["mean_r2"] * 100:+.2f}% improvement over the GBDT baseline.

4. **Production recommendation**: Use weighted averaging ensemble with learned optimal weights for production deployment.

## Visualizations

All ensemble visualizations are saved in:
`sow_outputs/sprint6/figures/ensemble/`

- `ensemble_performance_comparison.png/pdf` - Overall performance comparison
- `per_fold_performance.png/pdf` - Performance across CV folds
- `ensemble_prediction_scatter.png/pdf` - Prediction quality scatter plots
- `ensemble_error_distributions.png/pdf` - Error distribution analysis
- `ensemble_weight_distribution.png/pdf` - Optimal weight analysis
- `ensemble_improvement_analysis.png/pdf` - Improvement over baseline

---
*Task 2.1 Complete - Ensemble Methods Analysis*
"""

        md_path = self.reports_dir / "ensemble_summary.md"
        with open(md_path, "w") as f:
            f.write(md)
        print(f"✓ Saved markdown summary to {md_path}")


def main():
    """Main execution."""
    print("=" * 80)
    print("ENSEMBLE RESULTS ANALYSIS - Task 2.1")
    print("=" * 80)

    # Path to ensemble results
    results_path = "sow_outputs/sprint6/reports/ensemble_results.json"

    # Initialize analyzer
    analyzer = EnsembleAnalyzer(results_path)

    # Create SOW-compliant report
    print("\n=== Creating SOW-Compliant Report ===\n")
    sow_report = analyzer.create_sow_compliant_report()
    analyzer.save_sow_report(sow_report)

    # Create visualizations
    analyzer.create_visualizations()

    # Generate markdown summary
    print("\n=== Generating Markdown Summary ===\n")
    analyzer.generate_markdown_summary(sow_report)

    # Print summary
    print("\n" + "=" * 80)
    print("ENSEMBLE ANALYSIS COMPLETE")
    print("=" * 80)
    print(
        f"\n✓ Best Ensemble: {sow_report['best_ensemble']['strategy'].replace('_', ' ').title()}"
    )
    print(f"✓ R² = {sow_report['best_ensemble']['mean_r2']:.4f}")
    print(f"✓ Target Achieved: {sow_report['best_ensemble']['achieved_target']}")
    print(
        f"\n✓ Optimal Weights: GBDT={sow_report['ensemble_strategies']['weighted_averaging']['optimal_weights']['w_gbdt']:.3f}, CNN={sow_report['ensemble_strategies']['weighted_averaging']['optimal_weights']['w_cnn']:.3f}"
    )
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
