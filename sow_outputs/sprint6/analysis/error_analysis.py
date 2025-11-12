#!/usr/bin/env python3
"""
Sprint 6 - Phase 1, Task 1.3: Comprehensive Error Analysis

This script performs comprehensive error analysis on the GBDT CBH prediction model:
1. Identifies worst-performing samples
2. Analyzes error correlations with input features
3. Performs per-flight error analysis
4. Generates statistical significance tests
5. Creates detailed visualizations

Author: Sprint 6 Agent
Date: 2025
"""

import json
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 10

print("=" * 80)
print("Sprint 6 - Phase 1, Task 1.3: Comprehensive Error Analysis")
print("=" * 80)

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[3]
INTEGRATED_FEATURES = (
    PROJECT_ROOT / "sow_outputs/integrated_features/Integrated_Features.hdf5"
)
OUTPUT_DIR = PROJECT_ROOT / "sow_outputs/sprint6"
FIGURES_DIR = OUTPUT_DIR / "figures/error_analysis"
REPORTS_DIR = OUTPUT_DIR / "reports"

# Create directories
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

print(f"Project Root: {PROJECT_ROOT}")
print(f"Integrated Features: {INTEGRATED_FEATURES}")
print(f"Output Directory: {OUTPUT_DIR}")
print(f"✓ Error analysis output directory: {FIGURES_DIR}")


class ComprehensiveErrorAnalyzer:
    """Performs comprehensive error analysis on CBH predictions."""

    def __init__(self, random_seed=42):
        self.random_seed = random_seed
        self.figures_dir = FIGURES_DIR
        self.reports_dir = REPORTS_DIR

    def load_data(
        self, hdf5_path: Path
    ) -> Tuple[np.ndarray, np.ndarray, List[str], pd.DataFrame, Dict]:
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
                if key != "derived_geometric_H":
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

        # Create metadata dataframe with all auxiliary features
        metadata = pd.DataFrame(
            {
                "flight_id": flight_ids,
                "sample_id": sample_ids,
                "cbh_km": cbh_km,
            }
        )

        # Add auxiliary features for correlation analysis
        if "sza_deg" in feature_names:
            idx = feature_names.index("sza_deg")
            metadata["sza"] = X[:, idx]
        if "saa_deg" in feature_names:
            idx = feature_names.index("saa_deg")
            metadata["saa"] = X[:, idx]
        if "altitude_km" in feature_names:
            idx = feature_names.index("altitude_km")
            metadata["altitude"] = X[:, idx]

        # Add ERA5 features if available
        for i, name in enumerate(era5_feature_names):
            if name in ["blh", "lcl", "t2m", "d2m"]:
                metadata[name] = era5_features[:, i]

        return X, y, feature_names, metadata, flight_mapping

    def train_model(self, X: np.ndarray, y: np.ndarray) -> GradientBoostingRegressor:
        """Train GBDT model on full dataset."""
        print("\n" + "=" * 80)
        print("Training GBDT Model on Full Dataset")
        print("=" * 80)

        # Normalize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train model
        model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.05,
            min_samples_split=10,
            min_samples_leaf=4,
            subsample=0.8,
            random_state=self.random_seed,
            verbose=0,
        )

        print("Training model...")
        model.fit(X_scaled, y)
        print("✓ Model trained")

        # Make predictions
        predictions = model.predict(X_scaled)

        # Calculate metrics
        r2 = r2_score(y, predictions)
        mae = mean_absolute_error(y, predictions)
        rmse = np.sqrt(mean_squared_error(y, predictions))

        print(f"\nFull Dataset Performance:")
        print(f"  R² = {r2:.4f}")
        print(f"  MAE = {mae * 1000:.1f} m")
        print(f"  RMSE = {rmse * 1000:.1f} m")

        return model, scaler, predictions

    def identify_worst_samples(
        self,
        errors: np.ndarray,
        metadata: pd.DataFrame,
        threshold_km: float = 0.2,
        n_samples: int = 50,
    ) -> Dict:
        """Identify worst-performing samples."""
        print("\n" + "=" * 80)
        print("Identifying Worst Samples")
        print("=" * 80)

        abs_errors = np.abs(errors)
        worst_indices = np.where(abs_errors > threshold_km)[0]

        print(f"Threshold: {threshold_km * 1000:.0f} m")
        print(f"Samples exceeding threshold: {len(worst_indices)}")
        print(f"Percentage: {len(worst_indices) / len(errors) * 100:.2f}%")

        # Get top N worst
        top_worst_indices = np.argsort(abs_errors)[-n_samples:][::-1]

        worst_samples = {
            "threshold_km": threshold_km,
            "n_flagged": int(len(worst_indices)),
            "percentage_flagged": float(len(worst_indices) / len(errors) * 100),
            "n_samples": n_samples,
            "sample_ids": metadata.iloc[top_worst_indices]["sample_id"].tolist(),
            "flight_ids": metadata.iloc[top_worst_indices]["flight_id"].tolist(),
            "errors_km": abs_errors[top_worst_indices].tolist(),
            "mean_error_km": float(abs_errors[worst_indices].mean()),
            "max_error_km": float(abs_errors.max()),
            "min_error_km": float(abs_errors.min()),
        }

        print(f"\nTop {n_samples} Worst Samples:")
        print(f"  Mean error: {worst_samples['mean_error_km'] * 1000:.1f} m")
        print(f"  Max error: {worst_samples['max_error_km'] * 1000:.1f} m")

        return worst_samples

    def correlation_analysis(self, errors: np.ndarray, metadata: pd.DataFrame) -> Dict:
        """Analyze correlation between errors and input features."""
        print("\n" + "=" * 80)
        print("Correlation Analysis")
        print("=" * 80)

        abs_errors = np.abs(errors)
        correlation_results = {}

        # Features to analyze
        feature_columns = ["sza", "saa", "altitude", "blh", "lcl", "t2m", "d2m"]

        for feature_name in feature_columns:
            if feature_name not in metadata.columns:
                continue

            feature_values = metadata[feature_name].values

            # Remove NaN values
            valid_mask = ~np.isnan(feature_values) & ~np.isnan(abs_errors)
            if valid_mask.sum() < 10:
                continue

            # Calculate correlation
            correlation, p_value = stats.pearsonr(
                feature_values[valid_mask], abs_errors[valid_mask]
            )

            correlation_results[f"error_vs_{feature_name}"] = {
                "correlation": float(correlation),
                "p_value": float(p_value),
                "significant": bool(p_value < 0.05),
            }

            print(f"  {feature_name}:")
            print(f"    Correlation: {correlation:.4f}")
            print(f"    P-value: {p_value:.4e}")
            print(f"    Significant: {'Yes' if p_value < 0.05 else 'No'}")

        return correlation_results

    def per_flight_error_analysis(
        self, errors: np.ndarray, metadata: pd.DataFrame, flight_mapping: Dict
    ) -> Dict:
        """Analyze errors per flight."""
        print("\n" + "=" * 80)
        print("Per-Flight Error Analysis")
        print("=" * 80)

        abs_errors = np.abs(errors)
        per_flight_results = {}

        # Reverse flight mapping (name -> id)
        flight_name_to_id = {v: k for k, v in flight_mapping.items()}

        for flight_name in ["F1", "F2", "F4"]:
            if flight_name not in flight_name_to_id:
                continue

            flight_id = flight_name_to_id[flight_name]
            flight_mask = metadata["flight_id"] == flight_id

            if flight_mask.sum() == 0:
                continue

            flight_errors = abs_errors[flight_mask]

            per_flight_results[flight_name] = {
                "mean_error_km": float(flight_errors.mean()),
                "std_error_km": float(flight_errors.std()),
                "median_error_km": float(np.median(flight_errors)),
                "n_samples": int(flight_mask.sum()),
            }

            print(f"\n{flight_name}:")
            print(f"  Samples: {per_flight_results[flight_name]['n_samples']}")
            print(
                f"  Mean error: {per_flight_results[flight_name]['mean_error_km'] * 1000:.1f} m"
            )
            print(
                f"  Std error: {per_flight_results[flight_name]['std_error_km'] * 1000:.1f} m"
            )

        return per_flight_results

    def statistical_significance_tests(
        self, errors: np.ndarray, metadata: pd.DataFrame, flight_mapping: Dict
    ) -> Dict:
        """Perform statistical significance tests across flights."""
        print("\n" + "=" * 80)
        print("Statistical Significance Tests")
        print("=" * 80)

        abs_errors = np.abs(errors)

        # Reverse flight mapping
        flight_name_to_id = {v: k for k, v in flight_mapping.items()}

        # Group errors by flight
        flight_error_groups = []
        flight_names = []

        for flight_name in ["F1", "F2", "F4"]:
            if flight_name not in flight_name_to_id:
                continue

            flight_id = flight_name_to_id[flight_name]
            flight_mask = metadata["flight_id"] == flight_id

            if flight_mask.sum() > 0:
                flight_error_groups.append(abs_errors[flight_mask])
                flight_names.append(flight_name)

        # ANOVA test
        if len(flight_error_groups) >= 2:
            f_statistic, p_value = stats.f_oneway(*flight_error_groups)

            results = {
                "anova_across_flights": {
                    "f_statistic": float(f_statistic),
                    "p_value": float(p_value),
                },
                "conclusion": (
                    "Significant differences between flights"
                    if p_value < 0.05
                    else "No significant differences between flights"
                ),
                "flights_tested": flight_names,
            }

            print(f"ANOVA Test:")
            print(f"  F-statistic: {f_statistic:.4f}")
            print(f"  P-value: {p_value:.4e}")
            print(f"  Conclusion: {results['conclusion']}")
        else:
            results = {
                "anova_across_flights": {"f_statistic": 0.0, "p_value": 1.0},
                "conclusion": "Insufficient flights for ANOVA",
                "flights_tested": flight_names,
            }

        return results

    def create_visualizations(
        self,
        errors: np.ndarray,
        predictions: np.ndarray,
        targets: np.ndarray,
        metadata: pd.DataFrame,
        flight_mapping: Dict,
    ):
        """Create comprehensive error analysis visualizations."""
        print("\n" + "=" * 80)
        print("Creating Visualizations")
        print("=" * 80)

        abs_errors = np.abs(errors)

        # 1. Error distribution
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Histogram
        axes[0].hist(errors * 1000, bins=50, edgecolor="black", alpha=0.7)
        axes[0].axvline(0, color="red", linestyle="--", linewidth=2, label="Zero Error")
        axes[0].set_xlabel("Prediction Error (m)")
        axes[0].set_ylabel("Frequency")
        axes[0].set_title("Error Distribution")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Box plot
        axes[1].boxplot(abs_errors * 1000, vert=True)
        axes[1].set_ylabel("Absolute Error (m)")
        axes[1].set_title("Absolute Error Box Plot")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            self.figures_dir / "error_distribution.png", dpi=300, bbox_inches="tight"
        )
        plt.savefig(self.figures_dir / "error_distribution.pdf", bbox_inches="tight")
        plt.close()
        print("✓ Saved error_distribution.png/pdf")

        # 2. Error vs. predictions
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(
            predictions * 1000,
            errors * 1000,
            c=abs_errors * 1000,
            cmap="hot",
            alpha=0.6,
            s=20,
        )
        ax.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.5)
        ax.set_xlabel("Predicted CBH (m)")
        ax.set_ylabel("Prediction Error (m)")
        ax.set_title("Error vs. Predicted CBH")
        plt.colorbar(scatter, ax=ax, label="Absolute Error (m)")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            self.figures_dir / "error_vs_predictions.png", dpi=300, bbox_inches="tight"
        )
        plt.savefig(self.figures_dir / "error_vs_predictions.pdf", bbox_inches="tight")
        plt.close()
        print("✓ Saved error_vs_predictions.png/pdf")

        # 3. Per-flight error analysis
        flight_name_to_id = {v: k for k, v in flight_mapping.items()}
        flight_errors = {}

        for flight_name in ["F1", "F2", "F4"]:
            if flight_name in flight_name_to_id:
                flight_id = flight_name_to_id[flight_name]
                flight_mask = metadata["flight_id"] == flight_id
                if flight_mask.sum() > 0:
                    flight_errors[flight_name] = abs_errors[flight_mask] * 1000

        if flight_errors:
            fig, ax = plt.subplots(figsize=(8, 6))
            positions = list(range(len(flight_errors)))
            ax.boxplot(
                flight_errors.values(), labels=flight_errors.keys(), positions=positions
            )
            ax.set_xlabel("Flight")
            ax.set_ylabel("Absolute Error (m)")
            ax.set_title("Error Distribution by Flight")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(
                self.figures_dir / "error_by_flight.png", dpi=300, bbox_inches="tight"
            )
            plt.savefig(self.figures_dir / "error_by_flight.pdf", bbox_inches="tight")
            plt.close()
            print("✓ Saved error_by_flight.png/pdf")

        # 4. Error correlation heatmap
        correlation_features = ["sza", "saa", "altitude", "blh", "lcl", "t2m", "d2m"]
        available_features = [f for f in correlation_features if f in metadata.columns]

        if available_features:
            corr_data = metadata[available_features].copy()
            corr_data["abs_error"] = abs_errors

            # Remove NaN values
            corr_data = corr_data.dropna()

            if len(corr_data) > 10:
                corr_matrix = corr_data.corr()

                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(
                    corr_matrix,
                    annot=True,
                    fmt=".3f",
                    cmap="coolwarm",
                    center=0,
                    square=True,
                    ax=ax,
                )
                ax.set_title("Feature Correlation Matrix (including Absolute Error)")
                plt.tight_layout()
                plt.savefig(
                    self.figures_dir / "correlation_heatmap.png",
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.savefig(
                    self.figures_dir / "correlation_heatmap.pdf", bbox_inches="tight"
                )
                plt.close()
                print("✓ Saved correlation_heatmap.png/pdf")

        # 5. Error vs. specific features
        if "sza" in metadata.columns:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(metadata["sza"], abs_errors * 1000, alpha=0.5, s=20)
            ax.set_xlabel("Solar Zenith Angle (degrees)")
            ax.set_ylabel("Absolute Error (m)")
            ax.set_title("Error vs. Solar Zenith Angle")
            ax.grid(True, alpha=0.3)

            # Add trend line
            z = np.polyfit(
                metadata["sza"].dropna(), abs_errors[~metadata["sza"].isna()] * 1000, 1
            )
            p = np.poly1d(z)
            x_trend = np.linspace(metadata["sza"].min(), metadata["sza"].max(), 100)
            ax.plot(x_trend, p(x_trend), "r--", linewidth=2, label="Trend")
            ax.legend()

            plt.tight_layout()
            plt.savefig(
                self.figures_dir / "error_vs_sza.png", dpi=300, bbox_inches="tight"
            )
            plt.savefig(self.figures_dir / "error_vs_sza.pdf", bbox_inches="tight")
            plt.close()
            print("✓ Saved error_vs_sza.png/pdf")

        print(f"\n✓ All visualizations saved to: {self.figures_dir}")

    def generate_systematic_bias_report(
        self,
        worst_samples: Dict,
        correlation_results: Dict,
        per_flight_error: Dict,
        statistical_tests: Dict,
    ):
        """Generate a comprehensive markdown report on systematic biases."""
        print("\n" + "=" * 80)
        print("Generating Systematic Bias Report")
        print("=" * 80)

        report_lines = [
            "# Sprint 6 - Task 1.3: Comprehensive Error Analysis Report",
            "",
            f"**Generated**: {datetime.now().isoformat()}",
            "",
            "---",
            "",
            "## 1. Worst-Performing Samples",
            "",
            f"**Threshold**: {worst_samples['threshold_km'] * 1000:.0f} m",
            f"**Samples exceeding threshold**: {worst_samples['n_flagged']} ({worst_samples['percentage_flagged']:.2f}%)",
            f"**Mean error (flagged samples)**: {worst_samples['mean_error_km'] * 1000:.1f} m",
            f"**Maximum error**: {worst_samples['max_error_km'] * 1000:.1f} m",
            "",
            "### Top Worst Samples",
            "",
            "| Sample ID | Flight ID | Error (m) |",
            "|-----------|-----------|-----------|",
        ]

        for i in range(min(10, len(worst_samples["sample_ids"]))):
            report_lines.append(
                f"| {worst_samples['sample_ids'][i]} | "
                f"{worst_samples['flight_ids'][i]} | "
                f"{worst_samples['errors_km'][i] * 1000:.1f} |"
            )

        report_lines.extend(
            [
                "",
                "---",
                "",
                "## 2. Correlation Analysis",
                "",
                "Analysis of error correlations with input features:",
                "",
                "| Feature | Correlation | P-value | Significant |",
                "|---------|-------------|---------|-------------|",
            ]
        )

        for feature_name, stats in correlation_results.items():
            feature_display = feature_name.replace("error_vs_", "")
            report_lines.append(
                f"| {feature_display} | {stats['correlation']:.4f} | "
                f"{stats['p_value']:.4e} | {'Yes' if stats['significant'] else 'No'} |"
            )

        report_lines.extend(
            [
                "",
                "### Interpretation",
                "",
            ]
        )

        # Add interpretation based on correlations
        significant_correlations = [
            (name, stats)
            for name, stats in correlation_results.items()
            if stats["significant"]
        ]

        if significant_correlations:
            report_lines.append("**Significant correlations found:**")
            report_lines.append("")
            for name, stats in significant_correlations:
                feature_name = name.replace("error_vs_", "")
                direction = "positive" if stats["correlation"] > 0 else "negative"
                report_lines.append(
                    f"- **{feature_name}**: {direction} correlation "
                    f"(r={stats['correlation']:.4f}, p={stats['p_value']:.4e})"
                )
        else:
            report_lines.append("No significant correlations found (p < 0.05).")

        report_lines.extend(
            [
                "",
                "---",
                "",
                "## 3. Per-Flight Error Analysis",
                "",
                "| Flight | N Samples | Mean Error (m) | Std Error (m) | Median Error (m) |",
                "|--------|-----------|----------------|---------------|------------------|",
            ]
        )

        for flight_name, stats in per_flight_error.items():
            report_lines.append(
                f"| {flight_name} | {stats['n_samples']} | "
                f"{stats['mean_error_km'] * 1000:.1f} | "
                f"{stats['std_error_km'] * 1000:.1f} | "
                f"{stats['median_error_km'] * 1000:.1f} |"
            )

        report_lines.extend(
            [
                "",
                "---",
                "",
                "## 4. Statistical Significance Tests",
                "",
                "### ANOVA Across Flights",
                "",
                f"**F-statistic**: {statistical_tests['anova_across_flights']['f_statistic']:.4f}",
                f"**P-value**: {statistical_tests['anova_across_flights']['p_value']:.4e}",
                f"**Conclusion**: {statistical_tests['conclusion']}",
                "",
                f"**Flights tested**: {', '.join(statistical_tests['flights_tested'])}",
                "",
                "---",
                "",
                "## 5. Summary and Recommendations",
                "",
            ]
        )

        # Add summary based on findings
        if statistical_tests["anova_across_flights"]["p_value"] < 0.05:
            report_lines.append(
                "- **Flight-specific errors**: Significant differences detected between flights. "
                "Consider flight-specific calibration or domain adaptation."
            )
        else:
            report_lines.append(
                "- **Flight consistency**: No significant differences between flights. "
                "Model generalizes well across flights."
            )

        if significant_correlations:
            report_lines.append(
                "- **Feature correlations**: Errors are correlated with specific input features. "
                "Consider feature engineering or specialized handling of these conditions."
            )

        if worst_samples["percentage_flagged"] > 10:
            report_lines.append(
                f"- **High error rate**: {worst_samples['percentage_flagged']:.1f}% of samples "
                f"exceed {worst_samples['threshold_km'] * 1000:.0f} m error threshold. "
                "Consider ensemble methods or uncertainty quantification."
            )

        report_lines.extend(
            [
                "",
                "---",
                "",
                f"**Report generated**: {datetime.now().isoformat()}",
                "",
            ]
        )

        # Save report
        report_path = self.reports_dir / "error_analysis_systematic_bias_report.md"
        with open(report_path, "w") as f:
            f.write("\n".join(report_lines))

        print(f"✓ Systematic bias report saved: {report_path}")

    def save_error_analysis_report(
        self,
        worst_samples: Dict,
        correlation_results: Dict,
        per_flight_error: Dict,
        statistical_tests: Dict,
    ):
        """Save error analysis report in JSON format."""
        report = {
            "worst_samples": worst_samples,
            "correlation_analysis": correlation_results,
            "per_flight_error": per_flight_error,
            "statistical_tests": statistical_tests,
            "timestamp": datetime.now().isoformat(),
        }

        report_path = self.reports_dir / "error_analysis_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"\n{'=' * 80}")
        print("Error Analysis Report Summary")
        print(f"{'=' * 80}")
        print(f"Worst Samples: {worst_samples['n_flagged']} exceed threshold")
        print(f"Correlation Analysis: {len(correlation_results)} features analyzed")
        print(f"Per-Flight Analysis: {len(per_flight_error)} flights")
        print(f"Statistical Tests: {statistical_tests['conclusion']}")
        print(f"\n✓ Report saved: {report_path}")

        return report


def main():
    """Main execution function."""

    print("\n" + "=" * 80)
    print("Starting Error Analysis")
    print("=" * 80)

    # Initialize analyzer
    analyzer = ComprehensiveErrorAnalyzer(random_seed=42)

    # Load data
    X, y, feature_names, metadata, flight_mapping = analyzer.load_data(
        INTEGRATED_FEATURES
    )

    # Train model
    model, scaler, predictions = analyzer.train_model(X, y)

    # Calculate errors
    errors = predictions - y
    abs_errors = np.abs(errors)

    print(f"\n{'=' * 80}")
    print("Error Statistics")
    print(f"{'=' * 80}")
    print(f"Mean absolute error: {abs_errors.mean() * 1000:.1f} m")
    print(f"Std absolute error: {abs_errors.std() * 1000:.1f} m")
    print(f"Median absolute error: {np.median(abs_errors) * 1000:.1f} m")
    print(f"Max absolute error: {abs_errors.max() * 1000:.1f} m")

    # Identify worst samples
    worst_samples = analyzer.identify_worst_samples(
        errors, metadata, threshold_km=0.2, n_samples=50
    )

    # Correlation analysis
    correlation_results = analyzer.correlation_analysis(errors, metadata)

    # Per-flight error analysis
    per_flight_error = analyzer.per_flight_error_analysis(
        errors, metadata, flight_mapping
    )

    # Statistical significance tests
    statistical_tests = analyzer.statistical_significance_tests(
        errors, metadata, flight_mapping
    )

    # Create visualizations
    analyzer.create_visualizations(errors, predictions, y, metadata, flight_mapping)

    # Generate systematic bias report (Markdown)
    analyzer.generate_systematic_bias_report(
        worst_samples, correlation_results, per_flight_error, statistical_tests
    )

    # Save JSON report
    analyzer.save_error_analysis_report(
        worst_samples, correlation_results, per_flight_error, statistical_tests
    )

    print(f"\n{'=' * 80}")
    print("✓ Task 1.3 Complete: Comprehensive Error Analysis")
    print(f"{'=' * 80}")
    print(f"All outputs saved to:")
    print(f"  Figures: {FIGURES_DIR}")
    print(f"  Reports: {REPORTS_DIR}")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
