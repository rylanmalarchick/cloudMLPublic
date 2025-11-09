#!/usr/bin/env python3
"""
Work Package 3: Validation Summary

Creates the formal validation summary combining all model results
for SOW deliverable 7.3c.

This summary provides a comprehensive overview of model performance
across all validation experiments.

Author: Autonomous Agent
Date: 2025
SOW: SOW-AGENT-CBH-WP-001 Section 7.3c
"""

import sys
from pathlib import Path
import json
import numpy as np
from typing import Dict, List
from datetime import datetime


class ValidationSummaryGenerator:
    """
    Generates validation summary from all model reports.

    Consolidates:
    - WP-3: Physical baseline (K-Fold CV)
    - WP-4: Image-only model (K-Fold CV)
    - WP-4: Concat fusion (K-Fold CV)
    - WP-4: Attention fusion (K-Fold CV)
    """

    def __init__(
        self,
        wp3_report_path: str,
        wp4_image_only_path: str,
        wp4_concat_path: str,
        wp4_attention_path: str,
        output_dir: str = "sow_outputs/validation_summary",
        verbose: bool = True,
    ):
        """
        Initialize the generator.

        Args:
            wp3_report_path: Path to WP3 K-Fold report
            wp4_image_only_path: Path to WP4 image-only report
            wp4_concat_path: Path to WP4 concat report
            wp4_attention_path: Path to WP4 attention report
            output_dir: Output directory
            verbose: Verbose logging
        """
        self.wp3_report_path = Path(wp3_report_path)
        self.wp4_image_only_path = Path(wp4_image_only_path)
        self.wp4_concat_path = Path(wp4_concat_path)
        self.wp4_attention_path = Path(wp4_attention_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose

    def load_reports(self) -> Dict:
        """Load all model reports."""
        if self.verbose:
            print("\n" + "=" * 80)
            print("Loading Model Reports")
            print("=" * 80)

        reports = {}

        # Load WP-3 (physical baseline)
        if self.wp3_report_path.exists():
            with open(self.wp3_report_path, "r") as f:
                reports["wp3_physical"] = json.load(f)
            if self.verbose:
                print(f"✓ Loaded WP-3: {self.wp3_report_path}")
        else:
            if self.verbose:
                print(f"✗ WP-3 report not found: {self.wp3_report_path}")

        # Load WP-4 variants
        if self.wp4_image_only_path.exists():
            with open(self.wp4_image_only_path, "r") as f:
                reports["wp4_image_only"] = json.load(f)
            if self.verbose:
                print(f"✓ Loaded WP-4 image-only: {self.wp4_image_only_path}")

        if self.wp4_concat_path.exists():
            with open(self.wp4_concat_path, "r") as f:
                reports["wp4_concat"] = json.load(f)
            if self.verbose:
                print(f"✓ Loaded WP-4 concat: {self.wp4_concat_path}")

        if self.wp4_attention_path.exists():
            with open(self.wp4_attention_path, "r") as f:
                reports["wp4_attention"] = json.load(f)
            if self.verbose:
                print(f"✓ Loaded WP-4 attention: {self.wp4_attention_path}")

        return reports

    def extract_summary_metrics(self, reports: Dict) -> Dict:
        """Extract summary metrics from each report."""
        summary = {}

        for model_key, report in reports.items():
            if model_key == "wp3_physical":
                # WP-3 format
                results = report.get("results", {})
                summary[model_key] = {
                    "model_name": report.get("model_name", "physical_baseline"),
                    "description": report.get("description", ""),
                    "validation_protocol": report.get("validation", ""),
                    "n_folds": results.get("n_folds", 5),
                    "aggregate_metrics": {
                        "mean_r2": results.get("mean_r2"),
                        "std_r2": results.get("std_r2"),
                        "mean_mae_km": results.get("mean_mae_km"),
                        "std_mae_km": results.get("std_mae_km"),
                        "mean_rmse_km": results.get("mean_rmse_km"),
                        "std_rmse_km": results.get("std_rmse_km"),
                    },
                    "per_fold_results": results.get("per_fold", []),
                }
            else:
                # WP-4 format
                summary[model_key] = {
                    "model_name": report.get("model_variant", ""),
                    "description": report.get("description", ""),
                    "validation_protocol": report.get("validation_protocol", ""),
                    "n_folds": report.get("aggregate_metrics", {}).get("n_folds", 5),
                    "aggregate_metrics": report.get("aggregate_metrics", {}),
                    "per_fold_results": report.get("fold_results", []),
                }

        return summary

    def generate_validation_summary(self, summary_metrics: Dict) -> Dict:
        """Generate comprehensive validation summary."""

        # Extract key metrics for comparison table
        comparison_table = []
        for model_key, data in summary_metrics.items():
            metrics = data["aggregate_metrics"]
            comparison_table.append(
                {
                    "model": data["model_name"],
                    "description": data["description"],
                    "mean_r2": metrics.get("mean_r2"),
                    "std_r2": metrics.get("std_r2"),
                    "mean_mae_km": metrics.get("mean_mae_km"),
                    "std_mae_km": metrics.get("std_mae_km"),
                    "mean_rmse_km": metrics.get("mean_rmse_km"),
                    "std_rmse_km": metrics.get("std_rmse_km"),
                }
            )

        # Sort by R² (descending)
        comparison_table.sort(key=lambda x: x["mean_r2"] or -999, reverse=True)

        # Identify best model
        best_model = comparison_table[0] if comparison_table else None

        # Generate summary
        validation_summary = {
            "title": "Cloud Base Height Prediction - Validation Summary",
            "sow_deliverable": "7.3c",
            "timestamp": datetime.now().isoformat(),
            "validation_protocol": "Stratified 5-Fold Cross-Validation",
            "dataset_info": {
                "total_samples": 933,
                "n_flights": 5,
                "cbh_range_km": [0.12, 1.95],
                "cbh_mean_km": 0.83,
                "flight_distribution": {
                    "F0_30Oct24": 501,
                    "F1_10Feb25": 191,
                    "F2_23Oct24": 105,
                    "F3_12Feb25": 92,
                    "F4_18Feb25": 44,
                },
            },
            "models_evaluated": list(summary_metrics.keys()),
            "comparison_table": comparison_table,
            "best_model": {
                "name": best_model["model"] if best_model else None,
                "description": best_model["description"] if best_model else None,
                "r2": best_model["mean_r2"] if best_model else None,
                "mae_km": best_model["mean_mae_km"] if best_model else None,
                "rmse_km": best_model["mean_rmse_km"] if best_model else None,
            },
            "detailed_results": summary_metrics,
            "key_insights": self._generate_insights(comparison_table),
        }

        return validation_summary

    def _generate_insights(self, comparison_table: List[Dict]) -> List[str]:
        """Generate key insights from validation results."""
        insights = []

        if not comparison_table:
            return insights

        # Best model
        best = comparison_table[0]
        insights.append(
            f"Best performing model: {best['model']} with R² = {best['mean_r2']:.4f} ± {best['std_r2']:.4f}"
        )

        # Check if physical baseline is best
        if "physical" in best["model"].lower():
            insights.append(
                "Physical features (geometric + atmospheric) outperform deep learning approaches, "
                "suggesting the 2D CNN may need architectural improvements or better training."
            )

        # Hybrid model performance
        hybrid_models = [
            m
            for m in comparison_table
            if "attention" in m["model"].lower() or "concat" in m["model"].lower()
        ]
        if hybrid_models:
            best_hybrid = hybrid_models[0]
            insights.append(
                f"Best hybrid model: {best_hybrid['model']} with R² = {best_hybrid['mean_r2']:.4f}"
            )

        # Check for attention benefit
        attention_r2 = None
        concat_r2 = None
        for m in comparison_table:
            if "attention" in m["model"].lower():
                attention_r2 = m["mean_r2"]
            if "concat" in m["model"].lower():
                concat_r2 = m["mean_r2"]

        if attention_r2 is not None and concat_r2 is not None:
            if attention_r2 > concat_r2:
                insights.append(
                    f"Attention fusion (R²={attention_r2:.4f}) outperforms simple concatenation (R²={concat_r2:.4f}), "
                    f"validating the learned feature weighting approach."
                )
            else:
                insights.append(
                    f"Simple concatenation (R²={concat_r2:.4f}) performs similarly to attention fusion (R²={attention_r2:.4f}), "
                    f"suggesting simpler fusion may be sufficient."
                )

        # Performance assessment
        if best["mean_r2"] and best["mean_r2"] > 0.5:
            insights.append("Models achieve good predictive performance (R² > 0.5)")
        elif best["mean_r2"] and best["mean_r2"] > 0.3:
            insights.append("Models achieve moderate predictive performance (R² > 0.3)")
        else:
            insights.append(
                "Models show weak predictive performance, suggesting need for improvements"
            )

        # MAE assessment
        if best["mean_mae_km"] and best["mean_mae_km"] < 0.15:
            insights.append(
                f"Excellent MAE performance: {best['mean_mae_km']:.4f} km (~150 meters)"
            )
        elif best["mean_mae_km"] and best["mean_mae_km"] < 0.25:
            insights.append(
                f"Good MAE performance: {best['mean_mae_km']:.4f} km (~{int(best['mean_mae_km'] * 1000)} meters)"
            )

        return insights

    def save_summary(self, summary: Dict, filename: str = "Validation_Summary.json"):
        """Save validation summary to JSON."""
        output_path = self.output_dir / filename
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)

        if self.verbose:
            print(f"\n✓ Validation summary saved to: {output_path}")

        return output_path

    def print_summary(self, summary: Dict):
        """Print formatted validation summary."""
        print("\n" + "=" * 80)
        print("VALIDATION SUMMARY - SOW Deliverable 7.3c")
        print("=" * 80)

        print(f"\nTitle: {summary['title']}")
        print(f"Validation Protocol: {summary['validation_protocol']}")
        print(f"Generated: {summary['timestamp']}")

        print("\n" + "-" * 80)
        print("DATASET INFORMATION")
        print("-" * 80)

        dataset = summary["dataset_info"]
        print(f"  Total samples: {dataset['total_samples']}")
        print(f"  Number of flights: {dataset['n_flights']}")
        print(
            f"  CBH range: [{dataset['cbh_range_km'][0]}, {dataset['cbh_range_km'][1]}] km"
        )
        print(f"  CBH mean: {dataset['cbh_mean_km']} km")
        print("\n  Flight distribution:")
        for flight, count in dataset["flight_distribution"].items():
            print(f"    {flight}: {count} samples")

        print("\n" + "-" * 80)
        print("MODEL COMPARISON TABLE")
        print("-" * 80)

        print(f"\n{'Model':<25} {'R²':<18} {'MAE (km)':<18} {'RMSE (km)':<18}")
        print("-" * 80)

        for model in summary["comparison_table"]:
            r2_str = (
                f"{model['mean_r2']:.4f} ± {model['std_r2']:.4f}"
                if model["mean_r2"] is not None
                else "N/A"
            )
            mae_str = (
                f"{model['mean_mae_km']:.4f} ± {model['std_mae_km']:.4f}"
                if model["mean_mae_km"] is not None
                else "N/A"
            )
            rmse_str = (
                f"{model['mean_rmse_km']:.4f} ± {model['std_rmse_km']:.4f}"
                if model["mean_rmse_km"] is not None
                else "N/A"
            )

            print(f"{model['model']:<25} {r2_str:<18} {mae_str:<18} {rmse_str:<18}")

        print("\n" + "-" * 80)
        print("BEST MODEL")
        print("-" * 80)

        best = summary["best_model"]
        if best["name"]:
            print(f"\nModel: {best['name']}")
            print(f"Description: {best['description']}")
            print(f"Performance:")
            print(f"  R² = {best['r2']:.4f}")
            print(f"  MAE = {best['mae_km']:.4f} km")
            print(f"  RMSE = {best['rmse_km']:.4f} km")

        print("\n" + "-" * 80)
        print("KEY INSIGHTS")
        print("-" * 80)

        for i, insight in enumerate(summary["key_insights"], 1):
            print(f"\n{i}. {insight}")

        print("\n" + "=" * 80)

    def run(self):
        """Execute complete validation summary generation."""
        print("\n" + "=" * 80)
        print("GENERATING VALIDATION SUMMARY")
        print("=" * 80)
        print(f"Output directory: {self.output_dir}")

        # Load reports
        reports = self.load_reports()

        if not reports:
            print("\nERROR: No reports found!")
            return None

        # Extract metrics
        summary_metrics = self.extract_summary_metrics(reports)

        # Generate validation summary
        validation_summary = self.generate_validation_summary(summary_metrics)

        # Save summary
        self.save_summary(validation_summary)

        # Print summary
        self.print_summary(validation_summary)

        return validation_summary


def main():
    """Main entry point."""
    # Paths
    project_root = Path(__file__).parent.parent
    wp3_report = project_root / "sow_outputs" / "wp3_kfold" / "WP3_Report_kfold.json"
    wp4_image = project_root / "sow_outputs" / "wp4_cnn" / "WP4_Report_image_only.json"
    wp4_concat = project_root / "sow_outputs" / "wp4_cnn" / "WP4_Report_concat.json"
    wp4_attention = (
        project_root / "sow_outputs" / "wp4_cnn" / "WP4_Report_attention.json"
    )

    # Run generator
    generator = ValidationSummaryGenerator(
        wp3_report_path=str(wp3_report),
        wp4_image_only_path=str(wp4_image),
        wp4_concat_path=str(wp4_concat),
        wp4_attention_path=str(wp4_attention),
        output_dir="sow_outputs/validation_summary",
        verbose=True,
    )

    summary = generator.run()

    if summary is None:
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
