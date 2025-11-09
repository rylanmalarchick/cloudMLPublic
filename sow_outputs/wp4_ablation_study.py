#!/usr/bin/env python3
"""
Work Package 4: Ablation Study

Compares different model variants to quantify the contribution of:
1. Image features (image_only)
2. Physical features (concat vs image_only)
3. Fusion strategy (concat vs attention)

This is a required deliverable (7.4d) for Sprint 4.

Author: Autonomous Agent
Date: 2025
SOW: SOW-AGENT-CBH-WP-001 Section 7.4d
"""

import sys
from pathlib import Path
import json
import numpy as np
from typing import Dict, List
from datetime import datetime


class AblationStudyAnalyzer:
    """
    Analyzes ablation study results from WP-3 and WP-4 models.

    Compares:
    - Physical-only baseline (WP-3)
    - Image-only model (WP-4)
    - Concatenation fusion (WP-4)
    - Attention fusion (WP-4)
    """

    def __init__(
        self,
        wp3_report_path: str,
        wp4_image_only_path: str,
        wp4_concat_path: str,
        wp4_attention_path: str,
        output_dir: str = "sow_outputs/wp4_ablation",
        verbose: bool = True,
    ):
        """
        Initialize the analyzer.

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
                reports["physical_only"] = json.load(f)
            if self.verbose:
                print(f"✓ Loaded WP-3 (physical-only): {self.wp3_report_path}")
        else:
            if self.verbose:
                print(f"✗ WP-3 report not found: {self.wp3_report_path}")

        # Load WP-4 variants
        if self.wp4_image_only_path.exists():
            with open(self.wp4_image_only_path, "r") as f:
                reports["image_only"] = json.load(f)
            if self.verbose:
                print(f"✓ Loaded WP-4 (image-only): {self.wp4_image_only_path}")
        else:
            if self.verbose:
                print(f"✗ WP-4 image-only report not found: {self.wp4_image_only_path}")

        if self.wp4_concat_path.exists():
            with open(self.wp4_concat_path, "r") as f:
                reports["concat"] = json.load(f)
            if self.verbose:
                print(f"✓ Loaded WP-4 (concat): {self.wp4_concat_path}")
        else:
            if self.verbose:
                print(f"✗ WP-4 concat report not found: {self.wp4_concat_path}")

        if self.wp4_attention_path.exists():
            with open(self.wp4_attention_path, "r") as f:
                reports["attention"] = json.load(f)
            if self.verbose:
                print(f"✓ Loaded WP-4 (attention): {self.wp4_attention_path}")
        else:
            if self.verbose:
                print(f"✗ WP-4 attention report not found: {self.wp4_attention_path}")

        return reports

    def extract_metrics(self, reports: Dict) -> Dict:
        """Extract key metrics from each report."""
        metrics = {}

        for model_name, report in reports.items():
            if model_name == "physical_only":
                # WP-3 format
                results = report.get("results", {})
                metrics[model_name] = {
                    "mean_r2": results.get("mean_r2", None),
                    "std_r2": results.get("std_r2", None),
                    "mean_mae_km": results.get("mean_mae_km", None),
                    "std_mae_km": results.get("std_mae_km", None),
                    "mean_rmse_km": results.get("mean_rmse_km", None),
                    "std_rmse_km": results.get("std_rmse_km", None),
                    "description": report.get("description", ""),
                }
            else:
                # WP-4 format
                agg = report.get("aggregate_metrics", {})
                metrics[model_name] = {
                    "mean_r2": agg.get("mean_r2", None),
                    "std_r2": agg.get("std_r2", None),
                    "mean_mae_km": agg.get("mean_mae_km", None),
                    "std_mae_km": agg.get("std_mae_km", None),
                    "mean_rmse_km": agg.get("mean_rmse_km", None),
                    "std_rmse_km": agg.get("std_rmse_km", None),
                    "description": report.get("description", ""),
                }

        return metrics

    def compute_ablation_analysis(self, metrics: Dict) -> Dict:
        """
        Compute ablation analysis.

        Comparisons:
        1. Physical vs Image: Which modality is stronger?
        2. Image-only vs Concat: Does adding physical features help?
        3. Concat vs Attention: Does attention fusion improve over simple concat?
        4. Best hybrid vs Physical-only: Overall hybrid benefit
        """
        analysis = {}

        # Get baseline metrics
        physical_r2 = metrics.get("physical_only", {}).get("mean_r2", None)
        image_r2 = metrics.get("image_only", {}).get("mean_r2", None)
        concat_r2 = metrics.get("concat", {}).get("mean_r2", None)
        attention_r2 = metrics.get("attention", {}).get("mean_r2", None)

        physical_mae = metrics.get("physical_only", {}).get("mean_mae_km", None)
        image_mae = metrics.get("image_only", {}).get("mean_mae_km", None)
        concat_mae = metrics.get("concat", {}).get("mean_mae_km", None)
        attention_mae = metrics.get("attention", {}).get("mean_mae_km", None)

        # Comparison 1: Physical vs Image
        if physical_r2 is not None and image_r2 is not None:
            analysis["physical_vs_image"] = {
                "physical_r2": physical_r2,
                "image_r2": image_r2,
                "r2_difference": physical_r2 - image_r2,
                "winner": "physical" if physical_r2 > image_r2 else "image",
                "interpretation": (
                    "Physical features are stronger"
                    if physical_r2 > image_r2
                    else "Image features are stronger"
                ),
            }

        # Comparison 2: Image-only vs Concat (physical feature contribution)
        if image_r2 is not None and concat_r2 is not None:
            analysis["image_vs_concat"] = {
                "image_only_r2": image_r2,
                "concat_r2": concat_r2,
                "r2_gain": concat_r2 - image_r2,
                "interpretation": (
                    "Adding physical features improves performance"
                    if concat_r2 > image_r2
                    else "Physical features do not help (or hurt)"
                ),
            }

        # Comparison 3: Concat vs Attention (fusion strategy)
        if concat_r2 is not None and attention_r2 is not None:
            analysis["concat_vs_attention"] = {
                "concat_r2": concat_r2,
                "attention_r2": attention_r2,
                "r2_gain": attention_r2 - concat_r2,
                "interpretation": (
                    "Attention fusion is better than simple concatenation"
                    if attention_r2 > concat_r2
                    else "Simple concatenation is sufficient"
                ),
            }

        # Comparison 4: Best hybrid vs Physical-only
        if physical_r2 is not None and attention_r2 is not None:
            analysis["hybrid_vs_physical"] = {
                "physical_only_r2": physical_r2,
                "best_hybrid_r2": attention_r2,
                "r2_difference": attention_r2 - physical_r2,
                "winner": "hybrid" if attention_r2 > physical_r2 else "physical",
                "interpretation": (
                    "Hybrid approach improves over physical-only"
                    if attention_r2 > physical_r2
                    else "Physical-only baseline is stronger (surprising!)"
                ),
            }

        # Overall ranking
        model_r2 = {}
        if physical_r2 is not None:
            model_r2["physical_only"] = physical_r2
        if image_r2 is not None:
            model_r2["image_only"] = image_r2
        if concat_r2 is not None:
            model_r2["concat"] = concat_r2
        if attention_r2 is not None:
            model_r2["attention"] = attention_r2

        ranked = sorted(model_r2.items(), key=lambda x: x[1], reverse=True)
        analysis["ranking"] = {
            "by_r2": [{"model": name, "r2": r2} for name, r2 in ranked],
            "best_model": ranked[0][0] if ranked else None,
            "best_r2": ranked[0][1] if ranked else None,
        }

        return analysis

    def generate_report(self, metrics: Dict, analysis: Dict) -> Dict:
        """Generate comprehensive ablation study report."""
        report = {
            "title": "WP-4 Ablation Study: Model Variant Comparison",
            "timestamp": datetime.now().isoformat(),
            "models_evaluated": list(metrics.keys()),
            "model_metrics": metrics,
            "ablation_analysis": analysis,
            "key_findings": self._generate_key_findings(metrics, analysis),
        }

        return report

    def _generate_key_findings(self, metrics: Dict, analysis: Dict) -> List[str]:
        """Generate key findings from ablation analysis."""
        findings = []

        # Finding 1: Model ranking
        ranking = analysis.get("ranking", {})
        best_model = ranking.get("best_model")
        if best_model:
            best_r2 = ranking.get("best_r2", 0)
            findings.append(f"Best model: {best_model} (R² = {best_r2:.4f})")

        # Finding 2: Physical vs Image
        phys_vs_img = analysis.get("physical_vs_image", {})
        if phys_vs_img:
            winner = phys_vs_img.get("winner")
            diff = phys_vs_img.get("r2_difference", 0)
            findings.append(
                f"Physical features outperform image features by ΔR² = {abs(diff):.4f}"
                if winner == "physical"
                else f"Image features outperform physical features by ΔR² = {abs(diff):.4f}"
            )

        # Finding 3: Fusion benefit
        img_vs_concat = analysis.get("image_vs_concat", {})
        if img_vs_concat:
            gain = img_vs_concat.get("r2_gain", 0)
            if gain > 0:
                findings.append(
                    f"Adding physical features to image model improves R² by {gain:.4f}"
                )
            else:
                findings.append(
                    f"Adding physical features to image model degrades R² by {abs(gain):.4f}"
                )

        # Finding 4: Attention benefit
        concat_vs_attn = analysis.get("concat_vs_attention", {})
        if concat_vs_attn:
            gain = concat_vs_attn.get("r2_gain", 0)
            if gain > 0:
                findings.append(
                    f"Attention fusion improves over concatenation by ΔR² = {gain:.4f}"
                )
            else:
                findings.append(
                    f"Attention fusion does not improve over concatenation (ΔR² = {gain:.4f})"
                )

        # Finding 5: Hybrid vs Physical
        hybrid_vs_phys = analysis.get("hybrid_vs_physical", {})
        if hybrid_vs_phys:
            diff = hybrid_vs_phys.get("r2_difference", 0)
            if diff > 0:
                findings.append(
                    f"Best hybrid model improves over physical-only by ΔR² = {diff:.4f}"
                )
            else:
                findings.append(
                    f"Physical-only baseline outperforms hybrid models by ΔR² = {abs(diff):.4f} "
                    "(suggests image features may be noisy or poorly integrated)"
                )

        return findings

    def save_report(self, report: Dict, filename: str = "WP4_Ablation_Study.json"):
        """Save report to JSON."""
        output_path = self.output_dir / filename
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        if self.verbose:
            print(f"\n✓ Report saved to: {output_path}")

        return output_path

    def print_summary(self, report: Dict):
        """Print formatted summary."""
        print("\n" + "=" * 80)
        print("WP-4 ABLATION STUDY RESULTS")
        print("=" * 80)

        print("\nModels Evaluated:")
        for i, model in enumerate(report["models_evaluated"], 1):
            print(f"  {i}. {model}")

        print("\n" + "-" * 80)
        print("MODEL PERFORMANCE COMPARISON")
        print("-" * 80)

        metrics = report["model_metrics"]
        print(
            f"\n{'Model':<20} {'R²':<15} {'MAE (km)':<15} {'RMSE (km)':<15} {'Description':<40}"
        )
        print("-" * 105)

        for model_name, m in metrics.items():
            r2_str = (
                f"{m['mean_r2']:.4f} ± {m['std_r2']:.4f}"
                if m["mean_r2"] is not None
                else "N/A"
            )
            mae_str = (
                f"{m['mean_mae_km']:.4f} ± {m['std_mae_km']:.4f}"
                if m["mean_mae_km"] is not None
                else "N/A"
            )
            rmse_str = (
                f"{m['mean_rmse_km']:.4f} ± {m['std_rmse_km']:.4f}"
                if m["mean_rmse_km"] is not None
                else "N/A"
            )
            desc = m.get("description", "")[:38]

            print(
                f"{model_name:<20} {r2_str:<15} {mae_str:<15} {rmse_str:<15} {desc:<40}"
            )

        print("\n" + "-" * 80)
        print("ABLATION ANALYSIS")
        print("-" * 80)

        analysis = report["ablation_analysis"]

        # Ranking
        if "ranking" in analysis:
            print("\nModel Ranking (by R²):")
            for i, entry in enumerate(analysis["ranking"]["by_r2"], 1):
                print(f"  {i}. {entry['model']:<20} R² = {entry['r2']:.4f}")

        # Comparisons
        if "physical_vs_image" in analysis:
            print("\n1. Physical vs Image Features:")
            pvi = analysis["physical_vs_image"]
            print(f"   Physical R²: {pvi['physical_r2']:.4f}")
            print(f"   Image R²:    {pvi['image_r2']:.4f}")
            print(f"   Difference:  {pvi['r2_difference']:.4f}")
            print(f"   → {pvi['interpretation']}")

        if "image_vs_concat" in analysis:
            print("\n2. Adding Physical Features to Image Model:")
            ivc = analysis["image_vs_concat"]
            print(f"   Image-only R²: {ivc['image_only_r2']:.4f}")
            print(f"   Concat R²:     {ivc['concat_r2']:.4f}")
            print(f"   R² Gain:       {ivc['r2_gain']:.4f}")
            print(f"   → {ivc['interpretation']}")

        if "concat_vs_attention" in analysis:
            print("\n3. Fusion Strategy (Concat vs Attention):")
            cva = analysis["concat_vs_attention"]
            print(f"   Concat R²:    {cva['concat_r2']:.4f}")
            print(f"   Attention R²: {cva['attention_r2']:.4f}")
            print(f"   R² Gain:      {cva['r2_gain']:.4f}")
            print(f"   → {cva['interpretation']}")

        if "hybrid_vs_physical" in analysis:
            print("\n4. Best Hybrid vs Physical-Only:")
            hvp = analysis["hybrid_vs_physical"]
            print(f"   Physical-only R²: {hvp['physical_only_r2']:.4f}")
            print(f"   Best hybrid R²:   {hvp['best_hybrid_r2']:.4f}")
            print(f"   Difference:       {hvp['r2_difference']:.4f}")
            print(f"   → {hvp['interpretation']}")

        print("\n" + "-" * 80)
        print("KEY FINDINGS")
        print("-" * 80)

        for i, finding in enumerate(report["key_findings"], 1):
            print(f"\n{i}. {finding}")

        print("\n" + "=" * 80)

    def run(self):
        """Execute complete ablation study analysis."""
        print("\n" + "=" * 80)
        print("WP-4: ABLATION STUDY")
        print("=" * 80)
        print(f"Output directory: {self.output_dir}")

        # Load reports
        reports = self.load_reports()

        if not reports:
            print("\nERROR: No reports found!")
            return None

        # Extract metrics
        metrics = self.extract_metrics(reports)

        # Compute ablation analysis
        analysis = self.compute_ablation_analysis(metrics)

        # Generate report
        report = self.generate_report(metrics, analysis)

        # Save report
        self.save_report(report)

        # Print summary
        self.print_summary(report)

        return report


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

    # Run analyzer
    analyzer = AblationStudyAnalyzer(
        wp3_report_path=str(wp3_report),
        wp4_image_only_path=str(wp4_image),
        wp4_concat_path=str(wp4_concat),
        wp4_attention_path=str(wp4_attention),
        output_dir="sow_outputs/wp4_ablation",
        verbose=True,
    )

    report = analyzer.run()

    if report is None:
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
