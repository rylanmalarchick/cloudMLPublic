#!/usr/bin/env python3
"""
Verification script for preprint numerical claims.

This script cross-references all quantitative claims in the academic preprint
against the canonical validation reports in results/cbh/reports/.

Usage:
    python scripts/verify_preprint_claims.py

Returns:
    Exit code 0 if all claims verified
    Exit code 1 if any discrepancies found
"""

import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

# Color codes for terminal output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"


class PrePrintVerifier:
    """Verify preprint claims against validation reports."""

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.reports_dir = repo_root / "results" / "cbh" / "reports"
        self.verification_results: List[Dict[str, Any]] = []

    def load_report(self, filename: str) -> Dict:
        """Load a JSON report file."""
        path = self.reports_dir / filename
        if not path.exists():
            return {}
        with open(path) as f:
            return json.load(f)

    def verify_claim(
        self,
        claim_name: str,
        expected: float,
        actual: float,
        tolerance: float = 0.001,
        source: str = "",
    ) -> bool:
        """Verify a numerical claim within tolerance."""
        matches = abs(expected - actual) <= tolerance
        status = " PASS" if matches else " FAIL"
        color = GREEN if matches else RED

        result = {
            "claim": claim_name,
            "expected": expected,
            "actual": actual,
            "tolerance": tolerance,
            "matches": matches,
            "source": source,
        }
        self.verification_results.append(result)

        print(f"{color}{status}{RESET} {claim_name}")
        print(f"  Expected: {expected:.4f}")
        print(f"  Actual:   {actual:.4f}")
        if source:
            print(f"  Source:   {source}")
        print()

        return matches

    def verify_tabular_cv_metrics(self) -> bool:
        """Verify GBDT tabular cross-validation metrics."""
        print(f"\n{BLUE}=== GBDT Tabular CV Metrics ==={RESET}\n")

        report = self.load_report("validation_report_tabular.json")
        if not report:
            print(f"{RED}ERROR: validation_report_tabular.json not found{RESET}\n")
            return False

        mean_metrics = report.get("mean_metrics", {})
        std_metrics = report.get("std_metrics", {})

        all_pass = True
        all_pass &= self.verify_claim(
            "GBDT Mean R²",
            expected=0.744,
            actual=mean_metrics.get("r2", 0),
            tolerance=0.001,
            source="validation_report_tabular.json:mean_metrics.r2",
        )

        all_pass &= self.verify_claim(
            "GBDT Std R²",
            expected=0.037,
            actual=std_metrics.get("r2", 0),
            tolerance=0.001,
            source="validation_report_tabular.json:std_metrics.r2",
        )

        all_pass &= self.verify_claim(
            "GBDT Mean MAE (m)",
            expected=117.4,
            actual=mean_metrics.get("mae_m", 0),
            tolerance=0.5,
            source="validation_report_tabular.json:mean_metrics.mae_m",
        )

        all_pass &= self.verify_claim(
            "GBDT Mean RMSE (m)",
            expected=187.3,
            actual=mean_metrics.get("rmse_m", 0),
            tolerance=0.5,
            source="validation_report_tabular.json:mean_metrics.rmse_m",
        )

        return all_pass

    def verify_cnn_cv_metrics(self) -> bool:
        """Verify CNN image cross-validation metrics."""
        print(f"\n{BLUE}=== CNN Image CV Metrics ==={RESET}\n")

        report = self.load_report("validation_report_images.json")
        if not report:
            print(f"{RED}ERROR: validation_report_images.json not found{RESET}\n")
            return False

        mean_metrics = report.get("mean_metrics", {})
        std_metrics = report.get("std_metrics", {})

        all_pass = True
        all_pass &= self.verify_claim(
            "CNN Mean R²",
            expected=0.320,
            actual=mean_metrics.get("r2", 0),
            tolerance=0.001,
            source="validation_report_images.json:mean_metrics.r2",
        )

        all_pass &= self.verify_claim(
            "CNN Std R²",
            expected=0.152,
            actual=std_metrics.get("r2", 0),
            tolerance=0.001,
            source="validation_report_images.json:std_metrics.r2",
        )

        all_pass &= self.verify_claim(
            "CNN Mean MAE (m)",
            expected=238.2,
            actual=mean_metrics.get("mae_m", 0),
            tolerance=0.5,
            source="validation_report_images.json:mean_metrics.mae_m",
        )

        return all_pass

    def verify_ensemble_metrics(self) -> bool:
        """Verify ensemble model metrics."""
        print(f"\n{BLUE}=== Ensemble Metrics ==={RESET}\n")

        report = self.load_report("ensemble_results.json")
        if not report:
            print(f"{RED}ERROR: ensemble_results.json not found{RESET}\n")
            return False

        strategies = report.get("ensemble_strategies", {})

        all_pass = True

        # Simple average ensemble
        simple = strategies.get("simple_avg", {})
        if simple:
            all_pass &= self.verify_claim(
                "Simple Average Ensemble R²",
                expected=0.662,
                actual=simple.get("mean_r2", 0),
                tolerance=0.001,
                source="ensemble_results.json:ensemble_strategies.simple_avg.mean_r2",
            )

        # Weighted ensemble
        weighted = strategies.get("weighted_avg", {})
        if weighted:
            all_pass &= self.verify_claim(
                "Weighted Ensemble R²",
                expected=0.739,
                actual=weighted.get("mean_r2", 0),
                tolerance=0.001,
                source="ensemble_results.json:ensemble_strategies.weighted_avg.mean_r2",
            )

        # Stacking ensemble
        stacking = strategies.get("stacking", {})
        if stacking:
            all_pass &= self.verify_claim(
                "Stacking Ensemble R²",
                expected=0.724,
                actual=stacking.get("mean_r2", 0),
                tolerance=0.001,
                source="ensemble_results.json:ensemble_strategies.stacking.mean_r2",
            )

        return all_pass

    def verify_uncertainty_metrics(self) -> bool:
        """Verify uncertainty quantification metrics."""
        print(f"\n{BLUE}=== Uncertainty Quantification Metrics ==={RESET}\n")

        report = self.load_report("uncertainty_quantification_report.json")
        if not report:
            print(f"{RED}ERROR: uncertainty_quantification_report.json not found{RESET}\n")
            return False

        agg_metrics = report.get("aggregated_metrics", {})

        all_pass = True
        all_pass &= self.verify_claim(
            "UQ 90% Coverage",
            expected=0.771,
            actual=agg_metrics.get("mean_coverage", 0),
            tolerance=0.001,
            source="uncertainty_quantification_report.json:aggregated_metrics.mean_coverage",
        )

        all_pass &= self.verify_claim(
            "UQ Mean Interval Width (m)",
            expected=533.4,
            actual=agg_metrics.get("mean_interval_width_m", 0),
            tolerance=1.0,
            source="uncertainty_quantification_report.json:aggregated_metrics.mean_interval_width_m",
        )

        # Uncertainty-error correlation
        corr = agg_metrics.get("mean_uncertainty_error_correlation", 0)
        all_pass &= self.verify_claim(
            "Uncertainty-Error Correlation",
            expected=0.485,
            actual=corr,
            tolerance=0.01,
            source="uncertainty_quantification_report.json:aggregated_metrics.mean_uncertainty_error_correlation",
        )

        return all_pass

    def verify_domain_adaptation(self) -> bool:
        """Verify domain adaptation (LOFO) metrics."""
        print(f"\n{BLUE}=== Domain Adaptation (LOFO) Metrics ==={RESET}\n")

        report = self.load_report("domain_adaptation_f4_report.json")
        if not report:
            print(f"{RED}ERROR: domain_adaptation_f4_report.json not found{RESET}\n")
            return False

        all_pass = True

        # 18Feb25 (F4) LOFO baseline
        all_pass &= self.verify_claim(
            "18Feb25 LOFO Baseline R²",
            expected=-0.978,
            actual=report.get("baseline_loo_r2", 0),
            tolerance=0.01,
            source="domain_adaptation_f4_report.json:baseline_loo_r2",
        )

        # Few-shot results
        few_shot = report.get("few_shot_experiments", {})

        if "5_samples" in few_shot:
            all_pass &= self.verify_claim(
                "Few-shot 5-sample R²",
                expected=-0.528,
                actual=few_shot["5_samples"].get("r2", 0),
                tolerance=0.01,
                source="domain_adaptation_f4_report.json:few_shot_experiments.5_samples.r2",
            )

        if "10_samples" in few_shot:
            all_pass &= self.verify_claim(
                "Few-shot 10-sample R²",
                expected=-0.220,
                actual=few_shot["10_samples"].get("r2", 0),
                tolerance=0.01,
                source="domain_adaptation_f4_report.json:few_shot_experiments.10_samples.r2",
            )

        if "20_samples" in few_shot:
            all_pass &= self.verify_claim(
                "Few-shot 20-sample R²",
                expected=-0.708,
                actual=few_shot["20_samples"].get("r2", 0),
                tolerance=0.01,
                source="domain_adaptation_f4_report.json:few_shot_experiments.20_samples.r2",
            )

        return all_pass

    def check_per_flight_data(self) -> None:
        """Check if per-flight performance data exists."""
        print(f"\n{BLUE}=== Per-Flight Performance Check ==={RESET}\n")

        # Check for any per-flight reports
        per_flight_reports = list(self.reports_dir.glob("*per*flight*.json"))

        if not per_flight_reports:
            print(f"{YELLOW}WARNING: No per-flight performance reports found{RESET}")
            print(
                f"  The preprint contains a per-flight performance table that CANNOT be verified."
            )
            print(f"  Recommendation: Remove this table or compute per-flight metrics.")
            print()
        else:
            print(f"{GREEN}Found per-flight reports:{RESET}")
            for report in per_flight_reports:
                print(f"  - {report.name}")
            print()

    def check_lofo_coverage(self) -> None:
        """Check LOFO coverage for all flights."""
        print(f"\n{BLUE}=== LOFO Coverage Check ==={RESET}\n")

        expected_flights = ["30Oct24", "23Oct24", "10Feb25", "12Feb25", "18Feb25"]
        lofo_reports = list(self.reports_dir.glob("*lofo*.json")) + list(
            self.reports_dir.glob("*domain*.json")
        )

        print(f"Expected flights: {', '.join(expected_flights)}")
        print(f"Found LOFO reports: {len(lofo_reports)}")

        for report in lofo_reports:
            print(f"  - {report.name}")

        print(
            f"\n{YELLOW}WARNING: Only 18Feb25 (F4) LOFO verified in domain_adaptation_f4_report.json{RESET}"
        )
        print(f"  The preprint contains LOFO results for other flights that CANNOT be verified.")
        print(f"  Recommendation: Remove unverified LOFO entries or re-run LOFO for all flights.")
        print()

    def verify_dataset_info(self) -> bool:
        """Verify dataset information (sample counts, flights)."""
        print(f"\n{BLUE}=== Dataset Information ==={RESET}\n")

        # Check total samples from validation report
        tabular_report = self.load_report("validation_report_tabular.json")
        if not tabular_report:
            return False

        agg_preds = tabular_report.get("aggregated_predictions", {})
        total_samples = len(agg_preds.get("y_true", []))

        all_pass = True
        all_pass &= self.verify_claim(
            "Total Dataset Samples",
            expected=933,
            actual=total_samples,
            tolerance=0,
            source="validation_report_tabular.json:aggregated_predictions.y_true length",
        )

        return all_pass

    def run_all_checks(self) -> bool:
        """Run all verification checks."""
        print(f"\n{BLUE}{'=' * 70}{RESET}")
        print(f"{BLUE}PREPRINT VERIFICATION REPORT{RESET}")
        print(f"{BLUE}{'=' * 70}{RESET}")

        all_pass = True

        # Core metrics
        all_pass &= self.verify_tabular_cv_metrics()
        all_pass &= self.verify_cnn_cv_metrics()
        all_pass &= self.verify_ensemble_metrics()
        all_pass &= self.verify_uncertainty_metrics()
        all_pass &= self.verify_domain_adaptation()
        all_pass &= self.verify_dataset_info()

        # Check for missing data
        self.check_per_flight_data()
        self.check_lofo_coverage()

        # Summary
        print(f"\n{BLUE}{'=' * 70}{RESET}")
        print(f"{BLUE}VERIFICATION SUMMARY{RESET}")
        print(f"{BLUE}{'=' * 70}{RESET}\n")

        passed = sum(1 for r in self.verification_results if r["matches"])
        total = len(self.verification_results)

        if all_pass:
            print(f"{GREEN} ALL VERIFIED CLAIMS PASSED ({passed}/{total}){RESET}\n")
        else:
            print(f"{RED} SOME CLAIMS FAILED ({passed}/{total} passed){RESET}\n")

        # List any failures
        failures = [r for r in self.verification_results if not r["matches"]]
        if failures:
            print(f"{RED}Failed claims:{RESET}")
            for fail in failures:
                print(f"  - {fail['claim']}: expected {fail['expected']}, got {fail['actual']}")
            print()

        # Warnings about unverifiable claims
        print(f"{YELLOW}WARNINGS:{RESET}")
        print(f"  - Per-flight performance table (lines ~412-422) CANNOT be verified")
        print(f"  - LOFO results for flights other than 18Feb25 CANNOT be verified")
        print(f"  - Test coverage (93.5%) claim needs direct verification via pytest-cov")
        print()

        return all_pass

    def export_results(self, output_path: Path) -> None:
        """Export verification results to JSON."""
        output = {
            "timestamp": Path(__file__).stat().st_mtime,
            "total_claims_checked": len(self.verification_results),
            "claims_passed": sum(1 for r in self.verification_results if r["matches"]),
            "claims_failed": sum(1 for r in self.verification_results if not r["matches"]),
            "results": self.verification_results,
        }

        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)

        print(f"Verification results exported to: {output_path}")


def main():
    """Main entry point."""
    repo_root = Path(__file__).parent.parent

    verifier = PrePrintVerifier(repo_root)
    all_pass = verifier.run_all_checks()

    # Export results
    output_path = repo_root / "docs" / "preprint_verification_results.json"
    verifier.export_results(output_path)

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
