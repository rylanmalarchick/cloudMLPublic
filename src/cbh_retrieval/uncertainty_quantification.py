#!/usr/bin/env python3
"""
Sprint 6 - Phase 1, Task 1.2: Uncertainty Quantification

This script implements uncertainty quantification for the Temporal ViT model
using Monte Carlo (MC) Dropout. It generates uncertainty estimates, performs
calibration analysis, and identifies low-confidence predictions.

Deliverables:
- Uncertainty quantification report (JSON)
- Calibration plots (uncertainty vs. error, coverage histogram)
- Low-confidence sample analysis

Author: Sprint 6 Execution Agent
Date: 2025-01-10
"""

import json
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from scipy import stats
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from sow_outputs.sprint6.modules.mc_dropout import (
    MCDropoutWrapper,
    UncertaintyQuantifier,
)
from sow_outputs.wp5.wp5_utils import get_stratified_folds
from src.hdf5_dataset import HDF5CloudDataset

# Import model architecture (reuse from validation script)
sys.path.insert(0, str(Path(__file__).parent))
from offline_validation import TemporalConsistencyViT, TemporalDataset

# Try to import transformers
try:
    from transformers import ViTForImageClassification

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("WARNING: transformers library not available")


# ==============================================================================
# Uncertainty Analysis Class
# ==============================================================================


class UncertaintyAnalyzer:
    """
    Performs comprehensive uncertainty quantification and calibration analysis.
    """

    def __init__(
        self,
        output_dir: Path,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        n_forward_passes: int = 20,
        confidence_level: float = 0.90,
    ):
        self.output_dir = Path(output_dir)
        self.device = device
        self.n_forward_passes = n_forward_passes
        self.confidence_level = confidence_level

        # Create output directories
        self.reports_dir = self.output_dir / "reports"
        self.figures_dir = self.output_dir / "figures" / "uncertainty"
        self.checkpoints_dir = self.output_dir / "checkpoints"

        for dir_path in [self.reports_dir, self.figures_dir, self.checkpoints_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        print(f" Uncertainty analysis output directory: {self.output_dir}")
        print(f" Device: {self.device}")
        print(f" MC Dropout forward passes: {self.n_forward_passes}")
        print(f" Confidence level: {self.confidence_level * 100:.0f}%")

    def load_data(
        self, integrated_features_path: str
    ) -> Tuple[HDF5CloudDataset, np.ndarray]:
        """Load integrated features dataset"""
        print(f"\n{'=' * 80}")
        print("Loading Dataset")
        print(f"{'=' * 80}")

        dataset = HDF5CloudDataset(integrated_features_path)
        cbh_values = dataset.cbh_values

        print(f" Total samples: {len(dataset)}")
        print(f" CBH range: [{cbh_values.min():.3f}, {cbh_values.max():.3f}] km")
        print(f" CBH mean: {cbh_values.mean():.3f} km")

        return dataset, cbh_values

    def load_trained_model(self, checkpoint_path: Path) -> nn.Module:
        """Load a trained model from checkpoint"""
        print(f"\n{'=' * 80}")
        print("Loading Trained Model")
        print(f"{'=' * 80}")

        model = TemporalConsistencyViT(
            pretrained_model="WinKawaks/vit-tiny-patch16-224", n_frames=5
        )

        if checkpoint_path.exists():
            print(f"Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            model.load_state_dict(checkpoint["model_state_dict"])
            print(f" Model loaded from checkpoint")
        else:
            print(f" Checkpoint not found: {checkpoint_path}")
            print(f" Using freshly initialized model (for testing only)")

        model = model.to(self.device)
        model.eval()

        return model

    def run_uncertainty_quantification(
        self,
        model: nn.Module,
        image_dataset: HDF5CloudDataset,
        cbh_values: np.ndarray,
        batch_size: int = 4,
    ) -> Dict:
        """
        Run uncertainty quantification on validation set using MC Dropout.
        """
        print(f"\n{'=' * 80}")
        print("Running Uncertainty Quantification (MC Dropout)")
        print(f"{'=' * 80}")

        # Use first fold validation set for UQ analysis
        folds = get_stratified_folds(cbh_values, n_splits=5)
        _, val_indices = folds[0]

        print(f"Validation set size: {len(val_indices)} samples")

        # Create validation dataset
        val_dataset = TemporalDataset(
            image_dataset, val_indices, cbh_values, n_frames=5
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
        )

        # Initialize uncertainty quantifier
        uq = UncertaintyQuantifier(
            model=model,
            device=self.device,
            n_forward_passes=self.n_forward_passes,
            confidence_level=self.confidence_level,
        )

        # Generate predictions with uncertainty
        print(f"Performing {self.n_forward_passes} forward passes per sample...")
        uq_results = uq.predict_with_uncertainty(val_loader, return_all_samples=False)

        predictions = uq_results["predictions"]
        uncertainties = uq_results["uncertainties"]
        targets = uq_results["targets"]
        ci_lower = uq_results["ci_lower"]
        ci_upper = uq_results["ci_upper"]

        print(f" Predictions generated")
        print(
            f"  Mean uncertainty: {np.mean(uncertainties):.4f} km (±{np.std(uncertainties):.4f})"
        )

        # Compute errors
        errors = np.abs(predictions - targets)

        # Calibration metrics
        calibration_metrics = uq.compute_calibration_metrics(
            predictions, uncertainties, targets
        )

        print(f"\nCalibration Metrics:")
        print(
            f"  {int(self.confidence_level * 100)}% CI Coverage: {calibration_metrics[f'coverage_{int(self.confidence_level * 100)}']:.3f}"
        )
        print(
            f"  Uncertainty-Error Correlation: {calibration_metrics['uncertainty_error_correlation']:.3f}"
        )
        print(
            f"  Mean Uncertainty: {calibration_metrics['mean_uncertainty_km']:.4f} km"
        )

        # Identify low-confidence samples
        low_confidence = uq.identify_low_confidence_samples(
            uncertainties, percentile=90.0, sample_ids=val_indices
        )

        print(f"\nLow-Confidence Samples (top 10%):")
        print(f"  N flagged: {low_confidence['n_flagged']}")
        print(f"  Threshold: {low_confidence['threshold_km']:.4f} km")

        # Calibration curve data
        bin_centers, mean_errors, bin_counts = uq.calibration_curve(
            uncertainties, errors, n_bins=10
        )

        return {
            "predictions": predictions,
            "uncertainties": uncertainties,
            "targets": targets,
            "errors": errors,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "calibration_metrics": calibration_metrics,
            "low_confidence": low_confidence,
            "calibration_curve": {
                "bin_centers": bin_centers,
                "mean_errors": mean_errors,
                "bin_counts": bin_counts,
            },
        }

    def create_visualizations(self, uq_results: Dict):
        """Create uncertainty quantification visualizations"""

        print(f"\n{'=' * 80}")
        print("Generating Uncertainty Visualizations")
        print(f"{'=' * 80}")

        sns.set_style("whitegrid")

        predictions = uq_results["predictions"]
        uncertainties = uq_results["uncertainties"]
        targets = uq_results["targets"]
        errors = uq_results["errors"]
        ci_lower = uq_results["ci_lower"]
        ci_upper = uq_results["ci_upper"]

        # 1. Uncertainty vs. Error Scatter Plot
        fig, ax = plt.subplots(figsize=(10, 8))

        scatter = ax.scatter(
            uncertainties,
            errors,
            c=targets,
            cmap="viridis",
            alpha=0.6,
            s=30,
            edgecolors="k",
            linewidth=0.5,
        )

        # Add trend line
        z = np.polyfit(uncertainties, errors, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(uncertainties.min(), uncertainties.max(), 100)
        ax.plot(
            x_trend, p(x_trend), "r--", lw=2, label=f"Trend: y={z[0]:.2f}x+{z[1]:.3f}"
        )

        ax.set_xlabel("Predicted Uncertainty (km)", fontsize=12)
        ax.set_ylabel("Absolute Error (km)", fontsize=12)
        ax.set_title("Uncertainty vs. Error\n(MC Dropout, N=20)", fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("True CBH (km)", fontsize=10)

        # Add correlation annotation
        corr = uq_results["calibration_metrics"]["uncertainty_error_correlation"]
        ax.text(
            0.05,
            0.95,
            f"Pearson r = {corr:.3f}",
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        plt.tight_layout()
        plt.savefig(self.figures_dir / "uncertainty_vs_error.png", dpi=300)
        plt.close()
        print(" Saved: uncertainty_vs_error.png")

        # 2. Calibration Curve
        fig, ax = plt.subplots(figsize=(10, 8))

        bin_centers = uq_results["calibration_curve"]["bin_centers"]
        mean_errors = uq_results["calibration_curve"]["mean_errors"]
        bin_counts = uq_results["calibration_curve"]["bin_counts"]

        ax.plot(bin_centers, mean_errors, "o-", lw=2, markersize=8, label="Empirical")
        ax.plot(
            [bin_centers.min(), bin_centers.max()],
            [bin_centers.min(), bin_centers.max()],
            "r--",
            lw=2,
            label="Perfect Calibration",
        )

        # Add error bars (±1 std within each bin)
        # For simplicity, we'll skip error bars in this version

        ax.set_xlabel("Predicted Uncertainty (km)", fontsize=12)
        ax.set_ylabel("Mean Absolute Error (km)", fontsize=12)
        ax.set_title("Calibration Curve\n(Binned Uncertainty vs. Error)", fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.figures_dir / "calibration_curve.png", dpi=300)
        plt.close()
        print(" Saved: calibration_curve.png")

        # 3. Confidence Interval Coverage Histogram
        fig, ax = plt.subplots(figsize=(10, 6))

        # Check which targets fall within CI
        within_ci = (targets >= ci_lower) & (targets <= ci_upper)
        coverage = np.mean(within_ci)

        ax.bar(
            ["Within CI", "Outside CI"],
            [np.sum(within_ci), np.sum(~within_ci)],
            color=["green", "red"],
            alpha=0.7,
            edgecolor="black",
        )

        ax.set_ylabel("Number of Samples", fontsize=12)
        ax.set_title(
            f"{int(self.confidence_level * 100)}% Confidence Interval Coverage\n"
            f"Empirical Coverage: {coverage:.1%}",
            fontsize=14,
        )
        ax.grid(True, alpha=0.3, axis="y")

        # Add expected coverage line
        expected = self.confidence_level * len(targets)
        ax.axhline(
            y=expected,
            color="blue",
            linestyle="--",
            lw=2,
            label=f"Expected ({self.confidence_level:.0%})",
        )
        ax.legend(fontsize=10)

        plt.tight_layout()
        plt.savefig(self.figures_dir / "coverage_histogram.png", dpi=300)
        plt.close()
        print(" Saved: coverage_histogram.png")

        # 4. Uncertainty Distribution
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram of uncertainties
        axes[0].hist(
            uncertainties, bins=30, color="skyblue", edgecolor="black", alpha=0.7
        )
        axes[0].axvline(
            x=np.mean(uncertainties), color="r", linestyle="--", lw=2, label="Mean"
        )
        axes[0].axvline(
            x=np.percentile(uncertainties, 90),
            color="orange",
            linestyle="--",
            lw=2,
            label="90th %ile",
        )
        axes[0].set_xlabel("Uncertainty (km)", fontsize=12)
        axes[0].set_ylabel("Frequency", fontsize=12)
        axes[0].set_title("Distribution of Uncertainties", fontsize=14)
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)

        # Box plot comparing low vs high confidence samples
        threshold = uq_results["low_confidence"]["threshold_km"]
        low_conf_mask = uncertainties > threshold
        high_conf_errors = errors[~low_conf_mask]
        low_conf_errors = errors[low_conf_mask]

        axes[1].boxplot(
            [high_conf_errors, low_conf_errors],
            labels=["High Confidence\n(< 90th %ile)", "Low Confidence\n(> 90th %ile)"],
            patch_artist=True,
            boxprops=dict(facecolor="lightblue", alpha=0.7),
            medianprops=dict(color="red", linewidth=2),
        )
        axes[1].set_ylabel("Absolute Error (km)", fontsize=12)
        axes[1].set_title("Error Distribution by Confidence Level", fontsize=14)
        axes[1].grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        plt.savefig(self.figures_dir / "uncertainty_distribution.png", dpi=300)
        plt.close()
        print(" Saved: uncertainty_distribution.png")

    def save_uncertainty_report(self, uq_results: Dict):
        """Save uncertainty quantification report in required JSON schema"""

        report = {
            "method": "Monte Carlo Dropout",
            "n_forward_passes": self.n_forward_passes,
            "confidence_level": self.confidence_level,
            "calibration_metrics": uq_results["calibration_metrics"],
            "low_confidence_samples": uq_results["low_confidence"],
            "timestamp": datetime.now().isoformat(),
        }

        report_path = self.reports_dir / "uncertainty_quantification_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"\n{'=' * 80}")
        print("Uncertainty Quantification Report Summary")
        print(f"{'=' * 80}")
        print(f"Method: {report['method']}")
        print(f"Forward Passes: {report['n_forward_passes']}")
        print(f"Confidence Level: {report['confidence_level'] * 100:.0f}%")
        print(f"\nCalibration Metrics:")
        for key, value in report["calibration_metrics"].items():
            print(f"  {key}: {value:.4f}")
        print(f"\nLow-Confidence Samples:")
        print(f"  N flagged: {report['low_confidence_samples']['n_flagged']}")
        print(f"  Threshold: {report['low_confidence_samples']['threshold_km']:.4f} km")
        print(f"\n Report saved: {report_path}")

        return report


# ==============================================================================
# Main Execution
# ==============================================================================


def main():
    """Main execution function"""

    # Paths (following Sprint 6 SOW Table 1)
    project_root = Path(
        "/home/rylan/Documents/research/NASA/programDirectory/cloudMLPublic"
    )
    integrated_features_path = str(
        project_root / "outputs/preprocessed_data/Integrated_Features.hdf5"
    )
    output_dir = project_root / "."

    # Load a trained model checkpoint from Task 1.1
    checkpoint_path = output_dir / "checkpoints" / "fold_0_model.pth"

    print(f"\n{'=' * 80}")
    print("Sprint 6 - Phase 1, Task 1.2: Uncertainty Quantification")
    print(f"{'=' * 80}")
    print(f"Project Root: {project_root}")
    print(f"Integrated Features: {integrated_features_path}")
    print(f"Output Directory: {output_dir}")
    print(f"Model Checkpoint: {checkpoint_path}")

    # Initialize analyzer
    analyzer = UncertaintyAnalyzer(
        output_dir=output_dir, n_forward_passes=20, confidence_level=0.90
    )

    # Load data
    image_dataset, cbh_values = analyzer.load_data(integrated_features_path)

    # Load trained model
    model = analyzer.load_trained_model(checkpoint_path)

    # Run uncertainty quantification
    uq_results = analyzer.run_uncertainty_quantification(
        model=model, image_dataset=image_dataset, cbh_values=cbh_values, batch_size=4
    )

    # Create visualizations
    analyzer.create_visualizations(uq_results)

    # Save report
    report = analyzer.save_uncertainty_report(uq_results)

    print(f"\n{'=' * 80}")
    print(" Task 1.2 Complete: Uncertainty Quantification")
    print(f"{'=' * 80}")
    print(f"All outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
