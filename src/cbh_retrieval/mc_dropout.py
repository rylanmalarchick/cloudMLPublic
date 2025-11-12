#!/usr/bin/env python3
"""
Sprint 6 - Phase 1, Task 1.2: Monte Carlo Dropout for Uncertainty Quantification

This module implements Monte Carlo (MC) Dropout for uncertainty quantification
in the Temporal ViT model. MC Dropout enables uncertainty estimates by performing
multiple forward passes with dropout enabled at inference time.

Key Features:
- Enable dropout at inference time
- Perform N forward passes to collect prediction distribution
- Compute mean, std, and confidence intervals
- Calibration analysis (uncertainty vs. error correlation)

Author: Sprint 6 Execution Agent
Date: 2025-01-10
"""

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")


class MCDropoutWrapper(nn.Module):
    """
    Wrapper for any PyTorch model to enable Monte Carlo Dropout.

    This wrapper recursively enables dropout layers during inference,
    allowing for uncertainty quantification through multiple stochastic
    forward passes.
    """

    def __init__(self, model: nn.Module, n_forward_passes: int = 20):
        """
        Args:
            model: The base model to wrap
            n_forward_passes: Number of forward passes for MC sampling
        """
        super().__init__()
        self.model = model
        self.n_forward_passes = n_forward_passes

    def enable_dropout(self):
        """Enable dropout layers during inference"""
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    def forward(self, x, return_uncertainty: bool = False):
        """
        Forward pass with optional uncertainty estimation.

        Args:
            x: Input tensor
            return_uncertainty: If True, perform MC Dropout and return uncertainty

        Returns:
            If return_uncertainty=False:
                predictions: (B,) or (B, 1) tensor of predictions
            If return_uncertainty=True:
                mean_pred: (B,) mean predictions
                std_pred: (B,) standard deviation (uncertainty)
                all_preds: (B, N) all MC samples
        """
        if not return_uncertainty:
            # Standard inference without dropout
            self.model.eval()
            with torch.no_grad():
                if hasattr(self.model, "forward"):
                    # Handle different model signatures
                    try:
                        output = self.model(x, predict_all_frames=False)
                    except TypeError:
                        output = self.model(x)
                else:
                    output = self.model(x)

                if isinstance(output, tuple):
                    output = output[-1]  # Take last output (center prediction)

                return output.squeeze(-1) if output.dim() > 1 else output

        else:
            # MC Dropout: multiple forward passes with dropout enabled
            self.enable_dropout()

            mc_predictions = []

            with torch.no_grad():
                for _ in range(self.n_forward_passes):
                    # Forward pass with dropout enabled
                    try:
                        output = self.model(x, predict_all_frames=False)
                    except TypeError:
                        output = self.model(x)

                    if isinstance(output, tuple):
                        output = output[-1]

                    pred = output.squeeze(-1) if output.dim() > 1 else output
                    mc_predictions.append(pred.cpu().numpy())

            # Stack predictions: (B, N)
            mc_predictions = np.stack(mc_predictions, axis=1)

            # Compute statistics
            mean_pred = np.mean(mc_predictions, axis=1)
            std_pred = np.std(mc_predictions, axis=1)

            return mean_pred, std_pred, mc_predictions


class UncertaintyQuantifier:
    """
    Comprehensive uncertainty quantification and calibration analysis.

    This class provides methods to:
    - Generate predictions with uncertainty estimates
    - Compute confidence intervals
    - Analyze calibration (coverage, reliability)
    - Identify low-confidence samples
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        n_forward_passes: int = 20,
        confidence_level: float = 0.90,
    ):
        """
        Args:
            model: The trained model
            device: Device for computation
            n_forward_passes: Number of MC Dropout samples
            confidence_level: Confidence level for intervals (e.g., 0.90 for 90% CI)
        """
        self.mc_model = MCDropoutWrapper(model, n_forward_passes=n_forward_passes)
        self.device = device
        self.n_forward_passes = n_forward_passes
        self.confidence_level = confidence_level

        # Z-score for confidence level (e.g., 1.645 for 90% CI)
        from scipy import stats

        self.z_score = stats.norm.ppf((1 + confidence_level) / 2)

    def predict_with_uncertainty(
        self, dataloader, return_all_samples: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Generate predictions with uncertainty estimates for entire dataset.

        Args:
            dataloader: PyTorch DataLoader
            return_all_samples: If True, return all MC samples

        Returns:
            Dictionary with:
                - predictions: (N,) mean predictions
                - uncertainties: (N,) standard deviations
                - ci_lower: (N,) lower confidence bounds
                - ci_upper: (N,) upper confidence bounds
                - targets: (N,) ground truth values
                - (optional) mc_samples: (N, n_passes) all MC samples
        """
        self.mc_model.to(self.device)

        all_predictions = []
        all_uncertainties = []
        all_targets = []
        all_mc_samples = [] if return_all_samples else None

        for batch in dataloader:
            # Unpack batch (handle different formats)
            if len(batch) == 3:
                frames, _, center_target = batch
            elif len(batch) == 2:
                frames, center_target = batch
            else:
                frames = batch[0]
                center_target = batch[-1]

            frames = frames.to(self.device)

            # MC Dropout inference
            mean_pred, std_pred, mc_preds = self.mc_model(
                frames, return_uncertainty=True
            )

            all_predictions.append(mean_pred)
            all_uncertainties.append(std_pred)
            all_targets.append(center_target.cpu().numpy())

            if return_all_samples:
                all_mc_samples.append(mc_preds)

        # Concatenate results
        predictions = np.concatenate(all_predictions)
        uncertainties = np.concatenate(all_uncertainties)
        targets = np.concatenate(all_targets)

        # Compute confidence intervals
        ci_lower = predictions - self.z_score * uncertainties
        ci_upper = predictions + self.z_score * uncertainties

        result = {
            "predictions": predictions,
            "uncertainties": uncertainties,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "targets": targets,
        }

        if return_all_samples:
            result["mc_samples"] = np.concatenate(all_mc_samples, axis=0)

        return result

    def compute_calibration_metrics(
        self, predictions: np.ndarray, uncertainties: np.ndarray, targets: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute calibration metrics.

        Args:
            predictions: (N,) predicted values
            uncertainties: (N,) uncertainty estimates (std)
            targets: (N,) ground truth values

        Returns:
            Dictionary with calibration metrics:
                - coverage_XX: fraction of targets within XX% CI
                - uncertainty_error_correlation: Pearson correlation
                - mean_uncertainty_km: average uncertainty
                - std_uncertainty_km: std of uncertainties
        """
        # Compute confidence intervals
        ci_lower = predictions - self.z_score * uncertainties
        ci_upper = predictions + self.z_score * uncertainties

        # Coverage: fraction of targets within CI
        coverage = np.mean((targets >= ci_lower) & (targets <= ci_upper))

        # Errors
        errors = np.abs(predictions - targets)

        # Correlation between uncertainty and error
        correlation = np.corrcoef(uncertainties, errors)[0, 1]

        return {
            f"coverage_{int(self.confidence_level * 100)}": float(coverage),
            "uncertainty_error_correlation": float(correlation),
            "mean_uncertainty_km": float(np.mean(uncertainties)),
            "std_uncertainty_km": float(np.std(uncertainties)),
        }

    def identify_low_confidence_samples(
        self,
        uncertainties: np.ndarray,
        percentile: float = 90.0,
        sample_ids: Optional[np.ndarray] = None,
    ) -> Dict[str, any]:
        """
        Identify samples with high uncertainty (low confidence).

        Args:
            uncertainties: (N,) uncertainty estimates
            percentile: Percentile threshold (e.g., 90 = top 10% uncertain)
            sample_ids: Optional array of sample IDs

        Returns:
            Dictionary with:
                - n_flagged: number of flagged samples
                - threshold_km: uncertainty threshold
                - flagged_sample_ids: list of flagged sample indices
                - flagged_uncertainties: uncertainties of flagged samples
        """
        threshold = np.percentile(uncertainties, percentile)
        flagged_mask = uncertainties > threshold
        flagged_indices = np.where(flagged_mask)[0]

        result = {
            "n_flagged": int(np.sum(flagged_mask)),
            "threshold_km": float(threshold),
            "flagged_sample_ids": flagged_indices.tolist(),
            "flagged_uncertainties": uncertainties[flagged_mask].tolist(),
        }

        if sample_ids is not None:
            result["flagged_sample_ids"] = sample_ids[flagged_mask].tolist()

        return result

    def calibration_curve(
        self, uncertainties: np.ndarray, errors: np.ndarray, n_bins: int = 10
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute calibration curve for plotting.

        The calibration curve bins samples by uncertainty and computes
        the mean error in each bin. For well-calibrated models, higher
        uncertainty should correlate with higher error.

        Args:
            uncertainties: (N,) uncertainty estimates
            errors: (N,) absolute errors
            n_bins: Number of bins

        Returns:
            bin_centers: (n_bins,) uncertainty bin centers
            mean_errors: (n_bins,) mean error per bin
            bin_counts: (n_bins,) number of samples per bin
        """
        # Create bins
        bin_edges = np.percentile(uncertainties, np.linspace(0, 100, n_bins + 1))
        bin_indices = np.digitize(uncertainties, bin_edges[1:-1])

        bin_centers = []
        mean_errors = []
        bin_counts = []

        for i in range(n_bins):
            mask = bin_indices == i
            if np.sum(mask) > 0:
                bin_centers.append(np.mean(uncertainties[mask]))
                mean_errors.append(np.mean(errors[mask]))
                bin_counts.append(np.sum(mask))

        return (np.array(bin_centers), np.array(mean_errors), np.array(bin_counts))

    def generate_report(
        self,
        predictions: np.ndarray,
        uncertainties: np.ndarray,
        targets: np.ndarray,
        sample_ids: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Generate complete uncertainty quantification report.

        Args:
            predictions: (N,) predictions
            uncertainties: (N,) uncertainties
            targets: (N,) ground truth
            sample_ids: Optional sample identifiers

        Returns:
            Complete report dictionary matching SOW schema
        """
        # Calibration metrics
        calibration_metrics = self.compute_calibration_metrics(
            predictions, uncertainties, targets
        )

        # Low-confidence samples
        low_confidence = self.identify_low_confidence_samples(
            uncertainties, percentile=90.0, sample_ids=sample_ids
        )

        report = {
            "method": "Monte Carlo Dropout",
            "n_forward_passes": self.n_forward_passes,
            "confidence_level": self.confidence_level,
            "calibration_metrics": calibration_metrics,
            "low_confidence_samples": low_confidence,
        }

        return report


def enable_mc_dropout(model: nn.Module):
    """
    Utility function to enable dropout in a trained model for MC Dropout.

    This function sets all Dropout layers to train mode while keeping
    the rest of the model in eval mode.

    Args:
        model: PyTorch model
    """
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


def compute_prediction_interval_coverage(
    predictions: np.ndarray,
    uncertainties: np.ndarray,
    targets: np.ndarray,
    confidence_level: float = 0.90,
) -> float:
    """
    Compute empirical coverage of prediction intervals.

    Args:
        predictions: (N,) predicted values
        uncertainties: (N,) standard deviations
        targets: (N,) ground truth values
        confidence_level: Nominal confidence level (e.g., 0.90)

    Returns:
        Empirical coverage (fraction of targets within intervals)
    """
    from scipy import stats

    z_score = stats.norm.ppf((1 + confidence_level) / 2)

    ci_lower = predictions - z_score * uncertainties
    ci_upper = predictions + z_score * uncertainties

    coverage = np.mean((targets >= ci_lower) & (targets <= ci_upper))

    return float(coverage)
