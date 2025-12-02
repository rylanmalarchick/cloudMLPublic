#!/usr/bin/env python3
"""Monte Carlo Dropout for Uncertainty Quantification.

This module implements Monte Carlo (MC) Dropout for uncertainty quantification
in the Temporal ViT model. MC Dropout enables uncertainty estimates by performing
multiple forward passes with dropout enabled at inference time.

Key Features:
    - Enable dropout at inference time for stochastic forward passes
    - Perform N forward passes to collect prediction distribution
    - Compute mean, standard deviation, and confidence intervals
    - Calibration analysis (uncertainty vs. error correlation)
    - Identify low-confidence samples for flagging

Example:
    Basic usage with a trained model::

        from src.cbh_retrieval.mc_dropout import UncertaintyQuantifier

        quantifier = UncertaintyQuantifier(model, n_forward_passes=50)
        results = quantifier.predict_with_uncertainty(dataloader)
        print(f"Mean uncertainty: {results['uncertainties'].mean():.2f} km")

Author: Sprint 6 Execution Agent
Date: 2025-01-10
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from sklearn.exceptions import ConvergenceWarning
from torch.utils.data import DataLoader

# Suppress sklearn convergence and numpy deprecation warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


class MCDropoutWrapper(nn.Module):
    """Wrapper for any PyTorch model to enable Monte Carlo Dropout.

    This wrapper recursively enables dropout layers during inference,
    allowing for uncertainty quantification through multiple stochastic
    forward passes.

    Attributes:
        model: The wrapped PyTorch model.
        n_forward_passes: Number of stochastic forward passes to perform.

    Example:
        Wrapping a model for MC Dropout inference::

            wrapped_model = MCDropoutWrapper(trained_model, n_forward_passes=50)
            mean, std, samples = wrapped_model(input_tensor, return_uncertainty=True)
    """

    def __init__(self, model: nn.Module, n_forward_passes: int = 20) -> None:
        """Initialize the MC Dropout wrapper.

        Args:
            model: The base PyTorch model to wrap. Must contain at least one
                ``nn.Dropout`` layer for MC Dropout to be effective.
            n_forward_passes: Number of forward passes for MC sampling.
                Higher values provide more stable uncertainty estimates but
                increase inference time linearly. Defaults to 20.
        """
        super().__init__()
        self.model = model
        self.n_forward_passes = n_forward_passes

    def enable_dropout(self) -> None:
        """Enable dropout layers during inference.

        Iterates through all modules in the model and sets any ``nn.Dropout``
        layers to training mode, which enables stochastic dropout during
        the forward pass even when the model is in eval mode.
        """
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    def forward(
        self, x: torch.Tensor, return_uncertainty: bool = False
    ) -> Union[torch.Tensor, Tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]]:
        """Perform forward pass with optional uncertainty estimation.

        Args:
            x: Input tensor of shape ``(B, ...)`` where B is the batch size.
                The remaining dimensions depend on the wrapped model's
                expected input format.
            return_uncertainty: If True, perform MC Dropout and return
                uncertainty estimates. If False, perform standard deterministic
                inference. Defaults to False.

        Returns:
            If ``return_uncertainty=False``:
                A tensor of predictions with shape ``(B,)``.

            If ``return_uncertainty=True``:
                A tuple containing:
                    - mean_pred: Mean predictions of shape ``(B,)``.
                    - std_pred: Standard deviation (uncertainty) of shape ``(B,)``.
                    - all_preds: All MC samples of shape ``(B, n_forward_passes)``.
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

            mc_predictions: List[NDArray[np.floating]] = []

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
            mc_predictions_array = np.stack(mc_predictions, axis=1)

            # Compute statistics
            mean_pred = np.mean(mc_predictions_array, axis=1)
            std_pred = np.std(mc_predictions_array, axis=1)

            return mean_pred, std_pred, mc_predictions_array


class UncertaintyQuantifier:
    """Comprehensive uncertainty quantification and calibration analysis.

    This class provides methods to generate predictions with uncertainty
    estimates, compute confidence intervals, analyze calibration, and
    identify low-confidence samples that may require manual review.

    Attributes:
        mc_model: The wrapped model with MC Dropout enabled.
        device: Device for computation (cuda or cpu).
        n_forward_passes: Number of MC Dropout forward passes.
        confidence_level: Confidence level for prediction intervals.
        z_score: Z-score corresponding to the confidence level.

    Example:
        Generate predictions with uncertainty::

            quantifier = UncertaintyQuantifier(model, n_forward_passes=50)
            results = quantifier.predict_with_uncertainty(test_loader)

            # Access results
            predictions = results["predictions"]
            uncertainties = results["uncertainties"]
            ci_lower = results["ci_lower"]
            ci_upper = results["ci_upper"]
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        n_forward_passes: int = 20,
        confidence_level: float = 0.90,
    ) -> None:
        """Initialize the uncertainty quantifier.

        Args:
            model: The trained PyTorch model to wrap.
            device: Device for computation. Defaults to "cuda" if available,
                otherwise "cpu".
            n_forward_passes: Number of MC Dropout samples to collect.
                Higher values provide more stable estimates. Defaults to 20.
            confidence_level: Confidence level for prediction intervals,
                expressed as a fraction (e.g., 0.90 for 90% CI). Defaults to 0.90.

        Raises:
            ValueError: If confidence_level is not in the range (0, 1).
        """
        if not 0 < confidence_level < 1:
            raise ValueError(
                f"confidence_level must be in (0, 1), got {confidence_level}"
            )

        self.mc_model = MCDropoutWrapper(model, n_forward_passes=n_forward_passes)
        self.device = device
        self.n_forward_passes = n_forward_passes
        self.confidence_level = confidence_level

        # Z-score for confidence level (e.g., 1.645 for 90% CI)
        from scipy import stats

        self.z_score: float = float(stats.norm.ppf((1 + confidence_level) / 2))

    def predict_with_uncertainty(
        self, dataloader: DataLoader, return_all_samples: bool = False
    ) -> Dict[str, NDArray[np.floating]]:
        """Generate predictions with uncertainty estimates for entire dataset.

        Performs MC Dropout inference on all batches in the dataloader and
        aggregates the results.

        Args:
            dataloader: PyTorch DataLoader containing the evaluation dataset.
                Expected to yield batches of (frames, targets) or
                (frames, aux_data, targets).
            return_all_samples: If True, include all individual MC samples
                in the returned dictionary. Defaults to False.

        Returns:
            Dictionary containing:
                - ``predictions``: Mean predictions of shape ``(N,)``.
                - ``uncertainties``: Standard deviations of shape ``(N,)``.
                - ``ci_lower``: Lower confidence bounds of shape ``(N,)``.
                - ``ci_upper``: Upper confidence bounds of shape ``(N,)``.
                - ``targets``: Ground truth values of shape ``(N,)``.
                - ``mc_samples`` (optional): All MC samples of shape
                  ``(N, n_forward_passes)``, only if ``return_all_samples=True``.
        """
        self.mc_model.to(self.device)

        all_predictions: List[NDArray[np.floating]] = []
        all_uncertainties: List[NDArray[np.floating]] = []
        all_targets: List[NDArray[np.floating]] = []
        all_mc_samples: List[NDArray[np.floating]] = []

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

        result: Dict[str, NDArray[np.floating]] = {
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
        self,
        predictions: NDArray[np.floating],
        uncertainties: NDArray[np.floating],
        targets: NDArray[np.floating],
    ) -> Dict[str, float]:
        """Compute calibration metrics for uncertainty estimates.

        Calibration measures how well the predicted uncertainties correspond
        to actual prediction errors. A well-calibrated model should have
        higher uncertainty for predictions with larger errors.

        Args:
            predictions: Predicted values of shape ``(N,)``.
            uncertainties: Uncertainty estimates (standard deviations)
                of shape ``(N,)``.
            targets: Ground truth values of shape ``(N,)``.

        Returns:
            Dictionary containing:
                - ``coverage_XX``: Fraction of targets within XX% confidence
                  interval, where XX is ``confidence_level * 100``.
                - ``uncertainty_error_correlation``: Pearson correlation
                  coefficient between uncertainties and absolute errors.
                  Higher values indicate better calibration.
                - ``mean_uncertainty_km``: Average uncertainty in kilometers.
                - ``std_uncertainty_km``: Standard deviation of uncertainties
                  in kilometers.
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
        uncertainties: NDArray[np.floating],
        percentile: float = 90.0,
        sample_ids: Optional[NDArray[np.integer]] = None,
    ) -> Dict[str, Any]:
        """Identify samples with high uncertainty (low confidence).

        Flags samples whose uncertainty exceeds a percentile threshold,
        indicating predictions that may require manual review or additional
        validation.

        Args:
            uncertainties: Uncertainty estimates of shape ``(N,)``.
            percentile: Percentile threshold for flagging. For example,
                90.0 flags the top 10% most uncertain samples. Defaults to 90.0.
            sample_ids: Optional array of sample identifiers of shape ``(N,)``.
                If provided, the returned flagged IDs will use these instead
                of integer indices.

        Returns:
            Dictionary containing:
                - ``n_flagged``: Number of flagged samples.
                - ``threshold_km``: Uncertainty threshold value in kilometers.
                - ``flagged_sample_ids``: List of indices (or IDs if
                  ``sample_ids`` provided) of flagged samples.
                - ``flagged_uncertainties``: List of uncertainty values for
                  flagged samples.
        """
        threshold = np.percentile(uncertainties, percentile)
        flagged_mask = uncertainties > threshold
        flagged_indices = np.where(flagged_mask)[0]

        result: Dict[str, Any] = {
            "n_flagged": int(np.sum(flagged_mask)),
            "threshold_km": float(threshold),
            "flagged_sample_ids": flagged_indices.tolist(),
            "flagged_uncertainties": uncertainties[flagged_mask].tolist(),
        }

        if sample_ids is not None:
            result["flagged_sample_ids"] = sample_ids[flagged_mask].tolist()

        return result

    def calibration_curve(
        self,
        uncertainties: NDArray[np.floating],
        errors: NDArray[np.floating],
        n_bins: int = 10,
    ) -> Tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.integer]]:
        """Compute calibration curve data for visualization.

        The calibration curve bins samples by uncertainty and computes the
        mean error in each bin. For well-calibrated models, higher uncertainty
        should correlate with higher error, producing a monotonically
        increasing curve.

        Args:
            uncertainties: Uncertainty estimates of shape ``(N,)``.
            errors: Absolute errors of shape ``(N,)``.
            n_bins: Number of uncertainty bins to create. Defaults to 10.

        Returns:
            A tuple containing:
                - ``bin_centers``: Array of shape ``(n_bins,)`` with the mean
                  uncertainty value in each bin.
                - ``mean_errors``: Array of shape ``(n_bins,)`` with the mean
                  absolute error in each bin.
                - ``bin_counts``: Array of shape ``(n_bins,)`` with the number
                  of samples in each bin.
        """
        # Create bins
        bin_edges = np.percentile(uncertainties, np.linspace(0, 100, n_bins + 1))
        bin_indices = np.digitize(uncertainties, bin_edges[1:-1])

        bin_centers: List[float] = []
        mean_errors: List[float] = []
        bin_counts: List[int] = []

        for i in range(n_bins):
            mask = bin_indices == i
            if np.sum(mask) > 0:
                bin_centers.append(float(np.mean(uncertainties[mask])))
                mean_errors.append(float(np.mean(errors[mask])))
                bin_counts.append(int(np.sum(mask)))

        return (
            np.array(bin_centers),
            np.array(mean_errors),
            np.array(bin_counts, dtype=np.int64),
        )

    def generate_report(
        self,
        predictions: NDArray[np.floating],
        uncertainties: NDArray[np.floating],
        targets: NDArray[np.floating],
        sample_ids: Optional[NDArray[np.integer]] = None,
    ) -> Dict[str, Any]:
        """Generate complete uncertainty quantification report.

        Combines calibration metrics and low-confidence sample identification
        into a comprehensive report dictionary suitable for logging or
        serialization.

        Args:
            predictions: Predicted values of shape ``(N,)``.
            uncertainties: Uncertainty estimates of shape ``(N,)``.
            targets: Ground truth values of shape ``(N,)``.
            sample_ids: Optional sample identifiers of shape ``(N,)`` for
                labeling flagged samples.

        Returns:
            Dictionary containing:
                - ``method``: The uncertainty quantification method name.
                - ``n_forward_passes``: Number of MC Dropout passes used.
                - ``confidence_level``: Confidence level for intervals.
                - ``calibration_metrics``: Dictionary of calibration statistics.
                - ``low_confidence_samples``: Dictionary identifying uncertain
                  samples that may need review.
        """
        # Calibration metrics
        calibration_metrics = self.compute_calibration_metrics(
            predictions, uncertainties, targets
        )

        # Low-confidence samples
        low_confidence = self.identify_low_confidence_samples(
            uncertainties, percentile=90.0, sample_ids=sample_ids
        )

        report: Dict[str, Any] = {
            "method": "Monte Carlo Dropout",
            "n_forward_passes": self.n_forward_passes,
            "confidence_level": self.confidence_level,
            "calibration_metrics": calibration_metrics,
            "low_confidence_samples": low_confidence,
        }

        return report


def enable_mc_dropout(model: nn.Module) -> None:
    """Enable dropout in a trained model for MC Dropout inference.

    This utility function sets all Dropout layers to training mode while
    keeping the rest of the model in eval mode. This is useful when you
    want to manually control MC Dropout behavior without using the
    ``MCDropoutWrapper`` class.

    Args:
        model: PyTorch model containing ``nn.Dropout`` layers. The model
            will be set to eval mode, with only dropout layers in train mode.

    Example:
        Manual MC Dropout inference::

            model.eval()
            enable_mc_dropout(model)

            predictions = []
            for _ in range(50):
                with torch.no_grad():
                    pred = model(input_tensor)
                predictions.append(pred)

            mean_pred = torch.stack(predictions).mean(dim=0)
            std_pred = torch.stack(predictions).std(dim=0)
    """
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


def compute_prediction_interval_coverage(
    predictions: NDArray[np.floating],
    uncertainties: NDArray[np.floating],
    targets: NDArray[np.floating],
    confidence_level: float = 0.90,
) -> float:
    """Compute empirical coverage of prediction intervals.

    Calculates the fraction of ground truth values that fall within the
    predicted confidence intervals. For well-calibrated models, the empirical
    coverage should approximately match the nominal confidence level.

    Args:
        predictions: Predicted values of shape ``(N,)``.
        uncertainties: Standard deviations (uncertainty estimates) of shape
            ``(N,)``. Used to construct symmetric confidence intervals around
            the predictions.
        targets: Ground truth values of shape ``(N,)``.
        confidence_level: Nominal confidence level for the prediction intervals,
            expressed as a fraction (e.g., 0.90 for 90% CI). Defaults to 0.90.

    Returns:
        Empirical coverage as a float in [0, 1], representing the fraction
        of targets that fall within their respective confidence intervals.

    Raises:
        ValueError: If confidence_level is not in the range (0, 1).

    Example:
        Evaluate prediction interval calibration::

            coverage = compute_prediction_interval_coverage(
                predictions=model_preds,
                uncertainties=model_stds,
                targets=ground_truth,
                confidence_level=0.90,
            )
            print(f"90% CI coverage: {coverage:.1%}")
            # Well-calibrated model should show ~90% coverage
    """
    from scipy import stats

    z_score = stats.norm.ppf((1 + confidence_level) / 2)

    ci_lower = predictions - z_score * uncertainties
    ci_upper = predictions + z_score * uncertainties

    coverage = np.mean((targets >= ci_lower) & (targets <= ci_upper))

    return float(coverage)
