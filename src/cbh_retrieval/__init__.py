"""
Cloud Base Height (CBH) Retrieval Module

Production-ready machine learning system for predicting cloud base height
from satellite imagery and atmospheric data.

Sprint 6 - Production Release
Performance: R² = 0.744 (GBDT), 0.7391 (Ensemble), MAE = 117.4m
"""

__version__ = "1.0.0"
__author__ = "NASA CBH Retrieval Team"

# Core models and datasets
# Ensemble methods
from .ensemble_models import (
    SimpleAverageEnsemble,
    StackingEnsemble,
    WeightedEnsemble,
)

# Analysis
from .error_analysis import (
    analyze_errors,
    compute_error_metrics,
    plot_error_distributions,
)
from .image_dataset import ImageCBHDataset
from .mc_dropout import MCDropoutModel
from .offline_validation_images import main as validate_images

# Validation
from .offline_validation_tabular import main as validate_tabular

# Visualization
from .performance_plots import (
    plot_feature_importance,
    plot_fold_metrics,
    plot_predictions_scatter,
)

# Training
from .train_production_model import train_production_model

# Uncertainty quantification
from .uncertainty_quantification_tabular import (
    compute_quantile_intervals,
    quantify_uncertainty,
)

__all__ = [
    # Version
    "__version__",
    "__author__",
    # Core
    "ImageCBHDataset",
    "MCDropoutModel",
    # Training
    "train_production_model",
    # Validation
    "validate_tabular",
    "validate_images",
    # Uncertainty
    "quantify_uncertainty",
    "compute_quantile_intervals",
    # Ensemble
    "WeightedEnsemble",
    "StackingEnsemble",
    "SimpleAverageEnsemble",
    # Analysis
    "analyze_errors",
    "compute_error_metrics",
    "plot_error_distributions",
    # Visualization
    "plot_predictions_scatter",
    "plot_feature_importance",
    "plot_fold_metrics",
]
