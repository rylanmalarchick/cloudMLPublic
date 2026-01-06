"""
Shared configuration for CBH Retrieval module.

This module centralizes hyperparameters, paths, and constants used across
the CBH retrieval pipeline to ensure consistency and ease of maintenance.

Canonical values validated in preprint (cloudml_academic_preprint.tex):
- GBDT: R² = 0.744, MAE = 117.4m
- Ensemble: R² = 0.7391
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

# =============================================================================
# Path Configuration
# =============================================================================

# Project root (relative to this file: src/cbh_retrieval/config.py)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Data paths
DATA_DIR = PROJECT_ROOT / "outputs" / "preprocessed_data"
INTEGRATED_FEATURES_PATH = DATA_DIR / "Integrated_Features.hdf5"

# Output paths
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUT_DIR / "models"
FIGURES_DIR = OUTPUT_DIR / "figures"
REPORTS_DIR = OUTPUT_DIR / "reports"
VALIDATION_DIR = OUTPUT_DIR / "validation"

# =============================================================================
# Model Hyperparameters
# =============================================================================


@dataclass(frozen=True)
class GBDTConfig:
    """
    Canonical GBDT hyperparameters for CBH prediction.
    
    These values were optimized during Sprint 6 and validated to achieve:
    - R² = 0.744
    - MAE = 117.4m
    
    Do not modify without re-running validation.
    """
    n_estimators: int = 200
    max_depth: int = 8
    learning_rate: float = 0.05
    min_samples_split: int = 10
    min_samples_leaf: int = 4
    subsample: float = 0.8
    random_state: int = 42
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for sklearn model initialization."""
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "subsample": self.subsample,
            "random_state": self.random_state,
        }


@dataclass(frozen=True)
class CNNConfig:
    """CNN hyperparameters for image-based CBH prediction."""
    learning_rate: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 50
    early_stopping_patience: int = 10
    dropout_rate: float = 0.3
    weight_decay: float = 1e-4


@dataclass(frozen=True)
class ValidationConfig:
    """Cross-validation configuration."""
    n_folds: int = 5
    random_seed: int = 42
    stratified_bins: int = 10


# Default instances
GBDT_CONFIG = GBDTConfig()
CNN_CONFIG = CNNConfig()
VALIDATION_CONFIG = ValidationConfig()

# =============================================================================
# Feature Configuration
# =============================================================================

# Canonical feature set (18 features)
GEOMETRIC_FEATURES: List[str] = [
    "solar_zenith_angle",
    "solar_azimuth_angle",
    "view_zenith_angle",
    "view_azimuth_angle",
    "relative_azimuth",
]

ERA5_FEATURES: List[str] = [
    "t2m",           # 2m temperature
    "d2m",           # 2m dewpoint
    "sp",            # Surface pressure
    "tcc",           # Total cloud cover
    "lcc",           # Low cloud cover
    "mcc",           # Medium cloud cover
    "hcc",           # High cloud cover
    "u10",           # 10m U wind
    "v10",           # 10m V wind
    "blh",           # Boundary layer height
    "cape",          # CAPE
    "tcwv",          # Total column water vapor
    "skt",           # Skin temperature
]

ALL_FEATURES: List[str] = GEOMETRIC_FEATURES + ERA5_FEATURES

# =============================================================================
# Physical Constants
# =============================================================================

# CBH range constraints (km)
CBH_MIN_KM: float = 0.0
CBH_MAX_KM: float = 10.0

# Conversion factors
KM_TO_M: float = 1000.0
M_TO_KM: float = 0.001

# =============================================================================
# Reported Performance Metrics (from validated experiments)
# =============================================================================


@dataclass(frozen=True)
class ReportedMetrics:
    """
    Validated performance metrics from Sprint 6 experiments.
    
    These are the canonical values reported in the preprint.
    """
    gbdt_r2: float = 0.744
    gbdt_mae_m: float = 117.4
    gbdt_rmse_m: float = 187.3
    
    ensemble_r2: float = 0.7391
    ensemble_mae_m: float = 120.2
    
    # Uncertainty quantification
    uq_coverage_90: float = 0.91  # 90% PI coverage (conformal prediction)
    uq_mpiw_m: float = 556.6      # Mean prediction interval width


REPORTED_METRICS = ReportedMetrics()

# =============================================================================
# Utility Functions
# =============================================================================


def ensure_dirs_exist() -> None:
    """Create output directories if they don't exist."""
    for dir_path in [MODELS_DIR, FIGURES_DIR, REPORTS_DIR, VALIDATION_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)


def get_gbdt_params(random_state: Optional[int] = None) -> Dict:
    """
    Get GBDT parameters dictionary, optionally with custom random state.
    
    Args:
        random_state: Override random state (e.g., for fold-specific seeding)
        
    Returns:
        Dictionary of GBDT hyperparameters
    """
    params = GBDT_CONFIG.to_dict()
    if random_state is not None:
        params["random_state"] = random_state
    return params
