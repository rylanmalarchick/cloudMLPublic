#!/usr/bin/env python3
"""
CBH-Specific Validators using AgentBible Framework

This module provides physics-aware validation for Cloud Base Height (CBH)
retrieval, building on the AgentBible core validators.

Validators ensure:
1. CBH values are physically plausible (0-10 km typical)
2. ERA5 features are non-zero (catch placeholder data)
3. No data leakage features are present
4. LCL-CBH consistency (LCL typically < CBH for marine stratocumulus)

Reference:
    Wood, R. (2012). "Stratocumulus Clouds." Monthly Weather Review, 140(8), 2373-2423.
    doi:10.1175/MWR-D-11-00121.1

Author: AgentBible-assisted development
Date: 2026-01-06
"""

from __future__ import annotations

import functools
from typing import Callable, Any

import numpy as np

from agentbible import validate_finite, validate_range, validate_non_negative
from agentbible.errors import ValidationError


class CBHValidationError(ValidationError):
    """Error raised when CBH-specific validation fails."""
    pass


class DataLeakageError(CBHValidationError):
    """Error raised when potential data leakage is detected."""
    pass


class ERA5DataError(CBHValidationError):
    """Error raised when ERA5 data appears invalid (e.g., all zeros)."""
    pass


# Physical constants for CBH validation
CBH_MIN_KM = 0.0      # Minimum valid CBH (fog at surface)
CBH_MAX_KM = 10.0     # Maximum valid CBH (high clouds)
CBH_PAPER_MIN_KM = 0.2  # Paper filter: minimum CBH
CBH_PAPER_MAX_KM = 2.0  # Paper filter: maximum CBH (marine boundary layer)

# Leakage features that must NOT be used
LEAKAGE_FEATURES = {'inversion_height', 'cbh_derived', 'target_encoded'}


def validate_cbh_range(
    min_km: float = CBH_MIN_KM,
    max_km: float = CBH_MAX_KM,
) -> Callable:
    """Decorator to validate CBH predictions are within physical bounds.
    
    Args:
        min_km: Minimum valid CBH in km (default: 0.0)
        max_km: Maximum valid CBH in km (default: 10.0)
        
    Returns:
        Decorator function
        
    Reference:
        Marine stratocumulus typically 0.3-1.5 km (Wood, 2012)
        Deep convection can reach 10+ km but rare in this dataset
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = func(*args, **kwargs)
            
            if isinstance(result, np.ndarray):
                if not np.all(np.isfinite(result)):
                    nan_count = np.sum(np.isnan(result))
                    inf_count = np.sum(np.isinf(result))
                    raise CBHValidationError(
                        f"CBH predictions contain non-finite values: "
                        f"{nan_count} NaN, {inf_count} Inf"
                    )
                
                out_of_range = (result < min_km) | (result > max_km)
                if np.any(out_of_range):
                    n_invalid = np.sum(out_of_range)
                    actual_min = np.min(result)
                    actual_max = np.max(result)
                    raise CBHValidationError(
                        f"CBH predictions out of range [{min_km}, {max_km}] km\n"
                        f"  Got: {n_invalid} values outside range\n"
                        f"  Actual range: [{actual_min:.3f}, {actual_max:.3f}] km\n"
                        f"  Function: {func.__name__}\n\n"
                        f"  Reference: Marine stratocumulus CBH typically 0.3-1.5 km\n"
                        f"  Guidance: Check for unit conversion errors (m vs km)"
                    )
            
            return result
        return wrapper
    return decorator


def validate_no_leakage(feature_names: list[str]) -> None:
    """Check that no leakage features are present in the feature set.
    
    Args:
        feature_names: List of feature names to check
        
    Raises:
        DataLeakageError: If any leakage features are found
        
    Note:
        The most common leakage in CBH prediction is:
        - inversion_height = CBH - BLH (directly encodes target)
        - Any feature derived from CBH
    """
    found_leakage = set(feature_names) & LEAKAGE_FEATURES
    if found_leakage:
        raise DataLeakageError(
            f"DATA LEAKAGE DETECTED\n\n"
            f"  Forbidden features found: {found_leakage}\n\n"
            f"  Why this matters:\n"
            f"    'inversion_height' is computed as CBH - BLH, which directly\n"
            f"    encodes the target variable. This caused R² = 0.999 in the\n"
            f"    original study, making results meaningless.\n\n"
            f"  Solution: Remove these features before training."
        )


def validate_era5_nonzero(
    features: dict[str, np.ndarray],
    min_variance: float = 1e-6,
) -> None:
    """Validate that ERA5 features are not placeholder zeros.
    
    Args:
        features: Dictionary mapping feature names to arrays
        min_variance: Minimum required variance (default: 1e-6)
        
    Raises:
        ERA5DataError: If any ERA5 feature has zero variance
        
    Note:
        The original study had ERA5 features that were all zeros
        because the integration script never ran. This validator
        catches that bug early.
    """
    era5_vars = ['blh', 't2m', 'd2m', 'sp', 'tcwv']
    
    for var in era5_vars:
        if var in features:
            data = features[var]
            variance = np.var(data)
            
            if variance < min_variance:
                raise ERA5DataError(
                    f"ERA5 FEATURE HAS ZERO VARIANCE\n\n"
                    f"  Feature: {var}\n"
                    f"  Variance: {variance:.2e}\n"
                    f"  Min/Max: [{np.min(data):.2f}, {np.max(data):.2f}]\n\n"
                    f"  Why this matters:\n"
                    f"    Zero-variance features indicate placeholder data that\n"
                    f"    was never populated with real ERA5 values.\n\n"
                    f"  Solution: Run the ERA5 integration script or check\n"
                    f"    that ERA5 files exist for all flight dates."
                )


def validate_lcl_cbh_consistency(
    lcl_m: np.ndarray,
    cbh_m: np.ndarray,
    tolerance: float = 0.1,
) -> dict[str, Any]:
    """Check LCL-CBH consistency for marine stratocumulus.
    
    For marine stratocumulus (the dominant cloud type in this dataset),
    LCL should typically be below CBH. Large discrepancies may indicate
    data quality issues.
    
    Args:
        lcl_m: Lifting Condensation Level in meters
        cbh_m: Cloud Base Height in meters
        tolerance: Fraction of samples allowed to violate (default: 10%)
        
    Returns:
        Dictionary with consistency metrics
        
    Reference:
        For marine Sc, cloud base is typically at or slightly above LCL.
        Large LCL > CBH indicates either:
        - Dry air above surface (decoupled boundary layer)
        - Data quality issues
        
    Note:
        This is a soft check (returns metrics) rather than raising an error,
        because LCL > CBH can occur in legitimate decoupled cases.
    """
    lcl_above_cbh = lcl_m > cbh_m
    fraction_above = np.mean(lcl_above_cbh)
    
    # Compute correlation
    valid_mask = np.isfinite(lcl_m) & np.isfinite(cbh_m)
    if np.sum(valid_mask) > 10:
        correlation = np.corrcoef(lcl_m[valid_mask], cbh_m[valid_mask])[0, 1]
    else:
        correlation = np.nan
    
    return {
        'lcl_above_cbh_fraction': fraction_above,
        'lcl_cbh_correlation': correlation,
        'n_samples': len(lcl_m),
        'is_consistent': fraction_above < tolerance,
        'warning': (
            f"LCL > CBH for {fraction_above*100:.1f}% of samples. "
            f"Expected < {tolerance*100:.0f}% for coupled marine Sc."
            if fraction_above >= tolerance else None
        ),
    }


def compute_temporal_autocorrelation(
    cbh: np.ndarray,
    timestamps: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute temporal autocorrelation of CBH time series.
    
    High autocorrelation (> 0.8) inflates random K-fold CV metrics
    because train/test samples are not independent.
    
    Args:
        cbh: CBH values (assumed ordered by time)
        timestamps: Optional timestamps to verify ordering
        
    Returns:
        Dictionary with autocorrelation metrics
        
    Reference:
        For CBH data at ~8-second intervals, lag-1 autocorrelation
        is typically ~0.92, which significantly inflates R² when
        using random train/test splits.
    """
    if len(cbh) < 10:
        return {'lag1_autocorr': np.nan, 'warning': 'Too few samples'}
    
    # Lag-1 autocorrelation
    lag1 = np.corrcoef(cbh[:-1], cbh[1:])[0, 1]
    
    # Lag-5 autocorrelation (for comparison)
    lag5 = np.corrcoef(cbh[:-5], cbh[5:])[0, 1] if len(cbh) > 10 else np.nan
    
    warning = None
    if lag1 > 0.8:
        warning = (
            f"High temporal autocorrelation (lag-1 = {lag1:.3f}). "
            f"Random K-fold CV will overestimate performance. "
            f"Use per-flight CV or time-series CV instead."
        )
    
    return {
        'lag1_autocorr': lag1,
        'lag5_autocorr': lag5,
        'n_samples': len(cbh),
        'warning': warning,
    }


# Convenience decorator combining common validations
def validate_cbh_prediction(func: Callable) -> Callable:
    """Combined decorator for CBH prediction functions.
    
    Applies:
    1. validate_finite (no NaN/Inf)
    2. validate_cbh_range (0-10 km)
    """
    @functools.wraps(func)
    @validate_finite
    @validate_cbh_range()
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)
    return wrapper
