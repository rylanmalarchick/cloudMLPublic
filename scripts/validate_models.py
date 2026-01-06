#!/usr/bin/env python3
"""
CBH Model Validation Framework

Implements three validation strategies to provide honest performance metrics:

1. Per-Flight K-Fold CV (PRIMARY METRIC)
   - 5-fold CV within each flight, then average across flights
   - Accounts for temporal autocorrelation within flights
   - Expected R² ~ 0.3-0.5 for honest generalization

2. Pooled K-Fold CV (FOR COMPARISON)
   - Random 5-fold across all flights
   - Inflated by temporal autocorrelation (~0.82-0.97 lag-1)
   - Expected R² ~ 0.7-0.9 (artificially high)

3. Leave-One-Flight-Out CV (LOFO-CV)
   - Train on 4 flights, test on 1
   - Tests cross-regime generalization
   - Expected R² ~ 0.0-0.2 (documents domain shift)

References:
    - Temporal autocorrelation in atmospheric data: Wilks (2011) Ch. 8
    - Cross-validation pitfalls in time series: Bergmeir & Benitez (2012)

Author: AgentBible-assisted development
Date: 2026-01-06
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Any

import h5py
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Add src to path for custom validators
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from cbh_validators import (
    validate_no_leakage,
    validate_era5_nonzero,
    compute_temporal_autocorrelation,
    CBHValidationError,
)


@dataclass
class ValidationResult:
    """Container for validation metrics."""
    strategy: str
    r2: float
    mae: float
    rmse: float
    n_samples: int
    n_folds: int = 5
    per_fold_r2: list[float] = field(default_factory=list)
    per_flight_r2: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'strategy': self.strategy,
            'r2': self.r2,
            'mae': self.mae,
            'rmse': self.rmse,
            'n_samples': self.n_samples,
            'n_folds': self.n_folds,
            'per_fold_r2': self.per_fold_r2,
            'per_flight_r2': self.per_flight_r2,
            'warnings': self.warnings,
        }


def load_clean_dataset(
    filepath: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], dict[int, str]]:
    """Load clean dataset and validate.
    
    Args:
        filepath: Path to clean HDF5 dataset
        
    Returns:
        X: Feature matrix (n_samples, n_features)
        y: Target CBH in km (n_samples,)
        flight_ids: Flight ID per sample (n_samples,)
        feature_names: List of feature names
        flight_mapping: Dict mapping flight_id int to name
        
    Raises:
        CBHValidationError: If dataset fails validation
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset not found: {filepath}")
    
    with h5py.File(filepath, 'r') as f:
        # Load features
        feature_names = json.loads(f.attrs['feature_names'])
        
        # Validate no leakage features
        validate_no_leakage(feature_names)
        
        # Build feature matrix
        X_parts = []
        features_dict = {}
        
        for name in feature_names:
            if name in f['atmospheric_features']:
                arr = f['atmospheric_features'][name][:]
            elif name in f['geometric_features']:
                arr = f['geometric_features'][name][:]
            else:
                raise KeyError(f"Feature '{name}' not found in dataset")
            X_parts.append(arr)
            features_dict[name] = arr
        
        X = np.column_stack(X_parts)
        
        # Validate ERA5 features are non-zero
        validate_era5_nonzero(features_dict)
        
        # Load target
        y = f['metadata/cbh_km'][:]
        
        # Load flight IDs
        flight_ids = f['metadata/flight_id'][:]
        
        # Get flight mapping
        flight_mapping_str = f.attrs.get('flight_mapping', '{}')
        flight_mapping_raw = json.loads(flight_mapping_str)
        flight_mapping = {int(k): v for k, v in flight_mapping_raw.items()}
    
    print(f"Loaded {len(y)} samples with {X.shape[1]} features")
    print(f"Features: {feature_names}")
    print(f"Flights: {list(flight_mapping.values())}")
    
    return X, y, flight_ids, feature_names, flight_mapping


def create_gbdt_model(**kwargs) -> GradientBoostingRegressor:
    """Create a GBDT model with reasonable defaults.
    
    Args:
        **kwargs: Override default parameters
        
    Returns:
        Configured GradientBoostingRegressor
    """
    defaults = {
        'n_estimators': 100,
        'max_depth': 4,
        'learning_rate': 0.1,
        'min_samples_split': 10,
        'min_samples_leaf': 5,
        'subsample': 0.8,
        'random_state': 42,
    }
    defaults.update(kwargs)
    return GradientBoostingRegressor(**defaults)


def per_flight_kfold_cv(
    X: np.ndarray,
    y: np.ndarray,
    flight_ids: np.ndarray,
    flight_mapping: dict[int, str],
    n_folds: int = 5,
    model_factory: Callable = create_gbdt_model,
) -> ValidationResult:
    """Per-flight K-fold cross-validation.
    
    For each flight, perform K-fold CV within that flight's samples,
    then average metrics across all flights.
    
    This is the PRIMARY VALID METRIC because it:
    - Respects temporal structure within flights
    - Averages across different atmospheric regimes
    - Provides honest estimate of within-campaign performance
    
    Args:
        X: Feature matrix
        y: Target values
        flight_ids: Flight ID per sample
        flight_mapping: Flight ID to name mapping
        n_folds: Number of folds per flight
        model_factory: Function to create model instances
        
    Returns:
        ValidationResult with aggregated metrics
    """
    print("\n" + "="*60)
    print("PER-FLIGHT K-FOLD CROSS-VALIDATION (PRIMARY METRIC)")
    print("="*60)
    
    unique_flights = np.unique(flight_ids)
    flight_results = {}
    all_y_true = []
    all_y_pred = []
    warnings = []
    
    for fid in unique_flights:
        flight_name = flight_mapping.get(fid, f"flight_{fid}")
        mask = flight_ids == fid
        X_flight = X[mask]
        y_flight = y[mask]
        n_samples = len(y_flight)
        
        print(f"\nFlight {flight_name}: {n_samples} samples")
        
        # Check if enough samples for k-fold
        if n_samples < n_folds * 2:
            msg = f"Flight {flight_name}: only {n_samples} samples, skipping k-fold"
            print(f"  WARNING: {msg}")
            warnings.append(msg)
            continue
        
        # Compute temporal autocorrelation for this flight
        autocorr = compute_temporal_autocorrelation(y_flight)
        if autocorr.get('warning'):
            print(f"  Autocorrelation warning: lag-1 = {autocorr['lag1_autocorr']:.3f}")
        
        # K-fold within flight
        kf = KFold(n_splits=n_folds, shuffle=False)  # NO shuffle to respect time order
        fold_r2s = []
        flight_y_true = []
        flight_y_pred = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X_flight)):
            X_train, X_test = X_flight[train_idx], X_flight[test_idx]
            y_train, y_test = y_flight[train_idx], y_flight[test_idx]
            
            model = model_factory()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            fold_r2 = r2_score(y_test, y_pred)
            fold_r2s.append(fold_r2)
            
            flight_y_true.extend(y_test)
            flight_y_pred.extend(y_pred)
        
        flight_r2 = np.mean(fold_r2s)
        flight_results[flight_name] = {
            'r2': flight_r2,
            'r2_std': np.std(fold_r2s),
            'n_samples': n_samples,
            'lag1_autocorr': autocorr['lag1_autocorr'],
        }
        
        print(f"  R² = {flight_r2:.4f} +/- {np.std(fold_r2s):.4f}")
        
        all_y_true.extend(flight_y_true)
        all_y_pred.extend(flight_y_pred)
    
    # Aggregate metrics
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    
    overall_r2 = r2_score(all_y_true, all_y_pred)
    overall_mae = mean_absolute_error(all_y_true, all_y_pred)
    overall_rmse = np.sqrt(mean_squared_error(all_y_true, all_y_pred))
    
    # Average R² across flights (the proper aggregate)
    avg_r2_across_flights = np.mean([r['r2'] for r in flight_results.values()])
    
    print(f"\n--- Summary ---")
    print(f"Average R² across flights: {avg_r2_across_flights:.4f}")
    print(f"Pooled R² (all predictions): {overall_r2:.4f}")
    print(f"MAE: {overall_mae*1000:.1f} m")
    print(f"RMSE: {overall_rmse*1000:.1f} m")
    
    return ValidationResult(
        strategy='per_flight_kfold',
        r2=avg_r2_across_flights,  # Use average across flights, not pooled
        mae=overall_mae,
        rmse=overall_rmse,
        n_samples=len(all_y_true),
        n_folds=n_folds,
        per_flight_r2={k: v['r2'] for k, v in flight_results.items()},
        warnings=warnings,
    )


def pooled_kfold_cv(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    model_factory: Callable = create_gbdt_model,
) -> ValidationResult:
    """Standard random K-fold cross-validation (inflated by autocorrelation).
    
    WARNING: This metric is inflated because random shuffling mixes
    temporally adjacent samples into train/test, and high autocorrelation
    (~0.82-0.97) causes train samples to "leak" information about test samples.
    
    Include this for comparison to show the inflation effect.
    
    Args:
        X: Feature matrix
        y: Target values
        n_folds: Number of folds
        model_factory: Function to create model instances
        
    Returns:
        ValidationResult with metrics (expect ~0.7-0.9 R²)
    """
    print("\n" + "="*60)
    print("POOLED K-FOLD CROSS-VALIDATION (INFLATED BY AUTOCORRELATION)")
    print("="*60)
    print("WARNING: This metric is artificially high due to temporal autocorrelation")
    
    # Compute overall autocorrelation
    autocorr = compute_temporal_autocorrelation(y)
    print(f"Lag-1 autocorrelation: {autocorr['lag1_autocorr']:.3f}")
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    all_y_true = []
    all_y_pred = []
    fold_r2s = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        model = model_factory()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        fold_r2 = r2_score(y_test, y_pred)
        fold_r2s.append(fold_r2)
        
        print(f"  Fold {fold_idx+1}: R² = {fold_r2:.4f}")
        
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
    
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    
    overall_r2 = np.mean(fold_r2s)
    overall_mae = mean_absolute_error(all_y_true, all_y_pred)
    overall_rmse = np.sqrt(mean_squared_error(all_y_true, all_y_pred))
    
    print(f"\n--- Summary ---")
    print(f"Mean R² across folds: {overall_r2:.4f} +/- {np.std(fold_r2s):.4f}")
    print(f"MAE: {overall_mae*1000:.1f} m")
    print(f"RMSE: {overall_rmse*1000:.1f} m")
    print(f"\nCAUTION: Compare to per-flight CV to see autocorrelation inflation")
    
    return ValidationResult(
        strategy='pooled_kfold',
        r2=overall_r2,
        mae=overall_mae,
        rmse=overall_rmse,
        n_samples=len(all_y_true),
        n_folds=n_folds,
        per_fold_r2=fold_r2s,
        warnings=[
            f"Inflated by temporal autocorrelation (lag-1 = {autocorr['lag1_autocorr']:.3f})",
            "Do not use this as primary metric",
        ],
    )


def lofo_cv(
    X: np.ndarray,
    y: np.ndarray,
    flight_ids: np.ndarray,
    flight_mapping: dict[int, str],
    model_factory: Callable = create_gbdt_model,
) -> ValidationResult:
    """Leave-One-Flight-Out Cross-Validation.
    
    Tests cross-regime generalization by training on all flights
    except one and testing on the held-out flight.
    
    Expected to fail (R² ~ 0.0-0.2 or negative) due to:
    - Domain shift between flights (different atmospheric regimes)
    - Limited number of flights (5)
    - ERA5 features may not transfer across regimes
    
    This documents the generalization limitation.
    
    Args:
        X: Feature matrix
        y: Target values
        flight_ids: Flight ID per sample
        flight_mapping: Flight ID to name mapping
        model_factory: Function to create model instances
        
    Returns:
        ValidationResult with metrics (expect poor performance)
    """
    print("\n" + "="*60)
    print("LEAVE-ONE-FLIGHT-OUT CROSS-VALIDATION (DOMAIN SHIFT TEST)")
    print("="*60)
    print("NOTE: Poor performance expected due to atmospheric regime differences")
    
    unique_flights = np.unique(flight_ids)
    flight_results = {}
    all_y_true = []
    all_y_pred = []
    
    for test_fid in unique_flights:
        flight_name = flight_mapping.get(test_fid, f"flight_{test_fid}")
        
        # Train on all other flights
        train_mask = flight_ids != test_fid
        test_mask = flight_ids == test_fid
        
        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
        
        print(f"\nTest flight: {flight_name}")
        print(f"  Train samples: {len(y_train)}, Test samples: {len(y_test)}")
        
        model = model_factory()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        flight_r2 = r2_score(y_test, y_pred)
        flight_mae = mean_absolute_error(y_test, y_pred)
        
        flight_results[flight_name] = {
            'r2': flight_r2,
            'mae': flight_mae,
            'n_test': len(y_test),
        }
        
        print(f"  R² = {flight_r2:.4f}, MAE = {flight_mae*1000:.1f} m")
        
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
    
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    
    # Average R² across held-out flights
    avg_r2 = np.mean([r['r2'] for r in flight_results.values()])
    overall_mae = mean_absolute_error(all_y_true, all_y_pred)
    overall_rmse = np.sqrt(mean_squared_error(all_y_true, all_y_pred))
    
    print(f"\n--- Summary ---")
    print(f"Average R² across held-out flights: {avg_r2:.4f}")
    print(f"Pooled MAE: {overall_mae*1000:.1f} m")
    print(f"Pooled RMSE: {overall_rmse*1000:.1f} m")
    
    # Interpretation
    if avg_r2 < 0:
        print("\nINTERPRETATION: Negative R² indicates model is worse than predicting mean.")
        print("This demonstrates severe domain shift between flights.")
    elif avg_r2 < 0.2:
        print("\nINTERPRETATION: Very low R² indicates poor cross-regime generalization.")
    
    return ValidationResult(
        strategy='lofo_cv',
        r2=avg_r2,
        mae=overall_mae,
        rmse=overall_rmse,
        n_samples=len(all_y_true),
        n_folds=len(unique_flights),
        per_flight_r2={k: v['r2'] for k, v in flight_results.items()},
        warnings=[
            "LOFO-CV tests cross-regime generalization (expected to fail)",
            "Domain shift between atmospheric regimes limits transferability",
        ],
    )


def run_all_validations(
    dataset_path: str | Path,
    output_dir: str | Path | None = None,
) -> dict[str, ValidationResult]:
    """Run all three validation strategies and save results.
    
    Args:
        dataset_path: Path to clean HDF5 dataset
        output_dir: Directory to save results (default: outputs/validation/)
        
    Returns:
        Dictionary of validation strategy -> ValidationResult
    """
    # Set up output directory
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "outputs" / "validation"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    X, y, flight_ids, feature_names, flight_mapping = load_clean_dataset(dataset_path)
    
    # Run all validations
    results = {}
    
    results['per_flight_kfold'] = per_flight_kfold_cv(
        X, y, flight_ids, flight_mapping
    )
    
    results['pooled_kfold'] = pooled_kfold_cv(X, y)
    
    results['lofo_cv'] = lofo_cv(X, y, flight_ids, flight_mapping)
    
    # Print summary comparison
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    print(f"{'Strategy':<25} {'R²':>8} {'MAE (m)':>10} {'RMSE (m)':>10}")
    print("-"*60)
    for name, res in results.items():
        print(f"{name:<25} {res.r2:>8.4f} {res.mae*1000:>10.1f} {res.rmse*1000:>10.1f}")
    
    print("\n" + "-"*60)
    print("INTERPRETATION:")
    print("-"*60)
    
    inflation = results['pooled_kfold'].r2 - results['per_flight_kfold'].r2
    print(f"1. Autocorrelation inflation: +{inflation:.3f} R²")
    print(f"   (pooled - per_flight = {results['pooled_kfold'].r2:.3f} - {results['per_flight_kfold'].r2:.3f})")
    
    print(f"\n2. Domain shift penalty: {results['lofo_cv'].r2:.3f} R²")
    print(f"   Cross-regime generalization is {'poor' if results['lofo_cv'].r2 < 0.2 else 'moderate'}")
    
    print(f"\n3. PRIMARY METRIC: Per-flight K-fold R² = {results['per_flight_kfold'].r2:.4f}")
    print(f"   This is the honest estimate of within-campaign performance.")
    
    # Save results
    results_dict = {k: v.to_dict() for k, v in results.items()}
    results_dict['metadata'] = {
        'dataset': str(dataset_path),
        'n_samples': int(X.shape[0]),
        'n_features': int(X.shape[1]),
        'feature_names': feature_names,
        'flights': list(flight_mapping.values()),
    }
    
    output_file = output_dir / "validation_results.json"
    with open(output_file, 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f"\nResults saved to: {output_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run CBH model validation with multiple strategies"
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='outputs/preprocessed_data/Clean_933_Integrated_Features.hdf5',
        help='Path to clean HDF5 dataset',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for results',
    )
    
    args = parser.parse_args()
    
    run_all_validations(args.dataset, args.output_dir)


if __name__ == '__main__':
    main()
