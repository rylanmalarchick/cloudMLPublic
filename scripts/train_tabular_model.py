#!/usr/bin/env python3
"""
Tabular GBDT Model Training for CBH Retrieval

This script trains GBDT models with proper validation strategies and generates
honest performance metrics. It implements:

1. Multiple validation strategies to document inflation effects
2. Feature importance analysis
3. Scatter plots (true vs predicted)
4. Saved model and metrics

Key findings from validation framework:
- Pooled K-fold R² ~ 0.92 (inflated by temporal autocorrelation)
- Per-flight K-fold R² ~ -0.7 to -1.0 (honest but strict)
- LOFO R² << 0 (severe domain shift)

The honest conclusion: ERA5 features alone do NOT generalize well for CBH
prediction, even within the same campaign.

Author: AgentBible-assisted development
Date: 2026-01-06
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from cbh_validators import (
    validate_no_leakage,
    validate_era5_nonzero,
    compute_temporal_autocorrelation,
)


def load_dataset(filepath: str | Path) -> dict[str, Any]:
    """Load clean dataset with all metadata."""
    filepath = Path(filepath)
    
    with h5py.File(filepath, 'r') as f:
        feature_names = json.loads(f.attrs['feature_names'])
        validate_no_leakage(feature_names)
        
        X_parts = []
        features_dict = {}
        
        for name in feature_names:
            if name in f['atmospheric_features']:
                arr = f['atmospheric_features'][name][:]
            elif name in f['geometric_features']:
                arr = f['geometric_features'][name][:]
            else:
                raise KeyError(f"Feature '{name}' not found")
            X_parts.append(arr)
            features_dict[name] = arr
        
        X = np.column_stack(X_parts)
        validate_era5_nonzero(features_dict)
        
        y = f['metadata/cbh_km'][:]
        flight_ids = f['metadata/flight_id'][:]
        
        flight_mapping_str = f.attrs.get('flight_mapping', '{}')
        flight_mapping = {int(k): v for k, v in json.loads(flight_mapping_str).items()}
    
    return {
        'X': X,
        'y': y,
        'flight_ids': flight_ids,
        'feature_names': feature_names,
        'flight_mapping': flight_mapping,
    }


def train_and_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    flight_ids: np.ndarray,
    flight_mapping: dict[int, str],
    feature_names: list[str],
    output_dir: Path,
) -> dict[str, Any]:
    """Train GBDT and evaluate with multiple strategies."""
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'n_samples': len(y),
        'n_features': X.shape[1],
        'feature_names': feature_names,
    }
    
    # ========================================
    # Strategy 1: Pooled K-fold (for comparison)
    # ========================================
    print("\n" + "="*60)
    print("STRATEGY 1: Pooled K-fold (inflated by autocorrelation)")
    print("="*60)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    pooled_y_true, pooled_y_pred = [], []
    
    for train_idx, test_idx in kf.split(X):
        model = GradientBoostingRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            min_samples_split=10, min_samples_leaf=5, random_state=42
        )
        model.fit(X[train_idx], y[train_idx])
        pooled_y_pred.extend(model.predict(X[test_idx]))
        pooled_y_true.extend(y[test_idx])
    
    pooled_y_true = np.array(pooled_y_true)
    pooled_y_pred = np.array(pooled_y_pred)
    
    results['pooled_kfold'] = {
        'r2': float(r2_score(pooled_y_true, pooled_y_pred)),
        'mae_km': float(mean_absolute_error(pooled_y_true, pooled_y_pred)),
        'rmse_km': float(np.sqrt(mean_squared_error(pooled_y_true, pooled_y_pred))),
        'warning': 'INFLATED by temporal autocorrelation - do not use as primary metric',
    }
    print(f"R² = {results['pooled_kfold']['r2']:.4f}")
    print(f"MAE = {results['pooled_kfold']['mae_km']*1000:.1f} m")
    
    # ========================================
    # Strategy 2: Per-flight K-fold (shuffled within flight)
    # ========================================
    print("\n" + "="*60)
    print("STRATEGY 2: Per-flight K-fold (shuffled within flight)")
    print("="*60)
    
    unique_flights = np.unique(flight_ids)
    flight_results = {}
    shuffled_y_true, shuffled_y_pred = [], []
    
    for fid in unique_flights:
        flight_name = flight_mapping.get(fid, f"flight_{fid}")
        mask = flight_ids == fid
        X_f, y_f = X[mask], y[mask]
        
        if len(y_f) < 20:
            print(f"  {flight_name}: skipped ({len(y_f)} samples)")
            continue
        
        kf = KFold(n_splits=5, shuffle=True, random_state=42)  # SHUFFLED
        f_true, f_pred = [], []
        
        for train_idx, test_idx in kf.split(X_f):
            model = GradientBoostingRegressor(
                n_estimators=100, max_depth=4, learning_rate=0.1,
                min_samples_split=10, min_samples_leaf=5, random_state=42
            )
            model.fit(X_f[train_idx], y_f[train_idx])
            f_pred.extend(model.predict(X_f[test_idx]))
            f_true.extend(y_f[test_idx])
        
        f_r2 = r2_score(f_true, f_pred)
        flight_results[flight_name] = f_r2
        shuffled_y_true.extend(f_true)
        shuffled_y_pred.extend(f_pred)
        print(f"  {flight_name}: R² = {f_r2:.4f} ({len(y_f)} samples)")
    
    shuffled_y_true = np.array(shuffled_y_true)
    shuffled_y_pred = np.array(shuffled_y_pred)
    
    avg_r2_shuffled = np.mean(list(flight_results.values()))
    results['per_flight_shuffled'] = {
        'r2': float(avg_r2_shuffled),
        'mae_km': float(mean_absolute_error(shuffled_y_true, shuffled_y_pred)),
        'rmse_km': float(np.sqrt(mean_squared_error(shuffled_y_true, shuffled_y_pred))),
        'per_flight_r2': flight_results,
        'note': 'Shuffled within-flight CV - moderate autocorrelation leakage',
    }
    print(f"\nAverage R² across flights: {avg_r2_shuffled:.4f}")
    
    # ========================================
    # Strategy 3: Per-flight K-fold (time-ordered, no shuffle)
    # ========================================
    print("\n" + "="*60)
    print("STRATEGY 3: Per-flight K-fold (time-ordered, strict)")
    print("="*60)
    
    flight_results_strict = {}
    strict_y_true, strict_y_pred = [], []
    
    for fid in unique_flights:
        flight_name = flight_mapping.get(fid, f"flight_{fid}")
        mask = flight_ids == fid
        X_f, y_f = X[mask], y[mask]
        
        if len(y_f) < 20:
            continue
        
        kf = KFold(n_splits=5, shuffle=False)  # NO SHUFFLE - time-ordered
        f_true, f_pred = [], []
        
        for train_idx, test_idx in kf.split(X_f):
            model = GradientBoostingRegressor(
                n_estimators=100, max_depth=4, learning_rate=0.1,
                min_samples_split=10, min_samples_leaf=5, random_state=42
            )
            model.fit(X_f[train_idx], y_f[train_idx])
            f_pred.extend(model.predict(X_f[test_idx]))
            f_true.extend(y_f[test_idx])
        
        f_r2 = r2_score(f_true, f_pred)
        flight_results_strict[flight_name] = f_r2
        strict_y_true.extend(f_true)
        strict_y_pred.extend(f_pred)
        print(f"  {flight_name}: R² = {f_r2:.4f}")
    
    strict_y_true = np.array(strict_y_true)
    strict_y_pred = np.array(strict_y_pred)
    
    avg_r2_strict = np.mean(list(flight_results_strict.values()))
    results['per_flight_strict'] = {
        'r2': float(avg_r2_strict),
        'mae_km': float(mean_absolute_error(strict_y_true, strict_y_pred)),
        'rmse_km': float(np.sqrt(mean_squared_error(strict_y_true, strict_y_pred))),
        'per_flight_r2': flight_results_strict,
        'note': 'Time-ordered CV - no autocorrelation leakage, but strict',
    }
    print(f"\nAverage R² across flights: {avg_r2_strict:.4f}")
    
    # ========================================
    # Strategy 4: LOFO-CV
    # ========================================
    print("\n" + "="*60)
    print("STRATEGY 4: Leave-One-Flight-Out (domain shift test)")
    print("="*60)
    
    lofo_results = {}
    lofo_y_true, lofo_y_pred = [], []
    
    for test_fid in unique_flights:
        flight_name = flight_mapping.get(test_fid, f"flight_{test_fid}")
        train_mask = flight_ids != test_fid
        test_mask = flight_ids == test_fid
        
        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
        
        model = GradientBoostingRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            min_samples_split=10, min_samples_leaf=5, random_state=42
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        f_r2 = r2_score(y_test, y_pred)
        lofo_results[flight_name] = f_r2
        lofo_y_true.extend(y_test)
        lofo_y_pred.extend(y_pred)
        print(f"  Test on {flight_name}: R² = {f_r2:.4f}")
    
    lofo_y_true = np.array(lofo_y_true)
    lofo_y_pred = np.array(lofo_y_pred)
    
    avg_r2_lofo = np.mean(list(lofo_results.values()))
    results['lofo_cv'] = {
        'r2': float(avg_r2_lofo),
        'mae_km': float(mean_absolute_error(lofo_y_true, lofo_y_pred)),
        'rmse_km': float(np.sqrt(mean_squared_error(lofo_y_true, lofo_y_pred))),
        'per_flight_r2': lofo_results,
        'note': 'Tests cross-regime generalization - expected to fail',
    }
    print(f"\nAverage R² across held-out flights: {avg_r2_lofo:.4f}")
    
    # ========================================
    # Train final model on all data (for feature importance)
    # ========================================
    print("\n" + "="*60)
    print("Training final model on all data (for analysis)")
    print("="*60)
    
    final_model = GradientBoostingRegressor(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        min_samples_split=10, min_samples_leaf=5, random_state=42
    )
    final_model.fit(X, y)
    
    # Feature importance
    importance = final_model.feature_importances_
    importance_dict = {name: float(imp) for name, imp in zip(feature_names, importance)}
    sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    
    print("\nFeature Importance:")
    for name, imp in sorted_importance:
        print(f"  {name}: {imp:.4f}")
    
    results['feature_importance'] = importance_dict
    
    # Save model
    model_path = output_dir / 'gbdt_model.joblib'
    joblib.dump(final_model, model_path)
    print(f"\nModel saved to: {model_path}")
    
    # ========================================
    # Generate scatter plots
    # ========================================
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Pooled K-fold
    ax = axes[0, 0]
    ax.scatter(pooled_y_true, pooled_y_pred, alpha=0.3, s=10)
    ax.plot([0, max(pooled_y_true.max(), pooled_y_pred.max())],
            [0, max(pooled_y_true.max(), pooled_y_pred.max())], 'r--', lw=2)
    ax.set_xlabel('True CBH (km)')
    ax.set_ylabel('Predicted CBH (km)')
    ax.set_title(f"Pooled K-fold (INFLATED)\nR² = {results['pooled_kfold']['r2']:.3f}")
    ax.set_aspect('equal', adjustable='box')
    
    # Per-flight shuffled
    ax = axes[0, 1]
    ax.scatter(shuffled_y_true, shuffled_y_pred, alpha=0.3, s=10)
    ax.plot([0, max(shuffled_y_true.max(), shuffled_y_pred.max())],
            [0, max(shuffled_y_true.max(), shuffled_y_pred.max())], 'r--', lw=2)
    ax.set_xlabel('True CBH (km)')
    ax.set_ylabel('Predicted CBH (km)')
    ax.set_title(f"Per-flight (shuffled)\nR² = {results['per_flight_shuffled']['r2']:.3f}")
    ax.set_aspect('equal', adjustable='box')
    
    # Per-flight strict
    ax = axes[1, 0]
    ax.scatter(strict_y_true, strict_y_pred, alpha=0.3, s=10)
    ax.plot([0, max(strict_y_true.max(), strict_y_pred.max())],
            [0, max(strict_y_true.max(), strict_y_pred.max())], 'r--', lw=2)
    ax.set_xlabel('True CBH (km)')
    ax.set_ylabel('Predicted CBH (km)')
    ax.set_title(f"Per-flight (time-ordered)\nR² = {results['per_flight_strict']['r2']:.3f}")
    ax.set_aspect('equal', adjustable='box')
    
    # LOFO
    ax = axes[1, 1]
    ax.scatter(lofo_y_true, lofo_y_pred, alpha=0.3, s=10)
    ax.plot([0, max(lofo_y_true.max(), lofo_y_pred.max())],
            [0, max(lofo_y_true.max(), lofo_y_pred.max())], 'r--', lw=2)
    ax.set_xlabel('True CBH (km)')
    ax.set_ylabel('Predicted CBH (km)')
    ax.set_title(f"LOFO-CV (domain shift)\nR² = {results['lofo_cv']['r2']:.3f}")
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    fig_path = output_dir / 'validation_comparison.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figure saved to: {fig_path}")
    
    # Feature importance bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    names, values = zip(*sorted_importance)
    y_pos = np.arange(len(names))
    ax.barh(y_pos, values, color='steelblue')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel('Feature Importance')
    ax.set_title('GBDT Feature Importance (10 features, NO inversion_height)')
    plt.tight_layout()
    fig_path = output_dir / 'feature_importance.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Feature importance plot saved to: {fig_path}")
    
    return results


def print_summary(results: dict[str, Any]):
    """Print summary comparison table."""
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    print(f"{'Strategy':<30} {'R²':>10} {'MAE (m)':>12} {'RMSE (m)':>12}")
    print("-"*70)
    
    strategies = [
        ('pooled_kfold', 'Pooled K-fold (INFLATED)'),
        ('per_flight_shuffled', 'Per-flight (shuffled)'),
        ('per_flight_strict', 'Per-flight (time-ordered)'),
        ('lofo_cv', 'LOFO-CV (domain shift)'),
    ]
    
    for key, label in strategies:
        if key in results:
            r = results[key]
            print(f"{label:<30} {r['r2']:>10.4f} {r['mae_km']*1000:>12.1f} {r['rmse_km']*1000:>12.1f}")
    
    print("\n" + "-"*70)
    print("INTERPRETATION:")
    print("-"*70)
    print("""
1. POOLED K-FOLD is INFLATED by temporal autocorrelation.
   - Random shuffling mixes adjacent samples into train/test
   - Lag-1 autocorrelation ~0.94 causes artificial performance boost

2. PER-FLIGHT SHUFFLED is MODERATELY INFLATED.
   - Shuffling within flight still mixes temporally adjacent samples
   - Better than pooled but still optimistic

3. PER-FLIGHT TIME-ORDERED is HONEST but STRICT.
   - Trains on one time segment, tests on another
   - Model must extrapolate to unseen time periods
   - Negative R² means model is worse than predicting mean

4. LOFO-CV shows SEVERE DOMAIN SHIFT.
   - Training on other flights does not transfer
   - Each flight has different atmospheric regime

CONCLUSION: ERA5 features alone do NOT generalize well for CBH prediction.
The original paper's R² ~ 0.74 was likely inflated by data leakage
(inversion_height) AND temporal autocorrelation (pooled K-fold).
""")


def main():
    parser = argparse.ArgumentParser(
        description="Train GBDT model for CBH with multiple validation strategies"
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
        default='outputs/tabular_model',
        help='Output directory',
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading dataset...")
    data = load_dataset(args.dataset)
    print(f"Loaded {len(data['y'])} samples with {data['X'].shape[1]} features")
    
    # Train and evaluate
    results = train_and_evaluate(
        X=data['X'],
        y=data['y'],
        flight_ids=data['flight_ids'],
        flight_mapping=data['flight_mapping'],
        feature_names=data['feature_names'],
        output_dir=output_dir,
    )
    
    # Print summary
    print_summary(results)
    
    # Save results
    results_path = output_dir / 'training_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")


if __name__ == '__main__':
    main()
