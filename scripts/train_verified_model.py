#!/usr/bin/env python3
"""
Train GBDT Model on Verified Dataset and Generate Metrics

This script:
1. Loads the verified dataset
2. Trains GBDT with 5-fold stratified CV
3. Generates validation metrics
4. Creates figures
5. Saves reports

Author: Automated rebuild
Date: 2026-01-06
"""

import h5py
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy import stats
import matplotlib.pyplot as plt

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parents[1]
VERIFIED_DATA = PROJECT_ROOT / "outputs/preprocessed_data/Verified_Integrated_Features.hdf5"
REPORTS_DIR = PROJECT_ROOT / "results/cbh/reports"
FIGURES_DIR = PROJECT_ROOT / "results/cbh/figures"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def load_verified_data():
    """Load verified dataset."""
    print("\n" + "=" * 80)
    print("Loading Verified Dataset")
    print("=" * 80)
    
    with h5py.File(VERIFIED_DATA, 'r') as f:
        cbh_km = f['metadata/cbh_km'][:]
        flight_ids = f['metadata/flight_id'][:]
        
        # Atmospheric features
        # NOTE: inversion_height is EXCLUDED because it's computed as (CBH - BLH),
        # which directly encodes the target variable and causes data leakage
        atmo_names = ['t2m', 'd2m', 'blh', 'sp', 'tcwv', 'lcl', 
                      'stability_index', 'moisture_gradient']
        atmo_features = []
        for name in atmo_names:
            if f'atmospheric_features/{name}' in f:
                atmo_features.append(f[f'atmospheric_features/{name}'][:])
        
        # Geometric features
        geo_names = ['sza_deg', 'saa_deg']
        geo_features = []
        for name in geo_names:
            if f'geometric_features/{name}' in f:
                geo_features.append(f[f'geometric_features/{name}'][:])
        
        X_atmo = np.column_stack(atmo_features)
        X_geo = np.column_stack(geo_features)
        X = np.concatenate([X_atmo, X_geo], axis=1)
        
        feature_names = atmo_names + geo_names
        y = cbh_km * 1000  # Convert to meters
    
    print(f"  Samples: {len(y)}")
    print(f"  Features: {X.shape[1]}")
    print(f"  CBH range: [{y.min():.0f}, {y.max():.0f}] m")
    print(f"  Feature names: {feature_names}")
    
    return X, y, flight_ids, feature_names

def create_stratified_bins(y, n_bins=10):
    """Create stratified bins for CBH values."""
    bins = np.percentile(y, np.linspace(0, 100, n_bins + 1))
    bins = np.unique(bins)
    bin_indices = np.digitize(y, bins) - 1
    bin_indices = np.clip(bin_indices, 0, len(bins) - 2)
    return bin_indices

def train_and_evaluate():
    """Train GBDT with 5-fold CV and compute metrics."""
    X, y, flight_ids, feature_names = load_verified_data()
    
    print("\n" + "=" * 80)
    print("Training GBDT with 5-Fold Stratified CV")
    print("=" * 80)
    
    # Create stratified bins
    y_bins = create_stratified_bins(y)
    
    # Initialize
    n_folds = 5
    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    fold_results = []
    all_y_true = []
    all_y_pred = []
    all_fold_idx = []
    feature_importance_sum = np.zeros(len(feature_names))
    
    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X, y_bins)):
        print(f"\n  Fold {fold_idx + 1}/{n_folds}:")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Train GBDT
        model = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=8,
            min_samples_split=10,
            min_samples_leaf=4,
            subsample=0.8,
            random_state=42,
            verbose=0
        )
        model.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred = model.predict(X_val_scaled)
        
        # Metrics
        r2 = r2_score(y_val, y_pred)
        mae = mean_absolute_error(y_val, y_pred)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        
        print(f"    R² = {r2:.4f}, MAE = {mae:.1f} m, RMSE = {rmse:.1f} m")
        
        fold_results.append({
            'fold': fold_idx,
            'r2': r2,
            'mae_m': mae,
            'rmse_m': rmse,
            'n_train': len(train_idx),
            'n_val': len(val_idx),
        })
        
        all_y_true.extend(y_val.tolist())
        all_y_pred.extend(y_pred.tolist())
        all_fold_idx.extend([fold_idx] * len(val_idx))
        
        feature_importance_sum += model.feature_importances_
    
    # Aggregate metrics
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    
    mean_metrics = {
        'r2': np.mean([f['r2'] for f in fold_results]),
        'mae_m': np.mean([f['mae_m'] for f in fold_results]),
        'rmse_m': np.mean([f['rmse_m'] for f in fold_results]),
    }
    std_metrics = {
        'r2': np.std([f['r2'] for f in fold_results]),
        'mae_m': np.std([f['mae_m'] for f in fold_results]),
        'rmse_m': np.std([f['rmse_m'] for f in fold_results]),
    }
    
    # Feature importance
    mean_importance = feature_importance_sum / n_folds
    feature_importance = {
        'feature_names': feature_names,
        'mean_importance': mean_importance.tolist(),
    }
    
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"\nMean Metrics (5-fold CV):")
    print(f"  R² = {mean_metrics['r2']:.4f} ± {std_metrics['r2']:.4f}")
    print(f"  MAE = {mean_metrics['mae_m']:.1f} ± {std_metrics['mae_m']:.1f} m")
    print(f"  RMSE = {mean_metrics['rmse_m']:.1f} ± {std_metrics['rmse_m']:.1f} m")
    
    print(f"\nFeature Importance (top 5):")
    sorted_idx = np.argsort(mean_importance)[::-1]
    for i in sorted_idx[:5]:
        print(f"  {feature_names[i]}: {mean_importance[i]:.4f}")
    
    # LCL correlation
    lcl_idx = feature_names.index('lcl')
    lcl_values = X[:, lcl_idx]
    
    # Compute LCL correlations
    corr_pred_lcl, p_pred = stats.pearsonr(all_y_pred, lcl_values)
    corr_true_lcl, p_true = stats.pearsonr(all_y_true, lcl_values)
    
    print(f"\nLCL Correlations:")
    print(f"  Predicted CBH vs LCL: r = {corr_pred_lcl:.3f} (p = {p_pred:.4f})")
    print(f"  True CBH vs LCL: r = {corr_true_lcl:.3f} (p = {p_true:.4f})")
    
    # Save report
    report = {
        'metadata': {
            'model_type': 'GradientBoostingRegressor',
            'n_folds': n_folds,
            'random_seed': 42,
            'validation_strategy': 'stratified_5fold',
            'timestamp': datetime.now().isoformat(),
            'data_source': 'Verified_Integrated_Features.hdf5',
            'n_samples': len(y),
        },
        'folds': fold_results,
        'mean_metrics': mean_metrics,
        'std_metrics': std_metrics,
        'feature_importance': feature_importance,
        'lcl_correlations': {
            'pred_vs_lcl': {'r': corr_pred_lcl, 'p': p_pred},
            'true_vs_lcl': {'r': corr_true_lcl, 'p': p_true},
        },
        'aggregated_predictions': {
            'y_true': (all_y_true / 1000).tolist(),  # Back to km
            'y_pred': (all_y_pred / 1000).tolist(),
        },
    }
    
    report_path = REPORTS_DIR / "validation_report_tabular_VERIFIED.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved report: {report_path}")
    
    # Create figures
    create_figures(all_y_true, all_y_pred, feature_names, mean_importance, lcl_values)
    
    return report

def create_figures(y_true, y_pred, feature_names, importance, lcl_values):
    """Create validation figures."""
    print("\n" + "=" * 80)
    print("Creating Figures")
    print("=" * 80)
    
    # 1. Scatter plot: True vs Predicted
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_true, y_pred, alpha=0.3, s=10)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
    
    # Regression line
    slope, intercept, r, p, se = stats.linregress(y_true, y_pred)
    ax.plot([min_val, max_val], [slope*min_val+intercept, slope*max_val+intercept], 
            'b-', linewidth=2, label=f'Fit (R²={r**2:.3f})')
    
    ax.set_xlabel('True CBH (m)', fontsize=12)
    ax.set_ylabel('Predicted CBH (m)', fontsize=12)
    ax.set_title('GBDT: True vs Predicted CBH (5-fold CV)', fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'true_vs_predicted_VERIFIED.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: true_vs_predicted_VERIFIED.png")
    
    # 2. Feature importance
    fig, ax = plt.subplots(figsize=(10, 6))
    sorted_idx = np.argsort(importance)
    ax.barh(range(len(feature_names)), importance[sorted_idx])
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels([feature_names[i] for i in sorted_idx])
    ax.set_xlabel('Feature Importance', fontsize=12)
    ax.set_title('GBDT Feature Importance', fontsize=14)
    ax.grid(alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'feature_importance_VERIFIED.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: feature_importance_VERIFIED.png")
    
    # 3. CBH vs LCL
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # True CBH vs LCL
    ax = axes[0]
    ax.scatter(lcl_values, y_true, alpha=0.3, s=10)
    slope, intercept, r, p, se = stats.linregress(lcl_values, y_true)
    x_line = np.array([lcl_values.min(), lcl_values.max()])
    ax.plot(x_line, slope*x_line + intercept, 'r-', linewidth=2, 
            label=f'r={r:.3f}, p={p:.2e}')
    ax.set_xlabel('LCL (m)', fontsize=12)
    ax.set_ylabel('True CBH (m)', fontsize=12)
    ax.set_title('True CBH vs LCL', fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Predicted CBH vs LCL
    ax = axes[1]
    ax.scatter(lcl_values, y_pred, alpha=0.3, s=10)
    slope, intercept, r, p, se = stats.linregress(lcl_values, y_pred)
    ax.plot(x_line, slope*x_line + intercept, 'r-', linewidth=2,
            label=f'r={r:.3f}, p={p:.2e}')
    ax.set_xlabel('LCL (m)', fontsize=12)
    ax.set_ylabel('Predicted CBH (m)', fontsize=12)
    ax.set_title('Predicted CBH vs LCL', fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'cbh_vs_lcl_VERIFIED.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: cbh_vs_lcl_VERIFIED.png")
    
    # 4. Error distribution
    errors = y_pred - y_true
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(errors, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(0, color='r', linestyle='--', linewidth=2)
    ax.axvline(errors.mean(), color='g', linestyle='-', linewidth=2, 
               label=f'Mean error: {errors.mean():.1f} m')
    ax.set_xlabel('Prediction Error (m)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Error Distribution', fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'error_distribution_VERIFIED.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: error_distribution_VERIFIED.png")

if __name__ == '__main__':
    report = train_and_evaluate()
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
