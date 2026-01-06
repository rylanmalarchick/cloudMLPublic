#!/usr/bin/env python3
"""
Quick Physics-Based Validation for CBH Predictions
Streamlined version that works with existing data structure
"""

import json
import sys
import warnings
from sklearn.exceptions import ConvergenceWarning
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Suppress sklearn convergence and numpy deprecation warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

print("=" * 80)
print("Quick Physics-Based Validation for CBH Predictions")
print("=" * 80)

# Paths
INTEGRATED_FEATURES = PROJECT_ROOT / "outputs/preprocessed_data/Integrated_Features.hdf5"
OUTPUT_DIR = PROJECT_ROOT / "outputs/physics_validation"
FIGURES_DIR = OUTPUT_DIR / "figures"
REPORTS_DIR = OUTPUT_DIR / "reports"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    """Load data with actual structure."""
    print("\nLoading data...")
    
    with h5py.File(INTEGRATED_FEATURES, "r") as f:
        # Load labels and metadata
        cbh_km = f["metadata/cbh_km"][:]
        flight_ids = f["metadata/flight_id"][:]
        
        # Load atmospheric features (individual arrays)
        atmo_group = f["atmospheric_features"]
        era5_names = ['blh', 'lcl', 'inversion_height', 'moisture_gradient',
                      'stability_index', 't2m', 'd2m', 'sp', 'tcwv']
        era5_features = []
        for name in era5_names:
            if name in atmo_group:
                era5_features.append(atmo_group[name][:])
        era5_features = np.column_stack(era5_features) if era5_features else np.zeros((len(cbh_km), 0))
        
        # Load geometric features (individual arrays)
        geo_group = f["geometric_features"]
        shadow_names = ['sza_deg', 'saa_deg', 'shadow_length_pixels', 
                       'shadow_angle_deg', 'shadow_detection_confidence']
        shadow_features = []
        for name in shadow_names:
            if name in geo_group:
                shadow_features.append(geo_group[name][:])
        shadow_features = np.column_stack(shadow_features) if shadow_features else np.zeros((len(cbh_km), 0))
        
        # Combine features
        X = np.concatenate([era5_features, shadow_features], axis=1)
        feature_names = era5_names + shadow_names
        y = cbh_km * 1000  # Convert to meters
        
    print(f"  Loaded {X.shape[0]} samples with {X.shape[1]} features")
    print(f"  CBH range: {y.min():.0f} - {y.max():.0f} m")
    
    return X, y, flight_ids, feature_names

def train_gbdt(X_train, y_train):
    """Train GBDT model (matching paper configuration)."""
    print("\nTraining GBDT model...")
    
    model = GradientBoostingRegressor(
        n_estimators=150,
        learning_rate=0.1,
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=4,
        subsample=0.8,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    print("  Training complete")
    
    return model

def create_cbh_vs_lcl_plots(X, y, predictions, feature_names):
    """Create CBH vs LCL scatter plots."""
    print("\nCreating CBH vs. LCL validation plots...")
    
    # Find LCL feature if it exists
    lcl_idx = None
    for i, name in enumerate(feature_names):
        if 'lcl' in name.lower():
            lcl_idx = i
            break
    
    if lcl_idx is None:
        print("  Warning: No LCL feature found, skipping LCL plots")
        return {}
    
    lcl_values = X[:, lcl_idx]
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Predicted CBH vs. LCL
    axes[0].scatter(lcl_values, predictions, alpha=0.5, s=20, c='blue', edgecolors='none')
    axes[0].plot([0, 3000], [0, 3000], 'r--', linewidth=2, label='1:1 line', alpha=0.7)
    axes[0].set_xlabel('LCL (m)', fontsize=12)
    axes[0].set_ylabel('Predicted CBH (m)', fontsize=12)
    axes[0].set_title('GBDT Predictions vs. LCL', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Compute correlation
    corr_pred, p_pred = pearsonr(lcl_values, predictions)
    axes[0].text(0.05, 0.95, f'r = {corr_pred:.3f}\np < 0.001', 
                transform=axes[0].transAxes, fontsize=10, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Right: True CBH vs. LCL
    axes[1].scatter(lcl_values, y, alpha=0.5, s=20, c='green', edgecolors='none')
    axes[1].plot([0, 3000], [0, 3000], 'r--', linewidth=2, label='1:1 line', alpha=0.7)
    axes[1].set_xlabel('LCL (m)', fontsize=12)
    axes[1].set_ylabel('True CBH (m)', fontsize=12)
    axes[1].set_title('Observed CBH vs. LCL', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    # Compute correlation
    corr_true, p_true = pearsonr(lcl_values, y)
    axes[1].text(0.05, 0.95, f'r = {corr_true:.3f}\np < 0.001', 
                transform=axes[1].transAxes, fontsize=10, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "cbh_vs_lcl_validation.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {FIGURES_DIR / 'cbh_vs_lcl_validation.png'}")
    
    return {
        'correlation_predicted_lcl': float(corr_pred),
        'correlation_true_lcl': float(corr_true),
        'p_value_predicted': float(p_pred),
        'p_value_true': float(p_true)
    }

def check_physical_constraints(predictions, X, feature_names):
    """Check physical plausibility of predictions."""
    print("\nChecking physical constraints...")
    
    # Constraint 1: CBH should not exceed tropopause (12 km)
    invalid_high = (predictions > 12000).sum()
    pct_high = invalid_high / len(predictions) * 100
    
    # Constraint 2: CBH should be positive
    invalid_neg = (predictions < 0).sum()
    pct_neg = invalid_neg / len(predictions) * 100
    
    # Constraint 3: Check correlations with physical variables
    correlations = {}
    
    # Find relevant features
    for i, name in enumerate(feature_names):
        if 'cape' in name.lower():
            corr, p = spearmanr(X[:, i], predictions)
            correlations['CAPE'] = {'correlation': float(corr), 'p_value': float(p), 
                                   'expected': 'negative', 'observed': 'negative' if corr < 0 else 'positive'}
        elif 'blh' in name.lower() or 'boundary' in name.lower():
            corr, p = spearmanr(X[:, i], predictions)
            correlations['BLH'] = {'correlation': float(corr), 'p_value': float(p),
                                  'expected': 'positive', 'observed': 'positive' if corr > 0 else 'negative'}
    
    results = {
        'cbh_below_tropopause': {
            'expected_pct': 100.0,
            'observed_pct': float(100 - pct_high),
            'violations': int(invalid_high),
            'total': int(len(predictions))
        },
        'cbh_positive': {
            'expected_pct': 100.0,
            'observed_pct': float(100 - pct_neg),
            'violations': int(invalid_neg),
            'total': int(len(predictions))
        },
        'physical_correlations': correlations
    }
    
    print(f"  Predictions > 12 km: {invalid_high} ({pct_high:.2f}%)")
    print(f"  Negative predictions: {invalid_neg} ({pct_neg:.2f}%)")
    for var, stats in correlations.items():
        print(f"  Correlation({var}, CBH): {stats['correlation']:.3f} (expected: {stats['expected']})")
    
    return results

def create_case_studies(X, y, predictions, feature_names):
    """Create case study analysis for best, worst, and median predictions."""
    print("\nCreating case study analysis...")
    
    errors = np.abs(predictions - y)
    
    # Find indices
    best_idx = np.argmin(errors)
    worst_idx = np.argmax(errors)
    median_idx = np.argsort(errors)[len(errors)//2]
    
    cases = {
        'best': {
            'index': int(best_idx),
            'true_cbh_m': float(y[best_idx]),
            'predicted_cbh_m': float(predictions[best_idx]),
            'error_m': float(errors[best_idx])
        },
        'worst': {
            'index': int(worst_idx),
            'true_cbh_m': float(y[worst_idx]),
            'predicted_cbh_m': float(predictions[worst_idx]),
            'error_m': float(errors[worst_idx])
        },
        'median': {
            'index': int(median_idx),
            'true_cbh_m': float(y[median_idx]),
            'predicted_cbh_m': float(predictions[median_idx]),
            'error_m': float(errors[median_idx])
        }
    }
    
    print(f"  Best case: Error = {cases['best']['error_m']:.1f} m")
    print(f"  Worst case: Error = {cases['worst']['error_m']:.1f} m")
    print(f"  Median case: Error = {cases['median']['error_m']:.1f} m")
    
    return cases

def main():
    # Load data
    X, y, flight_ids, feature_names = load_data()
    
    # Handle NaN values
    nan_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X = X[nan_mask]
    y = y[nan_mask]
    flight_ids = flight_ids[nan_mask]
    print(f"\nAfter removing NaN: {len(X)} samples")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = train_gbdt(X_train_scaled, y_train)
    
    # Make predictions
    predictions = model.predict(X_test_scaled)
    
    # Evaluate
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    
    print(f"\nModel Performance:")
    print(f"  R² = {r2:.3f}")
    print(f"  MAE = {mae:.1f} m")
    print(f"  RMSE = {rmse:.1f} m")
    
    # Physics validation
    lcl_results = create_cbh_vs_lcl_plots(X_test, y_test, predictions, feature_names)
    constraint_results = check_physical_constraints(predictions, X_test_scaled, feature_names)
    case_studies = create_case_studies(X_test, y_test, predictions, feature_names)
    
    # Save report
    report = {
        'model_performance': {
            'r2': float(r2),
            'mae_m': float(mae),
            'rmse_m': float(rmse),
            'n_test': int(len(y_test))
        },
        'lcl_correlation': lcl_results,
        'physical_constraints': constraint_results,
        'case_studies': case_studies,
        'metadata': {
            'n_features': int(X.shape[1]),
            'feature_names': feature_names,
            'timestamp': pd.Timestamp.now().isoformat()
        }
    }
    
    report_path = REPORTS_DIR / "physics_validation_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✓ Report saved: {report_path}")
    print("=" * 80)
    print("Physics Validation Complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()
