#!/usr/bin/env python3
"""
Physics-Based Validation for Cloud Base Height Predictions

Implements:
- LCL (Lifting Condensation Level) comparison
- Physical plausibility checks (CBH vs tropopause)
- Residual analysis correlated with atmospheric variables
- Case studies: success, failure, and median cases

Author: Preprint Revision Task 3
Date: 2025
"""

import json
import sys
import warnings
from datetime import datetime
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

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

print("=" * 80)
print("Physics-Based Validation for CBH Predictions")
print("=" * 80)

# Paths
INTEGRATED_FEATURES = PROJECT_ROOT / "outputs/preprocessed_data/Integrated_Features.hdf5"
OUTPUT_DIR = PROJECT_ROOT / "outputs/physics_validation"
FIGURES_DIR = OUTPUT_DIR / "figures"
REPORTS_DIR = OUTPUT_DIR / "reports"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Physical constants
TROPOPAUSE_HEIGHT_KM = 11.0  # Approximate tropopause height (km)
SURFACE_HEIGHT_KM = 0.0  # Surface level


def compute_lcl(temperature_k, dewpoint_k, pressure_pa):
    """
    Compute Lifting Condensation Level (LCL) height.
    
    Uses Espy's equation for LCL temperature and the hypsometric equation for height.
    
    Args:
        temperature_k: Temperature (K)
        dewpoint_k: Dewpoint temperature (K)
        pressure_pa: Pressure (Pa)
    
    Returns:
        lcl_height_km: LCL height (km)
    """
    # Espy's equation for LCL temperature
    # LCL_temp (°C) = Dewpoint - [(T - Dewpoint) / 4.4]
    T_c = temperature_k - 273.15
    Td_c = dewpoint_k - 273.15
    
    LCL_temp_c = Td_c - ((T_c - Td_c) / 4.4)
    
    # Convert to height using hypsometric equation
    # Simplified: ΔH ≈ (T - LCL_temp) * 125 m/°C
    delta_h_m = (T_c - LCL_temp_c) * 125
    
    # Convert to km
    lcl_height_km = delta_h_m / 1000.0
    
    return np.clip(lcl_height_km, 0, 15)  # Clip to reasonable range


def load_data():
    """Load data and compute derived physics variables."""
    print("\n" + "=" * 80)
    print("Loading Data and Computing Physics Variables")
    print("=" * 80)
    
    with h5py.File(INTEGRATED_FEATURES, "r") as f:
        # Load features
        features = f["tabular/features"][:]
        feature_names = [name.decode('utf-8') if isinstance(name, bytes) else name 
                        for name in f["tabular/feature_names"][:]]
        
        # Load metadata
        flight_ids = f["metadata/flight_id"][:]
        sample_ids = f["metadata/sample_id"][:]
        cbh_km = f["metadata/cbh_km"][:]
        flight_mapping = json.loads(f.attrs["flight_mapping"])
    
    # Convert to DataFrame
    df = pd.DataFrame(features, columns=feature_names)
    df['flight_id'] = flight_ids
    df['sample_id'] = sample_ids
    df['cbh_true_km'] = cbh_km
    
    print(f"Total samples: {len(df)}")
    print(f"Features: {len(feature_names)}")
    
    # Compute LCL if we have the required features
    if all(f in feature_names for f in ['surface_temp_k', 'surface_dewpoint_k', 'surface_pressure_pa']):
        print("\nComputing LCL...")
        df['lcl_km'] = compute_lcl(
            df['surface_temp_k'].values,
            df['surface_dewpoint_k'].values,
            df['surface_pressure_pa'].values
        )
        print(f"LCL range: [{df['lcl_km'].min():.3f}, {df['lcl_km'].max():.3f}] km")
    elif 'lcl' in feature_names:
        df['lcl_km'] = df['lcl']
        print(f"Using existing LCL feature")
    else:
        print("WARNING: LCL computation requires surface_temp_k, surface_dewpoint_k, surface_pressure_pa")
        print("Available features:", [f for f in feature_names if 'temp' in f.lower() or 'pressure' in f.lower()])
        # Use a proxy if available
        if 'blh' in feature_names:
            df['lcl_km'] = df['blh'] / 1000.0  # Convert BLH from m to km as LCL proxy
            print("Using BLH as LCL proxy")
        else:
            df['lcl_km'] = np.nan
    
    return df, feature_names, flight_mapping


def train_model_and_predict(df, feature_names):
    """Train GBDT model and generate predictions."""
    print("\n" + "=" * 80)
    print("Training Model and Generating Predictions")
    print("=" * 80)
    
    X = df[feature_names].values
    y = df['cbh_true_km'].values
    
    # Train/test split
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, np.arange(len(X)), test_size=0.2, random_state=42
    )
    
    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.05,
        min_samples_split=10,
        min_samples_leaf=4,
        random_state=42,
        subsample=0.8,
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Predict on full dataset
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    
    df['cbh_pred_km'] = y_pred
    df['residual_km'] = y_pred - y
    df['residual_m'] = df['residual_km'] * 1000
    
    print(f"Model trained on {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    return df, model, scaler


def validate_physical_plausibility(df):
    """Check for physical implausibilities."""
    print("\n" + "=" * 80)
    print("Physical Plausibility Checks")
    print("=" * 80)
    
    results = {
        'total_samples': len(df),
        'checks': [],
    }
    
    # Check 1: CBH < tropopause
    above_tropopause = df['cbh_pred_km'] > TROPOPAUSE_HEIGHT_KM
    n_above_tropopause = above_tropopause.sum()
    pct_above_tropopause = n_above_tropopause / len(df) * 100
    
    print(f"Check 1: CBH > Tropopause ({TROPOPAUSE_HEIGHT_KM} km)")
    print(f"  Violations: {n_above_tropopause}/{len(df)} ({pct_above_tropopause:.2f}%)")
    
    results['checks'].append({
        'name': 'CBH > Tropopause',
        'violations': int(n_above_tropopause),
        'percentage': float(pct_above_tropopause),
    })
    
    # Check 2: CBH > 0 (above surface)
    below_surface = df['cbh_pred_km'] < SURFACE_HEIGHT_KM
    n_below_surface = below_surface.sum()
    pct_below_surface = n_below_surface / len(df) * 100
    
    print(f"\nCheck 2: CBH < Surface ({SURFACE_HEIGHT_KM} km)")
    print(f"  Violations: {n_below_surface}/{len(df)} ({pct_below_surface:.2f}%)")
    
    results['checks'].append({
        'name': 'CBH < Surface',
        'violations': int(n_below_surface),
        'percentage': float(pct_below_surface),
    })
    
    # Check 3: Reasonable range (0-5 km for most clouds)
    REASONABLE_MAX_KM = 5.0
    unreasonable_high = df['cbh_pred_km'] > REASONABLE_MAX_KM
    n_unreasonable_high = unreasonable_high.sum()
    pct_unreasonable_high = n_unreasonable_high / len(df) * 100
    
    print(f"\nCheck 3: CBH > Reasonable Max ({REASONABLE_MAX_KM} km)")
    print(f"  Count: {n_unreasonable_high}/{len(df)} ({pct_unreasonable_high:.2f}%)")
    print(f"  Note: High clouds exist, but most CBH < 5 km")
    
    results['checks'].append({
        'name': 'CBH > Reasonable Max (5 km)',
        'violations': int(n_unreasonable_high),
        'percentage': float(pct_unreasonable_high),
    })
    
    return results


def compare_with_lcl(df):
    """Compare predictions with LCL."""
    print("\n" + "=" * 80)
    print("LCL Comparison Analysis")
    print("=" * 80)
    
    if 'lcl_km' not in df.columns or df['lcl_km'].isna().all():
        print("ERROR: LCL not available")
        return None
    
    # Remove NaN LCL values
    df_lcl = df[df['lcl_km'].notna()].copy()
    
    print(f"Samples with valid LCL: {len(df_lcl)}")
    
    # Correlation analysis
    corr_true_lcl, p_true_lcl = pearsonr(df_lcl['cbh_true_km'], df_lcl['lcl_km'])
    corr_pred_lcl, p_pred_lcl = pearsonr(df_lcl['cbh_pred_km'], df_lcl['lcl_km'])
    
    print(f"\nCorrelation Analysis:")
    print(f"  True CBH vs LCL: r={corr_true_lcl:.4f}, p={p_true_lcl:.4e}")
    print(f"  Pred CBH vs LCL: r={corr_pred_lcl:.4f}, p={p_pred_lcl:.4e}")
    
    # Difference from LCL
    df_lcl['diff_from_lcl_true'] = df_lcl['cbh_true_km'] - df_lcl['lcl_km']
    df_lcl['diff_from_lcl_pred'] = df_lcl['cbh_pred_km'] - df_lcl['lcl_km']
    
    print(f"\nDifference from LCL (km):")
    print(f"  True CBH: mean={df_lcl['diff_from_lcl_true'].mean():.3f}, std={df_lcl['diff_from_lcl_true'].std():.3f}")
    print(f"  Pred CBH: mean={df_lcl['diff_from_lcl_pred'].mean():.3f}, std={df_lcl['diff_from_lcl_pred'].std():.3f}")
    
    return {
        'n_samples': len(df_lcl),
        'corr_true_lcl': float(corr_true_lcl),
        'p_true_lcl': float(p_true_lcl),
        'corr_pred_lcl': float(corr_pred_lcl),
        'p_pred_lcl': float(p_pred_lcl),
        'mean_diff_true': float(df_lcl['diff_from_lcl_true'].mean()),
        'std_diff_true': float(df_lcl['diff_from_lcl_true'].std()),
        'mean_diff_pred': float(df_lcl['diff_from_lcl_pred'].mean()),
        'std_diff_pred': float(df_lcl['diff_from_lcl_pred'].std()),
    }


def correlate_errors_with_atmospheric_vars(df, feature_names):
    """Correlate prediction errors with atmospheric variables."""
    print("\n" + "=" * 80)
    print("Error Correlation with Atmospheric Variables")
    print("=" * 80)
    
    # Key atmospheric variables to check
    atmospheric_vars = [
        f for f in feature_names 
        if any(keyword in f.lower() for keyword in 
               ['temp', 'pressure', 'humidity', 'stability', 'blh', 'inversion'])
    ]
    
    if not atmospheric_vars:
        print("No atmospheric variables found")
        return None
    
    print(f"Analyzing {len(atmospheric_vars)} atmospheric variables")
    
    error_correlations = []
    
    for var in atmospheric_vars:
        if var in df.columns:
            valid_mask = df[var].notna() & df['residual_m'].notna()
            if valid_mask.sum() > 10:  # Need minimum samples
                corr, p_value = pearsonr(df.loc[valid_mask, var], df.loc[valid_mask, 'residual_m'])
                
                error_correlations.append({
                    'variable': var,
                    'correlation': corr,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                })
    
    error_corr_df = pd.DataFrame(error_correlations).sort_values('correlation', key=abs, ascending=False)
    
    print(f"\nTop 10 correlations with prediction error:")
    for _, row in error_corr_df.head(10).iterrows():
        sig_marker = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else ""
        print(f"  {row['variable']}: r={row['correlation']:.4f}, p={row['p_value']:.4e} {sig_marker}")
    
    return error_corr_df


def generate_case_studies(df):
    """Generate case studies: success, failure, median."""
    print("\n" + "=" * 80)
    print("Generating Case Studies")
    print("=" * 80)
    
    # Compute absolute error
    df['abs_error_m'] = np.abs(df['residual_m'])
    
    # Sort by absolute error
    df_sorted = df.sort_values('abs_error_m')
    
    # Select cases
    n_samples = len(df)
    
    # Success: lowest error
    success_cases = df_sorted.head(5)
    
    # Failure: highest error
    failure_cases = df_sorted.tail(5)
    
    # Median: around 50th percentile
    median_idx = n_samples // 2
    median_cases = df_sorted.iloc[median_idx-2:median_idx+3]
    
    case_studies = {
        'success': success_cases.to_dict('records'),
        'failure': failure_cases.to_dict('records'),
        'median': median_cases.to_dict('records'),
    }
    
    print("\nSuccess Cases (lowest error):")
    for i, case in enumerate(success_cases.itertuples(), 1):
        print(f"  {i}. True={case.cbh_true_km:.3f} km, Pred={case.cbh_pred_km:.3f} km, Error={case.abs_error_m:.1f} m")
    
    print("\nFailure Cases (highest error):")
    for i, case in enumerate(failure_cases.itertuples(), 1):
        print(f"  {i}. True={case.cbh_true_km:.3f} km, Pred={case.cbh_pred_km:.3f} km, Error={case.abs_error_m:.1f} m")
    
    print("\nMedian Cases:")
    for i, case in enumerate(median_cases.itertuples(), 1):
        print(f"  {i}. True={case.cbh_true_km:.3f} km, Pred={case.cbh_pred_km:.3f} km, Error={case.abs_error_m:.1f} m")
    
    return case_studies


def visualize_lcl_comparison(df):
    """Visualize CBH vs LCL comparison."""
    if 'lcl_km' not in df.columns or df['lcl_km'].isna().all():
        return
    
    df_lcl = df[df['lcl_km'].notna()].copy()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Scatter: CBH vs LCL
    axes[0].scatter(df_lcl['lcl_km'], df_lcl['cbh_true_km'], alpha=0.5, s=30, label='True CBH')
    axes[0].scatter(df_lcl['lcl_km'], df_lcl['cbh_pred_km'], alpha=0.5, s=30, label='Predicted CBH')
    
    lim = [0, max(df_lcl['lcl_km'].max(), df_lcl['cbh_true_km'].max(), df_lcl['cbh_pred_km'].max())]
    axes[0].plot(lim, lim, 'k--', lw=2, label='1:1 Line')
    
    axes[0].set_xlabel("LCL (km)", fontsize=12)
    axes[0].set_ylabel("CBH (km)", fontsize=12)
    axes[0].set_title("Cloud Base Height vs LCL", fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Residuals vs LCL
    axes[1].scatter(df_lcl['lcl_km'], df_lcl['residual_m'], alpha=0.5, s=30)
    axes[1].axhline(0, color='r', linestyle='--', lw=2, label='Zero Error')
    
    axes[1].set_xlabel("LCL (km)", fontsize=12)
    axes[1].set_ylabel("Residual (m)", fontsize=12)
    axes[1].set_title("Prediction Residuals vs LCL", fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "cbh_vs_lcl_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {FIGURES_DIR / 'cbh_vs_lcl_comparison.png'}")


def visualize_error_correlations(error_corr_df):
    """Visualize error correlations with atmospheric variables."""
    if error_corr_df is None or len(error_corr_df) == 0:
        return
    
    # Top 15 variables
    top_vars = error_corr_df.head(15)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['red' if row['significant'] else 'gray' for _, row in top_vars.iterrows()]
    
    ax.barh(top_vars['variable'], top_vars['correlation'], color=colors, edgecolor='black')
    ax.axvline(0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel("Pearson Correlation with Prediction Error (m)", fontsize=12)
    ax.set_ylabel("Atmospheric Variable", fontsize=12)
    ax.set_title("Error Correlation with Atmospheric Variables", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', edgecolor='black', label='Significant (p<0.05)'),
        Patch(facecolor='gray', edgecolor='black', label='Not Significant'),
    ]
    ax.legend(handles=legend_elements, loc='best')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "error_correlation_atmospheric.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {FIGURES_DIR / 'error_correlation_atmospheric.png'}")


def save_results(plausibility_results, lcl_results, error_corr_df, case_studies):
    """Save all results."""
    print("\n" + "=" * 80)
    print("Saving Results")
    print("=" * 80)
    
    # Save plausibility results
    with open(REPORTS_DIR / "physical_plausibility_results.json", "w") as f:
        json.dump(plausibility_results, f, indent=2)
    print(f"Saved: {REPORTS_DIR / 'physical_plausibility_results.json'}")
    
    # Save LCL results
    if lcl_results:
        with open(REPORTS_DIR / "lcl_comparison_results.json", "w") as f:
            json.dump(lcl_results, f, indent=2)
        print(f"Saved: {REPORTS_DIR / 'lcl_comparison_results.json'}")
    
    # Save error correlations
    if error_corr_df is not None and len(error_corr_df) > 0:
        error_corr_df.to_csv(REPORTS_DIR / "error_correlation_atmospheric.csv", index=False)
        print(f"Saved: {REPORTS_DIR / 'error_correlation_atmospheric.csv'}")
    
    # Save case studies
    with open(REPORTS_DIR / "case_studies.json", "w") as f:
        json.dump(case_studies, f, indent=2)
    print(f"Saved: {REPORTS_DIR / 'case_studies.json'}")
    
    # Aggregated report
    report = {
        'timestamp': datetime.now().isoformat(),
        'physical_plausibility': plausibility_results,
        'lcl_comparison': lcl_results,
        'case_studies_summary': {
            'n_success': len(case_studies['success']),
            'n_failure': len(case_studies['failure']),
            'n_median': len(case_studies['median']),
        },
    }
    
    with open(REPORTS_DIR / "physics_validation_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"Saved: {REPORTS_DIR / 'physics_validation_report.json'}")


def main():
    """Main execution."""
    # Load data
    df, feature_names, flight_mapping = load_data()
    
    # Train model and predict
    df, model, scaler = train_model_and_predict(df, feature_names)
    
    # Physical plausibility checks
    plausibility_results = validate_physical_plausibility(df)
    
    # LCL comparison
    lcl_results = compare_with_lcl(df)
    
    # Error correlation analysis
    error_corr_df = correlate_errors_with_atmospheric_vars(df, feature_names)
    
    # Generate case studies
    case_studies = generate_case_studies(df)
    
    # Visualizations
    visualize_lcl_comparison(df)
    visualize_error_correlations(error_corr_df)
    
    # Save results
    save_results(plausibility_results, lcl_results, error_corr_df, case_studies)
    
    print("\n" + "=" * 80)
    print("Physics-Based Validation Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
