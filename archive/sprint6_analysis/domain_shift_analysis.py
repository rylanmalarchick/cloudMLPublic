#!/usr/bin/env python3
"""
Domain Shift and Leave-One-Flight-Out Cross-Validation Analysis

Implements:
- Leave-one-flight-out (LOFO) cross-validation
- Maximum Mean Discrepancy (MMD) for domain divergence
- Kolmogorov-Smirnov tests per feature per flight
- PCA/t-SNE visualization colored by flight
- Statistical significance tests with confidence intervals

Author: Preprint Revision Task 2
Date: 2025
"""

import json
import sys
import warnings
from sklearn.exceptions import ConvergenceWarning
from datetime import datetime
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ks_2samp, ttest_ind
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.manifold import TSNE
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Suppress sklearn convergence and numpy deprecation warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

print("=" * 80)
print("Domain Shift and LOFO Cross-Validation Analysis")
print("=" * 80)

# Paths
INTEGRATED_FEATURES = PROJECT_ROOT / "outputs/preprocessed_data/Integrated_Features.hdf5"
OUTPUT_DIR = PROJECT_ROOT / "outputs/domain_analysis"
FIGURES_DIR = OUTPUT_DIR / "figures"
REPORTS_DIR = OUTPUT_DIR / "reports"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Minimum samples per flight to include
MIN_SAMPLES_PER_FLIGHT = 60


def compute_mmd(X, Y, kernel='rbf', gamma=None):
    """
    Compute Maximum Mean Discrepancy (MMD) between two distributions.
    
    Args:
        X: Samples from distribution P (n_samples_X, n_features)
        Y: Samples from distribution Q (n_samples_Y, n_features)
        kernel: Kernel type ('rbf', 'linear')
        gamma: RBF kernel bandwidth (default: 1 / n_features)
    
    Returns:
        mmd: MMD statistic
    """
    n_X = X.shape[0]
    n_Y = Y.shape[0]
    
    if gamma is None:
        gamma = 1.0 / X.shape[1]
    
    def rbf_kernel(X, Y, gamma):
        """RBF kernel matrix."""
        XX = np.sum(X**2, axis=1)[:, np.newaxis]
        YY = np.sum(Y**2, axis=1)[np.newaxis, :]
        XY = np.dot(X, Y.T)
        distances = XX + YY - 2 * XY
        return np.exp(-gamma * distances)
    
    def linear_kernel(X, Y):
        """Linear kernel matrix."""
        return np.dot(X, Y.T)
    
    if kernel == 'rbf':
        K_XX = rbf_kernel(X, X, gamma)
        K_YY = rbf_kernel(Y, Y, gamma)
        K_XY = rbf_kernel(X, Y, gamma)
    else:  # linear
        K_XX = linear_kernel(X, X)
        K_YY = linear_kernel(Y, Y)
        K_XY = linear_kernel(X, Y)
    
    # Compute MMD^2
    mmd_squared = (np.sum(K_XX) / (n_X * n_X) + 
                   np.sum(K_YY) / (n_Y * n_Y) - 
                   2 * np.sum(K_XY) / (n_X * n_Y))
    
    return np.sqrt(max(mmd_squared, 0))


def load_data():
    """Load data from integrated features."""
    print("\n" + "=" * 80)
    print("Loading Data")
    print("=" * 80)
    
    with h5py.File(INTEGRATED_FEATURES, "r") as f:
        # Load atmospheric features
        atmo_features = f["atmospheric_features/era5_features"][:]
        atmo_names = [name.decode('utf-8') if isinstance(name, bytes) else name 
                     for name in f["atmospheric_features/era5_feature_names"][:]]
        
        # Load geometric features
        geom_features = []
        geom_names = []
        for key in ['sza_deg', 'saa_deg', 'shadow_length_pixels', 'shadow_angle_deg', 
                    'shadow_detection_confidence', 'derived_geometric_H']:
            if key in f['geometric_features']:
                geom_features.append(f[f'geometric_features/{key}'][:])
                geom_names.append(key)
        
        geom_features = np.column_stack(geom_features) if geom_features else np.array([])
        
        # Combine features
        if geom_features.size > 0:
            features = np.column_stack([atmo_features, geom_features])
            feature_names = atmo_names + geom_names
        else:
            features = atmo_features
            feature_names = atmo_names
        
        # Load metadata
        flight_ids = f["metadata/flight_id"][:]
        sample_ids = f["metadata/sample_id"][:]
        cbh_km = f["metadata/cbh_km"][:]
        flight_mapping = json.loads(f.attrs["flight_mapping"])
    
    # Convert to DataFrame
    df = pd.DataFrame(features, columns=feature_names)
    df['flight_id'] = flight_ids
    df['sample_id'] = sample_ids
    df['cbh_km'] = cbh_km
    
    # Get flight names
    flight_id_to_name = {int(k): v for k, v in flight_mapping.items()}
    df['flight_name'] = df['flight_id'].map(flight_id_to_name)
    
    print(f"Total samples: {len(df)}")
    print(f"Features: {len(feature_names)}")
    print(f"Flights: {sorted(df['flight_name'].unique())}")
    print(f"\nSamples per flight:")
    flight_counts = df.groupby('flight_name').size().sort_values(ascending=False)
    for flight, count in flight_counts.items():
        print(f"  {flight}: {count} samples")
    
    return df, feature_names, flight_mapping


def filter_flights_by_size(df, min_samples=MIN_SAMPLES_PER_FLIGHT):
    """Filter out flights with too few samples."""
    flight_counts = df['flight_name'].value_counts()
    valid_flights = flight_counts[flight_counts >= min_samples].index.tolist()
    
    df_filtered = df[df['flight_name'].isin(valid_flights)].copy()
    
    print(f"\n" + "=" * 80)
    print(f"Filtering Flights (min {min_samples} samples)")
    print("=" * 80)
    print(f"Before filtering: {len(df)} samples, {df['flight_name'].nunique()} flights")
    print(f"After filtering: {len(df_filtered)} samples, {df_filtered['flight_name'].nunique()} flights")
    print(f"Retained flights: {valid_flights}")
    
    return df_filtered, valid_flights


def compute_ks_divergence_per_feature(df, feature_names):
    """Compute KS test for each feature comparing each flight to all others."""
    print("\n" + "=" * 80)
    print("Computing Kolmogorov-Smirnov Divergence Per Feature")
    print("=" * 80)
    
    flights = df['flight_name'].unique()
    n_flights = len(flights)
    n_features = len(feature_names)
    
    # Store results
    ks_results = []
    
    for flight in flights:
        flight_data = df[df['flight_name'] == flight]
        other_data = df[df['flight_name'] != flight]
        
        for feature in feature_names:
            flight_feature = flight_data[feature].values
            other_feature = other_data[feature].values
            
            # Remove NaN/Inf
            flight_feature = flight_feature[np.isfinite(flight_feature)]
            other_feature = other_feature[np.isfinite(other_feature)]
            
            if len(flight_feature) > 0 and len(other_feature) > 0:
                ks_stat, p_value = ks_2samp(flight_feature, other_feature)
                
                ks_results.append({
                    'flight': flight,
                    'feature': feature,
                    'ks_statistic': ks_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                })
    
    ks_df = pd.DataFrame(ks_results)
    
    # Summary
    print(f"Total tests: {len(ks_df)}")
    print(f"Significant tests (p < 0.05): {ks_df['significant'].sum()} ({ks_df['significant'].mean()*100:.1f}%)")
    
    # Top divergent features per flight
    print("\nTop 5 most divergent features per flight:")
    for flight in flights:
        flight_ks = ks_df[ks_df['flight'] == flight].sort_values('ks_statistic', ascending=False).head(5)
        print(f"\n{flight}:")
        for _, row in flight_ks.iterrows():
            print(f"  {row['feature']}: KS={row['ks_statistic']:.4f}, p={row['p_value']:.4e}")
    
    return ks_df


def compute_mmd_divergence(df, feature_names):
    """Compute MMD between each flight and all others."""
    print("\n" + "=" * 80)
    print("Computing Maximum Mean Discrepancy (MMD)")
    print("=" * 80)
    
    flights = df['flight_name'].unique()
    mmd_results = []
    
    # Prepare feature matrix (remove metadata columns)
    X_full = df[feature_names].values
    
    # Standardize
    scaler = StandardScaler()
    X_full_scaled = scaler.fit_transform(X_full)
    
    for flight in flights:
        flight_mask = df['flight_name'] == flight
        X_flight = X_full_scaled[flight_mask]
        X_other = X_full_scaled[~flight_mask]
        
        # Compute MMD
        mmd_rbf = compute_mmd(X_flight, X_other, kernel='rbf')
        mmd_linear = compute_mmd(X_flight, X_other, kernel='linear')
        
        mmd_results.append({
            'flight': flight,
            'mmd_rbf': mmd_rbf,
            'mmd_linear': mmd_linear,
            'n_samples': len(X_flight),
        })
        
        print(f"{flight}: MMD(RBF)={mmd_rbf:.4f}, MMD(Linear)={mmd_linear:.4f}, n={len(X_flight)}")
    
    mmd_df = pd.DataFrame(mmd_results)
    
    return mmd_df


def perform_lofo_cv(df, feature_names):
    """Perform leave-one-flight-out cross-validation."""
    print("\n" + "=" * 80)
    print("Leave-One-Flight-Out Cross-Validation")
    print("=" * 80)
    
    flights = df['flight_name'].unique()
    
    lofo_results = []
    
    for test_flight in flights:
        print(f"\n--- Testing on {test_flight} ---")
        
        # Split data
        train_mask = df['flight_name'] != test_flight
        test_mask = df['flight_name'] == test_flight
        
        X_train = df.loc[train_mask, feature_names].values
        y_train = df.loc[train_mask, 'cbh_km'].values
        X_test = df.loc[test_mask, feature_names].values
        y_test = df.loc[test_mask, 'cbh_km'].values
        
        print(f"Train: {len(X_train)} samples from {df[train_mask]['flight_name'].nunique()} flights")
        print(f"Test: {len(X_test)} samples from {test_flight}")
        
        # Standardize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            subsample=0.8,
        )
        
        model.fit(X_train_scaled, y_train)
        
        # Predict
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)
        
        # Metrics
        train_r2 = r2_score(y_train, y_train_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred) * 1000  # to meters
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred)) * 1000
        
        test_r2 = r2_score(y_test, y_test_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred) * 1000
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred)) * 1000
        
        print(f"Train: R²={train_r2:.4f}, MAE={train_mae:.1f}m, RMSE={train_rmse:.1f}m")
        print(f"Test:  R²={test_r2:.4f}, MAE={test_mae:.1f}m, RMSE={test_rmse:.1f}m")
        
        lofo_results.append({
            'test_flight': test_flight,
            'n_train': len(X_train),
            'n_test': len(X_test),
            'train_r2': train_r2,
            'train_mae_m': train_mae,
            'train_rmse_m': train_rmse,
            'test_r2': test_r2,
            'test_mae_m': test_mae,
            'test_rmse_m': test_rmse,
            'y_true': y_test.tolist(),
            'y_pred': y_test_pred.tolist(),
        })
    
    lofo_df = pd.DataFrame(lofo_results)
    
    # Aggregate statistics with confidence intervals
    print("\n" + "=" * 80)
    print("LOFO-CV Aggregated Results")
    print("=" * 80)
    
    test_r2_values = lofo_df['test_r2'].values
    test_mae_values = lofo_df['test_mae_m'].values
    test_rmse_values = lofo_df['test_rmse_m'].values
    
    # Compute mean and 95% CI
    from scipy import stats
    
    def compute_ci(values, confidence=0.95):
        mean = np.mean(values)
        sem = stats.sem(values)
        ci = stats.t.interval(confidence, len(values)-1, loc=mean, scale=sem)
        return mean, ci[0], ci[1]
    
    r2_mean, r2_ci_low, r2_ci_high = compute_ci(test_r2_values)
    mae_mean, mae_ci_low, mae_ci_high = compute_ci(test_mae_values)
    rmse_mean, rmse_ci_low, rmse_ci_high = compute_ci(test_rmse_values)
    
    print(f"Test R²:    {r2_mean:.4f} (95% CI: [{r2_ci_low:.4f}, {r2_ci_high:.4f}])")
    print(f"Test MAE:   {mae_mean:.1f} m (95% CI: [{mae_ci_low:.1f}, {mae_ci_high:.1f}])")
    print(f"Test RMSE:  {rmse_mean:.1f} m (95% CI: [{rmse_ci_low:.1f}, {rmse_ci_high:.1f}])")
    
    # Statistical tests
    print("\n" + "=" * 80)
    print("Statistical Significance Tests")
    print("=" * 80)
    
    # One-sample t-test: Is R² significantly different from 0?
    t_stat, p_value = stats.ttest_1samp(test_r2_values, 0)
    print(f"One-sample t-test (H0: R² = 0):")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.4e}")
    print(f"  Result: {'Reject H0' if p_value < 0.05 else 'Fail to reject H0'} (α=0.05)")
    
    return lofo_df, {
        'r2_mean': r2_mean, 'r2_ci_low': r2_ci_low, 'r2_ci_high': r2_ci_high,
        'mae_mean': mae_mean, 'mae_ci_low': mae_ci_low, 'mae_ci_high': mae_ci_high,
        'rmse_mean': rmse_mean, 'rmse_ci_low': rmse_ci_low, 'rmse_ci_high': rmse_ci_high,
        't_stat': t_stat, 'p_value': p_value,
    }


def visualize_pca_tsne(df, feature_names):
    """Generate PCA and t-SNE visualizations colored by flight."""
    print("\n" + "=" * 80)
    print("Generating PCA and t-SNE Visualizations")
    print("=" * 80)
    
    X = df[feature_names].values
    flights = df['flight_name'].values
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA
    print("Computing PCA...")
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    print(f"PCA explained variance: {pca.explained_variance_ratio_}")
    print(f"Total variance explained: {pca.explained_variance_ratio_.sum():.4f}")
    
    # t-SNE
    print("Computing t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(X_scaled)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    unique_flights = np.unique(flights)
    colors = sns.color_palette("husl", len(unique_flights))
    
    # PCA plot
    for i, flight in enumerate(unique_flights):
        mask = flights == flight
        axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1], 
                       label=flight, alpha=0.6, s=30, c=[colors[i]])
    
    axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%})", fontsize=12)
    axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%})", fontsize=12)
    axes[0].set_title("PCA: Feature Space Colored by Flight", fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # t-SNE plot
    for i, flight in enumerate(unique_flights):
        mask = flights == flight
        axes[1].scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                       label=flight, alpha=0.6, s=30, c=[colors[i]])
    
    axes[1].set_xlabel("t-SNE Dimension 1", fontsize=12)
    axes[1].set_ylabel("t-SNE Dimension 2", fontsize=12)
    axes[1].set_title("t-SNE: Feature Space Colored by Flight", fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "domain_shift_pca_tsne.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {FIGURES_DIR / 'domain_shift_pca_tsne.png'}")


def visualize_ks_heatmap(ks_df):
    """Generate heatmap of KS statistics per feature per flight."""
    print("\n" + "=" * 80)
    print("Generating KS Divergence Heatmap")
    print("=" * 80)
    
    # Pivot to heatmap format
    ks_pivot = ks_df.pivot(index='feature', columns='flight', values='ks_statistic')
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(ks_pivot, annot=False, cmap='YlOrRd', cbar_kws={'label': 'KS Statistic'}, ax=ax)
    ax.set_title("Kolmogorov-Smirnov Divergence per Feature per Flight", fontsize=14, fontweight='bold')
    ax.set_xlabel("Flight", fontsize=12)
    ax.set_ylabel("Feature", fontsize=12)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "ks_divergence_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {FIGURES_DIR / 'ks_divergence_heatmap.png'}")


def visualize_lofo_results(lofo_df):
    """Visualize LOFO-CV results."""
    print("\n" + "=" * 80)
    print("Generating LOFO-CV Visualizations")
    print("=" * 80)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    flights = lofo_df['test_flight'].values
    r2_vals = lofo_df['test_r2'].values
    mae_vals = lofo_df['test_mae_m'].values
    rmse_vals = lofo_df['test_rmse_m'].values
    
    # R² plot
    axes[0].barh(flights, r2_vals, color='steelblue', edgecolor='black')
    axes[0].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[0].set_xlabel("R²", fontsize=12)
    axes[0].set_ylabel("Test Flight", fontsize=12)
    axes[0].set_title("LOFO-CV: R² per Flight", fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='x')
    
    # MAE plot
    axes[1].barh(flights, mae_vals, color='coral', edgecolor='black')
    axes[1].set_xlabel("MAE (m)", fontsize=12)
    axes[1].set_ylabel("Test Flight", fontsize=12)
    axes[1].set_title("LOFO-CV: MAE per Flight", fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='x')
    
    # RMSE plot
    axes[2].barh(flights, rmse_vals, color='lightgreen', edgecolor='black')
    axes[2].set_xlabel("RMSE (m)", fontsize=12)
    axes[2].set_ylabel("Test Flight", fontsize=12)
    axes[2].set_title("LOFO-CV: RMSE per Flight", fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "lofo_cv_results.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {FIGURES_DIR / 'lofo_cv_results.png'}")


def save_results(ks_df, mmd_df, lofo_df, lofo_stats):
    """Save all results to JSON and CSV."""
    print("\n" + "=" * 80)
    print("Saving Results")
    print("=" * 80)
    
    # Save KS results
    ks_df.to_csv(REPORTS_DIR / "ks_divergence_results.csv", index=False)
    print(f"Saved: {REPORTS_DIR / 'ks_divergence_results.csv'}")
    
    # Save MMD results
    mmd_df.to_csv(REPORTS_DIR / "mmd_divergence_results.csv", index=False)
    print(f"Saved: {REPORTS_DIR / 'mmd_divergence_results.csv'}")
    
    # Save LOFO results
    lofo_df.to_csv(REPORTS_DIR / "lofo_cv_results.csv", index=False)
    print(f"Saved: {REPORTS_DIR / 'lofo_cv_results.csv'}")
    
    # Save aggregated report
    report = {
        "timestamp": datetime.now().isoformat(),
        "min_samples_per_flight": MIN_SAMPLES_PER_FLIGHT,
        "lofo_statistics": lofo_stats,
        "ks_summary": {
            "total_tests": len(ks_df),
            "significant_tests": int(ks_df['significant'].sum()),
            "percent_significant": float(ks_df['significant'].mean() * 100),
        },
        "mmd_summary": {
            "mean_mmd_rbf": float(mmd_df['mmd_rbf'].mean()),
            "std_mmd_rbf": float(mmd_df['mmd_rbf'].std()),
            "mean_mmd_linear": float(mmd_df['mmd_linear'].mean()),
            "std_mmd_linear": float(mmd_df['mmd_linear'].std()),
        },
    }
    
    with open(REPORTS_DIR / "domain_shift_analysis_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"Saved: {REPORTS_DIR / 'domain_shift_analysis_report.json'}")


def main():
    """Main execution."""
    # Load data
    df, feature_names, flight_mapping = load_data()
    
    # Filter flights
    df_filtered, valid_flights = filter_flights_by_size(df, MIN_SAMPLES_PER_FLIGHT)
    
    # Compute KS divergence
    ks_df = compute_ks_divergence_per_feature(df_filtered, feature_names)
    
    # Compute MMD divergence
    mmd_df = compute_mmd_divergence(df_filtered, feature_names)
    
    # Perform LOFO-CV
    lofo_df, lofo_stats = perform_lofo_cv(df_filtered, feature_names)
    
    # Visualizations
    visualize_pca_tsne(df_filtered, feature_names)
    visualize_ks_heatmap(ks_df)
    visualize_lofo_results(lofo_df)
    
    # Save results
    save_results(ks_df, mmd_df, lofo_df, lofo_stats)
    
    print("\n" + "=" * 80)
    print("Domain Shift Analysis Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
