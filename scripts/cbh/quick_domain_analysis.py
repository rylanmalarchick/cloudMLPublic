#!/usr/bin/env python3
"""
Quick Domain Shift Analysis for CBH Predictions
Streamlined version focusing on LOFO validation and KS divergence
"""

import json
import sys
import warnings
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ks_2samp
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

print("=" * 80)
print("Quick Domain Shift Analysis for CBH Predictions")
print("=" * 80)

# Paths
INTEGRATED_FEATURES = PROJECT_ROOT / "outputs/preprocessed_data/Integrated_Features.hdf5"
OUTPUT_DIR = PROJECT_ROOT / "outputs/domain_analysis"
FIGURES_DIR = OUTPUT_DIR / "figures"
REPORTS_DIR = OUTPUT_DIR / "reports"
TABLES_DIR = OUTPUT_DIR / "tables"

for dir in [FIGURES_DIR, REPORTS_DIR, TABLES_DIR]:
    dir.mkdir(parents=True, exist_ok=True)

MIN_SAMPLES_PER_FLIGHT = 60

def load_data():
    """Load data with actual structure."""
    print("\nLoading data...")
    
    with h5py.File(INTEGRATED_FEATURES, "r") as f:
        # Load labels and metadata
        cbh_km = f["metadata/cbh_km"][:]
        flight_ids = f["metadata/flight_id"][:]
        
        # Load atmospheric features (ERA5)
        era5_features = f["atmospheric_features/era5_features"][:]
        era5_names = [n.decode() if isinstance(n, bytes) else n 
                     for n in f["atmospheric_features/era5_feature_names"][:]]
        
        # Load shadow features
        shadow_features = f["shadow_features/shadow_features"][:]
        shadow_names = [n.decode() if isinstance(n, bytes) else n 
                       for n in f["shadow_features/shadow_feature_names"][:]]
        
        # Combine features
        X = np.concatenate([era5_features, shadow_features], axis=1)
        feature_names = era5_names + shadow_names
        y = cbh_km * 1000  # Convert to meters
        
    # Decode flight IDs if they're bytes
    if len(flight_ids) > 0 and isinstance(flight_ids[0], bytes):
        flight_ids = np.array([fid.decode() if isinstance(fid, bytes) else str(fid) for fid in flight_ids])
    else:
        flight_ids = np.array([str(fid) for fid in flight_ids])
    
    print(f"  Loaded {X.shape[0]} samples with {X.shape[1]} features")
    print(f"  CBH range: {y.min():.0f} - {y.max():.0f} m")
    
    return X, y, flight_ids, feature_names

def analyze_flights(X, y, flight_ids):
    """Analyze flight distribution and identify small flights."""
    print("\nAnalyzing flight distribution...")
    
    unique_flights = np.unique(flight_ids)
    flight_stats = []
    
    for flight in unique_flights:
        mask = flight_ids == flight
        n = mask.sum()
        cbh_vals = y[mask]
        
        flight_stats.append({
            'flight_id': flight,
            'n_samples': int(n),
            'mean_cbh_m': float(np.mean(cbh_vals)),
            'std_cbh_m': float(np.std(cbh_vals)),
            'min_cbh_m': float(np.min(cbh_vals)),
            'max_cbh_m': float(np.max(cbh_vals)),
            'include_in_lofo': n >= MIN_SAMPLES_PER_FLIGHT
        })
        
        print(f"  {flight}: n={n:3d}, mean={np.mean(cbh_vals):.0f}m, "
              f"std={np.std(cbh_vals):.0f}m, include={n >= MIN_SAMPLES_PER_FLIGHT}")
    
    return pd.DataFrame(flight_stats)

def leave_one_flight_out_cv(X, y, flight_ids, flight_stats_df):
    """Perform leave-one-flight-out cross-validation."""
    print("\nPerforming leave-one-flight-out cross-validation...")
    
    # Filter to flights with enough samples
    valid_flights = flight_stats_df[flight_stats_df['include_in_lofo']]['flight_id'].values
    
    if len(valid_flights) == 0:
        print("  Warning: No flights meet minimum sample criteria")
        return {}
    
    lofo_results = []
    
    for test_flight in valid_flights:
        # Split data
        test_mask = flight_ids == test_flight
        train_mask = ~test_mask
        
        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
        
        # Remove NaN
        train_valid = ~np.isnan(X_train).any(axis=1) & ~np.isnan(y_train)
        test_valid = ~np.isnan(X_test).any(axis=1) & ~np.isnan(y_test)
        
        X_train = X_train[train_valid]
        y_train = y_train[train_valid]
        X_test = X_test[test_valid]
        y_test = y_test[test_valid]
        
        if len(X_test) < 10:
            continue
        
        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=4,
            subsample=0.8,
            random_state=42
        )
        
        model.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred = model.predict(X_test_scaled)
        
        # Evaluate
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        lofo_results.append({
            'flight_id': test_flight,
            'n_test': int(len(y_test)),
            'n_train': int(len(y_train)),
            'r2': float(r2),
            'mae_m': float(mae),
            'rmse_m': float(rmse)
        })
        
        print(f"  {test_flight}: R²={r2:.3f}, MAE={mae:.1f}m (n_test={len(y_test)})")
    
    return lofo_results

def compute_ks_divergence(X, flight_ids, feature_names, flight_stats_df):
    """Compute Kolmogorov-Smirnov divergence across flights."""
    print("\nComputing K-S divergence across flights...")
    
    valid_flights = flight_stats_df[flight_stats_df['include_in_lofo']]['flight_id'].values
    
    if len(valid_flights) < 2:
        print("  Warning: Need at least 2 flights for KS analysis")
        return []
    
    ks_results = []
    
    # Compare all pairs of flights
    for i, flight_a in enumerate(valid_flights):
        for flight_b in valid_flights[i+1:]:
            mask_a = flight_ids == flight_a
            mask_b = flight_ids == flight_b
            
            X_a = X[mask_a]
            X_b = X[mask_b]
            
            # For each feature, compute KS statistic
            for feat_idx, feat_name in enumerate(feature_names):
                # Remove NaN
                vals_a = X_a[:, feat_idx]
                vals_b = X_b[:, feat_idx]
                
                vals_a = vals_a[~np.isnan(vals_a)]
                vals_b = vals_b[~np.isnan(vals_b)]
                
                if len(vals_a) < 5 or len(vals_b) < 5:
                    continue
                
                ks_stat, p_value = ks_2samp(vals_a, vals_b)
                
                ks_results.append({
                    'feature': feat_name,
                    'flight_a': flight_a,
                    'flight_b': flight_b,
                    'ks_statistic': float(ks_stat),
                    'p_value': float(p_value),
                    'significant': p_value < 0.001
                })
    
    print(f"  Computed {len(ks_results)} pairwise comparisons")
    
    return ks_results

def create_ks_heatmap(ks_results_df, feature_names):
    """Create heatmap of K-S statistics."""
    print("\nCreating K-S divergence heatmap...")
    
    if len(ks_results_df) == 0:
        print("  Skipping: no KS results")
        return
    
    # Get top features by average KS statistic
    feature_avg_ks = ks_results_df.groupby('feature')['ks_statistic'].mean().sort_values(ascending=False)
    top_features = feature_avg_ks.head(10).index.tolist()
    
    # Create pivot table for heatmap
    ks_subset = ks_results_df[ks_results_df['feature'].isin(top_features)]
    
    # Create flight pair labels
    ks_subset['flight_pair'] = ks_subset['flight_a'] + ' vs ' + ks_subset['flight_b']
    
    # Pivot
    heatmap_data = ks_subset.pivot_table(
        index='feature',
        columns='flight_pair',
        values='ks_statistic',
        aggfunc='mean'
    )
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='RdYlGn_r', 
                cbar_kws={'label': 'KS Statistic'}, ax=ax)
    ax.set_title('K-S Divergence Across Flights\n(Top 10 Most Divergent Features)', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Flight Pair', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "ks_divergence_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {FIGURES_DIR / 'ks_divergence_heatmap.png'}")

def create_pca_visualization(X, y, flight_ids, flight_stats_df):
    """Create PCA visualization of flight clustering."""
    print("\nCreating PCA flight clustering visualization...")
    
    # Remove NaN
    valid = ~np.isnan(X).any(axis=1)
    X_valid = X[valid]
    flight_ids_valid = flight_ids[valid]
    
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_valid)
    
    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    unique_flights = np.unique(flight_ids_valid)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_flights)))
    
    for i, flight in enumerate(unique_flights):
        mask = flight_ids_valid == flight
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                  label=flight, alpha=0.6, s=30, c=[colors[i]])
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', fontsize=12)
    ax.set_title('Feature Distribution by Flight (PCA)', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "pca_flight_clustering.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {FIGURES_DIR / 'pca_flight_clustering.png'}")
    print(f"  PC1 explains {pca.explained_variance_ratio_[0]*100:.1f}% variance")
    print(f"  PC2 explains {pca.explained_variance_ratio_[1]*100:.1f}% variance")
    
    return {
        'pc1_variance': float(pca.explained_variance_ratio_[0]),
        'pc2_variance': float(pca.explained_variance_ratio_[1]),
        'total_variance_2d': float(pca.explained_variance_ratio_[:2].sum())
    }

def main():
    # Load data
    X, y, flight_ids, feature_names = load_data()
    
    # Analyze flights
    flight_stats_df = analyze_flights(X, y, flight_ids)
    
    # Save flight stats table
    flight_stats_df.to_csv(TABLES_DIR / "flight_statistics.csv", index=False)
    print(f"\n✓ Saved: {TABLES_DIR / 'flight_statistics.csv'}")
    
    # Leave-one-flight-out CV
    lofo_results = leave_one_flight_out_cv(X, y, flight_ids, flight_stats_df)
    
    if lofo_results:
        lofo_df = pd.DataFrame(lofo_results)
        lofo_df.to_csv(TABLES_DIR / "lofo_results.csv", index=False)
        print(f"\n✓ Saved: {TABLES_DIR / 'lofo_results.csv'}")
    
    # K-S divergence
    ks_results = compute_ks_divergence(X, flight_ids, feature_names, flight_stats_df)
    
    if ks_results:
        ks_df = pd.DataFrame(ks_results)
        ks_df.to_csv(TABLES_DIR / "ks_divergence.csv", index=False)
        print(f"✓ Saved: {TABLES_DIR / 'ks_divergence.csv'}")
        
        # Create heatmap
        create_ks_heatmap(ks_df, feature_names)
    
    # PCA visualization
    pca_stats = create_pca_visualization(X, y, flight_ids, flight_stats_df)
    
    # Save comprehensive report
    report = {
        'flight_statistics': flight_stats_df.to_dict('records'),
        'lofo_results': lofo_results if lofo_results else [],
        'pca_analysis': pca_stats,
        'ks_summary': {
            'n_comparisons': len(ks_results) if ks_results else 0,
            'top_divergent_features': (
                pd.DataFrame(ks_results).groupby('feature')['ks_statistic']
                .mean().sort_values(ascending=False).head(5).to_dict()
                if ks_results else {}
            )
        },
        'metadata': {
            'min_samples_per_flight': MIN_SAMPLES_PER_FLIGHT,
            'n_total_samples': int(len(X)),
            'n_features': int(X.shape[1]),
            'feature_names': feature_names,
            'timestamp': pd.Timestamp.now().isoformat()
        }
    }
    
    report_path = REPORTS_DIR / "domain_analysis_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✓ Report saved: {report_path}")
    print("=" * 80)
    print("Domain Shift Analysis Complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()
