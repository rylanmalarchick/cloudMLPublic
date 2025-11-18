#!/usr/bin/env python3
"""
Physics-Informed Regularization Experiment for CBH Retrieval

Tests whether adding LCL-based regularization improves model performance
and physical constraint adherence, particularly for LOFO cross-flight validation.
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs" / "physics_informed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Physics constants
GRAVITY = 9.81  # m/s^2
R_DRY = 287.05  # J/(kg·K) - specific gas constant for dry air
CP = 1005.0  # J/(kg·K) - specific heat at constant pressure


def calculate_lcl_bolton(t2m_K, d2m_K, sp_Pa):
    """
    Calculate Lifting Condensation Level using Bolton (1980) formula.
    
    Parameters:
    -----------
    t2m_K : array-like
        2-meter temperature in Kelvin
    d2m_K : array-like
        2-meter dewpoint temperature in Kelvin
    sp_Pa : array-like
        Surface pressure in Pascals
    
    Returns:
    --------
    lcl_height : array-like
        LCL height in meters
    """
    # Temperature in Celsius
    t2m_C = t2m_K - 273.15
    d2m_C = d2m_K - 273.15
    
    # LCL temperature (Bolton 1980)
    T_lcl = 1.0 / (1.0 / (d2m_C + 273.15 - 56.0) + 
                   np.log(t2m_K / (d2m_K + 0.01)) / 800.0) + 56.0 - 273.15
    
    # LCL pressure (assuming dry adiabatic ascent)
    theta = t2m_K * (100000.0 / sp_Pa) ** (R_DRY / CP)
    P_lcl = 100000.0 * (T_lcl / theta) ** (CP / R_DRY)
    
    # LCL height using hypsometric equation
    T_avg = (t2m_K + (T_lcl + 273.15)) / 2.0
    lcl_height = (R_DRY * T_avg / GRAVITY) * np.log(sp_Pa / P_lcl)
    
    return lcl_height


class PhysicsInformedGBDT:
    """GBDT with physics-informed regularization."""
    
    def __init__(self, base_params, lcl_penalty_weight=0.0):
        """
        Parameters:
        -----------
        base_params : dict
            GBDT hyperparameters
        lcl_penalty_weight : float
            Weight for LCL deviation penalty (0 = no penalty, >0 = penalty)
        """
        self.base_params = base_params
        self.lcl_penalty_weight = lcl_penalty_weight
        self.model = GradientBoostingRegressor(**base_params)
        
    def fit(self, X, y, lcl_values):
        """Train with optional LCL-based sample weighting."""
        if self.lcl_penalty_weight > 0:
            # Calculate weights based on LCL proximity
            # Penalize samples where CBH is far from LCL
            lcl_deviation = np.abs(y - lcl_values)
            weights = np.exp(-self.lcl_penalty_weight * lcl_deviation / 1000.0)
            weights = weights / weights.sum() * len(weights)  # Normalize
            self.model.fit(X, y, sample_weight=weights)
        else:
            self.model.fit(X, y)
        return self
    
    def predict(self, X):
        return self.model.predict(X)
    
    def apply_physics_constraints(self, predictions, lcl_values, alpha=0.0):
        """
        Apply physics-informed post-processing.
        
        Parameters:
        -----------
        predictions : array-like
            Raw model predictions
        lcl_values : array-like
            LCL estimates
        alpha : float
            Blending weight (0 = no adjustment, 1 = pure LCL)
        """
        if alpha > 0:
            return (1 - alpha) * predictions + alpha * lcl_values
        return predictions


def load_data():
    """Load preprocessed dataset with features and LCL calculations."""
    print("Loading data...")
    
    # Load main dataset
    df = pd.read_csv(DATA_DIR / "processed_cbh_data.csv")
    
    # Calculate LCL
    df['lcl_height'] = calculate_lcl_bolton(
        df['t2m'].values,
        df['d2m'].values,
        df['sp'].values
    )
    
    print(f"Loaded {len(df)} samples")
    print(f"LCL range: {df['lcl_height'].min():.1f} - {df['lcl_height'].max():.1f} m")
    
    return df


def run_physics_informed_experiment(df, feature_cols, penalty_weights, blend_alphas):
    """
    Test physics-informed regularization with multiple hyperparameters.
    
    Parameters:
    -----------
    df : DataFrame
        Dataset with features and LCL
    feature_cols : list
        Feature column names
    penalty_weights : list
        LCL penalty weights to test
    blend_alphas : list
        Post-processing blend weights to test
    """
    results = []
    
    # Base GBDT parameters (from best model)
    base_params = {
        'n_estimators': 200,
        'learning_rate': 0.05,
        'max_depth': 5,
        'min_samples_split': 10,
        'min_samples_leaf': 5,
        'subsample': 0.8,
        'random_state': 42
    }
    
    # Get flight IDs
    flights = df['flight_id'].unique()
    print(f"\nFlights: {list(flights)}")
    
    # Leave-One-Flight-Out validation
    for test_flight in flights:
        print(f"\n{'='*80}")
        print(f"Testing on Flight: {test_flight}")
        print(f"{'='*80}")
        
        # Split data
        train_mask = df['flight_id'] != test_flight
        test_mask = df['flight_id'] == test_flight
        
        X_train = df.loc[train_mask, feature_cols].values
        y_train = df.loc[train_mask, 'cbh_m'].values
        lcl_train = df.loc[train_mask, 'lcl_height'].values
        
        X_test = df.loc[test_mask, feature_cols].values
        y_test = df.loc[test_mask, 'cbh_m'].values
        lcl_test = df.loc[test_mask, 'lcl_height'].values
        
        # Test configurations
        for penalty in penalty_weights:
            for alpha in blend_alphas:
                # Train model
                model = PhysicsInformedGBDT(base_params, lcl_penalty_weight=penalty)
                model.fit(X_train, y_train, lcl_train)
                
                # Predict
                y_pred_raw = model.predict(X_test)
                y_pred = model.apply_physics_constraints(y_pred_raw, lcl_test, alpha=alpha)
                
                # Evaluate
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                
                # LCL correlation
                lcl_corr, lcl_p = pearsonr(y_pred, lcl_test)
                
                # Physics constraints
                violations_high = np.sum(y_pred > 12000)  # Tropopause
                violations_low = np.sum(y_pred < 0)  # Below surface
                
                results.append({
                    'test_flight': test_flight,
                    'n_train': len(y_train),
                    'n_test': len(y_test),
                    'penalty_weight': penalty,
                    'blend_alpha': alpha,
                    'r2': r2,
                    'mae': mae,
                    'rmse': rmse,
                    'lcl_corr': lcl_corr,
                    'lcl_p': lcl_p,
                    'violations_high': violations_high,
                    'violations_low': violations_low,
                    'violation_rate': (violations_high + violations_low) / len(y_test)
                })
                
                print(f"  Penalty={penalty:.3f}, Alpha={alpha:.2f}: "
                      f"R²={r2:.3f}, MAE={mae:.1f}m, LCL_corr={lcl_corr:.3f}")
    
    return pd.DataFrame(results)


def analyze_results(results_df):
    """Analyze and visualize physics-informed regularization results."""
    print("\n" + "="*80)
    print("PHYSICS-INFORMED REGULARIZATION ANALYSIS")
    print("="*80)
    
    # Baseline (no regularization)
    baseline = results_df[(results_df['penalty_weight'] == 0) & 
                          (results_df['blend_alpha'] == 0)]
    
    print("\nBaseline (No Regularization) - LOFO Results:")
    print(f"  Mean R²: {baseline['r2'].mean():.3f} ± {baseline['r2'].std():.3f}")
    print(f"  Mean MAE: {baseline['mae'].mean():.1f} ± {baseline['mae'].std():.1f} m")
    print(f"  Violation Rate: {baseline['violation_rate'].mean():.3%}")
    
    # Best configuration by R²
    best_r2_idx = results_df['r2'].idxmax()
    best_r2_config = results_df.loc[best_r2_idx]
    
    print("\nBest Configuration (by R²):")
    print(f"  Penalty Weight: {best_r2_config['penalty_weight']}")
    print(f"  Blend Alpha: {best_r2_config['blend_alpha']}")
    print(f"  Mean R²: {results_df[
        (results_df['penalty_weight'] == best_r2_config['penalty_weight']) &
        (results_df['blend_alpha'] == best_r2_config['blend_alpha'])
    ]['r2'].mean():.3f}")
    
    # Aggregate by configuration
    agg_results = results_df.groupby(['penalty_weight', 'blend_alpha']).agg({
        'r2': ['mean', 'std'],
        'mae': ['mean', 'std'],
        'lcl_corr': ['mean', 'std'],
        'violation_rate': ['mean', 'std']
    }).round(3)
    
    print("\nAggregated Results by Configuration:")
    print(agg_results.to_string())
    
    # Save results
    results_df.to_csv(OUTPUT_DIR / "physics_informed_results.csv", index=False)
    agg_results.to_csv(OUTPUT_DIR / "physics_informed_summary.csv")
    
    # Generate plots
    plot_regularization_impact(results_df, baseline)
    
    return results_df, baseline


def plot_regularization_impact(results_df, baseline):
    """Visualize impact of physics-informed regularization."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Aggregate by configuration
    agg = results_df.groupby(['penalty_weight', 'blend_alpha']).agg({
        'r2': 'mean',
        'mae': 'mean',
        'lcl_corr': 'mean',
        'violation_rate': 'mean'
    }).reset_index()
    
    baseline_r2 = baseline['r2'].mean()
    baseline_mae = baseline['mae'].mean()
    
    # Plot 1: R² improvement
    ax1 = axes[0, 0]
    for alpha in sorted(agg['blend_alpha'].unique()):
        subset = agg[agg['blend_alpha'] == alpha]
        ax1.plot(subset['penalty_weight'], subset['r2'], marker='o', 
                label=f'α={alpha:.2f}')
    ax1.axhline(y=baseline_r2, color='red', linestyle='--', 
               label=f'Baseline (R²={baseline_r2:.3f})')
    ax1.set_xlabel('LCL Penalty Weight', fontweight='bold')
    ax1.set_ylabel('Mean R² (LOFO)', fontweight='bold')
    ax1.set_title('Physics-Informed Regularization Impact on R²', fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Plot 2: MAE improvement
    ax2 = axes[0, 1]
    for alpha in sorted(agg['blend_alpha'].unique()):
        subset = agg[agg['blend_alpha'] == alpha]
        ax2.plot(subset['penalty_weight'], subset['mae'], marker='o',
                label=f'α={alpha:.2f}')
    ax2.axhline(y=baseline_mae, color='red', linestyle='--',
               label=f'Baseline (MAE={baseline_mae:.1f}m)')
    ax2.set_xlabel('LCL Penalty Weight', fontweight='bold')
    ax2.set_ylabel('Mean MAE (m, LOFO)', fontweight='bold')
    ax2.set_title('Physics-Informed Regularization Impact on MAE', fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # Plot 3: LCL correlation
    ax3 = axes[1, 0]
    for alpha in sorted(agg['blend_alpha'].unique()):
        subset = agg[agg['blend_alpha'] == alpha]
        ax3.plot(subset['penalty_weight'], subset['lcl_corr'], marker='o',
                label=f'α={alpha:.2f}')
    ax3.set_xlabel('LCL Penalty Weight', fontweight='bold')
    ax3.set_ylabel('Mean LCL Correlation', fontweight='bold')
    ax3.set_title('LCL Correlation vs Regularization', fontweight='bold')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # Plot 4: Violation rate
    ax4 = axes[1, 1]
    for alpha in sorted(agg['blend_alpha'].unique()):
        subset = agg[agg['blend_alpha'] == alpha]
        ax4.plot(subset['penalty_weight'], subset['violation_rate'] * 100, marker='o',
                label=f'α={alpha:.2f}')
    ax4.set_xlabel('LCL Penalty Weight', fontweight='bold')
    ax4.set_ylabel('Constraint Violation Rate (%)', fontweight='bold')
    ax4.set_title('Physics Constraint Adherence', fontweight='bold')
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "physics_regularization_impact.png", dpi=300, bbox_inches='tight')
    print(f"\nSaved figure: {OUTPUT_DIR / 'physics_regularization_impact.png'}")
    plt.close()


def main():
    """Run physics-informed regularization experiment."""
    print("="*80)
    print("PHYSICS-INFORMED REGULARIZATION EXPERIMENT")
    print("="*80)
    
    # Load data
    df = load_data()
    
    # Define features (same as main model)
    feature_cols = [
        't2m', 'd2m', 'sp', 'tcwv', 'u10', 'v10', 'blh',
        'stability_index', 'moisture_gradient',
        'shadow_length_m', 'shadow_angle_deg', 'sza_deg',
        'sun_azimuth_deg', 'zenith_cos', 'pixel_brightness_mean'
    ]
    
    # Test configurations
    penalty_weights = [0.0, 0.001, 0.01, 0.1, 0.5, 1.0]
    blend_alphas = [0.0, 0.1, 0.2, 0.3]
    
    print(f"\nTesting {len(penalty_weights)} penalty weights × {len(blend_alphas)} blend alphas")
    print(f"Total configurations: {len(penalty_weights) * len(blend_alphas)}")
    
    # Run experiment
    results_df = run_physics_informed_experiment(
        df, feature_cols, penalty_weights, blend_alphas
    )
    
    # Analyze
    analyze_results(results_df)
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)
    print(f"Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
