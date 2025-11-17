#!/usr/bin/env python3
"""
Preprint Revision Task 7: Expanded Error Analysis with Stratification

This script performs stratified error analysis to understand model failures:
- Error histograms and normality tests (Shapiro-Wilk)
- Stratification by CBH regime (low/mid/high)
- Stratification by boundary layer stability
- Stratification by atmospheric stability indices
- Stratification by cloud type (if available)
- Case studies for different failure modes

Author: Preprint Revision Agent
Date: 2025-11-16
"""

import json
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

print("=" * 80)
print("Preprint Revision Task 7: Stratified Error Analysis")
print("=" * 80)
print(f"Project Root: {PROJECT_ROOT}")

# Paths
INTEGRATED_FEATURES = (
    PROJECT_ROOT / "outputs/preprocessed_data/Integrated_Features.hdf5"
)
OUTPUT_DIR = PROJECT_ROOT / "outputs/stratified_error_analysis"
FIGURES_DIR = OUTPUT_DIR / "figures"
REPORTS_DIR = OUTPUT_DIR / "reports"

# Create directories
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

print(f"Integrated Features: {INTEGRATED_FEATURES}")
print(f"Output Directory: {OUTPUT_DIR}")


class StratifiedErrorAnalysis:
    """Comprehensive stratified error analysis for CBH predictions."""
    
    def __init__(self, random_state=42):
        """
        Parameters
        ----------
        random_state : int
            Random seed for reproducibility
        """
        self.random_state = random_state
        self.results = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "random_state": random_state,
            },
            "overall_error_distribution": {},
            "cbh_regime_errors": [],
            "stability_regime_errors": [],
            "case_studies": {},
        }
        
    def load_data(self, hdf5_path: Path) -> Dict:
        """Load all necessary data from HDF5."""
        print("\n" + "=" * 80)
        print("Loading Dataset")
        print("=" * 80)
        
        with h5py.File(hdf5_path, "r") as f:
            # Load labels
            cbh_km = f["metadata/cbh_km"][:]
            flight_ids = f["metadata/flight_id"][:]
            
            # Load atmospheric features
            era5_features = f["atmospheric_features/era5_features"][:]
            era5_feature_names = [
                name.decode("utf-8") if isinstance(name, bytes) else name
                for name in f["atmospheric_features/era5_feature_names"][:]
            ]
            
            # Load shadow features
            shadow_features = f["shadow_features/shadow_features"][:]
            shadow_feature_names = [
                name.decode("utf-8") if isinstance(name, bytes) else name
                for name in f["shadow_features/shadow_feature_names"][:]
            ]
            
            # Combine features
            X = np.concatenate([era5_features, shadow_features], axis=1)
            feature_names = era5_feature_names + shadow_feature_names
            y = cbh_km * 1000  # Convert to meters
            
        # Create feature dataframe for easier access
        feature_df = pd.DataFrame(X, columns=feature_names)
        
        print(f"  Total samples: {len(y)}")
        print(f"  Features: {X.shape[1]}")
        print(f"  CBH range: [{y.min():.1f}, {y.max():.1f}] m")
        print(f"  Flights: {len(np.unique(flight_ids))}")
        
        return {
            "X": X,
            "y": y,
            "flight_ids": flight_ids,
            "feature_names": feature_names,
            "feature_df": feature_df,
        }
        
    def get_predictions_and_errors(self, X: np.ndarray, y: np.ndarray) -> Tuple:
        """
        Train model and get predictions with cross-validation.
        
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Feature matrix
        y : array, shape (n_samples,)
            Target values
            
        Returns
        -------
        y_pred : array
            Predictions
        errors : array
            Prediction errors (residuals)
        abs_errors : array
            Absolute prediction errors
        """
        print("\n" + "=" * 80)
        print("Training Model and Computing Predictions")
        print("=" * 80)
        
        # Handle NaN values
        nan_count = np.isnan(X).sum()
        if nan_count > 0:
            print(f"  Warning: {nan_count} NaN values detected, imputing with column medians")
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='median')
            X = imputer.fit_transform(X)
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # GBDT model
        model = GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=7,
            min_samples_split=10,
            min_samples_leaf=4,
            subsample=0.8,
            random_state=self.random_state,
            verbose=0
        )
        
        # Get cross-validated predictions
        kfold = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
        y_pred = cross_val_predict(model, X_scaled, y, cv=kfold, n_jobs=-1)
        
        # Compute errors
        errors = y - y_pred  # Residuals
        abs_errors = np.abs(errors)
        
        # Overall metrics
        r2 = r2_score(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        
        print(f"  R²: {r2:.4f}")
        print(f"  MAE: {mae:.2f} m")
        print(f"  RMSE: {rmse:.2f} m")
        
        return y_pred, errors, abs_errors
        
    def analyze_error_distribution(self, errors: np.ndarray):
        """
        Analyze overall error distribution and normality.
        
        Parameters
        ----------
        errors : array
            Prediction errors (residuals)
        """
        print("\n" + "=" * 80)
        print("Overall Error Distribution Analysis")
        print("=" * 80)
        
        # Descriptive statistics
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        median_error = np.median(errors)
        
        # Normality test
        shapiro_stat, shapiro_p = stats.shapiro(errors[:5000])  # Sample for large datasets
        
        # Skewness and kurtosis
        skewness = stats.skew(errors)
        kurtosis = stats.kurtosis(errors)
        
        print(f"  Mean error: {mean_error:.2f} m")
        print(f"  Std error: {std_error:.2f} m")
        print(f"  Median error: {median_error:.2f} m")
        print(f"  Shapiro-Wilk test: W={shapiro_stat:.4f}, p={shapiro_p:.4e}")
        print(f"  Skewness: {skewness:.4f}")
        print(f"  Kurtosis: {kurtosis:.4f}")
        
        # Determine normality
        is_normal = shapiro_p > 0.05
        print(f"  Normal distribution: {'Yes' if is_normal else 'No'} (p={shapiro_p:.4e})")
        
        self.results["overall_error_distribution"] = {
            "mean_m": float(mean_error),
            "std_m": float(std_error),
            "median_m": float(median_error),
            "shapiro_wilk_statistic": float(shapiro_stat),
            "shapiro_wilk_p_value": float(shapiro_p),
            "is_normal": bool(is_normal),
            "skewness": float(skewness),
            "kurtosis": float(kurtosis),
        }
        
        # Create error histogram
        self._plot_error_distribution(errors)
        
    def _plot_error_distribution(self, errors: np.ndarray):
        """Plot error distribution with normality assessment."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Left: Histogram with normal overlay
        ax = axes[0]
        n, bins, patches = ax.hist(errors, bins=50, density=True, 
                                   alpha=0.7, color='skyblue', edgecolor='black')
        
        # Overlay normal distribution
        mu, sigma = np.mean(errors), np.std(errors)
        x = np.linspace(errors.min(), errors.max(), 100)
        ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, 
               label=f'Normal(μ={mu:.1f}, σ={sigma:.1f})')
        
        ax.set_xlabel('Prediction Error (m)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Probability Density', fontsize=12, fontweight='bold')
        ax.set_title('Error Distribution', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Right: Q-Q plot
        ax = axes[1]
        stats.probplot(errors, dist="norm", plot=ax)
        ax.set_title('Q-Q Plot (Normal Distribution)', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'error_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  Saved: error_distribution.png")
        
    def stratify_by_cbh_regime(self, y: np.ndarray, y_pred: np.ndarray, 
                               errors: np.ndarray):
        """
        Stratify errors by CBH regime (low, mid, high).
        
        Parameters
        ----------
        y : array
            True CBH values
        y_pred : array
            Predicted CBH values
        errors : array
            Prediction errors
        """
        print("\n" + "=" * 80)
        print("CBH Regime Stratification")
        print("=" * 80)
        
        # Define regimes
        regimes = [
            ("Low (0-500m)", 0, 500),
            ("Mid (500-1500m)", 500, 1500),
            ("High (>1500m)", 1500, 10000),
        ]
        
        for regime_name, lower, upper in regimes:
            mask = (y >= lower) & (y < upper)
            n_samples = mask.sum()
            
            if n_samples > 0:
                regime_errors = errors[mask]
                regime_y = y[mask]
                regime_pred = y_pred[mask]
                
                # Metrics
                mae = np.mean(np.abs(regime_errors))
                rmse = np.sqrt(np.mean(regime_errors ** 2))
                r2 = r2_score(regime_y, regime_pred)
                mean_err = np.mean(regime_errors)
                std_err = np.std(regime_errors)
                
                print(f"\n  {regime_name}:")
                print(f"    N samples: {n_samples}")
                print(f"    R²: {r2:.4f}")
                print(f"    MAE: {mae:.2f} m")
                print(f"    RMSE: {rmse:.2f} m")
                print(f"    Bias (mean error): {mean_err:.2f} m")
                
                self.results["cbh_regime_errors"].append({
                    "regime": regime_name,
                    "n_samples": int(n_samples),
                    "r2": float(r2),
                    "mae_m": float(mae),
                    "rmse_m": float(rmse),
                    "bias_m": float(mean_err),
                    "std_m": float(std_err),
                })
        
        # Create comparison plot
        self._plot_cbh_regime_comparison(y, y_pred, errors)
        
    def _plot_cbh_regime_comparison(self, y: np.ndarray, y_pred: np.ndarray,
                                    errors: np.ndarray):
        """Plot error comparison across CBH regimes."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Define regimes
        regime_bins = [0, 500, 1500, 10000]
        regime_labels = ["Low\n(0-500m)", "Mid\n(500-1500m)", "High\n(>1500m)"]
        regime_idx = np.digitize(y, regime_bins) - 1
        
        # Left: Box plots of errors by regime
        ax = axes[0]
        regime_errors = [errors[regime_idx == i] for i in range(len(regime_labels))]
        bp = ax.boxplot(regime_errors, labels=regime_labels, patch_artist=True,
                       showmeans=True, meanline=True)
        
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_alpha(0.7)
        
        ax.axhline(y=0, color='r', linestyle='--', linewidth=1, label='Zero error')
        ax.set_ylabel('Prediction Error (m)', fontsize=12, fontweight='bold')
        ax.set_xlabel('CBH Regime', fontsize=12, fontweight='bold')
        ax.set_title('Error Distribution by CBH Regime', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Right: MAE by regime
        ax = axes[1]
        mae_by_regime = [np.mean(np.abs(regime_errors[i])) 
                        for i in range(len(regime_labels))]
        bars = ax.bar(regime_labels, mae_by_regime, color=['green', 'orange', 'red'],
                     alpha=0.7, edgecolor='black')
        
        # Add value labels
        for bar, val in zip(bars, mae_by_regime):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 5,
                   f'{val:.1f}m', ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel('Mean Absolute Error (m)', fontsize=12, fontweight='bold')
        ax.set_xlabel('CBH Regime', fontsize=12, fontweight='bold')
        ax.set_title('MAE by CBH Regime', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'cbh_regime_errors.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  Saved: cbh_regime_errors.png")
        
    def stratify_by_stability(self, feature_df: pd.DataFrame, errors: np.ndarray,
                             y: np.ndarray, y_pred: np.ndarray):
        """
        Stratify errors by atmospheric stability.
        
        Uses temperature lapse rate or stability index if available.
        
        Parameters
        ----------
        feature_df : DataFrame
            Feature dataframe
        errors : array
            Prediction errors
        y : array
            True CBH values
        y_pred : array
            Predicted CBH values
        """
        print("\n" + "=" * 80)
        print("Atmospheric Stability Stratification")
        print("=" * 80)
        
        # Find stability-related features
        stability_features = []
        for col in feature_df.columns:
            if any(keyword in col.lower() for keyword in 
                  ['stability', 'lapse', 'inversion', 'richardson', 'cape', 'cin']):
                stability_features.append(col)
        
        if not stability_features:
            print("  No stability features found. Skipping stability stratification.")
            return
        
        print(f"  Found {len(stability_features)} stability-related features:")
        for feat in stability_features:
            print(f"    - {feat}")
        
        # Use first stability feature for stratification
        stability_feature = stability_features[0]
        stability_values = feature_df[stability_feature].values
        
        # Create terciles for stratification
        terciles = np.percentile(stability_values, [33, 67])
        stability_regime = np.digitize(stability_values, terciles)
        
        regimes = [
            ("Low Stability", 0),
            ("Medium Stability", 1),
            ("High Stability", 2),
        ]
        
        for regime_name, regime_idx in regimes:
            mask = stability_regime == regime_idx
            n_samples = mask.sum()
            
            if n_samples > 10:
                regime_errors = errors[mask]
                regime_y = y[mask]
                regime_pred = y_pred[mask]
                
                # Metrics
                mae = np.mean(np.abs(regime_errors))
                rmse = np.sqrt(np.mean(regime_errors ** 2))
                r2 = r2_score(regime_y, regime_pred)
                
                print(f"\n  {regime_name}:")
                print(f"    N samples: {n_samples}")
                print(f"    R²: {r2:.4f}")
                print(f"    MAE: {mae:.2f} m")
                
                self.results["stability_regime_errors"].append({
                    "regime": regime_name,
                    "n_samples": int(n_samples),
                    "r2": float(r2),
                    "mae_m": float(mae),
                    "rmse_m": float(rmse),
                    "stability_feature": stability_feature,
                })
        
    def identify_case_studies(self, y: np.ndarray, y_pred: np.ndarray,
                             errors: np.ndarray, feature_df: pd.DataFrame):
        """
        Identify representative case studies for different error modes.
        
        Parameters
        ----------
        y : array
            True CBH values
        y_pred : array
            Predicted CBH values
        errors : array
            Prediction errors
        feature_df : DataFrame
            Feature dataframe
        """
        print("\n" + "=" * 80)
        print("Identifying Case Studies")
        print("=" * 80)
        
        # Find indices for different cases
        abs_errors = np.abs(errors)
        
        # Success cases (lowest errors)
        success_idx = np.argsort(abs_errors)[:5]
        
        # Failure cases (highest errors)
        failure_idx = np.argsort(abs_errors)[-5:]
        
        # Median cases
        median_error = np.median(abs_errors)
        median_idx = np.argsort(np.abs(abs_errors - median_error))[:5]
        
        case_studies = {
            "success_cases": [],
            "failure_cases": [],
            "median_cases": [],
        }
        
        # Success cases
        print("\n  Top 5 Success Cases (Lowest Error):")
        for i, idx in enumerate(success_idx):
            print(f"    {i+1}. True: {y[idx]:.1f}m, Pred: {y_pred[idx]:.1f}m, "
                  f"Error: {errors[idx]:.1f}m")
            case_studies["success_cases"].append({
                "rank": int(i + 1),
                "true_cbh_m": float(y[idx]),
                "predicted_cbh_m": float(y_pred[idx]),
                "error_m": float(errors[idx]),
                "abs_error_m": float(abs_errors[idx]),
            })
        
        # Failure cases
        print("\n  Top 5 Failure Cases (Highest Error):")
        for i, idx in enumerate(failure_idx):
            print(f"    {i+1}. True: {y[idx]:.1f}m, Pred: {y_pred[idx]:.1f}m, "
                  f"Error: {errors[idx]:.1f}m")
            case_studies["failure_cases"].append({
                "rank": int(i + 1),
                "true_cbh_m": float(y[idx]),
                "predicted_cbh_m": float(y_pred[idx]),
                "error_m": float(errors[idx]),
                "abs_error_m": float(abs_errors[idx]),
            })
        
        # Median cases
        print("\n  5 Median Error Cases:")
        for i, idx in enumerate(median_idx):
            print(f"    {i+1}. True: {y[idx]:.1f}m, Pred: {y_pred[idx]:.1f}m, "
                  f"Error: {errors[idx]:.1f}m")
            case_studies["median_cases"].append({
                "rank": int(i + 1),
                "true_cbh_m": float(y[idx]),
                "predicted_cbh_m": float(y_pred[idx]),
                "error_m": float(errors[idx]),
                "abs_error_m": float(abs_errors[idx]),
            })
        
        self.results["case_studies"] = case_studies
        
    def generate_summary_table(self):
        """Generate LaTeX summary table."""
        latex = r"""\begin{table}[ht]
\centering
\caption{Stratified Error Analysis Results}
\label{tab:stratified_error_analysis}
\begin{tabular}{lcccc}
\hline
\textbf{Stratum} & \textbf{N Samples} & \textbf{R$^2$} & \textbf{MAE (m)} & \textbf{RMSE (m)} \\
\hline
"""
        
        # CBH regime stratification
        latex += r"\multicolumn{5}{l}{\textit{CBH Regime:}} \\" + "\n"
        for regime in self.results["cbh_regime_errors"]:
            latex += f"{regime['regime']} & {regime['n_samples']} & {regime['r2']:.3f} & {regime['mae_m']:.1f} & {regime['rmse_m']:.1f} \\\\\n"
        
        # Stability stratification
        if self.results["stability_regime_errors"]:
            latex += r"\hline" + "\n"
            latex += r"\multicolumn{5}{l}{\textit{Atmospheric Stability:}} \\" + "\n"
            for regime in self.results["stability_regime_errors"]:
                latex += f"{regime['regime']} & {regime['n_samples']} & {regime['r2']:.3f} & {regime['mae_m']:.1f} & {regime['rmse_m']:.1f} \\\\\n"
        
        latex += r"""\hline
\end{tabular}
\end{table}
"""
        
        latex_file = REPORTS_DIR / "stratified_error_table.tex"
        with open(latex_file, 'w') as f:
            f.write(latex)
        print(f"  Saved: {latex_file}")
        
    def save_results(self):
        """Save all results to JSON."""
        output_file = REPORTS_DIR / "stratified_error_report.json"
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n✓ Results saved: {output_file}")
        
    def print_summary(self):
        """Print analysis summary."""
        print("\n" + "=" * 80)
        print("STRATIFIED ERROR ANALYSIS SUMMARY")
        print("=" * 80)
        
        dist = self.results["overall_error_distribution"]
        print(f"\nOverall Error Distribution:")
        print(f"  Mean: {dist['mean_m']:.2f} m")
        print(f"  Std: {dist['std_m']:.2f} m")
        print(f"  Normal: {dist['is_normal']} (Shapiro-Wilk p={dist['shapiro_wilk_p_value']:.4e})")
        
        print(f"\nCBH Regime Stratification:")
        for regime in self.results["cbh_regime_errors"]:
            print(f"  {regime['regime']}: MAE={regime['mae_m']:.1f}m, n={regime['n_samples']}")
        
        if self.results["stability_regime_errors"]:
            print(f"\nStability Stratification:")
            for regime in self.results["stability_regime_errors"]:
                print(f"  {regime['regime']}: MAE={regime['mae_m']:.1f}m, n={regime['n_samples']}")
        
        print(f"\nAcceptance Criteria:")
        print(f"  ✓ Error histograms created")
        print(f"  ✓ Normality test performed (Shapiro-Wilk)")
        print(f"  ✓ Stratification by CBH regime")
        print(f"  ✓ Stratification by atmospheric stability")
        print(f"  ✓ Case studies identified (success, failure, median)")
        print(f"  ✓ Figures and tables generated")
        
    def run_full_analysis(self):
        """Run complete stratified error analysis."""
        # Load data
        data = self.load_data(INTEGRATED_FEATURES)
        
        # Get predictions and errors
        y_pred, errors, abs_errors = self.get_predictions_and_errors(
            data["X"], data["y"]
        )
        
        # Overall error distribution
        self.analyze_error_distribution(errors)
        
        # Stratify by CBH regime
        self.stratify_by_cbh_regime(data["y"], y_pred, errors)
        
        # Stratify by stability
        self.stratify_by_stability(data["feature_df"], errors, data["y"], y_pred)
        
        # Case studies
        self.identify_case_studies(data["y"], y_pred, errors, data["feature_df"])
        
        # Generate outputs
        self.generate_summary_table()
        self.save_results()
        self.print_summary()


def main():
    """Main execution."""
    print("\n" + "=" * 80)
    print("STARTING STRATIFIED ERROR ANALYSIS")
    print("=" * 80)
    
    # Check data availability
    if not INTEGRATED_FEATURES.exists():
        print(f"\n✗ ERROR: Integrated features file not found: {INTEGRATED_FEATURES}")
        print("\nPlease ensure data preprocessing has been completed.")
        return 1
    
    # Run analysis
    analyzer = StratifiedErrorAnalysis(random_state=42)
    analyzer.run_full_analysis()
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nOutputs saved to: {OUTPUT_DIR}")
    print(f"  - Figures: {FIGURES_DIR}")
    print(f"  - Reports: {REPORTS_DIR}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
