#!/usr/bin/env python3
"""
Preprint Revision Task 5: Comprehensive Feature Ablation Study

This script performs exhaustive feature ablation analysis to understand:
1. Contribution of atmospheric vs. shadow features
2. Impact of removing top SHAP features individually
3. Feature redundancy and correlations
4. Minimal feature sets for acceptable performance

Ablation Studies:
- Atmospheric features only
- Shadow features only  
- All features (baseline)
- Remove top-5 SHAP features one at a time
- Feature correlation analysis and redundancy identification

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
import shap
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

print("=" * 80)
print("Preprint Revision Task 5: Comprehensive Feature Ablation Study")
print("=" * 80)
print(f"Project Root: {PROJECT_ROOT}")

# Paths
INTEGRATED_FEATURES = (
    PROJECT_ROOT / "outputs/preprocessed_data/Integrated_Features.hdf5"
)
OUTPUT_DIR = PROJECT_ROOT / "outputs/feature_ablation"
FIGURES_DIR = OUTPUT_DIR / "figures"
REPORTS_DIR = OUTPUT_DIR / "reports"

# Create directories
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

print(f"Integrated Features: {INTEGRATED_FEATURES}")
print(f"Output Directory: {OUTPUT_DIR}")


class FeatureAblationAnalysis:
    """Comprehensive feature ablation and redundancy analysis."""
    
    def __init__(self, n_folds=5, random_state=42):
        """
        Parameters
        ----------
        n_folds : int
            Number of cross-validation folds
        random_state : int
            Random seed for reproducibility
        """
        self.n_folds = n_folds
        self.random_state = random_state
        self.results = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "n_folds": n_folds,
                "random_state": random_state,
            },
            "baseline": {},
            "feature_group_ablation": [],
            "top_feature_ablation": [],
            "feature_importance": {},
            "feature_correlations": {},
        }
        
    def load_data(self, hdf5_path: Path) -> Tuple:
        """Load features and labels from HDF5."""
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
            
            # Convert CBH to meters
            y = cbh_km * 1000
            
        print(f"  Total samples: {len(y)}")
        print(f"  Atmospheric features: {len(era5_feature_names)}")
        print(f"  Shadow features: {len(shadow_feature_names)}")
        print(f"  CBH range: [{y.min():.1f}, {y.max():.1f}] m")
        
        return (era5_features, shadow_features, y, flight_ids,
                era5_feature_names, shadow_feature_names)
        
    def train_and_evaluate(self, X: np.ndarray, y: np.ndarray, 
                          name: str = "model") -> Dict:
        """
        Train GBDT model with cross-validation and return metrics.
        
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Feature matrix
        y : array, shape (n_samples,)
            Target values
        name : str
            Model identifier for logging
            
        Returns
        -------
        results : dict
            Cross-validation metrics
        """
        # Standardize features
        # Handle NaN values first
        nan_count = np.isnan(X).sum()
        if nan_count > 0:
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='median')
            X = imputer.fit_transform(X)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Cross-validation
        kfold = KFold(n_splits=self.n_folds, shuffle=True, 
                     random_state=self.random_state)
        
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
        
        # Compute CV scores
        r2_scores = cross_val_score(model, X_scaled, y, cv=kfold,
                                   scoring='r2', n_jobs=-1)
        
        # Compute MAE using cross_val_predict for more accurate estimate
        from sklearn.model_selection import cross_val_predict
        y_pred = cross_val_predict(model, X_scaled, y, cv=kfold, n_jobs=-1)
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)
        
        results = {
            "name": name,
            "n_features": int(X.shape[1]),
            "r2_mean": float(r2_scores.mean()),
            "r2_std": float(r2_scores.std()),
            "r2_cv": float(r2),
            "mae_m": float(mae),
            "rmse_m": float(rmse),
        }
        
        print(f"  {name}: R²={r2:.4f}, MAE={mae:.2f}m, n_features={X.shape[1]}")
        
        return results
        
    def baseline_analysis(self, era5_features, shadow_features, y):
        """Establish baseline with all features."""
        print("\n" + "=" * 80)
        print("Baseline Analysis: All Features")
        print("=" * 80)
        
        X_all = np.concatenate([era5_features, shadow_features], axis=1)
        results = self.train_and_evaluate(X_all, y, "All Features (Baseline)")
        self.results["baseline"] = results
        
        return X_all
        
    def feature_group_ablation(self, era5_features, shadow_features, y,
                               era5_names, shadow_names):
        """Ablation study: atmospheric only, shadow only, all features."""
        print("\n" + "=" * 80)
        print("Feature Group Ablation Study")
        print("=" * 80)
        
        ablations = [
            ("Atmospheric Only", era5_features, era5_names),
            ("Shadow Only", shadow_features, shadow_names),
        ]
        
        for name, X, feat_names in ablations:
            results = self.train_and_evaluate(X, y, name)
            results["feature_names"] = feat_names
            self.results["feature_group_ablation"].append(results)
            
    def compute_shap_importance(self, X: np.ndarray, y: np.ndarray,
                               feature_names: List[str]) -> pd.DataFrame:
        """
        Compute SHAP feature importance.
        
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Feature matrix
        y : array, shape (n_samples,)
            Target values
        feature_names : list of str
            Feature names
            
        Returns
        -------
        importance_df : DataFrame
            SHAP importance values sorted by mean absolute SHAP
        """
        print("\n" + "=" * 80)
        print("Computing SHAP Feature Importance")
        print("=" * 80)
        
        # Handle NaN values
        nan_count = np.isnan(X).sum()
        if nan_count > 0:
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='median')
            X = imputer.fit_transform(X)
        
        # Standardize and train model
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
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
        model.fit(X_scaled, y)
        
        # Compute SHAP values (using TreeExplainer for speed)
        print("  Computing SHAP values (this may take a few minutes)...")
        explainer = shap.TreeExplainer(model)
        
        # Use a subset for efficiency
        X_subset = X_scaled[:min(1000, len(X_scaled))]
        shap_values = explainer.shap_values(X_subset)
        
        # Compute mean absolute SHAP values
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'mean_abs_shap': mean_abs_shap,
            'gini_importance': model.feature_importances_,
        }).sort_values('mean_abs_shap', ascending=False)
        
        print("\nTop 10 Features by SHAP Importance:")
        print(importance_df.head(10).to_string(index=False))
        
        # Save full importance table
        importance_df.to_csv(REPORTS_DIR / "feature_importance.csv", index=False)
        
        # Store in results
        self.results["feature_importance"] = {
            "top_features": importance_df.head(10).to_dict('records'),
            "all_features": importance_df.to_dict('records'),
        }
        
        return importance_df
        
    def top_feature_ablation(self, X: np.ndarray, y: np.ndarray,
                            importance_df: pd.DataFrame,
                            feature_names: List[str]):
        """
        Remove top-5 SHAP features one at a time and measure performance drop.
        
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Full feature matrix
        y : array, shape (n_samples,)
            Target values
        importance_df : DataFrame
            SHAP importance rankings
        feature_names : list of str
            All feature names
        """
        print("\n" + "=" * 80)
        print("Top Feature Ablation Study")
        print("=" * 80)
        print("Removing top-5 SHAP features one at a time...\n")
        
        top_5_features = importance_df.head(5)['feature'].tolist()
        feature_indices = {name: i for i, name in enumerate(feature_names)}
        
        for feature_to_remove in top_5_features:
            # Get all features except the one to remove
            idx_to_remove = feature_indices[feature_to_remove]
            mask = np.ones(X.shape[1], dtype=bool)
            mask[idx_to_remove] = False
            X_ablated = X[:, mask]
            
            results = self.train_and_evaluate(
                X_ablated, y, 
                f"Remove: {feature_to_remove}"
            )
            results["removed_feature"] = feature_to_remove
            results["removed_shap_importance"] = float(
                importance_df[importance_df['feature'] == feature_to_remove]['mean_abs_shap'].iloc[0]
            )
            
            # Compute performance drop
            baseline_r2 = self.results["baseline"]["r2_cv"]
            r2_drop = baseline_r2 - results["r2_cv"]
            results["r2_drop"] = float(r2_drop)
            results["r2_drop_percent"] = float(r2_drop / baseline_r2 * 100)
            
            self.results["top_feature_ablation"].append(results)
            
            print(f"    R² drop: {r2_drop:.4f} ({r2_drop/baseline_r2*100:.1f}%)")
            
    def feature_correlation_analysis(self, X: np.ndarray, feature_names: List[str]):
        """
        Analyze feature correlations and identify redundancy.
        
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Feature matrix
        feature_names : list of str
            Feature names
        """
        print("\n" + "=" * 80)
        print("Feature Correlation Analysis")
        print("=" * 80)
        
        # Compute correlation matrix
        corr_matrix = np.corrcoef(X.T)
        
        # Find highly correlated pairs (|r| > 0.8, excluding diagonal)
        high_corr_pairs = []
        n_features = len(feature_names)
        for i in range(n_features):
            for j in range(i + 1, n_features):
                if abs(corr_matrix[i, j]) > 0.8:
                    high_corr_pairs.append({
                        "feature_1": feature_names[i],
                        "feature_2": feature_names[j],
                        "correlation": float(corr_matrix[i, j]),
                    })
        
        print(f"\nFound {len(high_corr_pairs)} highly correlated pairs (|r| > 0.8)")
        if high_corr_pairs:
            print("\nTop 10 Correlations:")
            sorted_pairs = sorted(high_corr_pairs, 
                                 key=lambda x: abs(x['correlation']), 
                                 reverse=True)
            for pair in sorted_pairs[:10]:
                print(f"  {pair['feature_1']} <-> {pair['feature_2']}: r={pair['correlation']:.3f}")
        
        self.results["feature_correlations"] = {
            "high_correlation_threshold": 0.8,
            "n_high_correlations": len(high_corr_pairs),
            "high_correlation_pairs": high_corr_pairs,
        }
        
        # Create correlation heatmap
        self._plot_correlation_heatmap(corr_matrix, feature_names)
        
        # Create hierarchical clustering dendrogram
        self._plot_feature_clustering(corr_matrix, feature_names)
        
    def _plot_correlation_heatmap(self, corr_matrix: np.ndarray, 
                                  feature_names: List[str]):
        """Plot feature correlation heatmap."""
        fig, ax = plt.subplots(figsize=(14, 12))
        
        # Use diverging colormap centered at 0
        sns.heatmap(corr_matrix, 
                   xticklabels=feature_names,
                   yticklabels=feature_names,
                   cmap='RdBu_r',
                   center=0,
                   vmin=-1, vmax=1,
                   square=True,
                   cbar_kws={"label": "Pearson Correlation"},
                   ax=ax)
        
        ax.set_title('Feature Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
        plt.xticks(rotation=90, ha='right', fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'feature_correlation_heatmap.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("  Saved: feature_correlation_heatmap.png")
        
    def _plot_feature_clustering(self, corr_matrix: np.ndarray,
                                 feature_names: List[str]):
        """Plot hierarchical clustering dendrogram to identify feature groups."""
        # Convert correlation to distance
        corr_dist = 1 - np.abs(corr_matrix)
        # Ensure perfect symmetry and zero diagonal for distance matrix
        corr_dist = (corr_dist + corr_dist.T) / 2
        np.fill_diagonal(corr_dist, 0)
        
        # Replace any NaN or inf values with 1 (maximum distance)
        corr_dist = np.nan_to_num(corr_dist, nan=1.0, posinf=1.0, neginf=1.0)
        
        # Perform hierarchical clustering  
        # Convert to condensed distance vector manually
        condensed = corr_dist[np.triu_indices(len(corr_dist), k=1)]
        linkage = hierarchy.linkage(condensed, method='average')
        
        fig, ax = plt.subplots(figsize=(12, 8))
        dendro = hierarchy.dendrogram(
            linkage,
            labels=feature_names,
            ax=ax,
            orientation='right',
            leaf_font_size=8
        )
        
        ax.set_title('Hierarchical Clustering of Features\n(Based on Absolute Correlation)',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Distance (1 - |correlation|)', fontsize=12)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'feature_clustering_dendrogram.png',
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("  Saved: feature_clustering_dendrogram.png")
        
    def create_summary_plots(self):
        """Create summary visualization of ablation results."""
        print("\n" + "=" * 80)
        print("Creating Summary Plots")
        print("=" * 80)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Left: Feature group ablation
        ax = axes[0]
        ablation_data = [self.results["baseline"]] + self.results["feature_group_ablation"]
        names = [d["name"] for d in ablation_data]
        r2_vals = [d["r2_cv"] for d in ablation_data]
        r2_stds = [d["r2_std"] for d in ablation_data]
        n_features = [d["n_features"] for d in ablation_data]
        
        x = np.arange(len(names))
        bars = ax.bar(x, r2_vals, yerr=r2_stds, capsize=5, 
                     color=['green', 'skyblue', 'coral'], alpha=0.7, edgecolor='black')
        
        # Add n_features labels on bars
        for i, (bar, n) in enumerate(zip(bars, n_features)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 0.02,
                   f'n={n}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_ylabel('R² Score', fontsize=12, fontweight='bold')
        ax.set_title('Feature Group Ablation Study', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=15, ha='right')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, max(r2_vals) * 1.15])
        
        # Right: Top feature removal impact
        ax = axes[1]
        if self.results["top_feature_ablation"]:
            top_ablation = self.results["top_feature_ablation"]
            features = [d["removed_feature"] for d in top_ablation]
            r2_drops = [d["r2_drop"] for d in top_ablation]
            
            # Shorten feature names for display
            features_short = [f[:30] + '...' if len(f) > 30 else f 
                            for f in features]
            
            bars = ax.barh(features_short, r2_drops, color='orangered', 
                          alpha=0.7, edgecolor='black')
            ax.set_xlabel('R² Drop When Feature Removed', fontsize=12, fontweight='bold')
            ax.set_title('Impact of Removing Top-5 SHAP Features', 
                        fontsize=14, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            
            # Add percentage labels
            for i, (bar, drop) in enumerate(zip(bars, r2_drops)):
                pct = top_ablation[i]["r2_drop_percent"]
                ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                       f'{pct:.1f}%', ha='left', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'ablation_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  Saved: ablation_summary.png")
        
    def generate_latex_table(self):
        """Generate LaTeX table for publication."""
        latex = r"""\begin{table}[ht]
\centering
\caption{Feature Ablation Study Results}
\label{tab:feature_ablation}
\begin{tabular}{lcccc}
\hline
\textbf{Configuration} & \textbf{N Features} & \textbf{R$^2$} & \textbf{MAE (m)} & \textbf{RMSE (m)} \\
\hline
"""
        
        # Baseline
        bl = self.results["baseline"]
        latex += f"All Features (Baseline) & {bl['n_features']} & {bl['r2_cv']:.3f} $\\pm$ {bl['r2_std']:.3f} & {bl['mae_m']:.1f} & {bl['rmse_m']:.1f} \\\\\n"
        
        # Feature groups
        for ab in self.results["feature_group_ablation"]:
            latex += f"{ab['name']} & {ab['n_features']} & {ab['r2_cv']:.3f} $\\pm$ {ab['r2_std']:.3f} & {ab['mae_m']:.1f} & {ab['rmse_m']:.1f} \\\\\n"
        
        latex += r"""\hline
\multicolumn{5}{l}{\textit{Top-5 SHAP Features Removed (Individual):}} \\
"""
        
        for ab in self.results["top_feature_ablation"]:
            feat_short = ab["removed_feature"][:40] + "..." if len(ab["removed_feature"]) > 40 else ab["removed_feature"]
            latex += f"{feat_short} & {ab['n_features']} & {ab['r2_cv']:.3f} & {ab['mae_m']:.1f} & {ab['rmse_m']:.1f} \\\\\n"
        
        latex += r"""\hline
\end{tabular}
\end{table}
"""
        
        latex_file = REPORTS_DIR / "feature_ablation_table.tex"
        with open(latex_file, 'w') as f:
            f.write(latex)
        print(f"  Saved: {latex_file}")
        
    def save_results(self):
        """Save all results to JSON."""
        output_file = REPORTS_DIR / "feature_ablation_report.json"
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n✓ Results saved: {output_file}")
        
    def print_summary(self):
        """Print analysis summary."""
        print("\n" + "=" * 80)
        print("FEATURE ABLATION ANALYSIS SUMMARY")
        print("=" * 80)
        
        bl = self.results["baseline"]
        print(f"\nBaseline (All Features):")
        print(f"  R²: {bl['r2_cv']:.4f} ± {bl['r2_std']:.4f}")
        print(f"  MAE: {bl['mae_m']:.2f} m")
        print(f"  N Features: {bl['n_features']}")
        
        print(f"\nFeature Group Ablation:")
        for ab in self.results["feature_group_ablation"]:
            r2_change = ab['r2_cv'] - bl['r2_cv']
            print(f"  {ab['name']}: R²={ab['r2_cv']:.4f} (Δ={r2_change:+.4f}), n={ab['n_features']}")
        
        print(f"\nTop-5 SHAP Feature Removal Impact:")
        for ab in self.results["top_feature_ablation"]:
            print(f"  {ab['removed_feature'][:50]}")
            print(f"    R² drop: {ab['r2_drop']:.4f} ({ab['r2_drop_percent']:.1f}%)")
        
        print(f"\nFeature Correlations:")
        corr = self.results["feature_correlations"]
        print(f"  High correlations (|r| > {corr['high_correlation_threshold']}): {corr['n_high_correlations']}")
        
        print(f"\nAcceptance Criteria:")
        print(f"  ✓ Atmospheric vs. shadow features ablated")
        print(f"  ✓ Top-5 SHAP features removed individually")
        print(f"  ✓ Feature correlation matrix generated")
        print(f"  ✓ Results table created")
        
    def run_full_analysis(self):
        """Run complete feature ablation analysis."""
        # Load data
        (era5_features, shadow_features, y, flight_ids, 
         era5_names, shadow_names) = self.load_data(INTEGRATED_FEATURES)
        
        # Baseline
        X_all = self.baseline_analysis(era5_features, shadow_features, y)
        all_feature_names = era5_names + shadow_names
        
        # Feature group ablation
        self.feature_group_ablation(era5_features, shadow_features, y,
                                    era5_names, shadow_names)
        
        # SHAP importance
        importance_df = self.compute_shap_importance(X_all, y, all_feature_names)
        
        # Top feature ablation
        self.top_feature_ablation(X_all, y, importance_df, all_feature_names)
        
        # Correlation analysis
        self.feature_correlation_analysis(X_all, all_feature_names)
        
        # Create visualizations
        self.create_summary_plots()
        
        # Generate outputs
        self.generate_latex_table()
        self.save_results()
        self.print_summary()


def main():
    """Main execution."""
    print("\n" + "=" * 80)
    print("STARTING FEATURE ABLATION ANALYSIS")
    print("=" * 80)
    
    # Check data availability
    if not INTEGRATED_FEATURES.exists():
        print(f"\n✗ ERROR: Integrated features file not found: {INTEGRATED_FEATURES}")
        print("\nPlease ensure data preprocessing has been completed.")
        print("Run: python src/data_preprocessing.py")
        return 1
    
    # Run analysis
    analyzer = FeatureAblationAnalysis(n_folds=5, random_state=42)
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
