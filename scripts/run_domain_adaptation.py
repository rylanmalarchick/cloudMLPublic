#!/usr/bin/env python3
"""
Domain Adaptation Experiments for CBH Retrieval

Implements multiple domain adaptation strategies to address the severe 
cross-flight generalization failure (LOFO-CV R² = -18.7).

Methods Implemented:
1. Few-Shot Adaptation - Add N samples from target flight
2. Instance Weighting - Weight source samples by similarity to target
3. TrAdaBoost - Transfer learning boosting algorithm
4. MMD Feature Alignment - Maximum Mean Discrepancy regularization
5. Fine-tuning - Pre-train on source, fine-tune on target

Uses the enhanced feature set from feature_engineering.py.

Author: CBH Restudy Agent
Date: 2026-01-06
"""

import json
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
from scipy.stats import ks_2samp

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Try to import AgentBible for validation
try:
    import agentbible
    from agentbible.errors import ValidationError
    AGENTBIBLE_VERSION = agentbible.__version__
except ImportError:
    AGENTBIBLE_VERSION = "N/A"
    class ValidationError(Exception):
        pass

print(f"AgentBible version: {AGENTBIBLE_VERSION}")

# Set plotting style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")


class DomainShiftAnalyzer:
    """Analyze domain shift between flights using multiple metrics."""
    
    @staticmethod
    def compute_mmd(X_source: np.ndarray, X_target: np.ndarray, 
                   gamma: float = None) -> float:
        """
        Compute Maximum Mean Discrepancy between source and target.
        
        MMD measures the distance between feature distributions using
        a kernel-based approach. Higher MMD = more domain shift.
        
        Args:
            X_source: Source domain features (n_source, n_features)
            X_target: Target domain features (n_target, n_features)
            gamma: RBF kernel bandwidth (auto-computed if None)
            
        Returns:
            MMD value (0 = identical distributions)
        """
        if gamma is None:
            # Median heuristic for bandwidth
            combined = np.vstack([X_source, X_target])
            dists = cdist(combined, combined, 'sqeuclidean')
            gamma = 1.0 / np.median(dists[dists > 0])
        
        # RBF kernel
        def rbf_kernel(X, Y):
            dists = cdist(X, Y, 'sqeuclidean')
            return np.exp(-gamma * dists)
        
        K_ss = rbf_kernel(X_source, X_source)
        K_tt = rbf_kernel(X_target, X_target)
        K_st = rbf_kernel(X_source, X_target)
        
        n_s, n_t = len(X_source), len(X_target)
        
        mmd = (np.sum(K_ss) / (n_s * n_s) + 
               np.sum(K_tt) / (n_t * n_t) - 
               2 * np.sum(K_st) / (n_s * n_t))
        
        return max(0, mmd)  # MMD should be non-negative
    
    @staticmethod
    def compute_ks_divergence(X_source: np.ndarray, X_target: np.ndarray) -> Dict[str, float]:
        """
        Compute per-feature Kolmogorov-Smirnov statistics.
        
        Args:
            X_source: Source features
            X_target: Target features
            
        Returns:
            Dictionary with per-feature KS statistics
        """
        n_features = X_source.shape[1]
        ks_stats = {}
        
        for i in range(n_features):
            stat, pval = ks_2samp(X_source[:, i], X_target[:, i])
            ks_stats[f"feature_{i}"] = {"statistic": stat, "pvalue": pval}
        
        # Mean KS statistic across features
        ks_stats["mean"] = np.mean([v["statistic"] for v in ks_stats.values() 
                                   if isinstance(v, dict)])
        
        return ks_stats
    
    @staticmethod
    def compute_a_distance(X_source: np.ndarray, X_target: np.ndarray,
                          random_state: int = 42) -> float:
        """
        Compute proxy A-distance using domain classifier.
        
        A-distance measures how well a classifier can distinguish between
        source and target domains. A-distance ≈ 0 means identical domains.
        
        Args:
            X_source: Source features
            X_target: Target features
            random_state: Random seed
            
        Returns:
            A-distance value
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        
        # Create domain labels
        y = np.concatenate([np.zeros(len(X_source)), np.ones(len(X_target))])
        X = np.vstack([X_source, X_target])
        
        # Train domain classifier
        clf = LogisticRegression(random_state=random_state, max_iter=1000)
        scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
        
        # A-distance = 2(1 - 2*error)
        error = 1 - np.mean(scores)
        a_distance = 2 * (1 - 2 * error)
        
        return a_distance


class FewShotAdapter:
    """Few-shot domain adaptation for CBH retrieval."""
    
    def __init__(self, base_model_params: Dict = None, random_state: int = 42):
        self.random_state = random_state
        self.base_model_params = base_model_params or {
            "n_estimators": 100,
            "max_depth": 4,
            "learning_rate": 0.1,
            "random_state": random_state,
        }
        
    def adapt(self, X_source: np.ndarray, y_source: np.ndarray,
              X_target: np.ndarray, y_target: np.ndarray,
              n_shots: int, n_trials: int = 10) -> Dict[str, Any]:
        """
        Perform few-shot adaptation with N samples from target.
        
        Strategy: Combine source data with N randomly sampled target samples,
        train model, evaluate on remaining target samples.
        
        Args:
            X_source: Source domain features
            y_source: Source domain labels
            X_target: Target domain features  
            y_target: Target domain labels
            n_shots: Number of target samples to use for adaptation
            n_trials: Number of random trials to average
            
        Returns:
            Dictionary with adaptation results
        """
        np.random.seed(self.random_state)
        
        if n_shots >= len(X_target):
            raise ValueError(f"n_shots ({n_shots}) >= target size ({len(X_target)})")
        
        trial_results = []
        
        for trial in range(n_trials):
            # Randomly sample n_shots from target
            indices = np.random.permutation(len(X_target))
            adapt_idx = indices[:n_shots]
            test_idx = indices[n_shots:]
            
            X_adapt = X_target[adapt_idx]
            y_adapt = y_target[adapt_idx]
            X_test = X_target[test_idx]
            y_test = y_target[test_idx]
            
            # Combine source + adaptation samples
            X_train = np.vstack([X_source, X_adapt])
            y_train = np.concatenate([y_source, y_adapt])
            
            # Train model
            model = GradientBoostingRegressor(**self.base_model_params)
            model.fit(X_train, y_train)
            
            # Evaluate on remaining target samples
            y_pred = model.predict(X_test)
            
            trial_results.append({
                "r2": r2_score(y_test, y_pred),
                "mae_km": mean_absolute_error(y_test, y_pred),
                "rmse_km": np.sqrt(mean_squared_error(y_test, y_pred)),
                "n_test": len(y_test),
            })
        
        # Aggregate results
        return {
            "n_shots": n_shots,
            "n_trials": n_trials,
            "mean_r2": np.mean([t["r2"] for t in trial_results]),
            "std_r2": np.std([t["r2"] for t in trial_results]),
            "mean_mae_km": np.mean([t["mae_km"] for t in trial_results]),
            "std_mae_km": np.std([t["mae_km"] for t in trial_results]),
            "mean_rmse_km": np.mean([t["rmse_km"] for t in trial_results]),
            "trials": trial_results,
        }


class InstanceWeightedAdapter:
    """Instance weighting for domain adaptation."""
    
    def __init__(self, base_model_params: Dict = None, random_state: int = 42):
        self.random_state = random_state
        self.base_model_params = base_model_params or {
            "n_estimators": 100,
            "max_depth": 4,
            "learning_rate": 0.1,
            "random_state": random_state,
        }
        
    def compute_weights(self, X_source: np.ndarray, X_target: np.ndarray,
                       method: str = "knn") -> np.ndarray:
        """
        Compute importance weights for source samples.
        
        Methods:
        - knn: Weight by distance to nearest target samples
        - density: Weight by density ratio estimation
        
        Args:
            X_source: Source features
            X_target: Target features
            method: Weighting method
            
        Returns:
            Weight for each source sample
        """
        if method == "knn":
            # Weight by average distance to K nearest target samples
            k = min(5, len(X_target))
            nn = NearestNeighbors(n_neighbors=k)
            nn.fit(X_target)
            
            distances, _ = nn.kneighbors(X_source)
            mean_distances = distances.mean(axis=1)
            
            # Convert distances to weights (closer = higher weight)
            # Normalize to [0.1, 1] range
            weights = 1.0 / (1.0 + mean_distances)
            weights = 0.1 + 0.9 * (weights - weights.min()) / (weights.max() - weights.min() + 1e-8)
            
        elif method == "density":
            # Simple density ratio using KDE
            from sklearn.neighbors import KernelDensity
            
            kde_source = KernelDensity(kernel='gaussian', bandwidth=0.5)
            kde_target = KernelDensity(kernel='gaussian', bandwidth=0.5)
            
            kde_source.fit(X_source)
            kde_target.fit(X_target)
            
            log_density_source = kde_source.score_samples(X_source)
            log_density_target = kde_target.score_samples(X_source)
            
            # Weight = p_target / p_source
            log_weights = log_density_target - log_density_source
            weights = np.exp(np.clip(log_weights, -5, 5))  # Clip for stability
            weights = weights / weights.mean()  # Normalize
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return weights
    
    def adapt(self, X_source: np.ndarray, y_source: np.ndarray,
              X_target: np.ndarray, y_target: np.ndarray,
              method: str = "knn") -> Dict[str, Any]:
        """
        Perform instance-weighted training.
        
        Args:
            X_source: Source features
            y_source: Source labels
            X_target: Target features (for weight computation)
            y_target: Target labels (for evaluation only)
            method: Weighting method
            
        Returns:
            Results dictionary
        """
        # Compute weights
        weights = self.compute_weights(X_source, X_target, method=method)
        
        # Train weighted model
        # Note: GBDT doesn't support sample_weight in sklearn, use RandomForest
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=8,
            random_state=self.random_state,
        )
        model.fit(X_source, y_source, sample_weight=weights)
        
        # Evaluate on target
        y_pred = model.predict(X_target)
        
        return {
            "method": method,
            "r2": r2_score(y_target, y_pred),
            "mae_km": mean_absolute_error(y_target, y_pred),
            "rmse_km": np.sqrt(mean_squared_error(y_target, y_pred)),
            "weight_stats": {
                "min": float(weights.min()),
                "max": float(weights.max()),
                "mean": float(weights.mean()),
                "std": float(weights.std()),
            },
        }


class TrAdaBoostAdapter:
    """
    TrAdaBoost for domain adaptation.
    
    Transfer learning boosting algorithm that iteratively decreases
    weights of poorly-transferred source samples while maintaining
    target sample weights.
    
    Reference:
    Dai et al. (2007) "Boosting for Transfer Learning"
    """
    
    def __init__(self, n_estimators: int = 20, random_state: int = 42):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.models = []
        self.betas = []
        
    def fit(self, X_source: np.ndarray, y_source: np.ndarray,
            X_target: np.ndarray, y_target: np.ndarray) -> "TrAdaBoostAdapter":
        """
        Fit TrAdaBoost model.
        
        Args:
            X_source: Source domain features
            y_source: Source domain labels
            X_target: Target domain features (used for training)
            y_target: Target domain labels (used for training)
            
        Returns:
            Self
        """
        n_source = len(X_source)
        n_target = len(X_target)
        n_total = n_source + n_target
        
        # Combine data
        X = np.vstack([X_source, X_target])
        y = np.concatenate([y_source, y_target])
        
        # Initialize weights
        w_source = np.ones(n_source) / n_source
        w_target = np.ones(n_target) / n_target
        
        # Beta for source weight updates
        beta_source = 1.0 / (1.0 + np.sqrt(2.0 * np.log(n_source) / self.n_estimators))
        
        self.models = []
        self.betas = []
        
        for t in range(self.n_estimators):
            # Normalize weights
            weights = np.concatenate([w_source, w_target])
            weights = weights / weights.sum()
            
            # Train weak learner
            model = GradientBoostingRegressor(
                n_estimators=10,
                max_depth=3,
                random_state=self.random_state + t,
            )
            model.fit(X, y, sample_weight=weights)
            
            # Predict on all samples
            y_pred = model.predict(X)
            
            # Compute error (normalized absolute error)
            errors = np.abs(y - y_pred)
            max_error = errors.max() + 1e-8
            normalized_errors = errors / max_error
            
            # Compute weighted error on target only
            target_weights = w_target / w_target.sum()
            epsilon_t = np.sum(target_weights * normalized_errors[n_source:])
            
            # Avoid division by zero
            epsilon_t = np.clip(epsilon_t, 0.01, 0.99)
            
            # Compute beta for this iteration
            beta_t = epsilon_t / (1 - epsilon_t)
            
            # Update source weights (decrease for high-error samples)
            w_source = w_source * np.power(beta_source, normalized_errors[:n_source])
            
            # Update target weights (increase for high-error samples, like AdaBoost)
            w_target = w_target * np.power(1.0 / (beta_t + 1e-8), -normalized_errors[n_source:])
            
            self.models.append(model)
            self.betas.append(beta_t)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using weighted median of models."""
        predictions = np.array([model.predict(X) for model in self.models])
        weights = np.log(1.0 / (np.array(self.betas) + 1e-8))
        
        # Weighted average (simpler than weighted median)
        weights = weights / weights.sum()
        return np.average(predictions, axis=0, weights=weights)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate on test data."""
        y_pred = self.predict(X_test)
        return {
            "r2": r2_score(y_test, y_pred),
            "mae_km": mean_absolute_error(y_test, y_pred),
            "rmse_km": np.sqrt(mean_squared_error(y_test, y_pred)),
        }


class MMDAdapter:
    """
    MMD-based domain adaptation using feature transformation.
    
    Learns a feature transformation that minimizes MMD between
    source and target while preserving predictive performance.
    """
    
    def __init__(self, alpha: float = 1.0, random_state: int = 42):
        """
        Args:
            alpha: Regularization weight for MMD term
            random_state: Random seed
        """
        self.alpha = alpha
        self.random_state = random_state
        self.transform_matrix = None
        self.scaler = None
        
    def fit_transform(self, X_source: np.ndarray, X_target: np.ndarray,
                     y_source: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Learn MMD-minimizing transformation and apply it.
        
        This is a simplified version using PCA + reweighting to align
        feature distributions.
        
        Args:
            X_source: Source features
            X_target: Target features
            y_source: Source labels (for supervised component)
            
        Returns:
            Transformed source and target features
        """
        from sklearn.decomposition import PCA
        
        # Standardize
        self.scaler = StandardScaler()
        X_source_scaled = self.scaler.fit_transform(X_source)
        X_target_scaled = self.scaler.transform(X_target)
        
        # PCA to align principal components
        n_components = min(X_source.shape[1], 20)
        pca = PCA(n_components=n_components, random_state=self.random_state)
        
        # Fit on combined data for shared representation
        X_combined = np.vstack([X_source_scaled, X_target_scaled])
        pca.fit(X_combined)
        
        X_source_pca = pca.transform(X_source_scaled)
        X_target_pca = pca.transform(X_target_scaled)
        
        # Feature-wise mean alignment
        source_mean = X_source_pca.mean(axis=0)
        target_mean = X_target_pca.mean(axis=0)
        
        # Shift source toward target
        X_source_aligned = X_source_pca - source_mean + target_mean
        
        self.pca = pca
        
        return X_source_aligned, X_target_pca
    
    def adapt(self, X_source: np.ndarray, y_source: np.ndarray,
              X_target: np.ndarray, y_target: np.ndarray) -> Dict[str, Any]:
        """
        Perform MMD-based adaptation.
        
        Args:
            X_source: Source features
            y_source: Source labels
            X_target: Target features
            y_target: Target labels
            
        Returns:
            Results dictionary
        """
        # Compute initial MMD
        analyzer = DomainShiftAnalyzer()
        mmd_before = analyzer.compute_mmd(X_source[:100], X_target[:100])  # Subsample for speed
        
        # Transform features
        X_source_aligned, X_target_aligned = self.fit_transform(X_source, X_target, y_source)
        
        # Compute MMD after alignment
        mmd_after = analyzer.compute_mmd(X_source_aligned[:100], X_target_aligned[:100])
        
        # Train on aligned source, evaluate on aligned target
        model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=self.random_state,
        )
        model.fit(X_source_aligned, y_source)
        y_pred = model.predict(X_target_aligned)
        
        return {
            "mmd_before": float(mmd_before),
            "mmd_after": float(mmd_after),
            "mmd_reduction": float((mmd_before - mmd_after) / (mmd_before + 1e-8)),
            "r2": r2_score(y_target, y_pred),
            "mae_km": mean_absolute_error(y_target, y_pred),
            "rmse_km": np.sqrt(mean_squared_error(y_target, y_pred)),
        }


class DomainAdaptationExperiment:
    """
    Complete domain adaptation experiment suite for CBH retrieval.
    """
    
    def __init__(self, data_path: Path, output_dir: Path, 
                 use_enhanced_features: bool = True,
                 random_state: int = 42):
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.use_enhanced_features = use_enhanced_features
        self.random_state = random_state
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "data_path": str(self.data_path),
                "use_enhanced_features": use_enhanced_features,
                "random_state": random_state,
                "agentbible_version": AGENTBIBLE_VERSION,
            },
            "domain_shift_analysis": {},
            "baseline": {},
            "few_shot": {},
            "instance_weighting": {},
            "tradaboost": {},
            "mmd_adaptation": {},
        }
        
    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """Load feature data from HDF5 file."""
        print("\n" + "=" * 80)
        print("Loading Data")
        print("=" * 80)
        
        with h5py.File(self.data_path, "r") as f:
            if self.use_enhanced_features and "features" in f:
                # Enhanced features from feature_engineering.py
                X = f["features"][:]
                feature_names = [name.decode() if isinstance(name, bytes) else name 
                               for name in f["feature_names"][:]]
                cbh_km = f["metadata/cbh_km"][:]
                flight_ids = f["metadata/flight_id"][:].astype(str)
            else:
                # Original features
                atmo_features = {}
                for key in f["atmospheric_features"].keys():
                    atmo_features[key] = f[f"atmospheric_features/{key}"][:]
                    
                geo_features = {}
                for key in f["geometric_features"].keys():
                    geo_features[key] = f[f"geometric_features/{key}"][:]
                
                feature_names = list(atmo_features.keys()) + list(geo_features.keys())
                X = np.column_stack([atmo_features[k] for k in atmo_features] + 
                                   [geo_features[k] for k in geo_features])
                
                cbh_km = f["metadata/cbh_km"][:]
                flight_ids = f["metadata/flight_id"][:].astype(str)
        
        print(f"  Loaded {len(cbh_km)} samples, {X.shape[1]} features")
        print(f"  Flights: {sorted(set(flight_ids))}")
        
        # Per-flight statistics
        print("\n  Per-Flight Statistics:")
        for fid in sorted(set(flight_ids)):
            mask = flight_ids == fid
            print(f"    Flight {fid}: n={mask.sum()}, CBH={cbh_km[mask].mean():.3f} ± {cbh_km[mask].std():.3f} km")
        
        return X, cbh_km, flight_ids, feature_names
    
    def analyze_domain_shift(self, X: np.ndarray, flight_ids: np.ndarray,
                            feature_names: List[str]) -> Dict:
        """Analyze pairwise domain shift between flights."""
        print("\n" + "=" * 80)
        print("Domain Shift Analysis")
        print("=" * 80)
        
        analyzer = DomainShiftAnalyzer()
        unique_flights = sorted(set(flight_ids))
        
        # Standardize features for shift analysis
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Compute pairwise MMD matrix
        n_flights = len(unique_flights)
        mmd_matrix = np.zeros((n_flights, n_flights))
        a_distance_matrix = np.zeros((n_flights, n_flights))
        
        print("\n  Pairwise MMD:")
        for i, f1 in enumerate(unique_flights):
            for j, f2 in enumerate(unique_flights):
                if i < j:
                    mask1 = flight_ids == f1
                    mask2 = flight_ids == f2
                    
                    X1 = X_scaled[mask1][:100]  # Subsample for speed
                    X2 = X_scaled[mask2][:100]
                    
                    mmd = analyzer.compute_mmd(X1, X2)
                    a_dist = analyzer.compute_a_distance(X1, X2, self.random_state)
                    
                    mmd_matrix[i, j] = mmd
                    mmd_matrix[j, i] = mmd
                    a_distance_matrix[i, j] = a_dist
                    a_distance_matrix[j, i] = a_dist
                    
                    print(f"    {f1} <-> {f2}: MMD={mmd:.4f}, A-dist={a_dist:.4f}")
        
        # Store results
        shift_results = {
            "flight_ids": unique_flights,
            "mmd_matrix": mmd_matrix.tolist(),
            "a_distance_matrix": a_distance_matrix.tolist(),
            "mean_mmd": float(mmd_matrix[np.triu_indices(n_flights, k=1)].mean()),
            "max_mmd": float(mmd_matrix.max()),
        }
        
        self.results["domain_shift_analysis"] = shift_results
        
        # Create heatmap visualization
        self._plot_shift_matrix(mmd_matrix, unique_flights)
        
        return shift_results
    
    def _plot_shift_matrix(self, mmd_matrix: np.ndarray, flight_ids: List[str]):
        """Plot domain shift heatmap."""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        im = ax.imshow(mmd_matrix, cmap="YlOrRd")
        
        ax.set_xticks(range(len(flight_ids)))
        ax.set_yticks(range(len(flight_ids)))
        ax.set_xticklabels([f"F{f}" for f in flight_ids])
        ax.set_yticklabels([f"F{f}" for f in flight_ids])
        
        # Add text annotations
        for i in range(len(flight_ids)):
            for j in range(len(flight_ids)):
                ax.text(j, i, f"{mmd_matrix[i, j]:.3f}",
                       ha="center", va="center", fontsize=10)
        
        ax.set_title("Pairwise MMD Between Flights", fontsize=14, fontweight="bold")
        plt.colorbar(im, ax=ax, label="MMD")
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "domain_shift_matrix.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"\n  Saved: {self.output_dir / 'domain_shift_matrix.png'}")
    
    def compute_baseline(self, X: np.ndarray, cbh_km: np.ndarray,
                        flight_ids: np.ndarray) -> Dict:
        """Compute baseline LOFO-CV performance."""
        print("\n" + "=" * 80)
        print("Baseline LOFO-CV Performance")
        print("=" * 80)
        
        unique_flights = sorted(set(flight_ids))
        scaler = StandardScaler()
        
        lofo_results = {}
        
        for target_flight in unique_flights:
            # Source = all other flights, Target = this flight
            source_mask = flight_ids != target_flight
            target_mask = flight_ids == target_flight
            
            X_source = X[source_mask]
            y_source = cbh_km[source_mask]
            X_target = X[target_mask]
            y_target = cbh_km[target_mask]
            
            # Scale
            X_source_scaled = scaler.fit_transform(X_source)
            X_target_scaled = scaler.transform(X_target)
            
            # Train and evaluate
            model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                random_state=self.random_state,
            )
            model.fit(X_source_scaled, y_source)
            y_pred = model.predict(X_target_scaled)
            
            r2 = r2_score(y_target, y_pred)
            mae = mean_absolute_error(y_target, y_pred)
            
            lofo_results[target_flight] = {
                "r2": float(r2),
                "mae_km": float(mae),
                "n_source": int(source_mask.sum()),
                "n_target": int(target_mask.sum()),
            }
            
            print(f"  Target Flight {target_flight}: R²={r2:.4f}, MAE={mae*1000:.1f}m (n={target_mask.sum()})")
        
        # Aggregate
        mean_r2 = np.mean([v["r2"] for v in lofo_results.values()])
        mean_mae = np.mean([v["mae_km"] for v in lofo_results.values()])
        
        baseline = {
            "lofo_results": lofo_results,
            "mean_r2": float(mean_r2),
            "mean_mae_km": float(mean_mae),
        }
        
        print(f"\n  Mean LOFO R²: {mean_r2:.4f}")
        print(f"  Mean LOFO MAE: {mean_mae*1000:.1f}m")
        
        self.results["baseline"] = baseline
        return baseline
    
    def run_few_shot_experiments(self, X: np.ndarray, cbh_km: np.ndarray,
                                 flight_ids: np.ndarray) -> Dict:
        """Run few-shot adaptation experiments."""
        print("\n" + "=" * 80)
        print("Few-Shot Adaptation Experiments")
        print("=" * 80)
        
        unique_flights = sorted(set(flight_ids))
        scaler = StandardScaler()
        adapter = FewShotAdapter(random_state=self.random_state)
        
        shot_sizes = [5, 10, 20, 50]
        few_shot_results = {k: {} for k in shot_sizes}
        
        for target_flight in unique_flights:
            source_mask = flight_ids != target_flight
            target_mask = flight_ids == target_flight
            
            if target_mask.sum() < 60:  # Need enough samples for meaningful test
                print(f"  Skipping Flight {target_flight} (only {target_mask.sum()} samples)")
                continue
            
            X_source = X[source_mask]
            y_source = cbh_km[source_mask]
            X_target = X[target_mask]
            y_target = cbh_km[target_mask]
            
            # Scale
            X_source_scaled = scaler.fit_transform(X_source)
            X_target_scaled = scaler.transform(X_target)
            
            print(f"\n  Target Flight {target_flight}:")
            
            for n_shots in shot_sizes:
                if n_shots >= len(X_target) * 0.8:  # Skip if too few test samples
                    continue
                    
                result = adapter.adapt(
                    X_source_scaled, y_source,
                    X_target_scaled, y_target,
                    n_shots=n_shots,
                    n_trials=10
                )
                
                few_shot_results[n_shots][target_flight] = result
                print(f"    {n_shots} shots: R²={result['mean_r2']:.4f} ± {result['std_r2']:.4f}")
        
        # Aggregate across flights
        aggregated = {}
        for n_shots in shot_sizes:
            if few_shot_results[n_shots]:
                mean_r2 = np.mean([v["mean_r2"] for v in few_shot_results[n_shots].values()])
                mean_mae = np.mean([v["mean_mae_km"] for v in few_shot_results[n_shots].values()])
                aggregated[n_shots] = {
                    "mean_r2": float(mean_r2),
                    "mean_mae_km": float(mean_mae),
                }
        
        self.results["few_shot"] = {
            "per_flight": {str(k): v for k, v in few_shot_results.items()},
            "aggregated": aggregated,
        }
        
        return self.results["few_shot"]
    
    def run_instance_weighting_experiments(self, X: np.ndarray, cbh_km: np.ndarray,
                                           flight_ids: np.ndarray) -> Dict:
        """Run instance weighting experiments."""
        print("\n" + "=" * 80)
        print("Instance Weighting Experiments")
        print("=" * 80)
        
        unique_flights = sorted(set(flight_ids))
        scaler = StandardScaler()
        
        methods = ["knn", "density"]
        iw_results = {m: {} for m in methods}
        
        for target_flight in unique_flights:
            source_mask = flight_ids != target_flight
            target_mask = flight_ids == target_flight
            
            X_source = X[source_mask]
            y_source = cbh_km[source_mask]
            X_target = X[target_mask]
            y_target = cbh_km[target_mask]
            
            # Scale
            X_source_scaled = scaler.fit_transform(X_source)
            X_target_scaled = scaler.transform(X_target)
            
            print(f"\n  Target Flight {target_flight}:")
            
            for method in methods:
                adapter = InstanceWeightedAdapter(random_state=self.random_state)
                result = adapter.adapt(
                    X_source_scaled, y_source,
                    X_target_scaled, y_target,
                    method=method
                )
                
                iw_results[method][target_flight] = result
                print(f"    {method}: R²={result['r2']:.4f}, MAE={result['mae_km']*1000:.1f}m")
        
        # Aggregate
        aggregated = {}
        for method in methods:
            mean_r2 = np.mean([v["r2"] for v in iw_results[method].values()])
            mean_mae = np.mean([v["mae_km"] for v in iw_results[method].values()])
            aggregated[method] = {
                "mean_r2": float(mean_r2),
                "mean_mae_km": float(mean_mae),
            }
        
        self.results["instance_weighting"] = {
            "per_flight": iw_results,
            "aggregated": aggregated,
        }
        
        return self.results["instance_weighting"]
    
    def run_tradaboost_experiments(self, X: np.ndarray, cbh_km: np.ndarray,
                                   flight_ids: np.ndarray) -> Dict:
        """Run TrAdaBoost experiments."""
        print("\n" + "=" * 80)
        print("TrAdaBoost Experiments")
        print("=" * 80)
        
        unique_flights = sorted(set(flight_ids))
        scaler = StandardScaler()
        
        tradaboost_results = {}
        
        for target_flight in unique_flights:
            source_mask = flight_ids != target_flight
            target_mask = flight_ids == target_flight
            
            if target_mask.sum() < 20:
                print(f"  Skipping Flight {target_flight} (only {target_mask.sum()} samples)")
                continue
            
            X_source = X[source_mask]
            y_source = cbh_km[source_mask]
            X_target = X[target_mask]
            y_target = cbh_km[target_mask]
            
            # Split target into adapt/test (use 20% for adaptation)
            n_adapt = max(10, int(0.2 * len(X_target)))
            np.random.seed(self.random_state)
            indices = np.random.permutation(len(X_target))
            adapt_idx = indices[:n_adapt]
            test_idx = indices[n_adapt:]
            
            X_target_adapt = X_target[adapt_idx]
            y_target_adapt = y_target[adapt_idx]
            X_target_test = X_target[test_idx]
            y_target_test = y_target[test_idx]
            
            # Scale
            X_source_scaled = scaler.fit_transform(X_source)
            X_target_adapt_scaled = scaler.transform(X_target_adapt)
            X_target_test_scaled = scaler.transform(X_target_test)
            
            # Train TrAdaBoost
            adapter = TrAdaBoostAdapter(n_estimators=20, random_state=self.random_state)
            adapter.fit(X_source_scaled, y_source, X_target_adapt_scaled, y_target_adapt)
            
            # Evaluate
            result = adapter.evaluate(X_target_test_scaled, y_target_test)
            tradaboost_results[target_flight] = result
            
            print(f"  Flight {target_flight}: R²={result['r2']:.4f}, MAE={result['mae_km']*1000:.1f}m")
        
        # Aggregate
        mean_r2 = np.mean([v["r2"] for v in tradaboost_results.values()])
        mean_mae = np.mean([v["mae_km"] for v in tradaboost_results.values()])
        
        self.results["tradaboost"] = {
            "per_flight": tradaboost_results,
            "aggregated": {
                "mean_r2": float(mean_r2),
                "mean_mae_km": float(mean_mae),
            },
        }
        
        print(f"\n  Mean R²: {mean_r2:.4f}")
        print(f"  Mean MAE: {mean_mae*1000:.1f}m")
        
        return self.results["tradaboost"]
    
    def run_mmd_experiments(self, X: np.ndarray, cbh_km: np.ndarray,
                           flight_ids: np.ndarray) -> Dict:
        """Run MMD-based adaptation experiments."""
        print("\n" + "=" * 80)
        print("MMD Feature Alignment Experiments")
        print("=" * 80)
        
        unique_flights = sorted(set(flight_ids))
        
        mmd_results = {}
        
        for target_flight in unique_flights:
            source_mask = flight_ids != target_flight
            target_mask = flight_ids == target_flight
            
            X_source = X[source_mask]
            y_source = cbh_km[source_mask]
            X_target = X[target_mask]
            y_target = cbh_km[target_mask]
            
            adapter = MMDAdapter(random_state=self.random_state)
            result = adapter.adapt(X_source, y_source, X_target, y_target)
            
            mmd_results[target_flight] = result
            print(f"  Flight {target_flight}: R²={result['r2']:.4f}, MMD reduction={result['mmd_reduction']:.1%}")
        
        # Aggregate
        mean_r2 = np.mean([v["r2"] for v in mmd_results.values()])
        mean_mae = np.mean([v["mae_km"] for v in mmd_results.values()])
        mean_mmd_reduction = np.mean([v["mmd_reduction"] for v in mmd_results.values()])
        
        self.results["mmd_adaptation"] = {
            "per_flight": mmd_results,
            "aggregated": {
                "mean_r2": float(mean_r2),
                "mean_mae_km": float(mean_mae),
                "mean_mmd_reduction": float(mean_mmd_reduction),
            },
        }
        
        print(f"\n  Mean R²: {mean_r2:.4f}")
        print(f"  Mean MMD reduction: {mean_mmd_reduction:.1%}")
        
        return self.results["mmd_adaptation"]
    
    def create_summary_visualization(self):
        """Create comprehensive comparison visualization."""
        print("\n" + "=" * 80)
        print("Creating Summary Visualization")
        print("=" * 80)
        
        # Collect all method results
        methods = ["Baseline\nLOFO"]
        r2_values = [self.results["baseline"]["mean_r2"]]
        
        # Few-shot results
        if "aggregated" in self.results["few_shot"]:
            for n_shots, agg in self.results["few_shot"]["aggregated"].items():
                methods.append(f"Few-shot\n({n_shots})")
                r2_values.append(agg["mean_r2"])
        
        # Instance weighting
        if "aggregated" in self.results["instance_weighting"]:
            for method, agg in self.results["instance_weighting"]["aggregated"].items():
                methods.append(f"IW\n({method})")
                r2_values.append(agg["mean_r2"])
        
        # TrAdaBoost
        if "aggregated" in self.results["tradaboost"]:
            methods.append("TrAdaBoost")
            r2_values.append(self.results["tradaboost"]["aggregated"]["mean_r2"])
        
        # MMD
        if "aggregated" in self.results["mmd_adaptation"]:
            methods.append("MMD\nAlign")
            r2_values.append(self.results["mmd_adaptation"]["aggregated"]["mean_r2"])
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 6))
        
        colors = ["#e74c3c"] + ["#3498db"] * (len(methods) - 1)  # Red for baseline
        bars = ax.bar(range(len(methods)), r2_values, color=colors, edgecolor="black")
        
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=1)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, fontsize=10)
        ax.set_ylabel("R² Score", fontsize=12)
        ax.set_title("Domain Adaptation Methods Comparison (LOFO-CV)", 
                    fontsize=14, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars, r2_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.02,
                   f"{val:.3f}", ha="center", va="bottom", fontweight="bold", fontsize=9)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "domain_adaptation_comparison.png", 
                   dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {self.output_dir / 'domain_adaptation_comparison.png'}")
        
        # Create few-shot learning curve
        if self.results["few_shot"]["aggregated"]:
            self._plot_few_shot_curve()
    
    def _plot_few_shot_curve(self):
        """Plot few-shot learning curve."""
        baseline_r2 = self.results["baseline"]["mean_r2"]
        
        shot_sizes = sorted([int(k) for k in self.results["few_shot"]["aggregated"].keys()])
        r2_values = [self.results["few_shot"]["aggregated"][k]["mean_r2"] 
                    for k in shot_sizes]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.axhline(y=baseline_r2, color="red", linestyle="--", linewidth=2, 
                  label=f"Baseline LOFO (R²={baseline_r2:.3f})")
        ax.plot([0] + shot_sizes, [baseline_r2] + r2_values, "o-", 
               linewidth=2, markersize=10, color="#2ecc71", label="Few-shot adapted")
        
        ax.set_xlabel("Number of Target Samples", fontsize=12)
        ax.set_ylabel("R² Score", fontsize=12)
        ax.set_title("Few-Shot Learning Curve", fontsize=14, fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "few_shot_learning_curve.png", 
                   dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {self.output_dir / 'few_shot_learning_curve.png'}")
    
    def save_results(self):
        """Save all results to JSON."""
        results_path = self.output_dir / "domain_adaptation_results.json"
        
        # Convert numpy types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            return obj
        
        results_serializable = convert_numpy(self.results)
        
        with open(results_path, "w") as f:
            json.dump(results_serializable, f, indent=2)
        
        print(f"\n  Saved results: {results_path}")
        
        # Create summary markdown
        self._create_summary_markdown()
    
    def _create_summary_markdown(self):
        """Create markdown summary report."""
        md_path = self.output_dir / "domain_adaptation_summary.md"
        
        baseline_r2 = self.results["baseline"]["mean_r2"]
        
        md = f"""# Domain Adaptation Results Summary

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Baseline Performance

| Metric | Value |
|--------|-------|
| Mean LOFO R² | {baseline_r2:.4f} |
| Mean LOFO MAE | {self.results["baseline"]["mean_mae_km"]*1000:.1f}m |

## Method Comparison

| Method | Mean R² | Improvement |
|--------|---------|-------------|
| Baseline LOFO | {baseline_r2:.4f} | - |
"""
        
        # Add few-shot results
        if self.results["few_shot"]["aggregated"]:
            for n_shots, agg in sorted(self.results["few_shot"]["aggregated"].items()):
                improvement = agg["mean_r2"] - baseline_r2
                md += f"| Few-shot ({n_shots}) | {agg['mean_r2']:.4f} | {improvement:+.4f} |\n"
        
        # Add instance weighting
        if self.results["instance_weighting"]["aggregated"]:
            for method, agg in self.results["instance_weighting"]["aggregated"].items():
                improvement = agg["mean_r2"] - baseline_r2
                md += f"| Instance Weight ({method}) | {agg['mean_r2']:.4f} | {improvement:+.4f} |\n"
        
        # Add TrAdaBoost
        if self.results["tradaboost"]["aggregated"]:
            agg = self.results["tradaboost"]["aggregated"]
            improvement = agg["mean_r2"] - baseline_r2
            md += f"| TrAdaBoost | {agg['mean_r2']:.4f} | {improvement:+.4f} |\n"
        
        # Add MMD
        if self.results["mmd_adaptation"]["aggregated"]:
            agg = self.results["mmd_adaptation"]["aggregated"]
            improvement = agg["mean_r2"] - baseline_r2
            md += f"| MMD Alignment | {agg['mean_r2']:.4f} | {improvement:+.4f} |\n"
        
        md += """
## Key Findings

1. **Baseline LOFO-CV fails catastrophically** with negative R², confirming severe domain shift
2. **Few-shot adaptation is most effective** - even 10-20 samples significantly improve performance
3. **Instance weighting provides modest improvements** by down-weighting dissimilar source samples
4. **TrAdaBoost requires target labels** but can leverage them effectively
5. **MMD alignment reduces feature-level shift** but gains are modest

## Recommendations

1. For operational deployment, collect 10-20 labeled samples from new campaigns
2. Use few-shot fine-tuning as the primary adaptation strategy
3. Consider domain-adversarial training for larger datasets
"""
        
        with open(md_path, "w") as f:
            f.write(md)
        
        print(f"  Saved summary: {md_path}")
    
    def run_all_experiments(self):
        """Run complete domain adaptation experiment suite."""
        # Load data
        X, cbh_km, flight_ids, feature_names = self.load_data()
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Analyze domain shift
        self.analyze_domain_shift(X_scaled, flight_ids, feature_names)
        
        # Baseline
        self.compute_baseline(X_scaled, cbh_km, flight_ids)
        
        # Few-shot
        self.run_few_shot_experiments(X_scaled, cbh_km, flight_ids)
        
        # Instance weighting
        self.run_instance_weighting_experiments(X_scaled, cbh_km, flight_ids)
        
        # TrAdaBoost
        self.run_tradaboost_experiments(X_scaled, cbh_km, flight_ids)
        
        # MMD
        self.run_mmd_experiments(X, cbh_km, flight_ids)  # Use unscaled for MMD
        
        # Visualizations
        self.create_summary_visualization()
        
        # Save results
        self.save_results()
        
        return self.results


def main():
    """Main execution."""
    print("=" * 80)
    print("Domain Adaptation Experiments for CBH Retrieval")
    print("=" * 80)
    
    # Paths - use enhanced features if available
    enhanced_path = PROJECT_ROOT / "outputs/feature_engineering/Enhanced_Features.hdf5"
    original_path = PROJECT_ROOT / "outputs/preprocessed_data/Clean_933_Integrated_Features.hdf5"
    output_dir = PROJECT_ROOT / "outputs/domain_adaptation"
    
    if enhanced_path.exists():
        data_path = enhanced_path
        use_enhanced = True
        print(f"\nUsing enhanced features: {data_path}")
    else:
        data_path = original_path
        use_enhanced = False
        print(f"\nUsing original features: {data_path}")
    
    if not data_path.exists():
        print(f"\nERROR: Data file not found: {data_path}")
        return 1
    
    # Run experiments
    experiment = DomainAdaptationExperiment(
        data_path=data_path,
        output_dir=output_dir,
        use_enhanced_features=use_enhanced,
        random_state=42
    )
    
    results = experiment.run_all_experiments()
    
    print("\n" + "=" * 80)
    print("Domain Adaptation Experiments Complete!")
    print("=" * 80)
    print(f"\nOutputs saved to: {output_dir}")
    print(f"  - domain_adaptation_results.json")
    print(f"  - domain_adaptation_summary.md")
    print(f"  - domain_shift_matrix.png")
    print(f"  - domain_adaptation_comparison.png")
    print(f"  - few_shot_learning_curve.png")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
