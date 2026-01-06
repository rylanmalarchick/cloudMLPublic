#!/usr/bin/env python3
"""
Uncertainty Quantification for CBH Retrieval with Conformal Prediction

Implements rigorous uncertainty quantification using:
1. Split Conformal Prediction - Distribution-free prediction intervals
2. Adaptive Conformal Prediction - Handles distribution shift
3. Quantile Regression - Direct interval estimation
4. Per-flight calibration analysis

Conformal prediction provides guaranteed coverage under exchangeability,
which is an important caveat for time-series CBH data with autocorrelation.

References:
- Angelopoulos & Bates (2021) "A Gentle Introduction to Conformal Prediction"
- Gibbs & Candes (2021) "Adaptive Conformal Inference Under Distribution Shift"

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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_absolute_error

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# AgentBible integration
try:
    import agentbible
    from agentbible.errors import ValidationError
    AGENTBIBLE_VERSION = agentbible.__version__
except ImportError:
    AGENTBIBLE_VERSION = "N/A"
    class ValidationError(Exception):
        pass

print(f"AgentBible version: {AGENTBIBLE_VERSION}")

# Plotting style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")


class SplitConformalPredictor:
    """
    Split Conformal Prediction for Regression.
    
    Provides prediction intervals with guaranteed finite-sample coverage
    under the exchangeability assumption.
    
    Algorithm:
    1. Split data into train, calibration, and test sets
    2. Fit model on training data
    3. Compute nonconformity scores on calibration data
    4. Use calibration quantile for prediction intervals on test data
    
    Reference: Lei et al. (2018) "Distribution-Free Predictive Inference for Regression"
    """
    
    def __init__(self, base_model, alpha: float = 0.1):
        """
        Args:
            base_model: Fitted sklearn regression model
            alpha: Miscoverage level (1 - alpha is target coverage)
        """
        self.base_model = base_model
        self.alpha = alpha
        self.calibration_scores = None
        self.quantile = None
        
    def calibrate(self, X_cal: np.ndarray, y_cal: np.ndarray):
        """
        Calibrate conformal predictor using holdout calibration set.
        
        Args:
            X_cal: Calibration features
            y_cal: Calibration labels
        """
        # Get predictions on calibration set
        y_pred_cal = self.base_model.predict(X_cal)
        
        # Compute nonconformity scores (absolute residuals)
        self.calibration_scores = np.abs(y_cal - y_pred_cal)
        
        # Compute conformal quantile with finite-sample correction
        n = len(self.calibration_scores)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        q_level = min(q_level, 1.0)  # Ensure valid quantile
        
        self.quantile = np.quantile(self.calibration_scores, q_level)
        
        return {
            "n_calibration": n,
            "target_coverage": 1 - self.alpha,
            "quantile_level": float(q_level),
            "interval_halfwidth_km": float(self.quantile),
        }
    
    def predict(self, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict with conformal prediction intervals.
        
        Args:
            X_test: Test features
            
        Returns:
            y_pred: Point predictions
            y_lower: Lower bounds
            y_upper: Upper bounds
        """
        if self.quantile is None:
            raise ValueError("Must call calibrate() before predict()")
        
        y_pred = self.base_model.predict(X_test)
        y_lower = y_pred - self.quantile
        y_upper = y_pred + self.quantile
        
        return y_pred, y_lower, y_upper
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate coverage and interval width on test set."""
        y_pred, y_lower, y_upper = self.predict(X_test)
        
        # Coverage
        covered = (y_lower <= y_test) & (y_test <= y_upper)
        coverage = np.mean(covered)
        
        # Interval width
        interval_width = np.mean(y_upper - y_lower)
        
        # Point prediction metrics
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        return {
            "coverage": float(coverage),
            "target_coverage": 1 - self.alpha,
            "mean_interval_width_km": float(interval_width),
            "r2": float(r2),
            "mae_km": float(mae),
            "n_test": len(y_test),
        }


class AdaptiveConformalPredictor:
    """
    Adaptive Conformal Prediction for handling distribution shift.
    
    Updates prediction intervals online to maintain coverage under
    gradual distribution changes (e.g., across flights, time drift).
    
    Reference: Gibbs & Candes (2021) "Adaptive Conformal Inference Under Distribution Shift"
    """
    
    def __init__(self, base_model, alpha: float = 0.1, gamma: float = 0.01):
        """
        Args:
            base_model: Fitted sklearn regression model
            alpha: Target miscoverage level
            gamma: Learning rate for quantile updates (smaller = more stable)
        """
        self.base_model = base_model
        self.alpha = alpha
        self.gamma = gamma
        
    def predict_adaptive(self, X_test: np.ndarray, y_test: np.ndarray,
                        initial_quantile: float) -> Dict[str, Any]:
        """
        Online adaptive prediction with quantile updates.
        
        Args:
            X_test: Test features (ordered by time/flight)
            y_test: Test labels
            initial_quantile: Initial interval half-width
            
        Returns:
            Dictionary with predictions, intervals, and tracking metrics
        """
        n_test = len(X_test)
        y_pred = self.base_model.predict(X_test)
        
        # Initialize
        quantile_t = initial_quantile
        y_lower = np.zeros(n_test)
        y_upper = np.zeros(n_test)
        quantile_history = []
        coverage_history = []
        
        for t in range(n_test):
            # Prediction interval at time t
            y_lower[t] = y_pred[t] - quantile_t
            y_upper[t] = y_pred[t] + quantile_t
            
            quantile_history.append(quantile_t)
            
            # Check coverage
            covered = (y_lower[t] <= y_test[t] <= y_upper[t])
            coverage_history.append(float(covered))
            
            # Adaptive update using error feedback
            # If under-covering (err > 0), increase quantile
            # If over-covering (err < 0), decrease quantile
            err = self.alpha - (1 - float(covered))
            quantile_t = quantile_t * np.exp(self.gamma * err)
            
            # Clip to reasonable range
            quantile_t = np.clip(quantile_t, 0.01, 2.0)
        
        # Final coverage
        final_coverage = np.mean(coverage_history)
        
        return {
            "y_pred": y_pred,
            "y_lower": y_lower,
            "y_upper": y_upper,
            "quantile_history": np.array(quantile_history),
            "coverage_history": np.array(coverage_history),
            "final_coverage": float(final_coverage),
            "mean_interval_width_km": float(np.mean(y_upper - y_lower)),
            "r2": float(r2_score(y_test, y_pred)),
            "mae_km": float(mean_absolute_error(y_test, y_pred)),
        }


class QuantileRegressionPredictor:
    """
    Quantile Regression using Gradient Boosting.
    
    Directly estimates conditional quantiles for prediction intervals.
    """
    
    def __init__(self, alpha: float = 0.1, random_state: int = 42):
        """
        Args:
            alpha: Miscoverage level (predicts alpha/2 and 1-alpha/2 quantiles)
        """
        self.alpha = alpha
        self.lower_quantile = alpha / 2
        self.upper_quantile = 1 - alpha / 2
        self.random_state = random_state
        self.models = {}
        
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """Fit quantile regression models."""
        base_params = {
            "n_estimators": 100,
            "max_depth": 4,
            "learning_rate": 0.1,
            "min_samples_split": 5,
            "random_state": self.random_state,
        }
        
        # Lower quantile model
        self.models["lower"] = GradientBoostingRegressor(
            loss="quantile", alpha=self.lower_quantile, **base_params
        )
        self.models["lower"].fit(X_train, y_train)
        
        # Median model
        self.models["median"] = GradientBoostingRegressor(
            loss="quantile", alpha=0.5, **base_params
        )
        self.models["median"].fit(X_train, y_train)
        
        # Upper quantile model
        self.models["upper"] = GradientBoostingRegressor(
            loss="quantile", alpha=self.upper_quantile, **base_params
        )
        self.models["upper"].fit(X_train, y_train)
        
    def predict(self, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict with quantile intervals."""
        y_pred = self.models["median"].predict(X_test)
        y_lower = self.models["lower"].predict(X_test)
        y_upper = self.models["upper"].predict(X_test)
        
        # Ensure proper ordering (lower < median < upper)
        y_lower = np.minimum(y_lower, y_pred)
        y_upper = np.maximum(y_upper, y_pred)
        
        return y_pred, y_lower, y_upper
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate coverage and interval width."""
        y_pred, y_lower, y_upper = self.predict(X_test)
        
        covered = (y_lower <= y_test) & (y_test <= y_upper)
        coverage = np.mean(covered)
        interval_width = np.mean(y_upper - y_lower)
        
        return {
            "coverage": float(coverage),
            "target_coverage": 1 - self.alpha,
            "mean_interval_width_km": float(interval_width),
            "r2": float(r2_score(y_test, y_pred)),
            "mae_km": float(mean_absolute_error(y_test, y_pred)),
        }


class UncertaintyQuantificationExperiment:
    """
    Complete uncertainty quantification experiment for CBH retrieval.
    """
    
    def __init__(self, data_path: Path, output_dir: Path, 
                 alpha: float = 0.1, random_state: int = 42):
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.alpha = alpha
        self.random_state = random_state
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "data_path": str(self.data_path),
                "target_coverage": 1 - alpha,
                "alpha": alpha,
                "random_state": random_state,
                "agentbible_version": AGENTBIBLE_VERSION,
            },
            "split_conformal": {},
            "adaptive_conformal": {},
            "quantile_regression": {},
            "per_flight_calibration": {},
            "comparison": {},
        }
        
    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """Load feature data from HDF5 file."""
        print("\n" + "=" * 80)
        print("Loading Data")
        print("=" * 80)
        
        with h5py.File(self.data_path, "r") as f:
            if "features" in f:
                # Enhanced features
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
        print(f"  CBH range: [{cbh_km.min():.3f}, {cbh_km.max():.3f}] km")
        
        return X, cbh_km, flight_ids, feature_names
    
    def run_split_conformal(self, X: np.ndarray, y: np.ndarray,
                           flight_ids: np.ndarray) -> Dict:
        """Run split conformal prediction experiment."""
        print("\n" + "=" * 80)
        print("Split Conformal Prediction")
        print("=" * 80)
        
        # Use per-flight shuffled CV for realistic evaluation
        unique_flights = sorted(set(flight_ids))
        # Filter to flights with enough samples
        valid_flights = [f for f in unique_flights 
                        if np.sum(flight_ids == f) >= 50]
        
        all_results = []
        
        for target_flight in valid_flights:
            # All other flights for training, target for test
            train_mask = flight_ids != target_flight
            test_mask = flight_ids == target_flight
            
            X_train_full = X[train_mask]
            y_train_full = y[train_mask]
            X_test = X[test_mask]
            y_test = y[test_mask]
            
            # Split train into train + calibration (70/30)
            n_cal = int(0.3 * len(X_train_full))
            indices = np.random.RandomState(self.random_state).permutation(len(X_train_full))
            
            X_cal = X_train_full[indices[:n_cal]]
            y_cal = y_train_full[indices[:n_cal]]
            X_train = X_train_full[indices[n_cal:]]
            y_train = y_train_full[indices[n_cal:]]
            
            # Scale
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_cal_scaled = scaler.transform(X_cal)
            X_test_scaled = scaler.transform(X_test)
            
            # Train base model
            model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                random_state=self.random_state,
            )
            model.fit(X_train_scaled, y_train)
            
            # Calibrate conformal predictor
            cp = SplitConformalPredictor(model, alpha=self.alpha)
            cal_info = cp.calibrate(X_cal_scaled, y_cal)
            
            # Evaluate on test
            result = cp.evaluate(X_test_scaled, y_test)
            result["flight"] = target_flight
            result["calibration_info"] = cal_info
            
            all_results.append(result)
            
            print(f"  Flight {target_flight}: coverage={result['coverage']:.1%}, "
                  f"width={result['mean_interval_width_km']*1000:.1f}m, R²={result['r2']:.3f}")
        
        # Aggregate
        mean_coverage = np.mean([r["coverage"] for r in all_results])
        mean_width = np.mean([r["mean_interval_width_km"] for r in all_results])
        mean_r2 = np.mean([r["r2"] for r in all_results])
        
        aggregated = {
            "mean_coverage": float(mean_coverage),
            "target_coverage": 1 - self.alpha,
            "coverage_gap": float(abs(mean_coverage - (1 - self.alpha))),
            "mean_interval_width_km": float(mean_width),
            "mean_r2": float(mean_r2),
            "per_flight_results": all_results,
        }
        
        print(f"\n  Overall: coverage={mean_coverage:.1%} (target={1-self.alpha:.0%}), "
              f"width={mean_width*1000:.1f}m")
        
        self.results["split_conformal"] = aggregated
        return aggregated
    
    def run_adaptive_conformal(self, X: np.ndarray, y: np.ndarray,
                               flight_ids: np.ndarray) -> Dict:
        """Run adaptive conformal prediction experiment."""
        print("\n" + "=" * 80)
        print("Adaptive Conformal Prediction")
        print("=" * 80)
        
        unique_flights = sorted(set(flight_ids))
        valid_flights = [f for f in unique_flights 
                        if np.sum(flight_ids == f) >= 50]
        
        all_results = []
        
        for target_flight in valid_flights:
            train_mask = flight_ids != target_flight
            test_mask = flight_ids == target_flight
            
            X_train = X[train_mask]
            y_train = y[train_mask]
            X_test = X[test_mask]
            y_test = y[test_mask]
            
            # Scale
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train base model
            model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                random_state=self.random_state,
            )
            model.fit(X_train_scaled, y_train)
            
            # Initial quantile from training residuals
            y_pred_train = model.predict(X_train_scaled)
            initial_quantile = np.quantile(np.abs(y_train - y_pred_train), 1 - self.alpha)
            
            # Adaptive conformal prediction
            acp = AdaptiveConformalPredictor(model, alpha=self.alpha, gamma=0.01)
            result = acp.predict_adaptive(X_test_scaled, y_test, initial_quantile)
            
            result["flight"] = target_flight
            result["initial_quantile"] = float(initial_quantile)
            
            # Remove large arrays from result for storage
            result_summary = {k: v for k, v in result.items() 
                            if not isinstance(v, np.ndarray)}
            result_summary["final_quantile"] = float(result["quantile_history"][-1])
            
            all_results.append(result_summary)
            
            print(f"  Flight {target_flight}: coverage={result['final_coverage']:.1%}, "
                  f"width={result['mean_interval_width_km']*1000:.1f}m, "
                  f"quantile: {initial_quantile*1000:.1f}→{result['quantile_history'][-1]*1000:.1f}m")
        
        # Aggregate
        mean_coverage = np.mean([r["final_coverage"] for r in all_results])
        mean_width = np.mean([r["mean_interval_width_km"] for r in all_results])
        
        aggregated = {
            "mean_coverage": float(mean_coverage),
            "target_coverage": 1 - self.alpha,
            "mean_interval_width_km": float(mean_width),
            "per_flight_results": all_results,
        }
        
        print(f"\n  Overall: coverage={mean_coverage:.1%}, width={mean_width*1000:.1f}m")
        
        self.results["adaptive_conformal"] = aggregated
        return aggregated
    
    def run_quantile_regression(self, X: np.ndarray, y: np.ndarray,
                                flight_ids: np.ndarray) -> Dict:
        """Run quantile regression experiment."""
        print("\n" + "=" * 80)
        print("Quantile Regression")
        print("=" * 80)
        
        unique_flights = sorted(set(flight_ids))
        valid_flights = [f for f in unique_flights 
                        if np.sum(flight_ids == f) >= 50]
        
        all_results = []
        
        for target_flight in valid_flights:
            train_mask = flight_ids != target_flight
            test_mask = flight_ids == target_flight
            
            X_train = X[train_mask]
            y_train = y[train_mask]
            X_test = X[test_mask]
            y_test = y[test_mask]
            
            # Scale
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train quantile regression
            qr = QuantileRegressionPredictor(alpha=self.alpha, random_state=self.random_state)
            qr.fit(X_train_scaled, y_train)
            
            # Evaluate
            result = qr.evaluate(X_test_scaled, y_test)
            result["flight"] = target_flight
            
            all_results.append(result)
            
            print(f"  Flight {target_flight}: coverage={result['coverage']:.1%}, "
                  f"width={result['mean_interval_width_km']*1000:.1f}m, R²={result['r2']:.3f}")
        
        # Aggregate
        mean_coverage = np.mean([r["coverage"] for r in all_results])
        mean_width = np.mean([r["mean_interval_width_km"] for r in all_results])
        mean_r2 = np.mean([r["r2"] for r in all_results])
        
        aggregated = {
            "mean_coverage": float(mean_coverage),
            "target_coverage": 1 - self.alpha,
            "mean_interval_width_km": float(mean_width),
            "mean_r2": float(mean_r2),
            "per_flight_results": all_results,
        }
        
        print(f"\n  Overall: coverage={mean_coverage:.1%}, width={mean_width*1000:.1f}m")
        
        self.results["quantile_regression"] = aggregated
        return aggregated
    
    def run_per_flight_calibration(self, X: np.ndarray, y: np.ndarray,
                                   flight_ids: np.ndarray) -> Dict:
        """Analyze calibration stratified by flight."""
        print("\n" + "=" * 80)
        print("Per-Flight Calibration Analysis")
        print("=" * 80)
        
        unique_flights = sorted(set(flight_ids))
        valid_flights = [f for f in unique_flights 
                        if np.sum(flight_ids == f) >= 30]
        
        # Train a global model using all data with K-fold CV
        kf = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        all_preds = np.zeros_like(y)
        all_lower = np.zeros_like(y)
        all_upper = np.zeros_like(y)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        for train_idx, test_idx in kf.split(X_scaled):
            # Train quantile regression on this fold
            qr = QuantileRegressionPredictor(alpha=self.alpha, random_state=self.random_state)
            qr.fit(X_scaled[train_idx], y[train_idx])
            
            pred, lower, upper = qr.predict(X_scaled[test_idx])
            all_preds[test_idx] = pred
            all_lower[test_idx] = lower
            all_upper[test_idx] = upper
        
        # Compute per-flight calibration
        per_flight_results = {}
        
        for flight in valid_flights:
            mask = flight_ids == flight
            
            covered = (all_lower[mask] <= y[mask]) & (y[mask] <= all_upper[mask])
            coverage = np.mean(covered)
            width = np.mean(all_upper[mask] - all_lower[mask])
            r2 = r2_score(y[mask], all_preds[mask])
            
            per_flight_results[flight] = {
                "coverage": float(coverage),
                "mean_interval_width_km": float(width),
                "r2": float(r2),
                "n_samples": int(mask.sum()),
                "mean_cbh_km": float(y[mask].mean()),
                "std_cbh_km": float(y[mask].std()),
            }
            
            print(f"  Flight {flight}: coverage={coverage:.1%}, width={width*1000:.1f}m, "
                  f"n={mask.sum()}, CBH={y[mask].mean():.2f}±{y[mask].std():.2f} km")
        
        # Overall
        covered = (all_lower <= y) & (y <= all_upper)
        overall_coverage = np.mean(covered)
        
        self.results["per_flight_calibration"] = {
            "per_flight": per_flight_results,
            "overall_coverage": float(overall_coverage),
            "target_coverage": 1 - self.alpha,
        }
        
        print(f"\n  Overall coverage: {overall_coverage:.1%} (target: {1-self.alpha:.0%})")
        
        return self.results["per_flight_calibration"]
    
    def create_comparison(self):
        """Create method comparison table."""
        print("\n" + "=" * 80)
        print("Method Comparison")
        print("=" * 80)
        
        comparison = {}
        
        if self.results["split_conformal"]:
            comparison["Split Conformal"] = {
                "coverage": self.results["split_conformal"]["mean_coverage"],
                "width_m": self.results["split_conformal"]["mean_interval_width_km"] * 1000,
            }
        
        if self.results["adaptive_conformal"]:
            comparison["Adaptive Conformal"] = {
                "coverage": self.results["adaptive_conformal"]["mean_coverage"],
                "width_m": self.results["adaptive_conformal"]["mean_interval_width_km"] * 1000,
            }
        
        if self.results["quantile_regression"]:
            comparison["Quantile Regression"] = {
                "coverage": self.results["quantile_regression"]["mean_coverage"],
                "width_m": self.results["quantile_regression"]["mean_interval_width_km"] * 1000,
            }
        
        self.results["comparison"] = comparison
        
        # Print table
        print(f"\n  {'Method':<25} {'Coverage':<12} {'Width (m)':<12}")
        print("  " + "-" * 50)
        for method, metrics in comparison.items():
            cov = metrics["coverage"]
            cov_str = f"{cov:.1%}"
            if abs(cov - (1 - self.alpha)) < 0.02:
                cov_str += " ✓"
            print(f"  {method:<25} {cov_str:<12} {metrics['width_m']:<12.1f}")
        
        return comparison
    
    def create_visualizations(self):
        """Create uncertainty quantification visualizations."""
        print("\n" + "=" * 80)
        print("Creating Visualizations")
        print("=" * 80)
        
        # 1. Method comparison bar chart
        self._plot_method_comparison()
        
        # 2. Calibration curves
        self._plot_calibration_by_flight()
        
        # 3. Coverage vs interval width tradeoff
        self._plot_coverage_width_tradeoff()
        
        print(f"  Visualizations saved to {self.output_dir}")
    
    def _plot_method_comparison(self):
        """Plot coverage and width comparison across methods."""
        comparison = self.results["comparison"]
        
        if not comparison:
            return
        
        methods = list(comparison.keys())
        coverages = [comparison[m]["coverage"] for m in methods]
        widths = [comparison[m]["width_m"] for m in methods]
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Coverage
        ax = axes[0]
        colors = ["#2ecc71" if abs(c - (1-self.alpha)) < 0.02 else "#e74c3c" 
                 for c in coverages]
        bars = ax.bar(methods, coverages, color=colors, edgecolor="black")
        ax.axhline(y=1-self.alpha, color="red", linestyle="--", linewidth=2,
                  label=f"Target ({1-self.alpha:.0%})")
        ax.set_ylabel("Coverage", fontsize=12)
        ax.set_title("Coverage by Method", fontsize=14, fontweight="bold")
        ax.legend()
        ax.set_ylim([0, 1.1])
        
        for bar, cov in zip(bars, coverages):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f"{cov:.1%}", ha="center", fontweight="bold")
        
        # Width
        ax = axes[1]
        ax.bar(methods, widths, color="#3498db", edgecolor="black")
        ax.set_ylabel("Mean Interval Width (m)", fontsize=12)
        ax.set_title("Interval Width by Method", fontsize=14, fontweight="bold")
        
        for bar, w in zip(ax.patches, widths):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                   f"{w:.0f}m", ha="center", fontweight="bold")
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "uq_method_comparison.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved: uq_method_comparison.png")
    
    def _plot_calibration_by_flight(self):
        """Plot per-flight calibration analysis."""
        if not self.results["per_flight_calibration"]:
            return
        
        per_flight = self.results["per_flight_calibration"]["per_flight"]
        
        flights = list(per_flight.keys())
        coverages = [per_flight[f]["coverage"] for f in flights]
        widths = [per_flight[f]["mean_interval_width_km"] * 1000 for f in flights]
        n_samples = [per_flight[f]["n_samples"] for f in flights]
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Coverage by flight
        ax = axes[0]
        colors = ["#2ecc71" if abs(c - (1-self.alpha)) < 0.05 else "#e74c3c" 
                 for c in coverages]
        bars = ax.bar([f"F{f}" for f in flights], coverages, color=colors, edgecolor="black")
        ax.axhline(y=1-self.alpha, color="red", linestyle="--", linewidth=2,
                  label=f"Target ({1-self.alpha:.0%})")
        ax.set_ylabel("Coverage", fontsize=12)
        ax.set_title("Coverage by Flight", fontsize=14, fontweight="bold")
        ax.legend()
        ax.set_ylim([0, 1.1])
        
        # Width by flight
        ax = axes[1]
        ax.bar([f"F{f}" for f in flights], widths, color="#3498db", edgecolor="black")
        ax.set_ylabel("Interval Width (m)", fontsize=12)
        ax.set_title("Interval Width by Flight", fontsize=14, fontweight="bold")
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "uq_per_flight_calibration.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved: uq_per_flight_calibration.png")
    
    def _plot_coverage_width_tradeoff(self):
        """Plot coverage vs width tradeoff across methods and flights."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        markers = {"Split Conformal": "o", "Adaptive Conformal": "s", "Quantile Regression": "^"}
        colors = {"Split Conformal": "#e74c3c", "Adaptive Conformal": "#3498db", "Quantile Regression": "#2ecc71"}
        
        for method_name, method_results in [
            ("Split Conformal", self.results["split_conformal"]),
            ("Quantile Regression", self.results["quantile_regression"]),
        ]:
            if not method_results or "per_flight_results" not in method_results:
                continue
            
            for r in method_results["per_flight_results"]:
                ax.scatter(r["mean_interval_width_km"] * 1000, r["coverage"],
                          marker=markers[method_name], c=colors[method_name],
                          s=100, alpha=0.7, label=method_name if r == method_results["per_flight_results"][0] else "")
        
        ax.axhline(y=1-self.alpha, color="gray", linestyle="--", linewidth=1,
                  label=f"Target coverage ({1-self.alpha:.0%})")
        
        ax.set_xlabel("Mean Interval Width (m)", fontsize=12)
        ax.set_ylabel("Coverage", fontsize=12)
        ax.set_title("Coverage vs Interval Width Tradeoff", fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "uq_coverage_width_tradeoff.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved: uq_coverage_width_tradeoff.png")
    
    def save_results(self):
        """Save all results to JSON."""
        results_path = self.output_dir / "uncertainty_quantification_results.json"
        
        with open(results_path, "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\n  Saved results: {results_path}")
        
        # Create markdown summary
        self._create_summary_markdown()
    
    def _create_summary_markdown(self):
        """Create markdown summary report."""
        md_path = self.output_dir / "uncertainty_quantification_summary.md"
        
        target = 1 - self.alpha
        
        md = f"""# Uncertainty Quantification Results Summary

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Configuration

| Parameter | Value |
|-----------|-------|
| Target Coverage | {target:.0%} |
| Alpha | {self.alpha} |

## Method Comparison

| Method | Coverage | Width (m) | Status |
|--------|----------|-----------|--------|
"""
        
        for method, metrics in self.results["comparison"].items():
            cov = metrics["coverage"]
            status = "✓" if abs(cov - target) < 0.02 else "✗"
            md += f"| {method} | {cov:.1%} | {metrics['width_m']:.0f} | {status} |\n"
        
        md += """
## Key Findings

1. **Split Conformal Prediction** provides finite-sample coverage guarantees under exchangeability
2. **Adaptive Conformal** adjusts intervals online to handle distribution shift across flights
3. **Quantile Regression** directly estimates conditional quantiles but may miscalibrate

## Caveats

- **Temporal autocorrelation** violates exchangeability assumption for conformal prediction
- **Domain shift** between flights causes calibration to vary
- **Small flights** (< 50 samples) have high variance in coverage estimates

## Recommendations

1. Use **Split Conformal** for flight-specific deployment with dedicated calibration set
2. Use **Adaptive Conformal** when processing sequential data with potential drift
3. Monitor coverage at deployment and recalibrate periodically
"""
        
        with open(md_path, "w") as f:
            f.write(md)
        
        print(f"  Saved summary: {md_path}")
    
    def run_all_experiments(self):
        """Run complete uncertainty quantification experiment suite."""
        # Load data
        X, cbh_km, flight_ids, feature_names = self.load_data()
        
        # Run experiments
        self.run_split_conformal(X, cbh_km, flight_ids)
        self.run_adaptive_conformal(X, cbh_km, flight_ids)
        self.run_quantile_regression(X, cbh_km, flight_ids)
        self.run_per_flight_calibration(X, cbh_km, flight_ids)
        
        # Comparison
        self.create_comparison()
        
        # Visualizations
        self.create_visualizations()
        
        # Save
        self.save_results()
        
        return self.results


def main():
    """Main execution."""
    print("=" * 80)
    print("Uncertainty Quantification for CBH Retrieval")
    print("=" * 80)
    
    # Paths - use enhanced features if available
    enhanced_path = PROJECT_ROOT / "outputs/feature_engineering/Enhanced_Features.hdf5"
    original_path = PROJECT_ROOT / "outputs/preprocessed_data/Clean_933_Integrated_Features.hdf5"
    output_dir = PROJECT_ROOT / "outputs/uncertainty"
    
    if enhanced_path.exists():
        data_path = enhanced_path
        print(f"\nUsing enhanced features: {data_path}")
    else:
        data_path = original_path
        print(f"\nUsing original features: {data_path}")
    
    if not data_path.exists():
        print(f"\nERROR: Data file not found: {data_path}")
        return 1
    
    # Run experiments
    experiment = UncertaintyQuantificationExperiment(
        data_path=data_path,
        output_dir=output_dir,
        alpha=0.1,  # Target 90% coverage
        random_state=42
    )
    
    results = experiment.run_all_experiments()
    
    print("\n" + "=" * 80)
    print("Uncertainty Quantification Complete!")
    print("=" * 80)
    print(f"\nOutputs saved to: {output_dir}")
    print(f"  - uncertainty_quantification_results.json")
    print(f"  - uncertainty_quantification_summary.md")
    print(f"  - uq_method_comparison.png")
    print(f"  - uq_per_flight_calibration.png")
    print(f"  - uq_coverage_width_tradeoff.png")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
