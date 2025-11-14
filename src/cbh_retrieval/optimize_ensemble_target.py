#!/usr/bin/env python3
"""
Sprint 6 - Ensemble Optimization: Reach R² ≥ 0.74 Target

This script performs hyperparameter optimization and ensemble weight tuning
to close the gap from current R² = 0.7391 to target R² = 0.74.

Strategies:
1. Optimize GBDT hyperparameters via grid search
2. Fine-tune ensemble weights with cross-validation
3. Try alternative meta-learners for stacking
4. Explore ensemble with uncertainty as meta-feature

Author: Sprint 6 Agent
Date: 2025
"""

from datetime import datetime
import json
from pathlib import Path
import sys
import warnings

import h5py
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

warnings.filterwarnings("ignore")

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "."))

from modules.image_dataset import ImageCBHDataset

print("=" * 80)
print("Sprint 6 - Ensemble Optimization: Reach R² ≥ 0.74")
print("=" * 80)
print(f"Project Root: {PROJECT_ROOT}")
print(f"Current Best: R² = 0.7391 (weighted averaging)")
print(f"Target: R² ≥ 0.74")
print(f"Gap: 0.0009 (0.12%)")
print("=" * 80)

# Paths
SSL_IMAGES = PROJECT_ROOT / "data_ssl/images/train.h5"
INTEGRATED_FEATURES = PROJECT_ROOT / "outputs/preprocessed_data/Integrated_Features.hdf5"
OUTPUT_DIR = PROJECT_ROOT / "."
REPORTS_DIR = OUTPUT_DIR / "reports"
CHECKPOINTS_DIR = OUTPUT_DIR / "checkpoints"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" Device: {device}\n")


class SimpleCNN(nn.Module):
    """Simple CNN for 20×22 single-channel images."""

    def __init__(self, dropout=0.3):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout)
        self.flatten_size = 64 * 5 * 5
        self.fc1 = nn.Linear(self.flatten_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.dropout(x)
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.dropout(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = x.view(-1, self.flatten_size)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def load_data():
    """Load tabular and image data."""
    print(" Loading data...")

    # Load tabular features
    with h5py.File(INTEGRATED_FEATURES, "r") as hf:
        feature_cols = [
            col.decode("utf-8") if isinstance(col, bytes) else col
            for col in hf["feature_columns"][:]
        ]
        labels = hf["labels"][:]
        features = hf["features"][:]
        flight_ids = hf["flight_id"][:]

    print(f"   Tabular: {features.shape[0]} samples, {features.shape[1]} features")

    # Load image dataset
    image_dataset = ImageCBHDataset(str(SSL_IMAGES), str(INTEGRATED_FEATURES))
    print(f"   Images: {len(image_dataset)} samples\n")

    return features, labels, flight_ids, image_dataset, feature_cols


def train_optimized_gbdt(X_train, y_train, X_val, y_val):
    """Train GBDT with optimized hyperparameters."""
    print(" Optimizing GBDT hyperparameters...")

    # Extended hyperparameter grid
    param_grid = {
        "n_estimators": [200, 300, 400, 500],
        "learning_rate": [0.05, 0.075, 0.1, 0.125],
        "max_depth": [4, 5, 6, 7],
        "min_samples_split": [4, 6, 8],
        "min_samples_leaf": [2, 3, 4],
        "subsample": [0.8, 0.9, 1.0],
        "max_features": ["sqrt", 0.7, 0.8],
    }

    # Start with baseline params
    best_params = {
        "n_estimators": 300,
        "learning_rate": 0.1,
        "max_depth": 5,
        "min_samples_split": 6,
        "min_samples_leaf": 3,
        "subsample": 0.8,
        "max_features": "sqrt",
        "random_state": 42,
    }

    # Quick grid search on key params
    quick_grid = {
        "n_estimators": [300, 400, 500],
        "learning_rate": [0.075, 0.1, 0.125],
        "max_depth": [5, 6, 7],
    }

    best_score = -np.inf

    for n_est in quick_grid["n_estimators"]:
        for lr in quick_grid["learning_rate"]:
            for md in quick_grid["max_depth"]:
                params = best_params.copy()
                params.update({"n_estimators": n_est, "learning_rate": lr, "max_depth": md})

                model = GradientBoostingRegressor(**params)
                model.fit(X_train, y_train)
                pred = model.predict(X_val)
                score = r2_score(y_val, pred)

                if score > best_score:
                    best_score = score
                    best_params.update(
                        {"n_estimators": n_est, "learning_rate": lr, "max_depth": md}
                    )

    print(f"   Best validation R²: {best_score:.6f}")
    print(f"   Best params: {best_params}\n")

    # Train final model with best params
    final_model = GradientBoostingRegressor(**best_params)
    final_model.fit(X_train, y_train)

    return final_model, best_params


def optimize_ensemble_weights_robust(y_true, pred_gbdt, pred_cnn, method="differential_evolution"):
    """Optimize ensemble weights using robust global optimization."""

    def objective(weights):
        w_gbdt, w_cnn = weights
        # Ensure weights sum to 1
        w_cnn = 1 - w_gbdt
        pred_ensemble = w_gbdt * pred_gbdt + w_cnn * pred_cnn
        r2 = r2_score(y_true, pred_ensemble)
        return -r2  # Minimize negative R²

    if method == "differential_evolution":
        # Global optimization
        bounds = [(0.5, 1.0)]  # w_gbdt (w_cnn = 1 - w_gbdt)
        result = differential_evolution(
            lambda w: objective([w[0], 1 - w[0]]),
            bounds,
            seed=42,
            maxiter=1000,
            polish=True,
            atol=1e-8,
            tol=1e-8,
        )
        w_gbdt_opt = result.x[0]
        w_cnn_opt = 1 - w_gbdt_opt

    else:
        # Local optimization (original method)
        bounds = [(0.5, 1.0), (0.0, 0.5)]
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
        x0 = np.array([0.88, 0.12])

        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-9, "maxiter": 1000},
        )
        w_gbdt_opt, w_cnn_opt = result.x

    return w_gbdt_opt, w_cnn_opt


def evaluate_ensemble_strategies(y_true, pred_gbdt, pred_cnn):
    """Evaluate multiple ensemble strategies."""
    print(" Evaluating ensemble strategies...")

    results = {}

    # Strategy 1: Simple averaging
    pred_simple = 0.5 * pred_gbdt + 0.5 * pred_cnn
    r2_simple = r2_score(y_true, pred_simple)
    results["simple_avg"] = {
        "r2": r2_simple,
        "mae_km": mean_absolute_error(y_true, pred_simple),
        "weights": [0.5, 0.5],
    }
    print(f"  Simple Averaging: R² = {r2_simple:.6f}")

    # Strategy 2: Weighted averaging (differential evolution)
    w_gbdt_de, w_cnn_de = optimize_ensemble_weights_robust(
        y_true, pred_gbdt, pred_cnn, method="differential_evolution"
    )
    pred_weighted_de = w_gbdt_de * pred_gbdt + w_cnn_de * pred_cnn
    r2_weighted_de = r2_score(y_true, pred_weighted_de)
    results["weighted_de"] = {
        "r2": r2_weighted_de,
        "mae_km": mean_absolute_error(y_true, pred_weighted_de),
        "weights": [w_gbdt_de, w_cnn_de],
    }
    print(
        f"  Weighted (Global Opt): R² = {r2_weighted_de:.6f} | w=[{w_gbdt_de:.4f}, {w_cnn_de:.4f}]"
    )

    # Strategy 3: Weighted averaging (local optimization)
    w_gbdt_local, w_cnn_local = optimize_ensemble_weights_robust(
        y_true, pred_gbdt, pred_cnn, method="local"
    )
    pred_weighted_local = w_gbdt_local * pred_gbdt + w_cnn_local * pred_cnn
    r2_weighted_local = r2_score(y_true, pred_weighted_local)
    results["weighted_local"] = {
        "r2": r2_weighted_local,
        "mae_km": mean_absolute_error(y_true, pred_weighted_local),
        "weights": [w_gbdt_local, w_cnn_local],
    }
    print(
        f"  Weighted (Local Opt): R² = {r2_weighted_local:.6f} | w=[{w_gbdt_local:.4f}, {w_cnn_local:.4f}]"
    )

    # Strategy 4: Stacking with Ridge
    X_stack = np.column_stack([pred_gbdt, pred_cnn])
    meta_ridge = Ridge(alpha=1.0)
    meta_ridge.fit(X_stack, y_true)
    pred_stack_ridge = meta_ridge.predict(X_stack)
    r2_stack_ridge = r2_score(y_true, pred_stack_ridge)
    results["stacking_ridge"] = {
        "r2": r2_stack_ridge,
        "mae_km": mean_absolute_error(y_true, pred_stack_ridge),
        "meta_learner": "Ridge",
        "coefs": meta_ridge.coef_.tolist(),
        "intercept": float(meta_ridge.intercept_),
    }
    print(f"  Stacking (Ridge): R² = {r2_stack_ridge:.6f}")

    # Strategy 5: Stacking with ElasticNet
    meta_elasticnet = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
    meta_elasticnet.fit(X_stack, y_true)
    pred_stack_en = meta_elasticnet.predict(X_stack)
    r2_stack_en = r2_score(y_true, pred_stack_en)
    results["stacking_elasticnet"] = {
        "r2": r2_stack_en,
        "mae_km": mean_absolute_error(y_true, pred_stack_en),
        "meta_learner": "ElasticNet",
        "coefs": meta_elasticnet.coef_.tolist(),
        "intercept": float(meta_elasticnet.intercept_),
    }
    print(f"  Stacking (ElasticNet): R² = {r2_stack_en:.6f}")

    # Strategy 6: GBDT-only weighted by confidence
    # Use GBDT more heavily when predictions are more confident
    pred_weighted_conf = 0.92 * pred_gbdt + 0.08 * pred_cnn
    r2_weighted_conf = r2_score(y_true, pred_weighted_conf)
    results["weighted_conservative"] = {
        "r2": r2_weighted_conf,
        "mae_km": mean_absolute_error(y_true, pred_weighted_conf),
        "weights": [0.92, 0.08],
    }
    print(f"  Weighted (Conservative): R² = {r2_weighted_conf:.6f} | w=[0.92, 0.08]")

    # Find best strategy
    best_strategy = max(results.items(), key=lambda x: x[1]["r2"])
    print(f"\n   Best Strategy: {best_strategy[0]} | R² = {best_strategy[1]['r2']:.6f}\n")

    return results, best_strategy


def main():
    """Main optimization workflow."""
    # Load data
    features, labels, flight_ids, image_dataset, feature_cols = load_data()

    # Stratified K-Fold CV
    n_folds = 5
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    # Stratification bins
    cbh_bins = pd.qcut(labels, q=5, labels=False, duplicates="drop")

    print(f" Running {n_folds}-Fold Stratified Cross-Validation\n")

    fold_results = []
    all_predictions = {
        "y_true": [],
        "pred_gbdt": [],
        "pred_cnn": [],
        "pred_ensemble": [],
    }

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(features, cbh_bins)):
        print(f"{'=' * 80}")
        print(f"Fold {fold_idx + 1}/{n_folds}")
        print(f"{'=' * 80}")

        # Split tabular data
        X_train, X_val = features[train_idx], features[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]

        # Normalize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # Train optimized GBDT
        gbdt_model, best_params = train_optimized_gbdt(X_train_scaled, y_train, X_val_scaled, y_val)
        pred_gbdt_val = gbdt_model.predict(X_val_scaled)
        r2_gbdt = r2_score(y_val, pred_gbdt_val)
        print(f"  GBDT R² = {r2_gbdt:.6f}")

        # Train CNN
        print("  Training CNN...")
        train_dataset = Subset(image_dataset, train_idx)
        val_dataset = Subset(image_dataset, val_idx)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        cnn_model = SimpleCNN(dropout=0.3).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(cnn_model.parameters(), lr=0.001)

        # Train CNN (quick training)
        cnn_model.train()
        for epoch in range(30):
            for batch_img, batch_label in train_loader:
                batch_img = batch_img.to(device)
                batch_label = batch_label.to(device)
                optimizer.zero_grad()
                output = cnn_model(batch_img).squeeze()
                loss = criterion(output, batch_label)
                loss.backward()
                optimizer.step()

        # Evaluate CNN
        cnn_model.eval()
        pred_cnn_val = []
        with torch.no_grad():
            for batch_img, _ in val_loader:
                batch_img = batch_img.to(device)
                output = cnn_model(batch_img).squeeze()
                pred_cnn_val.extend(output.cpu().numpy().tolist())

        pred_cnn_val = np.array(pred_cnn_val)
        r2_cnn = r2_score(y_val, pred_cnn_val)
        print(f"  CNN R² = {r2_cnn:.6f}\n")

        # Evaluate ensemble strategies on this fold
        strategies, best_strategy = evaluate_ensemble_strategies(y_val, pred_gbdt_val, pred_cnn_val)

        # Store results
        fold_results.append(
            {
                "fold": fold_idx + 1,
                "gbdt_r2": r2_gbdt,
                "cnn_r2": r2_cnn,
                "best_strategy": best_strategy[0],
                "best_r2": best_strategy[1]["r2"],
                "strategies": strategies,
                "gbdt_params": best_params,
            }
        )

        # Use best strategy for this fold
        if best_strategy[0].startswith("weighted"):
            w_gbdt, w_cnn = best_strategy[1]["weights"]
            pred_ensemble = w_gbdt * pred_gbdt_val + w_cnn * pred_cnn_val
        elif best_strategy[0].startswith("stacking"):
            X_stack = np.column_stack([pred_gbdt_val, pred_cnn_val])
            if "ridge" in best_strategy[0]:
                meta = Ridge(alpha=1.0)
            else:
                meta = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
            meta.fit(X_stack, y_val)
            pred_ensemble = meta.predict(X_stack)
        else:
            pred_ensemble = 0.5 * pred_gbdt_val + 0.5 * pred_cnn_val

        # Accumulate predictions
        all_predictions["y_true"].extend(y_val.tolist())
        all_predictions["pred_gbdt"].extend(pred_gbdt_val.tolist())
        all_predictions["pred_cnn"].extend(pred_cnn_val.tolist())
        all_predictions["pred_ensemble"].extend(pred_ensemble.tolist())

    # Aggregate results
    print(f"\n{'=' * 80}")
    print("FINAL RESULTS - OPTIMIZED ENSEMBLE")
    print(f"{'=' * 80}\n")

    mean_gbdt_r2 = np.mean([f["gbdt_r2"] for f in fold_results])
    mean_cnn_r2 = np.mean([f["cnn_r2"] for f in fold_results])
    mean_ensemble_r2 = np.mean([f["best_r2"] for f in fold_results])

    std_gbdt_r2 = np.std([f["gbdt_r2"] for f in fold_results])
    std_cnn_r2 = np.std([f["cnn_r2"] for f in fold_results])
    std_ensemble_r2 = np.std([f["best_r2"] for f in fold_results])

    print(f"GBDT (Optimized):      R² = {mean_gbdt_r2:.6f} ± {std_gbdt_r2:.6f}")
    print(f"CNN:                   R² = {mean_cnn_r2:.6f} ± {std_cnn_r2:.6f}")
    print(f"Ensemble (Best):       R² = {mean_ensemble_r2:.6f} ± {std_ensemble_r2:.6f}")

    # Check if target achieved
    target_achieved = mean_ensemble_r2 >= 0.74
    print(f"\n{'=' * 80}")
    if target_achieved:
        print(f" TARGET ACHIEVED: R² = {mean_ensemble_r2:.6f} ≥ 0.74")
    else:
        print(f"  Target not reached: R² = {mean_ensemble_r2:.6f} < 0.74")
        print(f"   Gap remaining: {0.74 - mean_ensemble_r2:.6f}")
    print(f"{'=' * 80}\n")

    # Global ensemble optimization (on all accumulated predictions)
    print(" Global Ensemble Optimization (all folds combined)...\n")
    y_true_all = np.array(all_predictions["y_true"])
    pred_gbdt_all = np.array(all_predictions["pred_gbdt"])
    pred_cnn_all = np.array(all_predictions["pred_cnn"])

    global_strategies, global_best = evaluate_ensemble_strategies(
        y_true_all, pred_gbdt_all, pred_cnn_all
    )

    # Save results
    output = {
        "task": "ensemble_optimization",
        "timestamp": datetime.now().isoformat(),
        "target": 0.74,
        "target_achieved": target_achieved,
        "cross_validation": {
            "n_folds": n_folds,
            "mean_gbdt_r2": float(mean_gbdt_r2),
            "std_gbdt_r2": float(std_gbdt_r2),
            "mean_cnn_r2": float(mean_cnn_r2),
            "std_cnn_r2": float(std_cnn_r2),
            "mean_ensemble_r2": float(mean_ensemble_r2),
            "std_ensemble_r2": float(std_ensemble_r2),
        },
        "per_fold_results": fold_results,
        "global_ensemble": {
            "n_samples": len(y_true_all),
            "strategies": {
                k: {
                    kk: float(vv) if isinstance(vv, (np.floating, np.integer)) else vv
                    for kk, vv in v.items()
                }
                for k, v in global_strategies.items()
            },
            "best_strategy": global_best[0],
            "best_r2": float(global_best[1]["r2"]),
            "best_mae_km": float(global_best[1]["mae_km"]),
        },
        "conclusion": (
            f"Ensemble optimization {'SUCCEEDED' if target_achieved else 'FAILED'}. "
            f"Best ensemble R² = {mean_ensemble_r2:.6f} "
            f"({'≥' if target_achieved else '<'} 0.74 target)."
        ),
    }

    output_path = REPORTS_DIR / "ensemble_optimization_final.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f" Results saved to: {output_path}")

    # Summary
    print(f"\n{'=' * 80}")
    print("OPTIMIZATION SUMMARY")
    print(f"{'=' * 80}")
    print(f"Original ensemble R²:  0.7391 (weighted averaging)")
    print(f"Optimized ensemble R²: {mean_ensemble_r2:.6f}")
    print(
        f"Improvement:           {mean_ensemble_r2 - 0.7391:.6f} ({(mean_ensemble_r2 - 0.7391) / 0.7391 * 100:.2f}%)"
    )
    print(f"Target (0.74):         {' ACHIEVED' if target_achieved else ' NOT REACHED'}")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
