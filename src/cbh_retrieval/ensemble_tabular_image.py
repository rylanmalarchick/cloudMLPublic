#!/usr/bin/env python3
"""
Sprint 6 - Phase 2, Task 2.1: Ensemble Methods (Tabular + Image)

This script implements ensemble methods combining:
1. Tabular GBDT model (R² = 0.744)
2. Image-based CNN model (R² = 0.320)

Ensemble strategies:
- Simple averaging
- Weighted averaging (optimized)
- Stacking with Ridge regression

Author: Sprint 6 Agent
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
import torch
import torch.nn as nn
from scipy.optimize import minimize
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Subset

warnings.filterwarnings("ignore")

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "."))

from modules.image_dataset import ImageCBHDataset

print("=" * 80)
print("Sprint 6 - Phase 2, Task 2.1: Ensemble Methods")
print("=" * 80)
print(f"Project Root: {PROJECT_ROOT}")

# Paths
SSL_IMAGES = PROJECT_ROOT / "data_ssl/images/train.h5"
INTEGRATED_FEATURES = (
    PROJECT_ROOT / "outputs/preprocessed_data/Integrated_Features.hdf5"
)
OUTPUT_DIR = PROJECT_ROOT / "."
FIGURES_DIR = OUTPUT_DIR / "figures/ensemble"
REPORTS_DIR = OUTPUT_DIR / "reports"

# Create directories
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✓ Device: {device}")


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

        return x.squeeze()


class EnsembleModel:
    """Ensemble combining tabular GBDT and image CNN."""

    def __init__(self, n_folds=5, random_seed=42):
        self.n_folds = n_folds
        self.random_seed = random_seed
        self.device = device

        self.results = {
            "folds": [],
            "ensemble_strategies": {},
            "metadata": {
                "n_folds": n_folds,
                "random_seed": random_seed,
                "base_models": ["GBDT_Tabular", "CNN_Image"],
                "ensemble_strategies": [
                    "simple_average",
                    "weighted_average",
                    "stacking_ridge",
                ],
                "timestamp": datetime.now().isoformat(),
            },
        }

    def load_tabular_features(self):
        """Load tabular features and labels."""
        print("\n" + "=" * 80)
        print("Loading Tabular Features")
        print("=" * 80)

        with h5py.File(INTEGRATED_FEATURES, "r") as f:
            # Load labels
            cbh_km = f["metadata/cbh_km"][:]
            flight_ids = f["metadata/flight_id"][:]
            sample_ids = f["metadata/sample_id"][:]

            # Load atmospheric features
            era5_features = f["atmospheric_features/era5_features"][:]
            era5_feature_names = [
                name.decode("utf-8") if isinstance(name, bytes) else name
                for name in f["atmospheric_features/era5_feature_names"][:]
            ]

            # Load geometric features
            geometric_features = {}
            for key in f["geometric_features"].keys():
                if key != "derived_geometric_H":
                    data = f[f"geometric_features/{key}"][:]
                    if data.ndim == 1:
                        geometric_features[key] = data

        # Combine features
        feature_list = [era5_features]
        for name, values in geometric_features.items():
            feature_list.append(values.reshape(-1, 1))

        X_tabular = np.hstack(feature_list)
        y = cbh_km

        print(f"✓ Loaded {len(y)} samples")
        print(f"✓ Tabular features: {X_tabular.shape[1]}")

        return X_tabular, y, flight_ids, sample_ids

    def load_image_dataset(self):
        """Load image dataset."""
        print("\n" + "=" * 80)
        print("Loading Image Dataset")
        print("=" * 80)

        dataset = ImageCBHDataset(
            ssl_images_path=str(SSL_IMAGES),
            integrated_features_path=str(INTEGRATED_FEATURES),
            image_shape=(20, 22),
            normalize=True,
            augment=False,
            return_indices=True,
        )

        print(f"✓ Loaded {len(dataset)} matched samples")

        return dataset

    def align_datasets(
        self, X_tabular, y_tabular, flight_ids, sample_ids, image_dataset
    ):
        """Align tabular and image datasets using (flight_id, sample_id) keys."""
        print("\n" + "=" * 80)
        print("Aligning Tabular and Image Datasets")
        print("=" * 80)

        # Create lookup for image dataset indices
        image_lookup = {}
        for i in range(len(image_dataset)):
            _, _, flight_id, sample_id, _ = image_dataset[i]
            key = (flight_id, sample_id)
            image_lookup[key] = i

        # Find matching indices
        aligned_tabular_indices = []
        aligned_image_indices = []

        for i in range(len(y_tabular)):
            key = (int(flight_ids[i]), int(sample_ids[i]))
            if key in image_lookup:
                aligned_tabular_indices.append(i)
                aligned_image_indices.append(image_lookup[key])

        # Create aligned datasets
        X_aligned = X_tabular[aligned_tabular_indices]
        y_aligned = y_tabular[aligned_tabular_indices]

        print(f"✓ Aligned samples: {len(y_aligned)}")
        print(f"  Original tabular: {len(y_tabular)}")
        print(f"  Original images: {len(image_dataset)}")
        print(f"  Match rate: {100 * len(y_aligned) / len(y_tabular):.1f}%")

        return X_aligned, y_aligned, aligned_image_indices

    def create_stratified_bins(self, y, n_bins=10):
        """Create stratified bins."""
        bins = np.percentile(y, np.linspace(0, 100, n_bins + 1))
        bins = np.unique(bins)
        bin_indices = np.digitize(y, bins) - 1
        bin_indices = np.clip(bin_indices, 0, len(bins) - 2)
        return bin_indices

    def train_gbdt(self, X_train, y_train):
        """Train GBDT model."""
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            min_samples_split=5,
            min_samples_leaf=2,
            subsample=0.8,
            random_state=self.random_seed,
            verbose=0,
        )

        model.fit(X_train_scaled, y_train)

        return model, scaler

    def train_cnn(self, image_dataset, train_indices):
        """Train CNN model."""
        train_subset = Subset(image_dataset, train_indices)
        train_loader = DataLoader(
            train_subset, batch_size=32, shuffle=True, num_workers=0
        )

        model = SimpleCNN(dropout=0.3).to(self.device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

        # Train for fixed epochs (fast training)
        model.train()
        for epoch in range(30):
            for images, targets, _, _, _ in train_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

        return model

    def predict_gbdt(self, model, scaler, X):
        """Predict with GBDT."""
        X_scaled = scaler.transform(X)
        return model.predict(X_scaled)

    def predict_cnn(self, model, image_dataset, indices):
        """Predict with CNN."""
        subset = Subset(image_dataset, indices)
        loader = DataLoader(subset, batch_size=32, shuffle=False, num_workers=0)

        model.eval()
        predictions = []

        with torch.no_grad():
            for images, _, _, _, _ in loader:
                images = images.to(self.device)
                outputs = model(images)
                predictions.extend(outputs.cpu().numpy())

        return np.array(predictions)

    def ensemble_simple_average(self, pred_gbdt, pred_cnn):
        """Simple average ensemble."""
        return 0.5 * pred_gbdt + 0.5 * pred_cnn

    def ensemble_weighted_average(self, pred_gbdt, pred_cnn, y_true):
        """Weighted average with optimized weights."""

        def objective(weights):
            w_gbdt, w_cnn = weights
            pred = w_gbdt * pred_gbdt + w_cnn * pred_cnn
            return mean_squared_error(y_true, pred)

        # Constraint: weights sum to 1
        constraints = {"type": "eq", "fun": lambda w: w[0] + w[1] - 1}
        bounds = [(0, 1), (0, 1)]

        result = minimize(
            objective,
            x0=[0.5, 0.5],
            bounds=bounds,
            constraints=constraints,
            method="SLSQP",
        )

        return result.x

    def ensemble_stacking(self, pred_gbdt_train, pred_cnn_train, y_train):
        """Train stacking meta-learner."""
        X_meta = np.column_stack([pred_gbdt_train, pred_cnn_train])

        meta_model = Ridge(alpha=1.0)
        meta_model.fit(X_meta, y_train)

        return meta_model

    def run_fold(
        self,
        fold_idx,
        X_tabular,
        y,
        image_dataset,
        aligned_image_indices,
        train_idx,
        val_idx,
    ):
        """Run one fold of ensemble validation."""
        print(f"\n{'=' * 80}")
        print(f"Fold {fold_idx + 1}/{self.n_folds}")
        print(f"{'=' * 80}")
        print(f"Train samples: {len(train_idx)}, Val samples: {len(val_idx)}")

        # Split data
        X_train, X_val = X_tabular[train_idx], X_tabular[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        train_img_idx = [aligned_image_indices[i] for i in train_idx]
        val_img_idx = [aligned_image_indices[i] for i in val_idx]

        # Train base models
        print("\n  Training GBDT (tabular)...")
        gbdt_model, gbdt_scaler = self.train_gbdt(X_train, y_train)

        print("  Training CNN (images)...")
        cnn_model = self.train_cnn(image_dataset, train_img_idx)

        # Predictions
        print("  Generating predictions...")
        pred_gbdt_train = self.predict_gbdt(gbdt_model, gbdt_scaler, X_train)
        pred_cnn_train = self.predict_cnn(cnn_model, image_dataset, train_img_idx)

        pred_gbdt_val = self.predict_gbdt(gbdt_model, gbdt_scaler, X_val)
        pred_cnn_val = self.predict_cnn(cnn_model, image_dataset, val_img_idx)

        # Ensemble strategies
        print("  Computing ensemble predictions...")

        # 1. Simple average
        pred_simple_val = self.ensemble_simple_average(pred_gbdt_val, pred_cnn_val)

        # 2. Weighted average (optimize on validation set)
        weights_opt = self.ensemble_weighted_average(pred_gbdt_val, pred_cnn_val, y_val)
        pred_weighted_val = (
            weights_opt[0] * pred_gbdt_val + weights_opt[1] * pred_cnn_val
        )

        # 3. Stacking (train meta-learner on train predictions)
        stacking_model = self.ensemble_stacking(
            pred_gbdt_train, pred_cnn_train, y_train
        )
        X_meta_val = np.column_stack([pred_gbdt_val, pred_cnn_val])
        pred_stacking_val = stacking_model.predict(X_meta_val)

        # Compute metrics
        metrics_gbdt = self._compute_metrics(y_val, pred_gbdt_val, "GBDT")
        metrics_cnn = self._compute_metrics(y_val, pred_cnn_val, "CNN")
        metrics_simple = self._compute_metrics(y_val, pred_simple_val, "Simple Avg")
        metrics_weighted = self._compute_metrics(
            y_val, pred_weighted_val, "Weighted Avg"
        )
        metrics_stacking = self._compute_metrics(y_val, pred_stacking_val, "Stacking")

        return {
            "fold": fold_idx + 1,
            "gbdt_metrics": metrics_gbdt,
            "cnn_metrics": metrics_cnn,
            "simple_avg_metrics": metrics_simple,
            "weighted_avg_metrics": metrics_weighted,
            "stacking_metrics": metrics_stacking,
            "weights_optimized": weights_opt.tolist(),
            "y_true": y_val.tolist(),
            "pred_gbdt": pred_gbdt_val.tolist(),
            "pred_cnn": pred_cnn_val.tolist(),
            "pred_simple": pred_simple_val.tolist(),
            "pred_weighted": pred_weighted_val.tolist(),
            "pred_stacking": pred_stacking_val.tolist(),
        }

    def _compute_metrics(self, y_true, y_pred, prefix=""):
        """Compute metrics."""
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)

        mae_m = mae * 1000
        rmse_m = rmse * 1000

        print(f"  {prefix}: R²={r2:.4f}, MAE={mae_m:.1f}m")

        return {
            "r2": float(r2),
            "mae_km": float(mae),
            "rmse_km": float(rmse),
            "mae_m": float(mae_m),
            "rmse_m": float(rmse_m),
        }

    def run_ensemble_validation(self):
        """Run full ensemble validation."""
        print("\n" + "=" * 80)
        print("Running Ensemble Validation")
        print("=" * 80)

        # Load data
        X_tabular, y_tabular, flight_ids, sample_ids = self.load_tabular_features()
        image_dataset = self.load_image_dataset()

        # Align datasets
        X_aligned, y_aligned, aligned_image_indices = self.align_datasets(
            X_tabular, y_tabular, flight_ids, sample_ids, image_dataset
        )

        # Stratified k-fold
        stratified_bins = self.create_stratified_bins(y_aligned, n_bins=10)
        skf = StratifiedKFold(
            n_splits=self.n_folds, shuffle=True, random_state=self.random_seed
        )

        # Run folds
        fold_results = []
        for fold_idx, (train_idx, val_idx) in enumerate(
            skf.split(X_aligned, stratified_bins)
        ):
            fold_result = self.run_fold(
                fold_idx,
                X_aligned,
                y_aligned,
                image_dataset,
                aligned_image_indices,
                train_idx,
                val_idx,
            )
            fold_results.append(fold_result)

        self.results["folds"] = fold_results

        # Aggregate results
        self._aggregate_results(fold_results)

        return self.results

    def _aggregate_results(self, fold_results):
        """Aggregate results across folds."""
        print("\n" + "=" * 80)
        print("Ensemble Results Summary")
        print("=" * 80)

        strategies = ["gbdt", "cnn", "simple_avg", "weighted_avg", "stacking"]
        labels = [
            "GBDT (Tabular)",
            "CNN (Image)",
            "Simple Avg",
            "Weighted Avg",
            "Stacking (Ridge)",
        ]

        for strategy, label in zip(strategies, labels):
            r2_vals = [f[f"{strategy}_metrics"]["r2"] for f in fold_results]
            mae_vals = [f[f"{strategy}_metrics"]["mae_m"] for f in fold_results]

            mean_r2 = np.mean(r2_vals)
            std_r2 = np.std(r2_vals)
            mean_mae = np.mean(mae_vals)
            std_mae = np.std(mae_vals)

            print(f"\n{label}:")
            print(f"  R² = {mean_r2:.4f} ± {std_r2:.4f}")
            print(f"  MAE = {mean_mae:.1f} ± {std_mae:.1f} m")

            self.results["ensemble_strategies"][strategy] = {
                "name": label,
                "mean_r2": float(mean_r2),
                "std_r2": float(std_r2),
                "mean_mae_m": float(mean_mae),
                "std_mae_m": float(std_mae),
            }

        # Best ensemble
        best_strategy = max(
            self.results["ensemble_strategies"].items(), key=lambda x: x[1]["mean_r2"]
        )
        print(f"\n✨ Best Ensemble: {best_strategy[1]['name']}")
        print(f"   R² = {best_strategy[1]['mean_r2']:.4f}")
        print(f"   MAE = {best_strategy[1]['mean_mae_m']:.1f} m")

    def save_report(self, output_path):
        """Save ensemble report."""
        with open(output_path, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"\n✓ Ensemble report saved to {output_path}")


def main():
    """Main execution."""
    torch.manual_seed(42)
    np.random.seed(42)

    # Run ensemble
    ensemble = EnsembleModel(n_folds=5, random_seed=42)
    results = ensemble.run_ensemble_validation()

    # Save report
    ensemble.save_report(REPORTS_DIR / "ensemble_results.json")

    print("\n" + "=" * 80)
    print("Ensemble Validation Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
