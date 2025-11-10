#!/usr/bin/env python3
"""
WP5 Utilities: Common functions and classes for Sprint 5

This module provides shared utilities for all Sprint 5 work packages:
- Stratified K-Fold CV setup
- Data loading helpers
- Metric computation
- Report generation (unified JSON schema)
- Training loops with early stopping

Author: Sprint 5 Execution Agent
Date: 2025-11-10
"""

import sys
from pathlib import Path
import numpy as np
import h5py
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    from sklearn.model_selection import StratifiedKFold
except ImportError as e:
    print(f"Missing required package: {e}")
    sys.exit(1)


@dataclass
class FoldMetrics:
    """Metrics for a single K-Fold CV fold."""

    fold_id: int
    n_train: int
    n_val: int
    n_test: int
    r2: float
    mae_km: float
    rmse_km: float
    best_epoch: int
    train_loss: float
    val_loss: float
    predictions: np.ndarray = None
    targets: np.ndarray = None


class ImageOnlyDataset(Dataset):
    """
    Dataset that loads only images (no physical features).
    Used for WP-1 pre-trained backbone experiments.
    """

    def __init__(
        self,
        image_dataset,
        indices: List[int],
        cbh_values: np.ndarray = None,
        transform=None,
    ):
        """
        Args:
            image_dataset: HDF5CloudDataset instance
            indices: List of global indices to use
            cbh_values: Cached CBH values (km)
            transform: Optional transform for images
        """
        self.image_dataset = image_dataset
        self.indices = indices
        self.transform = transform

        # Cache CBH values to avoid repeated HDF5 reads
        if cbh_values is None:
            self.cbh_values = image_dataset.get_unscaled_y()
        else:
            self.cbh_values = cbh_values

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Get global index
        global_idx = self.indices[idx]

        # Get image from HDF5 dataset
        # Returns: (image, sza, saa, y, global_idx, local_idx)
        image, _, _, _, _, _ = self.image_dataset[global_idx]

        # Get unscaled CBH (in km)
        cbh_km = self.cbh_values[global_idx]

        # Convert image to tensor if needed
        if isinstance(image, torch.Tensor):
            image_tensor = image.float()
        else:
            image_tensor = torch.from_numpy(image).float()

        # Ensure image is 3D: (C, H, W)
        if image_tensor.dim() == 2:
            # (H, W) -> (1, H, W)
            image_tensor = image_tensor.unsqueeze(0)
        elif image_tensor.dim() == 3:
            # Already (C, H, W)
            pass
        else:
            raise ValueError(f"Unexpected image dimension: {image_tensor.shape}")

        # Apply transform if provided
        if self.transform is not None:
            image_tensor = self.transform(image_tensor)

        return image_tensor, torch.tensor(cbh_km).float()


class FusionDataset(Dataset):
    """
    Dataset that loads images AND physical features (ERA5, geometric).
    Used for WP-3 fusion experiments.
    """

    def __init__(
        self,
        image_dataset,
        era5_features: np.ndarray,
        geometric_features: np.ndarray,
        indices: List[int],
        cbh_values: np.ndarray = None,
        transform=None,
    ):
        """
        Args:
            image_dataset: HDF5CloudDataset instance
            era5_features: (N, n_era5_feat) ERA5 feature array
            geometric_features: (N, n_geo_feat) geometric feature array
            indices: List of global indices to use
            cbh_values: Cached CBH values (km)
            transform: Optional transform for images
        """
        self.image_dataset = image_dataset
        self.era5_features = era5_features
        self.geometric_features = geometric_features
        self.indices = indices
        self.transform = transform

        # Cache CBH values
        if cbh_values is None:
            self.cbh_values = image_dataset.get_unscaled_y()
        else:
            self.cbh_values = cbh_values

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Get global index
        global_idx = self.indices[idx]

        # Get image
        image, _, _, _, _, _ = self.image_dataset[global_idx]

        # Get CBH
        cbh_km = self.cbh_values[global_idx]

        # Get physical features
        era5_feat = self.era5_features[global_idx]
        geo_feat = self.geometric_features[global_idx]
        physical_feat = np.concatenate([era5_feat, geo_feat])

        # Convert to tensor
        if isinstance(image, torch.Tensor):
            image_tensor = image.float()
        else:
            image_tensor = torch.from_numpy(image).float()

        # Ensure 3D
        if image_tensor.dim() == 2:
            image_tensor = image_tensor.unsqueeze(0)
        elif image_tensor.dim() == 3:
            pass
        else:
            raise ValueError(f"Unexpected image dimension: {image_tensor.shape}")

        # Apply transform
        if self.transform is not None:
            image_tensor = self.transform(image_tensor)

        return (
            image_tensor,
            torch.from_numpy(physical_feat).float(),
            torch.tensor(cbh_km).float(),
        )


class EarlyStopper:
    """Early stopping helper."""

    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

        return self.early_stop


def compute_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """
    Compute R2, MAE, RMSE metrics.

    Args:
        predictions: (N,) array of predictions
        targets: (N,) array of targets

    Returns:
        Dictionary with r2, mae_km, rmse_km
    """
    predictions = predictions.flatten()
    targets = targets.flatten()

    r2 = r2_score(targets, predictions)
    mae = mean_absolute_error(targets, predictions)
    rmse = np.sqrt(mean_squared_error(targets, predictions))

    return {
        "r2": float(r2),
        "mae_km": float(mae),
        "rmse_km": float(rmse),
    }


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
    fusion_mode: bool = False,
    accumulation_steps: int = 1,
) -> float:
    """
    Train for one epoch.

    Args:
        model: Model to train
        train_loader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on
        fusion_mode: If True, expects (image, physical, target) batches
        accumulation_steps: Gradient accumulation steps (for VRAM saving)

    Returns:
        Average training loss
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    optimizer.zero_grad()

    for batch_idx, batch in enumerate(train_loader):
        if fusion_mode:
            images, physical, targets = batch
            images = images.to(device)
            physical = physical.to(device)
            targets = targets.to(device).unsqueeze(1)

            # Forward pass
            predictions = model(images, physical)
        else:
            images, targets = batch
            images = images.to(device)
            targets = targets.to(device).unsqueeze(1)

            # Forward pass
            predictions = model(images)

        # Compute loss
        loss = criterion(predictions, targets)

        # Normalize loss for gradient accumulation
        loss = loss / accumulation_steps

        # Backward pass
        loss.backward()

        # Update weights every accumulation_steps
        if (batch_idx + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps
        n_batches += 1

    # Final update if needed
    if n_batches % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

    return total_loss / n_batches


def validate_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: str,
    fusion_mode: bool = False,
) -> Tuple[float, Dict[str, float], np.ndarray, np.ndarray]:
    """
    Validate for one epoch.

    Args:
        model: Model to validate
        val_loader: Validation data loader
        criterion: Loss function
        device: Device
        fusion_mode: If True, expects (image, physical, target) batches

    Returns:
        Tuple of (avg_loss, metrics_dict, predictions, targets)
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in val_loader:
            if fusion_mode:
                images, physical, targets = batch
                images = images.to(device)
                physical = physical.to(device)
                targets = targets.to(device).unsqueeze(1)

                predictions = model(images, physical)
            else:
                images, targets = batch
                images = images.to(device)
                targets = targets.to(device).unsqueeze(1)

                predictions = model(images)

            loss = criterion(predictions, targets)
            total_loss += loss.item()
            n_batches += 1

            all_preds.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    avg_loss = total_loss / n_batches
    preds = np.concatenate(all_preds, axis=0).flatten()
    targets = np.concatenate(all_targets, axis=0).flatten()
    metrics = compute_metrics(preds, targets)

    return avg_loss, metrics, preds, targets


def get_stratified_folds(
    cbh_values: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create stratified K-fold splits based on CBH values.

    Args:
        cbh_values: (N,) array of CBH values in km
        n_splits: Number of folds
        random_state: Random seed

    Returns:
        List of (train_val_indices, test_indices) tuples
    """
    # Bin CBH values for stratification
    # Use quartiles to create 4 bins
    cbh_bins = np.digitize(cbh_values, bins=[0.3, 0.6, 0.9, 1.2])

    # Create stratified K-fold splitter
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Get all indices
    all_indices = np.arange(len(cbh_values))

    # Generate folds
    folds = list(skf.split(all_indices, cbh_bins))

    return folds


def generate_wp5_report(
    model_name: str,
    sprint_work_package: str,
    script_path: str,
    fold_results: List[FoldMetrics],
    hardware_config: Dict[str, str],
    additional_info: Dict = None,
) -> Dict:
    """
    Generate WP5 report following the mandated JSON schema.

    Args:
        model_name: e.g., "ResNet-50 Baseline"
        sprint_work_package: e.g., "WP-1: Pre-Trained Backbones"
        script_path: Path to the script
        fold_results: List of FoldMetrics
        hardware_config: Dict with gpu, vram, vram_mitigation_used
        additional_info: Optional additional fields

    Returns:
        Report dictionary following mandated schema
    """
    # Extract metrics from folds
    r2_values = [f.r2 for f in fold_results]
    mae_values = [f.mae_km for f in fold_results]
    rmse_values = [f.rmse_km for f in fold_results]

    # Build per-fold results
    per_fold = []
    for f in fold_results:
        fold_dict = {
            "fold_id": f.fold_id,
            "n_train": f.n_train,
            "n_val": f.n_val,
            "n_test": f.n_test,
            "r2": f.r2,
            "mae_km": f.mae_km,
            "rmse_km": f.rmse_km,
            "best_epoch": f.best_epoch,
            "train_loss": f.train_loss,
            "val_loss": f.val_loss,
        }
        per_fold.append(fold_dict)

    # Build report
    report = {
        "model_name": model_name,
        "sprint_work_package": sprint_work_package,
        "script_path": script_path,
        "baseline_to_beat": {
            "model": "Physical GBDT (Real ERA5)",
            "r2": 0.668,
            "mae_km": 0.137,
        },
        "validation_protocol": {
            "name": "Stratified 5-Fold Cross-Validation",
            "n_folds": len(fold_results),
            "stratify_by": "CBH Target",
            "forbidden_protocol": "Leave-One-Flight-Out (LOO) CV",
            "justification": "LOO CV failed (R2=-3.13) on F4 due to extreme domain shift.",
        },
        "aggregate_metrics": {
            "mean_r2": float(np.mean(r2_values)),
            "std_r2": float(np.std(r2_values)),
            "mean_mae_km": float(np.mean(mae_values)),
            "std_mae_km": float(np.std(mae_values)),
            "mean_rmse_km": float(np.mean(rmse_values)),
            "std_rmse_km": float(np.std(rmse_values)),
        },
        "per_fold_results": per_fold,
        "hardware_config": hardware_config,
        "timestamp": datetime.now().isoformat(),
    }

    # Add additional info if provided
    if additional_info:
        report.update(additional_info)

    return report


def save_report(report: Dict, output_path: Path):
    """Save report to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"✓ Report saved to {output_path}")


def print_fold_summary(fold_id: int, n_folds: int, metrics: FoldMetrics):
    """Print summary for a single fold."""
    print("\n" + "-" * 80)
    print(f"Fold {fold_id + 1}/{n_folds} Results:")
    print("-" * 80)
    print(f"  Training samples:   {metrics.n_train}")
    print(f"  Validation samples: {metrics.n_val}")
    print(f"  Test samples:       {metrics.n_test}")
    print(f"  Best epoch:         {metrics.best_epoch}")
    print(f"  Train loss:         {metrics.train_loss:.4f}")
    print(f"  Val loss:           {metrics.val_loss:.4f}")
    print(f"  Test R²:            {metrics.r2:.4f}")
    print(
        f"  Test MAE:           {metrics.mae_km:.4f} km ({metrics.mae_km * 1000:.1f} m)"
    )
    print(
        f"  Test RMSE:          {metrics.rmse_km:.4f} km ({metrics.rmse_km * 1000:.1f} m)"
    )
    print("-" * 80)


def print_aggregate_summary(report: Dict):
    """Print aggregate summary from report."""
    print("\n" + "=" * 80)
    print(f"AGGREGATE RESULTS: {report['model_name']}")
    print("=" * 80)

    agg = report["aggregate_metrics"]
    baseline = report["baseline_to_beat"]

    print(f"\nModel: {report['model_name']}")
    print(f"Work Package: {report['sprint_work_package']}")
    print(f"Validation: {report['validation_protocol']['name']}")

    print(f"\nAggregate Metrics ({report['validation_protocol']['n_folds']} folds):")
    print(f"  Mean R²:   {agg['mean_r2']:.4f} ± {agg['std_r2']:.4f}")
    print(
        f"  Mean MAE:  {agg['mean_mae_km']:.4f} ± {agg['std_mae_km']:.4f} km ({agg['mean_mae_km'] * 1000:.1f} ± {agg['std_mae_km'] * 1000:.1f} m)"
    )
    print(f"  Mean RMSE: {agg['mean_rmse_km']:.4f} ± {agg['std_rmse_km']:.4f} km")

    print(f"\nBaseline to Beat:")
    print(f"  Model: {baseline['model']}")
    print(f"  R²:    {baseline['r2']:.4f}")
    print(f"  MAE:   {baseline['mae_km']:.4f} km ({baseline['mae_km'] * 1000:.1f} m)")

    # Compare to baseline
    r2_diff = agg["mean_r2"] - baseline["r2"]
    mae_diff = agg["mean_mae_km"] - baseline["mae_km"]

    print(f"\nComparison to Baseline:")
    print(f"  ΔR²:   {r2_diff:+.4f} ({r2_diff / baseline['r2'] * 100:+.1f}%)")
    print(
        f"  ΔMAE:  {mae_diff:+.4f} km ({mae_diff / baseline['mae_km'] * 100:+.1f}%) = {mae_diff * 1000:+.1f} m"
    )

    if r2_diff > 0 and mae_diff < 0:
        print(f"\n✓ SUCCESS: Model outperforms baseline!")
    elif r2_diff > 0 or mae_diff < 0:
        print(f"\n⚠ PARTIAL: Model shows improvement in some metrics.")
    else:
        print(f"\n✗ FAILURE: Model does not outperform baseline.")

    print("=" * 80)
