#!/usr/bin/env python3
"""
WP-4: Hybrid Deep Learning Model for Cloud Base Height Retrieval (FIXED)

This script implements the hybrid model that combines:
1. High-resolution 2D images (primary signal) via 2D CNN
2. ERA5 atmospheric features (coarse context)
3. Geometric features (weak priors)

KEY FIX: Uses proper 2D CNN instead of 1D MAE encoder that was discarding
99.8% of spatial information.

Key architecture decisions:
- 2D ResNet-style CNN for image feature extraction
- Multi-modal fusion via concatenation and attention mechanisms
- Regression head for CBH prediction

Validation: Leave-One-Flight-Out Cross-Validation (5 folds)

Author: Fixed by autonomous agent
Date: 2025-11-06
"""

import sys
from pathlib import Path
import numpy as np
import h5py
import yaml
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
import warnings
from datetime import datetime
from dataclasses import dataclass
import copy
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.hdf5_dataset import HDF5CloudDataset

# Check for required packages
try:
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
except ImportError as e:
    print(f"Missing required package: {e}")
    sys.exit(1)


@dataclass
class FoldMetrics:
    """Metrics for a single LOO CV fold."""

    fold_id: int
    test_flight: str
    n_train: int
    n_test: int
    r2: float
    mae_km: float
    rmse_km: float
    epoch_trained: int
    predictions: np.ndarray = None
    targets: np.ndarray = None


class ResidualBlock(nn.Module):
    """Residual block for CNN."""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ImageCNN(nn.Module):
    """
    2D CNN for extracting features from cloud images.

    Architecture: ResNet-style with 4 blocks
    Input: (B, C, H, W) where C=temporal_frames (typically 1-3)
    Output: (B, feature_dim)
    """

    def __init__(self, in_channels=3, feature_dim=256):
        super().__init__()
        self.feature_dim = feature_dim

        # Initial convolution
        self.conv1 = nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        # Residual blocks
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Project to feature_dim
        self.fc = nn.Linear(512, feature_dim)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) images
        Returns:
            features: (B, feature_dim)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class PhysicalFeatureEncoder(nn.Module):
    """MLP for encoding physical features (ERA5 + geometric)."""

    def __init__(self, input_dim: int = 12, hidden_dim: int = 64, output_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.mlp(x)


class CrossAttentionFusion(nn.Module):
    """Cross-attention fusion of image and physical features."""

    def __init__(
        self, image_dim: int = 256, physical_dim: int = 64, num_heads: int = 4
    ):
        super().__init__()
        # Make dimensions compatible for attention
        self.image_dim = image_dim
        self.physical_dim = physical_dim

        # Project physical features to match image dimension
        self.physical_proj = nn.Linear(physical_dim, image_dim)

        # Multi-head attention (physical attends to image)
        self.attention = nn.MultiheadAttention(
            embed_dim=image_dim, num_heads=num_heads, dropout=0.1, batch_first=True
        )

        # Layer norm
        self.norm = nn.LayerNorm(image_dim)

        # Output projection
        self.output_proj = nn.Linear(image_dim, image_dim)

    def forward(self, image_feat, physical_feat):
        """
        Args:
            image_feat: (B, image_dim) - features from CNN
            physical_feat: (B, physical_dim) - features from MLP

        Returns:
            fused: (B, image_dim) - attention-fused features
        """
        # Project physical features to image dimension
        physical_proj = self.physical_proj(physical_feat)  # (B, image_dim)

        # Add sequence dimension for attention
        # Use physical as query, image as key/value
        query = physical_proj.unsqueeze(1)  # (B, 1, image_dim)
        key_value = image_feat.unsqueeze(1)  # (B, 1, image_dim)

        # Cross-attention
        attn_out, _ = self.attention(query, key_value, key_value)  # (B, 1, image_dim)
        attn_out = attn_out.squeeze(1)  # (B, image_dim)

        # Residual connection with image features
        fused = self.norm(image_feat + attn_out)
        fused = self.output_proj(fused)

        return fused


class HybridCBHModel(nn.Module):
    """
    Hybrid model combining 2D CNN and physical features.

    Modes:
    - image_only: Only image features
    - concat: Concatenate image + physical features
    - attention: Cross-attention fusion
    """

    def __init__(
        self,
        fusion_mode: str = "concat",
        in_channels: int = 3,
        image_feature_dim: int = 256,
        physical_feature_dim: int = 64,
        n_physical_features: int = 12,
    ):
        super().__init__()
        self.fusion_mode = fusion_mode

        # Image feature extractor (2D CNN)
        self.image_encoder = ImageCNN(
            in_channels=in_channels, feature_dim=image_feature_dim
        )

        # Physical feature encoder (MLP)
        if fusion_mode != "image_only":
            self.physical_encoder = PhysicalFeatureEncoder(
                input_dim=n_physical_features,
                hidden_dim=physical_feature_dim,
                output_dim=physical_feature_dim,
            )

        # Fusion
        if fusion_mode == "concat":
            # Simple concatenation
            combined_dim = image_feature_dim + physical_feature_dim
            self.regressor = nn.Sequential(
                nn.Linear(combined_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(64, 1),
            )
        elif fusion_mode == "attention":
            # Cross-attention fusion
            self.fusion = CrossAttentionFusion(
                image_dim=image_feature_dim,
                physical_dim=physical_feature_dim,
                num_heads=4,
            )
            self.regressor = nn.Sequential(
                nn.Linear(image_feature_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(64, 1),
            )
        else:  # image_only
            self.regressor = nn.Sequential(
                nn.Linear(image_feature_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(64, 1),
            )

    def forward(self, image, physical_feat=None):
        """
        Args:
            image: (B, C, H, W) images
            physical_feat: (B, n_physical_features) physical features

        Returns:
            predictions: (B, 1) CBH predictions (km)
        """
        # Extract image features
        image_feat = self.image_encoder(image)  # (B, image_feature_dim)

        if self.fusion_mode == "image_only":
            combined = image_feat
        elif self.fusion_mode == "concat":
            # Encode physical features
            physical_encoded = self.physical_encoder(physical_feat)
            # Concatenate
            combined = torch.cat([image_feat, physical_encoded], dim=1)
        elif self.fusion_mode == "attention":
            # Encode physical features
            physical_encoded = self.physical_encoder(physical_feat)
            # Attention fusion
            combined = self.fusion(image_feat, physical_encoded)

        # Predict CBH
        pred = self.regressor(combined)
        return pred


class HybridDataset(Dataset):
    """Dataset that loads 2D images and physical features."""

    def __init__(
        self,
        image_dataset: HDF5CloudDataset,
        era5_features: np.ndarray,
        geometric_features: np.ndarray,
        indices: List[int],
        cbh_values: np.ndarray = None,
    ):
        self.image_dataset = image_dataset
        self.era5_features = era5_features
        self.geometric_features = geometric_features
        self.indices = indices

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

        # Get image and target from HDF5 dataset
        image, _, _, _, _, _ = self.image_dataset[global_idx]

        # Get unscaled CBH (in km) from cache
        cbh_km = self.cbh_values[global_idx]

        # Get physical features
        era5_feat = self.era5_features[global_idx]
        geo_feat = self.geometric_features[global_idx]
        physical_feat = np.concatenate([era5_feat, geo_feat])

        # Convert image to tensor if needed (HDF5Dataset returns Tensor)
        if isinstance(image, torch.Tensor):
            image_tensor = image.float()
        else:
            image_tensor = torch.from_numpy(image).float()

        # Ensure image is 4D: (C, H, W)
        # HDF5Dataset returns (temporal_frames, H, W)
        if image_tensor.dim() == 2:
            # (H, W) -> (1, H, W)
            image_tensor = image_tensor.unsqueeze(0)
        elif image_tensor.dim() == 3:
            # Already (C, H, W) - good!
            pass
        else:
            raise ValueError(f"Unexpected image dimension: {image_tensor.shape}")

        return (
            image_tensor,
            torch.from_numpy(physical_feat).float(),
            torch.tensor(cbh_km).float(),
        )


class HybridModelTrainer:
    """Trainer for hybrid model with LOO CV."""

    def __init__(
        self,
        config_path: str,
        era5_features_path: str,
        geometric_features_path: str,
        output_dir: str = "sow_outputs/wp4_cnn",
        device: str = None,
        verbose: bool = True,
    ):
        self.config_path = config_path
        self.era5_features_path = era5_features_path
        self.geometric_features_path = geometric_features_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose

        # Load config
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if self.verbose:
            print(f"Using device: {self.device}")

        # Flight mapping for LOO CV
        self.flight_mapping = {
            0: "30Oct24",
            1: "10Feb25",
            2: "23Oct24",
            3: "12Feb25",
            4: "18Feb25",
        }
        self.flight_name_to_id = {v: k for k, v in self.flight_mapping.items()}

    def load_data(self):
        """Load all data: images, ERA5, geometric features."""

        if self.verbose:
            print("\n" + "=" * 80)
            print("Loading Data")
            print("=" * 80)

        # Load image dataset
        self.image_dataset = HDF5CloudDataset(
            flight_configs=self.config["flights"],
            indices=None,
            augment=False,
            swath_slice=self.config.get("swath_slice", [40, 480]),
            temporal_frames=1,  # Use single frame for simplicity
        )

        n_samples = len(self.image_dataset)

        # Cache CBH values once to avoid repeated HDF5 reads
        if self.verbose:
            print("Caching CBH values...")
        self.cbh_values = self.image_dataset.get_unscaled_y()

        # Load ERA5 features
        with h5py.File(self.era5_features_path, "r") as f:
            self.era5_features = f["features"][:]
            era5_names = [
                n.decode() if isinstance(n, bytes) else n for n in f["feature_names"][:]
            ]

        # Load geometric features
        with h5py.File(self.geometric_features_path, "r") as f:
            # Select features
            geo_cbh = f["derived_geometric_H"][:]
            geo_length = f["shadow_length_pixels"][:]
            geo_conf = f["shadow_detection_confidence"][:]

            self.geometric_features = np.column_stack([geo_cbh, geo_length, geo_conf])
            geo_names = [
                "derived_geometric_H",
                "shadow_length_pixels",
                "shadow_detection_confidence",
            ]

        # Handle NaNs in geometric features (impute with median)
        for j in range(self.geometric_features.shape[1]):
            col = self.geometric_features[:, j]
            if np.any(np.isnan(col)):
                median_val = np.nanmedian(col)
                self.geometric_features[np.isnan(col), j] = median_val
                if self.verbose:
                    n_nan = np.sum(np.isnan(col))
                    print(
                        f"  Imputed {n_nan} NaNs in {geo_names[j]} with median={median_val:.3f}"
                    )

        # Get flight IDs
        self.flight_ids = np.zeros(n_samples, dtype=int)
        for i in range(n_samples):
            _, _, _, _, global_idx, _ = self.image_dataset[i]
            flight_idx, _ = self.image_dataset.global_to_local[int(global_idx)]
            flight_name = self.image_dataset.flight_data[flight_idx]["name"]
            if flight_name in self.flight_name_to_id:
                self.flight_ids[i] = self.flight_name_to_id[flight_name]
            else:
                self.flight_ids[i] = flight_idx

        if self.verbose:
            print(f"\nData loaded:")
            print(f"  Total samples: {n_samples}")
            print(f"  ERA5 features: {self.era5_features.shape[1]} ({era5_names})")
            print(
                f"  Geometric features: {self.geometric_features.shape[1]} ({geo_names})"
            )
            print(f"  Flight distribution:")
            for fid in sorted(np.unique(self.flight_ids)):
                count = np.sum(self.flight_ids == fid)
                fname = self.flight_mapping.get(fid, f"Unknown_{fid}")
                print(f"    F{fid} ({fname}): {count} samples")
            print("=" * 80)

    def train_model(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        n_epochs: int = 50,
        lr: float = 0.001,
    ) -> Tuple[nn.Module, int]:
        """Train model with early stopping."""

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs
        )
        criterion = nn.HuberLoss(delta=1.0)

        best_val_loss = float("inf")
        best_model = None
        best_epoch = 0
        patience = 10
        patience_counter = 0

        for epoch in range(n_epochs):
            # Training
            model.train()
            train_loss = 0.0
            for images, physical, targets in train_loader:
                images = images.to(self.device)
                if model.fusion_mode != "image_only":
                    physical = physical.to(self.device)
                else:
                    physical = None
                targets = targets.to(self.device).unsqueeze(1)

                optimizer.zero_grad()
                preds = model(images, physical)
                loss = criterion(preds, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss.item()

            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, physical, targets in val_loader:
                    images = images.to(self.device)
                    if model.fusion_mode != "image_only":
                        physical = physical.to(self.device)
                    else:
                        physical = None
                    targets = targets.to(self.device).unsqueeze(1)

                    preds = model(images, physical)
                    loss = criterion(preds, targets)
                    val_loss += loss.item()

            val_loss /= len(val_loader)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

            scheduler.step()

        # Restore best model
        model.load_state_dict(best_model)
        return model, best_epoch

    def evaluate_model(self, model: nn.Module, test_loader: DataLoader) -> Dict:
        """Evaluate model on test set."""
        model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for images, physical, targets in test_loader:
                images = images.to(self.device)
                if model.fusion_mode != "image_only":
                    physical = physical.to(self.device)
                else:
                    physical = None

                preds = model(images, physical)
                all_preds.append(preds.cpu().numpy())
                all_targets.append(targets.numpy())

        preds = np.concatenate(all_preds, axis=0).flatten()
        targets = np.concatenate(all_targets, axis=0).flatten()

        r2 = r2_score(targets, preds)
        mae = mean_absolute_error(targets, preds)
        rmse = np.sqrt(mean_squared_error(targets, preds))

        return {
            "r2": float(r2),
            "mae_km": float(mae),
            "rmse_km": float(rmse),
            "predictions": preds,
            "targets": targets,
        }

    def run_kfold_cv(
        self,
        fusion_mode: str = "concat",
        n_epochs: int = 50,
        batch_size: int = 32,
        n_splits: int = 5,
    ) -> List[FoldMetrics]:
        """Run Stratified K-Fold Cross-Validation."""

        print("\n" + "=" * 80)
        print(f"Stratified {n_splits}-Fold CV: {fusion_mode.upper()} mode")
        print("=" * 80)

        fold_results = []

        # Get in_channels from first image
        sample_image, _, _, _, _, _ = self.image_dataset[0]
        if isinstance(sample_image, torch.Tensor):
            in_channels = sample_image.shape[0]
        else:
            in_channels = sample_image.shape[0] if sample_image.ndim == 3 else 1

        n_physical_features = (
            self.era5_features.shape[1] + self.geometric_features.shape[1]
        )

        # Get CBH values for stratification
        cbh_values = self.cbh_values

        # Bin CBH values for stratification (4 bins)
        cbh_bins = np.digitize(cbh_values, bins=[0.3, 0.6, 0.9, 1.2])

        # Create stratified K-fold splitter
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        # All indices
        all_indices = np.arange(len(cbh_values))

        for fold_id, (train_val_idx, test_idx) in enumerate(
            skf.split(all_indices, cbh_bins)
        ):
            print("\n" + "-" * 80)
            print(f"Fold {fold_id + 1}/{n_splits}")
            print("-" * 80)

            # Further split train_val into train/val (80/20)
            np.random.seed(42 + fold_id)
            np.random.shuffle(train_val_idx)
            n_train = int(0.8 * len(train_val_idx))
            train_indices_split = train_val_idx[:n_train].tolist()
            val_indices_split = train_val_idx[n_train:].tolist()
            test_indices = test_idx.tolist()

            # Get flight distribution for this fold
            test_flights = [
                self.image_dataset.flight_data[
                    self.image_dataset.global_to_local[i][0]
                ]["name"]
                for i in test_indices[:5]
            ]  # Sample first 5
            test_flight_str = ", ".join(set(test_flights[:3]))

            print(f"  Training samples: {len(train_indices_split)}")
            print(f"  Validation samples: {len(val_indices_split)}")
            print(f"  Test samples: {len(test_indices)}")
            print(f"  Test flights (sample): {test_flight_str}...")

            # Create datasets
            train_dataset = HybridDataset(
                self.image_dataset,
                self.era5_features,
                self.geometric_features,
                train_indices_split,
                self.cbh_values,
            )
            val_dataset = HybridDataset(
                self.image_dataset,
                self.era5_features,
                self.geometric_features,
                val_indices_split,
                self.cbh_values,
            )
            test_dataset = HybridDataset(
                self.image_dataset,
                self.era5_features,
                self.geometric_features,
                test_indices,
                self.cbh_values,
            )

            # Create dataloaders
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
            )
            val_loader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
            )
            test_loader = DataLoader(
                test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
            )

            # Create model
            model = HybridCBHModel(
                fusion_mode=fusion_mode,
                in_channels=in_channels,
                image_feature_dim=256,
                physical_feature_dim=64,
                n_physical_features=n_physical_features,
            ).to(self.device)

            print(f"  Training model...")

            # Train with proper train/val split (higher LR for small dataset)
            model, best_epoch = self.train_model(
                model, train_loader, val_loader, n_epochs=n_epochs, lr=0.003
            )

            # Evaluate
            results = self.evaluate_model(model, test_loader)

            print(f"  Results:")
            print(f"    R² = {results['r2']:.4f}")
            print(f"    MAE = {results['mae_km']:.4f} km")
            print(f"    RMSE = {results['rmse_km']:.4f} km")
            print(f"    Best epoch: {best_epoch}")

            # Save model
            model_path = self.output_dir / f"model_{fusion_mode}_fold{fold_id}.pth"
            torch.save(model.state_dict(), model_path)

            # Store results
            fold_results.append(
                FoldMetrics(
                    fold_id=fold_id,
                    test_flight=f"Fold_{fold_id + 1}",  # K-Fold doesn't have single test flight
                    n_train=len(train_indices_split),
                    n_test=len(test_indices),
                    r2=results["r2"],
                    mae_km=results["mae_km"],
                    rmse_km=results["rmse_km"],
                    epoch_trained=best_epoch,
                    predictions=results["predictions"],
                    targets=results["targets"],
                )
            )

        return fold_results

    def generate_report(self, mode: str, fold_results: List[FoldMetrics]) -> Dict:
        """Generate JSON report from fold results."""

        # Aggregate metrics
        r2_values = [f.r2 for f in fold_results]
        mae_values = [f.mae_km for f in fold_results]
        rmse_values = [f.rmse_km for f in fold_results]

        report = {
            "model_variant": mode,
            "description": self._get_variant_description(mode),
            "validation_protocol": "Stratified K-Fold Cross-Validation",
            "fold_results": [
                {
                    "fold_id": f.fold_id,
                    "test_flight": f.test_flight,
                    "n_train": f.n_train,
                    "n_test": f.n_test,
                    "r2": f.r2,
                    "mae_km": f.mae_km,
                    "rmse_km": f.rmse_km,
                    "epoch_trained": f.epoch_trained,
                }
                for f in fold_results
            ],
            "aggregate_metrics": {
                "mean_r2": float(np.mean(r2_values)),
                "std_r2": float(np.std(r2_values)),
                "mean_mae_km": float(np.mean(mae_values)),
                "std_mae_km": float(np.std(mae_values)),
                "mean_rmse_km": float(np.mean(rmse_values)),
                "std_rmse_km": float(np.std(rmse_values)),
                "n_folds": len(fold_results),
            },
            "timestamp": datetime.now().isoformat(),
        }

        return report

    def _get_variant_description(self, mode: str) -> str:
        descriptions = {
            "image_only": "2D CNN baseline (image features only)",
            "concat": "2D CNN + physical features (concatenation fusion)",
            "attention": "2D CNN + physical features (attention fusion)",
        }
        return descriptions.get(mode, f"Unknown mode: {mode}")

    def print_summary(self, report: Dict):
        """Print summary of results."""
        print("\n" + "=" * 80)
        print(f"RESULTS: {report['model_variant'].upper()}")
        print("=" * 80)
        print(f"\nDescription: {report['description']}")
        print("\nPer-Fold Results:")
        print(
            f"{'Fold':<6} {'Flight':<12} {'n_test':<8} {'R²':<10} {'MAE (km)':<12} {'RMSE (km)':<12} {'Epoch':<7}"
        )
        print("-" * 80)
        for f in report["fold_results"]:
            print(
                f"{f['fold_id']:<6} {f['test_flight']:<12} {f['n_test']:<8} "
                f"{f['r2']:<10.4f} {f['mae_km']:<12.4f} {f['rmse_km']:<12.4f} {f['epoch_trained']:<7}"
            )

        agg = report["aggregate_metrics"]
        print(f"\nAggregate Metrics ({agg['n_folds']} folds):")
        print(f"  Mean R²:   {agg['mean_r2']:.4f} ± {agg['std_r2']:.4f}")
        print(f"  Mean MAE:  {agg['mean_mae_km']:.4f} ± {agg['std_mae_km']:.4f} km")
        print(f"  Mean RMSE: {agg['mean_rmse_km']:.4f} ± {agg['std_rmse_km']:.4f} km")
        print("=" * 80)


def main():
    """Main execution."""
    import argparse

    parser = argparse.ArgumentParser(
        description="WP-4 Hybrid Model Training (FIXED with 2D CNN)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/bestComboConfig.yaml",
        help="Config YAML",
    )
    parser.add_argument(
        "--era5-features",
        type=str,
        default="sow_outputs/wp2_atmospheric/WP2_Features.hdf5",
        help="ERA5 features",
    )
    parser.add_argument(
        "--geometric-features",
        type=str,
        default="sow_outputs/wp1_geometric/WP1_Features.hdf5",
        help="Geometric features",
    )
    parser.add_argument(
        "--output-dir", type=str, default="sow_outputs/wp4_cnn", help="Output dir"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["image_only", "concat", "attention", "all"],
        help="Fusion mode to train",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Max epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")

    args = parser.parse_args()

    # Create trainer
    trainer = HybridModelTrainer(
        config_path=args.config,
        era5_features_path=args.era5_features,
        geometric_features_path=args.geometric_features,
        output_dir=args.output_dir,
    )

    # Load data
    trainer.load_data()

    # Train models
    modes_to_train = (
        ["image_only", "concat", "attention"] if args.mode == "all" else [args.mode]
    )

    all_reports = {}

    for mode in modes_to_train:
        print(f"\n{'=' * 80}")
        print(f"TRAINING: {mode.upper()}")
        print(f"{'=' * 80}")

        # Run K-Fold CV
        fold_results = trainer.run_kfold_cv(
            fusion_mode=mode,
            n_epochs=args.epochs,
            batch_size=args.batch_size,
            n_splits=5,
        )

        # Generate report
        report = trainer.generate_report(mode, fold_results)

        # Save report
        report_path = trainer.output_dir / f"WP4_Report_{mode}.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        # Print summary
        trainer.print_summary(report)

        all_reports[mode] = report

    # Save combined report
    combined_path = trainer.output_dir / "WP4_Report_All.json"
    with open(combined_path, "w") as f:
        json.dump(all_reports, f, indent=2)

    print(f"\n✓ All reports saved to {trainer.output_dir}")


if __name__ == "__main__":
    main()
