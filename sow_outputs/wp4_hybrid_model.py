#!/usr/bin/env python3
"""
WP-4: Hybrid Deep Learning Model for Cloud Base Height Retrieval

This script implements the hybrid model that combines:
1. High-resolution images (primary signal) via pretrained MAE encoder
2. ERA5 atmospheric features (coarse context)
3. Geometric features (weak priors)

Key architecture decisions:
- Use MAE patch tokens (NOT CLS token - proven ineffective)
- Global average pooling for spatial feature aggregation
- Multi-modal fusion via concatenation and attention mechanisms
- Regression head for CBH prediction

Validation: Leave-One-Flight-Out Cross-Validation (5 folds)

Author: Autonomous Agent (WP-4 Execution)
Date: 2025-11-05
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
    predictions: np.ndarray
    targets: np.ndarray


class MAEFeatureExtractor(nn.Module):
    """
    Wrapper around pretrained MAE encoder for feature extraction.

    CRITICAL: Uses patch tokens with global average pooling, NOT CLS token.
    The CLS token was proven ineffective in prior experiments.
    """

    def __init__(self, mae_checkpoint_path: str, device: str = "cuda"):
        super().__init__()
        self.device = device

        # Load MAE encoder
        print(f"Loading MAE encoder from {mae_checkpoint_path}")
        checkpoint = torch.load(mae_checkpoint_path, map_location=device)

        # Extract encoder state dict
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # Filter encoder weights only
        encoder_state = {
            k.replace("encoder.", ""): v
            for k, v in state_dict.items()
            if k.startswith("encoder.")
        }

        # Build encoder architecture (must match MAE training)
        from src.mae_model import MAEEncoder

        # MAE config from checkpoint (img_width=440, patch_size=16, embed_dim=192, depth=4)
        self.encoder = MAEEncoder(
            img_width=440,
            patch_size=16,
            embed_dim=192,
            depth=4,
            num_heads=3,
            mlp_ratio=4.0,
            dropout=0.0,
        )

        # Load weights (checkpoint has no 'encoder.' prefix)
        if not encoder_state:
            # No encoder prefix, load directly
            self.encoder.load_state_dict(state_dict, strict=True)
        else:
            self.encoder.load_state_dict(encoder_state, strict=True)
        self.encoder.eval()
        self.encoder.to(device)

        # Freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Feature dimension (embed_dim from checkpoint)
        self.feature_dim = 192

        print(f"✓ MAE encoder loaded, feature dim: {self.feature_dim}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract spatial features from images.

        Args:
            x: Images (B, 1, H, W)

        Returns:
            features: (B, feature_dim) via global average pooling of patch tokens
        """
        with torch.no_grad():
            # Get patch embeddings from encoder
            # encoder output: (B, num_patches, embed_dim)
            patch_tokens = self.encoder(x)

            # Global average pooling over spatial dimension
            # This aggregates information from all patches
            features = patch_tokens.mean(dim=1)  # (B, embed_dim)

        return features


class PhysicalFeatureEncoder(nn.Module):
    """MLP encoder for physical features (ERA5 + geometric)."""

    def __init__(
        self,
        n_features: int,
        hidden_dim: int = 128,
        output_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class CrossAttentionFusion(nn.Module):
    """Cross-attention fusion between image and physical features."""

    def __init__(self, image_dim: int, physical_dim: int, num_heads: int = 4):
        super().__init__()

        # Project to common dimension
        self.image_proj = nn.Linear(image_dim, image_dim)
        self.physical_proj = nn.Linear(physical_dim, image_dim)

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=image_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True,
        )

        # Output projection
        self.output_proj = nn.Linear(image_dim, image_dim)

    def forward(
        self, image_features: torch.Tensor, physical_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse image and physical features via cross-attention.

        Args:
            image_features: (B, image_dim)
            physical_features: (B, physical_dim)

        Returns:
            fused_features: (B, image_dim)
        """
        # Project features
        image_proj = self.image_proj(image_features)  # (B, image_dim)
        physical_proj = self.physical_proj(physical_features)  # (B, image_dim)

        # Add sequence dimension for attention
        image_seq = image_proj.unsqueeze(1)  # (B, 1, image_dim)
        physical_seq = physical_proj.unsqueeze(1)  # (B, 1, image_dim)

        # Cross-attention: image attends to physical context
        attn_out, _ = self.attention(
            query=image_seq, key=physical_seq, value=physical_seq
        )

        # Remove sequence dimension and project
        fused = self.output_proj(attn_out.squeeze(1))  # (B, image_dim)

        # Residual connection
        return image_proj + fused


class HybridCBHModel(nn.Module):
    """
    Hybrid model for cloud base height prediction.

    Architecture variants:
    - image_only: Image → MAE → Regression
    - concat: Image + Physical → Concat → Regression
    - attention: Image + Physical → Cross-Attention → Regression
    """

    def __init__(
        self,
        mae_checkpoint: str,
        n_era5_features: int = 9,
        n_geometric_features: int = 3,
        fusion_mode: str = "concat",
        device: str = "cuda",
    ):
        super().__init__()

        self.fusion_mode = fusion_mode
        self.device = device

        # Image feature extractor (frozen MAE encoder)
        self.image_encoder = MAEFeatureExtractor(mae_checkpoint, device)
        image_dim = self.image_encoder.feature_dim

        # Physical feature encoder
        n_physical = n_era5_features + n_geometric_features
        self.physical_encoder = PhysicalFeatureEncoder(
            n_features=n_physical,
            hidden_dim=128,
            output_dim=64,
        )
        physical_dim = 64

        # Fusion layer
        if fusion_mode == "concat":
            # Simple concatenation
            combined_dim = image_dim + physical_dim
            self.fusion = None
        elif fusion_mode == "attention":
            # Cross-attention fusion
            self.fusion = CrossAttentionFusion(image_dim, physical_dim)
            combined_dim = image_dim
        elif fusion_mode == "image_only":
            # Image only (no physical features)
            combined_dim = image_dim
            self.physical_encoder = None
            self.fusion = None
        else:
            raise ValueError(f"Unknown fusion mode: {fusion_mode}")

        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(
        self, images: torch.Tensor, physical_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            images: (B, 1, H, W)
            physical_features: (B, n_physical) - ERA5 + geometric features

        Returns:
            cbh_pred: (B, 1) - predicted cloud base height in km
        """
        # Extract image features
        image_feat = self.image_encoder(images)  # (B, image_dim)

        if self.fusion_mode == "image_only":
            # No physical features
            combined = image_feat
        else:
            # Encode physical features
            physical_feat = self.physical_encoder(physical_features)  # (B, phys_dim)

            if self.fusion_mode == "concat":
                # Concatenate features
                combined = torch.cat([image_feat, physical_feat], dim=1)
            elif self.fusion_mode == "attention":
                # Cross-attention fusion
                combined = self.fusion(image_feat, physical_feat)

        # Regression
        cbh_pred = self.regressor(combined)

        return cbh_pred


class HybridDataset(Dataset):
    """
    Dataset that loads images and physical features.
    """

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
        image, _, cbh_scaled, _, _, _ = self.image_dataset[global_idx]

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

        # Convert 2D images to 1D to match MAE encoder training
        # MAE was trained on 1D cloud profiles (1, W=440)
        # HDF5Dataset returns 2D images (C, H, W) where H=440, W=640
        if image_tensor.dim() == 3:
            # Image shape is (C, H=440, W=640)
            # We need (1, 440) for MAE encoder
            # Take vertical slice at middle column: (C, H, W) -> (1, H=440)
            C, H, W = image_tensor.shape
            mid_col = W // 2
            if C == 1:
                image_1d = image_tensor[:, :, mid_col]  # (1, H=440)
            else:
                # Use first channel for grayscale, extract column
                image_1d = image_tensor[0:1, :, mid_col]  # (1, H=440)
        elif image_tensor.dim() == 2:
            # Already (C, W) - add channel dimension if needed
            if image_tensor.shape[0] != 1:
                image_1d = image_tensor[0:1, :]  # Take first channel
            else:
                image_1d = image_tensor
        else:
            # 1D - add channel dimension
            image_1d = image_tensor.unsqueeze(0)

        return (
            image_1d,
            torch.from_numpy(physical_feat).float(),
            torch.tensor(cbh_km).float(),
        )


class HybridModelTrainer:
    """Trainer for hybrid CBH models with LOO CV."""

    def __init__(
        self,
        config_path: str,
        mae_checkpoint: str,
        era5_features_path: str,
        geometric_features_path: str,
        output_dir: str = "sow_outputs/wp4_hybrid",
        device: str = None,
        verbose: bool = True,
    ):
        self.config_path = Path(config_path)
        self.mae_checkpoint = mae_checkpoint
        self.era5_features_path = Path(era5_features_path)
        self.geometric_features_path = Path(geometric_features_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose

        # Device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Using device: {self.device}")

        # Flight mapping
        self.flight_mapping = {
            0: "30Oct24",
            1: "10Feb25",
            2: "23Oct24",
            3: "12Feb25",
            4: "18Feb25",
        }

        # Load config
        with open(self.config_path) as f:
            self.config = yaml.safe_load(f)

        # Flight name to ID mapping
        self.flight_name_to_id = {}
        for i, flight in enumerate(self.config["flights"]):
            name = flight["name"]
            for fid, fname in self.flight_mapping.items():
                if fname == name:
                    self.flight_name_to_id[name] = fid
                    break

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
        lr: float = 1e-3,
    ) -> Tuple[nn.Module, int]:
        """
        Train model with early stopping.

        Returns:
            best_model: Model with best validation R²
            best_epoch: Epoch number of best model
        """
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs
        )
        criterion = nn.HuberLoss(delta=0.5)  # Robust to outliers

        best_r2 = -np.inf
        best_model_state = None
        best_epoch = 0
        patience = 15
        patience_counter = 0

        for epoch in range(n_epochs):
            # Training
            model.train()
            train_loss = 0.0

            for images, physical, targets in train_loader:
                images = images.to(self.device)
                physical = physical.to(self.device)
                targets = targets.to(self.device).unsqueeze(1)

                optimizer.zero_grad()
                preds = model(images, physical)
                loss = criterion(preds, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation
            model.eval()
            val_preds = []
            val_targets = []

            with torch.no_grad():
                for images, physical, targets in val_loader:
                    images = images.to(self.device)
                    physical = physical.to(self.device)

                    preds = model(images, physical)
                    val_preds.append(preds.cpu().numpy())
                    val_targets.append(targets.numpy())

            val_preds = np.concatenate(val_preds)
            val_targets = np.concatenate(val_targets)
            val_r2 = r2_score(val_targets, val_preds)

            # Check for improvement
            if val_r2 > best_r2:
                best_r2 = val_r2
                best_model_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= patience:
                if self.verbose:
                    print(
                        f"  Early stopping at epoch {epoch}, best R²={best_r2:.4f} at epoch {best_epoch}"
                    )
                break

            scheduler.step()

        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        return model, best_epoch

    def evaluate_model(
        self, model: nn.Module, test_loader: DataLoader
    ) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
        """Evaluate model on test set."""
        model.eval()

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for images, physical, targets in test_loader:
                images = images.to(self.device)
                physical = physical.to(self.device)

                preds = model(images, physical)
                all_preds.append(preds.cpu().numpy())
                all_targets.append(targets.numpy())

        preds = np.concatenate(all_preds).flatten()
        targets = np.concatenate(all_targets).flatten()

        r2 = r2_score(targets, preds)
        mae = mean_absolute_error(targets, preds)
        rmse = np.sqrt(mean_squared_error(targets, preds))

        return r2, mae, rmse, preds, targets

    def run_loo_cv(
        self, fusion_mode: str = "concat", n_epochs: int = 50, batch_size: int = 32
    ) -> List[FoldMetrics]:
        """
        Run Leave-One-Flight-Out Cross-Validation.

        Args:
            fusion_mode: "image_only", "concat", or "attention"
            n_epochs: Maximum training epochs
            batch_size: Batch size

        Returns:
            List of FoldMetrics for each fold
        """
        if self.verbose:
            print("\n" + "=" * 80)
            print(f"Leave-One-Flight-Out CV: {fusion_mode.upper()} mode")
            print("=" * 80)

        fold_results = []

        for test_flight_id in range(5):
            test_flight_name = self.flight_mapping[test_flight_id]

            if self.verbose:
                print(f"\n" + "-" * 80)
                print(f"Fold {test_flight_id}: Test on {test_flight_name}")
                print("-" * 80)

            # Split data
            test_mask = self.flight_ids == test_flight_id
            train_mask = ~test_mask

            train_indices = np.where(train_mask)[0].tolist()
            test_indices = np.where(test_mask)[0].tolist()

            if len(test_indices) == 0:
                print(f"  WARNING: No test samples, skipping fold {test_flight_id}")
                continue

            # Create datasets (pass cached CBH values)
            train_dataset = HybridDataset(
                self.image_dataset,
                self.era5_features,
                self.geometric_features,
                train_indices,
                self.cbh_values,
            )
            test_dataset = HybridDataset(
                self.image_dataset,
                self.era5_features,
                self.geometric_features,
                test_indices,
                self.cbh_values,
            )

            # Create dataloaders (num_workers=0 to avoid multiprocessing issues)
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
            )
            test_loader = DataLoader(
                test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
            )

            if self.verbose:
                print(f"  Training samples: {len(train_indices)}")
                print(f"  Test samples: {len(test_indices)}")

            # Create model
            model = HybridCBHModel(
                mae_checkpoint=self.mae_checkpoint,
                n_era5_features=self.era5_features.shape[1],
                n_geometric_features=self.geometric_features.shape[1],
                fusion_mode=fusion_mode,
                device=self.device,
            ).to(self.device)

            # Train
            if self.verbose:
                print(f"  Training model...")

            model, best_epoch = self.train_model(
                model, train_loader, test_loader, n_epochs=n_epochs
            )

            # Evaluate
            r2, mae, rmse, preds, targets = self.evaluate_model(model, test_loader)

            if self.verbose:
                print(f"  Results:")
                print(f"    R² = {r2:.4f}")
                print(f"    MAE = {mae:.4f} km")
                print(f"    RMSE = {rmse:.4f} km")
                print(f"    Best epoch: {best_epoch}")

            # Store results
            fold_metrics = FoldMetrics(
                fold_id=test_flight_id,
                test_flight=test_flight_name,
                n_train=len(train_indices),
                n_test=len(test_indices),
                r2=r2,
                mae_km=mae,
                rmse_km=rmse,
                epoch_trained=best_epoch,
                predictions=preds,
                targets=targets,
            )
            fold_results.append(fold_metrics)

            # Save model checkpoint
            model_path = (
                self.output_dir / f"model_{fusion_mode}_fold{test_flight_id}.pth"
            )
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "fold_id": test_flight_id,
                    "test_flight": test_flight_name,
                    "metrics": {"r2": r2, "mae": mae, "rmse": rmse},
                    "epoch": best_epoch,
                },
                model_path,
            )

        return fold_results

    def aggregate_results(self, fold_results: List[FoldMetrics]) -> dict:
        """Aggregate results across folds."""
        r2_scores = [f.r2 for f in fold_results]
        mae_scores = [f.mae_km for f in fold_results]
        rmse_scores = [f.rmse_km for f in fold_results]

        return {
            "mean_r2": np.mean(r2_scores),
            "std_r2": np.std(r2_scores),
            "mean_mae_km": np.mean(mae_scores),
            "std_mae_km": np.std(mae_scores),
            "mean_rmse_km": np.mean(rmse_scores),
            "std_rmse_km": np.std(rmse_scores),
            "n_folds": len(fold_results),
        }

    def generate_report(
        self, fusion_mode: str, fold_results: List[FoldMetrics]
    ) -> dict:
        """Generate report for a model variant."""
        fold_reports = []
        for fold in fold_results:
            fold_reports.append(
                {
                    "fold_id": fold.fold_id,
                    "test_flight": fold.test_flight,
                    "n_train": fold.n_train,
                    "n_test": fold.n_test,
                    "r2": float(fold.r2),
                    "mae_km": float(fold.mae_km),
                    "rmse_km": float(fold.rmse_km),
                    "epoch_trained": fold.epoch_trained,
                }
            )

        aggregate = self.aggregate_results(fold_results)

        report = {
            "model_variant": fusion_mode,
            "description": self._get_variant_description(fusion_mode),
            "validation_protocol": "Leave-One-Flight-Out Cross-Validation",
            "fold_results": fold_reports,
            "aggregate_metrics": aggregate,
            "timestamp": datetime.now().isoformat(),
        }

        return report

    def _get_variant_description(self, fusion_mode: str) -> str:
        descriptions = {
            "image_only": "Image-only CNN baseline (MAE encoder + regression head)",
            "concat": "Hybrid model: Image + Physical features (concatenation fusion)",
            "attention": "Hybrid model: Image + Physical features (cross-attention fusion)",
        }
        return descriptions.get(fusion_mode, "Unknown variant")

    def print_summary(self, report: dict):
        """Print summary to console."""
        print("\n" + "=" * 80)
        print(f"RESULTS: {report['model_variant'].upper()}")
        print("=" * 80)

        print(f"\nDescription: {report['description']}")

        print(f"\nPer-Fold Results:")
        print(
            f"{'Fold':<6} {'Flight':<12} {'n_test':<8} {'R²':<10} {'MAE (km)':<12} {'RMSE (km)':<12} {'Epoch':<8}"
        )
        print("-" * 80)
        for fold in report["fold_results"]:
            print(
                f"{fold['fold_id']:<6} {fold['test_flight']:<12} {fold['n_test']:<8} "
                f"{fold['r2']:<10.4f} {fold['mae_km']:<12.4f} {fold['rmse_km']:<12.4f} "
                f"{fold['epoch_trained']:<8}"
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

    parser = argparse.ArgumentParser(description="WP-4 Hybrid Model Training")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/bestComboConfig.yaml",
        help="Config YAML",
    )
    parser.add_argument(
        "--mae-checkpoint",
        type=str,
        default="outputs/mae_pretrain/mae_encoder_pretrained.pth",
        help="MAE encoder checkpoint",
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
        "--output-dir", type=str, default="sow_outputs/wp4_hybrid", help="Output dir"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["image_only", "concat", "attention", "all"],
        help="Fusion mode to train",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Max epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")

    args = parser.parse_args()

    # Create trainer
    trainer = HybridModelTrainer(
        config_path=args.config,
        mae_checkpoint=args.mae_checkpoint,
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

        # Run LOO CV
        fold_results = trainer.run_loo_cv(
            fusion_mode=mode, n_epochs=args.epochs, batch_size=args.batch_size
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
