#!/usr/bin/env python3
"""
WP-5 Task 3.1: FiLM Conditioning for Cloud Base Height Retrieval

This script implements Feature-wise Linear Modulation (FiLM) to fuse
image features from ViT-Tiny with atmospheric features from ERA5.

Key Architecture:
- Image Encoder: ViT-Tiny (pre-trained, single frame)
- Atmospheric Features: ERA5 (BLH, LCL, stability, temperature, etc.)
- Fusion: FiLM layers that modulate image features based on ERA5 state
- FiLM: Learns gamma (gain) and beta (bias) from ERA5 to scale/shift image features

Validation: Stratified 5-Fold Cross-Validation (n_splits=5)
Target: R² > 0.55 (Sprint 5 WP-3 objective)

Author: Sprint 5 Execution Agent
Date: 2025-11-10
"""

import sys
from pathlib import Path
import numpy as np
import h5py
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import warnings
from typing import Dict, List
import argparse

warnings.filterwarnings("ignore")

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.hdf5_dataset import HDF5CloudDataset
from sow_outputs.wp5.wp5_utils import (
    FoldMetrics,
    EarlyStopper,
    compute_metrics,
    get_stratified_folds,
    generate_wp5_report,
    save_report,
    print_fold_summary,
    print_aggregate_summary,
)

# Try to import transformers
try:
    from transformers import ViTForImageClassification

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print(
        "WARNING: transformers library not available. Please install: pip install transformers"
    )


class FusionDataset(Dataset):
    """
    Dataset that loads images AND ERA5 features for FiLM fusion.
    """

    def __init__(
        self,
        image_dataset,
        era5_features: np.ndarray,
        indices: List[int],
        cbh_values: np.ndarray = None,
    ):
        """
        Args:
            image_dataset: HDF5CloudDataset instance
            era5_features: (N, n_era5_feat) ERA5 feature array
            indices: List of global indices to use
            cbh_values: Cached CBH values (km)
        """
        self.image_dataset = image_dataset
        self.era5_features = era5_features
        self.indices = indices

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

        # Get ERA5 features
        era5_feat = self.era5_features[global_idx]

        # Convert to tensor
        if isinstance(image, torch.Tensor):
            image_tensor = image.float()
        else:
            image_tensor = torch.from_numpy(image).float()

        # Ensure 3D
        if image_tensor.dim() == 2:
            image_tensor = image_tensor.unsqueeze(0)

        return (
            image_tensor,
            torch.from_numpy(era5_feat).float(),
            torch.tensor(cbh_km).float(),
        )


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) layer.

    Takes conditioning features (ERA5) and generates gain (gamma) and bias (beta)
    to modulate the image features.

    Formulation: output = gamma * x + beta
    where gamma and beta are learned from the conditioning features.
    """

    def __init__(self, feature_dim: int, condition_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.condition_dim = condition_dim

        # Network to generate gamma and beta from conditioning features
        # CRITICAL FIX: Add more layers and normalization for stability
        self.film_generator = nn.Sequential(
            nn.Linear(condition_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(feature_dim, feature_dim * 2),
        )

    def forward(self, x, condition):
        """
        Args:
            x: (B, D) image features
            condition: (B, C) conditioning features (ERA5)

        Returns:
            modulated: (B, D) FiLM-modulated features
        """
        # Generate gamma and beta
        film_params = self.film_generator(condition)  # (B, 2*D)
        gamma, beta = torch.chunk(film_params, 2, dim=-1)  # Each (B, D)

        # CRITICAL FIX: Initialize gamma near 1, beta near 0 for stable training
        # Apply sigmoid to gamma to keep it in (0, 2) range, centered at 1
        gamma = torch.sigmoid(gamma) * 2.0

        # Apply FiLM: gamma * x + beta
        modulated = gamma * x + beta

        return modulated


class FiLMViTRegressor(nn.Module):
    """
    ViT-Tiny with FiLM conditioning from ERA5 features.

    Architecture:
    1. ViT extracts image features
    2. ERA5 features condition the image features via FiLM
    3. Modulated features feed into regression head
    """

    def __init__(
        self,
        pretrained_model: str = "WinKawaks/vit-tiny-patch16-224",
        era5_feature_dim: int = 13,
    ):
        super().__init__()

        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers library required. Install with: pip install transformers"
            )

        print(f"Loading ViT model: {pretrained_model}")

        # Load pre-trained ViT
        try:
            self.vit = ViTForImageClassification.from_pretrained(
                pretrained_model,
                num_labels=1,
                ignore_mismatched_sizes=True,
            )
            self.feature_dim = self.vit.config.hidden_size
            print(f"✓ ViT loaded (feature dim: {self.feature_dim})")
        except Exception as e:
            print(f"Failed to load {pretrained_model}: {e}")
            raise

        # ERA5 feature encoder
        self.era5_encoder = nn.Sequential(
            nn.Linear(era5_feature_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
        )

        # FiLM layer to modulate ViT features with ERA5
        self.film_layer = FiLMLayer(
            feature_dim=self.feature_dim,
            condition_dim=64,
        )

        # Regression head
        self.regression_head = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

        print(f"✓ FiLM-ViT initialized")
        print(f"  Image encoder: ViT-Tiny (feature dim: {self.feature_dim})")
        print(f"  ERA5 features: {era5_feature_dim} -> 64 (encoded)")
        print(f"  Fusion: FiLM conditioning")

    def forward(self, image, era5_features):
        """
        Forward pass.

        Args:
            image: (B, 1, H, W) grayscale images
            era5_features: (B, era5_dim) atmospheric features

        Returns:
            predictions: (B, 1) CBH predictions in km
        """
        # Duplicate grayscale to RGB
        if image.size(1) == 1:
            image = image.repeat(1, 3, 1, 1)

        # Resize to 224x224
        if image.size(2) != 224 or image.size(3) != 224:
            image = torch.nn.functional.interpolate(
                image, size=(224, 224), mode="bilinear", align_corners=False
            )

        # Extract ViT features (CLS token)
        vit_outputs = self.vit.vit(image)
        image_features = vit_outputs.last_hidden_state[:, 0, :]  # (B, D)

        # Encode ERA5 features
        era5_encoded = self.era5_encoder(era5_features)  # (B, 64)

        # Apply FiLM conditioning
        modulated_features = self.film_layer(image_features, era5_encoded)

        # Regression
        predictions = self.regression_head(modulated_features)

        return predictions


class FiLMTrainer:
    """Trainer for FiLM-ViT model with Stratified K-Fold CV."""

    def __init__(
        self,
        config_path: str,
        era5_features_path: str,
        output_dir: str = "sow_outputs/wp5",
        pretrained_model: str = "WinKawaks/vit-tiny-patch16-224",
        device: str = None,
        verbose: bool = True,
    ):
        self.config_path = config_path
        self.era5_features_path = era5_features_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pretrained_model = pretrained_model
        self.verbose = verbose

        # Create subdirectories
        self.model_dir = self.output_dir / "models" / "film"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir = self.output_dir / "reports"
        self.report_dir.mkdir(parents=True, exist_ok=True)

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
            if self.device == "cuda":
                print(f"  GPU: {torch.cuda.get_device_name(0)}")
                print(
                    f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
                )

    def load_data(self):
        """Load image dataset and ERA5 features."""
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
            temporal_frames=1,
        )

        # Cache CBH values
        if self.verbose:
            print("Caching CBH values...")
        self.cbh_values = self.image_dataset.get_unscaled_y()

        # Load ERA5 features
        if self.verbose:
            print(f"Loading ERA5 features from {self.era5_features_path}")
        with h5py.File(self.era5_features_path, "r") as f:
            self.era5_features = f["features"][:]
            era5_names = [
                n.decode() if isinstance(n, bytes) else n for n in f["feature_names"][:]
            ]

        # CRITICAL FIX: Normalize ERA5 features (z-score normalization)
        # This prevents gradient explosion in FiLM layers
        self.era5_mean = np.mean(self.era5_features, axis=0)
        self.era5_std = np.std(self.era5_features, axis=0)
        self.era5_std[self.era5_std == 0] = 1.0  # Avoid division by zero
        self.era5_features_normalized = (
            self.era5_features - self.era5_mean
        ) / self.era5_std

        n_samples = len(self.image_dataset)

        if self.verbose:
            print(f"\n✓ Data loaded:")
            print(f"  Total samples: {n_samples}")
            print(f"  ERA5 features: {self.era5_features.shape[1]} features")
            print(f"  ERA5 feature names: {era5_names[:5]}... (showing first 5)")
            print(
                f"  ERA5 range (raw): [{self.era5_features.min():.2f}, {self.era5_features.max():.2f}]"
            )
            print(
                f"  ERA5 range (normalized): [{self.era5_features_normalized.min():.2f}, {self.era5_features_normalized.max():.2f}]"
            )
            print(
                f"  CBH range: [{self.cbh_values.min():.3f}, {self.cbh_values.max():.3f}] km"
            )
            print(f"  CBH mean: {self.cbh_values.mean():.3f} km")

    def train_single_fold(
        self,
        fold_id: int,
        train_indices: List[int],
        val_indices: List[int],
        test_indices: List[int],
        n_epochs: int = 50,
        batch_size: int = 16,
        learning_rate: float = 3e-5,
        accumulation_steps: int = 1,
    ) -> FoldMetrics:
        """Train and evaluate a single fold."""

        print(f"\n{'=' * 80}")
        print(f"Fold {fold_id + 1}/5")
        print(f"{'=' * 80}")
        print(f"  Training:   {len(train_indices)} samples")
        print(f"  Validation: {len(val_indices)} samples")
        print(f"  Test:       {len(test_indices)} samples")

        # Create datasets (use NORMALIZED ERA5 features)
        train_dataset = FusionDataset(
            self.image_dataset,
            self.era5_features_normalized,
            train_indices,
            self.cbh_values,
        )
        val_dataset = FusionDataset(
            self.image_dataset,
            self.era5_features_normalized,
            val_indices,
            self.cbh_values,
        )
        test_dataset = FusionDataset(
            self.image_dataset,
            self.era5_features_normalized,
            test_indices,
            self.cbh_values,
        )

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        # Create model
        model = FiLMViTRegressor(
            pretrained_model=self.pretrained_model,
            era5_feature_dim=self.era5_features.shape[1],
        ).to(self.device)

        # Optimizer
        optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=1e-4,
        )

        # Loss function
        criterion = nn.MSELoss()

        # CRITICAL FIX: Add gradient clipping to prevent explosion
        max_grad_norm = 1.0

        # Early stopping
        early_stopper = EarlyStopper(patience=10, min_delta=1e-4)

        # Training loop
        best_val_loss = float("inf")
        best_epoch = 0
        best_model_state = None

        for epoch in range(n_epochs):
            # Train
            model.train()
            total_train_loss = 0.0
            n_train_batches = 0

            optimizer.zero_grad()

            for batch_idx, (images, era5, targets) in enumerate(train_loader):
                images = images.to(self.device)
                era5 = era5.to(self.device)
                targets = targets.to(self.device).unsqueeze(1)

                # Forward
                predictions = model(images, era5)

                # Loss
                loss = criterion(predictions, targets)
                loss = loss / accumulation_steps

                # Backward
                loss.backward()

                # Update
                if (batch_idx + 1) % accumulation_steps == 0:
                    # CRITICAL FIX: Clip gradients before optimizer step
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()

                total_train_loss += loss.item() * accumulation_steps
                n_train_batches += 1

            # Final update
            if n_train_batches % accumulation_steps != 0:
                optimizer.step()
                optimizer.zero_grad()

            train_loss = total_train_loss / n_train_batches

            # Validate
            model.eval()
            total_val_loss = 0.0
            n_val_batches = 0
            all_preds = []
            all_targets = []

            with torch.no_grad():
                for images, era5, targets in val_loader:
                    images = images.to(self.device)
                    era5 = era5.to(self.device)
                    targets = targets.to(self.device).unsqueeze(1)

                    predictions = model(images, era5)

                    loss = criterion(predictions, targets)
                    total_val_loss += loss.item()
                    n_val_batches += 1

                    all_preds.append(predictions.cpu().numpy())
                    all_targets.append(targets.cpu().numpy())

            val_loss = total_val_loss / n_val_batches
            preds = np.concatenate(all_preds, axis=0).flatten()
            targets_np = np.concatenate(all_targets, axis=0).flatten()
            val_metrics = compute_metrics(preds, targets_np)

            # Print progress
            if self.verbose and (epoch + 1) % 5 == 0:
                print(
                    f"  Epoch {epoch + 1:3d}/{n_epochs}: "
                    f"Train Loss = {train_loss:.4f}, "
                    f"Val Loss = {val_loss:.4f}, "
                    f"Val R² = {val_metrics['r2']:.4f}, "
                    f"Val MAE = {val_metrics['mae_km']:.4f} km"
                )

            # Save best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                best_model_state = model.state_dict().copy()

            # Early stopping
            if early_stopper(val_loss):
                if self.verbose:
                    print(f"  Early stopping at epoch {epoch + 1}")
                break

        # Load best model
        model.load_state_dict(best_model_state)

        # Test
        model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for images, era5, targets in test_loader:
                images = images.to(self.device)
                era5 = era5.to(self.device)
                targets = targets.to(self.device).unsqueeze(1)

                predictions = model(images, era5)

                all_preds.append(predictions.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        test_preds = np.concatenate(all_preds, axis=0).flatten()
        test_targets = np.concatenate(all_targets, axis=0).flatten()
        test_metrics = compute_metrics(test_preds, test_targets)

        # Save model
        model_path = self.model_dir / f"film_vit_fold{fold_id}.pth"
        torch.save(best_model_state, model_path)
        if self.verbose:
            print(f"  ✓ Model saved to {model_path}")

        # Create fold metrics
        fold_metrics = FoldMetrics(
            fold_id=fold_id,
            n_train=len(train_indices),
            n_val=len(val_indices),
            n_test=len(test_indices),
            r2=test_metrics["r2"],
            mae_km=test_metrics["mae_km"],
            rmse_km=test_metrics["rmse_km"],
            best_epoch=best_epoch,
            train_loss=train_loss,
            val_loss=best_val_loss,
            predictions=test_preds,
            targets=test_targets,
        )

        # Print summary
        if self.verbose:
            print_fold_summary(fold_id, 5, fold_metrics)

        return fold_metrics

    def run_kfold_cv(
        self,
        n_epochs: int = 50,
        batch_size: int = 16,
        learning_rate: float = 3e-5,
        accumulation_steps: int = 1,
    ) -> List[FoldMetrics]:
        """Run Stratified 5-Fold Cross-Validation."""

        print("\n" + "=" * 80)
        print("STRATIFIED 5-FOLD CROSS-VALIDATION: FiLM-ViT Fusion")
        print("=" * 80)

        # Get stratified folds
        folds = get_stratified_folds(self.cbh_values, n_splits=5, random_state=42)

        fold_results = []

        for fold_id, (train_val_idx, test_idx) in enumerate(folds):
            # Split train_val into train/val
            np.random.seed(42 + fold_id)
            np.random.shuffle(train_val_idx)
            n_train = int(0.8 * len(train_val_idx))
            train_indices = train_val_idx[:n_train].tolist()
            val_indices = train_val_idx[n_train:].tolist()
            test_indices = test_idx.tolist()

            # Train fold
            fold_metrics = self.train_single_fold(
                fold_id=fold_id,
                train_indices=train_indices,
                val_indices=val_indices,
                test_indices=test_indices,
                n_epochs=n_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                accumulation_steps=accumulation_steps,
            )

            fold_results.append(fold_metrics)

        return fold_results


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(description="WP-5 Task 3.1: FiLM Fusion Training")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/bestComboConfig.yaml",
        help="Config YAML path",
    )
    parser.add_argument(
        "--era5-features",
        type=str,
        default="sow_outputs/wp2_atmospheric/WP2_Features.hdf5",
        help="ERA5 features HDF5 path",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="sow_outputs/wp5",
        help="Output directory",
    )
    parser.add_argument(
        "--pretrained-model",
        type=str,
        default="WinKawaks/vit-tiny-patch16-224",
        help="Pretrained ViT model name",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Max epochs per fold",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--accumulation-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Verbose output",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("WP-5 TASK 3.1: FiLM Fusion for Cloud Base Height Retrieval")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Config file:         {args.config}")
    print(f"  ERA5 features:       {args.era5_features}")
    print(f"  Output directory:    {args.output_dir}")
    print(f"  Pretrained model:    {args.pretrained_model}")
    print(f"  Max epochs:          {args.epochs}")
    print(f"  Batch size:          {args.batch_size}")
    print(f"  Learning rate:       {args.learning_rate}")
    print(f"  Accumulation steps:  {args.accumulation_steps}")

    # Create trainer
    trainer = FiLMTrainer(
        config_path=args.config,
        era5_features_path=args.era5_features,
        output_dir=args.output_dir,
        pretrained_model=args.pretrained_model,
        verbose=args.verbose,
    )

    # Load data
    trainer.load_data()

    # Run K-Fold CV
    fold_results = trainer.run_kfold_cv(
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        accumulation_steps=args.accumulation_steps,
    )

    # Generate report
    hardware_config = {
        "gpu": "NVIDIA GTX 1070 Ti",
        "vram": "8 GB",
        "vram_mitigation_used": f"gradient_accumulation_steps: {args.accumulation_steps}"
        if args.accumulation_steps > 1
        else "None (batch_size sufficient)",
    }

    report = generate_wp5_report(
        model_name="FiLM-ViT Fusion (Image + ERA5)",
        sprint_work_package="WP-3: Advanced Fusion",
        script_path="sow_outputs/wp5/wp5_film_fusion.py",
        fold_results=fold_results,
        hardware_config=hardware_config,
        additional_info={
            "architecture": {
                "image_encoder": f"ViT-Tiny ({args.pretrained_model})",
                "atmospheric_encoder": "MLP (ERA5_dim -> 64 -> 64)",
                "fusion_mechanism": "FiLM (Feature-wise Linear Modulation)",
                "film_operation": "gamma * image_features + beta (learned from ERA5)",
                "regression_head": "Linear(192->256), ReLU, Dropout(0.3), Linear(256->1)",
            },
            "training": {
                "optimizer": "AdamW",
                "learning_rate": args.learning_rate,
                "weight_decay": 1e-4,
                "loss_function": "MSELoss",
                "early_stopping_patience": 10,
                "batch_size": args.batch_size,
                "accumulation_steps": args.accumulation_steps,
            },
        },
    )

    # Save report
    report_path = trainer.report_dir / "WP5_FiLM_Report.json"
    save_report(report, report_path)

    # Print summary
    print_aggregate_summary(report)

    print("\n" + "=" * 80)
    print("WP-5 TASK 3.1: COMPLETE")
    print("=" * 80)
    print(f"✓ Models saved to:  {trainer.model_dir}")
    print(f"✓ Report saved to:  {report_path}")


if __name__ == "__main__":
    main()
