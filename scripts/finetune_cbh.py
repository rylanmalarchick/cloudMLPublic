#!/usr/bin/env python3
"""
Phase 3: Fine-tune Pre-trained MAE Encoder for CBH Regression

This script implements two-stage fine-tuning:
  Stage 1: Freeze encoder, train regression head only
  Stage 2: Unfreeze encoder, fine-tune end-to-end

Target: Beat classical baseline (GradientBoosting R² = 0.7464)
"""

import argparse
import os
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, Subset, Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler
from src.split_utils import (
    stratified_split_by_flight,
    analyze_split_balance,
    check_split_leakage,
)

# Add project root to path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.hdf5_dataset import HDF5CloudDataset
from src.mae_model import MAEEncoder


# ================================================================================
# DATASET WRAPPER
# ================================================================================


class DictDatasetWrapper(Dataset):
    """
    Wrapper that converts HDF5CloudDataset tuple output to dictionary format.

    HDF5CloudDataset returns: (img, sza, saa, y, global_idx, local_idx)
    This wrapper converts to: {"img": img, "sza": sza, "saa": saa, "y": y}
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, sza, saa, y, _, _ = self.dataset[idx]

        # img from HDF5CloudDataset is (temporal_frames, height, width)
        # With temporal_frames=1, it's (1, height, width)
        # MAE encoder expects (1, width) where width is flattened spatial

        if img.dim() == 3:
            # Take first (and only) frame if temporal_frames=1
            img = img[0]  # (height, width)
            # Flatten spatial dimension to create 1D signal
            img = img.flatten().unsqueeze(0)  # (1, height*width) = (1, 440)
        elif img.dim() == 2:
            # Already (height, width), just flatten
            img = img.flatten().unsqueeze(0)  # (1, 440)

        return {
            "img": img,
            "sza": sza.squeeze(),  # Remove extra dimension
            "saa": saa.squeeze(),  # Remove extra dimension
            "y": y.squeeze() if y.dim() > 0 else y,
        }


# ================================================================================
# MODEL DEFINITION
# ================================================================================


class CBHRegressionHead(nn.Module):
    """
    Regression head for Cloud Base Height prediction.

    Takes encoder embeddings + optional angle features and predicts CBH.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [256, 128],
        dropout: float = 0.3,
        activation: str = "gelu",
        use_batch_norm: bool = True,
        num_angles: int = 0,  # Number of angle features (0, 1, or 2)
    ):
        super().__init__()

        self.num_angles = num_angles
        total_input_dim = input_dim + num_angles

        # Choose activation function
        if activation == "relu":
            act_fn = nn.ReLU
        elif activation == "gelu":
            act_fn = nn.GELU
        elif activation == "silu":
            act_fn = nn.SiLU
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Build MLP
        layers = []
        prev_dim = total_input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(act_fn())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Final output layer (single scalar: CBH in km)
        layers.append(nn.Linear(prev_dim, 1))

        self.head = nn.Sequential(*layers)

    def forward(self, embeddings, angles=None):
        """
        Args:
            embeddings: (B, embed_dim) - encoder output
            angles: (B, num_angles) - optional angle features

        Returns:
            (B, 1) - predicted CBH in km
        """
        if self.num_angles > 0 and angles is not None:
            x = torch.cat([embeddings, angles], dim=1)
        else:
            x = embeddings

        return self.head(x)


class CBHModel(nn.Module):
    """
    Full model: Pre-trained MAE Encoder + Regression Head
    """

    def __init__(
        self,
        encoder: MAEEncoder,
        head_config: dict,
        num_angles: int = 0,
        embed_dim: int = 192,
    ):
        super().__init__()

        self.encoder = encoder
        self.num_angles = num_angles

        # Create regression head
        self.head = CBHRegressionHead(
            input_dim=embed_dim,
            hidden_dims=head_config.get("hidden_dims", [256, 128]),
            dropout=head_config.get("dropout", 0.3),
            activation=head_config.get("activation", "gelu"),
            use_batch_norm=head_config.get("use_batch_norm", True),
            num_angles=num_angles,
        )

    def forward(self, x, angles=None):
        """
        Args:
            x: (B, C, W) - input images
            angles: (B, num_angles) - optional angle features

        Returns:
            (B, 1) - predicted CBH in km
        """
        # Get encoder embeddings
        # Encoder returns (B, n_patches+1, embed_dim) with CLS token at position 0
        encoder_output = self.encoder(x)  # (B, n_patches+1, embed_dim)

        # Extract CLS token for downstream task
        embeddings = encoder_output[:, 0, :]  # (B, embed_dim)

        # Predict CBH
        cbh = self.head(embeddings, angles)

        return cbh

    def freeze_encoder(self):
        """Freeze encoder weights for Stage 1"""
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        """Unfreeze encoder weights for Stage 2"""
        for param in self.encoder.parameters():
            param.requires_grad = True


# ================================================================================
# TRAINING & EVALUATION
# ================================================================================


class CBHTrainer:
    """
    Trainer for two-stage fine-tuning
    """

    def __init__(self, config: dict, device: torch.device):
        self.config = config
        self.device = device

        # Set random seeds
        seed = config.get("random_seed", 42)
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        # Create output directories
        self.output_dir = Path(config["output"]["base_dir"])
        self.checkpoint_dir = (
            self.output_dir / config["output"]["subdirs"]["checkpoints"]
        )
        self.log_dir = self.output_dir / config["output"]["subdirs"]["logs"]
        self.plot_dir = self.output_dir / config["output"]["subdirs"]["plots"]
        self.pred_dir = self.output_dir / config["output"]["subdirs"]["predictions"]

        for d in [self.checkpoint_dir, self.log_dir, self.plot_dir, self.pred_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Setup logging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.writer = SummaryWriter(self.log_dir / timestamp)

        print("\n" + "=" * 80)
        print("CBH FINE-TUNING - PHASE 3")
        print("=" * 80)
        print(f"Device: {device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(
                f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
            )
        print(f"\nOutput directory: {self.output_dir}")
        print(f"TensorBoard logs: {self.log_dir / timestamp}")

    def load_data(self):
        """Load and split labeled dataset"""
        print("\nLoading labeled dataset...")

        # Load full dataset
        flight_configs = self.config["data"]["flights"]

        # Load base dataset WITHOUT augmentation
        # (Augmentation adds complexity; we have pre-trained features already)
        dataset = HDF5CloudDataset(
            flight_configs=flight_configs,
            swath_slice=self.config["data"]["swath_slice"],
            augment=False,  # No augmentation for fine-tuning
            temporal_frames=self.config["data"]["temporal_frames"],
            filter_type=self.config["data"]["filter_type"],
            cbh_min=self.config["data"]["cbh_min"],
            cbh_max=self.config["data"]["cbh_max"],
            flat_field_correction=self.config["data"]["flat_field_correction"],
            clahe_clip_limit=self.config["data"]["clahe_clip_limit"],
            zscore_normalize=self.config["data"]["zscore_normalize"],
            angles_mode=self.config["data"]["angles_mode"],
        )

        total_samples = len(dataset)
        print(f"Total CPL-aligned samples: {total_samples}")

        # Split into train/val/test using stratified splitting
        train_ratio = self.config["data"]["train_ratio"]
        val_ratio = self.config["data"]["val_ratio"]

        print("\nCreating stratified train/val/test splits...")
        train_indices, val_indices, test_indices = stratified_split_by_flight(
            dataset, train_ratio=train_ratio, val_ratio=val_ratio, seed=42, verbose=True
        )

        # Verify no leakage
        check_split_leakage(train_indices, val_indices, test_indices)

        # Create subsets for train/val/test
        train_subset = Subset(dataset, train_indices)
        val_subset = Subset(dataset, val_indices)
        test_subset = Subset(dataset, test_indices)

        # Wrap subsets to convert tuple output to dict
        train_dataset = DictDatasetWrapper(train_subset)
        val_dataset = DictDatasetWrapper(val_subset)
        test_dataset = DictDatasetWrapper(test_subset)

        print(f"  Training samples: {len(train_dataset)}")
        print(f"  Validation samples: {len(val_dataset)}")
        print(f"  Test samples: {len(test_dataset)}")

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config["training"]["stage1"]["batch_size"],
            shuffle=True,
            num_workers=self.config["data"]["num_workers"],
            pin_memory=self.config["data"]["pin_memory"],
            persistent_workers=self.config["data"]["persistent_workers"],
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config["training"]["stage1"]["batch_size"],
            shuffle=False,
            num_workers=self.config["data"]["num_workers"],
            pin_memory=self.config["data"]["pin_memory"],
            persistent_workers=self.config["data"]["persistent_workers"],
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config["training"]["stage1"]["batch_size"],
            shuffle=False,
            num_workers=self.config["data"]["num_workers"],
            pin_memory=self.config["data"]["pin_memory"],
            persistent_workers=self.config["data"]["persistent_workers"],
        )

        # Store dataset reference for y_scaler
        self.dataset = dataset
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

    def build_model(self):
        """Build model with pre-trained encoder"""
        print("\nBuilding model...")

        model_config = self.config["model"]

        # Load pre-trained encoder
        print(f"Loading pre-trained encoder from: {model_config['encoder_weights']}")

        # Create encoder with same config as pre-training
        encoder = MAEEncoder(
            img_width=model_config["img_width"],
            patch_size=model_config["patch_size"],
            embed_dim=model_config["embed_dim"],
            depth=model_config["depth"],
            num_heads=model_config["num_heads"],
            mlp_ratio=model_config["mlp_ratio"],
        )

        # Load weights
        checkpoint = torch.load(
            model_config["encoder_weights"],
            map_location=self.device,
            weights_only=False,
        )
        encoder.load_state_dict(checkpoint)
        print(" Pre-trained weights loaded successfully")

        # Determine number of angle features
        angles_mode = self.config["data"]["angles_mode"]
        if angles_mode == "both":
            num_angles = 2
        elif angles_mode in ["sza_only", "saa_only"]:
            num_angles = 1
        else:
            num_angles = 0

        print(f"Angle features: {num_angles} ({angles_mode})")

        # Create full model
        self.model = CBHModel(
            encoder=encoder,
            head_config=model_config["head"],
            num_angles=num_angles,
            embed_dim=model_config["embed_dim"],
        ).to(self.device)

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        encoder_params = sum(p.numel() for p in self.model.encoder.parameters())
        head_params = sum(p.numel() for p in self.model.head.parameters())

        print(f"  Total parameters: {total_params:,}")
        print(f"  Encoder parameters: {encoder_params:,}")
        print(f"  Head parameters: {head_params:,}")

    def train_stage(self, stage_config: dict, stage_name: str, start_epoch: int = 0):
        """
        Train a single stage (Stage 1 or Stage 2)
        """
        print("\n" + "=" * 80)
        print(f"STAGE: {stage_config['name']}")
        print(f"Description: {stage_config['description']}")
        print("=" * 80)

        # Setup mixed precision training if enabled
        use_amp = self.config["training"].get("use_amp", False)
        scaler = GradScaler("cuda") if use_amp else None
        if use_amp:
            print(f" Mixed precision training enabled (AMP)")

        # Freeze/unfreeze encoder
        if stage_config["freeze_encoder"]:
            self.model.freeze_encoder()
            trainable_params = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )
            print(f" Encoder frozen - Trainable parameters: {trainable_params:,}")
        else:
            self.model.unfreeze_encoder()
            trainable_params = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )
            print(f" Encoder unfrozen - Trainable parameters: {trainable_params:,}")

        # Setup optimizer
        opt_config = stage_config["optimizer"]
        if opt_config["name"].lower() == "adamw":
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=opt_config["lr"],
                weight_decay=opt_config["weight_decay"],
                betas=opt_config["betas"],
            )
        else:
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=opt_config["lr"],
                weight_decay=opt_config["weight_decay"],
            )

        # Setup scheduler
        sched_config = stage_config["scheduler"]
        warmup_epochs = sched_config.get("warmup_epochs", 0)
        total_epochs = stage_config["epochs"]

        if sched_config["name"] == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=total_epochs - warmup_epochs,
                eta_min=sched_config["min_lr"],
            )
        else:
            scheduler = None

        # Setup loss function
        loss_name = self.config["training"]["loss"]
        if loss_name == "mse":
            criterion = nn.MSELoss()
        elif loss_name == "mae":
            criterion = nn.L1Loss()
        elif loss_name == "huber":
            criterion = nn.HuberLoss()
        elif loss_name == "smooth_l1":
            criterion = nn.SmoothL1Loss()
        else:
            criterion = nn.MSELoss()

        print(
            f"\nOptimizer: {opt_config['name']} (lr={opt_config['lr']:.2e}, wd={opt_config['weight_decay']})"
        )
        print(f"Scheduler: {sched_config['name']}")
        print(f"Loss: {loss_name}")
        print(f"Epochs: {total_epochs}")
        print(f"Batch size: {stage_config['batch_size']}")
        if use_amp:
            print(f"Mixed precision: Enabled")

        # Early stopping
        early_stop_config = stage_config["early_stopping"]
        best_val_loss = float("inf")
        patience_counter = 0

        # Training loop
        print("\n" + "=" * 80)
        print("TRAINING")
        print("=" * 80)

        for epoch in range(total_epochs):
            global_epoch = start_epoch + epoch

            # Training
            train_loss, train_metrics = self.train_epoch(
                optimizer, criterion, epoch, global_epoch, stage_name, scaler
            )

            # Validation
            val_loss, val_metrics = self.validate_epoch(
                criterion, epoch, global_epoch, stage_name
            )

            # Learning rate step
            current_lr = optimizer.param_groups[0]["lr"]
            if scheduler and epoch >= warmup_epochs:
                scheduler.step()

            # Log to tensorboard
            self.writer.add_scalar(f"{stage_name}/train_loss", train_loss, global_epoch)
            self.writer.add_scalar(f"{stage_name}/val_loss", val_loss, global_epoch)
            self.writer.add_scalar(f"{stage_name}/lr", current_lr, global_epoch)

            for metric_name, value in train_metrics.items():
                self.writer.add_scalar(
                    f"{stage_name}/train_{metric_name}", value, global_epoch
                )
            for metric_name, value in val_metrics.items():
                self.writer.add_scalar(
                    f"{stage_name}/val_{metric_name}", value, global_epoch
                )

            # Print progress
            print(
                f"Epoch {epoch + 1:3d}/{total_epochs} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Val R²: {val_metrics['r2']:.4f} | "
                f"LR: {current_lr:.2e}"
            )

            # Save best checkpoint
            if val_loss < best_val_loss - early_stop_config["min_delta"]:
                best_val_loss = val_loss
                patience_counter = 0

                checkpoint_path = self.checkpoint_dir / f"{stage_name}_best.pth"
                torch.save(
                    {
                        "epoch": global_epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": val_loss,
                        "val_metrics": val_metrics,
                    },
                    checkpoint_path,
                )
                print(
                    f"     Saved best checkpoint (val_loss: {val_loss:.6f}, R²: {val_metrics['r2']:.4f})"
                )
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= early_stop_config["patience"]:
                print(
                    f"\n Early stopping triggered after {patience_counter} epochs without improvement"
                )
                break

        # Save final checkpoint
        checkpoint_path = self.checkpoint_dir / f"{stage_name}_final.pth"
        torch.save(
            {
                "epoch": global_epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_metrics": val_metrics,
            },
            checkpoint_path,
        )

        print(f"\n {stage_name} complete!")
        print(f"Best validation loss: {best_val_loss:.6f}")

        return global_epoch + 1  # Return next epoch number

    def train_epoch(
        self, optimizer, criterion, epoch, global_epoch, stage_name, scaler=None
    ):
        """Train for one epoch"""
        self.model.train()

        use_amp = scaler is not None

        total_loss = 0
        all_preds = []
        all_targets = []

        for batch_idx, batch in enumerate(self.train_loader):
            # Unpack batch
            img = batch["img"].to(self.device)  # (B, C, W)
            target = batch["y"].to(self.device)  # (B,)

            # Get angles if needed
            angles = None
            if self.model.num_angles > 0:
                angles_mode = self.config["data"]["angles_mode"]
                if angles_mode == "both":
                    angles = torch.stack([batch["sza"], batch["saa"]], dim=1).to(
                        self.device
                    )
                elif angles_mode == "sza_only":
                    angles = batch["sza"].unsqueeze(1).to(self.device)
                elif angles_mode == "saa_only":
                    angles = batch["saa"].unsqueeze(1).to(self.device)

            # Forward pass
            optimizer.zero_grad()

            if use_amp:
                # Mixed precision forward pass
                with autocast("cuda"):
                    pred = self.model(img, angles).squeeze(-1)  # (B,)
                    loss = criterion(pred, target)

                # Backward pass with gradient scaling
                scaler.scale(loss).backward()

                # Gradient clipping (with scaler)
                if self.config["training"].get("grad_clip"):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config["training"]["grad_clip"]
                    )

                # Optimizer step with scaler
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard precision
                pred = self.model(img, angles).squeeze(-1)  # (B,)
                loss = criterion(pred, target)

                # Backward pass
                loss.backward()

                # Gradient clipping
                if self.config["training"].get("grad_clip"):
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config["training"]["grad_clip"]
                    )

                optimizer.step()

            # Track metrics
            total_loss += loss.item()
            all_preds.append(pred.detach().cpu().numpy())
            all_targets.append(target.detach().cpu().numpy())

        # Compute metrics
        avg_loss = total_loss / len(self.train_loader)
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)

        metrics = self.compute_metrics(all_targets, all_preds)

        return avg_loss, metrics

    def validate_epoch(self, criterion, epoch, global_epoch, stage_name):
        """Validate for one epoch"""
        self.model.eval()

        total_loss = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in self.val_loader:
                # Unpack batch
                img = batch["img"].to(self.device)
                target = batch["y"].to(self.device)

                # Get angles if needed
                angles = None
                if self.model.num_angles > 0:
                    angles_mode = self.config["data"]["angles_mode"]
                    if angles_mode == "both":
                        angles = torch.stack([batch["sza"], batch["saa"]], dim=1).to(
                            self.device
                        )
                    elif angles_mode == "sza_only":
                        angles = batch["sza"].unsqueeze(1).to(self.device)
                    elif angles_mode == "saa_only":
                        angles = batch["saa"].unsqueeze(1).to(self.device)

                # Forward pass
                pred = self.model(img, angles).squeeze(-1)
                loss = criterion(pred, target)

                # Track metrics
                total_loss += loss.item()
                all_preds.append(pred.cpu().numpy())
                all_targets.append(target.cpu().numpy())

        # Compute metrics
        avg_loss = total_loss / len(self.val_loader)
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)

        metrics = self.compute_metrics(all_targets, all_preds)

        return avg_loss, metrics

    def compute_metrics(self, y_true, y_pred):
        """Compute evaluation metrics"""
        # Inverse transform if using scaler
        if hasattr(self.dataset, "y_scaler") and self.dataset.y_scaler is not None:
            y_true_km = self.dataset.y_scaler.inverse_transform(
                y_true.reshape(-1, 1)
            ).flatten()
            y_pred_km = self.dataset.y_scaler.inverse_transform(
                y_pred.reshape(-1, 1)
            ).flatten()
        else:
            y_true_km = y_true
            y_pred_km = y_pred

        metrics = {
            "r2": r2_score(y_true_km, y_pred_km),
            "mae": mean_absolute_error(y_true_km, y_pred_km),
            "rmse": np.sqrt(mean_squared_error(y_true_km, y_pred_km)),
            "mse": mean_squared_error(y_true_km, y_pred_km),
        }

        return metrics

    def evaluate_test_set(self):
        """Final evaluation on test set"""
        print("\n" + "=" * 80)
        print("FINAL TEST SET EVALUATION")
        print("=" * 80)

        self.model.eval()

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in self.test_loader:
                img = batch["img"].to(self.device)
                target = batch["y"].to(self.device)

                # Get angles if needed
                angles = None
                if self.model.num_angles > 0:
                    angles_mode = self.config["data"]["angles_mode"]
                    if angles_mode == "both":
                        angles = torch.stack([batch["sza"], batch["saa"]], dim=1).to(
                            self.device
                        )
                    elif angles_mode == "sza_only":
                        angles = batch["sza"].unsqueeze(1).to(self.device)
                    elif angles_mode == "saa_only":
                        angles = batch["saa"].unsqueeze(1).to(self.device)

                pred = self.model(img, angles).squeeze(-1)

                all_preds.append(pred.cpu().numpy())
                all_targets.append(target.cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)

        # Compute metrics
        metrics = self.compute_metrics(all_targets, all_preds)

        # Get baseline metrics
        baseline = self.config["evaluation"]["baseline"]

        print(f"\nTest Set Results:")
        print(f"  R²:   {metrics['r2']:.4f}  (baseline: {baseline['r2']:.4f})")
        print(f"  MAE:  {metrics['mae']:.4f} km  (baseline: {baseline['mae']:.4f} km)")
        print(
            f"  RMSE: {metrics['rmse']:.4f} km  (baseline: {baseline['rmse']:.4f} km)"
        )

        # Compare to baseline
        print("\nComparison to Classical Baseline ({}):")
        print(f"  R² improvement: {metrics['r2'] - baseline['r2']:+.4f}")
        print(f"  MAE improvement: {metrics['mae'] - baseline['mae']:+.4f} km")
        print(f"  RMSE improvement: {metrics['rmse'] - baseline['rmse']:+.4f} km")

        # Success threshold
        thresholds = self.config["evaluation"]["thresholds"]
        if metrics["r2"] >= thresholds["excellent"]:
            print(f"\n EXCELLENT performance! (R² >= {thresholds['excellent']})")
        elif metrics["r2"] >= thresholds["good"]:
            print(f"\n GOOD performance! (R² >= {thresholds['good']})")
        elif metrics["r2"] >= thresholds["acceptable"]:
            print(f"\n ACCEPTABLE performance (R² >= {thresholds['acceptable']})")
        else:
            print(f"\n Below acceptable threshold (R² < {thresholds['acceptable']})")

        # Save results
        self.plot_results(all_targets, all_preds, metrics)

        return metrics

    def plot_results(self, y_true, y_pred, metrics):
        """Generate evaluation plots"""
        # Inverse transform
        if hasattr(self.dataset, "y_scaler") and self.dataset.y_scaler is not None:
            y_true_km = self.dataset.y_scaler.inverse_transform(
                y_true.reshape(-1, 1)
            ).flatten()
            y_pred_km = self.dataset.y_scaler.inverse_transform(
                y_pred.reshape(-1, 1)
            ).flatten()
        else:
            y_true_km = y_true
            y_pred_km = y_pred

        # 1. Scatter plot: Predicted vs True
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Scatter
        ax = axes[0]
        ax.scatter(y_true_km, y_pred_km, alpha=0.5, s=20)
        ax.plot(
            [y_true_km.min(), y_true_km.max()],
            [y_true_km.min(), y_true_km.max()],
            "r--",
            lw=2,
            label="Perfect prediction",
        )
        ax.set_xlabel("True CBH (km)", fontsize=12)
        ax.set_ylabel("Predicted CBH (km)", fontsize=12)
        ax.set_title(f"Predicted vs True CBH\nR² = {metrics['r2']:.4f}", fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Residuals
        ax = axes[1]
        residuals = y_pred_km - y_true_km
        ax.hist(residuals, bins=50, alpha=0.7, edgecolor="black")
        ax.axvline(0, color="r", linestyle="--", lw=2)
        ax.set_xlabel("Residual (km)", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.set_title(
            f"Residual Distribution\nMAE = {metrics['mae']:.4f} km", fontsize=14
        )
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        plt.savefig(self.plot_dir / "test_results.png", dpi=150, bbox_inches="tight")
        plt.close()

        print(f"\n Plots saved to: {self.plot_dir}")

    def run(self):
        """Run full two-stage fine-tuning"""
        # Load data
        self.load_data()

        # Build model
        self.build_model()

        # Stage 1: Train head only
        next_epoch = self.train_stage(
            self.config["training"]["stage1"], "stage1_freeze", start_epoch=0
        )

        # Load best Stage 1 checkpoint
        checkpoint_path = self.checkpoint_dir / "stage1_freeze_best.pth"
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        print(f"\n Loaded best Stage 1 checkpoint from epoch {checkpoint['epoch']}")

        # Stage 2: Fine-tune all
        next_epoch = self.train_stage(
            self.config["training"]["stage2"], "stage2_finetune", start_epoch=next_epoch
        )

        # Load best Stage 2 checkpoint
        checkpoint_path = self.checkpoint_dir / "stage2_finetune_best.pth"
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        print(f"\n Loaded best Stage 2 checkpoint from epoch {checkpoint['epoch']}")

        # Final evaluation
        test_metrics = self.evaluate_test_set()

        # Save final model
        final_model_path = self.checkpoint_dir / "final_model.pth"
        torch.save(self.model.state_dict(), final_model_path)
        print(f"\n Final model saved to: {final_model_path}")

        print("\n" + "=" * 80)
        print("PHASE 3 COMPLETE!")
        print("=" * 80)
        print(f"\nFinal Test R²: {test_metrics['r2']:.4f}")
        print(f"View training curves with:")
        print(f"  tensorboard --logdir {self.log_dir}")

        self.writer.close()


# ================================================================================
# MAIN
# ================================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Phase 3: Fine-tune MAE encoder for CBH"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/ssl_finetune_cbh.yaml",
        help="Path to config file",
    )
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Setup device
    device = torch.device(
        config["hardware"]["device"] if torch.cuda.is_available() else "cpu"
    )

    # Run training
    trainer = CBHTrainer(config, device)
    trainer.run()


if __name__ == "__main__":
    main()
