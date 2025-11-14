#!/usr/bin/env python3
"""
Phase 2: MAE Self-Supervised Pre-Training Script

Trains a Masked Autoencoder (MAE) on all extracted images from Phase 1.
The encoder learns robust visual representations of cloud structure without labels.

Usage:
    python scripts/pretrain_mae.py --config configs/ssl_pretrain_mae.yaml

    OR with command-line overrides:

    python scripts/pretrain_mae.py \
        --config configs/ssl_pretrain_mae.yaml \
        --epochs 100 \
        --batch-size 128 \
        --learning-rate 1e-3
"""

import argparse
import os
import sys
from pathlib import Path
import time
from datetime import datetime

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ssl_dataset import SSLCloudDataset
from src.mae_model import MaskedAutoencoder, build_mae_small, build_mae_tiny


class MAETrainer:
    """Trainer for MAE self-supervised pre-training."""

    def __init__(self, config, args):
        """
        Initialize MAE trainer.

        Args:
            config: Configuration dictionary from YAML
            args: Command-line arguments (override config)
        """
        self.config = config
        self.args = args

        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\n{'=' * 80}")
        print("MAE SELF-SUPERVISED PRE-TRAINING - PHASE 2")
        print("=" * 80)
        print(f"Device: {self.device}")

        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(
                f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
            )

        # Setup paths
        self.setup_paths()

        # Load datasets
        self.load_datasets()

        # Build model
        self.build_model()

        # Setup optimizer and scheduler
        self.setup_optimizer()

        # Setup logging
        self.setup_logging()

        # Training state
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0

    def setup_paths(self):
        """Setup output directories."""
        self.output_dir = Path(self.config.get("output_dir", "outputs/mae_pretrain"))
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.logs_dir = self.output_dir / "logs"
        self.plots_dir = self.output_dir / "plots"

        for dir_path in [
            self.output_dir,
            self.checkpoint_dir,
            self.logs_dir,
            self.plots_dir,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)

        print(f"\nOutput directory: {self.output_dir}")

    def load_datasets(self):
        """Load training and validation datasets."""
        print("\nLoading datasets...")

        train_path = Path(self.config["data_dir"]) / "train.h5"
        val_path = Path(self.config["data_dir"]) / "val.h5"

        # Training dataset with augmentation
        self.train_dataset = SSLCloudDataset(
            train_path,
            augment=self.config.get("augment", True),
            normalize=True,
            return_metadata=False,
        )

        # Validation dataset without augmentation
        self.val_dataset = SSLCloudDataset(
            val_path,
            augment=False,
            normalize=True,
            return_metadata=False,
        )

        # Create dataloaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=self.config.get("num_workers", 4),
            pin_memory=True,
            drop_last=True,
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=self.config.get("num_workers", 4),
            pin_memory=True,
        )

        print(f"  Training samples: {len(self.train_dataset):,}")
        print(f"  Validation samples: {len(self.val_dataset):,}")
        print(f"  Batch size: {self.config['batch_size']}")
        print(f"  Training batches per epoch: {len(self.train_loader)}")

    def build_model(self):
        """Build MAE model."""
        print("\nBuilding model...")

        model_size = self.config.get("model_size", "small")

        if model_size == "tiny":
            self.model = build_mae_tiny()
        elif model_size == "small":
            self.model = build_mae_small()
        else:
            # Custom configuration
            self.model = MaskedAutoencoder(
                img_width=self.config.get("img_width", 440),
                patch_size=self.config.get("patch_size", 16),
                embed_dim=self.config.get("embed_dim", 192),
                depth=self.config.get("depth", 4),
                num_heads=self.config.get("num_heads", 3),
                decoder_embed_dim=self.config.get("decoder_embed_dim", 96),
                decoder_depth=self.config.get("decoder_depth", 2),
                decoder_num_heads=self.config.get("decoder_num_heads", 3),
                mask_ratio=self.config.get("mask_ratio", 0.75),
            )

        self.model = self.model.to(self.device)

        # Count parameters
        n_params = sum(p.numel() for p in self.model.parameters())
        n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print(f"  Model size: {model_size}")
        print(f"  Total parameters: {n_params:,}")
        print(f"  Trainable parameters: {n_trainable:,}")
        print(f"  Mask ratio: {self.model.mask_ratio:.1%}")

    def setup_optimizer(self):
        """Setup optimizer and learning rate scheduler."""
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config.get("weight_decay", 0.05),
            betas=(0.9, 0.95),
        )

        # Learning rate scheduler
        scheduler_type = self.config.get("scheduler", "cosine")

        if scheduler_type == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config["epochs"],
                eta_min=self.config.get("min_lr", 1e-6),
            )
        elif scheduler_type == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.get("scheduler_step", 30),
                gamma=self.config.get("scheduler_gamma", 0.1),
            )
        else:
            self.scheduler = None

        print(
            f"\nOptimizer: AdamW (lr={self.config['learning_rate']:.2e}, wd={self.config.get('weight_decay', 0.05)})"
        )
        print(f"Scheduler: {scheduler_type}")

    def setup_logging(self):
        """Setup TensorBoard logging."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = self.logs_dir / timestamp

        self.writer = SummaryWriter(log_dir)
        print(f"\nTensorBoard logs: {log_dir}")
        print(f"  View with: tensorboard --logdir {self.logs_dir}")

    def compute_loss(self, pred, mask, target):
        """
        Compute MAE reconstruction loss on masked patches only.

        Args:
            pred: (B, N) predicted pixel values
            mask: (B, N) binary mask (1 = masked, 0 = visible)
            target: (B, N) target pixel values

        Returns:
            loss: scalar loss value
        """
        # MSE loss on masked patches only
        loss = (pred - target) ** 2
        loss = (loss * mask).sum() / mask.sum()  # Mean over masked patches

        return loss

    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()

        epoch_loss = 0.0
        epoch_start = time.time()

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch + 1}/{self.config['epochs']}",
            leave=False,
        )

        for batch_idx, images in enumerate(pbar):
            images = images.to(self.device)

            # Forward pass
            pred, mask, target = self.model(images)

            # Compute loss
            loss = self.compute_loss(pred, mask, target)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.config.get("grad_clip", 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config["grad_clip"],
                )

            self.optimizer.step()

            # Update metrics
            epoch_loss += loss.item()
            self.global_step += 1

            # Update progress bar
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            # Log to TensorBoard
            if batch_idx % self.config.get("log_interval", 10) == 0:
                self.writer.add_scalar(
                    "train/batch_loss", loss.item(), self.global_step
                )

        # Compute epoch metrics
        avg_loss = epoch_loss / len(self.train_loader)
        epoch_time = time.time() - epoch_start

        # Log epoch metrics
        self.writer.add_scalar("train/epoch_loss", avg_loss, epoch)
        self.writer.add_scalar("train/epoch_time", epoch_time, epoch)
        self.writer.add_scalar(
            "train/learning_rate", self.optimizer.param_groups[0]["lr"], epoch
        )

        return avg_loss, epoch_time

    @torch.no_grad()
    def validate(self, epoch):
        """Validate model."""
        self.model.eval()

        val_loss = 0.0

        for images in tqdm(self.val_loader, desc="Validating", leave=False):
            images = images.to(self.device)

            # Forward pass
            pred, mask, target = self.model(images)

            # Compute loss
            loss = self.compute_loss(pred, mask, target)
            val_loss += loss.item()

        # Compute average
        avg_val_loss = val_loss / len(self.val_loader)

        # Log to TensorBoard
        self.writer.add_scalar("val/loss", avg_val_loss, epoch)

        return avg_val_loss

    @torch.no_grad()
    def visualize_reconstructions(self, epoch):
        """Visualize MAE reconstructions."""
        self.model.eval()

        # Get a batch from validation set
        images = next(iter(self.val_loader)).to(self.device)
        images = images[:8]  # Take first 8 images

        # Forward pass
        pred, mask, target = self.model(images)

        # Convert to numpy
        images_np = images.cpu().numpy()
        pred_np = pred.cpu().numpy()
        mask_np = mask.cpu().numpy()
        target_np = target.cpu().numpy()

        # Create figure
        fig, axes = plt.subplots(4, 8, figsize=(20, 10))

        for i in range(8):
            # Original image
            axes[0, i].plot(images_np[i, 0, :], linewidth=0.5)
            axes[0, i].set_title(f"Original {i}")
            axes[0, i].set_ylim(0, 1)

            # Target (patchified)
            axes[1, i].plot(target_np[i, :], linewidth=0.5, alpha=0.7)
            axes[1, i].set_title("Target")
            axes[1, i].set_ylim(0, 1)

            # Mask
            axes[2, i].plot(mask_np[i, :], linewidth=0.5)
            axes[2, i].set_title(f"Mask ({mask_np[i].mean():.1%})")
            axes[2, i].set_ylim(-0.1, 1.1)

            # Prediction
            axes[3, i].plot(pred_np[i, :], linewidth=0.5, alpha=0.7)
            axes[3, i].set_title("Reconstruction")
            axes[3, i].set_ylim(0, 1)

        plt.tight_layout()

        # Save figure
        save_path = self.plots_dir / f"reconstruction_epoch_{epoch + 1:03d}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        # Log to TensorBoard
        self.writer.add_figure("reconstructions", fig, epoch)

    def save_checkpoint(self, epoch, val_loss, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "config": self.config,
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        # Save latest checkpoint
        checkpoint_path = self.checkpoint_dir / "latest.pth"
        torch.save(checkpoint, checkpoint_path)

        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "best.pth"
            torch.save(checkpoint, best_path)
            print(f"     Saved best checkpoint (val_loss: {val_loss:.6f})")

        # Save periodic checkpoints
        if (epoch + 1) % self.config.get("save_interval", 20) == 0:
            periodic_path = self.checkpoint_dir / f"epoch_{epoch + 1:03d}.pth"
            torch.save(checkpoint, periodic_path)

    def save_encoder(self):
        """Save encoder weights for fine-tuning."""
        encoder_path = self.output_dir / "mae_encoder_pretrained.pth"
        torch.save(self.model.encoder.state_dict(), encoder_path)
        print(f"\n Saved encoder weights: {encoder_path}")

    def train(self):
        """Main training loop."""
        print(f"\n{'=' * 80}")
        print("STARTING TRAINING")
        print("=" * 80)
        print(f"Epochs: {self.config['epochs']}")
        print(
            f"Early stopping patience: {self.config.get('early_stopping_patience', 20)}"
        )
        print()

        for epoch in range(self.config["epochs"]):
            # Train
            train_loss, epoch_time = self.train_epoch(epoch)

            # Validate
            val_loss = self.validate(epoch)

            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()

            # Print epoch summary
            print(
                f"Epoch {epoch + 1:3d}/{self.config['epochs']} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"LR: {self.optimizer.param_groups[0]['lr']:.2e} | "
                f"Time: {epoch_time:.1f}s"
            )

            # Visualize reconstructions
            if (epoch + 1) % self.config.get("vis_interval", 5) == 0:
                self.visualize_reconstructions(epoch)

            # Check for improvement
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1

            # Save checkpoint
            self.save_checkpoint(epoch, val_loss, is_best)

            # Early stopping
            patience = self.config.get("early_stopping_patience", 20)
            if self.epochs_without_improvement >= patience:
                print(
                    f"\n  Early stopping triggered after {patience} epochs without improvement"
                )
                break

        # Training complete
        print(f"\n{'=' * 80}")
        print("TRAINING COMPLETE")
        print("=" * 80)
        print(f"Best validation loss: {self.best_val_loss:.6f}")

        # Save final encoder
        self.save_encoder()

        # Close TensorBoard writer
        self.writer.close()


def main():
    parser = argparse.ArgumentParser(
        description="MAE Self-Supervised Pre-Training - Phase 2"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/ssl_pretrain_mae.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (overrides config)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (overrides config)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate (overrides config)",
    )
    parser.add_argument(
        "--model-size",
        type=str,
        choices=["tiny", "small"],
        default=None,
        help="Model size (overrides config)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )

    args = parser.parse_args()

    # Load configuration
    if not os.path.exists(args.config):
        print(f" Error: Config file not found: {args.config}")
        sys.exit(1)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Override config with command-line arguments
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        config["learning_rate"] = args.learning_rate
    if args.model_size is not None:
        config["model_size"] = args.model_size

    # Create trainer
    trainer = MAETrainer(config, args)

    # Train
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n\n  Training interrupted by user")
        trainer.save_encoder()
        sys.exit(0)
    except Exception as e:
        print(f"\n\n Error during training: {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
