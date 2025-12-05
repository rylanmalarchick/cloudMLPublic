#!/usr/bin/env python3
"""SimCLR pretraining script for cloud image representations.

This script trains a SimCLR encoder on unlabeled SSL images (58,846 samples)
using contrastive learning. The learned representations will be evaluated
for cross-flight generalization via linear probe.

Paper 2 Research Question:
    Can unsupervised contrastive learning improve cross-flight generalization?
    Baseline cross-flight R² = -0.98 (catastrophic failure)

Usage:
    python experiments/paper2/simclr_pretrain.py --config configs/paper2_simclr.yaml
    
    Or with defaults:
    python experiments/paper2/simclr_pretrain.py

Author: Paper 2 Implementation
Date: 2025
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.paper2.simclr_encoder import SimCLRModel, build_simclr_small
from experiments.paper2.contrastive_dataset import (
    ContrastiveSSLDataset,
    ContrastiveAugmentation,
)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class SimCLRConfig:
    """Configuration for SimCLR pretraining."""
    
    # Data paths
    ssl_data_path: str = "data_ssl/images/train.h5"
    output_dir: str = "outputs/paper2_simclr"
    
    # Model architecture
    feature_dim: int = 256
    projection_dim: int = 128
    hidden_dim: int = 256
    temperature: float = 0.5
    
    # Training
    batch_size: int = 256
    num_epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    warmup_epochs: int = 10
    
    # Augmentation
    noise_std: float = 0.1
    brightness_range: Tuple[float, float] = (0.8, 1.2)
    contrast_range: Tuple[float, float] = (0.8, 1.2)
    blur_prob: float = 0.5
    
    # System
    num_workers: int = 4
    device: str = "cuda"
    seed: int = 42
    save_every: int = 10
    log_every: int = 50
    
    @classmethod
    def from_yaml(cls, path: str) -> "SimCLRConfig":
        """Load configuration from YAML file."""
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)
        
        # Handle tuple fields
        for key in ["brightness_range", "contrast_range"]:
            if key in config_dict and isinstance(config_dict[key], list):
                config_dict[key] = tuple(config_dict[key])
        
        return cls(**config_dict)
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        return {
            "ssl_data_path": self.ssl_data_path,
            "output_dir": self.output_dir,
            "feature_dim": self.feature_dim,
            "projection_dim": self.projection_dim,
            "hidden_dim": self.hidden_dim,
            "temperature": self.temperature,
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "warmup_epochs": self.warmup_epochs,
            "noise_std": self.noise_std,
            "brightness_range": list(self.brightness_range),
            "contrast_range": list(self.contrast_range),
            "blur_prob": self.blur_prob,
            "num_workers": self.num_workers,
            "device": self.device,
            "seed": self.seed,
            "save_every": self.save_every,
            "log_every": self.log_every,
        }


# =============================================================================
# Training Utilities
# =============================================================================


def setup_logging(output_dir: Path) -> logging.Logger:
    """Configure logging to file and console."""
    logger = logging.getLogger("simclr_pretrain")
    logger.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler
    log_file = output_dir / "training.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    
    return logger


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =============================================================================
# Training Loop
# =============================================================================


class SimCLRTrainer:
    """Trainer for SimCLR pretraining."""
    
    def __init__(
        self,
        model: SimCLRModel,
        train_loader: DataLoader,
        config: SimCLRConfig,
        output_dir: Path,
        logger: logging.Logger,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.config = config
        self.output_dir = output_dir
        self.logger = logger
        
        # Device setup
        self.device = torch.device(
            config.device if torch.cuda.is_available() else "cpu"
        )
        self.model = self.model.to(self.device)
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        # Learning rate scheduler with warmup
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            total_iters=config.warmup_epochs * len(train_loader),
        )
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=(config.num_epochs - config.warmup_epochs) * len(train_loader),
            eta_min=1e-6,
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[config.warmup_epochs * len(train_loader)],
        )
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_loss = float("inf")
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "learning_rate": [],
        }
    
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        epoch_start = time.time()
        
        for batch_idx, (view1, view2) in enumerate(self.train_loader):
            view1 = view1.to(self.device)
            view2 = view2.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            loss, _, _ = self.model(view1, view2)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # Log progress
            if self.global_step % self.config.log_every == 0:
                current_lr = self.scheduler.get_last_lr()[0]
                self.logger.info(
                    f"Epoch {self.epoch + 1} | "
                    f"Step {batch_idx + 1}/{len(self.train_loader)} | "
                    f"Loss: {loss.item():.4f} | "
                    f"LR: {current_lr:.2e}"
                )
        
        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / num_batches
        
        self.logger.info(
            f"Epoch {self.epoch + 1} complete | "
            f"Avg Loss: {avg_loss:.4f} | "
            f"Time: {epoch_time:.1f}s"
        )
        
        return avg_loss
    
    def save_checkpoint(self, filename: str, is_best: bool = False) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_loss": self.best_loss,
            "config": self.config.to_dict(),
            "history": self.history,
        }
        
        path = self.output_dir / filename
        torch.save(checkpoint, path)
        self.logger.info(f"Saved checkpoint: {path}")
        
        if is_best:
            best_path = self.output_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best model: {best_path}")
    
    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_loss = checkpoint["best_loss"]
        self.history = checkpoint.get("history", self.history)
        
        self.logger.info(f"Loaded checkpoint from {path} (epoch {self.epoch})")
    
    def train(self) -> Dict[str, List[float]]:
        """Run full training loop."""
        self.logger.info(f"Starting training for {self.config.num_epochs} epochs")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Model parameters: {count_parameters(self.model):,}")
        self.logger.info(f"Training samples: {len(self.train_loader.dataset):,}")
        self.logger.info(f"Batch size: {self.config.batch_size}")
        self.logger.info(f"Batches per epoch: {len(self.train_loader)}")
        
        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            
            # Train epoch
            avg_loss = self.train_epoch()
            
            # Record history
            self.history["train_loss"].append(avg_loss)
            self.history["learning_rate"].append(self.scheduler.get_last_lr()[0])
            
            # Check for best model
            is_best = avg_loss < self.best_loss
            if is_best:
                self.best_loss = avg_loss
            
            # Save checkpoints
            if (epoch + 1) % self.config.save_every == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pt")
            
            if is_best:
                self.save_checkpoint("checkpoint_latest.pt", is_best=True)
            else:
                self.save_checkpoint("checkpoint_latest.pt")
        
        # Save final model
        self.save_checkpoint("final_model.pt")
        
        # Save training history
        history_path = self.output_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
        
        self.logger.info(f"Training complete! Best loss: {self.best_loss:.4f}")
        
        return self.history


# =============================================================================
# Main
# =============================================================================


def main(args: argparse.Namespace) -> None:
    """Main training function."""
    # Load config
    if args.config:
        config = SimCLRConfig.from_yaml(args.config)
    else:
        config = SimCLRConfig()
    
    # Override config with CLI args
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.epochs:
        config.num_epochs = args.epochs
    if args.lr:
        config.learning_rate = args.lr
    if args.device:
        config.device = args.device
    if args.output_dir:
        config.output_dir = args.output_dir
    
    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(config.output_dir) / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config_path = output_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False)
    
    # Setup logging
    logger = setup_logging(output_dir)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Config: {config.to_dict()}")
    
    # Set seed
    set_seed(config.seed)
    logger.info(f"Random seed: {config.seed}")
    
    # Create dataset
    ssl_path = PROJECT_ROOT / config.ssl_data_path
    logger.info(f"Loading SSL data from: {ssl_path}")
    
    augmentation = ContrastiveAugmentation(
        image_shape=(20, 22),
        noise_std=config.noise_std,
        brightness_range=config.brightness_range,
        contrast_range=config.contrast_range,
        blur_prob=config.blur_prob,
    )
    
    dataset = ContrastiveSSLDataset(
        h5_path=str(ssl_path),
        image_shape=(20, 22),
        augmentation=augmentation,
    )
    
    # Create data loader
    train_loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=config.num_workers > 0,
        drop_last=True,  # Drop last incomplete batch for stable batch norm
    )
    
    # Create model
    model = SimCLRModel(
        in_channels=1,
        feature_dim=config.feature_dim,
        projection_dim=config.projection_dim,
        hidden_dim=config.hidden_dim,
        temperature=config.temperature,
    )
    
    # Create trainer and run
    trainer = SimCLRTrainer(
        model=model,
        train_loader=train_loader,
        config=config,
        output_dir=output_dir,
        logger=logger,
    )
    
    # Load checkpoint if resuming
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train
    try:
        history = trainer.train()
        logger.info("Training completed successfully!")
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        trainer.save_checkpoint("checkpoint_interrupted.pt")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        trainer.save_checkpoint("checkpoint_error.pt")
        raise


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="SimCLR pretraining for cloud images"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (overrides config)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of epochs (overrides config)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (overrides config)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Device to use",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (overrides config)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
