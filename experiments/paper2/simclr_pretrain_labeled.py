#!/usr/bin/env python3
"""SimCLR pretraining on LABELED data subset only.

This script trains SimCLR on the 2,321 labeled samples (instead of 58k unlabeled)
to eliminate distribution mismatch between pretraining and evaluation data.

Hypothesis:
    If SimCLR works on matched distribution, cross-flight R² should improve.
    If it still fails, the method itself doesn't help for this domain shift.

Usage:
    CUDA_VISIBLE_DEVICES=0 python experiments/paper2/simclr_pretrain_labeled.py

Author: Paper 2 Implementation
Date: December 2025
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.paper2.simclr_encoder import SimCLRModel
from experiments.paper2.contrastive_dataset import ContrastiveAugmentation


# =============================================================================
# Dataset for Labeled-Only Contrastive Pretraining
# =============================================================================


class ContrastiveLabeledOnlyDataset(Dataset[Tuple[Tensor, Tensor]]):
    """Dataset for contrastive pretraining using only labeled samples.
    
    This creates two augmented views per sample (like ContrastiveSSLDataset)
    but only uses the 2,321 samples that have CBH labels.
    
    Note: Labels are NOT used during pretraining - this is still unsupervised.
    We just restrict the data pool to match evaluation distribution.
    """
    
    def __init__(
        self,
        ssl_path: str,
        features_path: str,
        image_shape: Tuple[int, int] = (20, 22),
        augmentation: Optional[ContrastiveAugmentation] = None,
    ) -> None:
        self.ssl_path = ssl_path
        self.features_path = features_path
        self.image_shape = image_shape
        self.augmentation = augmentation or ContrastiveAugmentation(image_shape)
        
        # Match labeled samples to SSL images
        self._create_mapping()
        
        print(f"ContrastiveLabeledOnlyDataset initialized:")
        print(f"  SSL path: {ssl_path}")
        print(f"  Features path: {features_path}")
        print(f"  Matched samples: {len(self.ssl_indices):,}")
        print(f"  Image shape: {image_shape}")
    
    def _create_mapping(self) -> None:
        """Create mapping between labeled samples and SSL images."""
        # Load SSL metadata
        with h5py.File(self.ssl_path, "r") as f:
            ssl_metadata = f["metadata"][:]
        
        # Create lookup: (flight_id, sample_id) -> ssl_index
        ssl_lookup = {}
        for i, (flight_id, sample_id, _, _) in enumerate(ssl_metadata):
            ssl_lookup[(int(flight_id), int(sample_id))] = i
        
        # Load labeled data identifiers
        with h5py.File(self.features_path, "r") as f:
            flight_ids = f["metadata/flight_id"][:]
            sample_ids = f["metadata/sample_id"][:]
        
        # Get SSL indices for labeled samples only
        self.ssl_indices = []
        for i in range(len(flight_ids)):
            key = (int(flight_ids[i]), int(sample_ids[i]))
            if key in ssl_lookup:
                self.ssl_indices.append(ssl_lookup[key])
        
        self.ssl_indices = np.array(self.ssl_indices)
    
    def __len__(self) -> int:
        return len(self.ssl_indices)
    
    def _load_image(self, idx: int) -> np.ndarray:
        """Load image by dataset index (maps to SSL index internally)."""
        ssl_idx = self.ssl_indices[idx]
        with h5py.File(self.ssl_path, "r") as f:
            img_flat = f["images"][ssl_idx]
        return img_flat.reshape(self.image_shape)
    
    def _normalize(self, img: np.ndarray) -> np.ndarray:
        """Z-score normalize image."""
        mean = img.mean()
        std = img.std()
        if std > 0:
            return (img - mean) / std
        return img - mean
    
    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """Get two augmented views of the same image."""
        img = self._load_image(idx)
        
        # Generate two different augmented views
        view1 = self.augmentation(img)
        view2 = self.augmentation(img)
        
        # Normalize
        view1 = self._normalize(view1)
        view2 = self._normalize(view2)
        
        # Convert to tensors with channel dimension
        view1_tensor = torch.from_numpy(view1).float().unsqueeze(0)
        view2_tensor = torch.from_numpy(view2).float().unsqueeze(0)
        
        return view1_tensor, view2_tensor


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class SimCLRLabeledConfig:
    """Configuration for SimCLR pretraining on labeled data."""
    
    # Data paths
    ssl_data_path: str = "data_ssl/images/train.h5"
    features_path: str = "data/Integrated_Features_max.hdf5"
    output_dir: str = "outputs/paper2_simclr_labeled"
    
    # Model architecture
    feature_dim: int = 256
    projection_dim: int = 128
    hidden_dim: int = 256
    temperature: float = 0.5
    
    # Training - adjusted for smaller dataset
    batch_size: int = 128  # Smaller batch for 2.3k samples
    num_epochs: int = 200  # More epochs for smaller dataset
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    warmup_epochs: int = 20
    
    # Augmentation
    noise_std: float = 0.1
    brightness_range: Tuple[float, float] = (0.8, 1.2)
    contrast_range: Tuple[float, float] = (0.8, 1.2)
    blur_prob: float = 0.5
    
    # System
    num_workers: int = 4
    device: str = "cuda"
    seed: int = 42
    save_every: int = 20
    log_every: int = 10
    
    def to_dict(self) -> Dict:
        return {
            "ssl_data_path": self.ssl_data_path,
            "features_path": self.features_path,
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
# Training
# =============================================================================


def setup_logging(output_dir: Path) -> logging.Logger:
    """Configure logging."""
    logger = logging.getLogger("simclr_labeled")
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Clear existing handlers
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    log_file = output_dir / "training.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(console_format)
    logger.addHandler(file_handler)
    
    return logger


def set_seed(seed: int) -> None:
    """Set random seeds."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_epoch(
    model: SimCLRModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    epoch: int,
    config: SimCLRLabeledConfig,
    logger: logging.Logger,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, (view1, view2) in enumerate(loader):
        view1 = view1.to(device)
        view2 = view2.to(device)
        
        optimizer.zero_grad()
        loss, _, _ = model(view1, view2)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if (batch_idx + 1) % config.log_every == 0:
            lr = scheduler.get_last_lr()[0]
            logger.info(
                f"Epoch {epoch} | Step {batch_idx + 1}/{len(loader)} | "
                f"Loss: {loss.item():.4f} | LR: {lr:.2e}"
            )
    
    return total_loss / num_batches


def save_checkpoint(
    model: SimCLRModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    best_loss: float,
    config: SimCLRLabeledConfig,
    output_dir: Path,
    filename: str,
    logger: logging.Logger,
) -> None:
    """Save checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_loss": best_loss,
        "config": config.to_dict(),
    }
    path = output_dir / filename
    torch.save(checkpoint, path)
    logger.info(f"Saved checkpoint: {path}")


def main() -> None:
    """Main training function."""
    config = SimCLRLabeledConfig()
    
    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(config.output_dir) / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False)
    
    # Setup
    logger = setup_logging(output_dir)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Config: {config.to_dict()}")
    
    set_seed(config.seed)
    
    # Device
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Dataset - labeled only
    ssl_path = PROJECT_ROOT / config.ssl_data_path
    features_path = PROJECT_ROOT / config.features_path
    
    augmentation = ContrastiveAugmentation(
        image_shape=(20, 22),
        noise_std=config.noise_std,
        brightness_range=config.brightness_range,
        contrast_range=config.contrast_range,
        blur_prob=config.blur_prob,
    )
    
    dataset = ContrastiveLabeledOnlyDataset(
        ssl_path=str(ssl_path),
        features_path=str(features_path),
        image_shape=(20, 22),
        augmentation=augmentation,
    )
    
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=config.num_workers > 0,
        drop_last=True,
    )
    
    logger.info(f"Training samples: {len(dataset):,}")
    logger.info(f"Batches per epoch: {len(loader)}")
    
    # Model
    model = SimCLRModel(
        in_channels=1,
        feature_dim=config.feature_dim,
        projection_dim=config.projection_dim,
        hidden_dim=config.hidden_dim,
        temperature=config.temperature,
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {n_params:,}")
    
    # Optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        total_iters=config.warmup_epochs * len(loader),
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=(config.num_epochs - config.warmup_epochs) * len(loader),
        eta_min=1e-6,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[config.warmup_epochs * len(loader)],
    )
    
    # Training loop
    logger.info(f"\nStarting training for {config.num_epochs} epochs...")
    logger.info("=" * 60)
    
    best_loss = float("inf")
    history = {"train_loss": [], "learning_rate": []}
    
    start_time = time.time()
    
    for epoch in range(1, config.num_epochs + 1):
        epoch_start = time.time()
        
        avg_loss = train_epoch(
            model, loader, optimizer, scheduler, device, epoch, config, logger
        )
        
        epoch_time = time.time() - epoch_start
        history["train_loss"].append(avg_loss)
        history["learning_rate"].append(scheduler.get_last_lr()[0])
        
        logger.info(
            f"Epoch {epoch} complete | Avg Loss: {avg_loss:.4f} | Time: {epoch_time:.1f}s"
        )
        
        # Save checkpoints
        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss
            save_checkpoint(
                model, optimizer, scheduler, epoch, best_loss, config,
                output_dir, "best_model.pt", logger
            )
        
        if epoch % config.save_every == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, best_loss, config,
                output_dir, f"checkpoint_epoch_{epoch}.pt", logger
            )
        
        save_checkpoint(
            model, optimizer, scheduler, epoch, best_loss, config,
            output_dir, "checkpoint_latest.pt", logger
        )
    
    total_time = time.time() - start_time
    
    # Save final model and history
    save_checkpoint(
        model, optimizer, scheduler, config.num_epochs, best_loss, config,
        output_dir, "final_model.pt", logger
    )
    
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    
    logger.info("=" * 60)
    logger.info(f"Training complete!")
    logger.info(f"Total time: {total_time / 60:.1f} minutes")
    logger.info(f"Best loss: {best_loss:.4f}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 60)
    
    # Print next steps
    logger.info("\nNext step - run linear probe evaluation:")
    logger.info(f"  python experiments/paper2/linear_probe.py --checkpoint {output_dir}/best_model.pt")


if __name__ == "__main__":
    main()
