#!/usr/bin/env python3
"""
WP-5 Task 1.1: ResNet-50 Baseline for Cloud Base Height Retrieval

This script implements a pre-trained ResNet-50 backbone for CBH prediction.
The model uses ImageNet pre-trained weights and is fine-tuned on cloud imagery.

Key Architecture:
- Backbone: ResNet-50 (torchvision, ImageNet pre-trained)
- Input: 1-channel grayscale images (440x640)
- Strategy: Duplicate grayscale channel 3x to leverage RGB pre-trained weights
- Fine-tuning: Freeze first 3 blocks, train block 4 + regression head
- Regression Head: Linear(2048, 512), ReLU(), Dropout(0.3), Linear(512, 1)

Validation: Stratified 5-Fold Cross-Validation (n_splits=5)
Target: R² > 0.45 (Sprint 5 WP-1 objective)

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
from torch.utils.data import DataLoader
import torchvision.models as models
import warnings
from typing import Dict, List
import argparse

warnings.filterwarnings("ignore")

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.hdf5_dataset import HDF5CloudDataset
from sow_outputs.wp5.wp5_utils import (
    ImageOnlyDataset,
    FoldMetrics,
    EarlyStopper,
    compute_metrics,
    train_epoch,
    validate_epoch,
    get_stratified_folds,
    generate_wp5_report,
    save_report,
    print_fold_summary,
    print_aggregate_summary,
)


class ResNet50Regressor(nn.Module):
    """
    ResNet-50 backbone with regression head for CBH prediction.

    Architecture:
    - Backbone: Pre-trained ResNet-50 (ImageNet)
    - Input: 1-channel grayscale -> duplicated to 3-channel
    - Fine-tuning: Freeze layers 1-3, train layer 4 + head
    - Head: FC(2048 -> 512) -> ReLU -> Dropout(0.3) -> FC(512 -> 1)
    """

    def __init__(self, pretrained: bool = True, freeze_early_layers: bool = True):
        super().__init__()

        # Load pre-trained ResNet-50
        self.backbone = models.resnet50(pretrained=pretrained)

        # Modify first conv layer to accept 1-channel input
        # Strategy: Keep pretrained 3-channel weights, duplicate grayscale input 3x
        # This is done in forward() to preserve pretrained conv1 weights
        self.original_conv1 = self.backbone.conv1

        # Freeze early layers (layer1, layer2, layer3) if requested
        if freeze_early_layers:
            # Freeze initial conv, bn, layer1, layer2, layer3
            for param in self.backbone.conv1.parameters():
                param.requires_grad = False
            for param in self.backbone.bn1.parameters():
                param.requires_grad = False
            for param in self.backbone.layer1.parameters():
                param.requires_grad = False
            for param in self.backbone.layer2.parameters():
                param.requires_grad = False
            for param in self.backbone.layer3.parameters():
                param.requires_grad = False

            print("✓ Frozen: conv1, bn1, layer1, layer2, layer3")
            print("✓ Trainable: layer4, regression head")

        # Remove the original FC layer (1000 classes)
        self.backbone.fc = nn.Identity()

        # Create regression head
        # ResNet-50 layer4 output: 2048 channels
        self.regression_head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 1),
        )

        print(f"✓ ResNet-50 initialized (pretrained={pretrained})")
        print(f"  Output feature dim: 2048")
        print(f"  Regression head: 2048 -> 512 -> 1")

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: (B, 1, H, W) grayscale images

        Returns:
            predictions: (B, 1) CBH predictions in km
        """
        # Input: (B, 1, H, W)
        # Duplicate grayscale channel to RGB: (B, 1, H, W) -> (B, 3, H, W)
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)

        # Extract features using ResNet backbone
        # ResNet forward without FC layer (we replaced it with Identity)
        x = self.backbone(x)  # (B, 2048)

        # Regression head
        x = self.regression_head(x)  # (B, 1)

        return x


class ResNetTrainer:
    """Trainer for ResNet-50 baseline with Stratified K-Fold CV."""

    def __init__(
        self,
        config_path: str,
        output_dir: str = "sow_outputs/wp5",
        device: str = None,
        verbose: bool = True,
    ):
        self.config_path = config_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose

        # Create subdirectories
        self.model_dir = self.output_dir / "models" / "resnet"
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
        """Load image dataset."""
        if self.verbose:
            print("\n" + "=" * 80)
            print("Loading Data")
            print("=" * 80)

        # Load image dataset (no augmentation for evaluation)
        self.image_dataset = HDF5CloudDataset(
            flight_configs=self.config["flights"],
            indices=None,
            augment=False,
            swath_slice=self.config.get("swath_slice", [40, 480]),
            temporal_frames=1,  # Single frame for ResNet baseline
        )

        # Cache CBH values
        if self.verbose:
            print("Caching CBH values...")
        self.cbh_values = self.image_dataset.get_unscaled_y()

        n_samples = len(self.image_dataset)

        if self.verbose:
            print(f"\n✓ Data loaded:")
            print(f"  Total samples: {n_samples}")
            print(
                f"  CBH range: [{self.cbh_values.min():.3f}, {self.cbh_values.max():.3f}] km"
            )
            print(f"  CBH mean: {self.cbh_values.mean():.3f} km")
            print(f"  CBH std: {self.cbh_values.std():.3f} km")

    def train_single_fold(
        self,
        fold_id: int,
        train_indices: List[int],
        val_indices: List[int],
        test_indices: List[int],
        n_epochs: int = 50,
        batch_size: int = 16,
        learning_rate: float = 1e-4,
        accumulation_steps: int = 1,
    ) -> FoldMetrics:
        """Train and evaluate a single fold."""

        print(f"\n{'=' * 80}")
        print(f"Fold {fold_id + 1}/5")
        print(f"{'=' * 80}")
        print(f"  Training:   {len(train_indices)} samples")
        print(f"  Validation: {len(val_indices)} samples")
        print(f"  Test:       {len(test_indices)} samples")

        # Create datasets
        train_dataset = ImageOnlyDataset(
            self.image_dataset, train_indices, self.cbh_values
        )
        val_dataset = ImageOnlyDataset(self.image_dataset, val_indices, self.cbh_values)
        test_dataset = ImageOnlyDataset(
            self.image_dataset, test_indices, self.cbh_values
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
        model = ResNet50Regressor(pretrained=True, freeze_early_layers=True).to(
            self.device
        )

        # Optimizer (only trainable parameters)
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=learning_rate,
            weight_decay=1e-4,
        )

        # Loss function
        criterion = nn.MSELoss()

        # Early stopping
        early_stopper = EarlyStopper(patience=10, min_delta=1e-4)

        # Training loop
        best_val_loss = float("inf")
        best_epoch = 0
        best_model_state = None

        for epoch in range(n_epochs):
            # Train
            train_loss = train_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                self.device,
                fusion_mode=False,
                accumulation_steps=accumulation_steps,
            )

            # Validate
            val_loss, val_metrics, _, _ = validate_epoch(
                model, val_loader, criterion, self.device, fusion_mode=False
            )

            # Print progress
            if self.verbose and (epoch + 1) % 5 == 0:
                print(
                    f"  Epoch {epoch + 1:3d}/{n_epochs}: "
                    f"Train Loss = {train_loss:.4f}, "
                    f"Val Loss = {val_loss:.4f}, "
                    f"Val R² = {val_metrics['r2']:.4f}, "
                    f"Val MAE = {val_metrics['mae_km']:.4f} km"
                )

            # Save best model
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

        # Evaluate on test set
        test_loss, test_metrics, test_preds, test_targets = validate_epoch(
            model, test_loader, criterion, self.device, fusion_mode=False
        )

        # Save model
        model_path = self.model_dir / f"resnet50_fold{fold_id}.pth"
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

        # Print fold summary
        if self.verbose:
            print_fold_summary(fold_id, 5, fold_metrics)

        return fold_metrics

    def run_kfold_cv(
        self,
        n_epochs: int = 50,
        batch_size: int = 16,
        learning_rate: float = 1e-4,
        accumulation_steps: int = 1,
    ) -> List[FoldMetrics]:
        """Run Stratified 5-Fold Cross-Validation."""

        print("\n" + "=" * 80)
        print("STRATIFIED 5-FOLD CROSS-VALIDATION: ResNet-50 Baseline")
        print("=" * 80)

        # Get stratified folds
        folds = get_stratified_folds(self.cbh_values, n_splits=5, random_state=42)

        fold_results = []

        for fold_id, (train_val_idx, test_idx) in enumerate(folds):
            # Split train_val into train/val (80/20)
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
    parser = argparse.ArgumentParser(
        description="WP-5 Task 1.1: ResNet-50 Baseline Training"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/bestComboConfig.yaml",
        help="Config YAML path",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="sow_outputs/wp5",
        help="Output directory",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Max epochs per fold",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size (adjust for 8GB VRAM)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--accumulation-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps (for VRAM saving)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Verbose output",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("WP-5 TASK 1.1: ResNet-50 Baseline for Cloud Base Height Retrieval")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Config file:         {args.config}")
    print(f"  Output directory:    {args.output_dir}")
    print(f"  Max epochs:          {args.epochs}")
    print(f"  Batch size:          {args.batch_size}")
    print(f"  Learning rate:       {args.learning_rate}")
    print(f"  Accumulation steps:  {args.accumulation_steps}")

    # Create trainer
    trainer = ResNetTrainer(
        config_path=args.config,
        output_dir=args.output_dir,
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
        model_name="ResNet-50 Baseline",
        sprint_work_package="WP-1: Pre-Trained Backbones",
        script_path="sow_outputs/wp5/wp5_resnet_baseline.py",
        fold_results=fold_results,
        hardware_config=hardware_config,
        additional_info={
            "architecture": {
                "backbone": "ResNet-50 (ImageNet pre-trained)",
                "input_channels": 1,
                "input_handling": "Grayscale duplicated 3x to RGB",
                "frozen_layers": ["conv1", "bn1", "layer1", "layer2", "layer3"],
                "trainable_layers": ["layer4", "regression_head"],
                "regression_head": "Linear(2048->512), ReLU, Dropout(0.3), Linear(512->1)",
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
    report_path = trainer.report_dir / "WP5_ResNet_Report.json"
    save_report(report, report_path)

    # Print summary
    print_aggregate_summary(report)

    print("\n" + "=" * 80)
    print("WP-5 TASK 1.1: COMPLETE")
    print("=" * 80)
    print(f"✓ Models saved to:  {trainer.model_dir}")
    print(f"✓ Report saved to:  {report_path}")


if __name__ == "__main__":
    main()
