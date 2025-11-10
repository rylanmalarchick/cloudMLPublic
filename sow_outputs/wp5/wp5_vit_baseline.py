#!/usr/bin/env python3
"""
WP-5 Task 1.2: Vision Transformer (ViT-Tiny) Baseline for Cloud Base Height Retrieval

This script implements a pre-trained Vision Transformer (ViT-Tiny) for CBH prediction.
The model uses ImageNet-21k pre-trained weights and is fine-tuned on cloud imagery.

Key Architecture:
- Backbone: ViT-Tiny (google/vit-base-patch16-224 or WinKawaks/vit-tiny-patch16-224)
- Input: 1-channel grayscale images (440x640) resized to 224x224
- Strategy: Use transformers library with num_labels=1 for regression
- Patch size: 16x16
- Fine-tuning: Full model training (ViT is smaller, can train all layers)

Validation: Stratified 5-Fold Cross-Validation (n_splits=5)
Target: R² > 0.48 (Sprint 5 WP-1 objective)

Author: Sprint 5 Execution Agent
Date: 2025-11-10
"""

import sys
from pathlib import Path
import numpy as np
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
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

# Try to import transformers
try:
    from transformers import ViTForImageClassification, ViTConfig

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print(
        "WARNING: transformers library not available. Please install: pip install transformers"
    )


class ViTTinyRegressor(nn.Module):
    """
    Vision Transformer (ViT-Tiny) with regression head for CBH prediction.

    Architecture:
    - Backbone: ViT-Tiny (ImageNet-21k pre-trained)
    - Input: 1-channel grayscale -> converted to 3-channel (grayscale duplicated)
    - Input size: Resized to 224x224 (ViT standard)
    - Output: Single value (CBH in km)
    """

    def __init__(self, pretrained_model: str = "WinKawaks/vit-tiny-patch16-224"):
        super().__init__()

        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers library required. Install with: pip install transformers"
            )

        print(f"Loading ViT model: {pretrained_model}")

        # Load pre-trained ViT with regression head
        # Use num_labels=1 for regression, ignore_mismatched_sizes=True for head
        try:
            self.vit = ViTForImageClassification.from_pretrained(
                pretrained_model,
                num_labels=1,
                ignore_mismatched_sizes=True,
            )
            print(f"✓ ViT loaded from {pretrained_model}")
        except Exception as e:
            print(f"Failed to load {pretrained_model}: {e}")
            print("Falling back to random initialization...")
            config = ViTConfig(
                image_size=224,
                patch_size=16,
                num_channels=3,
                hidden_size=192,
                num_hidden_layers=12,
                num_attention_heads=3,
                intermediate_size=768,
                num_labels=1,
            )
            self.vit = ViTForImageClassification(config)
            print("✓ ViT initialized with random weights (tiny config)")

        # ViT expects 3-channel RGB input, we'll duplicate grayscale
        # This is handled in forward()

        print(f"✓ ViT initialized")
        print(f"  Input size: 224x224 (resized from 440x640)")
        print(f"  Patch size: 16x16")
        print(f"  Number of patches: {(224 // 16) ** 2}")

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: (B, 1, H, W) grayscale images (will be resized to 224x224)

        Returns:
            predictions: (B, 1) CBH predictions in km
        """
        # Input: (B, 1, H, W)
        # Duplicate grayscale channel to RGB: (B, 1, H, W) -> (B, 3, H, W)
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)

        # ViT expects 224x224 input
        # Use interpolate to resize
        if x.size(2) != 224 or x.size(3) != 224:
            x = torch.nn.functional.interpolate(
                x, size=(224, 224), mode="bilinear", align_corners=False
            )

        # Forward through ViT
        # Returns ViTForImageClassificationOutput with logits
        outputs = self.vit(x)
        logits = outputs.logits  # (B, 1)

        return logits


class ViTDataset(ImageOnlyDataset):
    """
    Extended ImageOnlyDataset with resizing transform for ViT.
    ViT requires 224x224 input.
    """

    def __init__(self, image_dataset, indices: List[int], cbh_values=None):
        super().__init__(image_dataset, indices, cbh_values, transform=None)

        # Define resize transform (handled in model forward, but keep for consistency)
        self.resize = T.Resize((224, 224), antialias=True)

    def __getitem__(self, idx):
        # Get image and target from parent
        image_tensor, cbh_km = super().__getitem__(idx)

        # Note: Resizing is now done in the model's forward pass
        # to avoid storing large 224x224 images in memory during data loading

        return image_tensor, cbh_km


class ViTTrainer:
    """Trainer for ViT-Tiny baseline with Stratified K-Fold CV."""

    def __init__(
        self,
        config_path: str,
        output_dir: str = "sow_outputs/wp5",
        pretrained_model: str = "WinKawaks/vit-tiny-patch16-224",
        device: str = None,
        verbose: bool = True,
    ):
        self.config_path = config_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pretrained_model = pretrained_model
        self.verbose = verbose

        # Create subdirectories
        self.model_dir = self.output_dir / "models" / "vit"
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
            temporal_frames=1,  # Single frame for ViT baseline
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

        # Create datasets (using ViTDataset for resize)
        train_dataset = ViTDataset(self.image_dataset, train_indices, self.cbh_values)
        val_dataset = ViTDataset(self.image_dataset, val_indices, self.cbh_values)
        test_dataset = ViTDataset(self.image_dataset, test_indices, self.cbh_values)

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
        model = ViTTinyRegressor(pretrained_model=self.pretrained_model).to(self.device)

        # Optimizer (all parameters trainable for ViT)
        optimizer = optim.AdamW(
            model.parameters(),
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
        model_path = self.model_dir / f"vit_tiny_fold{fold_id}.pth"
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
        print("STRATIFIED 5-FOLD CROSS-VALIDATION: ViT-Tiny Baseline")
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
        description="WP-5 Task 1.2: ViT-Tiny Baseline Training"
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
        "--pretrained-model",
        type=str,
        default="WinKawaks/vit-tiny-patch16-224",
        help="Pretrained ViT model name",
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
    print("WP-5 TASK 1.2: ViT-Tiny Baseline for Cloud Base Height Retrieval")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Config file:         {args.config}")
    print(f"  Output directory:    {args.output_dir}")
    print(f"  Pretrained model:    {args.pretrained_model}")
    print(f"  Max epochs:          {args.epochs}")
    print(f"  Batch size:          {args.batch_size}")
    print(f"  Learning rate:       {args.learning_rate}")
    print(f"  Accumulation steps:  {args.accumulation_steps}")

    # Create trainer
    trainer = ViTTrainer(
        config_path=args.config,
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
        model_name="ViT-Tiny Baseline",
        sprint_work_package="WP-1: Pre-Trained Backbones",
        script_path="sow_outputs/wp5/wp5_vit_baseline.py",
        fold_results=fold_results,
        hardware_config=hardware_config,
        additional_info={
            "architecture": {
                "backbone": f"ViT-Tiny ({args.pretrained_model})",
                "input_channels": 1,
                "input_handling": "Grayscale duplicated 3x to RGB, resized 440x640 -> 224x224",
                "patch_size": "16x16",
                "num_patches": (224 // 16) ** 2,
                "trainable_layers": "All layers (ViT is small enough)",
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
    report_path = trainer.report_dir / "WP5_ViT_Report.json"
    save_report(report, report_path)

    # Print summary
    print_aggregate_summary(report)

    print("\n" + "=" * 80)
    print("WP-5 TASK 1.2: COMPLETE")
    print("=" * 80)
    print(f"✓ Models saved to:  {trainer.model_dir}")
    print(f"✓ Report saved to:  {report_path}")


if __name__ == "__main__":
    main()
