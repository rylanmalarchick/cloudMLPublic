#!/usr/bin/env python3
"""
WP-5 Task 2.1: Temporal Modeling for Cloud Base Height Retrieval

This script implements temporal modeling by processing sequences of consecutive frames.
The model uses the best-performing WP-1 architecture (ViT-Tiny) as a frame encoder,
combined with temporal attention to aggregate information across time.

Key Architecture:
- Input: 5 consecutive frames (t-2, t-1, t, t+1, t+2)
- Frame Encoder: ViT-Tiny (pre-trained, shared across all frames)
- Temporal Aggregation: Temporal Attention mechanism
- Output: Single CBH prediction for center frame (t)

Validation: Stratified 5-Fold Cross-Validation (n_splits=5)
Target: R² > 0.50 (Sprint 5 WP-2 objective)

VRAM Mitigation: Gradient accumulation (5x memory usage from multi-frame input)

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
from torch.utils.data import Dataset, DataLoader
import warnings
from typing import Dict, List, Tuple
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


class TemporalDataset(Dataset):
    """
    Dataset that loads sequences of consecutive frames for temporal modeling.

    For each sample, returns a sequence of N frames centered on the target frame.
    Edge cases (near start/end of flight) are handled by padding with the edge frame.
    """

    def __init__(
        self,
        image_dataset: HDF5CloudDataset,
        indices: List[int],
        cbh_values: np.ndarray,
        n_frames: int = 5,
    ):
        """
        Args:
            image_dataset: HDF5CloudDataset instance
            indices: List of global indices to use
            cbh_values: Cached CBH values (km)
            n_frames: Number of frames in temporal sequence (must be odd)
        """
        self.image_dataset = image_dataset
        self.indices = indices
        self.cbh_values = cbh_values
        self.n_frames = n_frames
        self.temporal_offset = n_frames // 2

        if n_frames % 2 == 0:
            raise ValueError("n_frames must be odd")

        # Build mapping from global index to (flight_idx, local_idx)
        self.global_to_local = image_dataset.global_to_local

        # Get flight boundaries by scanning global_to_local
        self.flight_boundaries = {}
        for global_idx, (flight_idx, local_idx) in enumerate(self.global_to_local):
            if flight_idx not in self.flight_boundaries:
                self.flight_boundaries[flight_idx] = [global_idx, global_idx]
            else:
                self.flight_boundaries[flight_idx][1] = global_idx

        # Convert to tuples (start, end+1)
        for flight_idx in self.flight_boundaries:
            start, end = self.flight_boundaries[flight_idx]
            self.flight_boundaries[flight_idx] = (start, end + 1)

    def __len__(self):
        return len(self.indices)

    def _get_frame_sequence(self, center_idx: int) -> List[int]:
        """
        Get indices for temporal sequence centered on center_idx.
        Handle edge cases by clamping to flight boundaries.
        """
        # Get flight info for center frame
        flight_idx, local_idx = self.global_to_local[center_idx]
        start_global, end_global = self.flight_boundaries[flight_idx]

        # Generate sequence indices
        sequence_indices = []
        for offset in range(-self.temporal_offset, self.temporal_offset + 1):
            target_idx = center_idx + offset
            # Clamp to flight boundaries (don't cross flight boundaries)
            target_idx = max(start_global, min(end_global - 1, target_idx))
            sequence_indices.append(target_idx)

        return sequence_indices

    def __getitem__(self, idx):
        # Get center frame index
        center_global_idx = self.indices[idx]

        # Get sequence indices
        sequence_indices = self._get_frame_sequence(center_global_idx)

        # Load all frames in sequence
        frames = []
        for frame_idx in sequence_indices:
            image, _, _, _, _, _ = self.image_dataset[frame_idx]

            # Convert to tensor if needed
            if isinstance(image, torch.Tensor):
                image_tensor = image.float()
            else:
                image_tensor = torch.from_numpy(image).float()

            # Ensure 3D: (C, H, W)
            if image_tensor.dim() == 2:
                image_tensor = image_tensor.unsqueeze(0)

            frames.append(image_tensor)

        # Stack frames: (N, C, H, W)
        frames_tensor = torch.stack(frames, dim=0)

        # Get target CBH for center frame
        cbh_km = self.cbh_values[center_global_idx]

        return frames_tensor, torch.tensor(cbh_km).float()


class TemporalAttention(nn.Module):
    """
    Temporal attention mechanism to aggregate features across time.

    Computes attention weights over the temporal dimension and
    produces a weighted combination of frame features.
    """

    def __init__(self, feature_dim: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.feature_dim = feature_dim

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True,
        )

        # Layer norm
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, x):
        """
        Args:
            x: (B, T, D) frame features

        Returns:
            aggregated: (B, D) aggregated features
        """
        # Self-attention over temporal dimension
        attn_out, attn_weights = self.attention(x, x, x)

        # Add residual connection
        x = x + attn_out
        x = self.norm(x)

        # Global average pooling over time
        aggregated = x.mean(dim=1)  # (B, D)

        return aggregated


class TemporalViTRegressor(nn.Module):
    """
    Temporal model using ViT-Tiny as frame encoder + Temporal Attention.

    Architecture:
    1. Process each frame independently with shared ViT encoder
    2. Aggregate frame features using temporal attention
    3. Predict CBH from aggregated features
    """

    def __init__(
        self,
        pretrained_model: str = "WinKawaks/vit-tiny-patch16-224",
        n_frames: int = 5,
    ):
        super().__init__()

        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers library required. Install with: pip install transformers"
            )

        self.n_frames = n_frames

        print(f"Loading ViT frame encoder: {pretrained_model}")

        # Load pre-trained ViT
        try:
            # Load ViT for feature extraction (without classification head)
            self.vit = ViTForImageClassification.from_pretrained(
                pretrained_model,
                num_labels=1,
                ignore_mismatched_sizes=True,
            )
            # Remove the final classifier to get features
            self.feature_dim = self.vit.config.hidden_size
            print(f"✓ ViT encoder loaded (feature dim: {self.feature_dim})")
        except Exception as e:
            print(f"Failed to load {pretrained_model}: {e}")
            raise

        # Temporal attention
        self.temporal_attention = TemporalAttention(
            feature_dim=self.feature_dim,
            num_heads=4,
        )

        # Regression head
        self.regression_head = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

        print(f"✓ Temporal ViT initialized")
        print(f"  Input: {n_frames} frames × 224×224")
        print(f"  Frame encoder: ViT-Tiny")
        print(f"  Temporal aggregation: Multi-head attention (4 heads)")

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: (B, T, C, H, W) sequence of frames

        Returns:
            predictions: (B, 1) CBH predictions in km
        """
        B, T, C, H, W = x.shape

        # Duplicate grayscale to RGB if needed
        if C == 1:
            x = x.repeat(1, 1, 3, 1, 1)  # (B, T, 3, H, W)

        # Resize to 224x224 if needed
        if H != 224 or W != 224:
            # Reshape to (B*T, C, H, W) for batch resize
            x = x.view(B * T, 3, H, W)
            x = torch.nn.functional.interpolate(
                x, size=(224, 224), mode="bilinear", align_corners=False
            )
            x = x.view(B, T, 3, 224, 224)

        # Process each frame independently
        # Reshape: (B, T, C, H, W) -> (B*T, C, H, W)
        x = x.view(B * T, 3, 224, 224)

        # Extract features from ViT (before final classification)
        # We need the pooled output (CLS token)
        outputs = self.vit.vit(x)
        frame_features = outputs.last_hidden_state[:, 0, :]  # (B*T, D) - CLS token

        # Reshape back: (B*T, D) -> (B, T, D)
        frame_features = frame_features.view(B, T, self.feature_dim)

        # Temporal attention aggregation
        aggregated_features = self.temporal_attention(frame_features)  # (B, D)

        # Regression
        predictions = self.regression_head(aggregated_features)  # (B, 1)

        return predictions


class TemporalTrainer:
    """Trainer for Temporal ViT model with Stratified K-Fold CV."""

    def __init__(
        self,
        config_path: str,
        output_dir: str = "sow_outputs/wp5",
        pretrained_model: str = "WinKawaks/vit-tiny-patch16-224",
        n_frames: int = 5,
        device: str = None,
        verbose: bool = True,
    ):
        self.config_path = config_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pretrained_model = pretrained_model
        self.n_frames = n_frames
        self.verbose = verbose

        # Create subdirectories
        self.model_dir = self.output_dir / "models" / "temporal"
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

        # Load image dataset (no augmentation)
        self.image_dataset = HDF5CloudDataset(
            flight_configs=self.config["flights"],
            indices=None,
            augment=False,
            swath_slice=self.config.get("swath_slice", [40, 480]),
            temporal_frames=1,  # Single frame per load (we handle temporal in dataset)
        )

        # Cache CBH values
        if self.verbose:
            print("Caching CBH values...")
        self.cbh_values = self.image_dataset.get_unscaled_y()

        n_samples = len(self.image_dataset)

        if self.verbose:
            print(f"\n✓ Data loaded:")
            print(f"  Total samples: {n_samples}")
            print(f"  Temporal frames per sample: {self.n_frames}")
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
        batch_size: int = 8,
        learning_rate: float = 3e-5,
        accumulation_steps: int = 2,
    ) -> FoldMetrics:
        """Train and evaluate a single fold."""

        print(f"\n{'=' * 80}")
        print(f"Fold {fold_id + 1}/5")
        print(f"{'=' * 80}")
        print(f"  Training:   {len(train_indices)} samples")
        print(f"  Validation: {len(val_indices)} samples")
        print(f"  Test:       {len(test_indices)} samples")

        # Create temporal datasets
        train_dataset = TemporalDataset(
            self.image_dataset, train_indices, self.cbh_values, self.n_frames
        )
        val_dataset = TemporalDataset(
            self.image_dataset, val_indices, self.cbh_values, self.n_frames
        )
        test_dataset = TemporalDataset(
            self.image_dataset, test_indices, self.cbh_values, self.n_frames
        )

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,  # Reduced for temporal (more memory per sample)
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

        # Create model
        model = TemporalViTRegressor(
            pretrained_model=self.pretrained_model,
            n_frames=self.n_frames,
        ).to(self.device)

        # Optimizer
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
            model.train()
            total_train_loss = 0.0
            n_train_batches = 0

            optimizer.zero_grad()

            for batch_idx, (frames, targets) in enumerate(train_loader):
                frames = frames.to(self.device)
                targets = targets.to(self.device).unsqueeze(1)

                # Forward pass
                predictions = model(frames)

                # Compute loss
                loss = criterion(predictions, targets)
                loss = loss / accumulation_steps

                # Backward pass
                loss.backward()

                # Update weights every accumulation_steps
                if (batch_idx + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                total_train_loss += loss.item() * accumulation_steps
                n_train_batches += 1

            # Final update if needed
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
                for frames, targets in val_loader:
                    frames = frames.to(self.device)
                    targets = targets.to(self.device).unsqueeze(1)

                    predictions = model(frames)

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
        model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for frames, targets in test_loader:
                frames = frames.to(self.device)
                targets = targets.to(self.device).unsqueeze(1)

                predictions = model(frames)

                all_preds.append(predictions.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        test_preds = np.concatenate(all_preds, axis=0).flatten()
        test_targets = np.concatenate(all_targets, axis=0).flatten()
        test_metrics = compute_metrics(test_preds, test_targets)

        # Save model
        model_path = self.model_dir / f"temporal_vit_fold{fold_id}.pth"
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
        batch_size: int = 8,
        learning_rate: float = 3e-5,
        accumulation_steps: int = 2,
    ) -> List[FoldMetrics]:
        """Run Stratified 5-Fold Cross-Validation."""

        print("\n" + "=" * 80)
        print("STRATIFIED 5-FOLD CROSS-VALIDATION: Temporal ViT Model")
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
        description="WP-5 Task 2.1: Temporal Modeling Training"
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
        "--n-frames",
        type=int,
        default=5,
        help="Number of frames in temporal sequence",
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
        default=6,
        help="Batch size (reduced for temporal/VRAM)",
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
        default=3,
        help="Gradient accumulation steps (CRITICAL for temporal/VRAM)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Verbose output",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("WP-5 TASK 2.1: Temporal Modeling for Cloud Base Height Retrieval")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Config file:         {args.config}")
    print(f"  Output directory:    {args.output_dir}")
    print(f"  Pretrained model:    {args.pretrained_model}")
    print(f"  Temporal frames:     {args.n_frames}")
    print(f"  Max epochs:          {args.epochs}")
    print(f"  Batch size:          {args.batch_size}")
    print(f"  Learning rate:       {args.learning_rate}")
    print(f"  Accumulation steps:  {args.accumulation_steps}")
    print(f"  Effective batch:     {args.batch_size * args.accumulation_steps}")

    # Create trainer
    trainer = TemporalTrainer(
        config_path=args.config,
        output_dir=args.output_dir,
        pretrained_model=args.pretrained_model,
        n_frames=args.n_frames,
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
        "vram_mitigation_used": f"gradient_accumulation_steps: {args.accumulation_steps}, reduced batch_size: {args.batch_size}",
    }

    report = generate_wp5_report(
        model_name="Temporal ViT (Multi-Frame)",
        sprint_work_package="WP-2: Temporal Modeling",
        script_path="sow_outputs/wp5/wp5_temporal.py",
        fold_results=fold_results,
        hardware_config=hardware_config,
        additional_info={
            "architecture": {
                "frame_encoder": f"ViT-Tiny ({args.pretrained_model})",
                "temporal_frames": args.n_frames,
                "temporal_aggregation": "Multi-head Attention (4 heads)",
                "input_handling": "Sequence of frames, resized to 224x224",
                "edge_handling": "Clamp to flight boundaries (no cross-flight sequences)",
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
                "effective_batch_size": args.batch_size * args.accumulation_steps,
            },
        },
    )

    # Save report
    report_path = trainer.report_dir / "WP5_Temporal_Report.json"
    save_report(report, report_path)

    # Print summary
    print_aggregate_summary(report)

    print("\n" + "=" * 80)
    print("WP-5 TASK 2.1: COMPLETE")
    print("=" * 80)
    print(f"✓ Models saved to:  {trainer.model_dir}")
    print(f"✓ Report saved to:  {report_path}")


if __name__ == "__main__":
    main()
