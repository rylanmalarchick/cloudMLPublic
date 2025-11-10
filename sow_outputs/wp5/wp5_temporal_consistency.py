#!/usr/bin/env python3
"""
WP-5 Task 2.2: Temporal Consistency Regularization for Cloud Base Height Retrieval

This script extends the Temporal ViT model (Task 2.1) with temporal consistency loss.
The model now predicts CBH for all 5 frames in the sequence and is regularized to
produce smooth, physically plausible predictions across time.

Key Architecture:
- Input: 5 consecutive frames (t-2, t-1, t, t+1, t+2)
- Frame Encoder: ViT-Tiny (pre-trained, shared across all frames)
- Temporal Aggregation: Temporal Attention mechanism
- Output: CBH predictions for ALL 5 frames (not just center)
- Loss: L_total = L_mse(center_frame) + λ * L_temporal(smoothness)

Physics-Informed Constraint:
Cloud base height should not change drastically between consecutive frames
(temporal separation ~1 second). The temporal consistency loss penalizes
unrealistic jumps: L_temporal = mean(|pred_t - pred_{t-1}|)

Ablation Study:
Tests λ_temporal ∈ {0.05, 0.1, 0.2} to find optimal temporal regularization strength.

Validation: Stratified 5-Fold Cross-Validation (n_splits=5)
Target: R² > 0.50 (Sprint 5 WP-2 objective), with improved temporal stability

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
import json
from datetime import datetime

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
    save_report,
    print_fold_summary,
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
    Same as Task 2.1, but now we need CBH targets for all 5 frames.
    """

    def __init__(
        self,
        image_dataset: HDF5CloudDataset,
        indices: List[int],
        cbh_values: np.ndarray,
        n_frames: int = 5,
    ):
        self.image_dataset = image_dataset
        self.indices = indices
        self.cbh_values = cbh_values
        self.n_frames = n_frames
        self.temporal_offset = n_frames // 2

        if n_frames % 2 == 0:
            raise ValueError("n_frames must be odd")

        # Build mapping
        self.global_to_local = image_dataset.global_to_local

        # Get flight boundaries
        self.flight_boundaries = {}
        for global_idx, (flight_idx, local_idx) in enumerate(self.global_to_local):
            if flight_idx not in self.flight_boundaries:
                self.flight_boundaries[flight_idx] = [global_idx, global_idx]
            else:
                self.flight_boundaries[flight_idx][1] = global_idx

        for flight_idx in self.flight_boundaries:
            start, end = self.flight_boundaries[flight_idx]
            self.flight_boundaries[flight_idx] = (start, end + 1)

    def __len__(self):
        return len(self.indices)

    def _get_frame_sequence(self, center_idx: int) -> List[int]:
        """Get indices for temporal sequence centered on center_idx."""
        flight_idx, local_idx = self.global_to_local[center_idx]
        start_global, end_global = self.flight_boundaries[flight_idx]

        sequence_indices = []
        for offset in range(-self.temporal_offset, self.temporal_offset + 1):
            target_idx = center_idx + offset
            target_idx = max(start_global, min(end_global - 1, target_idx))
            sequence_indices.append(target_idx)

        return sequence_indices

    def __getitem__(self, idx):
        center_global_idx = self.indices[idx]
        sequence_indices = self._get_frame_sequence(center_global_idx)

        # Load all frames
        frames = []
        cbh_targets = []
        for frame_idx in sequence_indices:
            image, _, _, _, _, _ = self.image_dataset[frame_idx]

            if isinstance(image, torch.Tensor):
                image_tensor = image.float()
            else:
                image_tensor = torch.from_numpy(image).float()

            if image_tensor.dim() == 2:
                image_tensor = image_tensor.unsqueeze(0)

            frames.append(image_tensor)
            cbh_targets.append(self.cbh_values[frame_idx])

        frames_tensor = torch.stack(frames, dim=0)  # (N, C, H, W)
        cbh_targets_tensor = torch.tensor(cbh_targets).float()  # (N,)

        # Return: frames, all_targets, center_target
        center_cbh = cbh_targets[self.temporal_offset]

        return frames_tensor, cbh_targets_tensor, torch.tensor(center_cbh).float()


class TemporalAttention(nn.Module):
    """Temporal attention mechanism."""

    def __init__(self, feature_dim: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.feature_dim = feature_dim

        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, x):
        """
        Args:
            x: (B, T, D) frame features
        Returns:
            aggregated: (B, D) aggregated features
        """
        attn_out, attn_weights = self.attention(x, x, x)
        x = x + attn_out
        x = self.norm(x)
        aggregated = x.mean(dim=1)
        return aggregated


class TemporalConsistencyViT(nn.Module):
    """
    Temporal ViT with consistency loss (Task 2.2).

    Key difference from Task 2.1:
    - Predicts CBH for ALL 5 frames (not just center)
    - Uses temporal consistency regularization
    """

    def __init__(
        self,
        pretrained_model: str = "WinKawaks/vit-tiny-patch16-224",
        n_frames: int = 5,
    ):
        super().__init__()

        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library required")

        self.n_frames = n_frames

        print(f"Loading ViT frame encoder: {pretrained_model}")

        try:
            self.vit = ViTForImageClassification.from_pretrained(
                pretrained_model,
                num_labels=1,
                ignore_mismatched_sizes=True,
            )
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

        # Regression head for per-frame predictions
        self.frame_regression = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

        # Regression head for aggregated (center frame) prediction
        self.center_regression = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

        print(f"✓ Temporal Consistency ViT initialized")
        print(f"  Input: {n_frames} frames × 224×224")
        print(f"  Frame encoder: ViT-Tiny")
        print(f"  Outputs: {n_frames} per-frame predictions + 1 center prediction")
        print(f"  Temporal loss: penalizes |pred_t - pred_t-1|")

    def forward(self, x, predict_all_frames=True):
        """
        Forward pass.

        Args:
            x: (B, T, C, H, W) sequence of frames
            predict_all_frames: If True, return predictions for all frames

        Returns:
            If predict_all_frames=True:
                all_frame_preds: (B, T) predictions for all frames
                center_pred: (B, 1) aggregated prediction for center frame
            Else:
                center_pred: (B, 1) aggregated prediction only
        """
        B, T, C, H, W = x.shape

        # Duplicate grayscale to RGB
        if C == 1:
            x = x.repeat(1, 1, 3, 1, 1)

        # Resize to 224x224
        if H != 224 or W != 224:
            x = x.view(B * T, 3, H, W)
            x = torch.nn.functional.interpolate(
                x, size=(224, 224), mode="bilinear", align_corners=False
            )
            x = x.view(B, T, 3, 224, 224)

        # Process each frame
        x = x.view(B * T, 3, 224, 224)
        outputs = self.vit.vit(x)
        frame_features = outputs.last_hidden_state[:, 0, :]  # (B*T, D)
        frame_features = frame_features.view(B, T, self.feature_dim)

        if predict_all_frames:
            # Per-frame predictions
            frame_features_flat = frame_features.view(B * T, self.feature_dim)
            all_preds = self.frame_regression(frame_features_flat)  # (B*T, 1)
            all_preds = all_preds.view(B, T)  # (B, T)

        # Temporal attention for center frame
        aggregated = self.temporal_attention(frame_features)  # (B, D)
        center_pred = self.center_regression(aggregated)  # (B, 1)

        if predict_all_frames:
            return all_preds, center_pred
        else:
            return center_pred


class TemporalConsistencyTrainer:
    """Trainer with temporal consistency loss."""

    def __init__(
        self,
        config_path: str,
        output_dir: str = "sow_outputs/wp5",
        pretrained_model: str = "WinKawaks/vit-tiny-patch16-224",
        n_frames: int = 5,
        lambda_temporal: float = 0.1,
        device: str = None,
        verbose: bool = True,
    ):
        self.config_path = config_path
        self.output_dir = Path(output_dir)
        self.pretrained_model = pretrained_model
        self.n_frames = n_frames
        self.lambda_temporal = lambda_temporal
        self.verbose = verbose

        # Create subdirectories
        self.model_dir = self.output_dir / "models" / "temporal_consistency"
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

        self.image_dataset = HDF5CloudDataset(
            flight_configs=self.config["flights"],
            indices=None,
            augment=False,
            swath_slice=self.config.get("swath_slice", [40, 480]),
            temporal_frames=1,
        )

        if self.verbose:
            print("Caching CBH values...")
        self.cbh_values = self.image_dataset.get_unscaled_y()

        n_samples = len(self.image_dataset)

        if self.verbose:
            print(f"\n✓ Data loaded:")
            print(f"  Total samples: {n_samples}")
            print(f"  Temporal frames per sample: {self.n_frames}")
            print(f"  Lambda_temporal: {self.lambda_temporal}")

    def temporal_consistency_loss(self, predictions):
        """
        Compute temporal consistency loss.

        Args:
            predictions: (B, T) predictions for T frames

        Returns:
            loss: scalar temporal consistency loss
        """
        # L_temporal = mean(|pred_t - pred_{t-1}|)
        diffs = torch.abs(predictions[:, 1:] - predictions[:, :-1])
        return diffs.mean()

    def train_single_fold(
        self,
        fold_id: int,
        train_indices: List[int],
        val_indices: List[int],
        test_indices: List[int],
        n_epochs: int = 50,
        batch_size: int = 4,
        learning_rate: float = 2e-5,
        accumulation_steps: int = 4,
    ) -> FoldMetrics:
        """Train and evaluate a single fold."""

        print(f"\n{'=' * 80}")
        print(f"Fold {fold_id + 1}/5 (λ_temporal = {self.lambda_temporal})")
        print(f"{'=' * 80}")
        print(f"  Training:   {len(train_indices)} samples")
        print(f"  Validation: {len(val_indices)} samples")
        print(f"  Test:       {len(test_indices)} samples")

        # Create datasets
        train_dataset = TemporalDataset(
            self.image_dataset, train_indices, self.cbh_values, self.n_frames
        )
        val_dataset = TemporalDataset(
            self.image_dataset, val_indices, self.cbh_values, self.n_frames
        )
        test_dataset = TemporalDataset(
            self.image_dataset, test_indices, self.cbh_values, self.n_frames
        )

        # Data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
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
        model = TemporalConsistencyViT(
            pretrained_model=self.pretrained_model, n_frames=self.n_frames
        ).to(self.device)

        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        criterion_mse = nn.MSELoss()
        early_stopper = EarlyStopper(patience=10, min_delta=1e-4)

        best_val_loss = float("inf")
        best_epoch = 0
        best_model_state = None

        for epoch in range(n_epochs):
            # Train
            model.train()
            total_train_loss = 0.0
            total_mse_loss = 0.0
            total_temp_loss = 0.0
            n_train_batches = 0

            optimizer.zero_grad()

            for batch_idx, (frames, all_targets, center_target) in enumerate(
                train_loader
            ):
                frames = frames.to(self.device)
                all_targets = all_targets.to(self.device)
                center_target = center_target.to(self.device).unsqueeze(1)

                # Forward (predict all frames + center)
                all_preds, center_pred = model(frames, predict_all_frames=True)

                # MSE loss on center frame
                loss_mse = criterion_mse(center_pred, center_target)

                # Temporal consistency loss
                loss_temporal = self.temporal_consistency_loss(all_preds)

                # Total loss
                loss = loss_mse + self.lambda_temporal * loss_temporal
                loss = loss / accumulation_steps

                loss.backward()

                if (batch_idx + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                total_train_loss += loss.item() * accumulation_steps
                total_mse_loss += loss_mse.item()
                total_temp_loss += loss_temporal.item()
                n_train_batches += 1

            if n_train_batches % accumulation_steps != 0:
                optimizer.step()
                optimizer.zero_grad()

            train_loss = total_train_loss / n_train_batches
            train_mse = total_mse_loss / n_train_batches
            train_temp = total_temp_loss / n_train_batches

            # Validate
            model.eval()
            total_val_loss = 0.0
            n_val_batches = 0
            all_preds = []
            all_targets = []

            with torch.no_grad():
                for frames, _, center_target in val_loader:
                    frames = frames.to(self.device)
                    center_target = center_target.to(self.device).unsqueeze(1)

                    center_pred = model(frames, predict_all_frames=False)

                    loss = criterion_mse(center_pred, center_target)
                    total_val_loss += loss.item()
                    n_val_batches += 1

                    all_preds.append(center_pred.cpu().numpy())
                    all_targets.append(center_target.cpu().numpy())

            val_loss = total_val_loss / n_val_batches
            preds = np.concatenate(all_preds, axis=0).flatten()
            targets = np.concatenate(all_targets, axis=0).flatten()
            val_metrics = compute_metrics(preds, targets)

            # Print progress
            if self.verbose and (epoch + 1) % 5 == 0:
                print(
                    f"  Epoch {epoch + 1:3d}/{n_epochs}: "
                    f"MSE={train_mse:.4f}, Temp={train_temp:.4f}, "
                    f"Val Loss={val_loss:.4f}, Val R²={val_metrics['r2']:.4f}"
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
            for frames, _, center_target in test_loader:
                frames = frames.to(self.device)
                center_target = center_target.to(self.device).unsqueeze(1)

                center_pred = model(frames, predict_all_frames=False)

                all_preds.append(center_pred.cpu().numpy())
                all_targets.append(center_target.cpu().numpy())

        test_preds = np.concatenate(all_preds, axis=0).flatten()
        test_targets = np.concatenate(all_targets, axis=0).flatten()
        test_metrics = compute_metrics(test_preds, test_targets)

        # Save model
        model_path = (
            self.model_dir
            / f"temporal_consistency_lambda{self.lambda_temporal}_fold{fold_id}.pth"
        )
        torch.save(best_model_state, model_path)
        if self.verbose:
            print(f"  ✓ Model saved to {model_path}")

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

        if self.verbose:
            print_fold_summary(fold_id, 5, fold_metrics)

        return fold_metrics

    def run_kfold_cv(
        self,
        n_epochs: int = 50,
        batch_size: int = 4,
        learning_rate: float = 2e-5,
        accumulation_steps: int = 4,
    ) -> List[FoldMetrics]:
        """Run Stratified 5-Fold CV."""

        print("\n" + "=" * 80)
        print(f"STRATIFIED 5-FOLD CV: Temporal Consistency (λ={self.lambda_temporal})")
        print("=" * 80)

        folds = get_stratified_folds(self.cbh_values, n_splits=5, random_state=42)
        fold_results = []

        for fold_id, (train_val_idx, test_idx) in enumerate(folds):
            np.random.seed(42 + fold_id)
            np.random.shuffle(train_val_idx)
            n_train = int(0.8 * len(train_val_idx))
            train_indices = train_val_idx[:n_train].tolist()
            val_indices = train_val_idx[n_train:].tolist()
            test_indices = test_idx.tolist()

            fold_metrics = self.train_single_fold(
                fold_id,
                train_indices,
                val_indices,
                test_indices,
                n_epochs,
                batch_size,
                learning_rate,
                accumulation_steps,
            )

            fold_results.append(fold_metrics)

        return fold_results


def main():
    parser = argparse.ArgumentParser(
        description="WP-5 Task 2.2: Temporal Consistency Loss"
    )
    parser.add_argument("--config", type=str, default="configs/bestComboConfig.yaml")
    parser.add_argument("--output-dir", type=str, default="sow_outputs/wp5")
    parser.add_argument(
        "--pretrained-model", type=str, default="WinKawaks/vit-tiny-patch16-224"
    )
    parser.add_argument("--n-frames", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument(
        "--lambda-values",
        type=str,
        default="0.05,0.1,0.2",
        help="Comma-separated lambda values for ablation",
    )
    parser.add_argument("--verbose", action="store_true", default=True)

    args = parser.parse_args()

    print("=" * 80)
    print("WP-5 TASK 2.2: Temporal Consistency Regularization - Ablation Study")
    print("=" * 80)

    # Parse lambda values
    lambda_values = [float(x) for x in args.lambda_values.split(",")]
    print(f"\nAblation study with λ_temporal ∈ {lambda_values}")

    all_results = {}

    for lambda_val in lambda_values:
        print(f"\n{'#' * 80}")
        print(f"# ABLATION: λ_temporal = {lambda_val}")
        print(f"{'#' * 80}")

        trainer = TemporalConsistencyTrainer(
            config_path=args.config,
            output_dir=args.output_dir,
            pretrained_model=args.pretrained_model,
            n_frames=args.n_frames,
            lambda_temporal=lambda_val,
            verbose=args.verbose,
        )

        trainer.load_data()

        fold_results = trainer.run_kfold_cv(
            n_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            accumulation_steps=args.accumulation_steps,
        )

        # Aggregate metrics
        r2_values = [f.r2 for f in fold_results]
        mae_values = [f.mae_km for f in fold_results]
        rmse_values = [f.rmse_km for f in fold_results]

        result_summary = {
            "lambda_temporal": lambda_val,
            "aggregate_metrics": {
                "mean_r2": float(np.mean(r2_values)),
                "std_r2": float(np.std(r2_values)),
                "mean_mae_km": float(np.mean(mae_values)),
                "std_mae_km": float(np.std(mae_values)),
                "mean_rmse_km": float(np.mean(rmse_values)),
                "std_rmse_km": float(np.std(rmse_values)),
            },
            "per_fold_results": [
                {
                    "fold_id": f.fold_id,
                    "r2": f.r2,
                    "mae_km": f.mae_km,
                    "rmse_km": f.rmse_km,
                    "best_epoch": f.best_epoch,
                }
                for f in fold_results
            ],
        }

        all_results[f"lambda_{lambda_val}"] = result_summary

        print(f"\n{'=' * 80}")
        print(f"RESULTS: λ_temporal = {lambda_val}")
        print(f"{'=' * 80}")
        print(
            f"  Mean R²:   {result_summary['aggregate_metrics']['mean_r2']:.4f} ± {result_summary['aggregate_metrics']['std_r2']:.4f}"
        )
        print(
            f"  Mean MAE:  {result_summary['aggregate_metrics']['mean_mae_km']:.4f} ± {result_summary['aggregate_metrics']['std_mae_km']:.4f} km"
        )
        print(
            f"  Mean RMSE: {result_summary['aggregate_metrics']['mean_rmse_km']:.4f} ± {result_summary['aggregate_metrics']['std_rmse_km']:.4f} km"
        )

    # Generate final ablation report
    final_report = {
        "model_name": "Temporal ViT with Consistency Loss (Ablation Study)",
        "sprint_work_package": "WP-2: Temporal Modeling (Task 2.2)",
        "script_path": "sow_outputs/wp5/wp5_temporal_consistency.py",
        "baseline_to_beat": {
            "model": "Physical GBDT (Real ERA5)",
            "r2": 0.668,
            "mae_km": 0.137,
        },
        "validation_protocol": {
            "name": "Stratified 5-Fold Cross-Validation",
            "n_folds": 5,
            "stratify_by": "CBH Target",
            "forbidden_protocol": "Leave-One-Flight-Out (LOO) CV",
            "justification": "LOO CV failed (R2=-3.13) on F4 due to extreme domain shift.",
        },
        "ablation_results": all_results,
        "hardware_config": {
            "gpu": "NVIDIA GTX 1070 Ti",
            "vram": "8 GB",
            "vram_mitigation_used": f"gradient_accumulation_steps: {args.accumulation_steps}, batch_size: {args.batch_size}",
        },
        "architecture": {
            "frame_encoder": f"ViT-Tiny ({args.pretrained_model})",
            "temporal_frames": args.n_frames,
            "temporal_aggregation": "Multi-head Attention (4 heads)",
            "loss_function": "L_total = L_mse(center) + λ * L_temporal",
            "temporal_loss": "L_temporal = mean(|pred_t - pred_{t-1}|)",
            "lambda_values_tested": lambda_values,
        },
        "timestamp": datetime.now().isoformat(),
    }

    # Save report
    report_path = Path(args.output_dir) / "reports" / "WP5_Temporal_Loss_Ablation.json"
    save_report(final_report, report_path)

    # Print comparison
    print("\n" + "=" * 80)
    print("ABLATION STUDY SUMMARY")
    print("=" * 80)
    print(f"\n{'Lambda':<10} {'R²':<15} {'MAE (km)':<15} {'RMSE (km)':<15}")
    print("-" * 80)
    for lambda_val in lambda_values:
        result = all_results[f"lambda_{lambda_val}"]
        metrics = result["aggregate_metrics"]
        print(
            f"{lambda_val:<10.2f} "
            f"{metrics['mean_r2']:.4f} ± {metrics['std_r2']:.4f}   "
            f"{metrics['mean_mae_km']:.4f} ± {metrics['std_mae_km']:.4f}   "
            f"{metrics['mean_rmse_km']:.4f} ± {metrics['std_rmse_km']:.4f}"
        )

    # Find best lambda
    best_lambda = min(
        lambda_values,
        key=lambda l: all_results[f"lambda_{l}"]["aggregate_metrics"]["mean_mae_km"],
    )
    best_result = all_results[f"lambda_{best_lambda}"]

    print("\n" + "=" * 80)
    print(f"BEST LAMBDA: {best_lambda}")
    print("=" * 80)
    print(f"  R²:   {best_result['aggregate_metrics']['mean_r2']:.4f}")
    print(
        f"  MAE:  {best_result['aggregate_metrics']['mean_mae_km']:.4f} km ({best_result['aggregate_metrics']['mean_mae_km'] * 1000:.1f} m)"
    )
    print(f"  RMSE: {best_result['aggregate_metrics']['mean_rmse_km']:.4f} km")

    baseline_r2 = 0.668
    baseline_mae = 0.137
    if (
        best_result["aggregate_metrics"]["mean_r2"] > baseline_r2
        and best_result["aggregate_metrics"]["mean_mae_km"] < baseline_mae
    ):
        print("\n✓ SUCCESS: Best configuration beats baseline!")
    else:
        print("\n⚠ Best configuration does not beat baseline on all metrics")

    print(f"\n✓ Task 2.2 Complete - Report saved to {report_path}")


if __name__ == "__main__":
    main()
