#!/usr/bin/env python3
"""
Training Script for Spatial MAE Variants
=========================================

This script trains the three spatial MAE variants (pooling, CNN, attention)
for cloud base height estimation using Leave-One-Out cross-validation.

Week 1, Task 1.1: Spatial Feature Extraction

Usage:
    python scripts/train_spatial_mae.py --variant pooling --config configs/ssl_finetune_cbh.yaml
    python scripts/train_spatial_mae.py --variant cnn --config configs/ssl_finetune_cbh.yaml
    python scripts/train_spatial_mae.py --variant attention --config configs/ssl_finetune_cbh.yaml
    python scripts/train_spatial_mae.py --all  # Train all variants

Author: Cloud ML Research Team
Date: 2024
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import yaml
from tqdm import tqdm
import json
from datetime import datetime
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# Import project modules
from src.hdf5_dataset import HDF5CloudDataset
from src.models.spatial_mae import SpatialMAE, build_spatial_mae
from src.split_utils import stratified_split_by_flight, get_flight_labels
from src.mae_model import build_mae_small


class SpatialMAETrainer:
    """Trainer for spatial MAE variants."""

    def __init__(
        self,
        variant: str,
        config_path: str,
        encoder_path: str,
        output_dir: str = "outputs/spatial_mae",
        device: str = "cuda",
    ):
        """
        Initialize trainer.

        Args:
            variant: Spatial head variant ("pooling", "cnn", "attention")
            config_path: Path to config file
            encoder_path: Path to pretrained MAE encoder
            output_dir: Directory for outputs
            device: Device for training
        """
        self.variant = variant
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.config = self.load_config(config_path)
        self.encoder_path = encoder_path

        # Create output directory
        self.output_dir = Path(output_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_dir / f"{variant}_{timestamp}"
        self.run_dir.mkdir(exist_ok=True, parents=True)

        print("=" * 80)
        print(f"SPATIAL MAE TRAINING: Variant {variant.upper()}")
        print("=" * 80)
        print(f"Device: {self.device}")
        print(f"Encoder: {encoder_path}")
        print(f"Output: {self.run_dir}")
        print()

        # Results storage
        self.loo_results = []
        self.best_models = {}

    def load_config(self, config_path):
        """Load YAML configuration."""
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def load_dataset(self):
        """Load the full labeled dataset."""
        data_config = self.config["data"]

        # Load dataset using flight configs (not pre-built HDF5)
        dataset = HDF5CloudDataset(
            flight_configs=data_config["flights"],
            swath_slice=data_config.get("swath_slice", [40, 480]),
            temporal_frames=data_config.get("temporal_frames", 1),
            filter_type=data_config.get("filter_type", "basic"),
            flat_field_correction=data_config.get("flat_field_correction", True),
            clahe_clip_limit=data_config.get("clahe_clip_limit", 0.01),
            zscore_normalize=data_config.get("zscore_normalize", True),
            angles_mode=data_config.get("angles_mode", "both"),
        )
        return dataset

    def create_dataloaders(self, train_indices, val_indices, batch_size=16):
        """Create train and validation dataloaders."""
        dataset = self.load_dataset()

        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)

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

        return train_loader, val_loader

    def build_model(self, freeze_encoder=True):
        """Build spatial MAE model."""
        # Check if encoder exists
        if not Path(self.encoder_path).exists():
            print(f"Warning: Encoder not found at {self.encoder_path}")
            print("Creating model with random initialization...")
            mae = build_mae_small()
            model = SpatialMAE(encoder=mae.encoder, head_type=self.variant)
        else:
            model = build_spatial_mae(
                encoder_path=self.encoder_path,
                head_type=self.variant,
                freeze_encoder=freeze_encoder,
                device=self.device,
            )

        return model

    def train_epoch(self, model, train_loader, optimizer, criterion, scaler_y):
        """Train for one epoch."""
        model.train()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        pbar = tqdm(train_loader, desc="Training")
        for batch in pbar:
            # Unpack batch from HDF5CloudDataset
            # Returns: (img, sza, saa, y, global_idx, local_idx)
            images, sza, saa, targets, _, _ = batch

            # Move to device
            images = images.to(self.device)
            targets = targets.to(self.device).float()

            # Flatten images to 1D if needed
            # HDF5CloudDataset returns (B, temporal_frames, H, W)
            if len(images.shape) == 4:
                # Shape: (B, 1, H, W) -> (B, 1, W) by taking mean over H
                images = images.squeeze(1).mean(dim=1, keepdim=True)

            # Forward pass
            optimizer.zero_grad()
            predictions = model(images)

            # Reshape predictions and targets to match
            predictions = predictions.squeeze(-1)
            targets = targets.squeeze(-1) if targets.dim() > 1 else targets

            # Compute loss
            loss = criterion(predictions, targets)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Track metrics
            total_loss += loss.item()
            all_preds.extend(predictions.detach().cpu().numpy())
            all_targets.extend(targets.detach().cpu().numpy())

            # Update progress bar
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Compute epoch metrics
        avg_loss = total_loss / len(train_loader)
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)

        # Inverse transform predictions and targets
        all_preds_scaled = scaler_y.inverse_transform(
            all_preds.reshape(-1, 1)
        ).flatten()
        all_targets_scaled = scaler_y.inverse_transform(
            all_targets.reshape(-1, 1)
        ).flatten()

        r2 = r2_score(all_targets_scaled, all_preds_scaled)
        mae = mean_absolute_error(all_targets_scaled, all_preds_scaled)
        rmse = np.sqrt(mean_squared_error(all_targets_scaled, all_preds_scaled))

        return avg_loss, r2, mae, rmse

    def validate(self, model, val_loader, criterion, scaler_y):
        """Validate model."""
        model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                # Unpack batch from HDF5CloudDataset
                # Returns: (img, sza, saa, y, global_idx, local_idx)
                images, sza, saa, targets, _, _ = batch

                # Move to device
                images = images.to(self.device)
                targets = targets.to(self.device).float()

                # Flatten images to 1D if needed
                # HDF5CloudDataset returns (B, temporal_frames, H, W)
                if len(images.shape) == 4:
                    # Shape: (B, 1, H, W) -> (B, 1, W) by taking mean over H
                    images = images.squeeze(1).mean(dim=1, keepdim=True)

                # Forward pass
                predictions = model(images)
                predictions = predictions.squeeze(-1)
                targets = targets.squeeze(-1) if targets.dim() > 1 else targets

                # Compute loss
                loss = criterion(predictions, targets)

                # Track metrics
                total_loss += loss.item()
                all_preds.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        # Compute metrics
        avg_loss = total_loss / len(val_loader)
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)

        # Inverse transform
        all_preds_scaled = scaler_y.inverse_transform(
            all_preds.reshape(-1, 1)
        ).flatten()
        all_targets_scaled = scaler_y.inverse_transform(
            all_targets.reshape(-1, 1)
        ).flatten()

        r2 = r2_score(all_targets_scaled, all_preds_scaled)
        mae = mean_absolute_error(all_targets_scaled, all_preds_scaled)
        rmse = np.sqrt(mean_squared_error(all_targets_scaled, all_preds_scaled))

        return avg_loss, r2, mae, rmse, all_preds_scaled, all_targets_scaled

    def train_fold(self, fold_idx, train_indices, val_indices, epochs=50):
        """Train one fold of LOO CV."""
        print(f"\n{'=' * 80}")
        print(f"FOLD {fold_idx + 1}")
        print(f"{'=' * 80}")
        print(f"Train samples: {len(train_indices)}")
        print(f"Val samples: {len(val_indices)}")

        # Create dataloaders
        train_loader, val_loader = self.create_dataloaders(
            train_indices, val_indices, batch_size=16
        )

        # Get scaler for this fold
        dataset = self.load_dataset()
        # Dataset returns tuple: (img, sza, saa, y, global_idx, local_idx)
        train_cbh = [dataset[i][3].item() for i in train_indices]  # y is at index 3
        from sklearn.preprocessing import StandardScaler

        scaler_y = StandardScaler()
        scaler_y.fit(np.array(train_cbh).reshape(-1, 1))

        # Build model
        model = self.build_model(freeze_encoder=True)
        model = model.to(self.device)

        # Setup training
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=1e-3,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )

        # Training loop
        best_val_loss = float("inf")
        best_val_r2 = -float("inf")
        patience_counter = 0
        patience = 10

        history = {
            "train_loss": [],
            "val_loss": [],
            "train_r2": [],
            "val_r2": [],
        }

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")

            # Train
            train_loss, train_r2, train_mae, train_rmse = self.train_epoch(
                model, train_loader, optimizer, criterion, scaler_y
            )

            # Validate
            val_loss, val_r2, val_mae, val_rmse, val_preds, val_targets = self.validate(
                model, val_loader, criterion, scaler_y
            )

            # Update scheduler
            scheduler.step(val_loss)

            # Log metrics
            print(
                f"Train Loss: {train_loss:.4f}, R²: {train_r2:.4f}, MAE: {train_mae:.1f}m"
            )
            print(f"Val   Loss: {val_loss:.4f}, R²: {val_r2:.4f}, MAE: {val_mae:.1f}m")

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_r2"].append(train_r2)
            history["val_r2"].append(val_r2)

            # Save best model
            if val_r2 > best_val_r2:
                best_val_r2 = val_r2
                best_val_loss = val_loss
                patience_counter = 0

                # Save checkpoint
                checkpoint_path = self.run_dir / f"best_model_fold{fold_idx}.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_r2": val_r2,
                        "val_mae": val_mae,
                        "val_rmse": val_rmse,
                        "scaler_y": scaler_y,
                    },
                    checkpoint_path,
                )
                print(f" Saved best model (R²={val_r2:.4f})")
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        # Load best model for final evaluation
        checkpoint = torch.load(
            self.run_dir / f"best_model_fold{fold_idx}.pt", weights_only=False
        )
        model.load_state_dict(checkpoint["model_state_dict"])

        # Final validation
        val_loss, val_r2, val_mae, val_rmse, val_preds, val_targets = self.validate(
            model, val_loader, criterion, scaler_y
        )

        return {
            "fold": fold_idx,
            "val_r2": val_r2,
            "val_mae": val_mae,
            "val_rmse": val_rmse,
            "predictions": val_preds,
            "targets": val_targets,
            "history": history,
        }

    def run_loo_cv(self, epochs=50):
        """Run Leave-One-Out cross-validation."""
        print("\n" + "=" * 80)
        print("LEAVE-ONE-OUT CROSS-VALIDATION")
        print("=" * 80)

        # Load dataset and get flight labels
        dataset = self.load_dataset()
        flight_labels = get_flight_labels(dataset)
        unique_flights = sorted(set(flight_labels))

        # Map flight indices to flight names
        flight_configs = self.config["data"]["flights"]
        flight_names = {
            i: flight_configs[i]["name"] for i in range(len(flight_configs))
        }

        print(f"Total samples: {len(dataset)}")
        print(f"Unique flights: {len(unique_flights)}")
        print(f"Flights: {[flight_names[f] for f in unique_flights]}")
        print()

        # Run LOO CV
        all_results = []

        for fold_idx, test_flight in enumerate(unique_flights):
            test_flight_name = flight_names[test_flight]
            print(f"\n{'#' * 80}")
            print(
                f"# FOLD {fold_idx + 1}/{len(unique_flights)}: Holding out {test_flight_name}"
            )
            print(f"{'#' * 80}")

            # Split data
            train_indices = [i for i, f in enumerate(flight_labels) if f != test_flight]
            val_indices = [i for i, f in enumerate(flight_labels) if f == test_flight]

            # Train fold
            fold_result = self.train_fold(fold_idx, train_indices, val_indices, epochs)
            fold_result["test_flight"] = test_flight_name
            fold_result["test_flight_idx"] = int(test_flight)
            all_results.append(fold_result)

            # Print fold summary
            print(f"\n{'=' * 80}")
            print(f"FOLD {fold_idx + 1} SUMMARY: {test_flight_name}")
            print(f"{'=' * 80}")
            print(f"R²:   {fold_result['val_r2']:.4f}")
            print(f"MAE:  {fold_result['val_mae']:.1f} m")
            print(f"RMSE: {fold_result['val_rmse']:.1f} m")
            print()

        # Aggregate results
        self.loo_results = all_results
        self.save_results()
        self.plot_results()

        return all_results

    def save_results(self):
        """Save LOO CV results to JSON."""
        results_summary = {
            "variant": self.variant,
            "timestamp": datetime.now().isoformat(),
            "folds": [],
            "aggregate": {},
        }

        # Per-fold results
        all_r2 = []
        all_mae = []
        all_rmse = []

        for result in self.loo_results:
            fold_data = {
                "fold": int(result["fold"]),
                "test_flight": result["test_flight"],
                "test_flight_idx": result["test_flight_idx"],
                "r2": float(result["val_r2"]),
                "mae": float(result["val_mae"]),
                "rmse": float(result["val_rmse"]),
            }
            results_summary["folds"].append(fold_data)

            all_r2.append(result["val_r2"])
            all_mae.append(result["val_mae"])
            all_rmse.append(result["val_rmse"])

        # Aggregate statistics
        results_summary["aggregate"] = {
            "mean_r2": float(np.mean(all_r2)),
            "std_r2": float(np.std(all_r2)),
            "mean_mae": float(np.mean(all_mae)),
            "std_mae": float(np.std(all_mae)),
            "mean_rmse": float(np.mean(all_rmse)),
            "std_rmse": float(np.std(all_rmse)),
        }

        # Save to JSON
        results_path = self.run_dir / "loo_results.json"
        with open(results_path, "w") as f:
            json.dump(results_summary, f, indent=2)

        print(f"\n Results saved to {results_path}")

        # Print summary
        print("\n" + "=" * 80)
        print("AGGREGATE LOO CV RESULTS")
        print("=" * 80)
        print(
            f"Mean R²:   {results_summary['aggregate']['mean_r2']:.4f} ± {results_summary['aggregate']['std_r2']:.4f}"
        )
        print(
            f"Mean MAE:  {results_summary['aggregate']['mean_mae']:.1f} ± {results_summary['aggregate']['std_mae']:.1f} m"
        )
        print(
            f"Mean RMSE: {results_summary['aggregate']['mean_rmse']:.1f} ± {results_summary['aggregate']['std_rmse']:.1f} m"
        )
        print()

    def plot_results(self):
        """Plot LOO CV results."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Extract data
        flights = [r["test_flight"] for r in self.loo_results]
        r2_scores = [r["val_r2"] for r in self.loo_results]
        mae_scores = [r["val_mae"] for r in self.loo_results]

        # Plot 1: R² per fold
        ax = axes[0, 0]
        ax.bar(flights, r2_scores)
        ax.axhline(0, color="red", linestyle="--", label="Baseline (mean)")
        ax.set_xlabel("Test Flight")
        ax.set_ylabel("R²")
        ax.set_title(f"R² per Fold - {self.variant.upper()}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: MAE per fold
        ax = axes[0, 1]
        ax.bar(flights, mae_scores, color="orange")
        ax.set_xlabel("Test Flight")
        ax.set_ylabel("MAE (m)")
        ax.set_title(f"MAE per Fold - {self.variant.upper()}")
        ax.grid(True, alpha=0.3)

        # Plot 3: Predictions vs Targets (all folds combined)
        ax = axes[1, 0]
        for result in self.loo_results:
            ax.scatter(
                result["targets"],
                result["predictions"],
                alpha=0.5,
                label=result["test_flight"],
                s=20,
            )
        # Perfect prediction line
        all_targets = np.concatenate([r["targets"] for r in self.loo_results])
        min_val, max_val = all_targets.min(), all_targets.max()
        ax.plot([min_val, max_val], [min_val, max_val], "k--", label="Perfect")
        ax.set_xlabel("True CBH (m)")
        ax.set_ylabel("Predicted CBH (m)")
        ax.set_title("Predictions vs Targets (All Folds)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Plot 4: Training history (first fold as example)
        ax = axes[1, 1]
        history = self.loo_results[0]["history"]
        epochs = range(1, len(history["train_r2"]) + 1)
        ax.plot(epochs, history["train_r2"], label="Train R²", marker="o", markersize=3)
        ax.plot(epochs, history["val_r2"], label="Val R²", marker="s", markersize=3)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("R²")
        ax.set_title(f"Training History (Fold 1: {self.loo_results[0]['test_flight']})")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = self.run_dir / "loo_results.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f" Plots saved to {plot_path}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Train Spatial MAE variants for CBH estimation"
    )
    parser.add_argument(
        "--variant",
        type=str,
        choices=["pooling", "cnn", "attention", "all"],
        default="pooling",
        help="Spatial head variant to train",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/ssl_finetune_cbh.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="models/mae_pretrained.pt",
        help="Path to pretrained MAE encoder",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of epochs per fold",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/spatial_mae",
        help="Output directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for training (cuda or cpu)",
    )

    args = parser.parse_args()

    # Determine which variants to train
    if args.variant == "all":
        variants = ["pooling", "cnn", "attention"]
    else:
        variants = [args.variant]

    # Train each variant
    all_results = {}
    for variant in variants:
        print("\n" + "=" * 80)
        print(f"TRAINING VARIANT: {variant.upper()}")
        print("=" * 80)

        trainer = SpatialMAETrainer(
            variant=variant,
            config_path=args.config,
            encoder_path=args.encoder,
            output_dir=args.output,
            device=args.device,
        )

        results = trainer.run_loo_cv(epochs=args.epochs)
        all_results[variant] = results

    # Compare variants if multiple were trained
    if len(variants) > 1:
        print("\n" + "=" * 80)
        print("COMPARISON OF VARIANTS")
        print("=" * 80)

        for variant in variants:
            results = all_results[variant]
            mean_r2 = np.mean([r["val_r2"] for r in results])
            mean_mae = np.mean([r["val_mae"] for r in results])
            print(f"{variant:10s}: R²={mean_r2:6.4f}, MAE={mean_mae:6.1f}m")

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
