#!/usr/bin/env python3
"""
Leave-One-Out (LOO) Per-Flight Cross-Validation
================================================

This script validates the hybrid MAE+GBDT model using leave-one-out cross-validation
at the flight level. For each of the 5 flights, we:

1. Train on 4 flights
2. Test on the remaining flight
3. Report R², MAE, RMSE per fold

This ensures the model generalizes across different flights/conditions and isn't
overfitting to specific flight characteristics.

Author: CloudML Validation Suite
Date: 2024
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
from datetime import datetime
import pandas as pd

# Import project modules
from src.hdf5_dataset import HDF5CloudDataset
from src.mae_model import MAEEncoder


class LOOValidator:
    """
    Leave-One-Out per-flight cross-validation for hybrid MAE+GBDT.
    """

    def __init__(self, config_path, encoder_path, device="cuda"):
        """
        Initialize the LOO validator.

        Args:
            config_path: Path to SSL fine-tuning config
            encoder_path: Path to pre-trained MAE encoder weights
            device: Device to run inference on
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.config = self.load_config(config_path)
        self.encoder_path = encoder_path

        print("=" * 80)
        print("LEAVE-ONE-OUT PER-FLIGHT CROSS-VALIDATION")
        print("=" * 80)
        print(f"Device: {self.device}")
        print(f"Encoder: {encoder_path}")
        print(f"Config: {config_path}")
        print()

        # Create output directory
        self.output_dir = Path("outputs/loo_validation")
        self.output_dir.mkdir(exist_ok=True, parents=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_dir / timestamp
        self.run_dir.mkdir(exist_ok=True, parents=True)

        print(f"Output directory: {self.run_dir}")
        print()

    def load_config(self, config_path):
        """Load YAML configuration."""
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def load_encoder(self):
        """Load pre-trained MAE encoder."""
        print("Loading pre-trained MAE encoder...")

        model_config = self.config["model"]
        encoder = MAEEncoder(
            img_width=model_config["img_width"],
            patch_size=model_config["patch_size"],
            embed_dim=model_config["embed_dim"],
            depth=model_config["depth"],
            num_heads=model_config["num_heads"],
            mlp_ratio=model_config["mlp_ratio"],
        )

        checkpoint = torch.load(
            self.encoder_path, map_location=self.device, weights_only=False
        )
        encoder.load_state_dict(checkpoint)
        encoder.to(self.device)
        encoder.eval()

        print(f"✓ Encoder loaded successfully")
        print(f"  Embedding dimension: {model_config['embed_dim']}")
        print()

        return encoder

    def load_flight_datasets(self):
        """
        Load datasets for each flight with UNIFIED scaler across all flights.

        Returns:
            flight_datasets: List of (flight_name, dataset, indices) tuples
            unified_scalers: Dict with shared SZA, SAA, Y scalers
        """
        print("Loading per-flight datasets with unified scaler...")

        # STEP 1: Create a combined dataset to fit unified scalers
        print("  Step 1: Fitting unified scalers on all flights combined...")
        unified_dataset = HDF5CloudDataset(
            flight_configs=self.config["data"]["flights"],
            swath_slice=self.config["data"]["swath_slice"],
            temporal_frames=self.config["data"]["temporal_frames"],
            filter_type=self.config["data"]["filter_type"],
            cbh_min=self.config["data"]["cbh_min"],
            cbh_max=self.config["data"]["cbh_max"],
            flat_field_correction=self.config["data"]["flat_field_correction"],
            clahe_clip_limit=self.config["data"]["clahe_clip_limit"],
            zscore_normalize=self.config["data"]["zscore_normalize"],
            angles_mode=self.config["data"]["angles_mode"],
            augment=False,
        )

        # Extract the unified scalers
        unified_scalers = {
            "sza_scaler": unified_dataset.sza_scaler,
            "saa_scaler": unified_dataset.saa_scaler,
            "y_scaler": unified_dataset.y_scaler,
        }

        print(f"  ✓ Unified scalers fitted on {len(unified_dataset)} total samples")
        print(
            f"    SZA: mean={unified_dataset.sza_scaler.mean_[0]:.3f}, std={unified_dataset.sza_scaler.scale_[0]:.3f}"
        )
        print(
            f"    SAA: mean={unified_dataset.saa_scaler.mean_[0]:.3f}, std={unified_dataset.saa_scaler.scale_[0]:.3f}"
        )
        print(
            f"    Y:   mean={unified_dataset.y_scaler.mean_[0]:.3f}, std={unified_dataset.y_scaler.scale_[0]:.3f}"
        )
        print()

        # STEP 2: Create individual flight datasets with shared scalers
        print("  Step 2: Loading individual flights with unified scalers...")
        flight_datasets = []
        flights = self.config["data"]["flights"]

        for flight_config in flights:
            flight_name = flight_config["name"]

            # Create dataset with single flight but SHARED scalers
            dataset = HDF5CloudDataset(
                flight_configs=[flight_config],
                swath_slice=self.config["data"]["swath_slice"],
                temporal_frames=self.config["data"]["temporal_frames"],
                filter_type=self.config["data"]["filter_type"],
                cbh_min=self.config["data"]["cbh_min"],
                cbh_max=self.config["data"]["cbh_max"],
                flat_field_correction=self.config["data"]["flat_field_correction"],
                clahe_clip_limit=self.config["data"]["clahe_clip_limit"],
                zscore_normalize=self.config["data"]["zscore_normalize"],
                angles_mode=self.config["data"]["angles_mode"],
                augment=False,
                # CRITICAL: Pass in the unified scalers
                sza_scaler=unified_scalers["sza_scaler"],
                saa_scaler=unified_scalers["saa_scaler"],
                y_scaler=unified_scalers["y_scaler"],
            )

            n_samples = len(dataset)
            indices = list(range(n_samples))

            flight_datasets.append((flight_name, dataset, indices))
            print(f"    {flight_name}: {n_samples} samples")

        print(f"\n  Total flights: {len(flight_datasets)}")
        print(f"  All flights now use the same scaler ✓")
        print()

        return flight_datasets, unified_scalers

    def extract_embeddings(self, encoder, dataset, indices, batch_size=64):
        """
        Extract MAE embeddings for a dataset.

        Args:
            encoder: Pre-trained MAE encoder
            dataset: Dataset to extract from
            indices: List of indices to use
            batch_size: Batch size for extraction

        Returns:
            embeddings: (N, embed_dim) numpy array
            angles: (N, 2) numpy array [SZA, SAA]
            targets: (N,) numpy array (CBH values in km)
        """
        subset = Subset(dataset, indices)
        dataloader = DataLoader(
            subset, batch_size=batch_size, shuffle=False, num_workers=4
        )

        embeddings_list = []
        angles_list = []
        targets_list = []

        # Get y_scaler from dataset for inverse transform
        y_scaler = dataset.y_scaler

        with torch.no_grad():
            for batch in dataloader:
                img_stack, sza, saa, y_scaled, global_idx, local_idx = batch

                # Flatten image to 1D signal
                img = img_stack.mean(dim=1)  # Average temporal frames
                img = img.flatten(start_dim=1).unsqueeze(1)  # (batch, 1, 440)

                # Move to device
                img = img.to(self.device)

                # Forward pass through encoder
                encoded = encoder(img)

                # Extract CLS token
                cls_embeddings = encoded[:, 0, :]

                # Store embeddings
                embeddings_list.append(cls_embeddings.cpu().numpy())

                # Stack and squeeze angles
                angles_batch = torch.stack([sza, saa], dim=1).cpu().numpy()
                if angles_batch.ndim == 3:
                    angles_batch = angles_batch.squeeze(-1)
                angles_list.append(angles_batch)

                # Convert y_scaled back to km using inverse transform
                y_scaled_np = y_scaled.cpu().numpy().reshape(-1, 1)
                y_unscaled = y_scaler.inverse_transform(y_scaled_np).flatten()
                targets_list.append(y_unscaled)

        # Concatenate all batches
        embeddings = np.concatenate(embeddings_list, axis=0)
        angles = np.concatenate(angles_list, axis=0)
        targets = np.concatenate(targets_list, axis=0)

        return embeddings, angles, targets

    def create_hybrid_features(self, embeddings, angles):
        """Combine MAE embeddings with angle features."""
        return np.concatenate([embeddings, angles], axis=1)

    def train_gbdt(self, X_train, y_train):
        """
        Train GradientBoosting regressor.

        Args:
            X_train: Training features
            y_train: Training targets

        Returns:
            model: Trained GBDT model
            scaler: Fitted StandardScaler
        """
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # Train GBDT with default params (no tuning for LOO to save time)
        model = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=2,
            subsample=0.8,
            random_state=42,
            verbose=0,
        )

        model.fit(X_train_scaled, y_train)

        return model, scaler

    def evaluate(self, model, scaler, X_test, y_test):
        """
        Evaluate model on test set.

        Args:
            model: Trained model
            scaler: Fitted scaler
            X_test: Test features
            y_test: Test targets

        Returns:
            metrics: Dict of evaluation metrics
            predictions: Model predictions
        """
        # Scale features
        X_test_scaled = scaler.transform(X_test)

        # Predict
        y_pred = model.predict(X_test_scaled)

        # Compute metrics
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        metrics = {
            "r2": float(r2),
            "mae": float(mae),
            "rmse": float(rmse),
            "n_samples": len(y_test),
        }

        return metrics, y_pred

    def run_loo_cv(self, encoder, flight_datasets):
        """
        Run leave-one-out cross-validation.

        Args:
            encoder: Pre-trained MAE encoder
            flight_datasets: List of (flight_name, dataset, indices) tuples

        Returns:
            results: Dict containing per-fold and aggregate metrics
        """
        n_flights = len(flight_datasets)
        fold_results = []

        print("=" * 80)
        print("RUNNING LEAVE-ONE-OUT CROSS-VALIDATION")
        print("=" * 80)
        print(f"Total folds: {n_flights}")
        print()

        for test_idx in range(n_flights):
            test_flight_name, test_dataset, test_indices = flight_datasets[test_idx]

            print(f"\n{'=' * 80}")
            print(f"FOLD {test_idx + 1}/{n_flights}: Testing on {test_flight_name}")
            print(f"{'=' * 80}\n")

            # Collect training flights
            train_flights = [
                (name, ds, idx)
                for i, (name, ds, idx) in enumerate(flight_datasets)
                if i != test_idx
            ]
            train_flight_names = [name for name, _, _ in train_flights]
            print(f"Training on: {', '.join(train_flight_names)}")
            print()

            # Extract embeddings for test flight
            print(f"Extracting embeddings for {test_flight_name} (test)...")
            test_embeddings, test_angles, test_targets = self.extract_embeddings(
                encoder, test_dataset, test_indices, batch_size=64
            )
            test_features = self.create_hybrid_features(test_embeddings, test_angles)
            print(f"✓ Test features: {test_features.shape}")
            print()

            # Extract embeddings for training flights
            train_embeddings_list = []
            train_angles_list = []
            train_targets_list = []

            for train_flight_name, train_dataset, train_indices in train_flights:
                print(f"Extracting embeddings for {train_flight_name} (train)...")
                emb, ang, tgt = self.extract_embeddings(
                    encoder, train_dataset, train_indices, batch_size=64
                )
                train_embeddings_list.append(emb)
                train_angles_list.append(ang)
                train_targets_list.append(tgt)
                print(f"✓ Features: {emb.shape}")

            # Concatenate training data
            train_embeddings = np.concatenate(train_embeddings_list, axis=0)
            train_angles = np.concatenate(train_angles_list, axis=0)
            train_targets = np.concatenate(train_targets_list, axis=0)
            train_features = self.create_hybrid_features(train_embeddings, train_angles)

            print(f"\nTotal training samples: {train_features.shape[0]}")
            print(f"Total test samples: {test_features.shape[0]}")
            print()

            # Train GBDT
            print("Training GBDT...")
            model, scaler = self.train_gbdt(train_features, train_targets)
            print("✓ Training complete")
            print()

            # Evaluate
            print("Evaluating...")
            metrics, predictions = self.evaluate(
                model, scaler, test_features, test_targets
            )

            print(f"\nResults for {test_flight_name}:")
            print(f"  R² = {metrics['r2']:.4f}")
            print(f"  MAE = {metrics['mae']:.4f} km ({metrics['mae'] * 1000:.1f} m)")
            print(f"  RMSE = {metrics['rmse']:.4f} km ({metrics['rmse'] * 1000:.1f} m)")
            print()

            # Store results
            fold_results.append(
                {
                    "fold": test_idx + 1,
                    "test_flight": test_flight_name,
                    "train_flights": train_flight_names,
                    "metrics": metrics,
                    "predictions": predictions.tolist(),
                    "targets": test_targets.tolist(),
                }
            )

        return fold_results

    def aggregate_results(self, fold_results):
        """
        Aggregate results across all folds.

        Args:
            fold_results: List of per-fold results

        Returns:
            aggregated: Dict of aggregate statistics
        """
        # Extract metrics
        r2_scores = [fold["metrics"]["r2"] for fold in fold_results]
        mae_scores = [fold["metrics"]["mae"] for fold in fold_results]
        rmse_scores = [fold["metrics"]["rmse"] for fold in fold_results]
        n_samples = [fold["metrics"]["n_samples"] for fold in fold_results]

        # Compute statistics
        aggregated = {
            "r2": {
                "mean": float(np.mean(r2_scores)),
                "std": float(np.std(r2_scores)),
                "min": float(np.min(r2_scores)),
                "max": float(np.max(r2_scores)),
                "per_fold": r2_scores,
            },
            "mae": {
                "mean": float(np.mean(mae_scores)),
                "std": float(np.std(mae_scores)),
                "min": float(np.min(mae_scores)),
                "max": float(np.max(mae_scores)),
                "per_fold": mae_scores,
            },
            "rmse": {
                "mean": float(np.mean(rmse_scores)),
                "std": float(np.std(rmse_scores)),
                "min": float(np.min(rmse_scores)),
                "max": float(np.max(rmse_scores)),
                "per_fold": rmse_scores,
            },
            "n_samples_per_fold": n_samples,
            "total_samples": sum(n_samples),
        }

        return aggregated

    def plot_results(self, fold_results, aggregated):
        """
        Create visualization of LOO CV results.

        Args:
            fold_results: List of per-fold results
            aggregated: Aggregated statistics
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle(
            "Leave-One-Out Per-Flight Cross-Validation Results",
            fontsize=16,
            fontweight="bold",
        )

        # 1. Per-fold metrics bar plot
        ax = axes[0, 0]
        flight_names = [fold["test_flight"] for fold in fold_results]
        r2_scores = [fold["metrics"]["r2"] for fold in fold_results]
        x = np.arange(len(flight_names))
        bars = ax.bar(x, r2_scores, alpha=0.7, edgecolor="black")

        # Color bars by performance
        for i, bar in enumerate(bars):
            if r2_scores[i] >= 0.75:
                bar.set_color("green")
            elif r2_scores[i] >= 0.60:
                bar.set_color("orange")
            else:
                bar.set_color("red")

        ax.axhline(
            aggregated["r2"]["mean"],
            color="blue",
            linestyle="--",
            label=f"Mean: {aggregated['r2']['mean']:.3f}",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(flight_names, rotation=45, ha="right")
        ax.set_ylabel("R² Score")
        ax.set_title("Per-Flight R² Scores")
        ax.legend()
        ax.grid(alpha=0.3)

        # 2. Per-fold MAE and RMSE
        ax = axes[0, 1]
        mae_scores = [
            fold["metrics"]["mae"] * 1000 for fold in fold_results
        ]  # to meters
        rmse_scores = [fold["metrics"]["rmse"] * 1000 for fold in fold_results]
        x = np.arange(len(flight_names))
        width = 0.35
        ax.bar(
            x - width / 2, mae_scores, width, label="MAE", alpha=0.7, edgecolor="black"
        )
        ax.bar(
            x + width / 2,
            rmse_scores,
            width,
            label="RMSE",
            alpha=0.7,
            edgecolor="black",
        )
        ax.axhline(
            aggregated["mae"]["mean"] * 1000, color="blue", linestyle="--", alpha=0.5
        )
        ax.axhline(
            aggregated["rmse"]["mean"] * 1000, color="orange", linestyle="--", alpha=0.5
        )
        ax.set_xticks(x)
        ax.set_xticklabels(flight_names, rotation=45, ha="right")
        ax.set_ylabel("Error (meters)")
        ax.set_title("Per-Flight MAE and RMSE")
        ax.legend()
        ax.grid(alpha=0.3)

        # 3. Predictions vs True scatter (all folds combined)
        ax = axes[1, 0]
        colors = plt.cm.tab10(np.linspace(0, 1, len(fold_results)))
        for i, fold in enumerate(fold_results):
            y_true = np.array(fold["targets"])
            y_pred = np.array(fold["predictions"])
            ax.scatter(
                y_true,
                y_pred,
                alpha=0.6,
                s=30,
                label=fold["test_flight"],
                color=colors[i],
            )

        # Plot diagonal
        all_targets = np.concatenate([fold["targets"] for fold in fold_results])
        lim_min = min(
            all_targets.min(), min([min(fold["predictions"]) for fold in fold_results])
        )
        lim_max = max(
            all_targets.max(), max([max(fold["predictions"]) for fold in fold_results])
        )
        ax.plot(
            [lim_min, lim_max],
            [lim_min, lim_max],
            "k--",
            alpha=0.5,
            label="Perfect prediction",
        )

        ax.set_xlabel("True CBH (km)")
        ax.set_ylabel("Predicted CBH (km)")
        ax.set_title("Predictions vs True (All Folds)")
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_aspect("equal", adjustable="box")

        # 4. Summary statistics table
        ax = axes[1, 1]
        ax.axis("off")

        summary_text = (
            f"LOO Cross-Validation Summary\n"
            f"{'=' * 40}\n\n"
            f"Number of folds: {len(fold_results)}\n"
            f"Total samples: {aggregated['total_samples']}\n\n"
            f"R² Score:\n"
            f"  Mean ± Std: {aggregated['r2']['mean']:.4f} ± {aggregated['r2']['std']:.4f}\n"
            f"  Min: {aggregated['r2']['min']:.4f}\n"
            f"  Max: {aggregated['r2']['max']:.4f}\n\n"
            f"MAE (meters):\n"
            f"  Mean ± Std: {aggregated['mae']['mean'] * 1000:.1f} ± {aggregated['mae']['std'] * 1000:.1f}\n"
            f"  Min: {aggregated['mae']['min'] * 1000:.1f}\n"
            f"  Max: {aggregated['mae']['max'] * 1000:.1f}\n\n"
            f"RMSE (meters):\n"
            f"  Mean ± Std: {aggregated['rmse']['mean'] * 1000:.1f} ± {aggregated['rmse']['std'] * 1000:.1f}\n"
            f"  Min: {aggregated['rmse']['min'] * 1000:.1f}\n"
            f"  Max: {aggregated['rmse']['max'] * 1000:.1f}\n"
        )

        ax.text(
            0.1,
            0.9,
            summary_text,
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment="top",
            family="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
        )

        plt.tight_layout()

        # Save figure
        save_path = self.run_dir / "loo_validation_results.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Plot saved to {save_path}")

        plt.close()

    def save_results(self, fold_results, aggregated):
        """
        Save results to JSON files.

        Args:
            fold_results: List of per-fold results
            aggregated: Aggregated statistics
        """
        # Save per-fold results
        fold_path = self.run_dir / "fold_results.json"
        with open(fold_path, "w") as f:
            json.dump(fold_results, f, indent=2)
        print(f"✓ Fold results saved to {fold_path}")

        # Save aggregated results
        agg_path = self.run_dir / "aggregated_metrics.json"
        with open(agg_path, "w") as f:
            json.dump(aggregated, f, indent=2)
        print(f"✓ Aggregated metrics saved to {agg_path}")

        # Save summary table as CSV
        csv_path = self.run_dir / "summary_table.csv"
        summary_data = []
        for fold in fold_results:
            summary_data.append(
                {
                    "Fold": fold["fold"],
                    "Test_Flight": fold["test_flight"],
                    "R2": fold["metrics"]["r2"],
                    "MAE_km": fold["metrics"]["mae"],
                    "RMSE_km": fold["metrics"]["rmse"],
                    "N_Samples": fold["metrics"]["n_samples"],
                }
            )

        df = pd.DataFrame(summary_data)
        df.to_csv(csv_path, index=False)
        print(f"✓ Summary table saved to {csv_path}")

    def run(self):
        """
        Run the complete LOO validation pipeline.
        """
        # Load encoder
        encoder = self.load_encoder()

        # Load per-flight datasets with unified scaler
        flight_datasets, unified_scalers = self.load_flight_datasets()

        # Run LOO cross-validation
        fold_results = self.run_loo_cv(encoder, flight_datasets)

        # Aggregate results
        aggregated = self.aggregate_results(fold_results)

        # Print summary
        print("\n" + "=" * 80)
        print("FINAL RESULTS")
        print("=" * 80)
        print(
            f"\nR² Score: {aggregated['r2']['mean']:.4f} ± {aggregated['r2']['std']:.4f}"
        )
        print(
            f"MAE: {aggregated['mae']['mean']:.4f} km ({aggregated['mae']['mean'] * 1000:.1f} m) "
            f"± {aggregated['mae']['std'] * 1000:.1f} m"
        )
        print(
            f"RMSE: {aggregated['rmse']['mean']:.4f} km ({aggregated['rmse']['mean'] * 1000:.1f} m) "
            f"± {aggregated['rmse']['std'] * 1000:.1f} m"
        )
        print()

        # Plot results
        self.plot_results(fold_results, aggregated)

        # Save results
        self.save_results(fold_results, aggregated)

        print("\n" + "=" * 80)
        print("LOO VALIDATION COMPLETE")
        print("=" * 80)
        print(f"Results saved to: {self.run_dir}")
        print()

        return fold_results, aggregated


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Leave-One-Out per-flight cross-validation for hybrid MAE+GBDT"
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
        default="outputs/mae_pretrain/mae_encoder_pretrained.pth",
        help="Path to pre-trained encoder",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda/cpu)",
    )

    args = parser.parse_args()

    # Run validation
    validator = LOOValidator(
        config_path=args.config,
        encoder_path=args.encoder,
        device=args.device,
    )

    fold_results, aggregated = validator.run()


if __name__ == "__main__":
    main()
