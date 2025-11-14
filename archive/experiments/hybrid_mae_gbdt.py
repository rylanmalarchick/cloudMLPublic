#!/usr/bin/env python3
"""
Strategy 1: Hybrid MAE + GradientBoosting Framework
====================================================

This script implements the "High-Confidence" Strategy 1 from the technical
scope of work document:

1. Extract deep features (embeddings) from the pre-trained MAE encoder
2. Concatenate with solar angle metadata (SZA, SAA)
3. Train a GradientBoosting regressor on the hybrid feature set
4. Compare against baseline (R² = 0.75 with hand-crafted features)

Expected outcome: MAE's learned features should outperform hand-crafted features,
potentially achieving R² > 0.75 on the CBH regression task.

Author: Based on cloudmlpublic technical analysis
Date: November 2024
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
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from src.split_utils import (
    stratified_split_by_flight,
    analyze_split_balance,
    check_split_leakage,
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
from datetime import datetime

# Import project modules
from src.hdf5_dataset import HDF5CloudDataset
from src.mae_model import MAEEncoder


class HybridMAEGBDT:
    """
    Hybrid model combining MAE embeddings with GradientBoosting regression.
    """

    def __init__(self, config_path, encoder_path, device="cuda"):
        """
        Initialize the hybrid model.

        Args:
            config_path: Path to SSL fine-tuning config (for data loading)
            encoder_path: Path to pre-trained MAE encoder weights
            device: Device to run inference on
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.config = self.load_config(config_path)
        self.encoder_path = encoder_path

        print("=" * 80)
        print("HYBRID MAE + GRADIENTBOOSTING FRAMEWORK")
        print("=" * 80)
        print(f"Device: {self.device}")
        print(f"Encoder: {encoder_path}")
        print(f"Config: {config_path}")
        print()

        # Create output directory
        self.output_dir = Path("outputs/hybrid_mae_gbdt")
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

        # Get encoder config from main config
        model_config = self.config["model"]
        encoder = MAEEncoder(
            img_width=model_config["img_width"],
            patch_size=model_config["patch_size"],
            embed_dim=model_config["embed_dim"],
            depth=model_config["depth"],
            num_heads=model_config["num_heads"],
            mlp_ratio=model_config["mlp_ratio"],
        )

        # Load pre-trained weights
        checkpoint = torch.load(
            self.encoder_path, map_location=self.device, weights_only=False
        )
        encoder.load_state_dict(checkpoint)
        encoder.to(self.device)
        encoder.eval()

        print(f" Encoder loaded successfully")
        print(f"  Embedding dimension: {model_config['embed_dim']}")
        print(
            f"  Architecture: {model_config['depth']} blocks, {model_config['num_heads']} heads"
        )
        print()

        return encoder

    def load_dataset(self):
        """Load the labeled CPL-aligned dataset."""
        print("Loading labeled dataset...")

        # Create dataset using config parameters
        dataset = HDF5CloudDataset(
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
        )

        total_samples = len(dataset)
        print(f"Total CPL-aligned samples: {total_samples}")

        # Split into train/val/test (same as fine-tuning)
        train_ratio = self.config["data"]["train_ratio"]
        val_ratio = self.config["data"]["val_ratio"]

        # Use stratified split to ensure balanced flight representation
        print("\nCreating stratified train/val/test splits...")
        train_indices, val_indices, test_indices = stratified_split_by_flight(
            dataset, train_ratio=train_ratio, val_ratio=val_ratio, seed=42, verbose=True
        )

        # Verify no leakage
        check_split_leakage(train_indices, val_indices, test_indices)

        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)
        test_dataset = Subset(dataset, test_indices)

        print(f"  Training samples: {len(train_dataset)}")
        print(f"  Validation samples: {len(val_dataset)}")
        print(f"  Test samples: {len(test_dataset)}")
        print()

        return dataset, train_dataset, val_dataset, test_dataset

    def extract_embeddings(self, encoder, dataset_subset, batch_size=64):
        """
        Extract MAE embeddings (CLS tokens) for a dataset.

        Args:
            encoder: Pre-trained MAE encoder
            dataset_subset: Dataset or Subset to extract from
            batch_size: Batch size for extraction

        Returns:
            embeddings: (N, embed_dim) numpy array
            angles: (N, 2) numpy array [SZA, SAA]
            targets: (N,) numpy array (CBH values in km)
        """
        dataloader = DataLoader(
            dataset_subset, batch_size=batch_size, shuffle=False, num_workers=4
        )

        embeddings_list = []
        angles_list = []
        targets_list = []

        print(f"Extracting embeddings from {len(dataset_subset)} samples...")

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting"):
                img_stack, sza, saa, y_scaled, y_unscaled, _ = batch

                # Flatten image to 1D signal (same as fine-tuning)
                # img_stack shape: (batch, temporal_frames, H, W)
                img = img_stack.mean(dim=1)  # Average temporal frames if > 1
                img = img.flatten(start_dim=1).unsqueeze(1)  # (batch, 1, 440)

                # Move to device
                img = img.to(self.device)

                # Forward pass through encoder
                # Returns: (batch, num_patches + 1, embed_dim)
                encoded = encoder(img)

                # Extract CLS token (first token)
                cls_embeddings = encoded[:, 0, :]  # (batch, embed_dim)

                # Store
                embeddings_list.append(cls_embeddings.cpu().numpy())
                # Stack angles and squeeze to get (batch, 2) shape
                angles_batch = torch.stack([sza, saa], dim=1).cpu().numpy()
                if angles_batch.ndim == 3:
                    angles_batch = angles_batch.squeeze(
                        -1
                    )  # (batch, 2, 1) -> (batch, 2)
                angles_list.append(angles_batch)
                targets_list.append(y_unscaled.cpu().numpy())

        # Concatenate all batches
        embeddings = np.concatenate(embeddings_list, axis=0)
        angles = np.concatenate(angles_list, axis=0)
        targets = np.concatenate(targets_list, axis=0)

        print(f" Embeddings extracted: {embeddings.shape}")
        print(f"  Embeddings: {embeddings.shape}")
        print(f"  Angles: {angles.shape}")
        print(f"  Targets: {targets.shape}")
        print()

        return embeddings, angles, targets

    def create_hybrid_features(self, embeddings, angles):
        """
        Combine MAE embeddings with angle features.

        Args:
            embeddings: (N, embed_dim) MAE CLS tokens
            angles: (N, 2) [SZA, SAA]

        Returns:
            features: (N, embed_dim + 2) hybrid feature matrix
        """
        return np.concatenate([embeddings, angles], axis=1)

    def train_gbdt(self, X_train, y_train, X_val, y_val, tune_hyperparams=True):
        """
        Train GradientBoosting regressor on hybrid features.

        Args:
            X_train, y_train: Training features and targets
            X_val, y_val: Validation features and targets
            tune_hyperparams: Whether to run grid search

        Returns:
            model: Trained GradientBoostingRegressor
            scaler: Fitted feature scaler
        """
        print("Training GradientBoosting regressor...")

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        if tune_hyperparams:
            print("Running hyperparameter tuning with GridSearchCV...")
            param_grid = {
                "n_estimators": [100, 200, 300],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.05, 0.1],
                "min_samples_split": [2, 5, 10],
            }

            gbdt = GradientBoostingRegressor(random_state=42)
            grid_search = GridSearchCV(
                gbdt,
                param_grid,
                cv=5,
                scoring="r2",
                n_jobs=-1,
                verbose=1,
            )
            grid_search.fit(X_train_scaled, y_train)

            best_model = grid_search.best_estimator_
            print(f" Best parameters: {grid_search.best_params_}")
            print(f"  Best CV R²: {grid_search.best_score_:.4f}")

        else:
            # Use baseline parameters from diagnostics
            best_model = GradientBoostingRegressor(
                n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
            )
            best_model.fit(X_train_scaled, y_train)
            print(" Trained with baseline parameters")

        # Validation performance
        y_val_pred = best_model.predict(X_val_scaled)
        val_r2 = r2_score(y_val, y_val_pred)
        val_mae = mean_absolute_error(y_val, y_val_pred)
        val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))

        # Convert to km if targets are in meters
        if y_val.max() > 10:
            val_mae_km = val_mae / 1000.0
            val_rmse_km = val_rmse / 1000.0
        else:
            val_mae_km = val_mae
            val_rmse_km = val_rmse

        print(f"\nValidation Performance:")
        print(f"  R²:   {val_r2:.4f}")
        print(f"  MAE:  {val_mae_km:.4f} km")
        print(f"  RMSE: {val_rmse_km:.4f} km")
        print()

        return best_model, scaler

    def evaluate(self, model, scaler, X_test, y_test):
        """
        Evaluate model on test set.

        Args:
            model: Trained model
            scaler: Fitted scaler
            X_test, y_test: Test features and targets

        Returns:
            results: Dictionary of metrics
            predictions: Test set predictions
        """
        print("Evaluating on test set...")

        X_test_scaled = scaler.transform(X_test)
        y_pred = model.predict(X_test_scaled)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        # Convert to km if targets are in meters (check range)
        if y_test.max() > 10:  # Likely in meters if max > 10
            mae_km = mae / 1000.0
            rmse_km = rmse / 1000.0
        else:
            mae_km = mae
            rmse_km = rmse

        results = {
            "test_r2": float(r2),
            "test_mae": float(mae_km),
            "test_rmse": float(rmse_km),
            "n_samples": len(y_test),
        }

        print(f"\nTest Set Results:")
        print(f"  R²:   {r2:.4f}")
        print(f"  MAE:  {mae_km:.4f} km")
        print(f"  RMSE: {rmse_km:.4f} km")
        print()

        return results, y_pred

    def compare_to_baseline(self, results):
        """
        Compare results to baseline performance.
        """
        # Baselines from diagnostics and SSL
        baseline_classical = {"r2": 0.7464, "mae": 0.1265, "rmse": 0.1929}
        baseline_ssl = {"r2": 0.3665, "mae": 0.2211, "rmse": 0.2993}

        print("=" * 80)
        print("COMPARISON TO BASELINES")
        print("=" * 80)

        print("\n1. Classical ML (Hand-crafted features + GBDT):")
        print(
            f"   R² = {baseline_classical['r2']:.4f}, MAE = {baseline_classical['mae']:.4f} km"
        )

        print("\n2. SSL End-to-End (MAE + Neural Head):")
        print(f"   R² = {baseline_ssl['r2']:.4f}, MAE = {baseline_ssl['mae']:.4f} km")

        print("\n3. Hybrid (MAE Embeddings + GBDT) [THIS WORK]:")
        print(f"   R² = {results['test_r2']:.4f}, MAE = {results['test_mae']:.4f} km")

        print("\n" + "-" * 80)

        # Improvements
        r2_improvement_classical = results["test_r2"] - baseline_classical["r2"]
        r2_improvement_ssl = results["test_r2"] - baseline_ssl["r2"]

        print("\nImprovement over Classical ML:")
        print(f"  ΔR² = {r2_improvement_classical:+.4f}")
        if r2_improvement_classical > 0:
            print(f"   HYBRID WINS! ({abs(r2_improvement_classical):.2%} better)")
        else:
            print(
                f"   Classical still better ({abs(r2_improvement_classical):.2%} gap)"
            )

        print("\nImprovement over SSL End-to-End:")
        print(f"  ΔR² = {r2_improvement_ssl:+.4f}")
        if r2_improvement_ssl > 0:
            print(
                f"   Hybrid better than end-to-end ({abs(r2_improvement_ssl):.2%} improvement)"
            )

        print("=" * 80)
        print()

        return {
            "baseline_classical": baseline_classical,
            "baseline_ssl": baseline_ssl,
            "hybrid": results,
            "improvement_over_classical": float(r2_improvement_classical),
            "improvement_over_ssl": float(r2_improvement_ssl),
        }

    def plot_results(self, y_test, y_pred, results, comparison):
        """
        Create visualization plots.
        """
        print("Creating plots...")

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # 1. Scatter plot: Predicted vs True
        ax = axes[0, 0]
        ax.scatter(y_test, y_pred, alpha=0.5, s=20)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
        ax.set_xlabel("True CBH (km)", fontsize=12)
        ax.set_ylabel("Predicted CBH (km)", fontsize=12)
        ax.set_title(
            f"Hybrid MAE+GBDT: R² = {results['test_r2']:.4f}",
            fontsize=14,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3)
        ax.text(
            0.05,
            0.95,
            f"MAE = {results['test_mae']:.4f} km\nRMSE = {results['test_rmse']:.4f} km",
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        # 2. Residual plot
        ax = axes[0, 1]
        residuals = y_test - y_pred
        ax.scatter(y_pred, residuals, alpha=0.5, s=20)
        ax.axhline(y=0, color="r", linestyle="--", lw=2)
        ax.set_xlabel("Predicted CBH (km)", fontsize=12)
        ax.set_ylabel("Residuals (km)", fontsize=12)
        ax.set_title("Residual Distribution", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)

        # 3. Residual histogram
        ax = axes[1, 0]
        ax.hist(residuals, bins=30, edgecolor="black", alpha=0.7)
        ax.axvline(x=0, color="r", linestyle="--", lw=2)
        ax.set_xlabel("Residual (km)", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.set_title(
            f"Residual Distribution (μ={residuals.mean():.4f}, σ={residuals.std():.4f})",
            fontsize=14,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3)

        # 4. Model comparison bar chart
        ax = axes[1, 1]
        models = [
            "Classical\n(Hand-crafted)",
            "SSL\n(End-to-End)",
            "Hybrid\n(MAE+GBDT)",
        ]
        r2_scores = [
            comparison["baseline_classical"]["r2"],
            comparison["baseline_ssl"]["r2"],
            comparison["hybrid"]["test_r2"],
        ]
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
        bars = ax.bar(models, r2_scores, color=colors, alpha=0.7, edgecolor="black")

        # Highlight best
        best_idx = np.argmax(r2_scores)
        bars[best_idx].set_edgecolor("gold")
        bars[best_idx].set_linewidth(3)

        ax.set_ylabel("Test R²", fontsize=12)
        ax.set_title("Model Comparison", fontsize=14, fontweight="bold")
        ax.set_ylim([0, 1.0])
        ax.grid(True, alpha=0.3, axis="y")

        # Add values on bars
        for i, (bar, score) in enumerate(zip(bars, r2_scores)):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.02,
                f"{score:.4f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        plt.tight_layout()
        plot_path = self.run_dir / "hybrid_results.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f" Plot saved to: {plot_path}")
        plt.close()

    def save_results(self, results, comparison, model, scaler):
        """
        Save results and model artifacts.
        """
        print("Saving results...")

        # Save metrics
        metrics = {
            "test_results": results,
            "comparison": comparison,
            "timestamp": datetime.now().isoformat(),
        }
        metrics_path = self.run_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f" Metrics saved to: {metrics_path}")

        # Save model (optional - GBDT is small)
        import joblib

        model_path = self.run_dir / "gbdt_model.pkl"
        scaler_path = self.run_dir / "feature_scaler.pkl"
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        print(f" Model saved to: {model_path}")
        print(f" Scaler saved to: {scaler_path}")

    def run(self, tune_hyperparams=True):
        """
        Execute the full hybrid pipeline.

        Args:
            tune_hyperparams: Whether to tune GBDT hyperparameters
        """
        # 1. Load encoder
        encoder = self.load_encoder()

        # 2. Load dataset
        full_dataset, train_dataset, val_dataset, test_dataset = self.load_dataset()

        # 3. Extract embeddings for each split
        print("=" * 80)
        print("STEP 1: EXTRACTING MAE EMBEDDINGS")
        print("=" * 80)
        print()

        train_emb, train_angles, train_y = self.extract_embeddings(
            encoder, train_dataset
        )
        val_emb, val_angles, val_y = self.extract_embeddings(encoder, val_dataset)
        test_emb, test_angles, test_y = self.extract_embeddings(encoder, test_dataset)

        # 4. Create hybrid features
        print("=" * 80)
        print("STEP 2: CREATING HYBRID FEATURE SETS")
        print("=" * 80)
        print()

        X_train = self.create_hybrid_features(train_emb, train_angles)
        X_val = self.create_hybrid_features(val_emb, val_angles)
        X_test = self.create_hybrid_features(test_emb, test_angles)

        print(f"Train features: {X_train.shape}")
        print(f"Val features:   {X_val.shape}")
        print(f"Test features:  {X_test.shape}")
        print()

        # 5. Train GradientBoosting
        print("=" * 80)
        print("STEP 3: TRAINING GRADIENTBOOSTING REGRESSOR")
        print("=" * 80)
        print()

        model, scaler = self.train_gbdt(
            X_train, train_y, X_val, val_y, tune_hyperparams=tune_hyperparams
        )

        # 6. Evaluate
        print("=" * 80)
        print("STEP 4: FINAL EVALUATION")
        print("=" * 80)
        print()

        results, y_pred = self.evaluate(model, scaler, X_test, test_y)

        # 7. Compare to baselines
        comparison = self.compare_to_baseline(results)

        # 8. Visualize
        self.plot_results(test_y, y_pred, results, comparison)

        # 9. Save
        self.save_results(results, comparison, model, scaler)

        print("=" * 80)
        print("HYBRID PIPELINE COMPLETE!")
        print("=" * 80)
        print(f"\nResults saved to: {self.run_dir}")
        print()

        return results, comparison


def main():
    """Main execution."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Hybrid MAE + GradientBoosting for CBH regression"
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
        "--no-tune",
        action="store_true",
        help="Skip hyperparameter tuning (use baseline params)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda/cpu)",
    )

    args = parser.parse_args()

    # Verify encoder exists
    if not Path(args.encoder).exists():
        print(f"ERROR: Encoder not found at {args.encoder}")
        print("Please run Phase 2 pre-training first:")
        print("  ./scripts/run_phase2_pretrain.sh")
        sys.exit(1)

    # Verify config exists
    if not Path(args.config).exists():
        print(f"ERROR: Config not found at {args.config}")
        sys.exit(1)

    # Run hybrid pipeline
    hybrid = HybridMAEGBDT(
        config_path=args.config, encoder_path=args.encoder, device=args.device
    )

    results, comparison = hybrid.run(tune_hyperparams=not args.no_tune)

    # Print final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"\nHybrid MAE+GBDT Test R²: {results['test_r2']:.4f}")

    if results["test_r2"] > 0.7464:
        print("\n SUCCESS! Hybrid approach BEATS classical baseline! ")
    elif results["test_r2"] > 0.3665:
        print("\n Hybrid approach beats SSL end-to-end (but not classical yet)")
    else:
        print("\n Hybrid approach underperforms - may need architecture tuning")

    print("=" * 80)


if __name__ == "__main__":
    main()
