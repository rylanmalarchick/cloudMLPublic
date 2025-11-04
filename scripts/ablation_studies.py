#!/usr/bin/env python3
"""
Ablation Studies for Hybrid MAE+GBDT
=====================================

This script performs systematic ablation studies to understand which components
contribute to the hybrid model's performance:

1. Feature ablations:
   - MAE embeddings only (no angles)
   - Angles only (no MAE embeddings)
   - Hand-crafted features only (baseline)
   - MAE + angles (full hybrid)
   - MAE + angles + hand-crafted features
   - Angles + hand-crafted (no MAE)

2. MAE pre-training ablations:
   - Pre-trained MAE (current)
   - Random initialization (no pre-training)
   - Random embeddings (sanity check)

3. Architecture ablations:
   - Different GBDT models (GradientBoosting, RandomForest, XGBoost if available)

This helps prove causality and identify the most important components.

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
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from src.split_utils import (
    stratified_split_by_flight,
    analyze_split_balance,
    check_split_leakage,
)
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


class AblationStudies:
    """
    Systematic ablation studies for hybrid MAE+GBDT model.
    """

    def __init__(self, config_path, encoder_path, device="cuda"):
        """
        Initialize ablation studies.

        Args:
            config_path: Path to SSL fine-tuning config
            encoder_path: Path to pre-trained MAE encoder weights
            device: Device to run inference on
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.config = self.load_config(config_path)
        self.encoder_path = encoder_path

        print("=" * 80)
        print("ABLATION STUDIES FOR HYBRID MAE+GBDT")
        print("=" * 80)
        print(f"Device: {self.device}")
        print(f"Encoder: {encoder_path}")
        print(f"Config: {config_path}")
        print()

        # Create output directory
        self.output_dir = Path("outputs/ablation_studies")
        self.output_dir.mkdir(exist_ok=True, parents=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_dir / timestamp
        self.run_dir.mkdir(exist_ok=True, parents=True)

        print(f"Output directory: {self.run_dir}")
        print()

        # Store results
        self.ablation_results = []

    def load_config(self, config_path):
        """Load YAML configuration."""
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def load_encoder(self, random_init=False):
        """
        Load MAE encoder.

        Args:
            random_init: If True, use random initialization (no pre-training)

        Returns:
            encoder: MAE encoder model
        """
        model_config = self.config["model"]
        encoder = MAEEncoder(
            img_width=model_config["img_width"],
            patch_size=model_config["patch_size"],
            embed_dim=model_config["embed_dim"],
            depth=model_config["depth"],
            num_heads=model_config["num_heads"],
            mlp_ratio=model_config["mlp_ratio"],
        )

        if not random_init:
            # Load pre-trained weights
            checkpoint = torch.load(
                self.encoder_path, map_location=self.device, weights_only=False
            )
            encoder.load_state_dict(checkpoint)

        encoder.to(self.device)
        encoder.eval()

        return encoder

    def load_dataset(self):
        """Load the labeled CPL-aligned dataset."""
        print("Loading labeled dataset...")

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
            augment=False,
        )

        total_samples = len(dataset)
        print(f"Total CPL-aligned samples: {total_samples}")

        # Split into train/val/test
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
        """Extract MAE embeddings for a dataset."""
        dataloader = DataLoader(
            dataset_subset, batch_size=batch_size, shuffle=False, num_workers=4
        )

        embeddings_list = []
        angles_list = []
        targets_list = []
        images_list = []

        # Get y_scaler from the base dataset
        if isinstance(dataset_subset, Subset):
            y_scaler = dataset_subset.dataset.y_scaler
        else:
            y_scaler = dataset_subset.y_scaler

        with torch.no_grad():
            for batch in dataloader:
                img_stack, sza, saa, y_scaled, global_idx, local_idx = batch

                # Flatten image to 1D signal
                img = img_stack.mean(dim=1)
                img = img.flatten(start_dim=1).unsqueeze(1)

                # Store raw images for hand-crafted features
                images_list.append(img.cpu().numpy())

                # Move to device
                img = img.to(self.device)

                # Forward pass through encoder
                encoded = encoder(img)

                # Extract CLS token
                cls_embeddings = encoded[:, 0, :]

                # Store
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
        images = np.concatenate(images_list, axis=0)

        return embeddings, angles, targets, images

    def compute_hand_crafted_features(self, images):
        """
        Compute hand-crafted features from images.

        Args:
            images: (N, 1, 440) numpy array of flattened images

        Returns:
            features: (N, n_features) numpy array
        """
        features_list = []

        for img in images:
            img_flat = img.squeeze()  # (440,)

            # Statistical features
            mean = np.mean(img_flat)
            std = np.std(img_flat)
            min_val = np.min(img_flat)
            max_val = np.max(img_flat)
            median = np.median(img_flat)

            # Percentiles
            p25 = np.percentile(img_flat, 25)
            p75 = np.percentile(img_flat, 75)
            iqr = p75 - p25

            # Gradient features
            gradient = np.gradient(img_flat)
            grad_mean = np.mean(gradient)
            grad_std = np.std(gradient)
            grad_max = np.max(np.abs(gradient))

            # Zero crossings (proxies for texture)
            zero_crossings = np.sum(np.diff(np.sign(img_flat - mean)) != 0)

            # Combine all features
            feat_vec = [
                mean,
                std,
                min_val,
                max_val,
                median,
                p25,
                p75,
                iqr,
                grad_mean,
                grad_std,
                grad_max,
                zero_crossings,
            ]

            features_list.append(feat_vec)

        return np.array(features_list)

    def train_gbdt(self, X_train, y_train, model_type="gbdt"):
        """
        Train a regressor.

        Args:
            X_train: Training features
            y_train: Training targets
            model_type: 'gbdt', 'rf', or 'xgboost'

        Returns:
            model: Trained model
            scaler: Fitted StandardScaler
        """
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # Train model
        if model_type == "gbdt":
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
        elif model_type == "rf":
            model = RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
                verbose=0,
            )
        elif model_type == "xgboost":
            try:
                import xgboost as xgb

                model = xgb.XGBRegressor(
                    n_estimators=200,
                    learning_rate=0.1,
                    max_depth=5,
                    subsample=0.8,
                    random_state=42,
                    verbosity=0,
                )
            except ImportError:
                print("XGBoost not available, falling back to GBDT")
                model = GradientBoostingRegressor(
                    n_estimators=200,
                    learning_rate=0.1,
                    max_depth=5,
                    random_state=42,
                    verbose=0,
                )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        model.fit(X_train_scaled, y_train)

        return model, scaler

    def evaluate(self, model, scaler, X_test, y_test):
        """Evaluate model on test set."""
        X_test_scaled = scaler.transform(X_test)
        y_pred = model.predict(X_test_scaled)

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

    def run_ablation(
        self,
        name,
        description,
        X_train,
        y_train,
        X_test,
        y_test,
        model_type="gbdt",
    ):
        """
        Run a single ablation experiment.

        Args:
            name: Experiment name
            description: Experiment description
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
            model_type: Model type to use

        Returns:
            result: Dict containing experiment results
        """
        print(f"\n{'=' * 80}")
        print(f"ABLATION: {name}")
        print(f"{'=' * 80}")
        print(f"Description: {description}")
        print(f"Training samples: {X_train.shape[0]}")
        print(f"Test samples: {X_test.shape[0]}")
        print(f"Feature dimension: {X_train.shape[1]}")
        print(f"Model type: {model_type}")
        print()

        # Train model
        print("Training model...")
        model, scaler = self.train_gbdt(X_train, y_train, model_type=model_type)
        print("✓ Training complete")

        # Evaluate
        print("Evaluating...")
        metrics, predictions = self.evaluate(model, scaler, X_test, y_test)

        print(f"\nResults:")
        print(f"  R² = {metrics['r2']:.4f}")
        print(f"  MAE = {metrics['mae']:.4f} km ({metrics['mae'] * 1000:.1f} m)")
        print(f"  RMSE = {metrics['rmse']:.4f} km ({metrics['rmse'] * 1000:.1f} m)")

        # Store results
        result = {
            "name": name,
            "description": description,
            "model_type": model_type,
            "n_features": X_train.shape[1],
            "metrics": metrics,
            "predictions": predictions.tolist(),
            "targets": y_test.tolist(),
        }

        self.ablation_results.append(result)

        return result

    def run_all_ablations(self):
        """Run all ablation experiments."""
        print("\n" + "=" * 80)
        print("RUNNING ALL ABLATION STUDIES")
        print("=" * 80)
        print()

        # Load dataset
        dataset, train_dataset, val_dataset, test_dataset = self.load_dataset()

        # =========================================================================
        # PART 1: FEATURE ABLATIONS
        # =========================================================================
        print("\n" + "=" * 80)
        print("PART 1: FEATURE ABLATIONS")
        print("=" * 80)

        # Load pre-trained encoder
        print("\nLoading pre-trained MAE encoder...")
        encoder_pretrained = self.load_encoder(random_init=False)
        print("✓ Pre-trained encoder loaded")

        # Extract all features
        print("\nExtracting embeddings and features for train set...")
        (
            train_embeddings,
            train_angles,
            train_targets,
            train_images,
        ) = self.extract_embeddings(encoder_pretrained, train_dataset)
        train_handcrafted = self.compute_hand_crafted_features(train_images)
        print(f"✓ Train embeddings: {train_embeddings.shape}")
        print(f"✓ Train angles: {train_angles.shape}")
        print(f"✓ Train hand-crafted: {train_handcrafted.shape}")

        print("\nExtracting embeddings and features for test set...")
        (
            test_embeddings,
            test_angles,
            test_targets,
            test_images,
        ) = self.extract_embeddings(encoder_pretrained, test_dataset)
        test_handcrafted = self.compute_hand_crafted_features(test_images)
        print(f"✓ Test embeddings: {test_embeddings.shape}")
        print(f"✓ Test angles: {test_angles.shape}")
        print(f"✓ Test hand-crafted: {test_handcrafted.shape}")

        # 1. MAE embeddings only (no angles)
        self.run_ablation(
            name="MAE_only",
            description="MAE embeddings only, no angle features",
            X_train=train_embeddings,
            y_train=train_targets,
            X_test=test_embeddings,
            y_test=test_targets,
        )

        # 2. Angles only (no MAE embeddings)
        self.run_ablation(
            name="Angles_only",
            description="Solar angles only (SZA, SAA), no MAE embeddings",
            X_train=train_angles,
            y_train=train_targets,
            X_test=test_angles,
            y_test=test_targets,
        )

        # 3. Hand-crafted features only (baseline)
        self.run_ablation(
            name="HandCrafted_only",
            description="Hand-crafted statistical features only",
            X_train=train_handcrafted,
            y_train=train_targets,
            X_test=test_handcrafted,
            y_test=test_targets,
        )

        # 4. MAE + angles (current hybrid)
        X_train_mae_angles = np.concatenate([train_embeddings, train_angles], axis=1)
        X_test_mae_angles = np.concatenate([test_embeddings, test_angles], axis=1)
        self.run_ablation(
            name="MAE_plus_Angles",
            description="MAE embeddings + solar angles (current hybrid)",
            X_train=X_train_mae_angles,
            y_train=train_targets,
            X_test=X_test_mae_angles,
            y_test=test_targets,
        )

        # 5. MAE + angles + hand-crafted (full feature set)
        X_train_full = np.concatenate(
            [train_embeddings, train_angles, train_handcrafted], axis=1
        )
        X_test_full = np.concatenate(
            [test_embeddings, test_angles, test_handcrafted], axis=1
        )
        self.run_ablation(
            name="Full_Features",
            description="MAE embeddings + angles + hand-crafted features",
            X_train=X_train_full,
            y_train=train_targets,
            X_test=X_test_full,
            y_test=test_targets,
        )

        # 6. Angles + hand-crafted (no MAE)
        X_train_angles_hc = np.concatenate([train_angles, train_handcrafted], axis=1)
        X_test_angles_hc = np.concatenate([test_angles, test_handcrafted], axis=1)
        self.run_ablation(
            name="Angles_plus_HandCrafted",
            description="Solar angles + hand-crafted features (no MAE)",
            X_train=X_train_angles_hc,
            y_train=train_targets,
            X_test=X_test_angles_hc,
            y_test=test_targets,
        )

        # =========================================================================
        # PART 2: MAE PRE-TRAINING ABLATIONS
        # =========================================================================
        print("\n" + "=" * 80)
        print("PART 2: MAE PRE-TRAINING ABLATIONS")
        print("=" * 80)

        # 7. Random initialization (no pre-training)
        print("\nLoading MAE with random initialization (no pre-training)...")
        encoder_random = self.load_encoder(random_init=True)
        print("✓ Random encoder loaded")

        print("\nExtracting embeddings with random encoder (train)...")
        random_train_embeddings, _, _, _ = self.extract_embeddings(
            encoder_random, train_dataset
        )
        print("\nExtracting embeddings with random encoder (test)...")
        random_test_embeddings, _, _, _ = self.extract_embeddings(
            encoder_random, test_dataset
        )

        X_train_random = np.concatenate([random_train_embeddings, train_angles], axis=1)
        X_test_random = np.concatenate([random_test_embeddings, test_angles], axis=1)

        self.run_ablation(
            name="Random_MAE_plus_Angles",
            description="Random MAE initialization (no pre-training) + angles",
            X_train=X_train_random,
            y_train=train_targets,
            X_test=X_test_random,
            y_test=test_targets,
        )

        # 8. Random embeddings (sanity check)
        random_embeddings_train = np.random.randn(*train_embeddings.shape)
        random_embeddings_test = np.random.randn(*test_embeddings.shape)

        X_train_noise = np.concatenate([random_embeddings_train, train_angles], axis=1)
        X_test_noise = np.concatenate([random_embeddings_test, test_angles], axis=1)

        self.run_ablation(
            name="Noise_plus_Angles",
            description="Random noise embeddings + angles (sanity check)",
            X_train=X_train_noise,
            y_train=train_targets,
            X_test=X_test_noise,
            y_test=test_targets,
        )

        # =========================================================================
        # PART 3: MODEL ABLATIONS
        # =========================================================================
        print("\n" + "=" * 80)
        print("PART 3: MODEL ABLATIONS")
        print("=" * 80)

        # 9. RandomForest instead of GradientBoosting
        self.run_ablation(
            name="MAE_Angles_RandomForest",
            description="MAE + angles with RandomForest regressor",
            X_train=X_train_mae_angles,
            y_train=train_targets,
            X_test=X_test_mae_angles,
            y_test=test_targets,
            model_type="rf",
        )

        # 10. XGBoost (if available)
        self.run_ablation(
            name="MAE_Angles_XGBoost",
            description="MAE + angles with XGBoost regressor",
            X_train=X_train_mae_angles,
            y_train=train_targets,
            X_test=X_test_mae_angles,
            y_test=test_targets,
            model_type="xgboost",
        )

        print("\n" + "=" * 80)
        print("ALL ABLATIONS COMPLETE")
        print("=" * 80)
        print()

        return self.ablation_results

    def plot_results(self):
        """Create comprehensive visualization of ablation results."""
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

        fig.suptitle(
            "Ablation Study Results: Hybrid MAE+GBDT",
            fontsize=16,
            fontweight="bold",
        )

        # 1. R² comparison (bar plot)
        ax1 = fig.add_subplot(gs[0, :])
        names = [r["name"] for r in self.ablation_results]
        r2_scores = [r["metrics"]["r2"] for r in self.ablation_results]

        colors = []
        for name in names:
            if "MAE_plus_Angles" in name and "Random" not in name:
                colors.append("green")
            elif "Full_Features" in name:
                colors.append("darkgreen")
            elif "Random" in name or "Noise" in name:
                colors.append("red")
            elif "HandCrafted" in name:
                colors.append("blue")
            else:
                colors.append("gray")

        x = np.arange(len(names))
        bars = ax1.bar(x, r2_scores, color=colors, alpha=0.7, edgecolor="black")

        # Add value labels on bars
        for i, (bar, score) in enumerate(zip(bars, r2_scores)):
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{score:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        ax1.set_xticks(x)
        ax1.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
        ax1.set_ylabel("R² Score", fontsize=11)
        ax1.set_title("R² Score Comparison Across Ablations", fontsize=12)
        ax1.grid(alpha=0.3, axis="y")
        ax1.axhline(0.75, color="red", linestyle="--", alpha=0.5, label="Target (0.75)")
        ax1.legend()

        # 2. MAE comparison (bar plot)
        ax2 = fig.add_subplot(gs[1, 0])
        mae_scores = [r["metrics"]["mae"] * 1000 for r in self.ablation_results]
        ax2.bar(x, mae_scores, color=colors, alpha=0.7, edgecolor="black")
        ax2.set_xticks(x)
        ax2.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax2.set_ylabel("MAE (meters)", fontsize=11)
        ax2.set_title("Mean Absolute Error Comparison", fontsize=12)
        ax2.grid(alpha=0.3, axis="y")

        # 3. RMSE comparison (bar plot)
        ax3 = fig.add_subplot(gs[1, 1])
        rmse_scores = [r["metrics"]["rmse"] * 1000 for r in self.ablation_results]
        ax3.bar(x, rmse_scores, color=colors, alpha=0.7, edgecolor="black")
        ax3.set_xticks(x)
        ax3.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax3.set_ylabel("RMSE (meters)", fontsize=11)
        ax3.set_title("Root Mean Squared Error Comparison", fontsize=12)
        ax3.grid(alpha=0.3, axis="y")

        # 4. Feature importance table
        ax4 = fig.add_subplot(gs[2, 0])
        ax4.axis("off")

        # Create summary table
        table_data = []
        for r in self.ablation_results:
            table_data.append(
                [
                    r["name"][:25],
                    f"{r['metrics']['r2']:.4f}",
                    f"{r['metrics']['mae'] * 1000:.1f}",
                    f"{r['n_features']}",
                ]
            )

        table = ax4.table(
            cellText=table_data,
            colLabels=["Experiment", "R²", "MAE (m)", "N Features"],
            cellLoc="left",
            loc="center",
            bbox=[0, 0, 1, 1],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.5)

        # Color code rows
        for i in range(len(table_data)):
            cell = table[(i + 1, 0)]
            cell.set_facecolor(colors[i])
            cell.set_alpha(0.3)

        ax4.set_title("Detailed Results Table", fontsize=12, pad=10)

        # 5. Key insights text
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.axis("off")

        # Find best and worst
        best_idx = np.argmax(r2_scores)
        worst_idx = np.argmin(r2_scores)
        best = self.ablation_results[best_idx]
        worst = self.ablation_results[worst_idx]

        # Find pre-training impact
        mae_angles_idx = next(
            (
                i
                for i, r in enumerate(self.ablation_results)
                if r["name"] == "MAE_plus_Angles"
            ),
            None,
        )
        random_mae_idx = next(
            (
                i
                for i, r in enumerate(self.ablation_results)
                if r["name"] == "Random_MAE_plus_Angles"
            ),
            None,
        )

        insights_text = "KEY INSIGHTS\n" + "=" * 40 + "\n\n"

        insights_text += f"Best Model:\n"
        insights_text += f"  {best['name']}\n"
        insights_text += f"  R² = {best['metrics']['r2']:.4f}\n"
        insights_text += f"  MAE = {best['metrics']['mae'] * 1000:.1f} m\n\n"

        insights_text += f"Worst Model:\n"
        insights_text += f"  {worst['name']}\n"
        insights_text += f"  R² = {worst['metrics']['r2']:.4f}\n"
        insights_text += f"  MAE = {worst['metrics']['mae'] * 1000:.1f} m\n\n"

        if mae_angles_idx is not None and random_mae_idx is not None:
            pretrained_r2 = self.ablation_results[mae_angles_idx]["metrics"]["r2"]
            random_r2 = self.ablation_results[random_mae_idx]["metrics"]["r2"]
            improvement = pretrained_r2 - random_r2
            insights_text += f"Pre-training Impact:\n"
            insights_text += f"  Pretrained: R² = {pretrained_r2:.4f}\n"
            insights_text += f"  Random: R² = {random_r2:.4f}\n"
            insights_text += f"  Δ = {improvement:+.4f}\n\n"

        insights_text += "Recommendations:\n"
        if best["name"] == "Full_Features":
            insights_text += "  • Full feature set yields best results\n"
        elif best["name"] == "MAE_plus_Angles":
            insights_text += "  • MAE+angles sufficient\n"
            insights_text += "  • Hand-crafted features not needed\n"

        if mae_angles_idx is not None and random_mae_idx is not None:
            improvement = pretrained_r2 - random_r2
            insights_text += f"  • Pre-training is {'critical' if improvement > 0.1 else 'beneficial'}\n"

        ax5.text(
            0.05,
            0.95,
            insights_text,
            transform=ax5.transAxes,
            fontsize=10,
            verticalalignment="top",
            family="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
        )

        # Save figure
        save_path = self.run_dir / "ablation_results.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Plot saved to {save_path}")

        plt.close()

    def save_results(self):
        """Save ablation results to JSON and CSV."""
        # Save full results
        results_path = self.run_dir / "ablation_results.json"
        with open(results_path, "w") as f:
            json.dump(self.ablation_results, f, indent=2)
        print(f"✓ Results saved to {results_path}")

        # Save summary CSV
        summary_data = []
        for r in self.ablation_results:
            summary_data.append(
                {
                    "Experiment": r["name"],
                    "Description": r["description"],
                    "Model_Type": r["model_type"],
                    "N_Features": r["n_features"],
                    "R2": r["metrics"]["r2"],
                    "MAE_km": r["metrics"]["mae"],
                    "RMSE_km": r["metrics"]["rmse"],
                }
            )

        df = pd.DataFrame(summary_data)
        csv_path = self.run_dir / "ablation_summary.csv"
        df.to_csv(csv_path, index=False)
        print(f"✓ Summary table saved to {csv_path}")

    def run(self):
        """Run complete ablation study pipeline."""
        # Run all ablations
        results = self.run_all_ablations()

        # Plot results
        self.plot_results()

        # Save results
        self.save_results()

        # Print summary
        print("\n" + "=" * 80)
        print("ABLATION STUDY SUMMARY")
        print("=" * 80)

        for r in results:
            print(f"\n{r['name']}:")
            print(f"  Description: {r['description']}")
            print(f"  R² = {r['metrics']['r2']:.4f}")
            print(f"  MAE = {r['metrics']['mae'] * 1000:.1f} m")
            print(f"  RMSE = {r['metrics']['rmse'] * 1000:.1f} m")

        print("\n" + "=" * 80)
        print("ABLATION STUDIES COMPLETE")
        print("=" * 80)
        print(f"Results saved to: {self.run_dir}")
        print()

        return results


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Ablation studies for hybrid MAE+GBDT")
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

    # Run ablation studies
    ablation = AblationStudies(
        config_path=args.config,
        encoder_path=args.encoder,
        device=args.device,
    )

    results = ablation.run()


if __name__ == "__main__":
    main()
