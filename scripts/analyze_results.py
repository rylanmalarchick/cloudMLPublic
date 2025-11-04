#!/usr/bin/env python3
"""
Physical Sanity Checks and Results Analysis
============================================

This script performs comprehensive physical validation and analysis of hybrid
MAE+GBDT results:

1. Physical bounds checking:
   - Ensure predictions are within valid CBH range (0.1-2.0 km)
   - Flag outliers and physically implausible predictions

2. Residual analysis:
   - Plot residuals vs predicted values
   - Residuals vs solar zenith angle (SZA)
   - Residuals vs solar azimuth angle (SAA)
   - Check for systematic biases

3. Per-flight analysis:
   - Performance breakdown by flight
   - Identify flights with anomalous performance
   - Visualize per-flight prediction quality

4. Error case analysis:
   - Identify worst predictions
   - Analyze characteristics of failure cases
   - Visualize examples of good/bad predictions

5. Statistical validation:
   - Normality test on residuals
   - Homoscedasticity check
   - Correlation analysis

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
from scipy import stats

# Import project modules
from src.hdf5_dataset import HDF5CloudDataset
from src.mae_model import MAEEncoder


class ResultsAnalyzer:
    """
    Comprehensive physical validation and analysis of hybrid MAE+GBDT results.
    """

    def __init__(self, config_path, encoder_path, device="cuda"):
        """
        Initialize results analyzer.

        Args:
            config_path: Path to SSL fine-tuning config
            encoder_path: Path to pre-trained MAE encoder weights
            device: Device to run inference on
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.config = self.load_config(config_path)
        self.encoder_path = encoder_path

        print("=" * 80)
        print("PHYSICAL SANITY CHECKS AND RESULTS ANALYSIS")
        print("=" * 80)
        print(f"Device: {self.device}")
        print(f"Encoder: {encoder_path}")
        print(f"Config: {config_path}")
        print()

        # Create output directory
        self.output_dir = Path("outputs/results_analysis")
        self.output_dir.mkdir(exist_ok=True, parents=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_dir / timestamp
        self.run_dir.mkdir(exist_ok=True, parents=True)

        print(f"Output directory: {self.run_dir}")
        print()

        # Physical bounds for CBH
        self.cbh_min = 0.1  # km
        self.cbh_max = 2.0  # km

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
        print()

        return encoder

    def load_dataset_with_flight_info(self):
        """
        Load dataset and track which samples belong to which flight.

        Returns:
            dataset: Full dataset
            splits: Dict with train/val/test subsets
            flight_indices: Dict mapping flight names to sample indices
        """
        print("Loading labeled dataset with flight tracking...")

        flight_configs = self.config["data"]["flights"]
        flight_indices = {}
        all_datasets = []
        offset = 0

        # Load each flight separately to track indices
        for flight_config in flight_configs:
            flight_name = flight_config["name"]

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
            )

            n_samples = len(dataset)
            flight_indices[flight_name] = list(range(offset, offset + n_samples))
            all_datasets.append(dataset)
            offset += n_samples

            print(
                f"  {flight_name}: {n_samples} samples (indices {flight_indices[flight_name][0]}-{flight_indices[flight_name][-1]})"
            )

        # Now load the combined dataset
        dataset = HDF5CloudDataset(
            flight_configs=flight_configs,
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
        print(f"\nTotal samples: {total_samples}")

        # Create stratified splits to ensure balanced flight representation
        train_ratio = self.config["data"]["train_ratio"]
        val_ratio = self.config["data"]["val_ratio"]

        print("\nCreating stratified train/val/test splits...")
        train_indices, val_indices, test_indices = stratified_split_by_flight(
            dataset, train_ratio=train_ratio, val_ratio=val_ratio, seed=42, verbose=True
        )

        # Verify no leakage
        check_split_leakage(train_indices, val_indices, test_indices)

        splits = {
            "train": Subset(dataset, train_indices),
            "val": Subset(dataset, val_indices),
            "test": Subset(dataset, test_indices),
            "train_indices": train_indices,
            "val_indices": val_indices,
            "test_indices": test_indices,
        }

        print(f"  Training: {len(train_indices)}")
        print(f"  Validation: {len(val_indices)}")
        print(f"  Test: {len(test_indices)}")
        print()

        return dataset, splits, flight_indices

    def extract_embeddings_with_metadata(self, encoder, dataset_subset, batch_size=64):
        """
        Extract MAE embeddings along with all metadata.

        Returns:
            data: Dict containing embeddings, angles, targets, and metadata
        """
        dataloader = DataLoader(
            dataset_subset, batch_size=batch_size, shuffle=False, num_workers=4
        )

        embeddings_list = []
        sza_list = []
        saa_list = []
        targets_list = []
        global_idx_list = []
        local_idx_list = []

        # Get y_scaler from dataset
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

                # Move to device
                img = img.to(self.device)

                # Forward pass through encoder
                encoded = encoder(img)

                # Extract CLS token
                cls_embeddings = encoded[:, 0, :]

                # Store
                embeddings_list.append(cls_embeddings.cpu().numpy())
                sza_list.append(sza.cpu().numpy())
                saa_list.append(saa.cpu().numpy())

                # Convert y_scaled back to km
                y_scaled_np = y_scaled.cpu().numpy().reshape(-1, 1)
                y_unscaled = y_scaler.inverse_transform(y_scaled_np).flatten()
                targets_list.append(y_unscaled)

                global_idx_list.append(global_idx.cpu().numpy())
                local_idx_list.append(local_idx.cpu().numpy())

        # Concatenate
        embeddings = np.concatenate(embeddings_list, axis=0)
        sza = np.concatenate(sza_list, axis=0).squeeze()
        saa = np.concatenate(saa_list, axis=0).squeeze()
        targets = np.concatenate(targets_list, axis=0)
        global_indices = np.concatenate(global_idx_list, axis=0)
        local_indices = np.concatenate(local_idx_list, axis=0)

        data = {
            "embeddings": embeddings,
            "sza": sza,
            "saa": saa,
            "targets": targets,
            "global_indices": global_indices,
            "local_indices": local_indices,
        }

        return data

    def train_and_predict(self, train_data, test_data):
        """
        Train GBDT and make predictions.

        Args:
            train_data: Training data dict
            test_data: Test data dict

        Returns:
            predictions: Test predictions
            model: Trained model
            scaler: Fitted scaler
        """
        # Create features
        X_train = np.concatenate(
            [
                train_data["embeddings"],
                train_data["sza"].reshape(-1, 1),
                train_data["saa"].reshape(-1, 1),
            ],
            axis=1,
        )
        y_train = train_data["targets"]

        X_test = np.concatenate(
            [
                test_data["embeddings"],
                test_data["sza"].reshape(-1, 1),
                test_data["saa"].reshape(-1, 1),
            ],
            axis=1,
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train GBDT
        print("Training GBDT...")
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
        print("✓ Training complete")
        print()

        # Predict
        predictions = model.predict(X_test_scaled)

        return predictions, model, scaler

    def check_physical_bounds(self, predictions, targets):
        """
        Check if predictions are within physical bounds.

        Args:
            predictions: Model predictions
            targets: True values

        Returns:
            report: Dict with bounds checking results
        """
        print("=" * 80)
        print("PHYSICAL BOUNDS CHECKING")
        print("=" * 80)
        print()

        n_total = len(predictions)
        n_below_min = np.sum(predictions < self.cbh_min)
        n_above_max = np.sum(predictions > self.cbh_max)
        n_valid = n_total - n_below_min - n_above_max

        pct_below = 100 * n_below_min / n_total
        pct_above = 100 * n_above_max / n_total
        pct_valid = 100 * n_valid / n_total

        print(f"Physical bounds: [{self.cbh_min} - {self.cbh_max}] km")
        print(f"Total predictions: {n_total}")
        print(f"  Valid (within bounds): {n_valid} ({pct_valid:.2f}%)")
        print(f"  Below minimum: {n_below_min} ({pct_below:.2f}%)")
        print(f"  Above maximum: {n_above_max} ({pct_above:.2f}%)")
        print()

        # Check targets too
        targets_below = np.sum(targets < self.cbh_min)
        targets_above = np.sum(targets > self.cbh_max)
        print(f"Ground truth bounds check:")
        print(f"  Below minimum: {targets_below}")
        print(f"  Above maximum: {targets_above}")
        print()

        report = {
            "bounds": {"min": self.cbh_min, "max": self.cbh_max},
            "predictions": {
                "total": int(n_total),
                "valid": int(n_valid),
                "below_min": int(n_below_min),
                "above_max": int(n_above_max),
                "pct_valid": float(pct_valid),
            },
            "targets": {
                "below_min": int(targets_below),
                "above_max": int(targets_above),
            },
        }

        return report

    def analyze_residuals(self, predictions, targets, sza, saa):
        """
        Analyze residual patterns.

        Args:
            predictions: Model predictions
            targets: True values
            sza: Solar zenith angles
            saa: Solar azimuth angles

        Returns:
            analysis: Dict with residual analysis results
        """
        print("=" * 80)
        print("RESIDUAL ANALYSIS")
        print("=" * 80)
        print()

        residuals = predictions - targets

        # Basic statistics
        mean_residual = np.mean(residuals)
        std_residual = np.std(residuals)
        median_residual = np.median(residuals)

        print(f"Residual statistics:")
        print(f"  Mean: {mean_residual:.4f} km ({mean_residual * 1000:.1f} m)")
        print(f"  Std: {std_residual:.4f} km ({std_residual * 1000:.1f} m)")
        print(f"  Median: {median_residual:.4f} km ({median_residual * 1000:.1f} m)")
        print()

        # Test for normality
        statistic, p_value = stats.shapiro(residuals[: min(5000, len(residuals))])
        print(f"Shapiro-Wilk normality test:")
        print(f"  Statistic: {statistic:.4f}")
        print(f"  p-value: {p_value:.4e}")
        print(f"  Normal distribution: {'Yes' if p_value > 0.05 else 'No'}")
        print()

        # Check for bias vs angles
        corr_sza = np.corrcoef(residuals, sza)[0, 1]
        corr_saa = np.corrcoef(residuals, saa)[0, 1]

        print(f"Residual correlation with angles:")
        print(f"  SZA: {corr_sza:.4f}")
        print(f"  SAA: {corr_saa:.4f}")
        print()

        # Check for heteroscedasticity (residuals vs predictions)
        corr_hetero = np.corrcoef(np.abs(residuals), predictions)[0, 1]
        print(f"Heteroscedasticity check:")
        print(f"  Correlation(|residuals|, predictions): {corr_hetero:.4f}")
        print(f"  Homoscedastic: {'Yes' if abs(corr_hetero) < 0.3 else 'No'}")
        print()

        analysis = {
            "mean": float(mean_residual),
            "std": float(std_residual),
            "median": float(median_residual),
            "normality_test": {
                "statistic": float(statistic),
                "p_value": float(p_value),
                "is_normal": bool(p_value > 0.05),
            },
            "angle_correlation": {
                "sza": float(corr_sza),
                "saa": float(corr_saa),
            },
            "heteroscedasticity": {
                "correlation": float(corr_hetero),
                "is_homoscedastic": bool(abs(corr_hetero) < 0.3),
            },
        }

        return analysis, residuals

    def analyze_per_flight(self, predictions, targets, test_indices, flight_indices):
        """
        Analyze performance per flight.

        Args:
            predictions: Model predictions
            targets: True values
            test_indices: Indices of test samples
            flight_indices: Dict mapping flight names to indices

        Returns:
            per_flight: Dict with per-flight metrics
        """
        print("=" * 80)
        print("PER-FLIGHT ANALYSIS")
        print("=" * 80)
        print()

        per_flight = {}

        for flight_name, flight_idx in flight_indices.items():
            # Find which test samples belong to this flight
            mask = np.isin(test_indices, flight_idx)
            n_samples = np.sum(mask)

            if n_samples == 0:
                print(f"{flight_name}: No test samples")
                continue

            flight_preds = predictions[mask]
            flight_targets = targets[mask]

            # Compute metrics
            r2 = r2_score(flight_targets, flight_preds)
            mae = mean_absolute_error(flight_targets, flight_preds)
            rmse = np.sqrt(mean_squared_error(flight_targets, flight_preds))

            print(f"{flight_name}:")
            print(f"  Samples: {n_samples}")
            print(f"  R²: {r2:.4f}")
            print(f"  MAE: {mae:.4f} km ({mae * 1000:.1f} m)")
            print(f"  RMSE: {rmse:.4f} km ({rmse * 1000:.1f} m)")
            print()

            per_flight[flight_name] = {
                "n_samples": int(n_samples),
                "r2": float(r2),
                "mae": float(mae),
                "rmse": float(rmse),
            }

        return per_flight

    def identify_failure_cases(self, predictions, targets, sza, saa, n_worst=10):
        """
        Identify and analyze worst predictions.

        Args:
            predictions: Model predictions
            targets: True values
            sza: Solar zenith angles
            saa: Solar azimuth angles
            n_worst: Number of worst cases to analyze

        Returns:
            failure_cases: Dict with failure case analysis
        """
        print("=" * 80)
        print("FAILURE CASE ANALYSIS")
        print("=" * 80)
        print()

        errors = np.abs(predictions - targets)
        worst_indices = np.argsort(errors)[-n_worst:][::-1]

        print(f"Top {n_worst} worst predictions:")
        print()

        failure_cases = []

        for i, idx in enumerate(worst_indices):
            error = errors[idx]
            pred = predictions[idx]
            true = targets[idx]
            sza_val = sza[idx]
            saa_val = saa[idx]

            print(f"{i + 1}. Sample {idx}:")
            print(f"   True: {true:.4f} km, Predicted: {pred:.4f} km")
            print(f"   Error: {error:.4f} km ({error * 1000:.1f} m)")
            print(f"   SZA: {sza_val:.2f}°, SAA: {saa_val:.2f}°")
            print()

            failure_cases.append(
                {
                    "index": int(idx),
                    "true": float(true),
                    "predicted": float(pred),
                    "error": float(error),
                    "sza": float(sza_val),
                    "saa": float(saa_val),
                }
            )

        # Analyze characteristics of failure cases
        worst_sza = sza[worst_indices]
        worst_saa = saa[worst_indices]
        worst_targets = targets[worst_indices]

        print(f"Failure case characteristics:")
        print(f"  Mean SZA: {np.mean(worst_sza):.2f}° (overall: {np.mean(sza):.2f}°)")
        print(f"  Mean SAA: {np.mean(worst_saa):.2f}° (overall: {np.mean(saa):.2f}°)")
        print(
            f"  Mean CBH: {np.mean(worst_targets):.3f} km (overall: {np.mean(targets):.3f} km)"
        )
        print()

        return failure_cases

    def plot_comprehensive_analysis(
        self, predictions, targets, sza, saa, residuals, per_flight, bounds_report
    ):
        """Create comprehensive analysis plots."""
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

        fig.suptitle(
            "Comprehensive Results Analysis: Hybrid MAE+GBDT",
            fontsize=16,
            fontweight="bold",
        )

        # 1. Predictions vs True
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.scatter(targets, predictions, alpha=0.5, s=20)
        lim_min = min(targets.min(), predictions.min())
        lim_max = max(targets.max(), predictions.max())
        ax1.plot([lim_min, lim_max], [lim_min, lim_max], "r--", label="Perfect")
        ax1.set_xlabel("True CBH (km)")
        ax1.set_ylabel("Predicted CBH (km)")
        ax1.set_title("Predictions vs True")
        ax1.legend()
        ax1.grid(alpha=0.3)
        ax1.set_aspect("equal", adjustable="box")

        # Add R² annotation
        r2 = r2_score(targets, predictions)
        ax1.text(
            0.05,
            0.95,
            f"R² = {r2:.4f}",
            transform=ax1.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat"),
        )

        # 2. Residual distribution
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.hist(residuals * 1000, bins=50, alpha=0.7, edgecolor="black")
        ax2.axvline(0, color="red", linestyle="--", label="Zero error")
        ax2.set_xlabel("Residual (meters)")
        ax2.set_ylabel("Frequency")
        ax2.set_title("Residual Distribution")
        ax2.legend()
        ax2.grid(alpha=0.3)

        # 3. Residuals vs Predictions
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.scatter(predictions, residuals * 1000, alpha=0.5, s=20)
        ax3.axhline(0, color="red", linestyle="--")
        ax3.set_xlabel("Predicted CBH (km)")
        ax3.set_ylabel("Residual (meters)")
        ax3.set_title("Residuals vs Predictions")
        ax3.grid(alpha=0.3)

        # 4. Residuals vs SZA
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.scatter(sza, residuals * 1000, alpha=0.5, s=20)
        ax4.axhline(0, color="red", linestyle="--")
        ax4.set_xlabel("Solar Zenith Angle (°)")
        ax4.set_ylabel("Residual (meters)")
        ax4.set_title("Residuals vs SZA")
        ax4.grid(alpha=0.3)

        # Add correlation
        corr = np.corrcoef(sza, residuals)[0, 1]
        ax4.text(
            0.05,
            0.95,
            f"Corr = {corr:.3f}",
            transform=ax4.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat"),
        )

        # 5. Residuals vs SAA
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.scatter(saa, residuals * 1000, alpha=0.5, s=20)
        ax5.axhline(0, color="red", linestyle="--")
        ax5.set_xlabel("Solar Azimuth Angle (°)")
        ax5.set_ylabel("Residual (meters)")
        ax5.set_title("Residuals vs SAA")
        ax5.grid(alpha=0.3)

        corr = np.corrcoef(saa, residuals)[0, 1]
        ax5.text(
            0.05,
            0.95,
            f"Corr = {corr:.3f}",
            transform=ax5.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat"),
        )

        # 6. Error vs True CBH
        ax6 = fig.add_subplot(gs[1, 2])
        errors = np.abs(residuals)
        ax6.scatter(targets, errors * 1000, alpha=0.5, s=20)
        ax6.set_xlabel("True CBH (km)")
        ax6.set_ylabel("Absolute Error (meters)")
        ax6.set_title("Error vs True CBH")
        ax6.grid(alpha=0.3)

        # 7. Per-flight performance
        if per_flight:
            ax7 = fig.add_subplot(gs[2, 0])
            flight_names = list(per_flight.keys())
            r2_scores = [per_flight[f]["r2"] for f in flight_names]
            colors = [
                "green" if r2 > 0.75 else "orange" if r2 > 0.5 else "red"
                for r2 in r2_scores
            ]
            x = np.arange(len(flight_names))
            bars = ax7.bar(x, r2_scores, color=colors, alpha=0.7, edgecolor="black")
            ax7.set_xticks(x)
            ax7.set_xticklabels(flight_names, rotation=45, ha="right")
            ax7.set_ylabel("R² Score")
            ax7.set_title("Per-Flight Performance")
            ax7.grid(alpha=0.3, axis="y")

            # Add value labels
            for i, (bar, score) in enumerate(zip(bars, r2_scores)):
                height = bar.get_height()
                ax7.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{score:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

        # 8. Q-Q plot for normality check
        ax8 = fig.add_subplot(gs[2, 1])
        stats.probplot(residuals, dist="norm", plot=ax8)
        ax8.set_title("Q-Q Plot (Normality Check)")
        ax8.grid(alpha=0.3)

        # 9. Physical bounds summary
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.axis("off")

        bounds_text = (
            f"PHYSICAL BOUNDS CHECK\n"
            f"{'=' * 35}\n\n"
            f"Valid range: [{self.cbh_min}-{self.cbh_max}] km\n\n"
            f"Predictions:\n"
            f"  Valid: {bounds_report['predictions']['pct_valid']:.1f}%\n"
            f"  Below min: {bounds_report['predictions']['below_min']}\n"
            f"  Above max: {bounds_report['predictions']['above_max']}\n\n"
            f"Overall Metrics:\n"
            f"  R²: {r2:.4f}\n"
            f"  MAE: {mean_absolute_error(targets, predictions) * 1000:.1f} m\n"
            f"  RMSE: {np.sqrt(mean_squared_error(targets, predictions)) * 1000:.1f} m\n\n"
            f"Residuals:\n"
            f"  Mean: {np.mean(residuals) * 1000:.1f} m\n"
            f"  Std: {np.std(residuals) * 1000:.1f} m\n"
        )

        ax9.text(
            0.1,
            0.9,
            bounds_text,
            transform=ax9.transAxes,
            fontsize=10,
            verticalalignment="top",
            family="monospace",
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.3),
        )

        # Save
        save_path = self.run_dir / "comprehensive_analysis.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Analysis plot saved to {save_path}")

        plt.close()

    def save_results(self, results):
        """Save all analysis results to JSON."""
        results_path = self.run_dir / "analysis_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"✓ Results saved to {results_path}")

        # Save per-flight as CSV
        if "per_flight" in results and results["per_flight"]:
            df_data = []
            for flight, metrics in results["per_flight"].items():
                df_data.append(
                    {
                        "Flight": flight,
                        "N_Samples": metrics["n_samples"],
                        "R2": metrics["r2"],
                        "MAE_km": metrics["mae"],
                        "RMSE_km": metrics["rmse"],
                    }
                )
            df = pd.DataFrame(df_data)
            csv_path = self.run_dir / "per_flight_metrics.csv"
            df.to_csv(csv_path, index=False)
            print(f"✓ Per-flight metrics saved to {csv_path}")

    def run(self):
        """Run complete analysis pipeline."""
        print("Starting comprehensive analysis...\n")

        # Load encoder
        encoder = self.load_encoder()

        # Load dataset with flight tracking
        dataset, splits, flight_indices = self.load_dataset_with_flight_info()

        # Extract embeddings and metadata
        print("Extracting embeddings for training set...")
        train_data = self.extract_embeddings_with_metadata(encoder, splits["train"])
        print(f"✓ Train data extracted: {train_data['embeddings'].shape}\n")

        print("Extracting embeddings for test set...")
        test_data = self.extract_embeddings_with_metadata(encoder, splits["test"])
        print(f"✓ Test data extracted: {test_data['embeddings'].shape}\n")

        # Train and predict
        predictions, model, scaler = self.train_and_predict(train_data, test_data)

        targets = test_data["targets"]
        sza = test_data["sza"]
        saa = test_data["saa"]

        # Run all analyses
        results = {}

        # 1. Physical bounds check
        bounds_report = self.check_physical_bounds(predictions, targets)
        results["physical_bounds"] = bounds_report

        # 2. Residual analysis
        residual_analysis, residuals = self.analyze_residuals(
            predictions, targets, sza, saa
        )
        results["residual_analysis"] = residual_analysis

        # 3. Per-flight analysis
        per_flight = self.analyze_per_flight(
            predictions, targets, splits["test_indices"], flight_indices
        )
        results["per_flight"] = per_flight

        # 4. Failure case analysis
        failure_cases = self.identify_failure_cases(predictions, targets, sza, saa)
        results["failure_cases"] = failure_cases

        # 5. Overall metrics
        results["overall_metrics"] = {
            "r2": float(r2_score(targets, predictions)),
            "mae": float(mean_absolute_error(targets, predictions)),
            "rmse": float(np.sqrt(mean_squared_error(targets, predictions))),
            "n_samples": len(targets),
        }

        # Plot comprehensive analysis
        self.plot_comprehensive_analysis(
            predictions, targets, sza, saa, residuals, per_flight, bounds_report
        )

        # Save results
        self.save_results(results)

        # Print summary
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)
        print(f"\nOverall Performance:")
        print(f"  R² = {results['overall_metrics']['r2']:.4f}")
        print(f"  MAE = {results['overall_metrics']['mae'] * 1000:.1f} m")
        print(f"  RMSE = {results['overall_metrics']['rmse'] * 1000:.1f} m")
        print(f"\nPhysical Validity:")
        print(f"  {bounds_report['predictions']['pct_valid']:.1f}% within bounds")
        print(f"\nResults saved to: {self.run_dir}")
        print()

        return results


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Physical sanity checks and results analysis"
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

    # Run analysis
    analyzer = ResultsAnalyzer(
        config_path=args.config,
        encoder_path=args.encoder,
        device=args.device,
    )

    results = analyzer.run()


if __name__ == "__main__":
    main()
