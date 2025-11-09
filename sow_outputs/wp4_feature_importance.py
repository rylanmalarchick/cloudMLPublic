#!/usr/bin/env python3
"""
Work Package 4: Feature Importance Analysis

Computes permutation feature importance for the best hybrid model (attention)
to understand which features contribute most to CBH prediction.

This is a required deliverable (7.3b) for Sprint 3.

Author: Autonomous Agent
Date: 2025
SOW: SOW-AGENT-CBH-WP-001 Section 7.3b
"""

import sys
from pathlib import Path
import numpy as np
import h5py
import yaml
import json
import torch
import torch.nn as nn
from typing import Dict, List, Tuple
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.hdf5_dataset import HDF5CloudDataset
from sow_outputs.wp4_cnn_model import HybridCNN2D, HybridDataset


class FeatureImportanceAnalyzer:
    """
    Computes feature importance using permutation importance.

    For hybrid models with images + physical features, we analyze:
    1. Physical feature importance (permutation on each feature)
    2. Image vs physical contribution (ablation)
    """

    def __init__(
        self,
        model_path: str,
        era5_features_path: str,
        geometric_features_path: str,
        config_path: str,
        fusion_mode: str = "attention",
        output_dir: str = "sow_outputs/wp4_feature_importance",
        device: str = "cuda",
        verbose: bool = True,
    ):
        """
        Initialize the analyzer.

        Args:
            model_path: Path to trained model weights (.pth)
            era5_features_path: Path to WP2_Features.hdf5
            geometric_features_path: Path to WP1_Features.hdf5
            config_path: Path to config YAML
            fusion_mode: Model fusion mode (attention, concat, image_only)
            output_dir: Output directory
            device: Device for computation
            verbose: Verbose logging
        """
        self.model_path = Path(model_path)
        self.era5_features_path = Path(era5_features_path)
        self.geometric_features_path = Path(geometric_features_path)
        self.config_path = Path(config_path)
        self.fusion_mode = fusion_mode
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.verbose = verbose

        # Load config
        with open(self.config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Flight mapping
        self.flight_mapping = {
            0: "30Oct24",
            1: "10Feb25",
            2: "23Oct24",
            3: "12Feb25",
            4: "18Feb25",
        }

        self.flight_name_to_id = {}
        for fid, fname in self.flight_mapping.items():
            self.flight_name_to_id[fname] = fid

    def load_data(self):
        """Load dataset and features."""
        if self.verbose:
            print("\n" + "=" * 80)
            print("Loading Data")
            print("=" * 80)

        # Load image dataset
        self.image_dataset = HDF5CloudDataset(
            flight_configs=self.config["flights"],
            indices=None,
            swath_slice=(0, 440),
            filter_type="basic",
            augment=False,
            temporal_frames=1,
            angles_mode="none",
        )

        n_samples = len(self.image_dataset)

        if self.verbose:
            print(f"Total samples: {n_samples}")

        # Cache CBH values
        self.cbh_values = self.image_dataset.get_unscaled_y()

        # Load ERA5 features
        with h5py.File(self.era5_features_path, "r") as f:
            self.era5_features = f["features"][:]
            self.era5_names = [
                n.decode() if isinstance(n, bytes) else n for n in f["feature_names"][:]
            ]

        # Load geometric features
        with h5py.File(self.geometric_features_path, "r") as f:
            geo_cbh = f["derived_geometric_H"][:]
            geo_length = f["shadow_length_pixels"][:]
            geo_conf = f["shadow_detection_confidence"][:]

            self.geometric_features = np.column_stack([geo_cbh, geo_length, geo_conf])
            self.geo_names = [
                "derived_geometric_H",
                "shadow_length_pixels",
                "shadow_detection_confidence",
            ]

        # Handle NaNs in geometric features
        for j in range(self.geometric_features.shape[1]):
            col = self.geometric_features[:, j]
            if np.any(np.isnan(col)):
                median_val = np.nanmedian(col)
                self.geometric_features[np.isnan(col), j] = median_val

        # Combine physical features
        self.physical_features = np.concatenate(
            [self.geometric_features, self.era5_features], axis=1
        )
        self.physical_names = self.geo_names + self.era5_names

        if self.verbose:
            print(f"Physical features: {len(self.physical_names)}")
            print(f"  Geometric: {self.geo_names}")
            print(f"  ERA5: {self.era5_names}")

    def load_model(self):
        """Load trained model."""
        if self.verbose:
            print("\n" + "=" * 80)
            print("Loading Model")
            print("=" * 80)

        # Determine dimensions
        n_physical = self.physical_features.shape[1]

        # Create model
        self.model = HybridCNN2D(
            in_channels=1,
            n_physical_features=n_physical,
            fusion_mode=self.fusion_mode,
        )

        # Load weights
        if self.model_path.exists():
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            if self.verbose:
                print(f"✓ Loaded model from: {self.model_path}")
        else:
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        self.model = self.model.to(self.device)
        self.model.eval()

    def evaluate_model(
        self, era5_features: np.ndarray, geometric_features: np.ndarray
    ) -> Tuple[float, float]:
        """
        Evaluate model with given features.

        Args:
            era5_features: ERA5 feature matrix
            geometric_features: Geometric feature matrix

        Returns:
            r2, mae_km
        """
        # Create dataset
        all_indices = list(range(len(self.image_dataset)))
        physical_features = np.concatenate([geometric_features, era5_features], axis=1)

        dataset = HybridDataset(
            image_dataset=self.image_dataset,
            era5_features=physical_features[:, len(self.geo_names) :],
            geometric_features=physical_features[:, : len(self.geo_names)],
            indices=all_indices,
            cbh_values=self.cbh_values,
        )

        # Create dataloader
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=32, shuffle=False, num_workers=0
        )

        # Evaluate
        predictions = []
        targets = []

        with torch.no_grad():
            for batch in loader:
                images = batch[0].to(self.device)
                physical = batch[1].to(self.device)
                y = batch[2].to(self.device)

                # Forward pass
                if self.fusion_mode == "image_only":
                    y_pred = self.model(images, None)
                else:
                    y_pred = self.model(images, physical)

                predictions.append(y_pred.cpu().numpy())
                targets.append(y.cpu().numpy())

        predictions = np.concatenate(predictions)
        targets = np.concatenate(targets)

        # Compute metrics
        r2 = 1 - np.sum((targets - predictions) ** 2) / np.sum(
            (targets - targets.mean()) ** 2
        )
        mae = np.mean(np.abs(targets - predictions))

        return r2, mae

    def compute_permutation_importance(
        self, n_repeats: int = 10
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute permutation importance for each physical feature.

        Args:
            n_repeats: Number of permutation repeats per feature

        Returns:
            Dictionary of feature importances
        """
        if self.verbose:
            print("\n" + "=" * 80)
            print("Computing Permutation Importance")
            print("=" * 80)

        # Baseline performance
        baseline_r2, baseline_mae = self.evaluate_model(
            self.era5_features, self.geometric_features
        )

        if self.verbose:
            print(f"Baseline performance:")
            print(f"  R² = {baseline_r2:.4f}")
            print(f"  MAE = {baseline_mae:.4f} km")

        # Permutation importance
        importances = {}

        # Geometric features
        if self.verbose:
            print("\nGeometric features:")

        for j, name in enumerate(self.geo_names):
            r2_drops = []
            mae_increases = []

            for repeat in range(n_repeats):
                # Permute feature
                geo_permuted = self.geometric_features.copy()
                np.random.shuffle(geo_permuted[:, j])

                # Evaluate
                r2, mae = self.evaluate_model(self.era5_features, geo_permuted)

                r2_drops.append(baseline_r2 - r2)
                mae_increases.append(mae - baseline_mae)

            importances[name] = {
                "mean_r2_drop": float(np.mean(r2_drops)),
                "std_r2_drop": float(np.std(r2_drops)),
                "mean_mae_increase_km": float(np.mean(mae_increases)),
                "std_mae_increase_km": float(np.std(mae_increases)),
            }

            if self.verbose:
                print(
                    f"  {name}: R² drop = {importances[name]['mean_r2_drop']:.4f} ± {importances[name]['std_r2_drop']:.4f}"
                )

        # ERA5 features
        if self.verbose:
            print("\nERA5 features:")

        for j, name in enumerate(self.era5_names):
            r2_drops = []
            mae_increases = []

            for repeat in range(n_repeats):
                # Permute feature
                era5_permuted = self.era5_features.copy()
                np.random.shuffle(era5_permuted[:, j])

                # Evaluate
                r2, mae = self.evaluate_model(era5_permuted, self.geometric_features)

                r2_drops.append(baseline_r2 - r2)
                mae_increases.append(mae - baseline_mae)

            importances[name] = {
                "mean_r2_drop": float(np.mean(r2_drops)),
                "std_r2_drop": float(np.std(r2_drops)),
                "mean_mae_increase_km": float(np.mean(mae_increases)),
                "std_mae_increase_km": float(np.std(mae_increases)),
            }

            if self.verbose:
                print(
                    f"  {name}: R² drop = {importances[name]['mean_r2_drop']:.4f} ± {importances[name]['std_r2_drop']:.4f}"
                )

        return importances, baseline_r2, baseline_mae

    def generate_report(
        self,
        importances: Dict,
        baseline_r2: float,
        baseline_mae: float,
    ) -> Dict:
        """Generate feature importance report."""

        # Sort features by importance
        sorted_features = sorted(
            importances.items(), key=lambda x: x[1]["mean_r2_drop"], reverse=True
        )

        report = {
            "model_variant": self.fusion_mode,
            "model_path": str(self.model_path),
            "baseline_performance": {
                "r2": float(baseline_r2),
                "mae_km": float(baseline_mae),
            },
            "feature_importance": {
                name: {
                    "mean_r2_drop": imp["mean_r2_drop"],
                    "std_r2_drop": imp["std_r2_drop"],
                    "mean_mae_increase_km": imp["mean_mae_increase_km"],
                    "std_mae_increase_km": imp["std_mae_increase_km"],
                }
                for name, imp in sorted_features
            },
            "top_5_features": [name for name, _ in sorted_features[:5]],
            "timestamp": datetime.now().isoformat(),
        }

        return report

    def save_report(self, report: Dict, filename: str = "WP4_Feature_Importance.json"):
        """Save report to JSON."""
        output_path = self.output_dir / filename
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        if self.verbose:
            print(f"\n✓ Report saved to: {output_path}")

    def print_summary(self, report: Dict):
        """Print formatted summary."""
        print("\n" + "=" * 80)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("=" * 80)

        print(f"\nModel: {report['model_variant']}")
        print(f"Baseline R²: {report['baseline_performance']['r2']:.4f}")
        print(f"Baseline MAE: {report['baseline_performance']['mae_km']:.4f} km")

        print("\nTop 5 Most Important Features:")
        print(f"{'Rank':<6} {'Feature':<30} {'R² Drop':<15} {'MAE Increase (km)':<20}")
        print("-" * 80)

        for i, feature_name in enumerate(report["top_5_features"]):
            imp = report["feature_importance"][feature_name]
            print(
                f"{i + 1:<6} {feature_name:<30} "
                f"{imp['mean_r2_drop']:.4f} ± {imp['std_r2_drop']:.4f}   "
                f"{imp['mean_mae_increase_km']:.4f} ± {imp['std_mae_increase_km']:.4f}"
            )

        print("\nAll Features (sorted by importance):")
        print(f"{'Feature':<35} {'R² Drop':<15} {'MAE Increase (km)':<20}")
        print("-" * 80)

        for feature_name, imp in report["feature_importance"].items():
            print(
                f"{feature_name:<35} "
                f"{imp['mean_r2_drop']:.4f} ± {imp['std_r2_drop']:.4f}   "
                f"{imp['mean_mae_increase_km']:.4f} ± {imp['std_mae_increase_km']:.4f}"
            )

        print("=" * 80)

    def run(self):
        """Execute complete feature importance analysis."""
        print("\n" + "=" * 80)
        print("WP-4: FEATURE IMPORTANCE ANALYSIS")
        print("=" * 80)
        print(f"Model: {self.fusion_mode}")
        print(f"Model path: {self.model_path}")
        print(f"Output directory: {self.output_dir}")

        # Load data
        self.load_data()

        # Load model
        self.load_model()

        # Compute permutation importance
        importances, baseline_r2, baseline_mae = self.compute_permutation_importance(
            n_repeats=10
        )

        # Generate report
        report = self.generate_report(importances, baseline_r2, baseline_mae)

        # Save report
        self.save_report(report)

        # Print summary
        self.print_summary(report)

        return report


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute feature importance for WP-4 hybrid model"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model weights (.pth)",
    )
    parser.add_argument(
        "--fusion-mode",
        type=str,
        default="attention",
        choices=["attention", "concat", "image_only"],
        help="Fusion mode",
    )
    parser.add_argument(
        "--era5-features",
        type=str,
        default="sow_outputs/wp2_atmospheric/WP2_Features.hdf5",
        help="ERA5 features path",
    )
    parser.add_argument(
        "--geometric-features",
        type=str,
        default="sow_outputs/wp1_geometric/WP1_Features.hdf5",
        help="Geometric features path",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Config YAML path",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="sow_outputs/wp4_feature_importance",
        help="Output directory",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device (cuda or cpu)"
    )

    args = parser.parse_args()

    # Check paths
    if not Path(args.model_path).exists():
        print(f"ERROR: Model not found: {args.model_path}")
        return 1

    if not Path(args.era5_features).exists():
        print(f"ERROR: ERA5 features not found: {args.era5_features}")
        return 1

    if not Path(args.geometric_features).exists():
        print(f"ERROR: Geometric features not found: {args.geometric_features}")
        return 1

    if not Path(args.config).exists():
        print(f"ERROR: Config not found: {args.config}")
        return 1

    # Run analyzer
    analyzer = FeatureImportanceAnalyzer(
        model_path=args.model_path,
        era5_features_path=args.era5_features,
        geometric_features_path=args.geometric_features,
        config_path=args.config,
        fusion_mode=args.fusion_mode,
        output_dir=args.output_dir,
        device=args.device,
        verbose=True,
    )

    report = analyzer.run()

    return 0


if __name__ == "__main__":
    sys.exit(main())
