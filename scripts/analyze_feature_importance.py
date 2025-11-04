#!/usr/bin/env python3
"""
Analyze feature importance to understand what the GBDT model relies on.

This script:
1. Trains GBDT on different feature combinations
2. Extracts feature importance from the trained models
3. Analyzes which features (MAE dimensions vs angles) contribute most
4. Creates visualizations of feature importance patterns
5. Performs permutation importance analysis

Usage:
    python scripts/analyze_feature_importance.py \
        --config configs/ssl_finetune_cbh.yaml \
        --encoder outputs/mae_pretrain/mae_encoder_pretrained.pth \
        --output outputs/feature_importance
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import yaml
from torch.utils.data import DataLoader, Subset
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from tqdm import tqdm
import json
from datetime import datetime

from src.hdf5_dataset import HDF5CloudDataset
from src.mae_model import MaskedAutoencoder
from src.split_utils import stratified_split_by_flight


class FeatureImportanceAnalyzer:
    def __init__(self, config_path, encoder_path, output_dir):
        self.config_path = config_path
        self.encoder_path = encoder_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load config
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

    def load_encoder(self):
        """Load pre-trained MAE encoder."""
        print("\nLoading MAE encoder...")

        encoder = MaskedAutoencoder(
            img_width=self.config["model"]["img_width"],
            patch_size=self.config["model"]["patch_size"],
            embed_dim=self.config["model"]["embed_dim"],
            depth=self.config["model"]["depth"],
            num_heads=self.config["model"]["num_heads"],
            decoder_embed_dim=96,  # Default, not used for inference
            decoder_depth=2,
            decoder_num_heads=3,
            mlp_ratio=self.config["model"]["mlp_ratio"],
        ).to(self.device)

        checkpoint = torch.load(
            self.encoder_path, map_location=self.device, weights_only=False
        )
        # Handle both direct state dict and wrapped dict
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        # Load only encoder weights (ignore decoder)
        encoder_state_dict = {
            k: v
            for k, v in state_dict.items()
            if k.startswith("encoder.") or not k.startswith("decoder.")
        }
        # Remove 'encoder.' prefix if present
        encoder_state_dict = {
            k.replace("encoder.", ""): v for k, v in encoder_state_dict.items()
        }
        encoder.encoder.load_state_dict(encoder_state_dict, strict=False)
        encoder.eval()

        print(f"✓ Encoder loaded (embed_dim={self.config['model']['embed_dim']})")
        return encoder

    def load_dataset(self):
        """Load dataset and create stratified splits."""
        print("\nLoading dataset...")

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

        print(f"✓ Dataset loaded: {len(dataset)} samples")

        # Create stratified splits
        train_idx, val_idx, test_idx = stratified_split_by_flight(
            dataset,
            train_ratio=self.config["data"]["train_ratio"],
            val_ratio=self.config["data"]["val_ratio"],
            seed=42,
            verbose=False,
        )

        train_dataset = Subset(dataset, train_idx)
        test_dataset = Subset(dataset, test_idx)

        print(f"  Train: {len(train_dataset)}, Test: {len(test_dataset)}")

        return dataset, train_dataset, test_dataset

    def extract_features(self, encoder, dataset_subset, batch_size=64):
        """Extract MAE embeddings and angles."""
        dataloader = DataLoader(
            dataset_subset, batch_size=batch_size, shuffle=False, num_workers=4
        )

        embeddings_list = []
        angles_list = []
        targets_list = []

        # Get base dataset for y_scaler
        if isinstance(dataset_subset, Subset):
            base_dataset = dataset_subset.dataset
        else:
            base_dataset = dataset_subset

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting", leave=False):
                img_stack, sza, saa, y_scaled, _, _ = batch
                img_stack = img_stack.to(self.device)

                # Handle temporal dimension: (B, C, T, W) -> (B, C, W)
                # Take middle frame if temporal_frames > 1
                if img_stack.dim() == 4:
                    temporal_frames = img_stack.shape[2]
                    mid_frame = temporal_frames // 2
                    img_stack = img_stack[:, :, mid_frame, :]

                # Extract MAE embeddings (CLS token)
                cls_token = encoder.forward_encoder(img_stack)
                embeddings_list.append(cls_token.cpu().numpy())

                # Store angles
                angles_list.append(np.column_stack([sza.numpy(), saa.numpy()]))

                # Unscale targets
                y_unscaled = base_dataset.y_scaler.inverse_transform(
                    y_scaled.numpy().reshape(-1, 1)
                ).flatten()
                targets_list.append(y_unscaled)

        embeddings = np.vstack(embeddings_list)
        angles = np.vstack(angles_list)
        targets = np.concatenate(targets_list)

        return embeddings, angles, targets

    def train_and_analyze(self, X_train, y_train, X_test, y_test, feature_names):
        """Train GBDT and extract feature importance."""
        print(f"  Training GBDT on {X_train.shape[1]} features...")

        # Train GBDT
        model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
            verbose=0,
        )
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        print(f"  R² = {r2:.4f}, MAE = {mae * 1000:.1f} m")

        # Get feature importance
        importance = model.feature_importances_

        # Permutation importance (more reliable but slower)
        print("  Computing permutation importance...")
        perm_importance = permutation_importance(
            model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
        )

        return {
            "model": model,
            "r2": r2,
            "mae": mae,
            "rmse": rmse,
            "feature_importance": importance,
            "permutation_importance": perm_importance.importances_mean,
            "permutation_std": perm_importance.importances_std,
            "feature_names": feature_names,
        }

    def run_ablations(
        self,
        embeddings_train,
        angles_train,
        targets_train,
        embeddings_test,
        angles_test,
        targets_test,
    ):
        """Run feature importance analysis on different feature combinations."""
        print("\nRunning feature importance ablations...")

        results = {}

        # 1. MAE + Angles (full model)
        print("\n[1] MAE + Angles")
        X_train = np.hstack([embeddings_train, angles_train])
        X_test = np.hstack([embeddings_test, angles_test])
        feature_names = [f"MAE_dim_{i}" for i in range(embeddings_train.shape[1])] + [
            "SZA",
            "SAA",
        ]
        results["mae_angles"] = self.train_and_analyze(
            X_train, targets_train, X_test, targets_test, feature_names
        )

        # 2. Angles only
        print("\n[2] Angles only")
        feature_names = ["SZA", "SAA"]
        results["angles_only"] = self.train_and_analyze(
            angles_train, targets_train, angles_test, targets_test, feature_names
        )

        # 3. MAE only
        print("\n[3] MAE only")
        feature_names = [f"MAE_dim_{i}" for i in range(embeddings_train.shape[1])]
        results["mae_only"] = self.train_and_analyze(
            embeddings_train,
            targets_train,
            embeddings_test,
            targets_test,
            feature_names,
        )

        return results

    def plot_feature_importance(self, results):
        """Create visualizations of feature importance."""
        print("\nCreating feature importance visualizations...")

        # 1. Full model feature importance
        self._plot_full_importance(results["mae_angles"])

        # 2. Compare angle importance across models
        self._plot_angle_importance_comparison(results)

        # 3. Top MAE dimensions
        self._plot_top_mae_dimensions(results["mae_angles"])

        # 4. Permutation vs built-in importance
        self._plot_importance_comparison(results["mae_angles"])

    def _plot_full_importance(self, result):
        """Plot all features sorted by importance."""
        importance = result["feature_importance"]
        feature_names = result["feature_names"]

        # Sort by importance
        indices = np.argsort(importance)[::-1]
        sorted_importance = importance[indices]
        sorted_names = [feature_names[i] for i in indices]

        # Take top 30 for visibility
        top_k = min(30, len(sorted_importance))

        fig, ax = plt.subplots(figsize=(10, 8))
        y_pos = np.arange(top_k)

        # Color MAE dims differently from angles
        colors = [
            "tab:blue" if "MAE" in name else "tab:orange"
            for name in sorted_names[:top_k]
        ]

        ax.barh(y_pos, sorted_importance[:top_k], color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_names[:top_k], fontsize=8)
        ax.set_xlabel("Feature Importance")
        ax.set_title(f"Top {top_k} Features (MAE+Angles)\nR² = {result['r2']:.4f}")
        ax.grid(axis="x", alpha=0.3)

        # Add legend
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor="tab:blue", label="MAE dimensions"),
            Patch(facecolor="tab:orange", label="Angles (SZA/SAA)"),
        ]
        ax.legend(handles=legend_elements, loc="lower right")

        plt.tight_layout()
        output_path = self.output_dir / "feature_importance_full.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {output_path}")

    def _plot_angle_importance_comparison(self, results):
        """Compare angle importance across different models."""
        fig, ax = plt.subplots(figsize=(10, 6))

        models = ["angles_only", "mae_angles"]
        model_labels = ["Angles Only", "MAE+Angles"]
        x = np.arange(2)  # SZA, SAA
        width = 0.35

        for i, (model_key, label) in enumerate(zip(models, model_labels)):
            result = results[model_key]
            feature_names = result["feature_names"]

            # Find SZA and SAA indices
            sza_idx = feature_names.index("SZA")
            saa_idx = feature_names.index("SAA")

            importances = [
                result["feature_importance"][sza_idx],
                result["feature_importance"][saa_idx],
            ]

            ax.bar(x + i * width, importances, width, label=label, alpha=0.8)

        ax.set_ylabel("Feature Importance")
        ax.set_title("Angle Feature Importance Across Models")
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(["SZA", "SAA"])
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        output_path = self.output_dir / "angle_importance_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {output_path}")

    def _plot_top_mae_dimensions(self, result):
        """Plot importance of top MAE dimensions."""
        feature_names = result["feature_names"]
        importance = result["feature_importance"]

        # Extract MAE dimension importances
        mae_importances = []
        mae_dims = []
        for i, name in enumerate(feature_names):
            if name.startswith("MAE_dim_"):
                mae_importances.append(importance[i])
                dim_num = int(name.split("_")[-1])
                mae_dims.append(dim_num)

        mae_importances = np.array(mae_importances)
        mae_dims = np.array(mae_dims)

        # Sort and take top 20
        top_k = min(20, len(mae_importances))
        indices = np.argsort(mae_importances)[::-1][:top_k]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(range(top_k), mae_importances[indices])
        ax.set_xlabel("Rank")
        ax.set_ylabel("Feature Importance")
        ax.set_title(f"Top {top_k} MAE Dimensions by Importance")
        ax.grid(axis="y", alpha=0.3)

        # Add dimension numbers as text
        for i, idx in enumerate(indices):
            ax.text(
                i,
                mae_importances[idx],
                f"D{mae_dims[idx]}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        plt.tight_layout()
        output_path = self.output_dir / "top_mae_dimensions.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {output_path}")

    def _plot_importance_comparison(self, result):
        """Compare built-in vs permutation importance."""
        importance = result["feature_importance"]
        perm_importance = result["permutation_importance"]
        feature_names = result["feature_names"]

        # Take top 20 by built-in importance
        top_k = min(20, len(importance))
        indices = np.argsort(importance)[::-1][:top_k]

        fig, ax = plt.subplots(figsize=(10, 8))
        y_pos = np.arange(top_k)
        width = 0.4

        ax.barh(
            y_pos - width / 2, importance[indices], width, label="Built-in", alpha=0.8
        )
        ax.barh(
            y_pos + width / 2,
            perm_importance[indices],
            width,
            label="Permutation",
            alpha=0.8,
        )

        ax.set_yticks(y_pos)
        ax.set_yticklabels([feature_names[i] for i in indices], fontsize=8)
        ax.set_xlabel("Importance")
        ax.set_title("Built-in vs Permutation Importance (Top 20)")
        ax.legend()
        ax.grid(axis="x", alpha=0.3)

        plt.tight_layout()
        output_path = self.output_dir / "importance_method_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {output_path}")

    def save_results(self, results):
        """Save numerical results to JSON."""
        print("\nSaving numerical results...")

        output = {
            "timestamp": datetime.now().isoformat(),
            "ablations": {},
        }

        for model_name, result in results.items():
            output["ablations"][model_name] = {
                "r2": float(result["r2"]),
                "mae": float(result["mae"]),
                "rmse": float(result["rmse"]),
                "n_features": len(result["feature_names"]),
            }

            # For full model, save detailed feature importance
            if model_name == "mae_angles":
                importance = result["feature_importance"]
                feature_names = result["feature_names"]

                # Top 20 features
                top_k = min(20, len(importance))
                indices = np.argsort(importance)[::-1][:top_k]

                output["top_features"] = [
                    {
                        "name": feature_names[i],
                        "importance": float(importance[i]),
                        "permutation_importance": float(
                            result["permutation_importance"][i]
                        ),
                    }
                    for i in indices
                ]

                # Aggregate statistics
                mae_mask = np.array(["MAE" in name for name in feature_names])
                angle_mask = np.array(
                    [name in ["SZA", "SAA"] for name in feature_names]
                )

                output["aggregate_importance"] = {
                    "mae_total": float(importance[mae_mask].sum()),
                    "mae_mean": float(importance[mae_mask].mean()),
                    "angle_total": float(importance[angle_mask].sum()),
                    "angle_mean": float(importance[angle_mask].mean()),
                }

        output_path = self.output_dir / "feature_importance_analysis.json"
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"  Saved: {output_path}")

        return output

    def run_full_analysis(self):
        """Run complete feature importance analysis."""
        print("=" * 80)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("=" * 80)

        # Load model and data
        encoder = self.load_encoder()
        dataset, train_dataset, test_dataset = self.load_dataset()

        # Extract features
        print("\nExtracting train features...")
        embeddings_train, angles_train, targets_train = self.extract_features(
            encoder, train_dataset
        )

        print("Extracting test features...")
        embeddings_test, angles_test, targets_test = self.extract_features(
            encoder, test_dataset
        )

        # Run ablations
        results = self.run_ablations(
            embeddings_train,
            angles_train,
            targets_train,
            embeddings_test,
            angles_test,
            targets_test,
        )

        # Create visualizations
        self.plot_feature_importance(results)

        # Save results
        output = self.save_results(results)

        # Print summary
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)
        print(f"Results saved to: {self.output_dir}")
        print("\nKey Findings:")

        agg = output.get("aggregate_importance", {})
        if agg:
            mae_total = agg.get("mae_total", 0)
            angle_total = agg.get("angle_total", 0)
            total = mae_total + angle_total

            print(
                f"  MAE dimensions:  {mae_total:.3f} ({100 * mae_total / total:.1f}%)"
            )
            print(
                f"  Angles (SZA/SAA): {angle_total:.3f} ({100 * angle_total / total:.1f}%)"
            )

            if angle_total > mae_total:
                print("\n⚠️  Angles dominate feature importance!")
                print("   MAE embeddings contribute less to predictions.")


def main():
    parser = argparse.ArgumentParser(description="Analyze feature importance")
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
        "--output",
        type=str,
        default=None,
        help="Output directory (default: outputs/feature_importance/TIMESTAMP)",
    )

    args = parser.parse_args()

    # Set default output directory with timestamp
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"outputs/feature_importance/{timestamp}"

    # Run analysis
    analyzer = FeatureImportanceAnalyzer(args.config, args.encoder, args.output)
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()
