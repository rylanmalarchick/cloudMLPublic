#!/usr/bin/env python3
"""
Work Package 4: Hybrid Model Integration and Validation

This module implements the final hybrid models combining physical features
(geometric + atmospheric) with MAE spatial embeddings and solar angles.

Executes ablation studies to determine the contribution of each feature type
and validates the complete physics-constrained CBH retrieval approach.

Author: Autonomous Agent
Date: 2025
SOW: SOW-AGENT-CBH-WP-001 Section 6
"""

import sys
from pathlib import Path
import numpy as np
import h5py
import yaml
import json
import pickle
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# ML imports
try:
    import torch
    import xgboost as xgb
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    from sklearn.inspection import permutation_importance

    TORCH_AVAILABLE = True
    XGBOOST_AVAILABLE = True
except ImportError as e:
    print(f"ERROR: Required packages not available: {e}")
    TORCH_AVAILABLE = False
    XGBOOST_AVAILABLE = False

from src.hdf5_dataset import HDF5CloudDataset
from src.mae_model import MAEEncoder


@dataclass
class ModelVariant:
    """Container for model variant specification."""

    id: str
    description: str
    features: List[str]
    use_geometric: bool = False
    use_atmospheric: bool = False
    use_mae: bool = False
    use_angles: bool = False


@dataclass
class FoldMetrics:
    """Container for per-fold evaluation metrics."""

    fold_id: int
    test_flight: str
    n_train: int
    n_test: int
    r2: float
    mae_km: float
    rmse_km: float
    predictions: np.ndarray = None
    targets: np.ndarray = None


class HybridModelValidator:
    """
    Validates hybrid models combining physical and learned features.

    Implements ablation studies to understand feature contributions.
    """

    def __init__(
        self,
        wp1_features_path: str,
        wp2_features_path: str,
        config_path: str,
        mae_encoder_path: str = "outputs/mae_pretrain/mae_encoder_pretrained.pth",
        output_dir: str = "sow_outputs/wp4_hybrid",
        verbose: bool = True,
    ):
        """
        Initialize the hybrid model validator.

        Args:
            wp1_features_path: Path to WP1_Features.hdf5
            wp2_features_path: Path to WP2_Features.hdf5
            config_path: Path to config YAML
            mae_encoder_path: Path to pretrained MAE encoder
            output_dir: Directory for outputs
            verbose: Enable verbose logging
        """
        self.wp1_path = Path(wp1_features_path)
        self.wp2_path = Path(wp2_features_path)
        self.config_path = Path(config_path)
        self.mae_encoder_path = Path(mae_encoder_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose

        # Flight mapping (as per SOW Section 2.2)
        self.flight_mapping = {
            0: "30Oct24",  # F_0: n=501
            1: "10Feb25",  # F_1: n=191
            2: "23Oct24",  # F_2: n=105
            3: "12Feb25",  # F_3: n=92
            4: "18Feb25",  # F_4: n=44
        }

        # Load config
        with open(self.config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Flight name to ID mapping
        self.flight_name_to_id = {}
        for i, flight in enumerate(self.config["flights"]):
            name = flight["name"]
            if name in ["30Oct24", "10Feb25", "23Oct24", "12Feb25", "18Feb25"]:
                for fid, fname in self.flight_mapping.items():
                    if fname == name:
                        self.flight_name_to_id[name] = fid
                        break

        # Define model variants for ablation study
        self.model_variants = [
            ModelVariant(
                id="M_PHYSICAL_ONLY",
                description="Physical Baseline (control)",
                features=["geometric", "atmospheric"],
                use_geometric=True,
                use_atmospheric=True,
            ),
            ModelVariant(
                id="M_PHYSICAL_ANGLES",
                description="Physical + Solar Angles",
                features=["geometric", "atmospheric", "angles"],
                use_geometric=True,
                use_atmospheric=True,
                use_angles=True,
            ),
            ModelVariant(
                id="M_PHYSICAL_MAE",
                description="Physical + MAE Spatial",
                features=["geometric", "atmospheric", "mae_spatial"],
                use_geometric=True,
                use_atmospheric=True,
                use_mae=True,
            ),
            ModelVariant(
                id="M_HYBRID_FULL",
                description="Full Hybrid (all features)",
                features=["geometric", "atmospheric", "mae_spatial", "angles"],
                use_geometric=True,
                use_atmospheric=True,
                use_mae=True,
                use_angles=True,
            ),
        ]

    def load_mae_encoder(self) -> torch.nn.Module:
        """
        Load pretrained MAE encoder.

        Returns:
            MAE encoder model
        """
        if self.verbose:
            print(f"\nLoading MAE encoder from: {self.mae_encoder_path}")

        if not self.mae_encoder_path.exists():
            raise FileNotFoundError(f"MAE encoder not found: {self.mae_encoder_path}")

        # Load checkpoint
        checkpoint = torch.load(self.mae_encoder_path, map_location="cpu")

        # Initialize encoder
        encoder = MAEEncoder(
            img_size=256,
            patch_size=16,
            in_chans=3,
            embed_dim=256,
            depth=6,
            num_heads=8,
        )

        # Load weights
        if "model_state_dict" in checkpoint:
            encoder.load_state_dict(checkpoint["model_state_dict"], strict=False)
        elif "encoder_state_dict" in checkpoint:
            encoder.load_state_dict(checkpoint["encoder_state_dict"], strict=False)
        else:
            # Try loading directly
            encoder.load_state_dict(checkpoint, strict=False)

        encoder.eval()

        if self.verbose:
            print(f"  MAE encoder loaded successfully")

        return encoder

    def extract_mae_embeddings(self, encoder: torch.nn.Module) -> np.ndarray:
        """
        Extract MAE spatial embeddings using global average pooling.

        CRITICAL: Uses patch tokens with global pooling, NOT CLS token.

        Args:
            encoder: Pretrained MAE encoder

        Returns:
            Embedding matrix (n_samples, embedding_dim)
        """
        if self.verbose:
            print("\nExtracting MAE spatial embeddings...")
            print("  Method: Global average pooling of patch tokens (NOT CLS token)")

        # Load dataset
        dataset = HDF5CloudDataset(
            flight_configs=self.config["flights"],
            indices=None,
            augment=False,
            swath_slice=self.config.get("swath_slice", [40, 480]),
        )

        embeddings_list = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoder = encoder.to(device)

        with torch.no_grad():
            for i in range(len(dataset)):
                # Get image
                img_stack, _, _, _, _, _ = dataset[i]

                # Use center frame
                img = img_stack[1:2, :, :]  # Shape: (1, H, W)
                img = img.unsqueeze(0)  # Shape: (1, 1, H, W)

                # Expand to 3 channels if needed
                if img.shape[1] == 1:
                    img = img.repeat(1, 3, 1, 1)

                img = img.to(device)

                # Forward through encoder
                embeddings = encoder(img)  # Shape: (1, N_patches, embed_dim)

                # Global average pooling over patches (NOT CLS token)
                # This is the critical difference from failed approaches
                spatial_embedding = embeddings.mean(dim=1)  # Shape: (1, embed_dim)

                embeddings_list.append(spatial_embedding.cpu().numpy())

                if self.verbose and (i + 1) % 100 == 0:
                    print(f"  Processed {i + 1}/{len(dataset)} samples")

        embeddings_array = np.vstack(embeddings_list)

        if self.verbose:
            print(f"  Extracted embeddings shape: {embeddings_array.shape}")
            print(f"  Embedding dimension: {embeddings_array.shape[1]}")

        return embeddings_array

    def load_all_features(
        self, include_mae: bool = True
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, List[str]]:
        """
        Load and organize all feature types.

        Args:
            include_mae: Whether to extract MAE embeddings

        Returns:
            feature_dict: Dictionary of feature arrays by type
            y: Target CBH in km
            flight_ids: Flight ID for each sample
            all_feature_names: List of all feature names
        """
        if self.verbose:
            print("\n" + "=" * 80)
            print("Loading All Features")
            print("=" * 80)

        feature_dict = {}
        all_feature_names = []

        # Load WP1 geometric features
        with h5py.File(self.wp1_path, "r") as f:
            n_samples = len(f["sample_id"])

            geometric_features = []
            geometric_names = []

            for key in [
                "derived_geometric_H",
                "shadow_length_pixels",
                "shadow_detection_confidence",
            ]:
                if key in f:
                    geometric_features.append(f[key][:])
                    geometric_names.append(f"geo_{key}")

            feature_dict["geometric"] = (
                np.column_stack(geometric_features)
                if geometric_features
                else np.zeros((n_samples, 0))
            )
            all_feature_names.extend(geometric_names)

            if self.verbose:
                print(f"\nGeometric features: {len(geometric_names)}")

        # Load WP2 atmospheric features
        with h5py.File(self.wp2_path, "r") as f:
            atmospheric_features = []
            atmospheric_names = []

            for key in [
                "blh_m",
                "lcl_m",
                "inversion_height_m",
                "moisture_gradient",
                "stability_index",
            ]:
                if key in f:
                    atmospheric_features.append(f[key][:])
                    atmospheric_names.append(f"atm_{key}")

            feature_dict["atmospheric"] = (
                np.column_stack(atmospheric_features)
                if atmospheric_features
                else np.zeros((n_samples, 0))
            )
            all_feature_names.extend(atmospheric_names)

            if self.verbose:
                print(f"Atmospheric features: {len(atmospheric_names)}")

        # Load dataset for angles and targets
        dataset = HDF5CloudDataset(
            flight_configs=self.config["flights"],
            indices=None,
            augment=False,
            swath_slice=self.config.get("swath_slice", [40, 480]),
        )

        # Extract angles
        sza_list = []
        saa_list = []
        flight_ids = np.zeros(len(dataset), dtype=int)

        for i in range(len(dataset)):
            _, sza_tensor, saa_tensor, _, global_idx, _ = dataset[i]
            sza_list.append(sza_tensor.item())
            saa_list.append(saa_tensor.item())

            flight_idx, _ = dataset.global_to_local[int(global_idx)]
            flight_name = dataset.flight_data[flight_idx]["name"]
            if flight_name in self.flight_name_to_id:
                flight_ids[i] = self.flight_name_to_id[flight_name]
            else:
                flight_ids[i] = flight_idx

        feature_dict["angles"] = np.column_stack([sza_list, saa_list])
        angle_names = ["angle_sza", "angle_saa"]
        all_feature_names.extend(angle_names)

        if self.verbose:
            print(f"Angle features: {len(angle_names)}")

        # Extract MAE embeddings if requested
        if include_mae:
            encoder = self.load_mae_encoder()
            mae_embeddings = self.extract_mae_embeddings(encoder)
            feature_dict["mae"] = mae_embeddings

            mae_names = [f"mae_dim{i}" for i in range(mae_embeddings.shape[1])]
            all_feature_names.extend(mae_names)

            if self.verbose:
                print(f"MAE features: {len(mae_names)}")
        else:
            feature_dict["mae"] = np.zeros((n_samples, 0))

        # Get targets
        y = dataset.get_unscaled_y()

        if self.verbose:
            print(f"\nTotal samples: {len(y)}")
            print(f"Total features: {len(all_feature_names)}")

        return feature_dict, y, flight_ids, all_feature_names

    def build_feature_matrix(
        self, feature_dict: Dict[str, np.ndarray], variant: ModelVariant
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Build feature matrix for a specific model variant.

        Args:
            feature_dict: Dictionary of all features
            variant: Model variant specification

        Returns:
            X: Feature matrix
            feature_names: List of feature names
        """
        features_to_include = []
        names_to_include = []

        if variant.use_geometric:
            features_to_include.append(feature_dict["geometric"])
            n_geo = feature_dict["geometric"].shape[1]
            names_to_include.extend([f"geo_{i}" for i in range(n_geo)])

        if variant.use_atmospheric:
            features_to_include.append(feature_dict["atmospheric"])
            n_atm = feature_dict["atmospheric"].shape[1]
            names_to_include.extend([f"atm_{i}" for i in range(n_atm)])

        if variant.use_mae:
            features_to_include.append(feature_dict["mae"])
            n_mae = feature_dict["mae"].shape[1]
            names_to_include.extend([f"mae_{i}" for i in range(n_mae)])

        if variant.use_angles:
            features_to_include.append(feature_dict["angles"])
            names_to_include.extend(["sza", "saa"])

        X = (
            np.hstack(features_to_include)
            if features_to_include
            else np.zeros((len(feature_dict["geometric"]), 0))
        )

        return X, names_to_include

    def train_gbdt(self, X_train: np.ndarray, y_train: np.ndarray) -> xgb.XGBRegressor:
        """
        Train GBDT model.

        Args:
            X_train: Training features
            y_train: Training targets

        Returns:
            Trained XGBoost model
        """
        params = {
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
            "gamma": 0.1,
            "random_state": 42,
            "n_jobs": -1,
            "tree_method": "hist",
        }

        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)

        return model

    def evaluate_fold(
        self, model: xgb.XGBRegressor, X_test: np.ndarray, y_test: np.ndarray
    ) -> Tuple[float, float, float, np.ndarray]:
        """
        Evaluate model on test set.

        Args:
            model: Trained model
            X_test: Test features
            y_test: Test targets

        Returns:
            r2, mae, rmse, predictions
        """
        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        return r2, mae, rmse, y_pred

    def run_loo_cv_for_variant(
        self,
        variant: ModelVariant,
        feature_dict: Dict[str, np.ndarray],
        y: np.ndarray,
        flight_ids: np.ndarray,
    ) -> List[FoldMetrics]:
        """
        Run LOO CV for a specific model variant.

        Args:
            variant: Model variant
            feature_dict: All features
            y: Targets
            flight_ids: Flight IDs

        Returns:
            List of fold metrics
        """
        if self.verbose:
            print(f"\n{'=' * 80}")
            print(f"Variant: {variant.id}")
            print(f"Description: {variant.description}")
            print(f"Features: {', '.join(variant.features)}")
            print(f"{'=' * 80}")

        # Build feature matrix
        X, feature_names = self.build_feature_matrix(feature_dict, variant)

        # Handle missing values
        X_clean = X.copy()
        for j in range(X.shape[1]):
            col = X[:, j]
            if np.any(np.isnan(col)) or np.any(np.isinf(col)):
                median_val = np.nanmedian(col[np.isfinite(col)])
                X_clean[~np.isfinite(col), j] = median_val

        fold_results = []

        # Run LOO CV
        for test_flight_id in range(5):
            test_flight_name = self.flight_mapping[test_flight_id]

            test_mask = flight_ids == test_flight_id
            train_mask = ~test_mask

            X_train = X_clean[train_mask]
            y_train = y[train_mask]
            X_test = X_clean[test_mask]
            y_test = y[test_mask]

            if len(y_test) == 0 or len(y_train) < 10:
                continue

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train
            model = self.train_gbdt(X_train_scaled, y_train)

            # Evaluate
            r2, mae, rmse, y_pred = self.evaluate_fold(model, X_test_scaled, y_test)

            if self.verbose:
                print(
                    f"  Fold {test_flight_id} ({test_flight_name}): R²={r2:.4f}, MAE={mae:.4f} km"
                )

            fold_results.append(
                FoldMetrics(
                    fold_id=test_flight_id,
                    test_flight=test_flight_name,
                    n_train=len(y_train),
                    n_test=len(y_test),
                    r2=r2,
                    mae_km=mae,
                    rmse_km=rmse,
                    predictions=y_pred,
                    targets=y_test,
                )
            )

        return fold_results

    def compute_feature_importance(
        self,
        feature_dict: Dict[str, np.ndarray],
        y: np.ndarray,
        flight_ids: np.ndarray,
        top_k: int = 20,
    ) -> Dict:
        """
        Compute feature importance for best model (M_HYBRID_FULL).

        Args:
            feature_dict: All features
            y: Targets
            flight_ids: Flight IDs
            top_k: Number of top features to report

        Returns:
            Feature importance dictionary
        """
        if self.verbose:
            print(f"\n{'=' * 80}")
            print("Computing Feature Importance (M_HYBRID_FULL)")
            print(f"{'=' * 80}")

        # Use full hybrid model
        variant = self.model_variants[-1]  # M_HYBRID_FULL
        X, feature_names = self.build_feature_matrix(feature_dict, variant)

        # Clean data
        X_clean = X.copy()
        for j in range(X.shape[1]):
            col = X[:, j]
            if np.any(np.isnan(col)) or np.any(np.isinf(col)):
                median_val = np.nanmedian(col[np.isfinite(col)])
                X_clean[~np.isfinite(col), j] = median_val

        # Train on all data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_clean)

        model = self.train_gbdt(X_scaled, y)

        # Get feature importances
        importances = model.feature_importances_

        # Sort by importance
        indices = np.argsort(importances)[::-1]

        # Categorize features
        def categorize_feature(name):
            if name.startswith("geo_"):
                return "Geometric"
            elif name.startswith("atm_"):
                return "Atmospheric"
            elif name.startswith("mae_"):
                return "MAE"
            elif name in ["sza", "saa"]:
                return "Angle"
            else:
                return "Unknown"

        # Build importance report
        top_features = []
        for i in range(min(top_k, len(indices))):
            idx = indices[i]
            top_features.append(
                {
                    "rank": i + 1,
                    "feature": feature_names[idx],
                    "importance": float(importances[idx]),
                    "category": categorize_feature(feature_names[idx]),
                }
            )

        # Category summary
        category_importance = {"Geometric": 0, "Atmospheric": 0, "MAE": 0, "Angle": 0}
        for i, idx in enumerate(indices):
            category = categorize_feature(feature_names[idx])
            category_importance[category] += importances[idx]

        if self.verbose:
            print(f"\nTop {top_k} Features:")
            for feat in top_features[:10]:
                print(
                    f"  {feat['rank']:2d}. {feat['feature']:30s} {feat['importance']:.4f} ({feat['category']})"
                )

            print(f"\nCategory Summary:")
            for cat, imp in sorted(
                category_importance.items(), key=lambda x: x[1], reverse=True
            ):
                print(f"  {cat:15s}: {imp:.4f}")

        return {
            "top_features": top_features,
            "category_importance": category_importance,
            "n_features": len(feature_names),
        }

    def save_final_features(self, feature_dict: Dict[str, np.ndarray], y: np.ndarray):
        """
        Save consolidated feature store.

        Args:
            feature_dict: All features
            y: Targets
        """
        output_path = self.output_dir / "final_features.hdf5"

        with h5py.File(output_path, "w") as f:
            # Save each feature type
            for key, features in feature_dict.items():
                f.create_dataset(key, data=features, compression="gzip")

            # Save target
            f.create_dataset("target_cbh_km", data=y, compression="gzip")

            # Metadata
            f.attrs["n_samples"] = len(y)
            f.attrs["timestamp"] = datetime.now().isoformat()

        if self.verbose:
            print(f"\nFinal features saved to: {output_path}")

    def generate_report(
        self, results: Dict[str, List[FoldMetrics]], feature_importance: Dict
    ) -> Dict:
        """
        Generate comprehensive WP4 report.

        Args:
            results: Results for all variants
            feature_importance: Feature importance analysis

        Returns:
            Complete report
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "model_variants": {},
            "feature_importance": feature_importance,
        }

        for variant_id, fold_results in results.items():
            r2_values = [f.r2 for f in fold_results]
            mae_values = [f.mae_km for f in fold_results]
            rmse_values = [f.rmse_km for f in fold_results]

            report["model_variants"][variant_id] = {
                "description": next(
                    v.description for v in self.model_variants if v.id == variant_id
                ),
                "n_folds": len(fold_results),
                "aggregate_metrics": {
                    "mean_r2": float(np.mean(r2_values)),
                    "std_r2": float(np.std(r2_values)),
                    "mean_mae_km": float(np.mean(mae_values)),
                    "std_mae_km": float(np.std(mae_values)),
                    "mean_rmse_km": float(np.mean(rmse_values)),
                    "std_rmse_km": float(np.std(rmse_values)),
                },
                "folds": [
                    {
                        "fold_id": f.fold_id,
                        "test_flight": f.test_flight,
                        "r2": f.r2,
                        "mae_km": f.mae_km,
                        "rmse_km": f.rmse_km,
                    }
                    for f in fold_results
                ],
            }

        return report

    def run(self) -> Dict:
        """
        Execute complete WP-4 pipeline.

        Returns:
            Complete report
        """
        if self.verbose:
            print("\n" + "=" * 80)
            print("WORK PACKAGE 4: HYBRID MODEL INTEGRATION")
            print("=" * 80)

        # Load all features
        feature_dict, y, flight_ids, all_feature_names = self.load_all_features(
            include_mae=True
        )

        # Save final features
        self.save_final_features(feature_dict, y)

        # Run ablation studies
        results = {}
        for variant in self.model_variants:
            fold_results = self.run_loo_cv_for_variant(
                variant, feature_dict, y, flight_ids
            )
            results[variant.id] = fold_results

        # Feature importance
        feature_importance = self.compute_feature_importance(
            feature_dict, y, flight_ids
        )

        # Generate report
        report = self.generate_report(results, feature_importance)

        # Save reports
        with open(self.output_dir / "WP4_Report.json", "w") as f:
            json.dump(report, f, indent=2)

        with open(self.output_dir / "WP4_Feature_Importance.json", "w") as f:
            json.dump(feature_importance, f, indent=2)

        if self.verbose:
            print(f"\nReports saved to: {self.output_dir}")

        return report


def main():
    """Main execution."""
    import argparse

    parser = argparse.ArgumentParser(description="WP-4: Hybrid Model Integration")
    parser.add_argument(
        "--wp1-features",
        type=str,
        default="sow_outputs/wp1_geometric/WP1_Features.hdf5",
    )
    parser.add_argument(
        "--wp2-features",
        type=str,
        default="sow_outputs/wp2_atmospheric/WP2_Features.hdf5",
    )
    parser.add_argument("--config", type=str, default="configs/bestComboConfig.yaml")
    parser.add_argument(
        "--mae-encoder",
        type=str,
        default="outputs/mae_pretrain/mae_encoder_pretrained.pth",
    )
    parser.add_argument("--output-dir", type=str, default="sow_outputs/wp4_hybrid")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    validator = HybridModelValidator(
        wp1_features_path=args.wp1_features,
        wp2_features_path=args.wp2_features,
        config_path=args.config,
        mae_encoder_path=args.mae_encoder,
        output_dir=args.output_dir,
        verbose=args.verbose,
    )

    report = validator.run()

    return 0


if __name__ == "__main__":
    sys.exit(main())
