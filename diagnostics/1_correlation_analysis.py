#!/usr/bin/env python3
"""
Diagnostic 1: Correlation Analysis (Batch-Processed)
=====================================================

Goal: Determine if ANY simple features correlate with optical depth.
This version processes data in batches to avoid OOM errors.

Expected runtime: ~15-20 minutes

Success criteria:
- If ANY feature has r² > 0.1 → Signal exists, proceed to D2
- If ALL features have r² < 0.05 → Task likely not learnable from these features
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from pathlib import Path
import json
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

# Import project modules
from src.hdf5_dataset import HDF5CloudDataset
from src.main_utils import load_all_flights_metadata_for_scalers


def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def extract_simple_features_batch(images, sza, saa):
    """
    Extract simple hand-crafted features from a BATCH of images and metadata.
    """
    if torch.is_tensor(images):
        images = images.cpu().numpy()
    if torch.is_tensor(sza):
        sza = sza.cpu().numpy()
    if torch.is_tensor(saa):
        saa = saa.cpu().numpy()

    N = images.shape[0]
    features = {}

    # Mean frame (average over temporal dimension)
    mean_frame = np.mean(images, axis=1)  # (N, H, W)

    # Basic intensity features
    features["mean_intensity"] = np.mean(mean_frame, axis=(1, 2))
    features["std_intensity"] = np.std(mean_frame, axis=(1, 2))
    features["min_intensity"] = np.min(mean_frame, axis=(1, 2))
    features["max_intensity"] = np.max(mean_frame, axis=(1, 2))
    features["median_intensity"] = np.median(mean_frame, axis=(1, 2))

    # Spatial gradients
    grad_x = np.abs(np.diff(mean_frame, axis=2))
    grad_y = np.abs(np.diff(mean_frame, axis=1))
    features["mean_grad_x"] = np.mean(grad_x, axis=(1, 2))
    features["mean_grad_y"] = np.mean(grad_y, axis=(1, 2))
    features["std_grad_x"] = np.std(grad_x, axis=(1, 2))
    features["std_grad_y"] = np.std(grad_y, axis=(1, 2))

    # Temporal variation
    if images.shape[1] > 1:
        features["temporal_std"] = np.std(images, axis=1).mean(axis=(1, 2))
        features["temporal_range"] = (
            np.max(images, axis=1) - np.min(images, axis=1)
        ).mean(axis=(1, 2))
    else:
        features["temporal_std"] = np.zeros(N)
        features["temporal_range"] = np.zeros(N)

    # Percentiles
    features["p25_intensity"] = np.percentile(mean_frame, 25, axis=(1, 2))
    features["p75_intensity"] = np.percentile(mean_frame, 75, axis=(1, 2))
    features["iqr_intensity"] = features["p75_intensity"] - features["p25_intensity"]

    # Metadata
    features["sza"] = sza.flatten()
    features["saa"] = saa.flatten()
    features["sza_cos"] = np.cos(np.deg2rad(sza.flatten()))
    features["sza_sin"] = np.sin(np.deg2rad(sza.flatten()))
    features["saa_cos"] = np.cos(np.deg2rad(saa.flatten()))
    features["saa_sin"] = np.sin(np.deg2rad(saa.flatten()))

    return features


def compute_correlations(features, targets):
    """
    Compute Pearson and Spearman correlations for all features.
    """
    results = []
    for feature_name, feature_values in tqdm(
        features.items(), desc="Computing correlations"
    ):
        # Pearson
        pearson_r, pearson_p = pearsonr(feature_values, targets)
        pearson_r2 = pearson_r**2

        # Spearman
        spearman_rho, spearman_p = spearmanr(feature_values, targets)
        spearman_r2 = spearman_rho**2

        results.append(
            {
                "feature": feature_name,
                "pearson_r": pearson_r,
                "pearson_r2": pearson_r2,
                "pearson_p": pearson_p,
                "spearman_rho": spearman_rho,
                "spearman_r2": spearman_r2,
                "spearman_p": spearman_p,
                "feature_mean": np.mean(feature_values),
                "feature_std": np.std(feature_values),
            }
        )

    df = pd.DataFrame(results)
    df = df.sort_values("pearson_r2", ascending=False)
    return df


def main():
    print("=" * 70)
    print("DIAGNOSTIC 1: CORRELATION ANALYSIS (BATCHED)")
    print("=" * 70)
    print()

    # Load config
    if Path("/content/drive/MyDrive").exists():
        config_path = (
            Path(__file__).parent.parent / "configs" / "colab_optimized_full_tuned.yaml"
        )
        print("Running on Colab")
    else:
        config_path = Path(__file__).parent.parent / "configs" / "local_diagnostic.yaml"
        print("Running locally")

    print(f"Loading config: {config_path}")
    config = load_config(config_path)
    print(f"Data directory: {config['data_directory']}")

    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True, parents=True)

    # Prepend data_directory to flight file paths
    print("\nLoading flight metadata for scalers...")
    data_dir = config["data_directory"]
    flight_configs = []
    for flight in config["flights"]:
        flight_configs.append(
            {
                "name": flight["name"],
                "iFileName": os.path.join(data_dir, flight["iFileName"]),
                "cFileName": os.path.join(data_dir, flight["cFileName"]),
                "nFileName": os.path.join(data_dir, flight["nFileName"]),
            }
        )

    # Load metadata arrays and fit scalers
    from sklearn.preprocessing import StandardScaler

    global_sza, global_saa, global_y = load_all_flights_metadata_for_scalers(
        flight_configs,
        swath_slice=config.get("swath_slice", (40, 480)),
        filter_type=config.get("filter_type", "basic"),
        cbh_min=config.get("cbh_min"),
        cbh_max=config.get("cbh_max"),
    )

    sza_scaler = StandardScaler().fit(global_sza)
    saa_scaler = StandardScaler().fit(global_saa)
    y_scaler = StandardScaler().fit(global_y)
    print(f"Scalers fitted on {len(global_sza)} samples")

    # Create dataset
    print("\nLoading dataset...")
    dataset = HDF5CloudDataset(
        flight_configs=flight_configs,
        swath_slice=config.get("swath_slice", (40, 480)),
        augment=False,
        temporal_frames=config.get("temporal_frames", 3),
        filter_type=config.get("filter_type", "basic"),
        cbh_min=config.get("cbh_min"),
        cbh_max=config.get("cbh_max"),
        sza_scaler=sza_scaler,
        saa_scaler=saa_scaler,
        y_scaler=y_scaler,
        flat_field_correction=config.get("flat_field_correction", True),
        clahe_clip_limit=config.get("clahe_clip_limit", 0.01),
        zscore_normalize=config.get("zscore_normalize", True),
        angles_mode=config.get("angles_mode", "both"),
    )
    print(f"\nDataset size: {len(dataset)} samples")

    # Create DataLoader for batch processing
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # Process in batches
    print("\nExtracting features in batches to conserve memory...")
    all_features_list = []
    all_targets_list = []

    for batch in tqdm(dataloader, desc="Processing batches"):
        img_stack, sza_tensor, saa_tensor, y_tensor, _, _ = batch

        # Extract features for the current batch
        batch_features = extract_simple_features_batch(
            img_stack, sza_tensor, saa_tensor
        )
        all_features_list.append(batch_features)

        # Store targets
        all_targets_list.append(y_tensor.cpu().numpy())

    # Consolidate features from all batches
    print("\nConsolidating features...")
    consolidated_features = {}
    if all_features_list:
        feature_names = all_features_list[0].keys()
        for name in feature_names:
            consolidated_features[name] = np.concatenate(
                [f[name] for f in all_features_list]
            )

    # Consolidate targets
    targets = np.concatenate(all_targets_list).flatten()

    print(f"Features extracted for {len(targets)} samples.")

    # Compute correlations
    print("\nComputing correlations with target (optical depth)...")
    correlation_df = compute_correlations(consolidated_features, targets)

    # Save results
    csv_path = output_dir / "correlation_results.csv"
    correlation_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

    # Print top results
    print("\n" + "=" * 70)
    print("TOP 10 FEATURES (by Pearson r²)")
    print("=" * 70)
    print(correlation_df.head(10).to_string(index=False))

    # Summary statistics
    max_r2 = correlation_df["pearson_r2"].max()
    max_feature = correlation_df.iloc[0]["feature"]

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total features tested: {len(correlation_df)}")
    print(f"Best feature: {max_feature}")
    print(f"Max Pearson r²: {max_r2:.4f} ({max_r2 * 100:.2f}% variance explained)")
    print(f"Features with r² > 0.10: {(correlation_df['pearson_r2'] > 0.10).sum()}")
    print(f"Features with r² > 0.05: {(correlation_df['pearson_r2'] > 0.05).sum()}")

    # Decision logic
    print("\n" + "=" * 70)
    print("DECISION")
    print("=" * 70)

    if max_r2 < 0.05:
        decision = "STOP - NOT LEARNABLE"
        explanation = (
            f"No features show meaningful correlation (max r² = {max_r2:.3f})."
        )
        recommendation = (
            "Task likely NOT learnable from these features.\n"
            "  Consider:\n"
            "  - Multi-wavelength IR data (CO2 absorption bands)\n"
            "  - Different target variable (cloud type instead of depth?)\n"
            "  - Additional sensors/data sources\n"
            "  - Reformulate as classification task"
        )
    elif max_r2 < 0.1:
        decision = "CAUTION - WEAK SIGNAL"
        explanation = f"Weak correlations found (max r² = {max_r2:.3f})."
        recommendation = (
            "Weak signal detected. Proceed to Diagnostic 2, but expect:\n"
            "  - Simple models: R² = 0.0-0.15\n"
            "  - Neural nets: R² = 0.0-0.20 (if training works)\n"
            "  - Low performance overall\n"
            "  - Consider if this level of accuracy is useful for your application"
        )
    else:
        decision = "PROCEED"
        explanation = f"Moderate correlations found (max r² = {max_r2:.3f})."
        recommendation = (
            "Signal definitely exists! Proceed to Diagnostic 2 (simple baselines).\n"
            "  Linear/RF models should get R² > 0, validating that\n"
            "  deep learning has potential to work on this task."
        )

    print(f"\nDecision: {decision}")
    print(f"\n{explanation}")
    print(f"\nRecommendation:\n{recommendation}")

    # Save summary
    summary = {
        "total_features": len(correlation_df),
        "total_samples": len(targets),
        "best_feature": max_feature,
        "max_pearson_r2": float(max_r2),
        "features_above_0.10": int((correlation_df["pearson_r2"] > 0.10).sum()),
        "features_above_0.05": int((correlation_df["pearson_r2"] > 0.05).sum()),
        "decision": decision,
        "explanation": explanation,
        "recommendation": recommendation,
        "target_stats": {
            "mean": float(np.mean(targets)),
            "std": float(np.std(targets)),
            "min": float(np.min(targets)),
            "max": float(np.max(targets)),
        },
    }

    json_path = output_dir / "correlation_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {json_path}")

    print("\n" + "=" * 70)
    print("NEXT STEP")
    print("=" * 70)
    if "STOP" in decision:
        print("Do NOT proceed to neural network training.")
        print("Consult with domain experts about data collection.")
    else:
        print("Run Diagnostic 2 to test simple baseline models:")
        print("  python diagnostics/2_simple_baselines.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
