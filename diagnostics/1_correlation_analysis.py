#!/usr/bin/env python3
"""
Diagnostic 1: Correlation Analysis
==================================

Goal: Determine if ANY simple features correlate with optical depth.
This will tell us if the task is fundamentally learnable from the data.

Expected runtime: ~30 minutes

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
from tqdm import tqdm
import yaml

# Import project modules
from src.hdf5_dataset import HDF5CloudDataset
from src.main_utils import load_all_flights_metadata_for_scalers


def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def extract_simple_features(images, sza, saa):
    """
    Extract simple hand-crafted features from images and metadata.

    Args:
        images: (N, T, H, W) tensor of IR images
        sza: (N,) tensor of solar zenith angles
        saa: (N,) tensor of solar azimuth angles

    Returns:
        dict of feature arrays
    """
    # Convert to numpy if needed
    if torch.is_tensor(images):
        images = images.cpu().numpy()
    if torch.is_tensor(sza):
        sza = sza.cpu().numpy()
    if torch.is_tensor(saa):
        saa = saa.cpu().numpy()

    N = images.shape[0]
    features = {}

    print("Extracting intensity statistics...")
    # Mean frame (average over temporal dimension)
    mean_frame = np.mean(images, axis=1)  # (N, H, W)

    # Basic intensity features
    features["mean_intensity"] = np.mean(mean_frame, axis=(1, 2))
    features["std_intensity"] = np.std(mean_frame, axis=(1, 2))
    features["min_intensity"] = np.min(mean_frame, axis=(1, 2))
    features["max_intensity"] = np.max(mean_frame, axis=(1, 2))
    features["median_intensity"] = np.median(mean_frame, axis=(1, 2))

    print("Extracting spatial features...")
    # Spatial gradients
    grad_x = np.abs(np.diff(mean_frame, axis=2))
    grad_y = np.abs(np.diff(mean_frame, axis=1))
    features["mean_grad_x"] = np.mean(grad_x, axis=(1, 2))
    features["mean_grad_y"] = np.mean(grad_y, axis=(1, 2))
    features["std_grad_x"] = np.std(grad_x, axis=(1, 2))
    features["std_grad_y"] = np.std(grad_y, axis=(1, 2))

    print("Extracting texture features...")
    # Texture (local variance)
    local_std = []
    for i in range(N):
        # Simple local variance (5x5 windows)
        kernel_size = 5
        H, W = mean_frame[i].shape
        local_vars = []
        for y in range(0, H - kernel_size, kernel_size):
            for x in range(0, W - kernel_size, kernel_size):
                patch = mean_frame[i, y : y + kernel_size, x : x + kernel_size]
                local_vars.append(np.std(patch))
        local_std.append(np.mean(local_vars))
    features["texture_local_std"] = np.array(local_std)

    print("Extracting temporal features...")
    # Temporal variation
    if images.shape[1] > 1:
        features["temporal_std"] = np.std(images, axis=1).mean(axis=(1, 2))
        features["temporal_range"] = (
            np.max(images, axis=1) - np.min(images, axis=1)
        ).mean(axis=(1, 2))
    else:
        features["temporal_std"] = np.zeros(N)
        features["temporal_range"] = np.zeros(N)

    print("Extracting spatial distribution features...")
    # Percentiles
    features["p25_intensity"] = np.percentile(mean_frame, 25, axis=(1, 2))
    features["p75_intensity"] = np.percentile(mean_frame, 75, axis=(1, 2))
    features["iqr_intensity"] = features["p75_intensity"] - features["p25_intensity"]

    # Skewness (simple)
    mean_vals = features["mean_intensity"][:, None, None]
    std_vals = features["std_intensity"][:, None, None]
    skewness = np.mean(((mean_frame - mean_vals) / (std_vals + 1e-8)) ** 3, axis=(1, 2))
    features["skewness"] = skewness

    # Center vs edge intensity
    H, W = mean_frame.shape[1], mean_frame.shape[2]
    center_h, center_w = H // 2, W // 2
    margin = min(H, W) // 4
    center_region = mean_frame[
        :, center_h - margin : center_h + margin, center_w - margin : center_w + margin
    ]
    features["center_intensity"] = np.mean(center_region, axis=(1, 2))

    edge_width = 20
    edge_regions = np.concatenate(
        [
            mean_frame[:, :edge_width, :],  # top
            mean_frame[:, -edge_width:, :],  # bottom
            mean_frame[:, :, :edge_width],  # left
            mean_frame[:, :, -edge_width:],  # right
        ],
        axis=1,
    )
    features["edge_intensity"] = np.mean(edge_regions, axis=(1, 2))
    features["center_edge_diff"] = (
        features["center_intensity"] - features["edge_intensity"]
    )

    print("Adding metadata features...")
    # Metadata
    features["sza"] = sza.flatten()
    features["saa"] = saa.flatten()

    # Derived angle features
    features["sza_cos"] = np.cos(np.deg2rad(sza.flatten()))
    features["sza_sin"] = np.sin(np.deg2rad(sza.flatten()))
    features["saa_cos"] = np.cos(np.deg2rad(saa.flatten()))
    features["saa_sin"] = np.sin(np.deg2rad(saa.flatten()))

    print(f"Extracted {len(features)} features from {N} samples")
    return features


def compute_correlations(features, targets):
    """
    Compute Pearson and Spearman correlations for all features.

    Args:
        features: dict of feature arrays
        targets: array of target values (optical depth)

    Returns:
        DataFrame with correlation results
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
    print("DIAGNOSTIC 1: CORRELATION ANALYSIS")
    print("=" * 70)
    print()

    # Load config - use local config if on local machine, Colab config if on Colab
    if Path("/content/drive/MyDrive").exists():
        # Running on Colab
        config_path = (
            Path(__file__).parent.parent / "configs" / "colab_optimized_full_tuned.yaml"
        )
        print(f"Running on Colab")
    else:
        # Running locally
        config_path = Path(__file__).parent.parent / "configs" / "local_diagnostic.yaml"
        print(f"Running locally")

    print(f"Loading config: {config_path}")
    config = load_config(config_path)
    print(f"Data directory: {config['data_directory']}")

    # Create output directory
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True, parents=True)

    # Prepend data_directory to flight file paths (like main.py does)
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

    # Fit scalers on the global data
    sza_scaler = StandardScaler().fit(global_sza)
    saa_scaler = StandardScaler().fit(global_saa)
    y_scaler = StandardScaler().fit(global_y)

    print(f"Scalers fitted on {len(global_sza)} samples")

    # Load dataset
    print("\nLoading dataset...")
    dataset = HDF5CloudDataset(
        flight_configs=flight_configs,
        swath_slice=config.get("swath_slice", (40, 480)),
        augment=False,  # No augmentation for diagnostics
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

    # Collect all data
    print("\nLoading all samples (this may take a few minutes)...")
    all_images = []
    all_sza = []
    all_saa = []
    all_targets = []

    for i in tqdm(range(len(dataset)), desc="Loading samples"):
        # Dataset returns: (img_stack, sza_tensor, saa_tensor, y_tensor, global_idx, local_idx)
        img_stack, sza_tensor, saa_tensor, y_tensor, _, _ = dataset[i]
        all_images.append(img_stack)
        all_sza.append(sza_tensor)
        all_saa.append(saa_tensor)
        all_targets.append(y_tensor)

    # Stack into arrays
    images = torch.stack(all_images)  # (N, T, H, W)
    sza = torch.stack(all_sza)  # (N, 1)
    saa = torch.stack(all_saa)  # (N, 1)
    targets = torch.stack(all_targets).numpy()  # (N,)

    print(f"\nData shapes:")
    print(f"  Images: {images.shape}")
    print(f"  SZA: {sza.shape}")
    print(f"  SAA: {saa.shape}")
    print(f"  Targets: {targets.shape}")

    # Extract features
    print("\nExtracting hand-crafted features...")
    features = extract_simple_features(images, sza, saa)

    # Compute correlations
    print("\nComputing correlations with target (optical depth)...")
    correlation_df = compute_correlations(features, targets)

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
            "  Consider if this level of accuracy is useful for your application,\n"
            "  but don't expect high performance (R² will be low)."
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
