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

# Import project modules
from src.config_loader import load_config
from src.data_loader import UnifiedCloudDataset


def extract_simple_features(images, sza, saa):
    """
    Extract simple hand-crafted features from images and metadata.

    Args:
        images: (N, T, H, W) tensor of IR images
        sza: (N,) tensor of solar zenith angles
        saa: (N,) tensor of solar azimuth angles

    Returns:
        dict of features (each N,)
    """
    # Flatten temporal dimension if needed
    if images.ndim == 4:
        # Use middle frame or average across time
        images_2d = images[:, images.shape[1] // 2, :, :]  # Middle frame
    else:
        images_2d = images

    features = {}

    print("Extracting intensity features...")
    # Basic intensity statistics
    features["mean_intensity"] = np.mean(images_2d, axis=(1, 2))
    features["std_intensity"] = np.std(images_2d, axis=(1, 2))
    features["min_intensity"] = np.min(images_2d, axis=(1, 2))
    features["max_intensity"] = np.max(images_2d, axis=(1, 2))
    features["median_intensity"] = np.median(images_2d, axis=(1, 2))
    features["range_intensity"] = features["max_intensity"] - features["min_intensity"]

    print("Extracting spatial features...")
    # Spatial variation features
    features["horizontal_gradient"] = np.mean(
        np.abs(np.diff(images_2d, axis=2)), axis=(1, 2)
    )
    features["vertical_gradient"] = np.mean(
        np.abs(np.diff(images_2d, axis=1)), axis=(1, 2)
    )
    features["total_gradient"] = (
        features["horizontal_gradient"] + features["vertical_gradient"]
    )

    # Texture features (simple)
    features["intensity_variance"] = features["std_intensity"] ** 2

    print("Extracting percentile features...")
    # Percentiles
    features["intensity_p10"] = np.percentile(images_2d, 10, axis=(1, 2))
    features["intensity_p25"] = np.percentile(images_2d, 25, axis=(1, 2))
    features["intensity_p75"] = np.percentile(images_2d, 75, axis=(1, 2))
    features["intensity_p90"] = np.percentile(images_2d, 90, axis=(1, 2))

    # Interquartile range
    features["intensity_iqr"] = features["intensity_p75"] - features["intensity_p25"]

    print("Adding metadata features...")
    # Metadata
    features["sza"] = sza
    features["saa"] = saa
    features["cos_sza"] = np.cos(np.radians(sza))
    features["sin_sza"] = np.sin(np.radians(sza))
    features["cos_saa"] = np.cos(np.radians(saa))
    features["sin_saa"] = np.sin(np.radians(saa))

    # Interaction features
    features["intensity_x_cos_sza"] = features["mean_intensity"] * features["cos_sza"]
    features["std_x_cos_sza"] = features["std_intensity"] * features["cos_sza"]

    return features


def compute_correlations(features, targets, feature_names):
    """
    Compute correlation coefficients between features and targets.

    Returns:
        DataFrame with correlation statistics
    """
    results = []

    for name in feature_names:
        feat = features[name]

        # Remove any NaN or inf values
        valid_mask = np.isfinite(feat) & np.isfinite(targets)
        feat_valid = feat[valid_mask]
        target_valid = targets[valid_mask]

        if len(feat_valid) < 10:
            print(f"Warning: {name} has too few valid samples")
            continue

        # Pearson correlation (linear)
        r_pearson, p_pearson = pearsonr(feat_valid, target_valid)
        r2_pearson = r_pearson**2

        # Spearman correlation (monotonic, nonlinear)
        r_spearman, p_spearman = spearmanr(feat_valid, target_valid)
        r2_spearman = r_spearman**2

        results.append(
            {
                "feature": name,
                "n_samples": len(feat_valid),
                "r_pearson": r_pearson,
                "r2_pearson": r2_pearson,
                "p_pearson": p_pearson,
                "r_spearman": r_spearman,
                "r2_spearman": r2_spearman,
                "p_spearman": p_spearman,
                "feat_mean": np.mean(feat_valid),
                "feat_std": np.std(feat_valid),
            }
        )

    df = pd.DataFrame(results)
    df = df.sort_values("r2_pearson", ascending=False)

    return df


def load_data(config_path):
    """Load dataset and extract features."""
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    config = load_config(config_path)

    # Create dataset
    print("\nCreating dataset...")
    dataset = UnifiedCloudDataset(
        config=config,
        mode="test",  # Use test set for unbiased analysis
        return_metadata=True,
    )

    print(f"Dataset size: {len(dataset)} samples")

    # Extract all data
    all_images = []
    all_sza = []
    all_saa = []
    all_targets = []

    print("\nExtracting data from dataset...")
    for i in tqdm(range(len(dataset)), desc="Loading samples"):
        img_stack, sza, saa, y, _, _ = dataset[i]

        # Convert to numpy
        all_images.append(img_stack.numpy())
        all_sza.append(sza.item())
        all_saa.append(saa.item())
        all_targets.append(y.item())

    # Stack into arrays
    all_images = np.array(all_images)
    all_sza = np.array(all_sza)
    all_saa = np.array(all_saa)
    all_targets = np.array(all_targets)

    print(f"\nData shapes:")
    print(f"  Images: {all_images.shape}")
    print(f"  SZA: {all_sza.shape}")
    print(f"  SAA: {all_saa.shape}")
    print(f"  Targets: {all_targets.shape}")

    return all_images, all_sza, all_saa, all_targets


def main():
    """Main diagnostic routine."""
    print("=" * 70)
    print("DIAGNOSTIC 1: CORRELATION ANALYSIS")
    print("=" * 70)
    print("\nGoal: Determine if features correlate with optical depth")
    print("This tells us if the task is fundamentally learnable.\n")

    # Configuration
    config_path = "configs/colab_optimized_full_tuned.yaml"
    output_dir = Path("diagnostics/results")
    output_dir.mkdir(exist_ok=True, parents=True)

    # Load data
    images, sza, saa, targets = load_data(config_path)

    # Extract features
    print("\n" + "=" * 70)
    print("EXTRACTING FEATURES")
    print("=" * 70)
    features = extract_simple_features(images, sza, saa)
    feature_names = list(features.keys())
    print(f"\nExtracted {len(feature_names)} features")

    # Compute correlations
    print("\n" + "=" * 70)
    print("COMPUTING CORRELATIONS")
    print("=" * 70)
    df_corr = compute_correlations(features, targets, feature_names)

    # Save results
    csv_path = output_dir / "correlation_results.csv"
    df_corr.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("TOP 10 FEATURES (by Pearson r²)")
    print("=" * 70)
    print(df_corr.head(10).to_string(index=False))

    # Key statistics
    print("\n" + "=" * 70)
    print("KEY STATISTICS")
    print("=" * 70)

    best_pearson = df_corr.iloc[0]
    best_spearman = df_corr.sort_values("r2_spearman", ascending=False).iloc[0]

    print(f"\nBest Pearson correlation:")
    print(f"  Feature: {best_pearson['feature']}")
    print(f"  r = {best_pearson['r_pearson']:.4f}")
    print(
        f"  r² = {best_pearson['r2_pearson']:.4f} ({best_pearson['r2_pearson'] * 100:.1f}% variance explained)"
    )
    print(f"  p-value = {best_pearson['p_pearson']:.2e}")

    print(f"\nBest Spearman correlation (nonlinear):")
    print(f"  Feature: {best_spearman['feature']}")
    print(f"  ρ = {best_spearman['r_spearman']:.4f}")
    print(
        f"  ρ² = {best_spearman['r2_spearman']:.4f} ({best_spearman['r2_spearman'] * 100:.1f}% variance explained)"
    )
    print(f"  p-value = {best_spearman['p_spearman']:.2e}")

    # Count strong correlations
    strong_pearson = (df_corr["r2_pearson"] > 0.1).sum()
    moderate_pearson = (
        (df_corr["r2_pearson"] > 0.05) & (df_corr["r2_pearson"] <= 0.1)
    ).sum()
    weak_pearson = (df_corr["r2_pearson"] <= 0.05).sum()

    print(f"\nCorrelation strength distribution:")
    print(f"  Strong (r² > 0.1):   {strong_pearson} features")
    print(f"  Moderate (0.05-0.1): {moderate_pearson} features")
    print(f"  Weak (< 0.05):       {weak_pearson} features")

    # Decision
    print("\n" + "=" * 70)
    print("DECISION")
    print("=" * 70)

    max_r2 = df_corr["r2_pearson"].max()
    max_r2_spearman = df_corr["r2_spearman"].max()

    if max_r2 < 0.05 and max_r2_spearman < 0.05:
        decision = "🔴 STOP"
        explanation = "No feature shows meaningful correlation (r² < 0.05)."
        recommendation = (
            "Task likely NOT learnable from these features. Consider:\n"
            "  - Multi-wavelength IR (need CO2 absorption bands)\n"
            "  - Different target (cloud type instead of optical depth?)\n"
            "  - Additional metadata (terrain, time of day, etc.)"
        )
    elif max_r2 < 0.1:
        decision = "🟡 PROCEED WITH CAUTION"
        explanation = f"Weak correlations found (max r² = {max_r2:.3f})."
        recommendation = (
            "Signal exists but is very weak. Proceed to Diagnostic 2 (simple baselines).\n"
            "  Deep learning may extract nonlinear relationships,\n"
            "  but don't expect high performance (R² will be low)."
        )
    else:
        decision = "✅ PROCEED"
        explanation = f"Moderate correlations found (max r² = {max_r2:.3f})."
        recommendation = (
            "Signal definitely exists! Proceed to Diagnostic 2 (simple baselines).\n"
            "  Linear/RF models should get R² > 0, validating that\n"
            "  deep learning has potential to work on this task."
        )

    print(f"\n{decision}")
    print(f"\nExplanation: {explanation}")
    print(f"\nRecommendation:")
    print(f"  {recommendation}")

    # Save summary
    summary = {
        "best_pearson_feature": best_pearson["feature"],
        "best_pearson_r2": float(best_pearson["r2_pearson"]),
        "best_spearman_feature": best_spearman["feature"],
        "best_spearman_r2": float(best_spearman["r2_spearman"]),
        "n_strong": int(strong_pearson),
        "n_moderate": int(moderate_pearson),
        "n_weak": int(weak_pearson),
        "decision": decision,
        "max_r2": float(max_r2),
        "target_mean": float(np.mean(targets)),
        "target_std": float(np.std(targets)),
        "target_range": [float(np.min(targets)), float(np.max(targets))],
        "n_samples": len(targets),
    }

    json_path = output_dir / "correlation_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n\nSummary saved to: {json_path}")
    print("=" * 70)
    print("DIAGNOSTIC 1 COMPLETE")
    print("=" * 70)

    return summary


if __name__ == "__main__":
    summary = main()
