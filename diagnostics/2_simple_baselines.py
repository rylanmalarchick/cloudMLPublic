#!/usr/bin/env python3
"""
Diagnostic 2: Simple Baseline Models
====================================

Goal: Determine if simple, classical ML models can achieve R² > 0.
This validates that a learnable signal exists and provides a performance
baseline for the neural network.

Expected runtime: ~1 hour (mostly for feature extraction)

Success criteria:
- If best R² > 0.1 → Signal exists, NN should work.
- If best R² < 0 → Data lacks learnable signal for any model.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from pathlib import Path
import json
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

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
    mean_frame = np.mean(images, axis=1)

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


def main():
    print("=" * 70)
    print("DIAGNOSTIC 2: SIMPLE BASELINE MODELS")
    print("=" * 70)
    print()

    # --- 1. Load Config and Data ---
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

    dataloader = DataLoader(
        dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True
    )

    # --- 2. Extract Features (Batch Processing) ---
    print("\nExtracting features in batches...")
    all_features_list = []
    all_targets_list = []
    for batch in tqdm(dataloader, desc="Processing batches"):
        img_stack, sza_tensor, saa_tensor, y_tensor, _, _ = batch
        batch_features = extract_simple_features_batch(
            img_stack, sza_tensor, saa_tensor
        )
        all_features_list.append(batch_features)
        all_targets_list.append(y_tensor.cpu().numpy())

    print("\nConsolidating features...")
    consolidated_features = {}
    if all_features_list:
        feature_names = all_features_list[0].keys()
        for name in feature_names:
            consolidated_features[name] = np.concatenate(
                [f[name] for f in all_features_list]
            )

    # Use unscaled targets for evaluation
    targets_unscaled = dataset.get_unscaled_y()
    X = pd.DataFrame(consolidated_features)
    y = targets_unscaled

    print(f"Feature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")

    # --- 3. Train and Evaluate Models ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale features
    feature_scaler = StandardScaler()
    X_train_scaled = feature_scaler.fit_transform(X_train)
    X_test_scaled = feature_scaler.transform(X_test)

    models = {
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.01),
        "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5),
        "SVR": SVR(kernel="rbf", C=1.0, epsilon=0.1),
        "RandomForest": RandomForestRegressor(
            n_estimators=100, random_state=42, max_depth=10, n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=100, random_state=42, max_depth=5
        ),
    }

    results = []
    print("\n" + "=" * 70)
    print("TRAINING AND EVALUATING BASELINE MODELS")
    print("=" * 70)
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        variance_ratio = np.var(y_pred) / np.var(y_test) if np.var(y_test) > 0 else 0

        results.append(
            {
                "model": name,
                "r2": r2,
                "mae": mae,
                "rmse": rmse,
                "variance_ratio": variance_ratio,
            }
        )
        print(f"  {name} Results: R²={r2:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}")

    # --- 4. Summarize and Decide ---
    df_results = pd.DataFrame(results).sort_values("r2", ascending=False)
    csv_path = output_dir / "baseline_results.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

    print("\n" + "=" * 70)
    print("BASELINE MODEL PERFORMANCE SUMMARY")
    print("=" * 70)
    print(df_results.to_string(index=False))

    best_model_res = df_results.iloc[0]
    best_model_name = best_model_res["model"]
    best_r2 = best_model_res["r2"]

    print("\n" + "=" * 70)
    print("COMPARISON WITH YOUR NEURAL NETWORK RUNS")
    print("=" * 70)
    print(f"Best simple model ({best_model_name}): R² = {best_r2:.4f}")
    print(f"Run 1 (neural net): R² = -0.0457")
    print(f"Run 2 (neural net): R² = -0.0226")
    print(f"Run 3 (neural net): R² = -0.2034")
    print(f"Run 4 (neural net): R² = -0.0655")

    if best_r2 > 0:
        print(f"\nCRITICAL: Simple models BEAT all your neural network runs!")
        print(
            f"   This means your neural network is UNDERPERFORMING due to training issues,"
        )
        print(f"   NOT because the task is impossible.")
    else:
        print(f"\nBoth simple models and neural networks fail (R² < 0).")
        print(f"   This suggests fundamental data/feature limitations.")

    print("\n" + "=" * 70)
    print("DECISION")
    print("=" * 70)

    if best_r2 < 0.0:
        decision = "STOP - NOT LEARNABLE"
        explanation = (
            f"No simple model could beat the mean baseline (best R² = {best_r2:.4f})."
        )
        recommendation = (
            "The features do not seem to contain a learnable signal for regression.\n"
            "  - STOP neural network experiments with the current setup.\n"
            "  - Re-evaluate the problem: Is classification (low/med/high) better?\n"
            "  - Consult domain experts about needing multi-spectral data."
        )
    elif best_r2 < 0.1:
        decision = "CAUTION - WEAK SIGNAL"
        explanation = f"Signal is very weak (best R² = {best_r2:.4f})."
        recommendation = (
            "Proceed to neural network training, but with low expectations.\n"
            "  - NN might achieve R² = 0.0-0.2, but high performance is unlikely.\n"
            "  - Consider if this level of accuracy is useful."
        )
    elif best_r2 < 0.3:
        decision = "LEARNABLE (MODERATE)"
        explanation = f"Best model R² = {best_r2:.4f} (moderate performance)"
        recommendation = (
            "Signal definitely exists! Simple models can learn patterns.\n\n"
            "  Deep learning should achieve R² = 0.2-0.4 with proper tuning.\n"
            "  Proceed to:\n"
            "  - Run 5 with increased variance lambda\n"
            "  - Expected: Useful but not perfect predictions"
        )
    else:
        decision = "STRONGLY LEARNABLE"
        explanation = f"Best model R² = {best_r2:.4f} (good performance!)"
        recommendation = (
            "Strong signal! Simple models work well.\n\n"
            "  Deep learning should achieve R² > 0.4 (possibly >0.5).\n"
            "  Your previous runs' failure is likely due to:\n"
            "  - Training issues (variance collapse, etc.)\n"
            "  - NOT a fundamental data problem\n\n"
            "  Proceed with confidence to Run 5 or architecture tuning!"
        )

    print(f"\nDecision: {decision}")
    print(f"\n{explanation}")
    print(f"\nRecommendation:\n{recommendation}")

    summary = {
        "best_model": best_model_name,
        "best_r2": float(best_r2),
        "decision": decision,
        "explanation": explanation,
        "recommendation": recommendation,
        "all_model_results": df_results.to_dict("records"),
    }

    json_path = output_dir / "baseline_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {json_path}")

    print("\n" + "=" * 70)
    print("NEXT STEP")
    print("=" * 70)
    if "STOP" in decision:
        print("Do NOT proceed to neural network training.")
        print(
            "Consult with domain experts about data collection or problem formulation."
        )
    else:
        print("Proceed to neural network training in the Colab notebook.")
        print("Use the 'OPTION A-TUNED' cell.")
    print("=" * 70)


if __name__ == "__main__":
    main()
