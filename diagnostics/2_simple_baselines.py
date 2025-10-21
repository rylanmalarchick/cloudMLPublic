#!/usr/bin/env python3
"""
Diagnostic 2: Simple Baselines
===============================

Goal: Can simple models (Linear Regression, Random Forest) beat the mean baseline?
This tells us if the signal is strong enough for ANY model to learn.

Expected runtime: ~1 hour

Success criteria:
- If any model gets R² > 0 → Signal is learnable, deep learning should work
- If all models get R² < 0 → Data doesn't contain learnable signal
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from pathlib import Path
import json
from tqdm import tqdm
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    explained_variance_score,
)
import warnings

warnings.filterwarnings("ignore")

# Import project modules
from src.config_loader import load_config
from src.data_loader import UnifiedCloudDataset


def extract_features(images, sza, saa):
    """
    Extract comprehensive hand-crafted features.
    """
    # Flatten temporal dimension - use middle frame
    if images.ndim == 4:
        images_2d = images[:, images.shape[1] // 2, :, :]
    else:
        images_2d = images

    n_samples = len(images_2d)
    features = []

    print("Extracting features for each sample...")
    for i in tqdm(range(n_samples), desc="Feature extraction"):
        img = images_2d[i]

        feat = []

        # Basic intensity statistics
        feat.append(np.mean(img))
        feat.append(np.std(img))
        feat.append(np.min(img))
        feat.append(np.max(img))
        feat.append(np.median(img))
        feat.append(np.max(img) - np.min(img))  # range

        # Percentiles
        feat.append(np.percentile(img, 10))
        feat.append(np.percentile(img, 25))
        feat.append(np.percentile(img, 50))
        feat.append(np.percentile(img, 75))
        feat.append(np.percentile(img, 90))

        # Spatial gradients
        grad_h = np.abs(np.diff(img, axis=1))
        grad_v = np.abs(np.diff(img, axis=0))
        feat.append(np.mean(grad_h))
        feat.append(np.std(grad_h))
        feat.append(np.max(grad_h))
        feat.append(np.mean(grad_v))
        feat.append(np.std(grad_v))
        feat.append(np.max(grad_v))

        # Center vs edge statistics
        h, w = img.shape
        center_h, center_w = h // 4, w // 4
        center = img[center_h : 3 * center_h, center_w : 3 * center_w]
        feat.append(np.mean(center))
        feat.append(np.std(center))

        # Metadata
        feat.append(sza[i])
        feat.append(saa[i])
        feat.append(np.cos(np.radians(sza[i])))
        feat.append(np.sin(np.radians(sza[i])))
        feat.append(np.cos(np.radians(saa[i])))
        feat.append(np.sin(np.radians(saa[i])))

        # Interaction features
        feat.append(np.mean(img) * np.cos(np.radians(sza[i])))
        feat.append(np.std(img) * np.cos(np.radians(sza[i])))

        features.append(feat)

    return np.array(features)


def load_data(config_path):
    """Load train and test datasets."""
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    config = load_config(config_path)

    # Load train set
    print("\nLoading TRAIN set...")
    train_dataset = UnifiedCloudDataset(
        config=config, mode="train", return_metadata=True
    )

    # Load test set
    print("Loading TEST set...")
    test_dataset = UnifiedCloudDataset(config=config, mode="test", return_metadata=True)

    print(f"Train size: {len(train_dataset)} samples")
    print(f"Test size: {len(test_dataset)} samples")

    # Extract train data
    train_images, train_sza, train_saa, train_targets = [], [], [], []
    print("\nExtracting train samples...")
    for i in tqdm(range(len(train_dataset)), desc="Train"):
        img_stack, sza, saa, y, _, _ = train_dataset[i]
        train_images.append(img_stack.numpy())
        train_sza.append(sza.item())
        train_saa.append(saa.item())
        train_targets.append(y.item())

    train_images = np.array(train_images)
    train_sza = np.array(train_sza)
    train_saa = np.array(train_saa)
    train_targets = np.array(train_targets)

    # Extract test data
    test_images, test_sza, test_saa, test_targets = [], [], [], []
    print("Extracting test samples...")
    for i in tqdm(range(len(test_dataset)), desc="Test"):
        img_stack, sza, saa, y, _, _ = test_dataset[i]
        test_images.append(img_stack.numpy())
        test_sza.append(sza.item())
        test_saa.append(saa.item())
        test_targets.append(y.item())

    test_images = np.array(test_images)
    test_sza = np.array(test_sza)
    test_saa = np.array(test_saa)
    test_targets = np.array(test_targets)

    return (
        train_images,
        train_sza,
        train_saa,
        train_targets,
        test_images,
        test_sza,
        test_saa,
        test_targets,
    )


def train_and_evaluate(model, model_name, X_train, y_train, X_test, y_test):
    """Train a model and evaluate it."""
    print(f"\n{'-' * 70}")
    print(f"Training {model_name}...")
    print(f"{'-' * 70}")

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Compute metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)

    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    # Variance ratio
    train_var_ratio = np.std(y_train_pred) / np.std(y_train)
    test_var_ratio = np.std(y_test_pred) / np.std(y_test)

    # Print results
    print(f"\nTrain Metrics:")
    print(f"  R²:   {train_r2:.4f}")
    print(f"  MAE:  {train_mae:.4f}")
    print(f"  RMSE: {train_rmse:.4f}")
    print(f"  Var Ratio: {train_var_ratio:.1%}")

    print(f"\nTest Metrics:")
    print(f"  R²:   {test_r2:.4f}")
    print(f"  MAE:  {test_mae:.4f}")
    print(f"  RMSE: {test_rmse:.4f}")
    print(f"  Var Ratio: {test_var_ratio:.1%}")

    # Feature importance (if available)
    feature_importance = None
    if hasattr(model, "feature_importances_"):
        feature_importance = model.feature_importances_.tolist()
        top_5_idx = np.argsort(model.feature_importances_)[-5:][::-1]
        print(f"\nTop 5 Most Important Features:")
        for idx in top_5_idx:
            print(f"  Feature {idx}: {model.feature_importances_[idx]:.4f}")
    elif hasattr(model, "coef_"):
        feature_importance = np.abs(model.coef_).tolist()
        top_5_idx = np.argsort(np.abs(model.coef_))[-5:][::-1]
        print(f"\nTop 5 Features by Coefficient Magnitude:")
        for idx in top_5_idx:
            print(f"  Feature {idx}: {np.abs(model.coef_[idx]):.4f}")

    results = {
        "model": model_name,
        "train_r2": float(train_r2),
        "test_r2": float(test_r2),
        "train_mae": float(train_mae),
        "test_mae": float(test_mae),
        "train_rmse": float(train_rmse),
        "test_rmse": float(test_rmse),
        "train_var_ratio": float(train_var_ratio),
        "test_var_ratio": float(test_var_ratio),
        "feature_importance": feature_importance,
        "overfitting": train_r2 - test_r2 > 0.1,
    }

    return results


def main():
    """Main diagnostic routine."""
    print("=" * 70)
    print("DIAGNOSTIC 2: SIMPLE BASELINES")
    print("=" * 70)
    print("\nGoal: Can simple models beat the mean baseline (R² > 0)?")
    print("This validates that the task has learnable signal.\n")

    # Configuration
    config_path = "configs/colab_optimized_full_tuned.yaml"
    output_dir = Path("diagnostics/results")
    output_dir.mkdir(exist_ok=True, parents=True)

    # Load data
    (
        train_images,
        train_sza,
        train_saa,
        train_targets,
        test_images,
        test_sza,
        test_saa,
        test_targets,
    ) = load_data(config_path)

    # Extract features
    print("\n" + "=" * 70)
    print("EXTRACTING FEATURES")
    print("=" * 70)
    X_train = extract_features(train_images, train_sza, train_saa)
    X_test = extract_features(test_images, test_sza, test_saa)

    print(f"\nFeature matrix shapes:")
    print(f"  Train: {X_train.shape}")
    print(f"  Test:  {X_test.shape}")

    # Standardize features
    print("\nStandardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define models to test
    models = [
        ("Ridge Regression (α=1.0)", Ridge(alpha=1.0)),
        ("Ridge Regression (α=10.0)", Ridge(alpha=10.0)),
        ("Lasso Regression (α=0.01)", Lasso(alpha=0.01, max_iter=5000)),
        ("ElasticNet (α=0.01)", ElasticNet(alpha=0.01, max_iter=5000)),
        (
            "Random Forest (n=50, d=10)",
            RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42),
        ),
        (
            "Random Forest (n=100, d=20)",
            RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42),
        ),
        (
            "Gradient Boosting (n=50)",
            GradientBoostingRegressor(n_estimators=50, max_depth=5, random_state=42),
        ),
    ]

    # Train and evaluate all models
    print("\n" + "=" * 70)
    print("TRAINING AND EVALUATING MODELS")
    print("=" * 70)

    all_results = []
    for model_name, model in models:
        try:
            results = train_and_evaluate(
                model,
                model_name,
                X_train_scaled,
                train_targets,
                X_test_scaled,
                test_targets,
            )
            all_results.append(results)
        except Exception as e:
            print(f"\nERROR training {model_name}: {e}")
            continue

    # Save results
    df_results = pd.DataFrame(all_results)
    csv_path = output_dir / "baseline_results.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"\n\nResults saved to: {csv_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print("\nAll Models (sorted by test R²):")
    df_sorted = df_results.sort_values("test_r2", ascending=False)
    print(
        df_sorted[
            ["model", "test_r2", "test_mae", "test_rmse", "test_var_ratio"]
        ].to_string(index=False)
    )

    # Best model
    best_model = df_sorted.iloc[0]
    print(f"\n{'=' * 70}")
    print("BEST MODEL")
    print(f"{'=' * 70}")
    print(f"\nModel: {best_model['model']}")
    print(f"Test R²: {best_model['test_r2']:.4f}")
    print(f"Test MAE: {best_model['test_mae']:.4f}")
    print(f"Test RMSE: {best_model['test_rmse']:.4f}")
    print(f"Variance Ratio: {best_model['test_var_ratio']:.1%}")

    # Decision
    print("\n" + "=" * 70)
    print("DECISION")
    print("=" * 70)

    best_r2 = best_model["test_r2"]

    if best_r2 < -0.05:
        decision = "🔴 STOP"
        explanation = f"Best model R² = {best_r2:.4f} (much worse than mean baseline)"
        recommendation = (
            "Data does NOT contain learnable signal for simple models.\n"
            "  Deep learning is VERY unlikely to work better.\n\n"
            "  Consider:\n"
            "  1. Check data quality (corrupted samples, wrong labels?)\n"
            "  2. Different features (multi-wavelength, polarization?)\n"
            "  3. Different target (cloud type/category instead of optical depth?)\n"
            "  4. Consult domain experts about physical feasibility"
        )
    elif best_r2 < 0:
        decision = "🟡 VERY WEAK SIGNAL"
        explanation = f"Best model R² = {best_r2:.4f} (slightly worse than mean)"
        recommendation = (
            "Simple models can't beat mean baseline.\n"
            "  Deep learning MIGHT extract nonlinear patterns, but unlikely.\n\n"
            "  Options:\n"
            "  1. Proceed to Diagnostic 3 (architecture ablation)\n"
            "  2. Try Run 5 but set expectations LOW (R² likely stays negative)\n"
            "  3. Consider reformulating task (classification, ranking, etc.)"
        )
    elif best_r2 < 0.1:
        decision = "🟡 WEAK BUT LEARNABLE"
        explanation = f"Best model R² = {best_r2:.4f} (barely positive)"
        recommendation = (
            "Signal exists but is VERY weak.\n"
            "  Simple models can barely beat mean baseline.\n\n"
            "  Deep learning should work slightly better, but expect:\n"
            "  - Final R² in range 0.05-0.15 (not great)\n"
            "  - Proceed to Diagnostic 3 to test architectures\n"
            "  - Consider if this level of performance is useful for your application"
        )
    elif best_r2 < 0.3:
        decision = "✅ LEARNABLE (MODERATE)"
        explanation = f"Best model R² = {best_r2:.4f} (moderate performance)"
        recommendation = (
            "Signal definitely exists! Simple models can learn patterns.\n\n"
            "  Deep learning should achieve R² = 0.2-0.4 with proper tuning.\n"
            "  Proceed to:\n"
            "  - Diagnostic 3 (architecture ablation)\n"
            "  - Run 5 with increased variance lambda\n"
            "  - Expected: Useful but not perfect predictions"
        )
    else:
        decision = "✅ STRONGLY LEARNABLE"
        explanation = f"Best model R² = {best_r2:.4f} (good performance!)"
        recommendation = (
            "Strong signal! Simple models work well.\n\n"
            "  Deep learning should achieve R² > 0.4 (possibly >0.5).\n"
            "  Your previous runs' failure is likely due to:\n"
            "  - Training issues (variance collapse, etc.)\n"
            "  - NOT a fundamental data problem\n\n"
            "  Proceed with confidence to Run 5 or architecture tuning!"
        )

    print(f"\n{decision}")
    print(f"\nExplanation: {explanation}")
    print(f"\nRecommendation:")
    for line in recommendation.split("\n"):
        print(f"  {line}")

    # Comparison with neural network results
    print("\n" + "=" * 70)
    print("COMPARISON WITH YOUR NEURAL NETWORK RUNS")
    print("=" * 70)
    print(f"\nBest simple model:  R² = {best_r2:.4f}")
    print(f"Run 1 (neural net): R² = -0.0457")
    print(f"Run 2 (neural net): R² = -0.0226")
    print(f"Run 3 (neural net): R² = -0.2034")
    print(f"Run 4 (neural net): R² = -0.0655")

    if best_r2 > 0:
        print(f"\n⚠️  CRITICAL: Simple models BEAT all your neural network runs!")
        print(
            f"   This means your neural network is UNDERPERFORMING due to training issues,"
        )
        print(f"   NOT because the task is impossible.")
    else:
        print(f"\n⚠️  Both simple models and neural networks fail (R² < 0).")
        print(f"   This suggests fundamental data/feature limitations.")

    # Save summary
    summary = {
        "best_model": best_model["model"],
        "best_r2": float(best_r2),
        "best_mae": float(best_model["test_mae"]),
        "best_rmse": float(best_model["test_rmse"]),
        "best_var_ratio": float(best_model["test_var_ratio"]),
        "decision": decision,
        "n_models_tested": len(all_results),
        "n_positive_r2": int((df_results["test_r2"] > 0).sum()),
        "mean_baseline_beaten": bool(best_r2 > 0),
    }

    json_path = output_dir / "baseline_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n\nSummary saved to: {json_path}")
    print("=" * 70)
    print("DIAGNOSTIC 2 COMPLETE")
    print("=" * 70)

    return summary


if __name__ == "__main__":
    summary = main()
