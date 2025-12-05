#!/usr/bin/env python3
"""Linear probe evaluation for SimCLR and MoCo representations.

This script evaluates learned contrastive representations using a simple linear
probe (linear regression) on the CBH prediction task. Uses leave-one-flight-out
(LOO) cross-validation to measure cross-flight generalization.

Research Question:
    Can unsupervised contrastive learning improve cross-flight generalization?
    Baseline supervised CNN cross-flight R² = -0.98 (catastrophic failure)

Evaluation Protocol:
    1. Load pretrained encoder (SimCLR or MoCo, freeze weights)
    2. Extract features for all labeled samples
    3. Train linear probe (Ridge regression) with LOO CV
    4. Report within-flight and cross-flight metrics

Usage:
    # SimCLR evaluation
    python experiments/paper2/linear_probe.py --checkpoint outputs/paper2_simclr_labeled/run_*/best_model.pt
    
    # MoCo evaluation
    python experiments/paper2/linear_probe.py --checkpoint outputs/paper2_moco_labeled/run_*/best_model.pt --model moco
    
    # Random baseline
    python experiments/paper2/linear_probe.py --random-baseline

Author: Paper 2 Implementation
Date: 2025
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Subset

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.paper2.simclr_encoder import SimCLRModel
from experiments.paper2.contrastive_dataset import ContrastiveLabeledDataset

# Lazy import MoCo to avoid import errors if not needed
MoCoModel = None


def _get_moco_model():
    """Lazy load MoCo model class."""
    global MoCoModel
    if MoCoModel is None:
        from experiments.paper2.moco_encoder import MoCoModel as _MoCoModel
        MoCoModel = _MoCoModel
    return MoCoModel


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class LinearProbeConfig:
    """Configuration for linear probe evaluation."""
    
    # Data paths
    ssl_data_path: str = "data_ssl/images/train.h5"
    features_path: str = "data/Integrated_Features_max.hdf5"
    output_dir: str = "outputs/paper2_simclr/linear_probe"
    
    # Model
    checkpoint_path: Optional[str] = None
    feature_dim: int = 256
    model_type: str = "simclr"  # "simclr" or "moco"
    
    # Evaluation
    ridge_alpha: float = 1.0  # Ridge regularization
    batch_size: int = 64
    
    # System
    device: str = "cuda"
    seed: int = 42


# =============================================================================
# Feature Extraction
# =============================================================================


def extract_features(
    encoder: nn.Module,
    dataset: ContrastiveLabeledDataset,
    device: torch.device,
    batch_size: int = 64,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract features from all samples using pretrained encoder.
    
    Args:
        encoder: Pretrained SimCLR encoder.
        dataset: Labeled dataset.
        device: Device to use.
        batch_size: Batch size for extraction.
    
    Returns:
        Tuple of (features, labels, flight_ids).
    """
    encoder.eval()
    
    all_features = []
    all_labels = []
    all_flight_ids = []
    
    # Create loader (no shuffling)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=2,
        pin_memory=True, persistent_workers=True
    )
    
    with torch.no_grad():
        for images, labels, flight_ids in loader:
            images = images.to(device)
            
            # Extract features (not projections)
            features = encoder.encode(images)
            
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())
            all_flight_ids.append(np.array(flight_ids))
    
    features = np.vstack(all_features)
    labels = np.concatenate(all_labels)
    flight_ids = np.concatenate(all_flight_ids)
    
    return features, labels, flight_ids


def extract_features_random(
    dataset: ContrastiveLabeledDataset,
    feature_dim: int = 256,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract random features (baseline comparison).
    
    Uses random projection as a baseline to compare against learned features.
    
    Args:
        dataset: Labeled dataset.
        feature_dim: Output feature dimension.
    
    Returns:
        Tuple of (features, labels, flight_ids).
    """
    all_features = []
    all_labels = []
    all_flight_ids = []
    
    # Random projection matrix (fixed seed)
    np.random.seed(42)
    projection = np.random.randn(20 * 22, feature_dim) / np.sqrt(20 * 22)
    
    for i in range(len(dataset)):
        image, label, flight_id = dataset[i]
        
        # Flatten image and project
        flat_img = image.numpy().flatten()
        features = flat_img @ projection
        
        all_features.append(features)
        all_labels.append(label.item())
        all_flight_ids.append(flight_id)
    
    features = np.vstack(all_features)
    labels = np.array(all_labels)
    flight_ids = np.array(all_flight_ids)
    
    return features, labels, flight_ids


# =============================================================================
# Leave-One-Flight-Out Cross-Validation
# =============================================================================


def loo_cross_validation(
    features: np.ndarray,
    labels: np.ndarray,
    flight_ids: np.ndarray,
    ridge_alpha: float = 1.0,
) -> Dict[str, Any]:
    """Perform leave-one-flight-out cross-validation.
    
    For each fold:
        - Train on N-1 flights
        - Test on held-out flight
    
    Args:
        features: Feature matrix of shape (n_samples, feature_dim).
        labels: CBH labels of shape (n_samples,).
        flight_ids: Flight IDs of shape (n_samples,).
        ridge_alpha: Ridge regularization strength.
    
    Returns:
        Dictionary with per-flight and aggregate metrics.
    """
    unique_flights = np.unique(flight_ids)
    n_flights = len(unique_flights)
    
    results = {
        "per_flight": {},
        "all_predictions": [],
        "all_labels": [],
        "all_flight_ids": [],
    }
    
    for test_flight in unique_flights:
        # Split data
        train_mask = flight_ids != test_flight
        test_mask = flight_ids == test_flight
        
        X_train = features[train_mask]
        y_train = labels[train_mask]
        X_test = features[test_mask]
        y_test = labels[test_mask]
        
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Ridge regression
        model = Ridge(alpha=ridge_alpha)
        model.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred = model.predict(X_test_scaled)
        
        # Compute metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        # Store per-flight results
        results["per_flight"][int(test_flight)] = {
            "n_samples": len(y_test),
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2),
            "y_true_mean": float(y_test.mean()),
            "y_true_std": float(y_test.std()),
            "y_pred_mean": float(y_pred.mean()),
            "y_pred_std": float(y_pred.std()),
        }
        
        # Accumulate predictions
        results["all_predictions"].extend(y_pred.tolist())
        results["all_labels"].extend(y_test.tolist())
        results["all_flight_ids"].extend([int(test_flight)] * len(y_test))
    
    # Compute aggregate metrics
    all_pred = np.array(results["all_predictions"])
    all_true = np.array(results["all_labels"])
    
    results["aggregate"] = {
        "cross_flight_mse": float(mean_squared_error(all_true, all_pred)),
        "cross_flight_rmse": float(np.sqrt(mean_squared_error(all_true, all_pred))),
        "cross_flight_mae": float(mean_absolute_error(all_true, all_pred)),
        "cross_flight_r2": float(r2_score(all_true, all_pred)),
        "n_flights": n_flights,
        "n_samples": len(all_true),
    }
    
    return results


def within_flight_evaluation(
    features: np.ndarray,
    labels: np.ndarray,
    flight_ids: np.ndarray,
    ridge_alpha: float = 1.0,
    test_fraction: float = 0.2,
) -> Dict[str, Any]:
    """Evaluate within-flight generalization (random train/test split).
    
    For comparison with cross-flight evaluation.
    
    Args:
        features: Feature matrix.
        labels: CBH labels.
        flight_ids: Flight IDs.
        ridge_alpha: Ridge regularization.
        test_fraction: Fraction of each flight to use for testing.
    
    Returns:
        Dictionary with within-flight metrics.
    """
    unique_flights = np.unique(flight_ids)
    
    all_pred = []
    all_true = []
    
    np.random.seed(42)
    
    for flight in unique_flights:
        mask = flight_ids == flight
        X_flight = features[mask]
        y_flight = labels[mask]
        
        n = len(y_flight)
        if n < 10:
            continue
        
        # Random split
        indices = np.random.permutation(n)
        n_test = max(2, int(n * test_fraction))
        
        test_idx = indices[:n_test]
        train_idx = indices[n_test:]
        
        X_train = X_flight[train_idx]
        y_train = y_flight[train_idx]
        X_test = X_flight[test_idx]
        y_test = y_flight[test_idx]
        
        # Standardize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train and predict
        model = Ridge(alpha=ridge_alpha)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        all_pred.extend(y_pred.tolist())
        all_true.extend(y_test.tolist())
    
    all_pred = np.array(all_pred)
    all_true = np.array(all_true)
    
    return {
        "within_flight_mse": float(mean_squared_error(all_true, all_pred)),
        "within_flight_rmse": float(np.sqrt(mean_squared_error(all_true, all_pred))),
        "within_flight_mae": float(mean_absolute_error(all_true, all_pred)),
        "within_flight_r2": float(r2_score(all_true, all_pred)),
        "n_samples": len(all_true),
    }


# =============================================================================
# Main Evaluation
# =============================================================================


def load_encoder(
    checkpoint_path: str,
    device: torch.device,
    model_type: str = "simclr",
) -> nn.Module:
    """Load pretrained encoder from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file.
        device: Device to load model to.
        model_type: Type of model ("simclr" or "moco").
    
    Returns:
        Loaded encoder model.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get config from checkpoint
    config = checkpoint.get("config", {})
    feature_dim = config.get("feature_dim", 256)
    projection_dim = config.get("projection_dim", 128)
    hidden_dim = config.get("hidden_dim", 256)
    
    if model_type == "moco":
        # Load MoCo model
        MoCoModelClass = _get_moco_model()
        queue_size = config.get("queue_size", 2048)
        momentum = config.get("momentum", 0.999)
        temperature = config.get("temperature", 0.07)
        
        model = MoCoModelClass(
            in_channels=1,
            feature_dim=feature_dim,
            projection_dim=projection_dim,
            hidden_dim=hidden_dim,
            queue_size=queue_size,
            momentum=momentum,
            temperature=temperature,
        )
    else:
        # Load SimCLR model
        temperature = config.get("temperature", 0.5)
        
        model = SimCLRModel(
            in_channels=1,
            feature_dim=feature_dim,
            projection_dim=projection_dim,
            hidden_dim=hidden_dim,
            temperature=temperature,
        )
    
    # Load weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    
    return model


def run_evaluation(
    config: LinearProbeConfig,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """Run full linear probe evaluation.
    
    Args:
        config: Evaluation configuration.
        logger: Logger instance.
    
    Returns:
        Dictionary with all evaluation results.
    """
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Set seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    
    # Load dataset
    ssl_path = PROJECT_ROOT / config.ssl_data_path
    features_path = PROJECT_ROOT / config.features_path
    
    logger.info(f"Loading labeled dataset...")
    logger.info(f"  SSL images: {ssl_path}")
    logger.info(f"  Features: {features_path}")
    
    dataset = ContrastiveLabeledDataset(
        ssl_path=str(ssl_path),
        features_path=str(features_path),
        augment=False,
    )
    
    logger.info(f"  Matched samples: {len(dataset)}")
    
    # Extract features
    if config.checkpoint_path:
        logger.info(f"Loading pretrained encoder: {config.checkpoint_path}")
        logger.info(f"  Model type: {config.model_type}")
        encoder = load_encoder(config.checkpoint_path, device, config.model_type)
        
        # Get training info from checkpoint
        checkpoint = torch.load(config.checkpoint_path, map_location=device)
        training_epoch = checkpoint.get("epoch", "unknown")
        best_loss = checkpoint.get("best_loss", "unknown")
        logger.info(f"  Training epoch: {training_epoch}")
        logger.info(f"  Best loss: {best_loss}")
        
        logger.info("Extracting features with pretrained encoder...")
        features, labels, flight_ids = extract_features(
            encoder, dataset, device, config.batch_size
        )
        method = config.model_type
    else:
        logger.info("Using random projection baseline...")
        features, labels, flight_ids = extract_features_random(
            dataset, config.feature_dim
        )
        method = "random"
    
    logger.info(f"Feature shape: {features.shape}")
    logger.info(f"Labels shape: {labels.shape}")
    logger.info(f"Unique flights: {np.unique(flight_ids)}")
    
    # Run LOO cross-validation
    logger.info("\n" + "=" * 60)
    logger.info("LEAVE-ONE-FLIGHT-OUT CROSS-VALIDATION")
    logger.info("=" * 60)
    
    loo_results = loo_cross_validation(
        features, labels, flight_ids, config.ridge_alpha
    )
    
    # Print per-flight results
    logger.info("\nPer-flight results (cross-flight test):")
    for flight_id, metrics in sorted(loo_results["per_flight"].items()):
        logger.info(
            f"  Flight {flight_id}: "
            f"R²={metrics['r2']:.3f}, "
            f"RMSE={metrics['rmse']:.3f} km, "
            f"MAE={metrics['mae']:.3f} km, "
            f"n={metrics['n_samples']}"
        )
    
    # Print aggregate results
    agg = loo_results["aggregate"]
    logger.info("\nAggregate cross-flight results:")
    logger.info(f"  Cross-flight R²:   {agg['cross_flight_r2']:.4f}")
    logger.info(f"  Cross-flight RMSE: {agg['cross_flight_rmse']:.4f} km")
    logger.info(f"  Cross-flight MAE:  {agg['cross_flight_mae']:.4f} km")
    
    # Run within-flight evaluation for comparison
    logger.info("\n" + "=" * 60)
    logger.info("WITHIN-FLIGHT EVALUATION (for comparison)")
    logger.info("=" * 60)
    
    within_results = within_flight_evaluation(
        features, labels, flight_ids, config.ridge_alpha
    )
    
    logger.info(f"  Within-flight R²:   {within_results['within_flight_r2']:.4f}")
    logger.info(f"  Within-flight RMSE: {within_results['within_flight_rmse']:.4f} km")
    logger.info(f"  Within-flight MAE:  {within_results['within_flight_mae']:.4f} km")
    
    # Compute generalization gap
    gen_gap = within_results["within_flight_r2"] - agg["cross_flight_r2"]
    logger.info(f"\n  Generalization gap (within - cross): {gen_gap:.4f}")
    
    # Summary comparison with baseline
    logger.info("\n" + "=" * 60)
    logger.info("COMPARISON WITH BASELINE")
    logger.info("=" * 60)
    logger.info("Baseline supervised CNN cross-flight R²: -0.98")
    logger.info(f"SimCLR linear probe cross-flight R²:   {agg['cross_flight_r2']:.4f}")
    improvement = agg["cross_flight_r2"] - (-0.98)
    logger.info(f"Improvement: {improvement:+.4f}")
    
    # Compile results
    results = {
        "method": method,
        "checkpoint": config.checkpoint_path,
        "config": {
            "ridge_alpha": config.ridge_alpha,
            "feature_dim": config.feature_dim,
            "seed": config.seed,
        },
        "data": {
            "n_samples": len(labels),
            "n_flights": len(np.unique(flight_ids)),
            "flight_ids": sorted(np.unique(flight_ids).tolist()),
        },
        "cross_flight": loo_results,
        "within_flight": within_results,
        "baseline_comparison": {
            "baseline_supervised_cnn_r2": -0.98,
            "simclr_linear_probe_r2": agg["cross_flight_r2"],
            "improvement": improvement,
            "generalization_gap": gen_gap,
        },
        "timestamp": datetime.now().isoformat(),
    }
    
    return results


def setup_logging(output_dir: Path) -> logging.Logger:
    """Setup logging to file and console."""
    logger = logging.getLogger("linear_probe")
    logger.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter("%(asctime)s - %(message)s", datefmt="%H:%M:%S")
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler
    log_file = output_dir / "linear_probe.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    
    return logger


def main(args: argparse.Namespace) -> None:
    """Main evaluation function."""
    # Create config
    config = LinearProbeConfig()
    
    if args.checkpoint:
        config.checkpoint_path = args.checkpoint
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.ridge_alpha:
        config.ridge_alpha = args.ridge_alpha
    if args.device:
        config.device = args.device
    if args.features_path:
        config.features_path = args.features_path
    if args.model:
        config.model_type = args.model
    
    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.random_baseline:
        output_dir = Path(config.output_dir) / f"random_baseline_{timestamp}"
    else:
        output_dir = Path(config.output_dir) / f"eval_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(output_dir)
    logger.info(f"Output directory: {output_dir}")
    
    if args.random_baseline:
        logger.info("Running RANDOM BASELINE evaluation")
        config.checkpoint_path = None
    else:
        if not config.checkpoint_path:
            logger.error("No checkpoint provided. Use --checkpoint or --random-baseline")
            return
    
    # Run evaluation
    try:
        results = run_evaluation(config, logger)
        
        # Save results
        results_path = output_dir / "results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"\nResults saved to: {results_path}")
        
        logger.info("\n" + "=" * 60)
        logger.info("EVALUATION COMPLETE")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Linear probe evaluation for SimCLR representations"
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to pretrained SimCLR checkpoint",
    )
    parser.add_argument(
        "--random-baseline",
        action="store_true",
        help="Run random projection baseline instead of SimCLR",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results",
    )
    parser.add_argument(
        "--ridge-alpha",
        type=float,
        default=None,
        help="Ridge regularization strength",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Device to use",
    )
    parser.add_argument(
        "--features-path",
        type=str,
        default=None,
        help="Path to integrated features HDF5",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="simclr",
        choices=["simclr", "moco"],
        help="Model type (simclr or moco)",
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
