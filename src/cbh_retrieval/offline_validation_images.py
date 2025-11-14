#!/usr/bin/env python3
"""
Sprint 6 - Phase 1, Task 1.1: Offline Validation (Image-Based)

This script performs stratified 5-fold cross-validation on the actual image data
(20×22 pixel arrays) using a Convolutional Neural Network (CNN) model.

This complements the tabular validation by using the raw image data.

Author: Sprint 6 Agent
Date: 2025
"""

import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset

warnings.filterwarnings("ignore")

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "."))

from modules.image_dataset import ImageCBHDataset

print("=" * 80)
print("Sprint 6 - Phase 1, Task 1.1: Offline Validation (Image-Based)")
print("=" * 80)
print(f"Project Root: {PROJECT_ROOT}")

# Paths
SSL_IMAGES = PROJECT_ROOT / "data_ssl/images/train.h5"
INTEGRATED_FEATURES = (
    PROJECT_ROOT / "outputs/preprocessed_data/Integrated_Features.hdf5"
)
OUTPUT_DIR = PROJECT_ROOT / "."
FIGURES_DIR = OUTPUT_DIR / "figures/validation_images"
REPORTS_DIR = OUTPUT_DIR / "reports"

# Create directories
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

print(f"SSL Images: {SSL_IMAGES}")
print(f"Integrated Features: {INTEGRATED_FEATURES}")
print(f"Output Directory: {OUTPUT_DIR}")

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" Device: {device}")


class SimpleCNN(nn.Module):
    """Simple CNN for 20×22 single-channel images."""

    def __init__(self, dropout=0.3):
        super(SimpleCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout)

        # Calculate flattened size: 20×22 -> pool -> pool = 5×5
        # After 2 pooling: 20/4 = 5, 22/4 = 5.5 (floor) = 5
        self.flatten_size = 64 * 5 * 5

        # Fully connected layers
        self.fc1 = nn.Linear(self.flatten_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

        self.relu = nn.ReLU()

    def forward(self, x):
        # Conv block 1
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.dropout(x)

        # Conv block 2
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.dropout(x)

        # Conv block 3
        x = self.relu(self.bn3(self.conv3(x)))

        # Flatten
        x = x.view(-1, self.flatten_size)

        # FC layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        return x.squeeze()


class ImageValidationAnalyzer:
    """Performs stratified 5-fold CV validation on image data."""

    def __init__(self, n_folds=5, random_seed=42):
        self.n_folds = n_folds
        self.random_seed = random_seed
        self.device = device

        self.results = {
            "folds": [],
            "mean_metrics": {},
            "std_metrics": {},
            "metadata": {
                "model_type": "SimpleCNN",
                "n_folds": n_folds,
                "random_seed": random_seed,
                "validation_strategy": "stratified_5fold",
                "image_shape": "(20, 22)",
                "timestamp": datetime.now().isoformat(),
            },
        }

    def load_data(self):
        """Load image dataset."""
        print("\n" + "=" * 80)
        print("Loading Image Dataset")
        print("=" * 80)

        dataset = ImageCBHDataset(
            ssl_images_path=str(SSL_IMAGES),
            integrated_features_path=str(INTEGRATED_FEATURES),
            image_shape=(20, 22),
            normalize=True,
            augment=False,
            return_indices=True,
        )

        # Get labels for stratification
        labels = []
        for i in range(len(dataset)):
            _, cbh, _, _, _ = dataset[i]
            labels.append(cbh.item())

        labels = np.array(labels)

        print(f"\n Dataset loaded: {len(dataset)} samples")
        print(f" CBH range: [{labels.min():.3f}, {labels.max():.3f}] km")

        return dataset, labels

    def create_stratified_bins(self, y, n_bins=10):
        """Create stratified bins for CBH values."""
        bins = np.percentile(y, np.linspace(0, 100, n_bins + 1))
        bins = np.unique(bins)
        bin_indices = np.digitize(y, bins) - 1
        bin_indices = np.clip(bin_indices, 0, len(bins) - 2)
        return bin_indices

    def train_epoch(self, model, train_loader, criterion, optimizer):
        """Train for one epoch."""
        model.train()
        total_loss = 0
        n_batches = 0

        for images, targets, _, _, _ in train_loader:
            images = images.to(self.device)
            targets = targets.to(self.device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / n_batches

    def evaluate(self, model, data_loader):
        """Evaluate model on data loader."""
        model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for images, targets, _, _, _ in data_loader:
                images = images.to(self.device)
                outputs = model(images)

                all_preds.extend(outputs.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        return np.array(all_preds), np.array(all_targets)

    def train_and_evaluate_fold(self, dataset, train_idx, val_idx, fold_idx):
        """Train and evaluate on a single fold."""
        print(f"\n{'=' * 80}")
        print(f"Fold {fold_idx + 1}/{self.n_folds}")
        print(f"{'=' * 80}")
        print(f"Train samples: {len(train_idx)}, Val samples: {len(val_idx)}")

        # Create data loaders
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_loader = DataLoader(
            train_subset, batch_size=32, shuffle=True, num_workers=0
        )
        val_loader = DataLoader(val_subset, batch_size=32, shuffle=False, num_workers=0)

        # Initialize model
        model = SimpleCNN(dropout=0.3).to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )

        # Training loop
        print("\nTraining CNN model...")
        n_epochs = 50
        best_val_loss = float("inf")
        patience_counter = 0
        patience = 10

        for epoch in range(n_epochs):
            train_loss = self.train_epoch(model, train_loader, criterion, optimizer)

            # Validation
            val_preds, val_targets = self.evaluate(model, val_loader)
            val_loss = mean_squared_error(val_targets, val_preds)

            scheduler.step(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch + 1}")
                break

            if (epoch + 1) % 10 == 0:
                print(
                    f"  Epoch {epoch + 1}/{n_epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}"
                )

        # Load best model
        model.load_state_dict(best_model_state)

        # Final evaluation
        train_preds, train_targets = self.evaluate(model, train_loader)
        val_preds, val_targets = self.evaluate(model, val_loader)

        # Compute metrics
        train_metrics = self._compute_metrics(train_targets, train_preds, "Train")
        val_metrics = self._compute_metrics(val_targets, val_preds, "Validation")

        return {
            "fold": fold_idx + 1,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "train_size": len(train_idx),
            "val_size": len(val_idx),
            "y_true": val_targets.tolist(),
            "y_pred": val_preds.tolist(),
        }

    def _compute_metrics(self, y_true, y_pred, prefix=""):
        """Compute regression metrics."""
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)

        # Convert to meters
        mae_m = mae * 1000
        rmse_m = rmse * 1000

        print(f"{prefix} Metrics:")
        print(f"  R² = {r2:.4f}")
        print(f"  MAE = {mae_m:.2f} m ({mae:.4f} km)")
        print(f"  RMSE = {rmse_m:.2f} m ({rmse:.4f} km)")

        return {
            "r2": float(r2),
            "mae_km": float(mae),
            "rmse_km": float(rmse),
            "mae_m": float(mae_m),
            "rmse_m": float(rmse_m),
        }

    def run_validation(self, dataset, labels):
        """Run stratified 5-fold cross-validation."""
        print("\n" + "=" * 80)
        print("Running Stratified 5-Fold Cross-Validation (Image-Based)")
        print("=" * 80)

        # Create stratified bins
        stratified_bins = self.create_stratified_bins(labels, n_bins=10)

        # Initialize k-fold
        skf = StratifiedKFold(
            n_splits=self.n_folds, shuffle=True, random_state=self.random_seed
        )

        # Run folds
        fold_results = []
        all_y_true = []
        all_y_pred = []

        indices = np.arange(len(dataset))

        for fold_idx, (train_idx, val_idx) in enumerate(
            skf.split(indices, stratified_bins)
        ):
            fold_result = self.train_and_evaluate_fold(
                dataset, train_idx, val_idx, fold_idx
            )

            fold_results.append(fold_result)
            all_y_true.extend(fold_result["y_true"])
            all_y_pred.extend(fold_result["y_pred"])

        # Store results
        self.results["folds"] = fold_results
        self.results["aggregated_predictions"] = {
            "y_true": all_y_true,
            "y_pred": all_y_pred,
        }

        # Aggregate metrics
        self._aggregate_metrics(fold_results)

        return self.results

    def _aggregate_metrics(self, fold_results):
        """Aggregate metrics across folds."""
        print("\n" + "=" * 80)
        print("Aggregated Results Across All Folds")
        print("=" * 80)

        # Extract validation metrics
        val_r2 = [f["val_metrics"]["r2"] for f in fold_results]
        val_mae_m = [f["val_metrics"]["mae_m"] for f in fold_results]
        val_rmse_m = [f["val_metrics"]["rmse_m"] for f in fold_results]

        mean_r2 = np.mean(val_r2)
        std_r2 = np.std(val_r2)
        mean_mae_m = np.mean(val_mae_m)
        std_mae_m = np.std(val_mae_m)
        mean_rmse_m = np.mean(val_rmse_m)
        std_rmse_m = np.std(val_rmse_m)

        print(f"Mean R² = {mean_r2:.4f} ± {std_r2:.4f}")
        print(f"Mean MAE = {mean_mae_m:.2f} ± {std_mae_m:.2f} m")
        print(f"Mean RMSE = {mean_rmse_m:.2f} ± {std_rmse_m:.2f} m")

        self.results["mean_metrics"] = {
            "r2": float(mean_r2),
            "mae_m": float(mean_mae_m),
            "rmse_m": float(mean_rmse_m),
        }

        self.results["std_metrics"] = {
            "r2": float(std_r2),
            "mae_m": float(std_mae_m),
            "rmse_m": float(std_rmse_m),
        }

    def generate_visualizations(self, output_dir):
        """Generate validation visualizations."""
        print("\n" + "=" * 80)
        print("Generating Visualizations")
        print("=" * 80)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        sns.set_style("whitegrid")
        plt.rcParams["figure.dpi"] = 300

        # Prediction scatter
        self._plot_predictions_scatter(output_dir)

        # Residuals
        self._plot_residuals(output_dir)

        # Per-fold metrics
        self._plot_fold_metrics(output_dir)

        print(f" Visualizations saved to {output_dir}")

    def _plot_predictions_scatter(self, output_dir):
        """Plot predicted vs true values."""
        y_true = np.array(self.results["aggregated_predictions"]["y_true"])
        y_pred = np.array(self.results["aggregated_predictions"]["y_pred"])

        fig, ax = plt.subplots(figsize=(8, 8))

        ax.scatter(y_true, y_pred, alpha=0.5, s=20)

        lim = [y_true.min(), y_true.max()]
        ax.plot(lim, lim, "r--", lw=2, label="Perfect Prediction")

        ax.set_xlabel("True CBH (km)", fontsize=12)
        ax.set_ylabel("Predicted CBH (km)", fontsize=12)
        ax.set_title(
            "CNN: Predicted vs True CBH (5-Fold CV)", fontsize=14, fontweight="bold"
        )
        ax.legend()
        ax.grid(True, alpha=0.3)

        r2 = self.results["mean_metrics"]["r2"]
        mae = self.results["mean_metrics"]["mae_m"]
        textstr = f"R² = {r2:.3f}\nMAE = {mae:.1f} m"
        ax.text(
            0.05,
            0.95,
            textstr,
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        plt.tight_layout()
        plt.savefig(
            output_dir / "predictions_scatter_cnn.png", dpi=300, bbox_inches="tight"
        )
        plt.close()
        print("   predictions_scatter_cnn.png")

    def _plot_residuals(self, output_dir):
        """Plot residuals distribution."""
        y_true = np.array(self.results["aggregated_predictions"]["y_true"])
        y_pred = np.array(self.results["aggregated_predictions"]["y_pred"])
        residuals = (y_pred - y_true) * 1000

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].hist(residuals, bins=50, edgecolor="black", alpha=0.7)
        axes[0].axvline(0, color="r", linestyle="--", lw=2, label="Zero Error")
        axes[0].set_xlabel("Residual (m)", fontsize=12)
        axes[0].set_ylabel("Frequency", fontsize=12)
        axes[0].set_title(
            "Distribution of Residuals (CNN)", fontsize=14, fontweight="bold"
        )
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].scatter(y_pred, residuals, alpha=0.5, s=20)
        axes[1].axhline(0, color="r", linestyle="--", lw=2, label="Zero Error")
        axes[1].set_xlabel("Predicted CBH (km)", fontsize=12)
        axes[1].set_ylabel("Residual (m)", fontsize=12)
        axes[1].set_title(
            "Residuals vs Predicted (CNN)", fontsize=14, fontweight="bold"
        )
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "residuals_cnn.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("   residuals_cnn.png")

    def _plot_fold_metrics(self, output_dir):
        """Plot metrics across folds."""
        folds = [f["fold"] for f in self.results["folds"]]
        r2_vals = [f["val_metrics"]["r2"] for f in self.results["folds"]]
        mae_vals = [f["val_metrics"]["mae_m"] for f in self.results["folds"]]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].bar(folds, r2_vals, color="steelblue", edgecolor="black")
        axes[0].axhline(np.mean(r2_vals), color="r", linestyle="--", lw=2, label="Mean")
        axes[0].set_xlabel("Fold", fontsize=12)
        axes[0].set_ylabel("R²", fontsize=12)
        axes[0].set_title("CNN: R² Across Folds", fontsize=14, fontweight="bold")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis="y")

        axes[1].bar(folds, mae_vals, color="coral", edgecolor="black")
        axes[1].axhline(
            np.mean(mae_vals), color="r", linestyle="--", lw=2, label="Mean"
        )
        axes[1].set_xlabel("Fold", fontsize=12)
        axes[1].set_ylabel("MAE (m)", fontsize=12)
        axes[1].set_title("CNN: MAE Across Folds", fontsize=14, fontweight="bold")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        plt.savefig(output_dir / "fold_metrics_cnn.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("   fold_metrics_cnn.png")

    def save_report(self, output_path):
        """Save validation report as JSON."""
        output_path = Path(output_path)
        with open(output_path, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"\n Validation report saved to {output_path}")


def main():
    """Main execution."""
    # Set seeds
    torch.manual_seed(42)
    np.random.seed(42)

    # Initialize analyzer
    analyzer = ImageValidationAnalyzer(n_folds=5, random_seed=42)

    # Load data
    dataset, labels = analyzer.load_data()

    # Run validation
    results = analyzer.run_validation(dataset, labels)

    # Generate visualizations
    analyzer.generate_visualizations(FIGURES_DIR)

    # Save report
    analyzer.save_report(REPORTS_DIR / "validation_report_images.json")

    print("\n" + "=" * 80)
    print("Image-Based Validation Complete!")
    print("=" * 80)
    print(
        f"Mean R² = {results['mean_metrics']['r2']:.4f} ± {results['std_metrics']['r2']:.4f}"
    )
    print(
        f"Mean MAE = {results['mean_metrics']['mae_m']:.1f} ± {results['std_metrics']['mae_m']:.1f} m"
    )
    print(
        f"Mean RMSE = {results['mean_metrics']['rmse_m']:.1f} ± {results['std_metrics']['rmse_m']:.1f} m"
    )
    print("\nOutputs:")
    print(f"  - Report: {REPORTS_DIR / 'validation_report_images.json'}")
    print(f"  - Figures: {FIGURES_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
