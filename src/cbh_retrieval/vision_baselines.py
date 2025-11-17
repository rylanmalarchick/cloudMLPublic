#!/usr/bin/env python3
"""
Fair Deep Learning Baselines for Cloud Base Height Retrieval

Implements transfer learning baselines (ResNet-18, EfficientNet-B0) with:
- ImageNet pre-training
- Cloud-specific data augmentation
- Ablation: scratch vs. pre-trained vs. augmented
- Computational cost benchmarking

Author: Preprint Revision Task 1
Date: 2025
"""

import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset, Subset

warnings.filterwarnings("ignore")

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

print("=" * 80)
print("Fair Deep Learning Baselines for CBH Retrieval")
print("=" * 80)
print(f"Project Root: {PROJECT_ROOT}")

# Paths
SSL_IMAGES = PROJECT_ROOT / "data_ssl/images/train.h5"
INTEGRATED_FEATURES = PROJECT_ROOT / "outputs/preprocessed_data/Integrated_Features.hdf5"
OUTPUT_DIR = PROJECT_ROOT / "outputs/vision_baselines"
FIGURES_DIR = OUTPUT_DIR / "figures"
REPORTS_DIR = OUTPUT_DIR / "reports"

# Create directories
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


class CloudAugmentedDataset(Dataset):
    """Dataset with cloud-specific augmentation."""

    def __init__(self, base_dataset, augment=False):
        self.base_dataset = base_dataset
        self.augment = augment

        if augment:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            ])
        else:
            self.transform = None

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, cbh, flight_id, sample_idx, extra = self.base_dataset[idx]

        # Convert to 3-channel for ImageNet models
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)

        # Apply augmentation
        if self.transform is not None:
            image = self.transform(image)

        return image, cbh, flight_id, sample_idx, extra


class ResNet18CBH(nn.Module):
    """ResNet-18 adapted for CBH regression."""

    def __init__(self, pretrained=True, dropout=0.3):
        super(ResNet18CBH, self).__init__()

        if pretrained:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            self.backbone = models.resnet18(weights=weights)
        else:
            self.backbone = models.resnet18(weights=None)

        # Modify first conv to accept custom image size
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.backbone.maxpool = nn.Identity()  # Remove maxpool for small images

        # Replace final layer for regression
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.backbone(x).squeeze()


class EfficientNetB0CBH(nn.Module):
    """EfficientNet-B0 adapted for CBH regression."""

    def __init__(self, pretrained=True, dropout=0.3):
        super(EfficientNetB0CBH, self).__init__()

        if pretrained:
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
            self.backbone = models.efficientnet_b0(weights=weights)
        else:
            self.backbone = models.efficientnet_b0(weights=None)

        # Replace classifier for regression
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.backbone(x).squeeze()


class VisionBaselineValidator:
    """Validates vision models with ablations and cost analysis."""

    def __init__(self, model_name, pretrained=True, augment=False, n_folds=5, random_seed=42):
        self.model_name = model_name
        self.pretrained = pretrained
        self.augment = augment
        self.n_folds = n_folds
        self.random_seed = random_seed
        self.device = device

        self.results = {
            "config": {
                "model": model_name,
                "pretrained": pretrained,
                "augment": augment,
                "n_folds": n_folds,
                "random_seed": random_seed,
                "timestamp": datetime.now().isoformat(),
            },
            "folds": [],
            "mean_metrics": {},
            "std_metrics": {},
            "computational_cost": {},
        }

    def create_model(self):
        """Create model instance."""
        if self.model_name == "resnet18":
            return ResNet18CBH(pretrained=self.pretrained).to(self.device)
        elif self.model_name == "efficientnet_b0":
            return EfficientNetB0CBH(pretrained=self.pretrained).to(self.device)
        else:
            raise ValueError(f"Unknown model: {self.model_name}")

    def load_data(self):
        """Load dataset."""
        print("\n" + "=" * 80)
        print("Loading Dataset")
        print("=" * 80)

        # Import here to avoid circular dependencies
        try:
            from modules.image_dataset import ImageCBHDataset
        except ImportError:
            # Try alternative import path
            sys.path.insert(0, str(PROJECT_ROOT / "src/cbh_retrieval"))
            from image_dataset import ImageCBHDataset

        base_dataset = ImageCBHDataset(
            ssl_images_path=str(SSL_IMAGES),
            integrated_features_path=str(INTEGRATED_FEATURES),
            image_shape=(20, 22),
            normalize=True,
            augment=False,  # We handle augmentation separately
            return_indices=True,
        )

        # Wrap with augmentation
        dataset = CloudAugmentedDataset(base_dataset, augment=self.augment)

        # Get labels for stratification
        labels = []
        for i in range(len(dataset)):
            _, cbh, _, _, _ = dataset[i]
            labels.append(cbh.item())

        labels = np.array(labels)

        print(f"Dataset loaded: {len(dataset)} samples")
        print(f"CBH range: [{labels.min():.3f}, {labels.max():.3f}] km")
        print(f"Augmentation: {self.augment}")

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

    def benchmark_inference(self, model, data_loader, n_iterations=100):
        """Benchmark inference time."""
        model.eval()
        times = []

        with torch.no_grad():
            # Warmup
            for i, (images, _, _, _, _) in enumerate(data_loader):
                if i >= 5:
                    break
                images = images.to(self.device)
                _ = model(images)

            # Benchmark
            for i, (images, _, _, _, _) in enumerate(data_loader):
                if i >= n_iterations:
                    break

                images = images.to(self.device)
                start = time.perf_counter()
                _ = model(images)
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                end = time.perf_counter()
                times.append((end - start) * 1000)  # Convert to ms

        return {
            "mean_ms": float(np.mean(times)),
            "std_ms": float(np.std(times)),
            "median_ms": float(np.median(times)),
        }

    def get_model_size(self, model):
        """Get model size in MB."""
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_mb = (param_size + buffer_size) / 1024**2
        return size_mb

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
            train_subset, batch_size=32, shuffle=True, num_workers=4
        )
        val_loader = DataLoader(
            val_subset, batch_size=32, shuffle=False, num_workers=4
        )

        # Initialize model
        model = self.create_model()
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )

        # Training loop
        print(f"\nTraining {self.model_name} ({'pretrained' if self.pretrained else 'scratch'})...")
        n_epochs = 100
        best_val_loss = float("inf")
        patience_counter = 0
        patience = 15
        train_start = time.perf_counter()

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
                print(f"  Epoch {epoch + 1}/{n_epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        train_time = time.perf_counter() - train_start

        # Load best model
        model.load_state_dict(best_model_state)

        # Final evaluation
        train_preds, train_targets = self.evaluate(model, train_loader)
        val_preds, val_targets = self.evaluate(model, val_loader)

        # Benchmark inference
        inference_stats = self.benchmark_inference(model, val_loader, n_iterations=50)

        # Model size
        model_size_mb = self.get_model_size(model)

        # Compute metrics
        train_metrics = self._compute_metrics(train_targets, train_preds, "Train")
        val_metrics = self._compute_metrics(val_targets, val_preds, "Validation")

        return {
            "fold": fold_idx + 1,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "train_size": len(train_idx),
            "val_size": len(val_idx),
            "train_time_sec": float(train_time),
            "inference_time_ms": inference_stats,
            "model_size_mb": float(model_size_mb),
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
        print(f"Running {self.n_folds}-Fold Cross-Validation")
        print(f"Model: {self.model_name}")
        print(f"Pretrained: {self.pretrained}")
        print(f"Augmentation: {self.augment}")
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

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(indices, stratified_bins)):
            fold_result = self.train_and_evaluate_fold(dataset, train_idx, val_idx, fold_idx)

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
        self._aggregate_computational_cost(fold_results)

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

    def _aggregate_computational_cost(self, fold_results):
        """Aggregate computational cost metrics."""
        train_times = [f["train_time_sec"] for f in fold_results]
        inference_times = [f["inference_time_ms"]["mean_ms"] for f in fold_results]
        model_sizes = [f["model_size_mb"] for f in fold_results]

        self.results["computational_cost"] = {
            "mean_train_time_sec": float(np.mean(train_times)),
            "std_train_time_sec": float(np.std(train_times)),
            "mean_inference_time_ms": float(np.mean(inference_times)),
            "std_inference_time_ms": float(np.std(inference_times)),
            "model_size_mb": float(np.mean(model_sizes)),
        }

        print("\nComputational Cost:")
        print(f"  Training time: {np.mean(train_times):.1f} ± {np.std(train_times):.1f} sec/fold")
        print(f"  Inference time: {np.mean(inference_times):.2f} ± {np.std(inference_times):.2f} ms/batch")
        print(f"  Model size: {np.mean(model_sizes):.2f} MB")

    def save_results(self, output_path):
        """Save validation results as JSON."""
        output_path = Path(output_path)
        with open(output_path, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"\nResults saved to {output_path}")


def run_ablation_study():
    """Run full ablation study comparing different configurations."""
    print("\n" + "=" * 80)
    print("Running Full Ablation Study")
    print("=" * 80)

    configurations = [
        # ResNet-18 ablations
        ("resnet18", False, False, "ResNet-18 (scratch, no augment)"),
        ("resnet18", True, False, "ResNet-18 (pretrained, no augment)"),
        ("resnet18", True, True, "ResNet-18 (pretrained, augmented)"),

        # EfficientNet-B0 ablations
        ("efficientnet_b0", False, False, "EfficientNet-B0 (scratch, no augment)"),
        ("efficientnet_b0", True, False, "EfficientNet-B0 (pretrained, no augment)"),
        ("efficientnet_b0", True, True, "EfficientNet-B0 (pretrained, augmented)"),
    ]

    all_results = {}

    for model_name, pretrained, augment, description in configurations:
        print(f"\n{'=' * 80}")
        print(f"Running: {description}")
        print(f"{'=' * 80}")

        validator = VisionBaselineValidator(
            model_name=model_name,
            pretrained=pretrained,
            augment=augment,
            n_folds=5,
            random_seed=42
        )

        # Set seeds
        torch.manual_seed(42)
        np.random.seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)

        # Load data
        dataset, labels = validator.load_data()

        # Run validation
        results = validator.run_validation(dataset, labels)

        # Save individual results
        config_name = f"{model_name}_{'pretrained' if pretrained else 'scratch'}_{'augment' if augment else 'noaugment'}"
        validator.save_results(REPORTS_DIR / f"{config_name}_results.json")

        all_results[description] = results

    # Save aggregated comparison
    with open(REPORTS_DIR / "ablation_study_comparison.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Generate comparison table
    generate_comparison_table(all_results)

    print("\n" + "=" * 80)
    print("Ablation Study Complete!")
    print("=" * 80)


def generate_comparison_table(all_results):
    """Generate comparison table for ablation study."""
    print("\n" + "=" * 80)
    print("Model Comparison Table")
    print("=" * 80)

    # Print header
    print(f"{'Model':<45} {'R²':>8} {'MAE (m)':>10} {'RMSE (m)':>10} {'Train (s)':>12} {'Infer (ms)':>12} {'Size (MB)':>10}")
    print("-" * 125)

    # Print rows
    for description, results in all_results.items():
        r2 = results["mean_metrics"]["r2"]
        mae = results["mean_metrics"]["mae_m"]
        rmse = results["mean_metrics"]["rmse_m"]
        train_time = results["computational_cost"]["mean_train_time_sec"]
        infer_time = results["computational_cost"]["mean_inference_time_ms"]
        model_size = results["computational_cost"]["model_size_mb"]

        print(f"{description:<45} {r2:>8.4f} {mae:>10.1f} {rmse:>10.1f} {train_time:>12.1f} {infer_time:>12.2f} {model_size:>10.2f}")

    print("=" * 125)

    # Save as LaTeX table
    save_latex_table(all_results)


def save_latex_table(all_results):
    """Save comparison as LaTeX table."""
    latex_path = REPORTS_DIR / "model_comparison_table.tex"

    with open(latex_path, "w") as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{Deep Learning Baseline Comparison: Ablation Study Results}\n")
        f.write("\\label{tab:vision_baselines}\n")
        f.write("\\begin{tabular}{lcccccc}\n")
        f.write("\\toprule\n")
        f.write("Model & R$^2$ & MAE (m) & RMSE (m) & Train (s) & Infer (ms) & Size (MB) \\\\\n")
        f.write("\\midrule\n")

        for description, results in all_results.items():
            r2 = results["mean_metrics"]["r2"]
            r2_std = results["std_metrics"]["r2"]
            mae = results["mean_metrics"]["mae_m"]
            mae_std = results["std_metrics"]["mae_m"]
            rmse = results["mean_metrics"]["rmse_m"]
            train_time = results["computational_cost"]["mean_train_time_sec"]
            infer_time = results["computational_cost"]["mean_inference_time_ms"]
            model_size = results["computational_cost"]["model_size_mb"]

            f.write(f"{description} & {r2:.3f} $\\pm$ {r2_std:.3f} & {mae:.1f} $\\pm$ {mae_std:.1f} & {rmse:.1f} & {train_time:.1f} & {infer_time:.2f} & {model_size:.1f} \\\\\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    print(f"\nLaTeX table saved to {latex_path}")


def main():
    """Main execution."""
    run_ablation_study()


if __name__ == "__main__":
    main()
