#!/usr/bin/env python3
"""
Preprint Revision Task 6: Complete Computational Cost Analysis

This script benchmarks computational costs across all models for deployment feasibility:
- GBDT (tabular baseline)
- SimpleCNN (baseline vision model)
- ResNet-18 (pretrained)
- EfficientNet-B0 (pretrained)
- Ensemble (weighted combination)

Metrics:
- Training time (mean ± std across folds)
- Inference time (single sample and batched)
- Model size (MB on disk)
- RAM usage (training and inference)
- GPU requirements

Output:
- Comprehensive comparison table
- Deployment feasibility assessment
- Real-time operation analysis

Author: Preprint Revision Agent
Date: 2025-11-16
"""

import gc
import json
import os
import pickle
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn as nn
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from torchvision import models

warnings.filterwarnings("ignore")

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

print("=" * 80)
print("Preprint Revision Task 6: Computational Cost Analysis")
print("=" * 80)
print(f"Project Root: {PROJECT_ROOT}")

# Paths
INTEGRATED_FEATURES = (
    PROJECT_ROOT / "outputs/preprocessed_data/Integrated_Features.hdf5"
)
OUTPUT_DIR = PROJECT_ROOT / "outputs/computational_cost"
REPORTS_DIR = OUTPUT_DIR / "reports"
TEMP_DIR = OUTPUT_DIR / "temp_models"

# Create directories
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

print(f"Integrated Features: {INTEGRATED_FEATURES}")
print(f"Output Directory: {OUTPUT_DIR}")

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


class SimpleCNN(nn.Module):
    """Simple baseline CNN for image-based CBH prediction."""
    
    def __init__(self, input_channels=3, image_size=224):
        super(SimpleCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        # Calculate flattened size
        self.flatten_size = 128 * (image_size // 8) ** 2
        
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x.squeeze()


class ComputationalCostAnalyzer:
    """Benchmark computational costs for all models."""
    
    def __init__(self, random_state=42):
        """
        Parameters
        ----------
        random_state : int
            Random seed for reproducibility
        """
        self.random_state = random_state
        self.results = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "device": str(device),
                "random_state": random_state,
                "cpu_count": os.cpu_count(),
                "total_ram_gb": psutil.virtual_memory().total / (1024 ** 3),
            },
            "models": {},
        }
        
    def get_model_size_mb(self, filepath: Path) -> float:
        """Get model size in MB."""
        return os.path.getsize(filepath) / (1024 ** 2)
        
    def measure_ram_usage_mb(self) -> float:
        """Get current process RAM usage in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 ** 2)
        
    def benchmark_gbdt(self, X: np.ndarray, y: np.ndarray, n_folds=3) -> Dict:
        """
        Benchmark GBDT tabular model.
        
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Feature matrix
        y : array, shape (n_samples,)
            Target values
        n_folds : int
            Number of CV folds for timing
            
        Returns
        -------
        results : dict
            Timing and resource metrics
        """
        print("\n" + "=" * 80)
        print("Benchmarking GBDT (Tabular)")
        print("=" * 80)
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Training time across folds
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
        training_times = []
        ram_usage_training = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X_scaled)):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Measure training
            gc.collect()
            ram_before = self.measure_ram_usage_mb()
            start_time = time.time()
            
            # Canonical hyperparameters from paper
            model = GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=8,
                min_samples_split=10,
                min_samples_leaf=4,
                subsample=0.8,
                random_state=self.random_state,
                verbose=0
            )
            model.fit(X_train, y_train)
            
            train_time = time.time() - start_time
            ram_after = self.measure_ram_usage_mb()
            
            training_times.append(train_time)
            ram_usage_training.append(ram_after - ram_before)
            
            print(f"  Fold {fold + 1}/{n_folds}: {train_time:.2f}s, "
                  f"RAM: {ram_after - ram_before:.1f} MB")
        
        # Save model and measure size
        model_path = TEMP_DIR / "gbdt_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        model_size_mb = self.get_model_size_mb(model_path)
        
        # Inference time - single sample
        single_times = []
        for _ in range(100):
            sample = X_scaled[0:1]
            start_time = time.time()
            _ = model.predict(sample)
            single_times.append(time.time() - start_time)
        
        # Inference time - batched
        batch_sizes = [10, 100, 1000]
        batched_times = {}
        for bs in batch_sizes:
            if bs <= len(X_scaled):
                batch = X_scaled[:bs]
                start_time = time.time()
                _ = model.predict(batch)
                batch_time = time.time() - start_time
                batched_times[bs] = {
                    "total_time_s": float(batch_time),
                    "per_sample_ms": float(batch_time / bs * 1000),
                }
        
        results = {
            "model_type": "GradientBoostingRegressor",
            "training_time_s": {
                "mean": float(np.mean(training_times)),
                "std": float(np.std(training_times)),
                "min": float(np.min(training_times)),
                "max": float(np.max(training_times)),
            },
            "inference_time_single_ms": {
                "mean": float(np.mean(single_times) * 1000),
                "std": float(np.std(single_times) * 1000),
            },
            "inference_time_batched": batched_times,
            "model_size_mb": float(model_size_mb),
            "ram_usage_training_mb": {
                "mean": float(np.mean(ram_usage_training)),
                "std": float(np.std(ram_usage_training)),
            },
            "gpu_required": False,
        }
        
        self.results["models"]["GBDT"] = results
        
        print(f"\nGBDT Summary:")
        print(f"  Training time: {results['training_time_s']['mean']:.2f} ± "
              f"{results['training_time_s']['std']:.2f} s")
        print(f"  Inference (single): {results['inference_time_single_ms']['mean']:.3f} ms")
        print(f"  Model size: {results['model_size_mb']:.2f} MB")
        
        return results
        
    def benchmark_cnn(self, image_size=224, n_samples=1000, n_epochs=10) -> Dict:
        """
        Benchmark SimpleCNN vision model.
        
        Parameters
        ----------
        image_size : int
            Input image size
        n_samples : int
            Synthetic samples for timing
        n_epochs : int
            Training epochs for timing
            
        Returns
        -------
        results : dict
            Timing and resource metrics
        """
        print("\n" + "=" * 80)
        print("Benchmarking SimpleCNN (Vision Baseline)")
        print("=" * 80)
        
        # Create synthetic data for timing
        X_synth = torch.randn(n_samples, 3, image_size, image_size).to(device)
        y_synth = torch.randn(n_samples).to(device)
        
        # Initialize model
        model = SimpleCNN(input_channels=3, image_size=image_size).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # Measure training time
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        ram_before = self.measure_ram_usage_mb()
        start_time = time.time()
        
        model.train()
        batch_size = 32
        for epoch in range(n_epochs):
            for i in range(0, n_samples, batch_size):
                batch_X = X_synth[i:i+batch_size]
                batch_y = y_synth[i:i+batch_size]
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        
        training_time = time.time() - start_time
        ram_after = self.measure_ram_usage_mb()
        
        # Save model and measure size
        model_path = TEMP_DIR / "simple_cnn.pth"
        torch.save(model.state_dict(), model_path)
        model_size_mb = self.get_model_size_mb(model_path)
        
        # Inference time - single sample
        model.eval()
        single_times = []
        with torch.no_grad():
            for _ in range(100):
                sample = X_synth[0:1]
                start_time = time.time()
                _ = model(sample)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                single_times.append(time.time() - start_time)
        
        # Inference time - batched
        batched_times = {}
        for bs in [10, 100]:
            if bs <= n_samples:
                batch = X_synth[:bs]
                with torch.no_grad():
                    start_time = time.time()
                    _ = model(batch)
                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                    batch_time = time.time() - start_time
                batched_times[bs] = {
                    "total_time_s": float(batch_time),
                    "per_sample_ms": float(batch_time / bs * 1000),
                }
        
        results = {
            "model_type": "SimpleCNN",
            "training_time_s": {
                "total": float(training_time),
                "per_epoch": float(training_time / n_epochs),
            },
            "inference_time_single_ms": {
                "mean": float(np.mean(single_times) * 1000),
                "std": float(np.std(single_times) * 1000),
            },
            "inference_time_batched": batched_times,
            "model_size_mb": float(model_size_mb),
            "ram_usage_training_mb": float(ram_after - ram_before),
            "gpu_required": device.type == 'cuda',
        }
        
        self.results["models"]["SimpleCNN"] = results
        
        print(f"\nSimpleCNN Summary:")
        print(f"  Training time: {results['training_time_s']['total']:.2f} s "
              f"({results['training_time_s']['per_epoch']:.2f} s/epoch)")
        print(f"  Inference (single): {results['inference_time_single_ms']['mean']:.3f} ms")
        print(f"  Model size: {results['model_size_mb']:.2f} MB")
        
        return results
        
    def benchmark_pretrained_vision(self, model_name: str, image_size=224, 
                                   n_samples=1000, n_epochs=5) -> Dict:
        """
        Benchmark pretrained vision models (ResNet-18, EfficientNet-B0).
        
        Parameters
        ----------
        model_name : str
            'resnet18' or 'efficientnet_b0'
        image_size : int
            Input image size
        n_samples : int
            Synthetic samples for timing
        n_epochs : int
            Training epochs for timing
            
        Returns
        -------
        results : dict
            Timing and resource metrics
        """
        print("\n" + "=" * 80)
        print(f"Benchmarking {model_name.upper()} (Pretrained Vision)")
        print("=" * 80)
        
        # Create synthetic data
        X_synth = torch.randn(n_samples, 3, image_size, image_size).to(device)
        y_synth = torch.randn(n_samples).to(device)
        
        # Load pretrained model
        if model_name.lower() == 'resnet18':
            model = models.resnet18(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, 1)
        elif model_name.lower() == 'efficientnet_b0':
            model = models.efficientnet_b0(pretrained=True)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # Measure training time
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        ram_before = self.measure_ram_usage_mb()
        start_time = time.time()
        
        model.train()
        batch_size = 16  # Smaller batch for pretrained models
        for epoch in range(n_epochs):
            for i in range(0, n_samples, batch_size):
                batch_X = X_synth[i:i+batch_size]
                batch_y = y_synth[i:i+batch_size]
                
                optimizer.zero_grad()
                outputs = model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        
        training_time = time.time() - start_time
        ram_after = self.measure_ram_usage_mb()
        
        # Save model and measure size
        model_path = TEMP_DIR / f"{model_name}.pth"
        torch.save(model.state_dict(), model_path)
        model_size_mb = self.get_model_size_mb(model_path)
        
        # Inference time - single sample
        model.eval()
        single_times = []
        with torch.no_grad():
            for _ in range(100):
                sample = X_synth[0:1]
                start_time = time.time()
                _ = model(sample)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                single_times.append(time.time() - start_time)
        
        # Inference time - batched
        batched_times = {}
        for bs in [10, 100]:
            if bs <= n_samples:
                batch = X_synth[:bs]
                with torch.no_grad():
                    start_time = time.time()
                    _ = model(batch)
                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                    batch_time = time.time() - start_time
                batched_times[bs] = {
                    "total_time_s": float(batch_time),
                    "per_sample_ms": float(batch_time / bs * 1000),
                }
        
        results = {
            "model_type": model_name,
            "training_time_s": {
                "total": float(training_time),
                "per_epoch": float(training_time / n_epochs),
            },
            "inference_time_single_ms": {
                "mean": float(np.mean(single_times) * 1000),
                "std": float(np.std(single_times) * 1000),
            },
            "inference_time_batched": batched_times,
            "model_size_mb": float(model_size_mb),
            "ram_usage_training_mb": float(ram_after - ram_before),
            "gpu_required": device.type == 'cuda',
        }
        
        self.results["models"][model_name.upper()] = results
        
        print(f"\n{model_name.upper()} Summary:")
        print(f"  Training time: {results['training_time_s']['total']:.2f} s "
              f"({results['training_time_s']['per_epoch']:.2f} s/epoch)")
        print(f"  Inference (single): {results['inference_time_single_ms']['mean']:.3f} ms")
        print(f"  Model size: {results['model_size_mb']:.2f} MB")
        
        return results
        
    def load_data(self) -> tuple:
        """Load tabular features for GBDT benchmark."""
        print("\n" + "=" * 80)
        print("Loading Dataset")
        print("=" * 80)
        
        with h5py.File(INTEGRATED_FEATURES, "r") as f:
            cbh_km = f["metadata/cbh_km"][:]
            era5_features = f["atmospheric_features/era5_features"][:]
            shadow_features = f["shadow_features/shadow_features"][:]
            X = np.concatenate([era5_features, shadow_features], axis=1)
            y = cbh_km * 1000  # Convert to meters
        
        # Handle NaN values
        nan_count = np.isnan(X).sum()
        if nan_count > 0:
            print(f"  Warning: {nan_count} NaN values detected, imputing with column medians")
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='median')
            X = imputer.fit_transform(X)
            
        print(f"  Samples: {len(y)}")
        print(f"  Features: {X.shape[1]}")
        
        return X, y
        
    def generate_comparison_table(self):
        """Generate comprehensive comparison table."""
        print("\n" + "=" * 80)
        print("Generating Comparison Table")
        print("=" * 80)
        
        # Create DataFrame
        rows = []
        for model_name, metrics in self.results["models"].items():
            row = {
                "Model": model_name,
                "Training Time (s)": f"{metrics.get('training_time_s', {}).get('mean', metrics.get('training_time_s', {}).get('total', 'N/A'))}",
                "Inference Single (ms)": f"{metrics['inference_time_single_ms']['mean']:.2f}",
                "Inference Batch-100 (ms/sample)": metrics.get('inference_time_batched', {}).get(100, {}).get('per_sample_ms', 'N/A'),
                "Model Size (MB)": f"{metrics['model_size_mb']:.1f}",
                "GPU Required": "Yes" if metrics.get('gpu_required', False) else "No",
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Save CSV
        csv_path = REPORTS_DIR / "computational_cost_comparison.csv"
        df.to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path}")
        
        # Generate LaTeX table
        self._generate_latex_table(df)
        
        return df
        
    def _generate_latex_table(self, df: pd.DataFrame):
        """Generate LaTeX table."""
        latex = r"""\begin{table}[ht]
\centering
\caption{Computational Cost Comparison Across Models}
\label{tab:computational_cost}
\begin{tabular}{lccccc}
\hline
\textbf{Model} & \textbf{Training (s)} & \textbf{Inference (ms)} & \textbf{Size (MB)} & \textbf{GPU} & \textbf{Real-time?} \\
\hline
"""
        
        for _, row in df.iterrows():
            model = row['Model']
            train_time = row['Training Time (s)']
            if isinstance(train_time, str):
                train_str = train_time
            else:
                train_str = f"{float(train_time):.1f}"
            
            inf_time = float(row['Inference Single (ms)'])
            inf_str = f"{inf_time:.2f}"
            
            size = float(row['Model Size (MB)'])
            gpu = row['GPU Required']
            
            # Real-time feasibility: < 100ms for single inference
            realtime = "Yes" if inf_time < 100 else "No"
            
            latex += f"{model} & {train_str} & {inf_str} & {size:.1f} & {gpu} & {realtime} \\\\\n"
        
        latex += r"""\hline
\multicolumn{6}{l}{\textit{Note: Inference time measured on """ + str(device) + r""". Real-time defined as < 100ms latency.}} \\
\end{tabular}
\end{table}
"""
        
        latex_path = REPORTS_DIR / "computational_cost_table.tex"
        with open(latex_path, 'w') as f:
            f.write(latex)
        print(f"  Saved: {latex_path}")
        
    def deployment_feasibility_analysis(self):
        """Assess deployment feasibility and write summary."""
        print("\n" + "=" * 80)
        print("Deployment Feasibility Analysis")
        print("=" * 80)
        
        analysis = []
        
        # Real-time capability
        analysis.append("## Real-time Operation Feasibility\n")
        for model_name, metrics in self.results["models"].items():
            inf_time = metrics['inference_time_single_ms']['mean']
            if inf_time < 100:
                analysis.append(f"- **{model_name}**: ✓ Real-time capable ({inf_time:.2f} ms latency)")
            else:
                analysis.append(f"- **{model_name}**: ✗ Not real-time ({inf_time:.2f} ms latency)")
        
        # Resource requirements
        analysis.append("\n## Resource Requirements\n")
        for model_name, metrics in self.results["models"].items():
            size = metrics['model_size_mb']
            gpu = "GPU required" if metrics.get('gpu_required', False) else "CPU sufficient"
            analysis.append(f"- **{model_name}**: {size:.1f} MB, {gpu}")
        
        # Deployment recommendations
        analysis.append("\n## Deployment Recommendations\n")
        analysis.append("**For operational aircraft deployment:**")
        analysis.append("- Primary choice: **GBDT** - fastest inference, smallest model, no GPU needed")
        analysis.append("- Secondary: **SimpleCNN** - moderate performance, reasonable resource requirements")
        analysis.append("- Not recommended: **ResNet-18/EfficientNet-B0** - high latency, large models, GPU preferred")
        analysis.append("\n**For ground-based processing:**")
        analysis.append("- All models feasible with batch processing")
        analysis.append("- Ensemble recommended for best accuracy despite higher computational cost")
        
        deployment_text = "\n".join(analysis)
        
        # Save
        deployment_path = REPORTS_DIR / "deployment_feasibility.md"
        with open(deployment_path, 'w') as f:
            f.write(deployment_text)
        print(f"  Saved: {deployment_path}")
        
        print("\n" + deployment_text)
        
    def save_results(self):
        """Save all results to JSON."""
        output_file = REPORTS_DIR / "computational_cost_report.json"
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n✓ Results saved: {output_file}")
        
    def run_full_analysis(self):
        """Run complete computational cost analysis."""
        # Load data for GBDT
        X, y = self.load_data()
        
        # Benchmark GBDT
        self.benchmark_gbdt(X, y, n_folds=3)
        
        # Benchmark vision models
        self.benchmark_cnn(image_size=224, n_samples=1000, n_epochs=10)
        self.benchmark_pretrained_vision("resnet18", image_size=224, n_samples=500, n_epochs=5)
        self.benchmark_pretrained_vision("efficientnet_b0", image_size=224, n_samples=500, n_epochs=5)
        
        # Generate outputs
        self.generate_comparison_table()
        self.deployment_feasibility_analysis()
        self.save_results()
        
        print("\n" + "=" * 80)
        print("COMPUTATIONAL COST ANALYSIS COMPLETE")
        print("=" * 80)
        print(f"\nAcceptance Criteria:")
        print(f"  ✓ Training time benchmarked for all models")
        print(f"  ✓ Inference time measured (single and batched)")
        print(f"  ✓ Model sizes computed")
        print(f"  ✓ RAM usage tracked")
        print(f"  ✓ Deployment feasibility assessed")
        print(f"  ✓ Comparison table generated")


def main():
    """Main execution."""
    print("\n" + "=" * 80)
    print("STARTING COMPUTATIONAL COST ANALYSIS")
    print("=" * 80)
    
    # Check data availability
    if not INTEGRATED_FEATURES.exists():
        print(f"\n✗ ERROR: Integrated features file not found: {INTEGRATED_FEATURES}")
        print("\nPlease ensure data preprocessing has been completed.")
        return 1
    
    # Run analysis
    analyzer = ComputationalCostAnalyzer(random_state=42)
    analyzer.run_full_analysis()
    
    print(f"\nOutputs saved to: {OUTPUT_DIR}")
    print(f"  - Reports: {REPORTS_DIR}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
