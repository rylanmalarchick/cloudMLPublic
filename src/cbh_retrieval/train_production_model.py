#!/usr/bin/env python3
"""
Sprint 6 - Phase 1, Task 1.4: Final Production Model Training

This script trains the final production GBDT model on the full dataset,
saves the model checkpoint, performs inference benchmarking, and generates
comprehensive reproducibility documentation.

Deliverables:
- Production model checkpoint (.pkl)
- Hyperparameters configuration (JSON)
- Training metrics report (JSON)
- Inference benchmark report (JSON)
- Reproducibility documentation (Markdown)

Author: Sprint 6 Agent
Date: 2025
"""

import json
import pickle
import platform
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import h5py
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300

print("=" * 80)
print("Sprint 6 - Phase 1, Task 1.4: Final Production Model Training")
print("=" * 80)

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[3]
INTEGRATED_FEATURES = (
    PROJECT_ROOT / "outputs/preprocessed_data/Integrated_Features.hdf5"
)
OUTPUT_DIR = PROJECT_ROOT / "."
CHECKPOINTS_DIR = OUTPUT_DIR / "checkpoints"
REPORTS_DIR = OUTPUT_DIR / "reports"
FIGURES_DIR = OUTPUT_DIR / "figures/production"

# Create directories
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

print(f"Project Root: {PROJECT_ROOT}")
print(f"Integrated Features: {INTEGRATED_FEATURES}")
print(f"Output Directory: {OUTPUT_DIR}")
print(f" Checkpoints directory: {CHECKPOINTS_DIR}")
print(f" Reports directory: {REPORTS_DIR}")
print(f" Figures directory: {FIGURES_DIR}")


def get_system_info() -> Dict:
    """Collect system information for reproducibility."""
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu": platform.processor(),
        "cpu_count": psutil.cpu_count(logical=False),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "timestamp": datetime.now().isoformat(),
    }


class ProductionModelTrainer:
    """Trains and saves the final production GBDT model."""

    def __init__(self, random_seed=42):
        self.random_seed = random_seed
        self.checkpoints_dir = CHECKPOINTS_DIR
        self.reports_dir = REPORTS_DIR
        self.figures_dir = FIGURES_DIR

    def load_data(self, hdf5_path: Path) -> Tuple[np.ndarray, np.ndarray, list, Dict]:
        """Load tabular features and labels from integrated features file."""
        print("\n" + "=" * 80)
        print("Loading Dataset")
        print("=" * 80)

        with h5py.File(hdf5_path, "r") as f:
            # Load labels
            cbh_km = f["metadata/cbh_km"][:]
            flight_ids = f["metadata/flight_id"][:]
            sample_ids = f["metadata/sample_id"][:]

            # Load atmospheric features
            era5_features = f["atmospheric_features/era5_features"][:]
            era5_feature_names = [
                name.decode("utf-8") if isinstance(name, bytes) else name
                for name in f["atmospheric_features/era5_feature_names"][:]
            ]

            # Load geometric features
            geometric_features = {}
            for key in f["geometric_features"].keys():
                if key != "derived_geometric_H":
                    data = f[f"geometric_features/{key}"][:]
                    if data.ndim == 1:
                        geometric_features[key] = data

            # Get flight mapping
            flight_mapping = json.loads(f.attrs["flight_mapping"])

        print(f" Loaded {len(cbh_km)} samples")
        print(f" CBH range: [{cbh_km.min():.3f}, {cbh_km.max():.3f}] km")
        print(
            f" Atmospheric features: {len(era5_feature_names)} ({', '.join(era5_feature_names)})"
        )
        print(
            f" Geometric features: {len(geometric_features)} ({', '.join(geometric_features.keys())})"
        )

        # Combine all features
        feature_list = [era5_features]
        feature_names = era5_feature_names.copy()

        for name, values in geometric_features.items():
            feature_list.append(values.reshape(-1, 1))
            feature_names.append(name)

        X = np.hstack(feature_list)
        y = cbh_km

        print(f"\n Total feature matrix shape: {X.shape}")
        print(f" Feature names ({len(feature_names)}): {feature_names}")

        # Create metadata
        metadata = {
            "n_samples": len(y),
            "n_features": X.shape[1],
            "feature_names": feature_names,
            "cbh_range_km": [float(cbh_km.min()), float(cbh_km.max())],
            "cbh_mean_km": float(cbh_km.mean()),
            "cbh_std_km": float(cbh_km.std()),
            "flight_mapping": flight_mapping,
        }

        return X, y, feature_names, metadata

    def train_production_model(
        self, X: np.ndarray, y: np.ndarray, feature_names: list
    ) -> Tuple[GradientBoostingRegressor, StandardScaler, Dict, Dict]:
        """Train production GBDT model on full dataset."""
        print("\n" + "=" * 80)
        print("Training Production Model")
        print("=" * 80)

        # Hyperparameters (based on validation results)
        hyperparameters = {
            "n_estimators": 200,
            "max_depth": 8,
            "learning_rate": 0.05,
            "min_samples_split": 10,
            "min_samples_leaf": 4,
            "subsample": 0.8,
            "random_state": self.random_seed,
            "verbose": 0,
        }

        print("\nHyperparameters:")
        for key, value in hyperparameters.items():
            print(f"  {key}: {value}")

        # Normalize features
        print("\nNormalizing features...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        print(" Features normalized")

        # Train model
        print("\nTraining model on full dataset...")
        start_time = time.time()

        model = GradientBoostingRegressor(**hyperparameters)
        model.fit(X_scaled, y)

        training_time = time.time() - start_time
        print(f" Model trained in {training_time:.2f} seconds")

        # Make predictions
        print("\nEvaluating on training set...")
        predictions = model.predict(X_scaled)

        # Calculate metrics
        r2 = r2_score(y, predictions)
        mae = mean_absolute_error(y, predictions)
        rmse = np.sqrt(mean_squared_error(y, predictions))

        metrics = {
            "r2": float(r2),
            "mae_km": float(mae),
            "mae_m": float(mae * 1000),
            "rmse_km": float(rmse),
            "rmse_m": float(rmse * 1000),
            "n_samples": len(y),
            "training_time_seconds": float(training_time),
        }

        print("\nTraining Set Performance:")
        print(f"  R² = {r2:.4f}")
        print(f"  MAE = {mae * 1000:.1f} m")
        print(f"  RMSE = {rmse * 1000:.1f} m")
        print(f"  Training time = {training_time:.2f} s")

        # Feature importance
        feature_importance = pd.DataFrame(
            {
                "feature": feature_names,
                "importance": model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)

        print("\nTop 10 Most Important Features:")
        for i, row in feature_importance.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")

        # Create feature importance dict
        feature_importance_dict = dict(
            zip(feature_importance["feature"], feature_importance["importance"])
        )

        return model, scaler, hyperparameters, metrics, feature_importance_dict

    def save_production_checkpoint(
        self,
        model: GradientBoostingRegressor,
        scaler: StandardScaler,
        hyperparameters: Dict,
        metrics: Dict,
        feature_names: list,
        feature_importance: Dict,
        metadata: Dict,
    ):
        """Save production model checkpoint and configuration."""
        print("\n" + "=" * 80)
        print("Saving Production Checkpoint")
        print("=" * 80)

        # Save model
        model_path = self.checkpoints_dir / "production_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        print(f" Model saved: {model_path}")

        # Save scaler
        scaler_path = self.checkpoints_dir / "production_scaler.pkl"
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)
        print(f" Scaler saved: {scaler_path}")

        # Save using joblib (alternative format)
        joblib_model_path = self.checkpoints_dir / "production_model.joblib"
        joblib.dump(model, joblib_model_path)
        print(f" Model saved (joblib): {joblib_model_path}")

        joblib_scaler_path = self.checkpoints_dir / "production_scaler.joblib"
        joblib.dump(scaler, joblib_scaler_path)
        print(f" Scaler saved (joblib): {joblib_scaler_path}")

        # Save configuration
        config = {
            "model_type": "GradientBoostingRegressor",
            "model_name": "CBH_Production_GBDT",
            "checkpoint_path": str(model_path),
            "scaler_path": str(scaler_path),
            "hyperparameters": hyperparameters,
            "metrics": metrics,
            "feature_names": feature_names,
            "feature_importance": {k: float(v) for k, v in feature_importance.items()},
            "metadata": metadata,
            "system_info": get_system_info(),
            "timestamp": datetime.now().isoformat(),
        }

        config_path = self.checkpoints_dir / "production_config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f" Configuration saved: {config_path}")

        return config

    def benchmark_inference(
        self, model: GradientBoostingRegressor, scaler: StandardScaler, X: np.ndarray
    ) -> Dict:
        """Benchmark inference performance on CPU."""
        print("\n" + "=" * 80)
        print("Benchmarking Inference Performance")
        print("=" * 80)

        X_scaled = scaler.transform(X)

        batch_sizes = [1, 4, 16, 32, 64, 128]
        benchmark_results = {"cpu_inference": {}}

        print("\nCPU Inference Benchmarking:")

        for batch_size in batch_sizes:
            n_iterations = min(100, len(X) // batch_size)
            if n_iterations == 0:
                continue

            times = []

            for i in range(n_iterations):
                start_idx = (i * batch_size) % (len(X) - batch_size)
                end_idx = start_idx + batch_size

                batch = X_scaled[start_idx:end_idx]

                start_time = time.time()
                _ = model.predict(batch)
                elapsed = (time.time() - start_time) * 1000  # Convert to ms

                times.append(elapsed)

            mean_time = np.mean(times)
            std_time = np.std(times)
            throughput = batch_size / (mean_time / 1000)  # samples/sec

            benchmark_results["cpu_inference"][f"batch_{batch_size}"] = {
                "mean_time_ms": float(mean_time),
                "std_time_ms": float(std_time),
                "throughput_samples_per_sec": float(throughput),
                "n_iterations": n_iterations,
            }

            print(
                f"  Batch {batch_size:3d}: {mean_time:6.2f} ± {std_time:5.2f} ms  "
                f"({throughput:7.1f} samples/sec)"
            )

        # Single sample latency (important for real-time applications)
        single_sample_times = []
        for i in range(1000):
            sample = X_scaled[i % len(X) : (i % len(X)) + 1]
            start_time = time.time()
            _ = model.predict(sample)
            elapsed = (time.time() - start_time) * 1000
            single_sample_times.append(elapsed)

        benchmark_results["single_sample_latency"] = {
            "mean_ms": float(np.mean(single_sample_times)),
            "std_ms": float(np.std(single_sample_times)),
            "median_ms": float(np.median(single_sample_times)),
            "p95_ms": float(np.percentile(single_sample_times, 95)),
            "p99_ms": float(np.percentile(single_sample_times, 99)),
        }

        print(f"\nSingle Sample Latency:")
        print(f"  Mean: {benchmark_results['single_sample_latency']['mean_ms']:.3f} ms")
        print(
            f"  Median: {benchmark_results['single_sample_latency']['median_ms']:.3f} ms"
        )
        print(f"  P95: {benchmark_results['single_sample_latency']['p95_ms']:.3f} ms")
        print(f"  P99: {benchmark_results['single_sample_latency']['p99_ms']:.3f} ms")

        return benchmark_results

    def save_benchmark_report(self, benchmark_results: Dict):
        """Save benchmark report."""
        print("\n" + "=" * 80)
        print("Saving Benchmark Report")
        print("=" * 80)

        report_path = self.reports_dir / "production_inference_benchmark.json"
        with open(report_path, "w") as f:
            json.dump(benchmark_results, f, indent=2)
        print(f" Benchmark report saved: {report_path}")

    def create_production_visualizations(
        self,
        model: GradientBoostingRegressor,
        scaler: StandardScaler,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list,
        feature_importance: Dict,
    ):
        """Create visualizations for production model."""
        print("\n" + "=" * 80)
        print("Creating Production Visualizations")
        print("=" * 80)

        X_scaled = scaler.transform(X)
        predictions = model.predict(X_scaled)
        errors = predictions - y

        # 1. Predictions vs. Actual
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(y * 1000, predictions * 1000, alpha=0.5, s=20)
        ax.plot(
            [y.min() * 1000, y.max() * 1000],
            [y.min() * 1000, y.max() * 1000],
            "r--",
            linewidth=2,
            label="Perfect Prediction",
        )
        ax.set_xlabel("True CBH (m)")
        ax.set_ylabel("Predicted CBH (m)")
        ax.set_title("Production Model: Predictions vs. Actual")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.figures_dir / "production_predictions.png", dpi=300)
        plt.savefig(self.figures_dir / "production_predictions.pdf")
        plt.close()
        print(" Saved production_predictions.png/pdf")

        # 2. Residuals
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(predictions * 1000, errors * 1000, alpha=0.5, s=20)
        ax.axhline(0, color="red", linestyle="--", linewidth=2)
        ax.set_xlabel("Predicted CBH (m)")
        ax.set_ylabel("Residual (m)")
        ax.set_title("Production Model: Residual Plot")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.figures_dir / "production_residuals.png", dpi=300)
        plt.savefig(self.figures_dir / "production_residuals.pdf")
        plt.close()
        print(" Saved production_residuals.png/pdf")

        # 3. Feature Importance
        importance_df = pd.DataFrame(
            {
                "feature": list(feature_importance.keys()),
                "importance": list(feature_importance.values()),
            }
        ).sort_values("importance", ascending=True)

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(importance_df["feature"].tail(15), importance_df["importance"].tail(15))
        ax.set_xlabel("Feature Importance")
        ax.set_title("Production Model: Top 15 Feature Importance")
        ax.grid(True, alpha=0.3, axis="x")
        plt.tight_layout()
        plt.savefig(self.figures_dir / "production_feature_importance.png", dpi=300)
        plt.savefig(self.figures_dir / "production_feature_importance.pdf")
        plt.close()
        print(" Saved production_feature_importance.png/pdf")

        print(f"\n All visualizations saved to: {self.figures_dir}")

    def generate_reproducibility_docs(
        self, config: Dict, benchmark_results: Dict, metadata: Dict
    ):
        """Generate comprehensive reproducibility documentation."""
        print("\n" + "=" * 80)
        print("Generating Reproducibility Documentation")
        print("=" * 80)

        doc_lines = [
            "# Production Model Documentation",
            "",
            f"**Generated**: {datetime.now().isoformat()}",
            "",
            "---",
            "",
            "## Model Information",
            "",
            f"**Model Type**: {config['model_type']}",
            f"**Model Name**: {config['model_name']}",
            f"**Checkpoint Path**: `{config['checkpoint_path']}`",
            f"**Scaler Path**: `{config['scaler_path']}`",
            "",
            "---",
            "",
            "## Training Dataset",
            "",
            f"**Number of Samples**: {metadata['n_samples']}",
            f"**Number of Features**: {metadata['n_features']}",
            f"**CBH Range**: [{metadata['cbh_range_km'][0]:.3f}, {metadata['cbh_range_km'][1]:.3f}] km",
            f"**CBH Mean**: {metadata['cbh_mean_km']:.3f} km",
            f"**CBH Std**: {metadata['cbh_std_km']:.3f} km",
            "",
            "### Features",
            "",
        ]

        for feat in metadata["feature_names"]:
            doc_lines.append(f"- {feat}")

        doc_lines.extend(
            [
                "",
                "---",
                "",
                "## Hyperparameters",
                "",
                "```json",
                json.dumps(config["hyperparameters"], indent=2),
                "```",
                "",
                "---",
                "",
                "## Performance Metrics",
                "",
                f"**R²**: {config['metrics']['r2']:.4f}",
                f"**MAE**: {config['metrics']['mae_m']:.1f} m",
                f"**RMSE**: {config['metrics']['rmse_m']:.1f} m",
                f"**Training Time**: {config['metrics']['training_time_seconds']:.2f} seconds",
                "",
                "---",
                "",
                "## Feature Importance (Top 10)",
                "",
                "| Rank | Feature | Importance |",
                "|------|---------|------------|",
            ]
        )

        importance_sorted = sorted(
            config["feature_importance"].items(), key=lambda x: x[1], reverse=True
        )
        for i, (feat, imp) in enumerate(importance_sorted[:10], 1):
            doc_lines.append(f"| {i} | {feat} | {imp:.4f} |")

        doc_lines.extend(
            [
                "",
                "---",
                "",
                "## Inference Performance",
                "",
                "### CPU Inference",
                "",
                "| Batch Size | Mean Time (ms) | Std Time (ms) | Throughput (samples/s) |",
                "|------------|----------------|---------------|------------------------|",
            ]
        )

        for batch_key, batch_stats in benchmark_results["cpu_inference"].items():
            batch_size = batch_key.replace("batch_", "")
            doc_lines.append(
                f"| {batch_size} | {batch_stats['mean_time_ms']:.2f} | "
                f"{batch_stats['std_time_ms']:.2f} | "
                f"{batch_stats['throughput_samples_per_sec']:.1f} |"
            )

        doc_lines.extend(
            [
                "",
                "### Single Sample Latency",
                "",
                f"- **Mean**: {benchmark_results['single_sample_latency']['mean_ms']:.3f} ms",
                f"- **Median**: {benchmark_results['single_sample_latency']['median_ms']:.3f} ms",
                f"- **P95**: {benchmark_results['single_sample_latency']['p95_ms']:.3f} ms",
                f"- **P99**: {benchmark_results['single_sample_latency']['p99_ms']:.3f} ms",
                "",
                "---",
                "",
                "## System Information",
                "",
                f"**Platform**: {config['system_info']['platform']}",
                f"**Python Version**: {config['system_info']['python_version']}",
                f"**CPU**: {config['system_info']['cpu']}",
                f"**CPU Count**: {config['system_info']['cpu_count']} physical, {config['system_info']['cpu_count_logical']} logical",
                f"**RAM**: {config['system_info']['ram_gb']} GB",
                "",
                "---",
                "",
                "## Usage Example",
                "",
                "```python",
                "import pickle",
                "import numpy as np",
                "",
                "# Load model and scaler",
                f"with open('{config['checkpoint_path']}', 'rb') as f:",
                "    model = pickle.load(f)",
                "",
                f"with open('{config['scaler_path']}', 'rb') as f:",
                "    scaler = pickle.load(f)",
                "",
                "# Prepare features (18 features in correct order)",
                f"# Feature order: {', '.join(metadata['feature_names'][:5])}...",
                "features = np.array([[...]])  # Shape: (n_samples, 18)",
                "",
                "# Normalize and predict",
                "features_scaled = scaler.transform(features)",
                "cbh_km = model.predict(features_scaled)",
                "```",
                "",
                "---",
                "",
                "## Model Card",
                "",
                "### Intended Use",
                "",
                "This model is designed for Cloud Base Height (CBH) retrieval from",
                "integrated features including atmospheric (ERA5) and geometric (shadow-based)",
                "measurements. It is intended for research purposes and operational CBH",
                "prediction in the NASA ACTIVATE field campaign.",
                "",
                "### Limitations",
                "",
                "- Trained on specific flight conditions (Flights F1, F2, F4)",
                "- Performance may degrade on out-of-distribution samples",
                "- Assumes availability of all 18 input features",
                "- Feature normalization is required before inference",
                "",
                "### Ethical Considerations",
                "",
                "- This model is for atmospheric science research",
                "- No personal data or sensitive information is used",
                "- Results should be validated against ground truth when available",
                "",
                "---",
                "",
                f"**Documentation generated**: {datetime.now().isoformat()}",
                "",
            ]
        )

        # Save documentation
        doc_path = self.reports_dir / "production_model_documentation.md"
        with open(doc_path, "w") as f:
            f.write("\n".join(doc_lines))
        print(f" Documentation saved: {doc_path}")


def main():
    """Main execution function."""

    print("\n" + "=" * 80)
    print("Starting Production Model Training")
    print("=" * 80)

    # Initialize trainer
    trainer = ProductionModelTrainer(random_seed=42)

    # Load data
    X, y, feature_names, metadata = trainer.load_data(INTEGRATED_FEATURES)

    # Train production model
    model, scaler, hyperparameters, metrics, feature_importance = (
        trainer.train_production_model(X, y, feature_names)
    )

    # Save checkpoint
    config = trainer.save_production_checkpoint(
        model,
        scaler,
        hyperparameters,
        metrics,
        feature_names,
        feature_importance,
        metadata,
    )

    # Benchmark inference
    benchmark_results = trainer.benchmark_inference(model, scaler, X)

    # Save benchmark report
    trainer.save_benchmark_report(benchmark_results)

    # Create visualizations
    trainer.create_production_visualizations(
        model, scaler, X, y, feature_names, feature_importance
    )

    # Generate reproducibility documentation
    trainer.generate_reproducibility_docs(config, benchmark_results, metadata)

    print("\n" + "=" * 80)
    print(" Task 1.4 Complete: Final Production Model Training")
    print("=" * 80)
    print("\nProduction Artifacts:")
    print(f"  Model: {CHECKPOINTS_DIR / 'production_model.pkl'}")
    print(f"  Scaler: {CHECKPOINTS_DIR / 'production_scaler.pkl'}")
    print(f"  Config: {CHECKPOINTS_DIR / 'production_config.json'}")
    print(f"  Documentation: {REPORTS_DIR / 'production_model_documentation.md'}")
    print(f"  Benchmark: {REPORTS_DIR / 'production_inference_benchmark.json'}")
    print(f"  Figures: {FIGURES_DIR}/")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
