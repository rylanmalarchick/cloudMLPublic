#!/usr/bin/env python3
"""
Sprint 6 - Phase 2, Task 2.2: Domain Adaptation for Flight F4

This script implements few-shot fine-tuning to mitigate the catastrophic
Leave-One-Flight-Out (LOO) failure on Flight F4 (R² = -3.13).

Key Challenge:
Flight F4 exhibits extreme domain shift (mean CBH = 0.249 km vs. 0.846 km for
other flights, a 2.2 standard deviation difference), causing model failure.

Solution:
Few-shot fine-tuning with 5, 10, and 20 samples from F4 to adapt the model
to the F4 domain while preserving general knowledge.

Deliverables:
- Few-shot fine-tuned models
- Domain adaptation results report
- Learning curves (R² vs. number of F4 samples)

Author: Sprint 6 Execution Agent
Date: 2025-01-10
"""

import json
import sys
import warnings
from sklearn.exceptions import ConvergenceWarning
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, r2_score
from torch.utils.data import DataLoader

# Suppress sklearn convergence and numpy deprecation warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from sow_outputs.wp5.wp5_utils import compute_metrics
from src.hdf5_dataset import HDF5CloudDataset

# Import model architecture
sys.path.insert(0, str(Path(__file__).parent.parent / "validation"))
from offline_validation import TemporalConsistencyViT, TemporalDataset

# ==============================================================================
# Domain Adaptation Class
# ==============================================================================


class DomainAdaptationF4:
    """
    Performs few-shot domain adaptation for Flight F4.
    """

    def __init__(
        self,
        output_dir: Path,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.output_dir = Path(output_dir)
        self.device = device

        # Create output directories
        self.models_dir = self.output_dir / "models" / "domain_adapted"
        self.reports_dir = self.output_dir / "reports"
        self.figures_dir = self.output_dir / "figures" / "domain_adaptation"

        for dir_path in [self.models_dir, self.reports_dir, self.figures_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        print(f" Domain adaptation initialized")
        print(f" Device: {self.device}")

    def load_data(
        self, integrated_features_path: str
    ) -> Tuple[HDF5CloudDataset, np.ndarray, np.ndarray]:
        """Load dataset and extract flight IDs"""
        print(f"\n{'=' * 80}")
        print("Loading Dataset")
        print(f"{'=' * 80}")

        dataset = HDF5CloudDataset(integrated_features_path)
        cbh_values = dataset.cbh_values

        # Extract flight IDs from dataset
        flight_ids = np.array(
            [dataset.global_to_local[i][0] for i in range(len(dataset))]
        )

        print(f" Total samples: {len(dataset)}")
        print(f" CBH range: [{cbh_values.min():.3f}, {cbh_values.max():.3f}] km")
        print(f" Unique flights: {np.unique(flight_ids)}")

        # Print per-flight statistics
        print(f"\nPer-Flight Statistics:")
        for flight_id in np.unique(flight_ids):
            flight_mask = flight_ids == flight_id
            flight_cbh = cbh_values[flight_mask]
            print(
                f"  Flight {flight_id}: N={np.sum(flight_mask)}, "
                f"Mean CBH={flight_cbh.mean():.3f} km, Std={flight_cbh.std():.3f} km"
            )

        return dataset, cbh_values, flight_ids

    def load_production_model(self, checkpoint_path: Path) -> nn.Module:
        """Load production model for fine-tuning"""
        print(f"\n{'=' * 80}")
        print("Loading Production Model")
        print(f"{'=' * 80}")

        model = TemporalConsistencyViT(
            pretrained_model="WinKawaks/vit-tiny-patch16-224", n_frames=5
        )

        if checkpoint_path.exists():
            print(f"Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            model.load_state_dict(checkpoint["model_state_dict"])
            print(f" Model loaded from checkpoint")
        else:
            print(f" Checkpoint not found: {checkpoint_path}")
            print(f" Using freshly initialized model (for testing only)")

        model = model.to(self.device)
        return model

    def get_f4_baseline(
        self,
        model: nn.Module,
        image_dataset: HDF5CloudDataset,
        cbh_values: np.ndarray,
        flight_ids: np.ndarray,
        f4_flight_id: int = 4,
    ) -> Dict:
        """Evaluate baseline LOO performance on F4 (catastrophic failure)"""
        print(f"\n{'=' * 80}")
        print("Baseline: Leave-One-Flight-Out (LOO) Evaluation on F4")
        print(f"{'=' * 80}")

        # Get F4 indices
        f4_mask = flight_ids == f4_flight_id
        f4_indices = np.where(f4_mask)[0].tolist()

        print(f"F4 samples: {len(f4_indices)}")
        print(f"F4 mean CBH: {cbh_values[f4_mask].mean():.3f} km")

        # Other flights mean CBH
        other_mask = ~f4_mask
        print(f"Other flights mean CBH: {cbh_values[other_mask].mean():.3f} km")
        print(
            f"Domain shift: {abs(cbh_values[f4_mask].mean() - cbh_values[other_mask].mean()):.3f} km"
        )

        # Evaluate on F4
        f4_dataset = TemporalDataset(image_dataset, f4_indices, cbh_values, n_frames=5)
        f4_loader = DataLoader(f4_dataset, batch_size=4, shuffle=False, num_workers=2)

        model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for frames, _, center_target in f4_loader:
                frames = frames.to(self.device)

                _, center_pred = model(frames, predict_all_frames=True)
                center_pred = center_pred.squeeze(1)

                all_preds.append(center_pred.cpu().numpy())
                all_targets.append(center_target.cpu().numpy())

        preds = np.concatenate(all_preds)
        targets = np.concatenate(all_targets)

        baseline_metrics = compute_metrics(preds, targets)

        print(f"\nBaseline LOO Results on F4:")
        print(f"  R²: {baseline_metrics['r2']:.4f}")
        print(f"  MAE: {baseline_metrics['mae_km'] * 1000:.1f} m")
        print(f"  RMSE: {baseline_metrics['rmse_km'] * 1000:.1f} m")

        if baseline_metrics["r2"] < 0:
            print(f"   Catastrophic failure detected (R² < 0)")

        return baseline_metrics

    def few_shot_fine_tune(
        self,
        model: nn.Module,
        image_dataset: HDF5CloudDataset,
        cbh_values: np.ndarray,
        few_shot_indices: List[int],
        n_epochs: int = 15,
        lr: float = 1e-5,
        lambda_temporal: float = 0.1,
    ) -> nn.Module:
        """Fine-tune model on few-shot samples"""

        # Create few-shot dataset
        few_shot_dataset = TemporalDataset(
            image_dataset, few_shot_indices, cbh_values, n_frames=5
        )
        few_shot_loader = DataLoader(
            few_shot_dataset, batch_size=2, shuffle=True, num_workers=2
        )

        # Freeze encoder, only train regression heads
        for param in model.vit.parameters():
            param.requires_grad = False
        for param in model.temporal_attention.parameters():
            param.requires_grad = False

        # Only train regression heads
        for param in model.frame_regression.parameters():
            param.requires_grad = True
        for param in model.center_regression.parameters():
            param.requires_grad = True

        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            weight_decay=1e-5,
        )
        criterion = nn.MSELoss()

        model.train()
        for epoch in range(n_epochs):
            epoch_losses = []

            for frames, all_targets, center_target in few_shot_loader:
                frames = frames.to(self.device)
                all_targets = all_targets.to(self.device)
                center_target = center_target.to(self.device)

                optimizer.zero_grad()

                all_preds, center_pred = model(frames, predict_all_frames=True)
                center_pred = center_pred.squeeze(1)

                # MSE loss on center frame
                loss_center = criterion(center_pred, center_target)

                # Temporal consistency loss
                temporal_diff = torch.abs(all_preds[:, 1:] - all_preds[:, :-1])
                loss_temporal = temporal_diff.mean()

                # Total loss
                loss = loss_center + lambda_temporal * loss_temporal

                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.item())

            if (epoch + 1) % 5 == 0:
                print(
                    f"  Epoch {epoch + 1}/{n_epochs}: Loss = {np.mean(epoch_losses):.6f}"
                )

        # Unfreeze all parameters for future use
        for param in model.parameters():
            param.requires_grad = True

        return model

    def evaluate_few_shot_experiments(
        self,
        base_model: nn.Module,
        image_dataset: HDF5CloudDataset,
        cbh_values: np.ndarray,
        flight_ids: np.ndarray,
        f4_flight_id: int = 4,
    ) -> Dict:
        """
        Evaluate few-shot fine-tuning with different numbers of F4 samples.
        """
        print(f"\n{'=' * 80}")
        print("Few-Shot Fine-Tuning Experiments")
        print(f"{'=' * 80}")

        # Get F4 indices
        f4_mask = flight_ids == f4_flight_id
        f4_indices = np.where(f4_mask)[0]

        # Shuffle F4 indices for random sampling
        np.random.seed(42)
        f4_indices_shuffled = np.random.permutation(f4_indices)

        # Split F4 into few-shot pool and test set
        n_test = max(20, len(f4_indices) // 2)  # Reserve half for testing (min 20)
        f4_test_indices = f4_indices_shuffled[:n_test].tolist()
        f4_pool_indices = f4_indices_shuffled[n_test:]

        print(f"F4 few-shot pool: {len(f4_pool_indices)} samples")
        print(f"F4 test set: {len(f4_test_indices)} samples")

        # Test dataset
        f4_test_dataset = TemporalDataset(
            image_dataset, f4_test_indices, cbh_values, n_frames=5
        )
        f4_test_loader = DataLoader(
            f4_test_dataset, batch_size=4, shuffle=False, num_workers=2
        )

        results = {}

        for n_shots in [5, 10, 20]:
            print(f"\n--- Few-Shot Experiment: {n_shots} samples ---")

            if n_shots > len(f4_pool_indices):
                print(f"   Not enough samples in pool, skipping")
                continue

            # Sample few-shot indices
            few_shot_indices = f4_pool_indices[:n_shots].tolist()

            # Clone base model for fine-tuning
            finetuned_model = TemporalConsistencyViT(
                pretrained_model="WinKawaks/vit-tiny-patch16-224", n_frames=5
            )
            finetuned_model.load_state_dict(base_model.state_dict())
            finetuned_model = finetuned_model.to(self.device)

            # Fine-tune on few-shot samples
            print(f"  Fine-tuning on {n_shots} F4 samples...")
            finetuned_model = self.few_shot_fine_tune(
                finetuned_model,
                image_dataset,
                cbh_values,
                few_shot_indices,
                n_epochs=15,
                lr=1e-5,
            )

            # Evaluate on F4 test set
            finetuned_model.eval()
            all_preds = []
            all_targets = []

            with torch.no_grad():
                for frames, _, center_target in f4_test_loader:
                    frames = frames.to(self.device)

                    _, center_pred = finetuned_model(frames, predict_all_frames=True)
                    center_pred = center_pred.squeeze(1)

                    all_preds.append(center_pred.cpu().numpy())
                    all_targets.append(center_target.cpu().numpy())

            preds = np.concatenate(all_preds)
            targets = np.concatenate(all_targets)

            metrics = compute_metrics(preds, targets)

            print(f"  Results:")
            print(f"    R²: {metrics['r2']:.4f}")
            print(f"    MAE: {metrics['mae_km'] * 1000:.1f} m")
            print(f"    RMSE: {metrics['rmse_km'] * 1000:.1f} m")

            results[f"{n_shots}_samples"] = {
                "r2": metrics["r2"],
                "mae_km": metrics["mae_km"],
                "rmse_km": metrics["rmse_km"],
                "improvement_over_baseline": metrics["r2"]
                - (-3.13),  # Baseline R² = -3.13
            }

            # Save fine-tuned checkpoint
            checkpoint_path = self.models_dir / f"f4_finetuned_{n_shots}shot.pth"
            torch.save(
                {
                    "model_state_dict": finetuned_model.state_dict(),
                    "n_shots": n_shots,
                    "f4_metrics": metrics,
                },
                checkpoint_path,
            )
            print(f"   Checkpoint saved: {checkpoint_path}")

        return results

    def create_visualizations(self, baseline_metrics: Dict, few_shot_results: Dict):
        """Create learning curves and comparison plots"""
        print(f"\n{'=' * 80}")
        print("Generating Domain Adaptation Visualizations")
        print(f"{'=' * 80}")

        sns.set_style("whitegrid")

        # Learning curve: R² vs. number of few-shot samples
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        n_samples = [0] + [int(k.split("_")[0]) for k in few_shot_results.keys()]
        r2_scores = [baseline_metrics["r2"]] + [
            v["r2"] for v in few_shot_results.values()
        ]
        mae_scores = [baseline_metrics["mae_km"] * 1000] + [
            v["mae_km"] * 1000 for v in few_shot_results.values()
        ]

        # R² curve
        axes[0].plot(n_samples, r2_scores, "o-", lw=2, markersize=10, color="blue")
        axes[0].axhline(
            y=0, color="red", linestyle="--", lw=2, label="R² = 0 (baseline failure)"
        )
        axes[0].axhline(
            y=0.7,
            color="green",
            linestyle="--",
            lw=1,
            alpha=0.5,
            label="Target R² ≈ 0.7",
        )
        axes[0].set_xlabel("Number of F4 Few-Shot Samples", fontsize=12)
        axes[0].set_ylabel("R² on F4 Test Set", fontsize=12)
        axes[0].set_title("Few-Shot Learning Curve: R²", fontsize=14)
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xlim(-1, max(n_samples) + 2)

        # Annotate points
        for x, y in zip(n_samples, r2_scores):
            axes[0].annotate(
                f"{y:.3f}",
                xy=(x, y),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
                fontsize=9,
            )

        # MAE curve
        axes[1].plot(n_samples, mae_scores, "o-", lw=2, markersize=10, color="orange")
        axes[1].set_xlabel("Number of F4 Few-Shot Samples", fontsize=12)
        axes[1].set_ylabel("MAE on F4 Test Set (m)", fontsize=12)
        axes[1].set_title("Few-Shot Learning Curve: MAE", fontsize=14)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xlim(-1, max(n_samples) + 2)

        # Annotate points
        for x, y in zip(n_samples, mae_scores):
            axes[1].annotate(
                f"{y:.1f}m",
                xy=(x, y),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
                fontsize=9,
            )

        plt.tight_layout()
        plt.savefig(self.figures_dir / "f4_learning_curves.png", dpi=300)
        plt.close()
        print(" Saved: f4_learning_curves.png")

        # Bar chart comparison
        fig, ax = plt.subplots(figsize=(10, 6))

        labels = ["Baseline\n(LOO)"] + [
            f"{k.split('_')[0]}-shot" for k in few_shot_results.keys()
        ]
        improvements = [0] + [
            v["improvement_over_baseline"] for v in few_shot_results.values()
        ]

        colors = ["red"] + [
            "green" if imp > 3 else "orange" for imp in improvements[1:]
        ]

        ax.bar(labels, improvements, color=colors, alpha=0.7, edgecolor="black")
        ax.axhline(y=0, color="black", linestyle="-", lw=1)
        ax.set_ylabel("Improvement in R² over Baseline", fontsize=12)
        ax.set_title(
            "Domain Adaptation: Improvement Over Baseline (R² = -3.13)", fontsize=14
        )
        ax.grid(True, alpha=0.3, axis="y")

        # Annotate bars
        for i, (label, imp) in enumerate(zip(labels, improvements)):
            ax.text(
                i,
                imp + 0.1,
                f"+{imp:.2f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

        plt.tight_layout()
        plt.savefig(self.figures_dir / "f4_improvement_comparison.png", dpi=300)
        plt.close()
        print(" Saved: f4_improvement_comparison.png")

    def save_domain_adaptation_report(
        self, baseline_metrics: Dict, few_shot_results: Dict
    ):
        """Save domain adaptation results report"""
        report = {
            "target_flight": "F4 (18Feb25)",
            "baseline_loo_r2": baseline_metrics["r2"],
            "domain_shift_description": "F4 mean CBH = 0.249 km, Other flights mean CBH = 0.846 km (2.2 std diff)",
            "few_shot_experiments": few_shot_results,
            "conclusion": self._generate_conclusion(baseline_metrics, few_shot_results),
            "timestamp": datetime.now().isoformat(),
        }

        report_path = self.reports_dir / "domain_adaptation_results.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"\n{'=' * 80}")
        print("Domain Adaptation Report Summary")
        print(f"{'=' * 80}")
        print(f"Baseline LOO R²: {baseline_metrics['r2']:.4f}")
        print(f"\nFew-Shot Results:")
        for key, value in few_shot_results.items():
            print(
                f"  {key}: R² = {value['r2']:.4f}, Improvement = +{value['improvement_over_baseline']:.4f}"
            )
        print(f"\nConclusion: {report['conclusion']}")
        print(f"\n Report saved: {report_path}")

        return report

    def _generate_conclusion(
        self, baseline_metrics: Dict, few_shot_results: Dict
    ) -> str:
        """Generate conclusion based on results"""
        max_improvement = max(
            [v["improvement_over_baseline"] for v in few_shot_results.values()],
            default=0,
        )
        max_r2 = max([v["r2"] for v in few_shot_results.values()], default=-999)

        if max_r2 > 0.5:
            return "Few-shot adaptation successful: R² > 0.5 achieved"
        elif max_improvement > 2.0:
            return "Moderate improvement: Significant R² gain but still below target"
        else:
            return "Insufficient improvement: Domain shift too severe for few-shot adaptation alone"


# ==============================================================================
# Main Execution
# ==============================================================================


def main():
    """Main execution function"""

    # Paths - use relative path from module location
    project_root = Path(__file__).resolve().parent.parent.parent
    integrated_features_path = str(
        project_root / "outputs/preprocessed_data/Integrated_Features.hdf5"
    )
    output_dir = project_root
    production_checkpoint = output_dir / "models" / "final_production_model.pth"

    # Fallback to fold checkpoint if production not available
    if not production_checkpoint.exists():
        production_checkpoint = output_dir / "checkpoints" / "fold_0_model.pth"

    print(f"\n{'=' * 80}")
    print("Sprint 6 - Phase 2, Task 2.2: Domain Adaptation for Flight F4")
    print(f"{'=' * 80}")
    print(f"Project Root: {project_root}")
    print(f"Integrated Features: {integrated_features_path}")
    print(f"Output Directory: {output_dir}")
    print(f"Production Checkpoint: {production_checkpoint}")

    # Initialize adapter
    adapter = DomainAdaptationF4(output_dir=output_dir)

    # Load data
    image_dataset, cbh_values, flight_ids = adapter.load_data(integrated_features_path)

    # Load production model
    base_model = adapter.load_production_model(production_checkpoint)

    # Get baseline LOO performance on F4
    baseline_metrics = adapter.get_f4_baseline(
        base_model, image_dataset, cbh_values, flight_ids, f4_flight_id=4
    )

    # Run few-shot experiments
    few_shot_results = adapter.evaluate_few_shot_experiments(
        base_model, image_dataset, cbh_values, flight_ids, f4_flight_id=4
    )

    # Create visualizations
    adapter.create_visualizations(baseline_metrics, few_shot_results)

    # Save report
    adapter.save_domain_adaptation_report(baseline_metrics, few_shot_results)

    print(f"\n{'=' * 80}")
    print(" Task 2.2 Complete: Domain Adaptation for Flight F4")
    print(f"{'=' * 80}")
    print(f"All outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
