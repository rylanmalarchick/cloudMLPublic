#!/usr/bin/env python3
"""
Visualize MAE embeddings to understand what they encode.

This script:
1. Extracts MAE embeddings for all labeled samples
2. Reduces dimensionality with t-SNE and UMAP
3. Creates scatter plots colored by:
   - CBH (target variable)
   - Flight ID (source)
   - SZA (solar zenith angle)
   - SAA (solar azimuth angle)
4. Computes per-dimension correlations with CBH
5. Identifies which dimensions (if any) encode CBH information

Usage:
    python scripts/visualize_embeddings.py \
        --config configs/ssl_finetune_cbh.yaml \
        --encoder outputs/mae_pretrain/mae_encoder_pretrained.pth \
        --output outputs/embedding_analysis
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import yaml
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
from datetime import datetime

try:
    import umap

    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("UMAP not available. Install with: pip install umap-learn")

from src.hdf5_dataset import HDF5CloudDataset
from src.mae_model import MaskedAutoencoder


class EmbeddingVisualizer:
    def __init__(self, config_path, encoder_path, output_dir):
        self.config_path = config_path
        self.encoder_path = encoder_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load config
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

    def load_encoder(self):
        """Load pre-trained MAE encoder."""
        print("\nLoading MAE encoder...")

        # Initialize model with default decoder params (we only need encoder)
        encoder = MaskedAutoencoder(
            img_width=self.config["model"]["img_width"],
            patch_size=self.config["model"]["patch_size"],
            embed_dim=self.config["model"]["embed_dim"],
            depth=self.config["model"]["depth"],
            num_heads=self.config["model"]["num_heads"],
            decoder_embed_dim=96,  # Default, not used for inference
            decoder_depth=2,
            decoder_num_heads=3,
            mlp_ratio=self.config["model"]["mlp_ratio"],
        ).to(self.device)

        # Load weights
        checkpoint = torch.load(
            self.encoder_path, map_location=self.device, weights_only=False
        )
        # Handle both direct state dict and wrapped dict
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        # Load only encoder weights (ignore decoder)
        encoder_state_dict = {
            k: v
            for k, v in state_dict.items()
            if k.startswith("encoder.") or not k.startswith("decoder.")
        }
        # Remove 'encoder.' prefix if present
        encoder_state_dict = {
            k.replace("encoder.", ""): v for k, v in encoder_state_dict.items()
        }
        encoder.encoder.load_state_dict(encoder_state_dict, strict=False)
        encoder.eval()

        print(f"✓ Encoder loaded")
        print(f"  Embedding dimension: {self.config['model']['embed_dim']}")

        return encoder

    def load_dataset(self):
        """Load labeled dataset."""
        print("\nLoading labeled dataset...")

        dataset = HDF5CloudDataset(
            flight_configs=self.config["data"]["flights"],
            swath_slice=self.config["data"]["swath_slice"],
            temporal_frames=self.config["data"]["temporal_frames"],
            filter_type=self.config["data"]["filter_type"],
            cbh_min=self.config["data"]["cbh_min"],
            cbh_max=self.config["data"]["cbh_max"],
            flat_field_correction=self.config["data"]["flat_field_correction"],
            clahe_clip_limit=self.config["data"]["clahe_clip_limit"],
            zscore_normalize=self.config["data"]["zscore_normalize"],
            angles_mode=self.config["data"]["angles_mode"],
            augment=False,
        )

        print(f"✓ Dataset loaded: {len(dataset)} samples")
        return dataset

    def extract_embeddings(self, encoder, dataset, batch_size=64):
        """Extract MAE embeddings and metadata for all samples."""
        print("\nExtracting embeddings and metadata...")

        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=4
        )

        embeddings_list = []
        sza_list = []
        saa_list = []
        cbh_list = []
        flight_list = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting"):
                img_stack, sza, saa, y_scaled, global_idx, local_idx = batch
                img_stack = img_stack.to(self.device)

                # Handle temporal dimension: (B, C, T, W) -> (B, C, W)
                # Take middle frame if temporal_frames > 1
                if img_stack.dim() == 4:
                    temporal_frames = img_stack.shape[2]
                    mid_frame = temporal_frames // 2
                    img_stack = img_stack[:, :, mid_frame, :]

                # Get encoder embeddings (CLS token)
                cls_token = encoder.forward_encoder(img_stack)
                embeddings_list.append(cls_token.cpu().numpy())

                # Store metadata
                sza_list.append(sza.numpy())
                saa_list.append(saa.numpy())

                # Unscale CBH to km
                y_unscaled = dataset.y_scaler.inverse_transform(
                    y_scaled.numpy().reshape(-1, 1)
                ).flatten()
                cbh_list.append(y_unscaled)

                # Get flight IDs
                for gidx in global_idx:
                    flight_idx, _ = dataset.global_to_local[int(gidx)]
                    flight_list.append(flight_idx)

        # Concatenate all batches
        embeddings = np.vstack(embeddings_list)
        sza = np.concatenate(sza_list).flatten()
        saa = np.concatenate(saa_list).flatten()
        cbh = np.concatenate(cbh_list).flatten()
        flight_ids = np.array(flight_list)

        # Get flight names
        flight_names = [dataset.flight_data[fid]["name"] for fid in flight_ids]

        print(f"✓ Extracted embeddings: {embeddings.shape}")
        print(f"  CBH range: [{cbh.min():.3f}, {cbh.max():.3f}] km")
        print(f"  SZA range: [{sza.min():.1f}, {sza.max():.1f}]°")
        print(f"  SAA range: [{saa.min():.1f}, {saa.max():.1f}]°")
        print(f"  Flights: {np.unique(flight_names)}")

        return {
            "embeddings": embeddings,
            "cbh": cbh,
            "sza": sza,
            "saa": saa,
            "flight_ids": flight_ids,
            "flight_names": flight_names,
        }

    def compute_correlations(self, data):
        """Compute per-dimension correlations with CBH and other variables."""
        print("\nComputing per-dimension correlations...")

        embeddings = data["embeddings"]
        cbh = data["cbh"]
        sza = data["sza"]
        saa = data["saa"]
        flight_ids = data["flight_ids"]

        n_dims = embeddings.shape[1]

        correlations = {
            "cbh": [],
            "sza": [],
            "saa": [],
            "flight": [],
        }

        for dim in range(n_dims):
            emb_dim = embeddings[:, dim]

            # Correlation with CBH
            corr_cbh = np.corrcoef(emb_dim, cbh)[0, 1]
            correlations["cbh"].append(corr_cbh)

            # Correlation with SZA
            corr_sza = np.corrcoef(emb_dim, sza)[0, 1]
            correlations["sza"].append(corr_sza)

            # Correlation with SAA
            corr_saa = np.corrcoef(emb_dim, saa)[0, 1]
            correlations["saa"].append(corr_saa)

            # Correlation with flight ID
            corr_flight = np.corrcoef(emb_dim, flight_ids)[0, 1]
            correlations["flight"].append(corr_flight)

        correlations = {k: np.array(v) for k, v in correlations.items()}

        # Print summary statistics
        print(f"\nCorrelation Summary (across {n_dims} dimensions):")
        for var_name, corrs in correlations.items():
            abs_corrs = np.abs(corrs)
            print(f"  {var_name:>10}:")
            print(f"    Max |r| = {abs_corrs.max():.4f}")
            print(f"    Mean |r| = {abs_corrs.mean():.4f}")
            print(f"    # dims with |r| > 0.3: {(abs_corrs > 0.3).sum()}")
            print(f"    # dims with |r| > 0.5: {(abs_corrs > 0.5).sum()}")

        return correlations

    def reduce_dimensionality(self, embeddings):
        """Reduce embeddings to 2D using PCA, t-SNE, and UMAP."""
        print("\nReducing dimensionality...")

        reductions = {}

        # PCA
        print("  Computing PCA...")
        pca = PCA(n_components=2, random_state=42)
        reductions["pca"] = pca.fit_transform(embeddings)
        print(f"    Explained variance: {pca.explained_variance_ratio_.sum():.1%}")

        # t-SNE
        print("  Computing t-SNE...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        reductions["tsne"] = tsne.fit_transform(embeddings)

        # UMAP
        if UMAP_AVAILABLE:
            print("  Computing UMAP...")
            umap_reducer = umap.UMAP(n_components=2, random_state=42)
            reductions["umap"] = umap_reducer.fit_transform(embeddings)
        else:
            print("  UMAP not available, skipping...")

        return reductions

    def plot_embeddings(self, reductions, data):
        """Create scatter plots of 2D embeddings colored by different variables."""
        print("\nCreating visualizations...")

        cbh = data["cbh"]
        sza = data["sza"]
        saa = data["saa"]
        flight_names = np.array(data["flight_names"])

        # Variables to color by
        color_vars = {
            "CBH (km)": cbh,
            "SZA (°)": sza,
            "SAA (°)": saa,
            "Flight": flight_names,
        }

        for method_name, coords in reductions.items():
            print(f"  Plotting {method_name.upper()}...")

            n_vars = len(color_vars)
            fig, axes = plt.subplots(1, n_vars, figsize=(6 * n_vars, 5))

            if n_vars == 1:
                axes = [axes]

            for ax, (var_name, var_data) in zip(axes, color_vars.items()):
                if var_name == "Flight":
                    # Categorical coloring for flights
                    unique_flights = np.unique(flight_names)
                    colors = plt.cm.tab10(np.arange(len(unique_flights)))

                    for i, flight in enumerate(unique_flights):
                        mask = flight_names == flight
                        ax.scatter(
                            coords[mask, 0],
                            coords[mask, 1],
                            c=[colors[i]],
                            label=flight,
                            alpha=0.6,
                            s=20,
                        )
                    ax.legend(loc="best", fontsize=8)
                else:
                    # Continuous coloring
                    scatter = ax.scatter(
                        coords[:, 0],
                        coords[:, 1],
                        c=var_data,
                        cmap="viridis",
                        alpha=0.6,
                        s=20,
                    )
                    plt.colorbar(scatter, ax=ax, label=var_name)

                ax.set_xlabel(f"{method_name.upper()} 1")
                ax.set_ylabel(f"{method_name.upper()} 2")
                ax.set_title(f"Colored by {var_name}")
                ax.grid(True, alpha=0.3)

            plt.tight_layout()
            output_path = self.output_dir / f"embeddings_{method_name}.png"
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"    Saved: {output_path}")

    def plot_correlation_heatmap(self, correlations):
        """Plot heatmap of per-dimension correlations."""
        print("\nCreating correlation heatmap...")

        # Convert to matrix
        n_dims = len(correlations["cbh"])
        corr_matrix = np.column_stack(
            [
                correlations["cbh"],
                correlations["sza"],
                correlations["saa"],
                correlations["flight"],
            ]
        ).T

        fig, ax = plt.subplots(figsize=(16, 4))
        im = ax.imshow(corr_matrix, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)

        ax.set_yticks(range(4))
        ax.set_yticklabels(["CBH", "SZA", "SAA", "Flight"])
        ax.set_xlabel("Embedding Dimension")
        ax.set_title("Per-Dimension Correlations with Variables")

        plt.colorbar(im, ax=ax, label="Pearson r")
        plt.tight_layout()

        output_path = self.output_dir / "correlation_heatmap.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {output_path}")

    def plot_top_dimensions(self, data, correlations, top_k=10):
        """Plot the top-k dimensions most correlated with CBH."""
        print(f"\nPlotting top {top_k} CBH-correlated dimensions...")

        embeddings = data["embeddings"]
        cbh = data["cbh"]

        # Get top dimensions by absolute correlation
        abs_corrs = np.abs(correlations["cbh"])
        top_indices = np.argsort(abs_corrs)[-top_k:][::-1]

        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.flatten()

        for i, dim_idx in enumerate(top_indices):
            ax = axes[i]
            corr_val = correlations["cbh"][dim_idx]

            ax.scatter(embeddings[:, dim_idx], cbh, alpha=0.3, s=10)
            ax.set_xlabel(f"Dimension {dim_idx}")
            ax.set_ylabel("CBH (km)")
            ax.set_title(f"Dim {dim_idx}: r = {corr_val:.4f}")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = self.output_dir / f"top_{top_k}_cbh_dimensions.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {output_path}")

    def save_results(self, correlations, data):
        """Save numerical results to JSON."""
        print("\nSaving numerical results...")

        results = {
            "timestamp": datetime.now().isoformat(),
            "n_samples": len(data["cbh"]),
            "n_dimensions": len(correlations["cbh"]),
            "correlation_stats": {},
        }

        for var_name, corrs in correlations.items():
            abs_corrs = np.abs(corrs)
            results["correlation_stats"][var_name] = {
                "max_abs": float(abs_corrs.max()),
                "mean_abs": float(abs_corrs.mean()),
                "n_strong": int((abs_corrs > 0.3).sum()),
                "n_very_strong": int((abs_corrs > 0.5).sum()),
                "top_10_dims": [int(x) for x in np.argsort(abs_corrs)[-10:][::-1]],
                "top_10_corrs": [float(x) for x in np.sort(abs_corrs)[-10:][::-1]],
            }

        output_path = self.output_dir / "embedding_analysis.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Saved: {output_path}")

    def run_full_analysis(self):
        """Run complete embedding analysis pipeline."""
        print("=" * 80)
        print("MAE EMBEDDING ANALYSIS")
        print("=" * 80)

        # Load model and data
        encoder = self.load_encoder()
        dataset = self.load_dataset()

        # Extract embeddings
        data = self.extract_embeddings(encoder, dataset)

        # Compute correlations
        correlations = self.compute_correlations(data)

        # Dimensionality reduction
        reductions = self.reduce_dimensionality(data["embeddings"])

        # Create visualizations
        self.plot_embeddings(reductions, data)
        self.plot_correlation_heatmap(correlations)
        self.plot_top_dimensions(data, correlations, top_k=10)

        # Save results
        self.save_results(correlations, data)

        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)
        print(f"Results saved to: {self.output_dir}")
        print("\nKey Findings:")
        print(f"  Max |r| with CBH: {np.abs(correlations['cbh']).max():.4f}")
        print(f"  Dims with |r| > 0.3: {(np.abs(correlations['cbh']) > 0.3).sum()}")
        print(f"  Dims with |r| > 0.5: {(np.abs(correlations['cbh']) > 0.5).sum()}")

        if np.abs(correlations["cbh"]).max() < 0.3:
            print("\n⚠️  WARNING: No embedding dimensions strongly correlate with CBH!")
            print("   This explains why MAE embeddings don't help with prediction.")


def main():
    parser = argparse.ArgumentParser(description="Visualize MAE embeddings")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/ssl_finetune_cbh.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="outputs/mae_pretrain/mae_encoder_pretrained.pth",
        help="Path to pre-trained encoder",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: outputs/embedding_analysis/TIMESTAMP)",
    )

    args = parser.parse_args()

    # Set default output directory with timestamp
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"outputs/embedding_analysis/{timestamp}"

    # Run analysis
    visualizer = EmbeddingVisualizer(args.config, args.encoder, args.output)
    visualizer.run_full_analysis()


if __name__ == "__main__":
    main()
