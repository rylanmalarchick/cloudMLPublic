#!/usr/bin/env python3
"""
Phase 1: Data Extraction Verification Script

This script verifies the integrity and statistics of extracted SSL data.

Usage:
    python scripts/verify_extraction.py --data-dir data_ssl/images
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import yaml
import matplotlib.pyplot as plt


def verify_hdf5_dataset(file_path: Path, split_name: str):
    """Verify a single HDF5 dataset file."""
    print(f"\n{'=' * 80}")
    print(f"Verifying {split_name.upper()} dataset: {file_path.name}")
    print("=" * 80)

    if not file_path.exists():
        print(f" ERROR: File not found!")
        return False

    try:
        with h5py.File(file_path, "r") as f:
            # Check required datasets
            if "images" not in f:
                print(" ERROR: 'images' dataset not found!")
                return False
            if "metadata" not in f:
                print(" ERROR: 'metadata' dataset not found!")
                return False

            images = f["images"]
            metadata = f["metadata"]

            # Basic info
            print(f"\n Dataset Information:")
            print(f"  Images shape: {images.shape}")
            print(f"  Images dtype: {images.dtype}")
            print(f"  Metadata shape: {metadata.shape}")
            print(f"  Metadata dtype: {metadata.dtype}")

            # Attributes
            if "n_samples" in f.attrs:
                print(f"  Number of samples: {f.attrs['n_samples']:,}")
            if "image_shape" in f.attrs:
                print(f"  Image shape: {tuple(f.attrs['image_shape'])}")

            # Metadata columns
            if "columns" in metadata.attrs:
                columns = metadata.attrs["columns"]
                print(f"  Metadata columns: {columns}")

            # File size
            file_size_mb = file_path.stat().st_size / (1024**2)
            print(f"  File size: {file_size_mb:.1f} MB")

            # Sample statistics
            print(f"\n Image Statistics:")

            # Load a sample for quick stats (first 1000 images)
            sample_size = min(1000, len(images))
            sample = images[:sample_size]

            print(f"  Computing stats on {sample_size:,} sample images...")
            print(f"  Min value: {np.min(sample):.2f}")
            print(f"  Max value: {np.max(sample):.2f}")
            print(f"  Mean value: {np.mean(sample):.2f}")
            print(f"  Std dev: {np.std(sample):.2f}")

            # Check for NaN/Inf
            n_nan = np.sum(np.isnan(sample))
            n_inf = np.sum(np.isinf(sample))

            if n_nan > 0:
                print(f"    WARNING: {n_nan} NaN values found in sample!")
            else:
                print(f"   No NaN values")

            if n_inf > 0:
                print(f"    WARNING: {n_inf} Inf values found in sample!")
            else:
                print(f"   No Inf values")

            # Metadata statistics
            print(f"\n Metadata Statistics:")

            meta_sample = metadata[:]
            if "columns" in metadata.attrs:
                columns = metadata.attrs["columns"]
                for i, col_name in enumerate(columns):
                    col_data = meta_sample[:, i]
                    print(f"  {col_name}:")
                    print(f"    Min: {np.min(col_data):.2f}")
                    print(f"    Max: {np.max(col_data):.2f}")
                    print(f"    Mean: {np.mean(col_data):.2f}")

                    if col_name == "flight_idx":
                        unique_flights = np.unique(col_data.astype(int))
                        print(f"    Unique flights: {list(unique_flights)}")

                        # Count samples per flight
                        print(f"    Samples per flight:")
                        for flight_idx in unique_flights:
                            count = np.sum(col_data == flight_idx)
                            print(f"      Flight {int(flight_idx)}: {count:,}")

            print(f"\n {split_name.upper()} dataset verification passed!")
            return True

    except Exception as e:
        print(f"\n ERROR during verification: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


def verify_npz_dataset(file_path: Path, split_name: str):
    """Verify a single NPZ dataset file."""
    print(f"\n{'=' * 80}")
    print(f"Verifying {split_name.upper()} dataset: {file_path.name}")
    print("=" * 80)

    if not file_path.exists():
        print(f" ERROR: File not found!")
        return False

    try:
        data = np.load(file_path)

        # Check required arrays
        if "images" not in data:
            print(" ERROR: 'images' array not found!")
            return False
        if "metadata" not in data:
            print(" ERROR: 'metadata' array not found!")
            return False

        images = data["images"]
        metadata = data["metadata"]

        # Basic info
        print(f"\n Dataset Information:")
        print(f"  Images shape: {images.shape}")
        print(f"  Images dtype: {images.dtype}")
        print(f"  Metadata shape: {metadata.shape}")
        print(f"  Metadata dtype: {metadata.dtype}")

        # File size
        file_size_mb = file_path.stat().st_size / (1024**2)
        print(f"  File size: {file_size_mb:.1f} MB")

        # Sample statistics
        print(f"\n Image Statistics:")
        print(f"  Min value: {np.min(images):.2f}")
        print(f"  Max value: {np.max(images):.2f}")
        print(f"  Mean value: {np.mean(images):.2f}")
        print(f"  Std dev: {np.std(images):.2f}")

        # Check for NaN/Inf
        n_nan = np.sum(np.isnan(images))
        n_inf = np.sum(np.isinf(images))

        if n_nan > 0:
            print(f"    WARNING: {n_nan} NaN values found!")
        else:
            print(f"   No NaN values")

        if n_inf > 0:
            print(f"    WARNING: {n_inf} Inf values found!")
        else:
            print(f"   No Inf values")

        print(f"\n {split_name.upper()} dataset verification passed!")
        return True

    except Exception as e:
        print(f"\n ERROR during verification: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


def load_statistics(stats_path: Path):
    """Load and display extraction statistics."""
    print(f"\n{'=' * 80}")
    print("Extraction Statistics")
    print("=" * 80)

    if not stats_path.exists():
        print(f"  WARNING: Statistics file not found: {stats_path}")
        return

    with open(stats_path, "r") as f:
        stats = yaml.safe_load(f)

    print(f"\n Summary:")
    print(f"  Total images: {stats.get('total_images', 'N/A'):,}")
    print(f"  Train images: {stats.get('train_images', 'N/A'):,}")
    print(f"  Val images: {stats.get('val_images', 'N/A'):,}")
    print(f"  Flights processed: {stats.get('flights_processed', 'N/A')}")
    print(f"  Flights failed: {stats.get('flights_failed', 0)}")

    if "flights" in stats:
        print(f"\n  Processed flights:")
        for flight in stats["flights"]:
            print(f"    - {flight}")

    if "config" in stats:
        print(f"\n  Configuration:")
        config = stats["config"]
        for key, value in config.items():
            print(f"  {key}: {value}")


def plot_sample_images(data_dir: Path, format_type: str, n_samples: int = 9):
    """Plot sample images from the dataset."""
    print(f"\n{'=' * 80}")
    print(f"Plotting {n_samples} sample images...")
    print("=" * 80)

    train_file = data_dir / f"train.{'h5' if format_type == 'hdf5' else 'npz'}"

    if not train_file.exists():
        print(f"  Cannot plot samples: {train_file} not found")
        return

    try:
        # Load sample images
        if format_type == "hdf5":
            with h5py.File(train_file, "r") as f:
                # Sample random indices
                n_total = f["images"].shape[0]
                indices = np.random.choice(
                    n_total, size=min(n_samples, n_total), replace=False
                )
                # Sort indices for HDF5 fancy indexing requirement
                indices = np.sort(indices)
                images = f["images"][indices]
        else:
            data = np.load(train_file)
            n_total = data["images"].shape[0]
            indices = np.random.choice(
                n_total, size=min(n_samples, n_total), replace=False
            )
            # Sort indices for consistency (not required for NPZ but good practice)
            indices = np.sort(indices)
            images = data["images"][indices]

        # Create plot
        n_rows = int(np.sqrt(n_samples))
        n_cols = int(np.ceil(n_samples / n_rows))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 12))
        axes = axes.flatten() if n_samples > 1 else [axes]

        for i, (ax, img) in enumerate(zip(axes, images)):
            ax.imshow(img, cmap="gray")
            ax.set_title(f"Sample {indices[i]}")
            ax.axis("off")

        # Hide unused subplots
        for i in range(len(images), len(axes)):
            axes[i].axis("off")

        plt.tight_layout()

        # Save plot
        output_path = data_dir / "sample_images.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f" Sample images saved to: {output_path}")

        # plt.show()  # Uncomment to display interactively

    except Exception as e:
        print(f" ERROR plotting samples: {str(e)}")
        import traceback

        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description="Verify extracted SSL dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing extracted datasets",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["hdf5", "npz"],
        default="hdf5",
        help="Dataset format (default: hdf5)",
    )
    parser.add_argument(
        "--plot-samples", action="store_true", help="Generate plot of sample images"
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=9,
        help="Number of sample images to plot (default: 9)",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    if not data_dir.exists():
        print(f" ERROR: Data directory not found: {data_dir}")
        sys.exit(1)

    print("\n" + "=" * 80)
    print("SSL DATA EXTRACTION VERIFICATION")
    print("=" * 80)
    print(f"Data directory: {data_dir}")
    print(f"Format: {args.format}")

    # Verify statistics file
    stats_path = data_dir / "extraction_stats.yaml"
    load_statistics(stats_path)

    # Verify datasets
    file_ext = "h5" if args.format == "hdf5" else "npz"
    train_file = data_dir / f"train.{file_ext}"
    val_file = data_dir / f"val.{file_ext}"

    verify_func = verify_hdf5_dataset if args.format == "hdf5" else verify_npz_dataset

    train_ok = verify_func(train_file, "train")
    val_ok = verify_func(val_file, "val")

    # Plot samples if requested
    if args.plot_samples:
        plot_sample_images(data_dir, args.format, args.n_samples)

    # Final summary
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)

    if train_ok and val_ok:
        print(" All verifications passed!")
        print("\nNext steps:")
        print("  1. Review extraction_stats.yaml for detailed statistics")
        print("  2. Proceed to Phase 2: SSL pre-training")
        print(
            "     python scripts/pretrain_mae.py --config configs/ssl_pretrain_mae.yaml"
        )
        sys.exit(0)
    else:
        print(" Some verifications failed. Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
