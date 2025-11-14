#!/usr/bin/env python3
"""
Phase 1: Data Extraction for Self-Supervised Learning

This script extracts ALL IR images from the flight HDF5 files (labeled + unlabeled)
and saves them in an efficient format for SSL pre-training.

Usage:
    python scripts/extract_all_images.py --config configs/ssl_extract.yaml

    OR with command-line overrides:

    python scripts/extract_all_images.py \
        --data-dir /path/to/data \
        --output-dir data_ssl/images \
        --format hdf5 \
        --train-split 0.95
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import yaml

import h5py
import numpy as np
from tqdm import tqdm
import gc


def linear_vignetting_correction(image, flat_ref):
    """
    Apply linear vignetting correction using a flat reference.
    Copied from data_preprocessing.py for standalone use.
    """
    median_flat = np.median(flat_ref)
    if median_flat == 0:
        return image
    correction_map = median_flat / (flat_ref + 1e-8)
    corrected = image * correction_map
    return corrected


class ImageExtractor:
    """Extracts and caches all IR images from flight HDF5 files."""

    def __init__(
        self,
        flight_configs: List[Dict],
        output_dir: str,
        swath_slice: Tuple[int, int] = (40, 480),
        format: str = "hdf5",
        train_split: float = 0.95,
        apply_vignetting_correction: bool = True,
        random_seed: int = 42,
    ):
        """
        Initialize the image extractor.

        Args:
            flight_configs: List of flight configuration dicts with iFileName, cFileName, nFileName
            output_dir: Directory to save extracted images
            swath_slice: Tuple of (start, end) indices for swath cropping
            format: Output format ('hdf5' or 'npz')
            train_split: Fraction of data to use for training (rest is validation)
            apply_vignetting_correction: Whether to apply flat-field correction
            random_seed: Random seed for reproducibility
        """
        self.flight_configs = flight_configs
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.swath_start, self.swath_end = swath_slice
        self.format = format
        self.train_split = train_split
        self.apply_vignetting = apply_vignetting_correction
        self.random_seed = random_seed

        np.random.seed(random_seed)

        # Statistics for reporting
        self.stats = {
            "total_images": 0,
            "train_images": 0,
            "val_images": 0,
            "flights_processed": 0,
            "flights_failed": 0,
        }

    def extract_all(self):
        """Main extraction pipeline."""
        print("\n" + "=" * 80)
        print("PHASE 1: DATA EXTRACTION FOR SELF-SUPERVISED LEARNING")
        print("=" * 80)
        print(f"\nConfiguration:")
        print(f"  Output directory: {self.output_dir}")
        print(f"  Output format: {self.format}")
        print(f"  Swath slice: [{self.swath_start}:{self.swath_end}]")
        print(f"  Train/Val split: {self.train_split:.1%} / {1 - self.train_split:.1%}")
        print(f"  Vignetting correction: {self.apply_vignetting}")
        print(f"  Number of flights: {len(self.flight_configs)}")
        print()

        # Extract images from all flights
        all_images = []
        all_metadata = []

        for flight_config in self.flight_configs:
            try:
                images, metadata = self._extract_flight(flight_config)
                all_images.append(images)
                all_metadata.append(metadata)
                self.stats["flights_processed"] += 1
            except Exception as e:
                print(
                    f"\n   ERROR: Failed to process flight {flight_config.get('name', 'unknown')}"
                )
                print(f"     {str(e)}")
                self.stats["flights_failed"] += 1
                continue

        if not all_images:
            raise RuntimeError("No flights were successfully processed!")

        # Concatenate all flights
        print("\n" + "-" * 80)
        print("Concatenating all flights...")
        all_images_array = np.concatenate(all_images, axis=0)
        all_metadata_array = np.concatenate(all_metadata, axis=0)

        self.stats["total_images"] = len(all_images_array)
        print(f"Total images extracted: {self.stats['total_images']:,}")

        # Create train/val split
        print("\nCreating train/validation split...")
        train_images, val_images, train_meta, val_meta = self._split_data(
            all_images_array, all_metadata_array
        )

        self.stats["train_images"] = len(train_images)
        self.stats["val_images"] = len(val_images)

        # Save datasets
        print("\nSaving datasets...")
        self._save_dataset(train_images, train_meta, "train")
        self._save_dataset(val_images, val_meta, "val")

        # Save statistics
        self._save_statistics()

        # Print summary
        self._print_summary()

        print("\n Phase 1 extraction complete!\n")

    def _extract_flight(self, config: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract all images from a single flight.

        Returns:
            images: Array of shape (N, H, W) with extracted images
            metadata: Array of shape (N, M) with flight_idx, frame_idx, SZA, SAA
        """
        flight_name = config.get("name", "unknown")
        print(f"\nProcessing flight: {flight_name}")
        print(f"  IR file: {Path(config['iFileName']).name}")

        # Open HDF5 files
        with h5py.File(config["iFileName"], "r") as hf:
            ds = hf["Product/Signal"]
            total_frames = ds.shape[0]
            print(f"  Total frames: {total_frames:,}")

            # Read first 100 frames for flat-field reference
            if self.apply_vignetting:
                block = ds[
                    0 : min(total_frames, 100), self.swath_start : self.swath_end, :
                ].astype(np.float32)
                flat_ref = np.mean(block, axis=0)
                corrected_flat = linear_vignetting_correction(flat_ref, flat_ref)
                median_cf = np.median(corrected_flat)
                del block
                gc.collect()
                print(f"  Flat-field reference computed (median: {median_cf:.2f})")
            else:
                flat_ref = None

            # Read navigation data for metadata
            with h5py.File(config["nFileName"], "r") as nf:
                SZA_raw = nf["nav/solarZenith"][:total_frames].reshape(-1, 1)
                SZA = np.nan_to_num(SZA_raw, nan=np.nanmean(SZA_raw))
                SAA_raw = nf["nav/sunAzGrd"][:total_frames].reshape(-1, 1)
                SAA = np.mod(np.nan_to_num(SAA_raw, nan=0.0), 360.0)
                print(
                    f"  Navigation data loaded (SZA range: {SZA.min():.1f}-{SZA.max():.1f}°)"
                )

            # Extract all images in chunks to manage memory
            chunk_size = 1000
            images_list = []

            print(f"  Extracting images in chunks of {chunk_size}...")
            for start_idx in tqdm(
                range(0, total_frames, chunk_size), desc=f"  {flight_name}"
            ):
                end_idx = min(start_idx + chunk_size, total_frames)

                # Read chunk
                chunk = ds[
                    start_idx:end_idx, self.swath_start : self.swath_end, :
                ].astype(np.float32)

                # Apply vignetting correction if enabled
                if self.apply_vignetting and flat_ref is not None:
                    for i in range(len(chunk)):
                        chunk[i] = linear_vignetting_correction(chunk[i], flat_ref)

                # Average across the 3 pixels dimension (nadir + 2 views)
                chunk_avg = np.mean(chunk, axis=2)  # Shape: (chunk_size, H, W)

                images_list.append(chunk_avg)

            # Concatenate all chunks
            images = np.concatenate(images_list, axis=0)
            print(f"   Extracted {len(images):,} images with shape {images.shape[1:]}")

        # Create metadata array: [flight_idx, frame_idx, SZA, SAA]
        flight_idx = config.get("flight_idx", 0)
        frame_indices = np.arange(total_frames).reshape(-1, 1)
        metadata = np.hstack(
            [
                np.full((total_frames, 1), flight_idx),
                frame_indices,
                SZA,
                SAA,
            ]
        )

        return images, metadata

    def _split_data(
        self, images: np.ndarray, metadata: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into train and validation sets.

        Uses stratified sampling by flight to ensure each flight is represented.
        """
        # Get unique flight indices
        flight_indices = metadata[:, 0].astype(int)
        unique_flights = np.unique(flight_indices)

        train_mask = np.zeros(len(images), dtype=bool)

        # For each flight, randomly sample train_split fraction
        for flight_idx in unique_flights:
            flight_mask = flight_indices == flight_idx
            flight_count = flight_mask.sum()

            # Get indices for this flight
            flight_positions = np.where(flight_mask)[0]

            # Shuffle and split
            np.random.shuffle(flight_positions)
            n_train = int(flight_count * self.train_split)
            train_positions = flight_positions[:n_train]

            train_mask[train_positions] = True

        # Split data
        train_images = images[train_mask]
        val_images = images[~train_mask]
        train_meta = metadata[train_mask]
        val_meta = metadata[~train_mask]

        print(
            f"  Train set: {len(train_images):,} images ({len(train_images) / len(images):.1%})"
        )
        print(
            f"  Val set:   {len(val_images):,} images ({len(val_images) / len(images):.1%})"
        )

        return train_images, val_images, train_meta, val_meta

    def _save_dataset(self, images: np.ndarray, metadata: np.ndarray, split: str):
        """Save a dataset split to disk."""
        if self.format == "hdf5":
            self._save_hdf5(images, metadata, split)
        elif self.format == "npz":
            self._save_npz(images, metadata, split)
        else:
            raise ValueError(f"Unsupported format: {self.format}")

    def _save_hdf5(self, images: np.ndarray, metadata: np.ndarray, split: str):
        """Save dataset in HDF5 format (recommended for large datasets)."""
        output_path = self.output_dir / f"{split}.h5"

        with h5py.File(output_path, "w") as f:
            # Store images with compression
            f.create_dataset(
                "images",
                data=images,
                dtype=np.float32,
                compression="gzip",
                compression_opts=4,
            )

            # Store metadata
            f.create_dataset(
                "metadata",
                data=metadata,
                dtype=np.float32,
            )

            # Store column names as attributes
            f["metadata"].attrs["columns"] = ["flight_idx", "frame_idx", "SZA", "SAA"]

            # Store dataset info
            f.attrs["n_samples"] = len(images)
            f.attrs["image_shape"] = images.shape[1:]
            f.attrs["split"] = split

        file_size_mb = output_path.stat().st_size / (1024**2)
        print(f"   Saved {split}.h5 ({file_size_mb:.1f} MB)")

    def _save_npz(self, images: np.ndarray, metadata: np.ndarray, split: str):
        """Save dataset in NPZ format (compressed numpy)."""
        output_path = self.output_dir / f"{split}.npz"

        np.savez_compressed(
            output_path,
            images=images,
            metadata=metadata,
            metadata_columns=["flight_idx", "frame_idx", "SZA", "SAA"],
        )

        file_size_mb = output_path.stat().st_size / (1024**2)
        print(f"   Saved {split}.npz ({file_size_mb:.1f} MB)")

    def _save_statistics(self):
        """Save extraction statistics to YAML."""
        stats_path = self.output_dir / "extraction_stats.yaml"

        # Add additional statistics
        extended_stats = {
            **self.stats,
            "config": {
                "swath_slice": [self.swath_start, self.swath_end],
                "train_split": self.train_split,
                "vignetting_correction": self.apply_vignetting,
                "format": self.format,
                "random_seed": self.random_seed,
            },
            "flights": [
                fc.get("name", f"flight_{i}")
                for i, fc in enumerate(self.flight_configs)
            ],
        }

        with open(stats_path, "w") as f:
            yaml.dump(extended_stats, f, default_flow_style=False)

        print(f"   Saved extraction_stats.yaml")

    def _print_summary(self):
        """Print extraction summary."""
        print("\n" + "=" * 80)
        print("EXTRACTION SUMMARY")
        print("=" * 80)
        print(
            f" Flights processed: {self.stats['flights_processed']}/{len(self.flight_configs)}"
        )
        if self.stats["flights_failed"] > 0:
            print(f" Flights failed: {self.stats['flights_failed']}")
        print(f" Total images: {self.stats['total_images']:,}")
        print(
            f"  - Training: {self.stats['train_images']:,} ({self.stats['train_images'] / self.stats['total_images']:.1%})"
        )
        print(
            f"  - Validation: {self.stats['val_images']:,} ({self.stats['val_images'] / self.stats['total_images']:.1%})"
        )
        print(f"\n Output directory: {self.output_dir}")
        print(f" Files created:")
        print(f"  - train.{self.format.replace('hdf5', 'h5')}")
        print(f"  - val.{self.format.replace('hdf5', 'h5')}")
        print(f"  - extraction_stats.yaml")
        print("=" * 80)


def load_flight_configs_from_yaml(config_path: str) -> List[Dict]:
    """Load flight configurations from a YAML config file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Extract flight configurations
    flights = config.get("flights", [])
    data_dir = config.get("data_directory", "")

    # Prepend data directory to file paths and add flight_idx
    for i, flight in enumerate(flights):
        flight["flight_idx"] = i
        for key in ["iFileName", "cFileName", "nFileName"]:
            if key in flight and not flight[key].startswith("/"):
                flight[key] = os.path.join(data_dir, flight[key])

    return flights


def main():
    parser = argparse.ArgumentParser(
        description="Extract all IR images for self-supervised learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML config file with flight configurations",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        help="Base directory containing flight data (overrides config)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data_ssl/images",
        help="Output directory for extracted images (default: data_ssl/images)",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["hdf5", "npz"],
        default="hdf5",
        help="Output format (default: hdf5)",
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.95,
        help="Fraction of data for training (default: 0.95)",
    )
    parser.add_argument(
        "--swath-start",
        type=int,
        default=40,
        help="Start index for swath slice (default: 40)",
    )
    parser.add_argument(
        "--swath-end",
        type=int,
        default=480,
        help="End index for swath slice (default: 480)",
    )
    parser.add_argument(
        "--no-vignetting",
        action="store_true",
        help="Disable vignetting correction",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    # Load flight configurations
    if args.config:
        print(f"Loading flight configurations from: {args.config}")
        flight_configs = load_flight_configs_from_yaml(args.config)
    else:
        print("ERROR: --config is required")
        sys.exit(1)

    if not flight_configs:
        print("ERROR: No flight configurations found!")
        sys.exit(1)

    # Override data directory if specified
    if args.data_dir:
        for flight in flight_configs:
            for key in ["iFileName", "cFileName", "nFileName"]:
                if key in flight:
                    flight[key] = os.path.join(
                        args.data_dir,
                        os.path.basename(os.path.dirname(flight[key])),
                        os.path.basename(flight[key]),
                    )

    # Initialize extractor
    extractor = ImageExtractor(
        flight_configs=flight_configs,
        output_dir=args.output_dir,
        swath_slice=(args.swath_start, args.swath_end),
        format=args.format,
        train_split=args.train_split,
        apply_vignetting_correction=not args.no_vignetting,
        random_seed=args.seed,
    )

    # Run extraction
    try:
        extractor.extract_all()
    except KeyboardInterrupt:
        print("\n\n  Extraction interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n ERROR: {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
