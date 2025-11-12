#!/usr/bin/env python3
"""
Sprint 6 - Image Dataset Loader

This module provides a dataset loader that matches SSL image data (20×22 pixel arrays)
to labeled CBH samples from the integrated features file.

The SSL data contains 58,846 unlabeled images (440 pixels = 20×22 reshaped).
The integrated features contain 933 labeled samples with CBH ground truth.
This loader matches them using (flight_id, sample_id) keys.

Author: Sprint 6 Agent
Date: 2025
"""

import json
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class ImageCBHDataset(Dataset):
    """
    Dataset that loads 20×22 pixel images from SSL data and matches them
    to CBH labels from integrated features.

    Features:
    - Loads pre-processed images from data_ssl/images/train.h5
    - Matches to labeled samples via (flight_id, sample_id)
    - Returns: (image, cbh_km, flight_id, sample_id)
    - Supports augmentation and normalization
    """

    def __init__(
        self,
        ssl_images_path: str,
        integrated_features_path: str,
        image_shape: Tuple[int, int] = (20, 22),
        normalize: bool = True,
        augment: bool = False,
        return_indices: bool = False,
    ):
        """
        Initialize dataset.

        Args:
            ssl_images_path: Path to SSL images HDF5 file
            integrated_features_path: Path to integrated features HDF5 file
            image_shape: Shape to reshape images to (height, width)
            normalize: Whether to z-score normalize images
            augment: Whether to apply data augmentation
            return_indices: Whether to return dataset indices
        """
        self.ssl_images_path = ssl_images_path
        self.integrated_features_path = integrated_features_path
        self.image_shape = image_shape
        self.normalize = normalize
        self.augment = augment
        self.return_indices = return_indices

        # Load data and create mapping
        self._load_data()
        self._create_mapping()

        print(f"ImageCBHDataset initialized:")
        print(f"  Total labeled samples: {len(self.labeled_samples)}")
        print(f"  Matched images: {len(self.valid_indices)}")
        print(f"  Image shape: {self.image_shape}")
        print(f"  Normalization: {self.normalize}")
        print(f"  Augmentation: {self.augment}")

    def _load_data(self):
        """Load SSL images and labeled samples metadata."""
        # Load SSL images metadata
        with h5py.File(self.ssl_images_path, "r") as f:
            self.ssl_metadata = f["metadata"][:]  # (N, 4) array
            # Columns: [flight_id, sample_id, ?, ?]
            self.n_ssl_images = self.ssl_metadata.shape[0]

        # Load labeled samples
        with h5py.File(self.integrated_features_path, "r") as f:
            self.labeled_flight_ids = f["metadata/flight_id"][:]
            self.labeled_sample_ids = f["metadata/sample_id"][:]
            self.labeled_cbh_km = f["metadata/cbh_km"][:]
            self.flight_mapping = json.loads(f.attrs["flight_mapping"])

        self.n_labeled = len(self.labeled_cbh_km)

        print(f"Loaded data:")
        print(f"  SSL images: {self.n_ssl_images} total")
        print(f"  Labeled samples: {self.n_labeled}")
        print(f"  Flight mapping: {self.flight_mapping}")

    def _create_mapping(self):
        """Create mapping from labeled samples to SSL image indices."""
        # Create dictionary: (flight_id, sample_id) -> ssl_index
        ssl_lookup = {}
        for i, (flight_id, sample_id) in enumerate(self.ssl_metadata[:, :2]):
            key = (int(flight_id), int(sample_id))
            ssl_lookup[key] = i

        # Find matches for labeled samples
        self.valid_indices = []
        self.ssl_indices = []
        self.labeled_samples = []

        for i in range(self.n_labeled):
            flight_id = int(self.labeled_flight_ids[i])
            sample_id = int(self.labeled_sample_ids[i])
            key = (flight_id, sample_id)

            if key in ssl_lookup:
                self.valid_indices.append(i)
                self.ssl_indices.append(ssl_lookup[key])
                self.labeled_samples.append(
                    {
                        "flight_id": flight_id,
                        "sample_id": sample_id,
                        "cbh_km": float(self.labeled_cbh_km[i]),
                        "ssl_index": ssl_lookup[key],
                        "labeled_index": i,
                    }
                )

        print(f"Mapping complete:")
        print(f"  Matched samples: {len(self.valid_indices)} / {self.n_labeled}")
        print(f"  Match rate: {100 * len(self.valid_indices) / self.n_labeled:.1f}%")

        if len(self.valid_indices) < self.n_labeled:
            missing = self.n_labeled - len(self.valid_indices)
            print(f"  WARNING: {missing} labeled samples have no matching images!")

    def __len__(self):
        """Return number of matched samples."""
        return len(self.valid_indices)

    def _load_image(self, ssl_index: int) -> np.ndarray:
        """Load a single image from SSL data."""
        with h5py.File(self.ssl_images_path, "r") as f:
            img_flat = f["images"][ssl_index]

        # Reshape to 2D
        img = img_flat.reshape(self.image_shape)

        return img

    def _normalize_image(self, img: np.ndarray) -> np.ndarray:
        """Z-score normalize image."""
        mean = img.mean()
        std = img.std()
        if std > 0:
            img = (img - mean) / std
        return img

    def _augment_image(self, img: np.ndarray) -> np.ndarray:
        """Apply data augmentation."""
        # Horizontal flip (50% chance)
        if np.random.rand() > 0.5:
            img = np.fliplr(img)

        # Vertical flip (50% chance)
        if np.random.rand() > 0.5:
            img = np.flipud(img)

        # Small random noise (10% std)
        noise = np.random.randn(*img.shape) * 0.1 * img.std()
        img = img + noise

        return img

    def __getitem__(self, idx: int):
        """
        Get item by index.

        Returns:
            image: (H, W) array, normalized if requested
            cbh_km: Cloud base height in km
            flight_id: Flight ID
            sample_id: Sample ID within flight
            (optional) labeled_index: Index in labeled dataset
        """
        # Get labeled sample info
        labeled_idx = self.valid_indices[idx]
        ssl_idx = self.ssl_indices[idx]
        sample_info = self.labeled_samples[idx]

        # Load image
        img = self._load_image(ssl_idx)

        # Normalize
        if self.normalize:
            img = self._normalize_image(img)

        # Augment (if training)
        if self.augment:
            img = self._augment_image(img)

        # Convert to tensor (add channel dimension: [1, H, W])
        img_tensor = torch.from_numpy(img).float().unsqueeze(0)

        # Get label
        cbh_km = sample_info["cbh_km"]
        cbh_tensor = torch.tensor(cbh_km, dtype=torch.float32)

        # Metadata
        flight_id = sample_info["flight_id"]
        sample_id = sample_info["sample_id"]

        if self.return_indices:
            return (
                img_tensor,
                cbh_tensor,
                flight_id,
                sample_id,
                labeled_idx,
            )
        else:
            return (
                img_tensor,
                cbh_tensor,
                flight_id,
                sample_id,
            )

    def get_flight_indices(self) -> Dict[int, List[int]]:
        """
        Get indices grouped by flight.

        Returns:
            Dictionary mapping flight_id to list of dataset indices
        """
        flight_indices = {}
        for i, sample in enumerate(self.labeled_samples):
            flight_id = sample["flight_id"]
            if flight_id not in flight_indices:
                flight_indices[flight_id] = []
            flight_indices[flight_id].append(i)

        return flight_indices

    def get_statistics(self) -> Dict:
        """
        Compute dataset statistics.

        Returns:
            Dictionary with statistics
        """
        # Load all images to compute stats (use subset for speed)
        n_samples = min(len(self), 1000)
        indices = np.random.choice(len(self), n_samples, replace=False)

        images = []
        cbhs = []

        for idx in indices:
            img_tensor, cbh_tensor, _, _ = self[idx]
            images.append(img_tensor.squeeze().numpy())
            cbhs.append(cbh_tensor.item())

        images = np.array(images)
        cbhs = np.array(cbhs)

        stats = {
            "n_samples": len(self),
            "image_shape": self.image_shape,
            "image_mean": float(images.mean()),
            "image_std": float(images.std()),
            "image_min": float(images.min()),
            "image_max": float(images.max()),
            "cbh_mean": float(cbhs.mean()),
            "cbh_std": float(cbhs.std()),
            "cbh_min": float(cbhs.min()),
            "cbh_max": float(cbhs.max()),
            "flight_distribution": {
                str(fid): len(indices)
                for fid, indices in self.get_flight_indices().items()
            },
        }

        return stats


class TemporalImageDataset(ImageCBHDataset):
    """
    Extension of ImageCBHDataset that loads temporal sequences.

    For each sample, loads T consecutive frames (e.g., T=5 for context).
    Handles boundary cases by padding with repeated frames.
    """

    def __init__(
        self,
        ssl_images_path: str,
        integrated_features_path: str,
        temporal_frames: int = 5,
        image_shape: Tuple[int, int] = (20, 22),
        normalize: bool = True,
        augment: bool = False,
        return_indices: bool = False,
    ):
        """
        Initialize temporal dataset.

        Args:
            temporal_frames: Number of consecutive frames to load
            Other args same as ImageCBHDataset
        """
        self.temporal_frames = temporal_frames
        self.temporal_offset = temporal_frames // 2

        super().__init__(
            ssl_images_path=ssl_images_path,
            integrated_features_path=integrated_features_path,
            image_shape=image_shape,
            normalize=normalize,
            augment=augment,
            return_indices=return_indices,
        )

        # Build temporal indices
        self._build_temporal_indices()

    def _build_temporal_indices(self):
        """Build indices for temporal sequences."""
        # Group samples by flight for temporal continuity
        flight_groups = {}
        for i, sample in enumerate(self.labeled_samples):
            flight_id = sample["flight_id"]
            sample_id = sample["sample_id"]

            if flight_id not in flight_groups:
                flight_groups[flight_id] = []

            flight_groups[flight_id].append(
                {
                    "dataset_idx": i,
                    "sample_id": sample_id,
                    "ssl_idx": sample["ssl_index"],
                }
            )

        # Sort each flight by sample_id for temporal order
        for flight_id in flight_groups:
            flight_groups[flight_id].sort(key=lambda x: x["sample_id"])

        # Build temporal sequences
        self.temporal_sequences = []

        for flight_id, samples in flight_groups.items():
            for i, sample in enumerate(samples):
                # Get temporal neighbors
                temporal_indices = []

                for offset in range(-self.temporal_offset, self.temporal_offset + 1):
                    neighbor_idx = i + offset

                    # Handle boundaries by clamping
                    neighbor_idx = max(0, min(len(samples) - 1, neighbor_idx))
                    temporal_indices.append(samples[neighbor_idx]["ssl_idx"])

                self.temporal_sequences.append(
                    {
                        "dataset_idx": sample["dataset_idx"],
                        "temporal_ssl_indices": temporal_indices,
                        "center_idx": i,
                    }
                )

        print(f"Temporal sequences built:")
        print(f"  Temporal frames: {self.temporal_frames}")
        print(f"  Total sequences: {len(self.temporal_sequences)}")

    def __getitem__(self, idx: int):
        """
        Get temporal sequence.

        Returns:
            images: (T, H, W) tensor of temporal sequence
            cbh_km: CBH of center frame
            flight_id: Flight ID
            sample_id: Sample ID
        """
        sequence_info = self.temporal_sequences[idx]
        temporal_ssl_indices = sequence_info["temporal_ssl_indices"]
        dataset_idx = sequence_info["dataset_idx"]

        # Load temporal images
        temporal_images = []
        for ssl_idx in temporal_ssl_indices:
            img = self._load_image(ssl_idx)

            # Normalize
            if self.normalize:
                img = self._normalize_image(img)

            temporal_images.append(img)

        # Stack into tensor (T, H, W)
        temporal_tensor = torch.from_numpy(np.array(temporal_images)).float()

        # Augment entire sequence consistently
        if self.augment:
            # Apply same flip to all frames
            if np.random.rand() > 0.5:
                temporal_tensor = torch.flip(temporal_tensor, dims=[2])  # Horizontal
            if np.random.rand() > 0.5:
                temporal_tensor = torch.flip(temporal_tensor, dims=[1])  # Vertical

        # Get label from center frame
        sample_info = self.labeled_samples[dataset_idx]
        cbh_tensor = torch.tensor(sample_info["cbh_km"], dtype=torch.float32)
        flight_id = sample_info["flight_id"]
        sample_id = sample_info["sample_id"]

        if self.return_indices:
            return (
                temporal_tensor,
                cbh_tensor,
                flight_id,
                sample_id,
                dataset_idx,
            )
        else:
            return (
                temporal_tensor,
                cbh_tensor,
                flight_id,
                sample_id,
            )
