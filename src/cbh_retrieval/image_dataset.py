#!/usr/bin/env python3
"""Sprint 6 - Image Dataset Loader.

This module provides a dataset loader that matches SSL image data (20×22 pixel arrays)
to labeled CBH samples from the integrated features file.

The SSL data contains 58,846 unlabeled images (440 pixels = 20×22 reshaped).
The integrated features contain 933 labeled samples with CBH ground truth.
This loader matches them using (flight_id, sample_id) keys.

Classes:
    ImageCBHDataset: Base dataset for loading images with CBH labels.
    TemporalImageDataset: Extended dataset that loads temporal sequences.

Example:
    >>> dataset = ImageCBHDataset(
    ...     ssl_images_path="data_ssl/images/train.h5",
    ...     integrated_features_path="data/integrated_features.h5",
    ...     normalize=True,
    ...     augment=False,
    ... )
    >>> image, cbh, flight_id, sample_id = dataset[0]

Author: Sprint 6 Agent
Date: 2025
"""

from __future__ import annotations

import atexit
import json
import threading
from typing import Any, Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor
from torch.utils.data import Dataset


# Global HDF5 file cache for efficient data loading
_h5_file_cache: Dict[str, h5py.File] = {}
_h5_cache_lock = threading.Lock()


def _get_h5_file(path: str) -> h5py.File:
    """Get a cached HDF5 file handle (thread-safe).
    
    Args:
        path: Path to the HDF5 file.
        
    Returns:
        Open HDF5 file handle.
    """
    if path not in _h5_file_cache:
        with _h5_cache_lock:
            if path not in _h5_file_cache:
                _h5_file_cache[path] = h5py.File(path, "r", swmr=True)
    return _h5_file_cache[path]


def _cleanup_h5_cache() -> None:
    """Close all cached HDF5 file handles on exit."""
    for f in _h5_file_cache.values():
        try:
            f.close()
        except Exception:
            pass
    _h5_file_cache.clear()


# Register cleanup function
atexit.register(_cleanup_h5_cache)


class ImageCBHDataset(Dataset[Tuple[Tensor, ...]]):
    """Dataset that loads 20×22 pixel images and matches them to CBH labels.

    This dataset loads pre-processed images from SSL (Self-Supervised Learning)
    data and matches them to labeled CBH (Cloud Base Height) samples using
    (flight_id, sample_id) composite keys.

    Attributes:
        ssl_images_path: Path to the SSL images HDF5 file.
        integrated_features_path: Path to the integrated features HDF5 file.
        image_shape: Shape of reshaped images as (height, width).
        normalize: Whether z-score normalization is applied.
        augment: Whether data augmentation is applied.
        return_indices: Whether dataset indices are returned.
        labeled_samples: List of matched sample information dictionaries.
        valid_indices: List of valid labeled sample indices.
        ssl_indices: List of corresponding SSL image indices.
        n_ssl_images: Total number of SSL images available.
        n_labeled: Total number of labeled samples.
        flight_mapping: Dictionary mapping flight names to IDs.

    Example:
        >>> dataset = ImageCBHDataset(
        ...     ssl_images_path="data_ssl/images/train.h5",
        ...     integrated_features_path="data/integrated_features.h5",
        ... )
        >>> len(dataset)
        933
        >>> image, cbh, flight_id, sample_id = dataset[0]
        >>> image.shape
        torch.Size([1, 20, 22])
    """

    ssl_images_path: str
    integrated_features_path: str
    image_shape: Tuple[int, int]
    normalize: bool
    augment: bool
    return_indices: bool
    labeled_samples: List[Dict[str, Any]]
    valid_indices: List[int]
    ssl_indices: List[int]
    n_ssl_images: int
    n_labeled: int
    flight_mapping: Dict[str, int]
    ssl_metadata: npt.NDArray[np.float64]
    labeled_flight_ids: npt.NDArray[np.int64]
    labeled_sample_ids: npt.NDArray[np.int64]
    labeled_cbh_km: npt.NDArray[np.float64]

    def __init__(
        self,
        ssl_images_path: str,
        integrated_features_path: str,
        image_shape: Tuple[int, int] = (20, 22),
        normalize: bool = True,
        augment: bool = False,
        return_indices: bool = False,
    ) -> None:
        """Initialize the ImageCBHDataset.

        Args:
            ssl_images_path: Path to the SSL images HDF5 file containing
                the raw image data and metadata.
            integrated_features_path: Path to the integrated features HDF5
                file containing labeled CBH samples.
            image_shape: Shape to reshape flattened images to, specified as
                (height, width). Defaults to (20, 22).
            normalize: Whether to apply z-score normalization to images.
                Defaults to True.
            augment: Whether to apply data augmentation (random flips and
                noise). Defaults to False.
            return_indices: Whether to return the labeled dataset index
                as an additional element. Defaults to False.

        Raises:
            FileNotFoundError: If either HDF5 file does not exist.
            KeyError: If required datasets are missing from HDF5 files.
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

    def _load_data(self) -> None:
        """Load SSL images and labeled samples metadata from HDF5 files.

        This method reads metadata from both the SSL images file and the
        integrated features file, storing them as instance attributes for
        later use in sample matching.

        Raises:
            FileNotFoundError: If either HDF5 file does not exist.
            KeyError: If required datasets are missing from HDF5 files.
        """
        # Load SSL images metadata
        with h5py.File(self.ssl_images_path, "r") as f:
            self.ssl_metadata = f["metadata"][:]  # type: ignore[index]  # (N, 4) array
            # Columns: [flight_id, sample_id, ?, ?]
            self.n_ssl_images = self.ssl_metadata.shape[0]

        # Load labeled samples
        with h5py.File(self.integrated_features_path, "r") as f:
            self.labeled_flight_ids = f["metadata/flight_id"][:]  # type: ignore[index]
            self.labeled_sample_ids = f["metadata/sample_id"][:]  # type: ignore[index]
            self.labeled_cbh_km = f["metadata/cbh_km"][:]  # type: ignore[index]
            self.flight_mapping = json.loads(f.attrs["flight_mapping"])  # type: ignore[arg-type]

        self.n_labeled = len(self.labeled_cbh_km)

        print(f"Loaded data:")
        print(f"  SSL images: {self.n_ssl_images} total")
        print(f"  Labeled samples: {self.n_labeled}")
        print(f"  Flight mapping: {self.flight_mapping}")

    def _create_mapping(self) -> None:
        """Create mapping from labeled samples to SSL image indices.

        Builds a lookup table to match labeled samples with their corresponding
        SSL images using (flight_id, sample_id) composite keys. Populates
        the valid_indices, ssl_indices, and labeled_samples attributes.

        Note:
            Samples without matching images are logged as warnings but do not
            raise exceptions. The match rate is printed to stdout.
        """
        # Create dictionary: (flight_id, sample_id) -> ssl_index
        ssl_lookup: Dict[Tuple[int, int], int] = {}
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

    def __len__(self) -> int:
        """Return the number of matched samples in the dataset.

        Returns:
            The number of samples that have both labeled CBH values
            and corresponding SSL images.
        """
        return len(self.valid_indices)

    def _load_image(self, ssl_index: int) -> npt.NDArray[np.float64]:
        """Load a single image from the SSL data file.

        Args:
            ssl_index: Index of the image in the SSL images HDF5 file.

        Returns:
            2D numpy array of shape (height, width) containing the
            image pixel values.

        Raises:
            IndexError: If ssl_index is out of bounds.
        """
        # Use cached file handle for efficiency (avoids reopening on every access)
        f = _get_h5_file(self.ssl_images_path)
        img_flat = f["images"][ssl_index]  # type: ignore[index]

        # Reshape to 2D
        img: npt.NDArray[np.float64] = img_flat.reshape(self.image_shape)  # type: ignore[union-attr]

        return img

    def _normalize_image(self, img: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Apply z-score normalization to an image.

        Normalizes the image to have zero mean and unit standard deviation.
        If the standard deviation is zero (constant image), returns the
        original image unchanged.

        Args:
            img: Input image array of any shape.

        Returns:
            Normalized image array with the same shape as input.
        """
        mean = img.mean()
        std = img.std()
        if std > 0:
            img = (img - mean) / std
        return img

    def _augment_image(self, img: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Apply random data augmentation to an image.

        Augmentation includes:
        - Horizontal flip (50% probability)
        - Vertical flip (50% probability)
        - Additive Gaussian noise (10% of image std)

        Args:
            img: Input image array of shape (height, width).

        Returns:
            Augmented image array with the same shape as input.
        """
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

    def __getitem__(
        self, idx: int
    ) -> Union[
        Tuple[Tensor, Tensor, int, int, int],
        Tuple[Tensor, Tensor, int, int],
    ]:
        """Get a sample by index.

        Args:
            idx: Index of the sample to retrieve (0 to len(dataset)-1).

        Returns:
            A tuple containing:
            - image: Tensor of shape (1, H, W) with the normalized image
            - cbh_km: Tensor containing the cloud base height in kilometers
            - flight_id: Integer flight identifier
            - sample_id: Integer sample identifier within the flight
            - labeled_index (optional): Index in the original labeled dataset,
              only included if return_indices=True

        Raises:
            IndexError: If idx is out of bounds.
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
        flight_id: int = sample_info["flight_id"]
        sample_id: int = sample_info["sample_id"]

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
        """Get dataset indices grouped by flight.

        Useful for leave-one-flight-out cross-validation or analyzing
        per-flight performance.

        Returns:
            Dictionary mapping flight_id to a list of dataset indices
            belonging to that flight.

        Example:
            >>> flight_indices = dataset.get_flight_indices()
            >>> train_indices = []
            >>> for fid, indices in flight_indices.items():
            ...     if fid != test_flight:
            ...         train_indices.extend(indices)
        """
        flight_indices: Dict[int, List[int]] = {}
        for i, sample in enumerate(self.labeled_samples):
            flight_id: int = sample["flight_id"]
            if flight_id not in flight_indices:
                flight_indices[flight_id] = []
            flight_indices[flight_id].append(i)

        return flight_indices

    def get_statistics(self) -> Dict[str, Any]:
        """Compute and return dataset statistics.

        Samples up to 1000 random images to compute statistics for
        efficiency on large datasets.

        Returns:
            Dictionary containing:
            - n_samples: Total number of samples in dataset
            - image_shape: Tuple of (height, width)
            - image_mean: Mean pixel value across sampled images
            - image_std: Standard deviation of pixel values
            - image_min: Minimum pixel value
            - image_max: Maximum pixel value
            - cbh_mean: Mean CBH value in kilometers
            - cbh_std: Standard deviation of CBH values
            - cbh_min: Minimum CBH value
            - cbh_max: Maximum CBH value
            - flight_distribution: Dict mapping flight_id to sample count

        Example:
            >>> stats = dataset.get_statistics()
            >>> print(f"CBH range: {stats['cbh_min']:.2f} - {stats['cbh_max']:.2f} km")
        """
        # Load all images to compute stats (use subset for speed)
        n_samples = min(len(self), 1000)
        indices = np.random.choice(len(self), n_samples, replace=False)

        images: List[npt.NDArray[np.float64]] = []
        cbhs: List[float] = []

        for idx in indices:
            img_tensor, cbh_tensor, _, _ = self[idx]  # type: ignore[misc]
            images.append(img_tensor.squeeze().numpy())
            cbhs.append(cbh_tensor.item())

        images_arr = np.array(images)
        cbhs_arr = np.array(cbhs)

        stats: Dict[str, Any] = {
            "n_samples": len(self),
            "image_shape": self.image_shape,
            "image_mean": float(images_arr.mean()),
            "image_std": float(images_arr.std()),
            "image_min": float(images_arr.min()),
            "image_max": float(images_arr.max()),
            "cbh_mean": float(cbhs_arr.mean()),
            "cbh_std": float(cbhs_arr.std()),
            "cbh_min": float(cbhs_arr.min()),
            "cbh_max": float(cbhs_arr.max()),
            "flight_distribution": {
                str(fid): len(indices)
                for fid, indices in self.get_flight_indices().items()
            },
        }

        return stats


class TemporalImageDataset(ImageCBHDataset):
    """Extension of ImageCBHDataset that loads temporal sequences.

    For each sample, loads T consecutive frames centered on the target frame.
    This provides temporal context for models that leverage time-series
    information. Boundary cases are handled by repeating edge frames.

    Attributes:
        temporal_frames: Number of consecutive frames in each sequence.
        temporal_offset: Number of frames before/after center frame.
        temporal_sequences: List of sequence metadata dictionaries.

    Example:
        >>> dataset = TemporalImageDataset(
        ...     ssl_images_path="data_ssl/images/train.h5",
        ...     integrated_features_path="data/integrated_features.h5",
        ...     temporal_frames=5,
        ... )
        >>> images, cbh, flight_id, sample_id = dataset[0]
        >>> images.shape
        torch.Size([5, 20, 22])
    """

    temporal_frames: int
    temporal_offset: int
    temporal_sequences: List[Dict[str, Any]]

    def __init__(
        self,
        ssl_images_path: str,
        integrated_features_path: str,
        temporal_frames: int = 5,
        image_shape: Tuple[int, int] = (20, 22),
        normalize: bool = True,
        augment: bool = False,
        return_indices: bool = False,
    ) -> None:
        """Initialize the TemporalImageDataset.

        Args:
            ssl_images_path: Path to the SSL images HDF5 file containing
                the raw image data and metadata.
            integrated_features_path: Path to the integrated features HDF5
                file containing labeled CBH samples.
            temporal_frames: Number of consecutive frames to load per sample.
                Should be an odd number for symmetric context. Defaults to 5.
            image_shape: Shape to reshape flattened images to, specified as
                (height, width). Defaults to (20, 22).
            normalize: Whether to apply z-score normalization to images.
                Defaults to True.
            augment: Whether to apply data augmentation (random flips).
                Note: Noise augmentation is not applied to temporal sequences.
                Defaults to False.
            return_indices: Whether to return the labeled dataset index
                as an additional element. Defaults to False.

        Raises:
            FileNotFoundError: If either HDF5 file does not exist.
            KeyError: If required datasets are missing from HDF5 files.
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

    def _build_temporal_indices(self) -> None:
        """Build indices for temporal sequences.

        Groups samples by flight and sorts by sample_id to establish
        temporal ordering. For each sample, identifies the indices of
        neighboring frames within the same flight.

        Note:
            Boundary frames are handled by clamping indices to valid
            range, effectively repeating the first/last frame.
        """
        # Group samples by flight for temporal continuity
        flight_groups: Dict[int, List[Dict[str, int]]] = {}
        for i, sample in enumerate(self.labeled_samples):
            flight_id: int = sample["flight_id"]
            sample_id: int = sample["sample_id"]

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
                temporal_indices: List[int] = []

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

    def __getitem__(
        self, idx: int
    ) -> Union[
        Tuple[Tensor, Tensor, int, int, int],
        Tuple[Tensor, Tensor, int, int],
    ]:
        """Get a temporal sequence by index.

        Args:
            idx: Index of the sequence to retrieve (0 to len(dataset)-1).

        Returns:
            A tuple containing:
            - images: Tensor of shape (T, H, W) with the temporal sequence.
              Unlike the base class, there is no channel dimension.
            - cbh_km: Tensor containing the cloud base height of the center
              frame in kilometers
            - flight_id: Integer flight identifier
            - sample_id: Integer sample identifier of the center frame
            - dataset_idx (optional): Index in the original dataset,
              only included if return_indices=True

        Raises:
            IndexError: If idx is out of bounds.
        """
        sequence_info = self.temporal_sequences[idx]
        temporal_ssl_indices: List[int] = sequence_info["temporal_ssl_indices"]
        dataset_idx: int = sequence_info["dataset_idx"]

        # Load temporal images
        temporal_images: List[npt.NDArray[np.float64]] = []
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
        flight_id: int = sample_info["flight_id"]
        sample_id: int = sample_info["sample_id"]

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
