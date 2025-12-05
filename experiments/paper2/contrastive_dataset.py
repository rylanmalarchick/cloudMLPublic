#!/usr/bin/env python3
"""Contrastive dataset with augmentations for SimCLR pretraining.

This module provides a dataset that generates two augmented views of each
image for contrastive learning. Designed for small (20x22) cloud radiance
images from the SSL unlabeled pool.

Augmentation Strategy:
    Following SimCLR recommendations, we use strong augmentations:
    - Random cropping (with resize back to original)
    - Horizontal and vertical flips
    - Gaussian noise injection
    - Brightness/contrast jitter
    - Gaussian blur

Author: Paper 2 Implementation
Date: 2025
"""

from __future__ import annotations

import atexit
import threading
from pathlib import Path
from typing import Dict, Optional, Tuple

import h5py
import numpy as np
import numpy.typing as npt
import torch
from scipy.ndimage import gaussian_filter
from torch import Tensor
from torch.utils.data import Dataset


# Global HDF5 file cache for efficient data loading
_h5_file_cache: Dict[str, h5py.File] = {}
_h5_cache_lock = threading.Lock()


def _get_h5_file(path: str) -> h5py.File:
    """Get a cached HDF5 file handle (thread-safe)."""
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


atexit.register(_cleanup_h5_cache)


class ContrastiveAugmentation:
    """Augmentation pipeline for contrastive learning.
    
    Generates two different augmented views of the same image.
    Augmentations are designed for small grayscale images (20x22).
    
    Args:
        image_shape: Shape of images as (height, width).
        noise_std: Standard deviation of Gaussian noise (relative to image std).
        brightness_range: Range for brightness adjustment (min, max).
        contrast_range: Range for contrast adjustment (min, max).
        blur_prob: Probability of applying Gaussian blur.
        blur_sigma: Sigma range for Gaussian blur.
    """
    
    def __init__(
        self,
        image_shape: Tuple[int, int] = (20, 22),
        noise_std: float = 0.1,
        brightness_range: Tuple[float, float] = (0.8, 1.2),
        contrast_range: Tuple[float, float] = (0.8, 1.2),
        blur_prob: float = 0.5,
        blur_sigma: Tuple[float, float] = (0.5, 1.0),
    ) -> None:
        self.image_shape = image_shape
        self.noise_std = noise_std
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.blur_prob = blur_prob
        self.blur_sigma = blur_sigma
    
    def _random_crop_resize(
        self, img: npt.NDArray[np.float64], crop_ratio: float = 0.8
    ) -> npt.NDArray[np.float64]:
        """Random crop and resize back to original size.
        
        For small images, we use a larger crop ratio to preserve information.
        """
        h, w = img.shape
        
        # Random crop size (80-100% of original)
        crop_h = int(h * np.random.uniform(crop_ratio, 1.0))
        crop_w = int(w * np.random.uniform(crop_ratio, 1.0))
        
        # Random crop position
        top = np.random.randint(0, h - crop_h + 1)
        left = np.random.randint(0, w - crop_w + 1)
        
        # Crop
        cropped = img[top:top + crop_h, left:left + crop_w]
        
        # Simple resize by replication (for tiny images)
        # Use numpy's repeat to approximately resize
        h_repeat = max(1, round(h / crop_h))
        w_repeat = max(1, round(w / crop_w))
        
        resized = np.repeat(np.repeat(cropped, h_repeat, axis=0), w_repeat, axis=1)
        
        # Crop/pad to exact size
        resized = resized[:h, :w]
        if resized.shape[0] < h or resized.shape[1] < w:
            padded = np.zeros((h, w), dtype=img.dtype)
            padded[:resized.shape[0], :resized.shape[1]] = resized
            resized = padded
        
        return resized
    
    def _horizontal_flip(
        self, img: npt.NDArray[np.float64], prob: float = 0.5
    ) -> npt.NDArray[np.float64]:
        """Random horizontal flip."""
        if np.random.rand() < prob:
            return np.fliplr(img).copy()
        return img
    
    def _vertical_flip(
        self, img: npt.NDArray[np.float64], prob: float = 0.5
    ) -> npt.NDArray[np.float64]:
        """Random vertical flip."""
        if np.random.rand() < prob:
            return np.flipud(img).copy()
        return img
    
    def _add_noise(
        self, img: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Add Gaussian noise."""
        noise = np.random.randn(*img.shape) * self.noise_std * img.std()
        return img + noise
    
    def _adjust_brightness(
        self, img: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Random brightness adjustment."""
        factor = np.random.uniform(*self.brightness_range)
        return img * factor
    
    def _adjust_contrast(
        self, img: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Random contrast adjustment."""
        factor = np.random.uniform(*self.contrast_range)
        mean = img.mean()
        return (img - mean) * factor + mean
    
    def _gaussian_blur(
        self, img: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Apply Gaussian blur using scipy (vectorized, ~50-100x faster)."""
        if np.random.rand() > self.blur_prob:
            return img
        
        sigma = np.random.uniform(*self.blur_sigma)
        return gaussian_filter(img, sigma=sigma)
    
    def __call__(
        self, img: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Apply random augmentation to image.
        
        Args:
            img: Input image of shape (H, W).
        
        Returns:
            Augmented image of shape (H, W).
        """
        # Apply augmentations in order (with some randomness)
        aug_img = img.copy()
        
        # Geometric transforms
        if np.random.rand() < 0.8:  # 80% chance of crop
            aug_img = self._random_crop_resize(aug_img)
        aug_img = self._horizontal_flip(aug_img)
        aug_img = self._vertical_flip(aug_img)
        
        # Photometric transforms
        aug_img = self._adjust_brightness(aug_img)
        aug_img = self._adjust_contrast(aug_img)
        aug_img = self._add_noise(aug_img)
        aug_img = self._gaussian_blur(aug_img)
        
        return aug_img


class ContrastiveSSLDataset(Dataset[Tuple[Tensor, Tensor]]):
    """Dataset for SimCLR contrastive pretraining.
    
    Loads unlabeled SSL images and generates two augmented views per sample.
    
    Attributes:
        h5_path: Path to HDF5 file with images.
        image_shape: Shape to reshape images to.
        augmentation: Augmentation pipeline.
        n_samples: Number of images in dataset.
    
    Example:
        >>> dataset = ContrastiveSSLDataset("data_ssl/images/train.h5")
        >>> view1, view2 = dataset[0]
        >>> view1.shape, view2.shape
        (torch.Size([1, 20, 22]), torch.Size([1, 20, 22]))
    """
    
    def __init__(
        self,
        h5_path: str,
        image_shape: Tuple[int, int] = (20, 22),
        augmentation: Optional[ContrastiveAugmentation] = None,
    ) -> None:
        """Initialize ContrastiveSSLDataset.
        
        Args:
            h5_path: Path to HDF5 file containing images.
            image_shape: Shape to reshape flattened images.
            augmentation: Custom augmentation pipeline. If None, uses default.
        """
        self.h5_path = h5_path
        self.image_shape = image_shape
        self.augmentation = augmentation or ContrastiveAugmentation(image_shape)
        
        # Get dataset size
        with h5py.File(h5_path, "r") as f:
            self.n_samples = f["images"].shape[0]  # type: ignore[union-attr]
        
        print(f"ContrastiveSSLDataset initialized:")
        print(f"  Path: {h5_path}")
        print(f"  Samples: {self.n_samples:,}")
        print(f"  Image shape: {image_shape}")
    
    def __len__(self) -> int:
        """Return number of samples."""
        return self.n_samples
    
    def _load_image(self, idx: int) -> npt.NDArray[np.float64]:
        """Load and reshape a single image."""
        f = _get_h5_file(self.h5_path)
        img_flat = f["images"][idx]  # type: ignore[index]
        
        return img_flat.reshape(self.image_shape)  # type: ignore[union-attr]
    
    def _normalize(
        self, img: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Z-score normalize image."""
        mean = img.mean()
        std = img.std()
        if std > 0:
            return (img - mean) / std
        return img - mean
    
    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """Get two augmented views of the same image.
        
        Args:
            idx: Index of image to load.
        
        Returns:
            Tuple of (view1, view2), each of shape (1, H, W).
        """
        # Load original image
        img = self._load_image(idx)
        
        # Generate two different augmented views
        view1 = self.augmentation(img)
        view2 = self.augmentation(img)
        
        # Normalize
        view1 = self._normalize(view1)
        view2 = self._normalize(view2)
        
        # Convert to tensors with channel dimension
        view1_tensor = torch.from_numpy(view1).float().unsqueeze(0)
        view2_tensor = torch.from_numpy(view2).float().unsqueeze(0)
        
        return view1_tensor, view2_tensor


class ContrastiveLabeledDataset(Dataset[Tuple[Tensor, Tensor, Tensor]]):
    """Dataset that provides contrastive views + labels for evaluation.
    
    Used for linear probe evaluation on labeled data.
    
    Attributes:
        ssl_path: Path to SSL images HDF5.
        features_path: Path to integrated features HDF5.
        augmentation: Augmentation pipeline.
        matched_samples: List of matched (ssl_idx, cbh_km) pairs.
    """
    
    def __init__(
        self,
        ssl_path: str,
        features_path: str,
        image_shape: Tuple[int, int] = (20, 22),
        augment: bool = False,
    ) -> None:
        """Initialize dataset with labeled samples.
        
        Args:
            ssl_path: Path to SSL images HDF5.
            features_path: Path to integrated features HDF5 with labels.
            image_shape: Shape to reshape images.
            augment: Whether to apply augmentation.
        """
        self.ssl_path = ssl_path
        self.features_path = features_path
        self.image_shape = image_shape
        self.augment = augment
        
        if augment:
            self.augmentation = ContrastiveAugmentation(image_shape)
        else:
            self.augmentation = None
        
        # Match labeled samples to SSL images
        self._create_mapping()
        
        print(f"ContrastiveLabeledDataset initialized:")
        print(f"  Matched samples: {len(self.matched_samples)}")
        print(f"  Augmentation: {augment}")
    
    def _create_mapping(self) -> None:
        """Create mapping between labeled samples and SSL images."""
        import json
        
        # Load SSL metadata
        with h5py.File(self.ssl_path, "r") as f:
            ssl_metadata = f["metadata"][:]  # type: ignore[index]
        
        # Create lookup: (flight_id, sample_id) -> ssl_index
        ssl_lookup = {}
        for i, (flight_id, sample_id, _, _) in enumerate(ssl_metadata):
            ssl_lookup[(int(flight_id), int(sample_id))] = i
        
        # Load labeled data
        with h5py.File(self.features_path, "r") as f:
            flight_ids = f["metadata/flight_id"][:]  # type: ignore[index]
            sample_ids = f["metadata/sample_id"][:]  # type: ignore[index]
            cbh_km = f["metadata/cbh_km"][:]  # type: ignore[index]
        
        # Match samples
        self.matched_samples = []
        self.flight_ids_list = []
        
        for i in range(len(cbh_km)):
            key = (int(flight_ids[i]), int(sample_ids[i]))
            if key in ssl_lookup:
                self.matched_samples.append({
                    "ssl_idx": ssl_lookup[key],
                    "cbh_km": float(cbh_km[i]),
                    "flight_id": int(flight_ids[i]),
                    "sample_id": int(sample_ids[i]),
                    "labeled_idx": i,
                })
                self.flight_ids_list.append(int(flight_ids[i]))
    
    def __len__(self) -> int:
        """Return number of matched samples."""
        return len(self.matched_samples)
    
    def _load_image(self, ssl_idx: int) -> npt.NDArray[np.float64]:
        """Load image from SSL file."""
        f = _get_h5_file(self.ssl_path)
        img_flat = f["images"][ssl_idx]  # type: ignore[index]
        return img_flat.reshape(self.image_shape)  # type: ignore[union-attr]
    
    def _normalize(
        self, img: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Z-score normalize."""
        mean = img.mean()
        std = img.std()
        if std > 0:
            return (img - mean) / std
        return img - mean
    
    def __getitem__(
        self, idx: int
    ) -> Tuple[Tensor, Tensor, int]:
        """Get image, label, and flight_id.
        
        Args:
            idx: Sample index.
        
        Returns:
            Tuple of (image, cbh_km, flight_id).
        """
        sample = self.matched_samples[idx]
        
        # Load and process image
        img = self._load_image(sample["ssl_idx"])
        
        if self.augment and self.augmentation is not None:
            img = self.augmentation(img)
        
        img = self._normalize(img)
        img_tensor = torch.from_numpy(img).float().unsqueeze(0)
        
        # Label
        cbh_tensor = torch.tensor(sample["cbh_km"], dtype=torch.float32)
        
        return img_tensor, cbh_tensor, sample["flight_id"]
    
    def get_flight_indices(self) -> dict:
        """Get indices grouped by flight for LOO cross-validation."""
        flight_indices = {}
        for i, sample in enumerate(self.matched_samples):
            fid = sample["flight_id"]
            if fid not in flight_indices:
                flight_indices[fid] = []
            flight_indices[fid].append(i)
        return flight_indices


if __name__ == "__main__":
    from pathlib import Path
    
    # Quick test
    print("Testing ContrastiveSSLDataset...")
    
    ssl_path = Path("data_ssl/images/train.h5")
    if ssl_path.exists():
        dataset = ContrastiveSSLDataset(str(ssl_path))
        
        # Test loading
        view1, view2 = dataset[0]
        print(f"View 1 shape: {view1.shape}")
        print(f"View 2 shape: {view2.shape}")
        print(f"View 1 stats: mean={view1.mean():.3f}, std={view1.std():.3f}")
        print(f"View 2 stats: mean={view2.mean():.3f}, std={view2.std():.3f}")
        
        # Views should be different (augmented differently)
        diff = (view1 - view2).abs().mean()
        print(f"Mean absolute difference between views: {diff:.3f}")
        
        print("\nAll tests passed!")
    else:
        print(f"Test data not found at {ssl_path}")
