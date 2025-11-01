"""
SSL Dataset Loader for Phase 1 Extracted Data

This dataset loads the extracted HDF5 files from Phase 1 for self-supervised
pre-training. It provides efficient random access to images with optional
augmentations suitable for SSL.
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from pathlib import Path


class SSLCloudDataset(Dataset):
    """
    Dataset for self-supervised learning from Phase 1 extracted images.

    Loads images from HDF5 files created by scripts/extract_all_images.py.
    Supports data augmentation for SSL pre-training.
    """

    def __init__(
        self,
        hdf5_path: str,
        augment: bool = True,
        normalize: bool = True,
        return_metadata: bool = False,
    ):
        """
        Initialize SSL dataset.

        Args:
            hdf5_path: Path to HDF5 file (train.h5 or val.h5)
            augment: Whether to apply augmentations
            normalize: Whether to normalize images to [0, 1]
            return_metadata: Whether to return metadata along with images
        """
        self.hdf5_path = Path(hdf5_path)
        self.augment = augment
        self.normalize = normalize
        self.return_metadata = return_metadata

        if not self.hdf5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")

        # Open HDF5 file to get metadata
        with h5py.File(self.hdf5_path, "r") as f:
            self.n_samples = f["images"].shape[0]
            self.image_shape = f["images"].shape[1:]

            # Get metadata column names if available
            if "columns" in f["metadata"].attrs:
                self.metadata_columns = list(f["metadata"].attrs["columns"])
            else:
                self.metadata_columns = ["flight_idx", "frame_idx", "SZA", "SAA"]

        print(f"Loaded SSL dataset from {self.hdf5_path.name}")
        print(f"  Samples: {self.n_samples:,}")
        print(f"  Image shape: {self.image_shape}")
        print(f"  Augmentation: {self.augment}")

        # Define augmentations for SSL
        # Note: Our images are 1D signals, so we use custom augmentations
        self.augment_enabled = self.augment
        self.transforms = None  # We'll apply augmentations manually

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        """
        Get a single image (and optionally metadata).

        For SSL, we typically return the image twice with different augmentations
        for contrastive learning, but for MAE we just need one view.
        """
        # Open HDF5 file (each worker gets its own handle)
        with h5py.File(self.hdf5_path, "r") as f:
            # Load image
            image = f["images"][idx].astype(np.float32)

            # Load metadata if requested
            if self.return_metadata:
                metadata = f["metadata"][idx].astype(np.float32)

        # Normalize to [0, 1] if requested
        if self.normalize:
            # Use robust normalization (percentile-based to handle outliers)
            p1, p99 = np.percentile(image, [1, 99])
            image = np.clip(image, p1, p99)
            image = (image - p1) / (p99 - p1 + 1e-8)

        # Convert to tensor (W,) -> (1, W)
        image = torch.from_numpy(image).float().unsqueeze(0)  # Add channel dimension

        # Apply augmentations (custom for 1D signals)
        if self.augment_enabled:
            image = self._apply_1d_augmentations(image)

        if self.return_metadata:
            return image, torch.from_numpy(metadata)
        else:
            return image

    def _apply_1d_augmentations(self, image):
        """
        Apply augmentations suitable for 1D signals.

        Args:
            image: (1, W) tensor

        Returns:
            Augmented (1, W) tensor
        """
        # Horizontal flip (reverse the signal)
        if torch.rand(1).item() < 0.5:
            image = torch.flip(image, dims=[1])

        # Random crop and resize (for multi-scale)
        if torch.rand(1).item() < 0.8:
            W = image.shape[1]
            scale = 0.8 + torch.rand(1).item() * 0.2  # Scale between 0.8 and 1.0
            crop_w = int(W * scale)

            # Random crop
            start_idx = torch.randint(0, W - crop_w + 1, (1,)).item()
            cropped = image[:, start_idx : start_idx + crop_w]

            # Resize back to original width using interpolation
            # Shape: (1, W_crop) -> (1, 1, W_crop) for interpolate
            cropped = cropped.unsqueeze(0)  # (1, 1, W_crop)
            resized = torch.nn.functional.interpolate(
                cropped, size=W, mode="linear", align_corners=False
            )
            image = resized.squeeze(0)  # Back to (1, W)

        # Intensity jitter (brightness and contrast)
        if torch.rand(1).item() < 0.8:
            # Brightness
            brightness_factor = 0.7 + torch.rand(1).item() * 0.6  # 0.7 to 1.3
            image = image * brightness_factor

            # Contrast
            mean_val = image.mean()
            contrast_factor = 0.7 + torch.rand(1).item() * 0.6  # 0.7 to 1.3
            image = (image - mean_val) * contrast_factor + mean_val

            # Clamp to valid range
            image = torch.clamp(image, 0, 1)

        return image

    def get_statistics(self, n_samples=1000):
        """
        Compute dataset statistics from a sample.
        Useful for normalization and verification.
        """
        indices = np.random.choice(
            self.n_samples, size=min(n_samples, self.n_samples), replace=False
        )
        indices = np.sort(indices)  # HDF5 requires sorted indices

        with h5py.File(self.hdf5_path, "r") as f:
            sample_images = f["images"][indices]

        stats = {
            "mean": float(np.mean(sample_images)),
            "std": float(np.std(sample_images)),
            "min": float(np.min(sample_images)),
            "max": float(np.max(sample_images)),
            "p01": float(np.percentile(sample_images, 1)),
            "p99": float(np.percentile(sample_images, 99)),
        }

        return stats


class SSLCloudDatasetWithAugmentations(Dataset):
    """
    SSL dataset that returns two augmented views of each image.
    Useful for contrastive learning methods (SimCLR, MoCo, etc.).

    For MAE, use SSLCloudDataset instead (single view is sufficient).
    """

    def __init__(
        self,
        hdf5_path: str,
        normalize: bool = True,
        strong_augment: bool = True,
    ):
        """
        Initialize dual-view SSL dataset.

        Args:
            hdf5_path: Path to HDF5 file
            normalize: Whether to normalize images
            strong_augment: Use stronger augmentations
        """
        self.hdf5_path = Path(hdf5_path)
        self.normalize = normalize

        if not self.hdf5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")

        with h5py.File(self.hdf5_path, "r") as f:
            self.n_samples = f["images"].shape[0]
            self.image_shape = f["images"].shape[1:]

        print(f"Loaded dual-view SSL dataset from {self.hdf5_path.name}")
        print(f"  Samples: {self.n_samples:,}")
        print(f"  Strong augmentation: {strong_augment}")

        # Store augmentation strength
        self.strong_augment = strong_augment

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        """Return two augmented views of the same image."""
        with h5py.File(self.hdf5_path, "r") as f:
            image = f["images"][idx].astype(np.float32)

        # Normalize
        if self.normalize:
            p1, p99 = np.percentile(image, [1, 99])
            image = np.clip(image, p1, p99)
            image = (image - p1) / (p99 - p1 + 1e-8)

        # Convert to tensor
        image = torch.from_numpy(image).float().unsqueeze(0)

        # Create two views with different augmentations
        view1 = self._apply_dual_view_augmentation(image, strength=1)
        view2 = self._apply_dual_view_augmentation(image, strength=2)

        return view1, view2

    def _apply_dual_view_augmentation(self, image, strength=1):
        """Apply augmentation for dual-view SSL (1D signals)."""
        W = image.shape[1]

        # Horizontal flip
        if torch.rand(1).item() < 0.5:
            image = torch.flip(image, dims=[1])

        # Random crop and resize
        if self.strong_augment:
            scale = 0.7 + torch.rand(1).item() * 0.3  # 0.7 to 1.0
        else:
            scale = 0.8 + torch.rand(1).item() * 0.2  # 0.8 to 1.0

        crop_w = int(W * scale)
        start_idx = torch.randint(0, W - crop_w + 1, (1,)).item()
        cropped = image[:, start_idx : start_idx + crop_w]

        # Resize back
        cropped = cropped.unsqueeze(0)
        resized = torch.nn.functional.interpolate(
            cropped, size=W, mode="linear", align_corners=False
        )
        image = resized.squeeze(0)

        # Intensity jitter
        brightness_range = 0.4 if self.strong_augment else 0.2
        brightness = 1.0 + (torch.rand(1).item() - 0.5) * 2 * brightness_range
        image = image * brightness

        contrast_range = 0.4 if self.strong_augment else 0.2
        mean_val = image.mean()
        contrast = 1.0 + (torch.rand(1).item() - 0.5) * 2 * contrast_range
        image = (image - mean_val) * contrast + mean_val

        # Gaussian noise (for strong augmentation)
        if self.strong_augment and strength == 1:
            noise = torch.randn_like(image) * 0.01
            image = image + noise

        image = torch.clamp(image, 0, 1)

        return image


def test_dataset():
    """Test function to verify dataset loading."""
    import matplotlib.pyplot as plt

    # Test with train.h5
    dataset = SSLCloudDataset(
        "data_ssl/images/train.h5",
        augment=True,
        normalize=True,
    )

    print(f"\nDataset length: {len(dataset)}")

    # Get statistics
    print("\nComputing dataset statistics...")
    stats = dataset.get_statistics(n_samples=1000)
    for key, value in stats.items():
        print(f"  {key}: {value:.4f}")

    # Visualize some samples
    print("\nVisualizing samples...")
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.flatten()

    for i in range(8):
        img = dataset[i * 1000]  # Sample every 1000th image
        axes[i].imshow(img.squeeze(), cmap="gray")
        axes[i].set_title(f"Sample {i * 1000}")
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig("data_ssl/images/dataset_samples.png", dpi=150)
    print("Saved visualization to data_ssl/images/dataset_samples.png")

    # Test dual-view dataset
    print("\nTesting dual-view dataset...")
    dual_dataset = SSLCloudDatasetWithAugmentations(
        "data_ssl/images/train.h5",
        normalize=True,
        strong_augment=True,
    )

    view1, view2 = dual_dataset[0]
    print(f"View 1 shape: {view1.shape}")
    print(f"View 2 shape: {view2.shape}")

    print("\n✅ Dataset tests passed!")


if __name__ == "__main__":
    test_dataset()
