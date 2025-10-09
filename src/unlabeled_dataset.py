import h5py
import torch
from torch.utils.data import Dataset
import numpy as np


class UnlabeledCloudDataset(Dataset):
    """Dataset for loading cloud imagery without labels for self-supervised learning."""

    def __init__(self, flight_paths, config, transform=None):
        self.flight_paths = flight_paths
        self.config = config
        self.transform = transform
        self.image_indices = self._collect_image_indices()

    def _collect_image_indices(self):
        """Collect all valid image indices from the flight HDF5 files."""
        indices = []
        for flight_path in self.flight_paths:
            try:
                with h5py.File(flight_path, "r") as f:
                    # Correct path to the image dataset in the H5 file
                    num_images = f["Product/Signal"].shape[0]
                    for i in range(num_images):
                        indices.append((flight_path, i))
            except Exception as e:
                print(f"Could not read {flight_path}: {e}")
        return indices

    def __len__(self):
        return len(self.image_indices)

    def __getitem__(self, idx):
        flight_path, img_idx = self.image_indices[idx]
        with h5py.File(flight_path, "r") as f:
            image = f["Product/Signal"][img_idx]

        # Basic preprocessing (e.g., normalization, handling NaNs)
        image = np.nan_to_num(image, nan=0.0)
        image = (image - np.mean(image)) / (np.std(image) + 1e-6)  # Normalize

        if self.transform:
            image = self.transform(image)

        return torch.tensor(image, dtype=torch.float32).unsqueeze(
            0
        )  # Add channel dimension
