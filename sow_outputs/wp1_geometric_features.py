#!/usr/bin/env python3
"""
Work Package 1: Geometric Feature Engineering for CBH Retrieval

This module implements shadow-based geometric feature extraction as specified
in the SOW-AGENT-CBH-WP-001 document.

Key Features:
- Shadow detection using gradient-based edge detection and thresholding
- Cloud-shadow pair identification using solar azimuth projection
- Geometric CBH estimation from shadow length and solar zenith angle
- Confidence scoring for shadow detections
- Robust handling of challenging scenarios (low contrast, multi-layer, broken clouds)

Author: Autonomous Agent
Date: 2025
"""

import sys
from pathlib import Path
import numpy as np
import h5py
import yaml
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import cv2
from scipy import ndimage
from skimage import filters, morphology, measure
import warnings

warnings.filterwarnings("ignore")

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.hdf5_dataset import HDF5CloudDataset


@dataclass
class ShadowDetection:
    """Container for shadow detection results."""

    shadow_length_pixels: float
    shadow_angle_deg: float
    cloud_edge_x: float
    cloud_edge_y: float
    shadow_edge_x: float
    shadow_edge_y: float
    confidence: float
    derived_cbh_km: float
    shadow_mask: Optional[np.ndarray] = None
    cloud_mask: Optional[np.ndarray] = None


class GeometricFeatureExtractor:
    """
    Extracts physics-based geometric features from cloud imagery using shadow analysis.

    Implements the algorithm described in SOW Section 3 for deriving cloud base height
    from shadow geometry and solar angles.
    """

    def __init__(
        self,
        image_scale_m_per_pixel: float = 7.0,  # Empirically calibrated scale factor
        min_shadow_length_pixels: int = 5,
        max_shadow_length_pixels: int = 200,
        edge_threshold: float = 0.1,
        shadow_intensity_threshold: float = 0.3,
        min_confidence: float = 0.3,
        verbose: bool = False,
    ):
        """
        Initialize the geometric feature extractor.

        Args:
            image_scale_m_per_pixel: Conversion factor from pixels to meters (ground sampling distance)
            min_shadow_length_pixels: Minimum valid shadow length
            max_shadow_length_pixels: Maximum valid shadow length
            edge_threshold: Threshold for edge detection (0-1)
            shadow_intensity_threshold: Relative intensity threshold for shadow detection
            min_confidence: Minimum confidence score for valid detections
            verbose: Enable verbose logging
        """
        self.scale_m_per_pixel = image_scale_m_per_pixel
        self.min_shadow_length = min_shadow_length_pixels
        self.max_shadow_length = max_shadow_length_pixels
        self.edge_threshold = edge_threshold
        self.shadow_threshold = shadow_intensity_threshold
        self.min_confidence = min_confidence
        self.verbose = verbose

    def detect_cloud_edges(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect cloud edges using gradient-based methods.

        Args:
            image: Grayscale image (H, W) normalized to [0, 1]

        Returns:
            edge_magnitude: Edge strength map
            edge_direction: Edge direction in radians
        """
        # Apply Gaussian smoothing to reduce noise
        smoothed = cv2.GaussianBlur(image, (5, 5), 1.0)

        # Compute gradients using Sobel operators
        grad_x = cv2.Sobel(smoothed, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(smoothed, cv2.CV_64F, 0, 1, ksize=3)

        # Compute magnitude and direction
        edge_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        edge_direction = np.arctan2(grad_y, grad_x)

        # Normalize magnitude
        if edge_magnitude.max() > 0:
            edge_magnitude = edge_magnitude / edge_magnitude.max()

        return edge_magnitude, edge_direction

    def detect_shadows(self, image: np.ndarray) -> np.ndarray:
        """
        Detect dark shadow regions using adaptive thresholding.

        Args:
            image: Grayscale image (H, W) normalized to [0, 1]

        Returns:
            shadow_mask: Binary mask where 1 indicates shadow regions
        """
        # Convert to uint8 for cv2 operations
        image_uint8 = (image * 255).astype(np.uint8)

        # Apply adaptive thresholding to handle varying illumination
        # Use mean-based adaptive threshold
        adaptive_thresh = cv2.adaptiveThreshold(
            image_uint8,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=31,
            C=10,
        )

        # Also use global thresholding based on relative darkness
        mean_intensity = np.mean(image)
        global_thresh = image < (mean_intensity * self.shadow_threshold)

        # Combine adaptive and global thresholding
        shadow_mask = np.logical_and(adaptive_thresh > 0, global_thresh).astype(
            np.uint8
        )

        # Morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_CLOSE, kernel)
        shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_OPEN, kernel)

        return shadow_mask

    def detect_bright_clouds(self, image: np.ndarray) -> np.ndarray:
        """
        Detect bright cloud regions.

        Args:
            image: Grayscale image (H, W) normalized to [0, 1]

        Returns:
            cloud_mask: Binary mask where 1 indicates cloud regions
        """
        # Detect bright regions (clouds are typically brighter than background)
        mean_intensity = np.mean(image)
        std_intensity = np.std(image)

        # Threshold at mean + 0.5 * std
        cloud_threshold = mean_intensity + 0.5 * std_intensity
        cloud_mask = (image > cloud_threshold).astype(np.uint8)

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        cloud_mask = cv2.morphologyEx(cloud_mask, cv2.MORPH_CLOSE, kernel)

        return cloud_mask

    def find_cloud_shadow_pairs(
        self, image: np.ndarray, saa_deg: float, sza_deg: float
    ) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """
        Identify corresponding cloud-shadow pairs by projecting along solar azimuth.

        Args:
            image: Grayscale image (H, W) normalized to [0, 1]
            saa_deg: Solar Azimuth Angle in degrees
            sza_deg: Solar Zenith Angle in degrees

        Returns:
            List of (cloud_region, shadow_region, confidence) tuples
        """
        # Detect clouds and shadows
        cloud_mask = self.detect_bright_clouds(image)
        shadow_mask = self.detect_shadows(image)

        # Label connected components
        cloud_labels = measure.label(cloud_mask, connectivity=2)
        shadow_labels = measure.label(shadow_mask, connectivity=2)

        cloud_props = measure.regionprops(cloud_labels)
        shadow_props = measure.regionprops(shadow_labels)

        # Convert SAA to projection direction
        # SAA is clockwise from north; we need direction from cloud to shadow
        # In image coordinates: x=column (east), y=row (south)
        # Shadow direction = SAA + 180 degrees (opposite of sun)
        shadow_direction_deg = (saa_deg + 180) % 360
        shadow_direction_rad = np.deg2rad(shadow_direction_deg)

        # Unit vector in shadow direction (image coordinates)
        dx = np.sin(shadow_direction_rad)  # East component
        dy = np.cos(shadow_direction_rad)  # South component (positive y is down)

        pairs = []

        for cloud_region in cloud_props:
            if cloud_region.area < 50:  # Skip small regions
                continue

            cloud_centroid = cloud_region.centroid  # (row, col)

            # Find the edge of the cloud closest to the shadow direction
            cloud_coords = cloud_region.coords  # (N, 2) array of (row, col)

            # Project cloud edge points in shadow direction
            best_shadow = None
            best_distance = float("inf")
            best_confidence = 0.0

            for shadow_region in shadow_props:
                if shadow_region.area < 30:  # Skip small shadows
                    continue

                shadow_centroid = shadow_region.centroid

                # Vector from cloud to shadow
                cloud_to_shadow = np.array(
                    [
                        shadow_centroid[0] - cloud_centroid[0],  # dy (rows)
                        shadow_centroid[1] - cloud_centroid[1],  # dx (cols)
                    ]
                )

                distance = np.linalg.norm(cloud_to_shadow)

                if (
                    distance < self.min_shadow_length
                    or distance > self.max_shadow_length
                ):
                    continue

                # Check if shadow is in the expected direction
                expected_dir = np.array([dy, dx])
                actual_dir = cloud_to_shadow / (distance + 1e-6)

                # Cosine similarity
                alignment = np.dot(expected_dir, actual_dir)

                # Confidence based on alignment and relative sizes
                size_ratio = shadow_region.area / cloud_region.area
                size_confidence = np.clip(size_ratio, 0, 1) * np.clip(
                    1.0 / (size_ratio + 0.1), 0, 1
                )

                confidence = alignment * 0.7 + size_confidence * 0.3

                if confidence > 0.3 and confidence > best_confidence:
                    best_shadow = shadow_region
                    best_confidence = confidence
                    best_distance = distance

            if best_shadow is not None:
                pairs.append((cloud_region, best_shadow, best_confidence))

        return pairs

    def compute_geometric_cbh(
        self, shadow_length_pixels: float, sza_deg: float
    ) -> float:
        """
        Compute cloud base height from shadow length and solar zenith angle.

        Formula: H = L * tan(SZA)

        The shadow length L relates to height H by: L = H / tan(SZA)
        Therefore: H = L * tan(SZA)

        Args:
            shadow_length_pixels: Shadow length in pixels
            sza_deg: Solar Zenith Angle in degrees

        Returns:
            Estimated cloud base height in kilometers
        """
        # Convert shadow length to meters
        shadow_length_m = shadow_length_pixels * self.scale_m_per_pixel

        # Convert SZA to radians
        sza_rad = np.deg2rad(sza_deg)

        # Avoid issues for sun near zenith (SZA < 5°) or near horizon (SZA > 85°)
        if sza_deg < 5.0 or sza_deg > 85.0:
            return np.nan

        # H = L * tan(SZA)
        cbh_m = shadow_length_m * np.tan(sza_rad)

        # Convert to kilometers
        cbh_km = cbh_m / 1000.0

        # Sanity check: reject physically impossible values
        if cbh_km < 0.0 or cbh_km > 10.0:
            return np.nan

        return cbh_km

    def assess_detection_confidence(
        self,
        image: np.ndarray,
        cloud_region,
        shadow_region,
        geometric_alignment: float,
        sza_deg: float,
    ) -> float:
        """
        Assess confidence in shadow detection based on multiple factors.

        Challenges addressed:
        1. Low-contrast surfaces (e.g., ocean) → Check shadow contrast
        2. Multi-layer clouds → Check for ambiguous shadows
        3. Broken cloud fields → Check cloud/shadow coherence

        Args:
            image: Input image
            cloud_region: Cloud region properties
            shadow_region: Shadow region properties
            geometric_alignment: Alignment with expected shadow direction (0-1)
            sza_deg: Solar zenith angle

        Returns:
            Confidence score (0-1)
        """
        confidence_factors = []

        # Factor 1: Geometric alignment (already computed)
        confidence_factors.append(geometric_alignment)

        # Factor 2: Shadow contrast
        shadow_mask = np.zeros_like(image, dtype=bool)
        shadow_coords = shadow_region.coords
        shadow_mask[shadow_coords[:, 0], shadow_coords[:, 1]] = True

        # Get intensity in shadow and surrounding
        shadow_intensity = np.mean(image[shadow_mask])

        # Sample surrounding region (dilate shadow mask)
        dilated_shadow = ndimage.binary_dilation(shadow_mask, iterations=10)
        surrounding_mask = dilated_shadow & ~shadow_mask
        if surrounding_mask.sum() > 0:
            surrounding_intensity = np.mean(image[surrounding_mask])
            contrast = (surrounding_intensity - shadow_intensity) / (
                surrounding_intensity + 1e-6
            )
            contrast_confidence = np.clip(contrast / 0.3, 0, 1)  # Expect 30% darker
        else:
            contrast_confidence = 0.3  # Unknown

        confidence_factors.append(contrast_confidence)

        # Factor 3: Size consistency
        cloud_area = cloud_region.area
        shadow_area = shadow_region.area
        size_ratio = shadow_area / (cloud_area + 1e-6)
        # Expect shadow to be similar size or smaller than cloud
        size_confidence = np.exp(-abs(np.log(size_ratio + 0.1)))
        confidence_factors.append(size_confidence)

        # Factor 4: Solar angle quality
        # Low sun angles (high SZA) are less reliable
        if sza_deg > 75:
            angle_confidence = 0.2
        elif sza_deg > 60:
            angle_confidence = 0.6
        else:
            angle_confidence = 1.0
        confidence_factors.append(angle_confidence)

        # Factor 5: Cloud/shadow shape coherence
        cloud_solidity = cloud_region.solidity
        shadow_solidity = shadow_region.solidity
        solidity_confidence = min(cloud_solidity, shadow_solidity)
        confidence_factors.append(solidity_confidence)

        # Weighted average of confidence factors
        weights = [0.25, 0.25, 0.15, 0.20, 0.15]
        overall_confidence = np.average(confidence_factors, weights=weights)

        return overall_confidence

    def extract_features_from_image(
        self, image: np.ndarray, sza_deg: float, saa_deg: float
    ) -> ShadowDetection:
        """
        Extract geometric features from a single image.

        Args:
            image: Grayscale image (H, W) normalized to [0, 1]
            sza_deg: Solar Zenith Angle in degrees
            saa_deg: Solar Azimuth Angle in degrees

        Returns:
            ShadowDetection object with extracted features
        """
        # Find cloud-shadow pairs
        pairs = self.find_cloud_shadow_pairs(image, saa_deg, sza_deg)

        if len(pairs) == 0:
            # No valid pairs found
            return ShadowDetection(
                shadow_length_pixels=0.0,
                shadow_angle_deg=saa_deg,
                cloud_edge_x=0.0,
                cloud_edge_y=0.0,
                shadow_edge_x=0.0,
                shadow_edge_y=0.0,
                confidence=0.0,
                derived_cbh_km=np.nan,
            )

        # Select the best pair (highest confidence)
        best_pair = max(pairs, key=lambda p: p[2])
        cloud_region, shadow_region, geometric_confidence = best_pair

        # Compute shadow length
        cloud_centroid = cloud_region.centroid
        shadow_centroid = shadow_region.centroid

        shadow_length_pixels = np.sqrt(
            (shadow_centroid[0] - cloud_centroid[0]) ** 2
            + (shadow_centroid[1] - cloud_centroid[1]) ** 2
        )

        # Compute geometric CBH
        derived_cbh_km = self.compute_geometric_cbh(shadow_length_pixels, sza_deg)

        # Assess overall confidence
        confidence = self.assess_detection_confidence(
            image, cloud_region, shadow_region, geometric_confidence, sza_deg
        )

        # Create shadow detection result
        detection = ShadowDetection(
            shadow_length_pixels=shadow_length_pixels,
            shadow_angle_deg=saa_deg,
            cloud_edge_x=cloud_centroid[1],  # col
            cloud_edge_y=cloud_centroid[0],  # row
            shadow_edge_x=shadow_centroid[1],  # col
            shadow_edge_y=shadow_centroid[0],  # row
            confidence=confidence,
            derived_cbh_km=derived_cbh_km
            if confidence > self.min_confidence
            else np.nan,
        )

        return detection


def load_dataset_with_metadata(config_path: str) -> Tuple[HDF5CloudDataset, List[Dict]]:
    """
    Load the full dataset with all flights and extract metadata.

    Args:
        config_path: Path to config YAML file

    Returns:
        dataset: HDF5CloudDataset instance
        metadata_list: List of metadata dicts for each sample
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Load dataset with all flights
    dataset = HDF5CloudDataset(
        flight_configs=config["flights"],
        indices=None,  # Load all samples
        augment=False,  # No augmentation for feature extraction
        swath_slice=config.get("swath_slice", [40, 480]),
    )

    # Extract metadata for each sample
    # Dataset returns: (img_stack, sza_tensor, saa_tensor, y_scaled, global_idx, local_idx)
    metadata_list = []
    for idx in range(len(dataset)):
        img_stack, sza_tensor, saa_tensor, y_scaled, global_idx, local_idx = dataset[
            idx
        ]

        # Get unscaled values directly from flight data
        flight_idx, _ = dataset.global_to_local[int(global_idx)]
        flight_info = dataset.flight_data[flight_idx]

        # Get raw (unscaled) SZA and SAA values
        sza_raw = flight_info["SZA_full"][local_idx, 0]
        saa_raw = flight_info["SAA_full"][local_idx, 0]
        cbh_km = flight_info["Y_full"][local_idx].item()

        metadata = {
            "index": idx,
            "sza_deg": sza_raw,
            "saa_deg": saa_raw,
            "cbh_km": cbh_km,
        }
        metadata_list.append(metadata)

    return dataset, metadata_list


def extract_all_geometric_features(
    config_path: str,
    output_path: str,
    image_scale_m_per_pixel: float = 7.0,
    verbose: bool = True,
) -> Dict:
    """
    Extract geometric features for all 933 labeled samples.

    Args:
        config_path: Path to configuration YAML
        output_path: Path to save output HDF5 file
        image_scale_m_per_pixel: Ground sampling distance
        verbose: Enable progress logging

    Returns:
        Dictionary with extraction statistics
    """
    if verbose:
        print("=" * 80)
        print("Work Package 1: Geometric Feature Engineering")
        print("=" * 80)
        print(f"\nConfiguration: {config_path}")
        print(f"Output: {output_path}")
        print(f"Image scale: {image_scale_m_per_pixel} m/pixel")

    # Initialize feature extractor
    extractor = GeometricFeatureExtractor(
        image_scale_m_per_pixel=image_scale_m_per_pixel, verbose=verbose
    )

    # Load dataset
    if verbose:
        print("\nLoading dataset...")
    dataset, metadata_list = load_dataset_with_metadata(config_path)
    n_samples = len(dataset)

    if verbose:
        print(f"Loaded {n_samples} samples")

    # Initialize feature arrays
    features = {
        "sample_id": np.arange(n_samples),
        "derived_geometric_H": np.full(n_samples, np.nan),
        "shadow_length_pixels": np.zeros(n_samples),
        "shadow_detection_confidence": np.zeros(n_samples),
        "cloud_edge_x": np.zeros(n_samples),
        "cloud_edge_y": np.zeros(n_samples),
        "shadow_edge_x": np.zeros(n_samples),
        "shadow_edge_y": np.zeros(n_samples),
        "shadow_angle_deg": np.zeros(n_samples),
        "sza_deg": np.zeros(n_samples),
        "saa_deg": np.zeros(n_samples),
        "true_cbh_km": np.zeros(n_samples),
    }

    # Extract features for each sample
    if verbose:
        print("\nExtracting geometric features...")
        print("-" * 80)

    successful = 0
    high_confidence = 0

    for idx in range(n_samples):
        if verbose and (idx % 100 == 0 or idx == n_samples - 1):
            print(
                f"Processing sample {idx + 1}/{n_samples} ({100 * (idx + 1) / n_samples:.1f}%)"
            )

        # Get sample
        img_stack, sza_tensor, saa_tensor, y_scaled, global_idx, local_idx = dataset[
            idx
        ]

        # Use center frame from temporal stack
        image = img_stack[1, :, :].numpy()  # Center frame, shape (H, W)

        # Normalize to [0, 1] if needed
        if image.max() > 1.0:
            image = image / 255.0

        # Get metadata
        meta = metadata_list[idx]
        sza_deg = meta["sza_deg"]
        saa_deg = meta["saa_deg"]
        cbh_km = meta["cbh_km"]

        # Extract features
        detection = extractor.extract_features_from_image(image, sza_deg, saa_deg)

        # Store features
        features["derived_geometric_H"][idx] = detection.derived_cbh_km
        features["shadow_length_pixels"][idx] = detection.shadow_length_pixels
        features["shadow_detection_confidence"][idx] = detection.confidence
        features["cloud_edge_x"][idx] = detection.cloud_edge_x
        features["cloud_edge_y"][idx] = detection.cloud_edge_y
        features["shadow_edge_x"][idx] = detection.shadow_edge_x
        features["shadow_edge_y"][idx] = detection.shadow_edge_y
        features["shadow_angle_deg"][idx] = detection.shadow_angle_deg
        features["sza_deg"][idx] = sza_deg
        features["saa_deg"][idx] = saa_deg
        features["true_cbh_km"][idx] = cbh_km

        # Track statistics
        if detection.confidence > 0.0:
            successful += 1
        if detection.confidence > 0.5:
            high_confidence += 1

    # Save to HDF5
    if verbose:
        print("\n" + "-" * 80)
        print(f"Saving features to {output_path}")

    with h5py.File(output_path, "w") as hf:
        # Save each feature array
        for key, value in features.items():
            hf.create_dataset(key, data=value, compression="gzip", compression_opts=4)

        # Save metadata
        hf.attrs["n_samples"] = n_samples
        hf.attrs["image_scale_m_per_pixel"] = image_scale_m_per_pixel
        hf.attrs["successful_detections"] = successful
        hf.attrs["high_confidence_detections"] = high_confidence

    # Compute statistics
    stats = {
        "n_samples": n_samples,
        "successful_detections": successful,
        "high_confidence_detections": high_confidence,
        "success_rate": successful / n_samples,
        "high_confidence_rate": high_confidence / n_samples,
        "mean_confidence": np.mean(features["shadow_detection_confidence"]),
        "valid_cbh_estimates": np.sum(~np.isnan(features["derived_geometric_H"])),
    }

    if verbose:
        print("\n" + "=" * 80)
        print("Extraction Complete!")
        print("=" * 80)
        print(f"Total samples: {stats['n_samples']}")
        print(
            f"Successful detections: {stats['successful_detections']} ({stats['success_rate'] * 100:.1f}%)"
        )
        print(
            f"High confidence (>0.5): {stats['high_confidence_detections']} ({stats['high_confidence_rate'] * 100:.1f}%)"
        )
        print(f"Mean confidence: {stats['mean_confidence']:.3f}")
        print(f"Valid CBH estimates: {stats['valid_cbh_estimates']}")
        print("=" * 80)

    return stats


def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="WP-1: Extract geometric features from cloud imagery"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/bestComboConfig.yaml",
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="sow_outputs/wp1_geometric/WP1_Features.hdf5",
        help="Output path for feature HDF5 file",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=50.0,
        help="Image scale in meters per pixel (ground sampling distance)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    # Run extraction
    stats = extract_all_geometric_features(
        config_path=args.config,
        output_path=args.output,
        image_scale_m_per_pixel=args.scale,
        verbose=args.verbose,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
