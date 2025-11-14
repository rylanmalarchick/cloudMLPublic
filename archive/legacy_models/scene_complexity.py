"""
Scene Complexity Module

Computes scene complexity scores based on image and LiDAR features.
Used to correlate model prediction errors with scene difficulty.
"""

import numpy as np
import warnings
import yaml
import os

warnings.filterwarnings("ignore", category=UserWarning)

# Try to import scikit-image features, with fallbacks
try:
    # Only import what we actually use
    SKIMAGE_AVAILABLE = True
except ImportError:
    print(
        "Warning: scikit-image not available. Using simplified complexity calculation."
    )
    SKIMAGE_AVAILABLE = False


def normalize_image(img):
    """Normalize image to 0-1 range for processing."""
    img = img.astype(np.float32)
    img = (img - img.min()) / (np.ptp(img) + 1e-8)
    return img


def img_entropy(img):
    """Compute local entropy using sliding window."""
    if not SKIMAGE_AVAILABLE:
        # Fallback: compute simple entropy (faster)
        hist, _ = np.histogram(img, bins=256, range=(0, 1))
        hist = hist[hist > 0] / hist.sum()
        return -np.sum(hist * np.log2(hist))

    # For speed, use simple entropy instead of sliding window
    hist, _ = np.histogram(img, bins=256, range=(0, 1))
    hist = hist[hist > 0] / hist.sum()
    return -np.sum(hist * np.log2(hist))


def glcm_features(img):
    """Compute GLCM features (contrast, homogeneity, energy)."""
    try:
        # Convert to uint8 for GLCM
        img_uint8 = (img * 255).astype(np.uint8)

        # Try to import GLCM functions
        try:
            from skimage.feature import greycomatrix, greycoprops
        except ImportError:
            # Fallback: compute simple texture metrics
            grad_x = np.gradient(img, axis=1)
            grad_y = np.gradient(img, axis=0)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

            contrast = np.std(gradient_magnitude)
            homogeneity = 1.0 / (1.0 + np.var(img))
            energy = np.mean(img**2)

            return np.array([contrast, homogeneity, energy])

        # Compute GLCM matrix
        glcm = greycomatrix(
            img_uint8,
            distances=[1, 2],
            angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
            symmetric=True,
            normed=True,
        )

        # Extract features
        contrast = greycoprops(glcm, "contrast").mean()
        homogeneity = greycoprops(glcm, "homogeneity").mean()
        energy = greycoprops(glcm, "energy").mean()

        return np.array([contrast, homogeneity, energy])
    except Exception as e:
        print(f"GLCM calculation failed: {e}")
        # Fallback: compute simple texture metrics
        grad_x = np.gradient(img, axis=1)
        grad_y = np.gradient(img, axis=0)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

        contrast = np.std(gradient_magnitude)
        homogeneity = 1.0 / (1.0 + np.var(img))
        energy = np.mean(img**2)

        return np.array([contrast, homogeneity, energy])


def local_contrast(img):
    """Compute local contrast using gradient magnitude."""
    try:
        # Compute gradient magnitude (more reliable than Laplacian)
        grad_x = np.gradient(img, axis=1)
        grad_y = np.gradient(img, axis=0)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        return np.std(gradient_magnitude)
    except Exception as e:
        print(f"Local contrast calculation failed: {e}")
        return 0.0


def image_based_metrics(img_stack):
    """
    Compute image-based complexity metrics for a temporal image stack.

    Args:
        img_stack: numpy array of shape (temporal_frames, height, width)

    Returns:
        dict: Dictionary of image-based metrics
    """
    metrics = {
        "entropy": [],
        "glcm_contrast": [],
        "glcm_homogeneity": [],
        "glcm_energy": [],
        "local_contrast": [],
        "global_contrast": [],
    }

    for t in range(img_stack.shape[0]):
        frame = img_stack[t]
        frame_norm = normalize_image(frame)

        # Compute metrics for this frame
        frame_entropy = img_entropy(frame_norm)
        glcm_feats = glcm_features(frame_norm)
        frame_local_contrast = local_contrast(frame_norm)
        frame_global_contrast = frame_norm.std()

        # Store metrics
        metrics["entropy"].append(frame_entropy)
        metrics["glcm_contrast"].append(glcm_feats[0])
        metrics["glcm_homogeneity"].append(glcm_feats[1])
        metrics["glcm_energy"].append(glcm_feats[2])
        metrics["local_contrast"].append(frame_local_contrast)
        metrics["global_contrast"].append(frame_global_contrast)

    # Average across temporal frames
    avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
    return avg_metrics


def lidar_based_metrics(cloud_layers, backscatter=None):
    """
    Compute LiDAR-based complexity metrics.

    Args:
        cloud_layers: List of tuples (base_height_km, top_height_km) for cloud layers
        backscatter: Optional array of backscatter intensities within cloud layers

    Returns:
        dict: Dictionary of LiDAR-based metrics
    """
    metrics = {}

    # Number of cloud layers
    metrics["num_layers"] = len(cloud_layers) if cloud_layers else 0

    # Vertical characteristics
    if cloud_layers:
        base_heights = [layer[0] for layer in cloud_layers if np.isfinite(layer[0])]
        top_heights = [layer[1] for layer in cloud_layers if np.isfinite(layer[1])]

        if len(base_heights) > 1:
            # spread across multiple layers
            metrics["height_spread"] = float(np.std(base_heights))
            metrics["height_range"] = float(np.ptp(base_heights))
        elif len(base_heights) == 1 and len(top_heights) == 1:
            # only one layer – use its thickness as a proxy for vertical complexity
            metrics["height_spread"] = float(top_heights[0] - base_heights[0])
            metrics["height_range"] = metrics["height_spread"]
        else:
            metrics["height_spread"] = 0.0
            metrics["height_range"] = 0.0
    else:
        metrics["height_spread"] = 0.0
        metrics["height_range"] = 0.0

    # Backscatter variance (if available)
    if backscatter is not None and len(backscatter) > 0:
        metrics["backscatter_variance"] = np.var(backscatter)
        metrics["backscatter_mean"] = np.mean(backscatter)
    else:
        metrics["backscatter_variance"] = 0.0
        metrics["backscatter_mean"] = 0.0

    return metrics


def compute_scene_complexity(
    img_stack, cloud_layers=None, backscatter=None, weights=None, normalize=True
):
    """
    Compute overall scene complexity score.

    Args:
        img_stack: numpy array of shape (temporal_frames, height, width)
        cloud_layers: List of tuples (base_height_km, top_height_km) for cloud layers
        backscatter: Optional array of backscatter intensities
        weights: Optional dict of weights for each metric
        normalize: Whether to normalize features before combining

    Returns:
        tuple: (complexity_score, feature_dict)
    """
    # Compute image-based metrics
    img_metrics = image_based_metrics(img_stack)

    # Compute LiDAR-based metrics
    lidar_metrics = lidar_based_metrics(cloud_layers, backscatter)

    # Combine all metrics
    all_metrics = {**img_metrics, **lidar_metrics}

    # Convert to feature vector
    feature_names = [
        "entropy",
        "glcm_contrast",
        "glcm_homogeneity",
        "glcm_energy",
        "local_contrast",
        "global_contrast",
        "num_layers",
        "height_spread",
        "height_range",
        "backscatter_variance",
        "backscatter_mean",
    ]

    feature_vector = np.array([all_metrics[name] for name in feature_names])

    # Load tuned weights if present
    global _DEFAULT_WEIGHTS
    try:
        if "_DEFAULT_WEIGHTS" not in globals():
            if os.path.exists("complexity_weights.yaml"):
                with open("complexity_weights.yaml", "r") as f:
                    _DEFAULT_WEIGHTS = yaml.safe_load(f)
            else:
                _DEFAULT_WEIGHTS = None
    except Exception:
        _DEFAULT_WEIGHTS = None

    # Default weights (adjusted for better complexity scoring)
    if weights is None:
        if _DEFAULT_WEIGHTS is not None:
            weights = _DEFAULT_WEIGHTS
        else:
            weights = {
                "entropy": 1.5,  # Still important but slightly reduced
                "glcm_contrast": 1.5,  # Texture contrast remains crucial
                "glcm_homogeneity": 0.5,  # Lower weight
                "glcm_energy": 0.5,  # Lower weight
                "local_contrast": 1.5,  # Keep similar to glcm_contrast
                "global_contrast": 1.0,
                "num_layers": 3.0,  # Strongly emphasize multiple layers
                "height_spread": 2.0,  # Vertical variability
                "height_range": 1.5,  # Range of bases
                "backscatter_variance": 1.5,  # Backscatter variability importance
                "backscatter_mean": 0.5,  # Minor influence
            }

    weight_vector = np.array([weights[name] for name in feature_names])

    # Normalize features if requested
    if normalize:
        # Simple feature scaling to [0, 1] using rough empirical bounds.
        bounds = {
            "entropy": (0, 9),
            "glcm_contrast": (0, 0.1),
            "glcm_homogeneity": (0.8, 1.0),
            "glcm_energy": (0, 1.0),
            "local_contrast": (0, 0.1),
            "global_contrast": (0, 0.4),
            "num_layers": (0, 5),
            "height_spread": (0, 5),
            "height_range": (0, 5),
            "backscatter_variance": (0, 0.2),
            "backscatter_mean": (0, 1),
        }
        normalized_features = []
        for name, value in zip(feature_names, feature_vector):
            low, high = bounds.get(name, (0, 1))
            norm_val = (value - low) / (high - low) if high > low else value
            normalized_features.append(norm_val)
        feature_vector = np.array(normalized_features)

    # Compute weighted complexity score
    complexity_score = float(
        (weight_vector * feature_vector).sum() / weight_vector.sum()
    )

    return complexity_score, all_metrics


def batch_compute_complexity(img_stacks, cloud_layers_list, backscatter_list=None):
    """
    Compute complexity scores for a batch of samples.

    Args:
        img_stacks: List of image stacks
        cloud_layers_list: List of cloud layer data
        backscatter_list: Optional list of backscatter data

    Returns:
        tuple: (complexity_scores, feature_dicts)
    """
    complexity_scores = []
    feature_dicts = []

    for i, img_stack in enumerate(img_stacks):
        cloud_layers = cloud_layers_list[i] if i < len(cloud_layers_list) else None
        backscatter = (
            backscatter_list[i]
            if backscatter_list and i < len(backscatter_list)
            else None
        )

        score, features = compute_scene_complexity(img_stack, cloud_layers, backscatter)
        complexity_scores.append(score)
        feature_dicts.append(features)

    return np.array(complexity_scores), feature_dicts


# Convenience function for quick testing
def quick_complexity_test(img_stack):
    """Quick test function that computes complexity with dummy LiDAR data."""
    # Create dummy cloud layers for testing
    dummy_layers = [(2.0, 3.0), (5.0, 6.0)]  # Two cloud layers
    dummy_backscatter = np.random.normal(0.5, 0.2, 100)  # Random backscatter

    score, features = compute_scene_complexity(
        img_stack, dummy_layers, dummy_backscatter
    )
    return score, features
