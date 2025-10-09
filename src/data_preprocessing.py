import numpy as np
from skimage.exposure import equalize_adapthist
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


def preprocess_image(data, clip_limit: float = 0.01, zscore: bool = True):
    """Pre-processing pipeline used by the CNN.

    Steps (configurable for ablation):
    1. Scale raw IR data to [0,1]
    2. CLAHE contrast enhancement (clip_limit tunable)
    3. Optional per-image z-score normalisation
    """
    # 1. Min-max scale to [0,1]
    d_min, d_max = data.min(), data.max()
    data_scaled = (data - d_min) / (d_max - d_min + 1e-8)

    # 2. CLAHE
    data_defog = equalize_adapthist(data_scaled, clip_limit=clip_limit)

    # 3. Optional z-score
    if not zscore:
        return data_defog.astype(np.float32)

    m, s = data_defog.mean(), data_defog.std()
    out = (data_defog - m) / (s + 1e-8)
    out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)
    return out.astype(np.float32)


def scale_data(data, scaler=None, fit=False):
    """
    For scalar features (SZA, SAA, heights), fit or apply a zero-mean unit-variance scaler.
    - If fit=True
    - Otherwise: transform using the provided scaler.
    This standardization prevents features with larger ranges from dominating training.
    """
    if fit:
        scaler = StandardScaler()
        return scaler.fit_transform(data), scaler
    return scaler.transform(data)


def linear_vignetting_correction(reference, image):
    """
    Remove linear sensor vignetting by regression.
    1) Fit a linear model: image ≈ a * reference + b, capturing pixel-wise brightness trend.
    2) Compute fitted background = a*reference + b.
    3) Subtract this trend (image - fitted) to flatten gradients.
    4) Add back the median intensity of the original image to preserve overall brightness scale.
    5) Clip to avoid near-zero artifacts.
    """
    h, w = reference.shape
    # Flatten to vectors for regression
    X = reference.flatten().reshape(-1, 1)
    y = image.flatten()
    # Fit linear mapping from reference to raw image values
    reg = LinearRegression().fit(X, y)
    # Reconstruct background (vignetting profile)
    fitted = reg.predict(X).reshape(h, w)
    # Subtract background trend, then restore median brightness
    residual = image - fitted
    corrected = residual + np.median(image)
    # Clip to enforce positive intensities
    return np.clip(corrected, 1e-3, None)


# Add helper for run-length filtering
