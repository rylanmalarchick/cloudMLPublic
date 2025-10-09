# in src/main_utils.py

import os
import torch
import yaml
import numpy as np
from sklearn.preprocessing import StandardScaler

from .hdf5_dataset import HDF5CloudDataset

"""
main_utils: Helper functions to configure the environment and prepare training data.
Provides both an in-memory loader (`prepare_data`) for small datasets and a
streaming HDF5-based loader (`prepare_streaming_data`) for large flight archives.
Each returns PyTorch DataLoaders that yield tuples of (IR image tensor,
solar angles, azimuth angles, and target cloud height).
"""

# --- NEW AND RESTORED FUNCTIONS ---


def get_device():
    """Returns the available device (CUDA or CPU)."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_config(config_path):
    """Loads a YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def save_model_and_scaler(model, scaler, path):
    """Saves the model state and scaler object to a file."""
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    # Ensure model is on CPU before saving to avoid GPU-specific info
    model.to("cpu")
    torch.save({"model_state_dict": model.state_dict(), "scaler": scaler}, path)
    # Move model back to the original device if needed
    model.to(get_device())


def load_model_and_scaler(path, device):
    """Loads the model state and scaler object from a file."""
    # This import is here to avoid circular dependencies
    from .pytorchmodel import MultimodalRegressionModel, get_model_config

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    scaler = checkpoint.get("scaler", None)  # Use .get for backward compatibility
    if scaler is None:
        raise ValueError("Scaler not found in checkpoint. Cannot proceed.")

    # A more robust way to get model config might be needed if it changes
    # For now, we use a dummy config based on the expected data shape
    dummy_config = get_model_config(image_shape=(3, 440, 440), temporal_frames=3)
    model = MultimodalRegressionModel(dummy_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    return model, scaler


# --- YOUR ORIGINAL FUNCTIONS (UNCHANGED) ---


def setup_environment(hpc_mode=False):
    """
    Set up OS variables for CUDA debugging and determine compute device.
    Prints CUDA version and device info, then returns a torch.device
    ('cuda' or 'cpu') for downstream model training and evaluation.
    """
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    try:
        print(f"torch.version.cuda: {torch.version.cuda}")
        print(f"torch.cuda.is_available: {torch.cuda.is_available()}")
        print(f"torch.cuda.device_count: {torch.cuda.device_count()}")
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            print(f"torch.cuda.get_device_name(0): {torch.cuda.get_device_name(0)}")
    except Exception as e:
        print(f"CUDA diagnostics failed: {e}")

    if hpc_mode and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        print("cuDNN benchmark mode enabled for HPC.")
    else:
        torch.backends.cudnn.benchmark = False

    print(f"CUDA_HOME: {os.environ.get('CUDA_HOME')}")
    print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH')}")
    return device


def prepare_streaming_data(
    flight_config_or_configs,
    swath_slice=(40, 480),
    temporal_frames=3,
    filter_type="basic",
    cbh_min=None,
    cbh_max=None,
    sza_scaler=None,
    saa_scaler=None,
    y_scaler=None,
    augment=False,
    flat_field_correction=True,
    clahe_clip_limit=0.01,
    zscore_normalize=True,
    angles_mode: str = "both",
):
    """
    Creates and returns a fully configured HDF5CloudDataset.
    """
    if isinstance(flight_config_or_configs, dict):
        flight_configs = [flight_config_or_configs]
    elif isinstance(flight_config_or_configs, list):
        flight_configs = flight_config_or_configs
    else:
        raise ValueError(
            "Input must be a flight config dict or a list of flight configs"
        )

    dataset = HDF5CloudDataset(
        flight_configs,
        indices=None,
        swath_slice=swath_slice,
        temporal_frames=temporal_frames,
        filter_type=filter_type,
        cbh_min=cbh_min,
        cbh_max=cbh_max,
        sza_scaler=sza_scaler,
        saa_scaler=saa_scaler,
        y_scaler=y_scaler,
        augment=augment,
        flat_field_correction=flat_field_correction,
        clahe_clip_limit=clahe_clip_limit,
        zscore_normalize=zscore_normalize,
        angles_mode=angles_mode,
    )

    print(
        f"DEBUG prepare_streaming_data: Final CPL filtered samples = {len(dataset.indices)}"
    )
    return dataset


def prepare_full_dataset_for_evaluation(
    flight_config_or_configs,
    temporal_frames=3,
    filter_type="basic",
    cbh_min=None,
    cbh_max=None,
    swath_slice=(40, 480),
    sza_scaler=None,
    saa_scaler=None,
    y_scaler=None,
    angles_mode: str = "both",
):
    """
    Prepare dataset for evaluation - returns ALL CPL samples without train/test split.
    """
    if isinstance(flight_config_or_configs, dict):
        flight_configs = [flight_config_or_configs]
    elif isinstance(flight_config_or_configs, list):
        flight_configs = flight_config_or_configs
    else:
        raise ValueError(
            "Input must be a flight config dict or a list of flight configs"
        )

    print(
        f"\n=== Preparing full dataset for evaluation: {[c['name'] for c in flight_configs]} ==="
    )

    dataset = HDF5CloudDataset(
        flight_configs,
        indices=None,
        swath_slice=swath_slice,
        temporal_frames=temporal_frames,
        filter_type=filter_type,
        cbh_min=cbh_min,
        cbh_max=cbh_max,
        sza_scaler=sza_scaler,
        saa_scaler=saa_scaler,
        y_scaler=y_scaler,
        angles_mode=angles_mode,
    )

    print(f"DEBUG prepare_full_dataset: Final CPL filtered samples = {len(dataset)}")
    return dataset.image_shape, dataset.y_scaler, dataset


def load_all_flights_metadata_for_scalers(
    datasets_info,
    swath_slice=(40, 480),
    filter_type="basic",
    cbh_min=None,
    cbh_max=None,
):
    """
    Loads metadata from all specified flights to collect global SZA, SAA, and Y_valid data
    for unified scaler fitting.
    """
    all_sza, all_saa, all_y_valid = [], [], []

    temp_dataset = HDF5CloudDataset(
        flight_configs=[datasets_info[0]],
        swath_slice=swath_slice,
        filter_type=filter_type,
        cbh_min=cbh_min,
        cbh_max=cbh_max,
        sza_scaler=StandardScaler(),
        saa_scaler=StandardScaler(),
        y_scaler=StandardScaler(),
    )
    temp_dataset.sza_scaler_pre = None
    temp_dataset.saa_scaler_pre = None
    temp_dataset.y_scaler_pre = None

    print("Phase 0: Collecting metadata for global scaler fitting...")
    for i, config in enumerate(datasets_info):
        try:
            print(f"  Loading flight {config['name']} metadata...")
            flight_info = temp_dataset._load_single_flight_metadata(config, i)
            all_sza.append(flight_info["SZA_full"])
            all_saa.append(flight_info["SAA_full"])
            all_y_valid.append(flight_info["y_valid"])
            print(f"    Success: {flight_info['n_samples']} samples")
        except Exception as e:
            print(f"    Error loading flight {config['name']} metadata: {e}")
            continue

    if not all_sza:
        raise ValueError("No valid flight metadata loaded for scaler fitting.")

    global_sza = np.vstack(all_sza)
    global_saa = np.vstack(all_saa)
    global_y_valid = np.vstack(all_y_valid)

    print(
        f"  Global metadata shapes: SZA={global_sza.shape}, SAA={global_saa.shape}, Y={global_y_valid.shape}"
    )
    return global_sza, global_saa, global_y_valid
