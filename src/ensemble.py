# src/ensemble.py

import torch
import numpy as np
import argparse
import os
from tqdm import tqdm
import yaml
from sklearn.preprocessing import StandardScaler

from .pytorchmodel import MultimodalRegressionModel, get_model_config
from .hdf5_dataset import HDF5CloudDataset
from torch.utils.data import DataLoader
from .main_utils import load_all_flights_metadata_for_scalers


def get_test_loader(flight_name, batch_size, config_path="config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    data_dir = config["data_directory"]
    temporal_frames = config["temporal_frames"]
    filter_type = config["filter_type"]
    cbh_min = config.get("cbh_min")
    cbh_max = config.get("cbh_max")
    swath_slice = config.get("swath_slice", [40, 480])
    flight_info = next((f for f in config["flights"] if f["name"] == flight_name), None)
    if flight_info is None:
        raise ValueError(f"Flight '{flight_name}' not found in config.yaml")
    datasets_info = [
        {
            "name": flight_info["name"],
            "iFileName": os.path.join(data_dir, flight_info["iFileName"]),
            "cFileName": os.path.join(data_dir, flight_info["cFileName"]),
            "nFileName": os.path.join(data_dir, flight_info["nFileName"]),
        }
    ]
    global_sza_data, global_saa_data, global_y_data = (
        load_all_flights_metadata_for_scalers(
            datasets_info,
            filter_type=filter_type,
            cbh_min=cbh_min,
            cbh_max=cbh_max,
            swath_slice=swath_slice,
        )
    )
    global_sza_scaler = StandardScaler().fit(global_sza_data)
    global_saa_scaler = StandardScaler().fit(global_saa_data)
    global_y_scaler = StandardScaler().fit(global_y_data)
    test_dataset = HDF5CloudDataset(
        flight_configs=datasets_info,
        augment=False,
        temporal_frames=temporal_frames,
        filter_type=filter_type,
        cbh_min=cbh_min,
        cbh_max=cbh_max,
        swath_slice=swath_slice,
        sza_scaler=global_sza_scaler,
        saa_scaler=global_saa_scaler,
        y_scaler=global_y_scaler,
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader


def get_model_paths(model_dir, model_filter, test_flight):
    all_files = os.listdir(model_dir)
    model_paths = []
    print(f"--- Building ensemble for test flight: {test_flight} ---")
    print(f"Searching in '{model_dir}' for models containing '{model_filter}'...")
    for f in all_files:
        if f.startswith("loo_") and model_filter in f and test_flight not in f:
            model_paths.append(os.path.join(model_dir, f))
    print(f"Found {len(model_paths)} models for the ensemble:")
    for path in model_paths:
        print(f"  - {os.path.basename(path)}")
    if len(model_paths) != 5:
        print(
            f"\nWarning: Expected 5 models for the ensemble, but found {len(model_paths)}. Please check your model files."
        )
    return model_paths


def predict(args):
    model_paths = get_model_paths(args.model_dir, args.model_filter, args.test_flight)
    if not model_paths:
        print("Error: No models found for the LOO ensemble.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    models = []
    model_config = get_model_config(image_shape=(3, 440, 440), temporal_frames=3)
    for path in model_paths:
        model = MultimodalRegressionModel(model_config)

        # --- THIS IS THE CRITICAL FIX ---
        # The error log tells us to set weights_only=False to load untrusted files.
        checkpoint = torch.load(path, map_location=device, weights_only=False)

        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        models.append(model)

    test_loader = get_test_loader(flight_name=args.test_flight, batch_size=1)

    all_preds = []
    with torch.no_grad():
        for inputs, sza_scaled, saa_scaled, _, _, _ in tqdm(
            test_loader, desc=f"Predicting on {args.test_flight}"
        ):
            inputs = inputs.to(device)
            sza_scaled = sza_scaled.to(device)
            saa_scaled = saa_scaled.to(device)

            accumulated_scalar_pred = 0.0
            for model in models:
                pred_map, _ = model(inputs, sza_scaled, saa_scaled)
                scalar_pred = torch.mean(pred_map)
                accumulated_scalar_pred += scalar_pred.item()

            final_ensemble_pred = accumulated_scalar_pred / len(models)
            all_preds.append(final_ensemble_pred)

    output_dir = os.path.join("models", "predictions")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(
        output_dir, f"ensemble_preds_{args.test_flight}_{args.model_filter}.npy"
    )
    np.save(output_path, np.array(all_preds))
    print(f"Ensemble predictions saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ensemble model predictions using LOO strategy"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="models/trained",
        help="Directory where trained models are stored",
    )
    parser.add_argument(
        "--test_flight",
        type=str,
        required=True,
        help='Flight to use for testing (e.g., "10Feb25")',
    )
    parser.add_argument(
        "--model_filter",
        type=str,
        required=True,
        help='Filter to select models (e.g., "combo7")',
    )
    args = parser.parse_args()
    predict(args)
