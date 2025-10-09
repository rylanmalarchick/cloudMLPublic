import argparse
import numpy as np
import torch
import yaml
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset

from src.hdf5_dataset import HDF5CloudDataset
from src.pipeline import get_device, load_model_and_scalers


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate model for conformal prediction"
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Configuration file path"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model checkpoint",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.9,
        help="Desired confidence level (e.g., 0.9 for 90%)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="calibration_term.txt",
        help="Output file for the calibration term",
    )
    parser.add_argument(
        "--calibration_split",
        type=float,
        default=0.2,
        help="Fraction of training data to use for calibration",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = get_device()

    print(f"Loading model and scalers from {args.model_path}...")
    model, sza_scaler, saa_scaler, y_scaler = load_model_and_scalers(
        args.model_path, config, device
    )
    model.eval()

    print("Identifying training flights for calibration set...")
    train_flight_configs = [
        flight
        for flight in config["flights"]
        if flight["name"] != config.get("validation_flight")
    ]

    if not train_flight_configs:
        raise ValueError("No training flights found to create a calibration set.")

    print(f"Found {len(train_flight_configs)} training flights.")

    full_train_dataset = HDF5CloudDataset(
        flight_configs=train_flight_configs,
        swath_slice=config.get("swath_slice", (40, 480)),
        augment=False,  # No augmentation for calibration
        temporal_frames=config["temporal_frames"],
        sza_scaler=sza_scaler,
        saa_scaler=saa_scaler,
        y_scaler=y_scaler,
    )

    print(f"Full training dataset has {len(full_train_dataset)} samples.")

    _, calib_indices = train_test_split(
        np.arange(len(full_train_dataset)),
        test_size=args.calibration_split,
        random_state=42,
    )

    calibration_dataset = Subset(full_train_dataset, calib_indices)
    print(f"Using {len(calibration_dataset)} samples for calibration.")

    calibration_loader = DataLoader(
        calibration_dataset,
        batch_size=config.get("batch_size", 8),
        shuffle=False,
        num_workers=os.cpu_count() // 2,
        pin_memory=True,
    )

    print("Calculating conformity scores (absolute errors) on calibration set...")
    conformity_scores = []

    with torch.no_grad():
        for img_stack, sza_scaled, saa_scaled, y_scaled, _, _ in tqdm(
            calibration_loader, desc="Calibrating"
        ):
            img_stack, sza_scaled, saa_scaled, y_scaled = (
                img_stack.to(device),
                sza_scaled.to(device),
                saa_scaled.to(device),
                y_scaled.to(device),
            )

            predictions_scaled, _ = model(img_stack, sza_scaled, saa_scaled)

            y_true_unscaled = y_scaler.inverse_transform(y_scaled.cpu().numpy())
            predictions_unscaled = y_scaler.inverse_transform(
                predictions_scaled.cpu().numpy()
            )

            errors = np.abs(y_true_unscaled - predictions_unscaled)
            conformity_scores.extend(errors.flatten())

    alpha = 1.0 - args.confidence
    q_level = np.ceil((len(calibration_dataset) + 1) * (1 - alpha)) / len(
        calibration_dataset
    )
    calibration_term = np.quantile(conformity_scores, q_level)

    print(
        f"\nCalibration term for {args.confidence:.0%} confidence: {calibration_term:.4f} km"
    )

    with open(args.output, "w") as f:
        f.write(str(calibration_term))
    print(f"Calibration term saved to {args.output}")


if __name__ == "__main__":
    main()
