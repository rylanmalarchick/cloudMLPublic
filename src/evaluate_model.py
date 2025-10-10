"""
`evaluate_model.py` provides functions to:
 - `evaluate_model`: compute loss, MAE, MSE on a test DataLoader, returning averaged metrics and scaled predictions.
"""

import numpy as np
import torch
import os

from .pytorchmodel import CustomLoss


def calculate_metrics(y_true, y_pred):
    """
    Calculates comprehensive metrics: MAE, MSE, RMSE, MAPE, R-squared, and error statistics.
    """
    errors = y_true - y_pred
    mae = np.mean(np.abs(errors))
    mse = np.mean(errors**2)
    rmse = np.sqrt(mse)
    mape = (
        np.mean(np.abs(errors / y_true)) * 100 if np.all(y_true != 0) else np.nan
    )  # Avoid division by zero

    # Calculate R^2 relative to the 1:1 line
    ss_res = np.sum((y_pred - y_true) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Additional error statistics
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    error_quantiles = np.percentile(errors, [25, 50, 75, 90, 95])

    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "mape": mape,
        "r2": r2,
        "mean_error": mean_error,
        "std_error": std_error,
        "error_25th_percentile": error_quantiles[0],
        "error_median": error_quantiles[1],
        "error_75th_percentile": error_quantiles[2],
        "error_90th_percentile": error_quantiles[3],
        "error_95th_percentile": error_quantiles[4],
    }


def load_calibration_term(calibration_file="calibration_term.txt"):
    """Load the calibration term from a file if it exists."""
    if calibration_file and os.path.exists(calibration_file):
        with open(calibration_file, "r") as f:
            try:
                return float(f.read().strip())
            except (ValueError, TypeError):
                return None
    return None


def evaluate_model_and_get_metrics(
    model,
    test_loader,
    device,
    scaler_Y,
    return_preds=False,
    loss_type="mse_mae",
    loss_alpha=0.6,
    huber_delta=1.0,
    calibration_file="calibration_term.txt",
    prediction_clip_range=None,
):
    """
    Run the trained model on the test DataLoader and compute metrics.

    Loss configuration:
    - loss_type: "mse_mae", "huber", "huber_mae"
    - loss_alpha: weight for first loss component (0-1)
    - huber_delta: threshold for Huber loss

    If return_preds=False: Returns (avg_loss, avg_mae, avg_mse)
    If return_preds=True: Returns (avg_loss, avg_mae, avg_mse, y_true, y_pred, indices)
    """
    model.to(device)
    model.eval()

    calibration_term = load_calibration_term(calibration_file)
    if calibration_term is not None:
        print(f"Loaded calibration term: {calibration_term:.4f} km")

    # Use the same loss as training for consistency
    criterion = CustomLoss(
        loss_type=loss_type, alpha=loss_alpha, huber_delta=huber_delta
    )

    total_loss = 0.0
    all_predictions = []
    all_true = []
    all_indices = []
    all_local_indices = []
    all_y_lower = []
    all_y_upper = []
    num_batches = 0

    with torch.no_grad():
        for (
            images,
            param1,
            param2,
            targets,
            global_indices,
            local_indices,
        ) in test_loader:
            # Move batch data to device
            images, param1, param2, targets = (
                images.to(device),
                param1.to(device),
                param2.to(device),
                targets.to(device),
            )

            outputs, _ = model(images, param1, param2)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            y_pred_scaled = outputs.cpu().numpy()
            y_true_scaled = targets.cpu().numpy()

            if scaler_Y:
                y_pred_unscaled = scaler_Y.inverse_transform(y_pred_scaled).flatten()
                y_true_unscaled = scaler_Y.inverse_transform(y_true_scaled).flatten()
            else:
                y_pred_unscaled = y_pred_scaled.flatten()
                y_true_unscaled = y_true_scaled.flatten()

            # --- NEW: Apply prediction clipping ---
            if prediction_clip_range:
                min_val, max_val = prediction_clip_range
                y_pred_unscaled = np.clip(y_pred_unscaled, min_val, max_val)

            all_true.extend(y_true_unscaled)
            all_predictions.extend(y_pred_unscaled)

            if return_preds:
                all_indices.extend(global_indices.cpu().numpy())
                all_local_indices.extend(local_indices.cpu().numpy())

            if calibration_term is not None:
                y_lower = y_pred_unscaled - calibration_term
                y_upper = y_pred_unscaled + calibration_term
                all_y_lower.extend(y_lower.tolist())
                all_y_upper.extend(y_upper.tolist())

            num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0

    y_true_final = np.array(all_true)
    y_pred_final = np.array(all_predictions)

    metrics = calculate_metrics(y_true_final, y_pred_final)
    metrics["loss"] = avg_loss

    # Add sample size and basic summary
    metrics["num_samples"] = len(y_true_final)
    metrics["prediction_range"] = [
        float(np.min(y_pred_final)),
        float(np.max(y_pred_final)),
    ]
    metrics["true_range"] = [float(np.min(y_true_final)), float(np.max(y_true_final))]

    if return_preds:
        metrics["y_true"] = y_true_final
        metrics["y_pred"] = y_pred_final
        metrics["indices"] = np.array(all_indices)
        metrics["local_indices"] = np.array(all_local_indices)
        if all_y_lower and all_y_upper:
            metrics["y_lower"] = np.array(all_y_lower)
            metrics["y_upper"] = np.array(all_y_upper)

    return metrics
