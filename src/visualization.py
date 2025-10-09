# src/visualization.py

import os
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from matplotlib.ticker import FuncFormatter


def plot_correlation_enhanced(y_true, y_pred, method_name="Method", save_path=None):
    # This function is correct and remains unchanged
    plt.figure(figsize=(10, 8))
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    if len(y_true_clean) > 1 and np.std(y_true_clean) > 0 and np.std(y_pred_clean) > 0:
        r_pearson = stats.pearsonr(y_true_clean, y_pred_clean)[0]
        f"{r_pearson**2:.3f}"
        q1_true, q3_true = np.percentile(y_true_clean, [15, 90])
        q1_pred, q3_pred = np.percentile(y_pred_clean, [15, 90])
        iqr_true = q3_true - q1_true
        iqr_pred = q3_pred - q1_pred
        outlier_mask = (
            (y_true_clean < q1_true - 2.5 * iqr_true)
            | (y_true_clean > q3_true + 2.5 * iqr_true)
            | (y_pred_clean < q1_pred - 2.5 * iqr_pred)
            | (y_pred_clean > q3_pred + 2.5 * iqr_pred)
        )
        y_true_robust = y_true_clean[~outlier_mask]
        y_pred_robust = y_pred_clean[~outlier_mask]
        if (
            len(y_true_robust) > 1
            and np.std(y_true_robust) > 0
            and np.std(y_pred_robust) > 0
        ):
            r_robust = stats.pearsonr(y_true_robust, y_pred_robust)[0]
            f"{r_robust**2:.3f}"
    else:
        outlier_mask = np.zeros_like(y_true_clean, dtype=bool)

    if np.sum(~outlier_mask) > 0:
        plt.scatter(
            y_true_clean[~outlier_mask],
            y_pred_clean[~outlier_mask],
            alpha=0.6,
            s=20,
            c="blue",
            label="Data",
        )
    if np.sum(outlier_mask) > 0:
        plt.scatter(
            y_true_clean[outlier_mask],
            y_pred_clean[outlier_mask],
            alpha=0.8,
            s=30,
            c="red",
            marker="x",
            label="Outliers",
        )

    plt.plot([0, 2], [0, 2], "r--", label="Perfect Prediction")
    plt.xlabel("True Height (km)")
    plt.ylabel("Predicted Height (km)")
    plt.title(f"{method_name}: Correlation Plot")
    plt.xlim(0, 2)
    plt.ylim(0, 2)
    plt.grid(True, alpha=0.3)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.legend()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_positions(
    y_true,
    y_pred,
    y_lower,
    y_upper,
    out_dir,
    model_name,
    timestamp,
    local_indices,
    max_original_index,
    x_axis_data,
    x_label,
):
    """
    Generates a plot showing true and predicted heights against a time-formatted x-axis.
    """
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(16, 6))

    # Helper to format time if needed
    def time_formatter(x, pos):
        hours = int(x // 3600) % 24
        minutes = int((x % 3600) // 60)
        seconds = int(x % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    # Build full x-axis vector
    if x_axis_data is not None and len(x_axis_data) == max_original_index:
        x_full = x_axis_data
    else:
        x_full = np.arange(max_original_index)

    # Create full-length arrays for true and predicted values
    y_true_full = np.full(max_original_index, np.nan, dtype=np.float32)
    y_pred_full = np.full(max_original_index, np.nan, dtype=np.float32)
    y_lower_full = np.full(max_original_index, np.nan, dtype=np.float32)
    y_upper_full = np.full(max_original_index, np.nan, dtype=np.float32)
    # Populate at valid sample indices
    y_true_full[local_indices] = y_true
    y_pred_full[local_indices] = y_pred
    if y_lower is not None and y_upper is not None:
        y_lower_full[local_indices] = y_lower
        y_upper_full[local_indices] = y_upper

    # Plot truth as points only (avoid interpolation artifacts)
    x_truth = x_full[local_indices]
    plt.scatter(x_truth, y_true, c="blue", s=18, alpha=0.7, label="CPL Lidar Truth")

    # Plot predictions as points
    x_pred = x_full[local_indices]
    plt.scatter(x_pred, y_pred, c="red", s=18, alpha=0.8, label="Model Prediction")

    # Confidence interval shading (optional; renders only where predictions exist)
    if y_lower is not None and y_upper is not None:
        # Draw vertical error bars at the predicted points instead of a filled band
        y_err_lower = y_pred - y_lower
        y_err_upper = y_upper - y_pred
        plt.errorbar(
            x_pred,
            y_pred,
            yerr=[y_err_lower, y_err_upper],
            fmt="none",
            ecolor="red",
            alpha=0.25,
            capsize=2,
        )

    plt.title(f"{model_name}: Combined True and Predicted Heights vs. {x_label}")
    plt.xlabel(x_label)
    plt.ylabel("Height (km)")

    if x_axis_data is not None and len(x_axis_data) > 0:
        plt.xlim(np.min(x_axis_data), np.max(x_axis_data))

    plt.ylim(0, 2)
    plt.legend()
    plt.grid(True, alpha=0.5)

    # Apply the custom time formatter to the x-axis
    ax = plt.gca()
    ax.xaxis.set_major_formatter(FuncFormatter(time_formatter))
    fig = plt.gcf()
    fig.autofmt_xdate()  # Rotates and aligns the tick labels nicely

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"combined_positions_absolute_{timestamp}.png"))
    plt.close()


def plot_flight_path(nav_data, local_indices, dataset, save_path):
    # This function is correct and remains unchanged
    lat = nav_data.get("lat")
    lon = nav_data.get("lon")
    if lat is None or lon is None:
        return

    plt.figure(figsize=(8, 8))
    plt.plot(lon, lat, label="Full Flight Path", color="gray", alpha=0.7)

    if local_indices is not None and len(local_indices) > 0:
        valid_indices = local_indices[local_indices < len(lon)]
        plt.scatter(
            lon[valid_indices], lat[valid_indices], c="red", label="Truth Samples", s=15
        )

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Flight Path with Truth Samples Highlighted")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_results(
    model,
    Y_test,
    Y_pred,
    Y_lower,
    Y_upper,
    raw_indices,
    nav_data,
    model_name,
    timestamp,
    dataset,
    output_base_dir=None,
    x_axis_data=None,
    x_label="Camera Frame Index",
):
    """
    Main wrapper to generate and save all evaluation plots.
    """
    out_dir = os.path.join(output_base_dir or "plots", f"{model_name}_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    # 1. Correlation Plot
    corr_dir = os.path.join(out_dir, "plots_correlation")
    plot_correlation_enhanced(
        Y_test,
        Y_pred,
        method_name=model_name,
        save_path=os.path.join(corr_dir, "correlation_enhanced.png"),
    )

    # 2. Time Series / Position Plot
    positions_dir = os.path.join(out_dir, "plots_positions")
    max_index = dataset.flight_data[0]["n_samples"] if dataset.flight_data else 0
    plot_positions(
        Y_test,
        Y_pred,
        Y_lower,
        Y_upper,
        positions_dir,
        model_name,
        timestamp,
        raw_indices,
        max_index,
        x_axis_data=x_axis_data,  # Pass the timestamps here
        x_label=x_label,
    )

    # 3. Flight Path Plot
    path_dir = os.path.join(out_dir, "plots_path")
    plot_flight_path(
        nav_data,
        raw_indices,
        dataset,
        save_path=os.path.join(path_dir, "flight_path_highlighted.png"),
    )
