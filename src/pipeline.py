# in src/pipeline.py

import os
import torch
import pandas as pd
import h5py  # Added for navigation data loading
import json
from torch.utils.data import DataLoader, WeightedRandomSampler

from .main_utils import prepare_streaming_data
from .pytorchmodel import get_model_config, get_model_class
from .train_model import train_model
from .evaluate_model import evaluate_model_and_get_metrics
from .visualization import plot_results


def run_pretraining(
    config,
    datasets_info,
    device,
    hpc_settings,
    save_suffix,
    sza_scaler,
    saa_scaler,
    y_scaler,
    log_dir,
):
    """Handles the pre-training phase on a specified flight."""
    if config.get("no_pretrain", False):
        print("\n--- Skipping Pre-training ---")
        return None

    pretrain_flight_name = config["pretrain_flight"]
    print(f"\n--- Pretraining on {pretrain_flight_name} ---")

    pretrain_flight_config = next(
        (f for f in datasets_info if f["name"] == pretrain_flight_name), None
    )
    if not pretrain_flight_config:
        raise ValueError(
            f"Pre-training flight '{pretrain_flight_name}' not found in dataset configs."
        )

    dataset = prepare_streaming_data(
        pretrain_flight_config,
        swath_slice=tuple(config["swath_slice"]),
        temporal_frames=config["temporal_frames"],
        sza_scaler=sza_scaler,
        saa_scaler=saa_scaler,
        y_scaler=y_scaler,
        augment=config.get("augment", True),
        flat_field_correction=config.get("flat_field_correction", True),
        clahe_clip_limit=config.get("clahe_clip_limit", 0.01),
        zscore_normalize=config.get("zscore_normalize", True),
        angles_mode=config.get("angles_mode", "both"),
    )

    val_split = 0.2
    indices = list(range(len(dataset)))
    split_idx = int(len(indices) * (1 - val_split))
    train_indices, val_indices = indices[:split_idx], indices[split_idx:]

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)

    # --- CORRECTED HPC DATALOADER SETTINGS ---
    # The hpc_settings dictionary is now directly passed to the DataLoader.
    # The 'hpc_mode' key is removed before passing to avoid errors.
    hpc_loader_settings = hpc_settings.copy()
    hpc_loader_settings.pop("hpc_mode", None)

    train_loader = DataLoader(train_dataset, shuffle=True, **hpc_loader_settings)
    val_loader = DataLoader(val_dataset, shuffle=False, **hpc_loader_settings)

    model_config = get_model_config(dataset.image_shape, config["temporal_frames"])
    model_config["use_spatial_attention"] = config.get("use_spatial_attention", True)
    model_config["use_temporal_attention"] = config.get("use_temporal_attention", True)
    model_class = get_model_class(
        config.get("architecture", {}).get("name", "transformer")
    )
    model = model_class(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    pretrain_save_name = f"pretrain_{pretrain_flight_name}{save_suffix}"
    pretrain_save_path = os.path.join("models", "trained", f"{pretrain_save_name}.pth")
    pretrain_log_dir = os.path.join(log_dir, "tensorboard")

    pretrain_config = config.copy()
    pretrain_config["save_name"] = pretrain_save_name

    model_pre, _ = train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        pretrain_config,
        pretrain_save_path,
        pretrain_log_dir,
        scaler={"sza": sza_scaler, "saa": saa_scaler, "y": y_scaler},
    )

    print(f"→ Saved model + scaler to {pretrain_save_path}")
    return pretrain_save_path


def run_final_training_and_evaluation(
    config,
    datasets_info,
    pre_ckpt,
    device,
    hpc_settings,
    save_suffix,
    sza_scaler,
    saa_scaler,
    y_scaler,
    log_dir,
):
    """Handles final model training and evaluation plotting."""
    if config.get("no_final", False):
        print("\n--- Skipping Final Training & Evaluation ---")
        return

    print("\n--- Final full-flight training ---")

    val_flight_name = config["validation_flight"]
    train_configs = [f for f in datasets_info if f["name"] != val_flight_name]
    val_config = next((f for f in datasets_info if f["name"] == val_flight_name), None)

    print(f"  Using {val_flight_name} as validation flight.")
    os.makedirs(os.path.join(log_dir, "csv"), exist_ok=True)

    train_dataset = prepare_streaming_data(
        train_configs,
        sza_scaler=sza_scaler,
        saa_scaler=saa_scaler,
        y_scaler=y_scaler,
        augment=config.get("augment", True),
        swath_slice=tuple(config["swath_slice"]),
        temporal_frames=config["temporal_frames"],
        filter_type=config.get("filter_type", "basic"),
        cbh_min=config.get("cbh_min"),
        cbh_max=config.get("cbh_max"),
        flat_field_correction=config.get("flat_field_correction", True),
        clahe_clip_limit=config.get("clahe_clip_limit", 0.01),
        zscore_normalize=config.get("zscore_normalize", True),
        angles_mode=config.get("angles_mode", "both"),
    )
    val_dataset = prepare_streaming_data(
        val_config,
        sza_scaler=sza_scaler,
        saa_scaler=saa_scaler,
        y_scaler=y_scaler,
        augment=False,
        swath_slice=tuple(config["swath_slice"]),
        temporal_frames=config["temporal_frames"],
        filter_type=config.get("filter_type", "basic"),
        cbh_min=config.get("cbh_min"),
        cbh_max=config.get("cbh_max"),
        flat_field_correction=config.get("flat_field_correction", True),
        clahe_clip_limit=config.get("clahe_clip_limit", 0.01),
        zscore_normalize=config.get("zscore_normalize", True),
        angles_mode=config.get("angles_mode", "both"),
    )

    pretrain_flight_name = config["pretrain_flight"]
    pretrain_flight_index = next(
        (i for i, f in enumerate(train_configs) if f["name"] == pretrain_flight_name),
        -1,
    )

    weights = [
        (
            config.get("overweight_factor", 3.0)
            if train_dataset.global_to_local[i][0] == pretrain_flight_index
            else 1.0
        )
        for i in range(len(train_dataset))
    ]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    hpc_loader_settings = hpc_settings.copy()
    hpc_loader_settings.pop("hpc_mode", None)
    train_loader = DataLoader(
        train_dataset, sampler=sampler, **hpc_loader_settings, drop_last=True
    )
    val_loader = DataLoader(val_dataset, shuffle=False, **hpc_loader_settings)

    model_config = get_model_config(
        train_dataset.image_shape, config["temporal_frames"]
    )
    model_config["use_spatial_attention"] = config.get("use_spatial_attention", True)
    model_config["use_temporal_attention"] = config.get("use_temporal_attention", True)
    model_class = get_model_class(
        config.get("architecture", {}).get("name", "transformer")
    )
    model = model_class(model_config).to(device)

    if pre_ckpt:
        print(f"Initialized model from {pre_ckpt}")
        model.load_state_dict(
            torch.load(pre_ckpt, weights_only=False)["model_state_dict"]
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    final_save_name = f"final_overweighted{save_suffix}"
    final_save_path = os.path.join("models", "trained", f"{final_save_name}.pth")
    final_log_dir = os.path.join(log_dir, "tensorboard")

    final_config = config.copy()
    final_config["save_name"] = final_save_name

    final_model, _ = train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        final_config,
        final_save_path,
        final_log_dir,
        scaler={"sza": sza_scaler, "saa": saa_scaler, "y": y_scaler},
    )
    print(f"→ Saved final overweighted model to {final_save_path}")

    # Extract final evaluation results for ablation study
    print("\n--- Final Model Evaluation ---")
    metrics = evaluate_model_and_get_metrics(
        final_model, val_loader, device, y_scaler, return_preds=True
    )
    print(
        f"Results: Loss={metrics['loss']:.4f}, MAE={metrics['mae']:.4f}, MSE={metrics['mse']:.4f}, RMSE={metrics.get('rmse', 'N/A'):.4f}, R²={metrics['r2']:.4f}"
    )

    # Save metrics summary to JSON
    metrics_summary = {
        k: v for k, v in metrics.items() if not isinstance(v, np.ndarray)
    }  # Exclude arrays
    metrics_json_path = os.path.join(log_dir, "csv", f"metrics_{final_save_name}.json")
    with open(metrics_json_path, "w") as f:
        json.dump(metrics_summary, f, indent=4)
    print(f"→ Saved metrics summary to {metrics_json_path}")

    # Save predictions to CSV
    predictions_df = pd.DataFrame(
        {
            "y_true": metrics["y_true"],
            "y_pred": metrics["y_pred"],
            "local_indices": metrics["local_indices"],
            "flight_name": val_config[
                "name"
            ],  # Add flight name for merging with complexity data
        }
    )
    predictions_csv_path = os.path.join(
        log_dir, "csv", f"predictions_{final_save_name}.csv"
    )
    predictions_df.to_csv(predictions_csv_path, index=False)
    print(f"→ Saved predictions to {predictions_csv_path}")

    if not config.get("no_plots", False):
        print("\n--- Generating plots for final model ---")

        # Ensure the validation dataset is the HDF5CloudDataset, not a Subset
        if isinstance(val_dataset, torch.utils.data.Subset):
            plot_dataset = val_dataset.dataset
        else:
            plot_dataset = val_dataset

        # Extract evaluation results
        metrics = evaluate_model_and_get_metrics(
            final_model, val_loader, device, y_scaler, return_preds=True
        )
        Y_test = metrics["y_true"]
        Y_pred = metrics["y_pred"]
        raw_indices = metrics["local_indices"]

        # Safely load navigation data for the specific validation flight
        nav_data = {"lat": None, "lon": None}
        try:
            val_flight_info = plot_dataset.flight_data[
                0
            ]  # Assumes single validation flight
            print(
                f"DEBUG: val_flight_info keys: {val_flight_info.keys()}"
            )  # Debug print
            nav_file_path = os.path.join(
                config["data_directory"], val_flight_info["nFileName"]
            )
            with h5py.File(nav_file_path, "r") as nf:
                try:
                    nav_data["lat"] = nf["nav/IWG_lat"][:]
                    nav_data["lon"] = nf["nav/IWG_lon"][:]
                except KeyError as ke:
                    print(f"Warning: Could not load navigation data for plotting: {ke}")
        except (KeyError, FileNotFoundError) as e:
            print(f"Warning: Could not load navigation data for plotting: {e}")

        # Get timestamp for plot folder naming
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

        plot_results(
            model=final_model,
            Y_test=Y_test,
            Y_pred=Y_pred,
            raw_indices=raw_indices,
            nav_data=nav_data,
            model_name=final_save_name,
            timestamp=timestamp,
            dataset=plot_dataset,
            output_base_dir="plots",
        )


def run_loo_evaluation(config, all_flight_configs, scaler_info, hpc_settings):
    """
    Performs Leave-One-Out (LOO) cross-validation.
    """
    print("\n--- Running Leave-One-Out (LOO) training ---")
    loo_results = []
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for i, hold_out_flight in enumerate(all_flight_configs):
        hold_out_name = hold_out_flight["name"]
        print(
            f"\n--- LOO: Hold-out {hold_out_name} ({i + 1}/{len(all_flight_configs)}) ---"
        )

        train_configs = [f for f in all_flight_configs if f["name"] != hold_out_name]

        print(f"Training on: {[c['name'] for c in train_configs]}")
        print(f"Validating on: {hold_out_name}")

        # Fit global scalers excluding hold-out flight
        from src.main_utils import load_all_flights_metadata_for_scalers

        global_sza, global_saa, global_y = load_all_flights_metadata_for_scalers(
            train_configs,
            filter_type=config.get("filter_type", "basic"),
            cbh_min=config.get("cbh_min"),
            cbh_max=config.get("cbh_max"),
            swath_slice=tuple(config["swath_slice"]),
        )
        from sklearn.preprocessing import StandardScaler

        sza_scaler_local = StandardScaler().fit(global_sza)
        saa_scaler_local = StandardScaler().fit(global_saa)
        y_scaler_local = StandardScaler().fit(global_y)

        train_dataset = prepare_streaming_data(
            train_configs,
            sza_scaler=sza_scaler_local,
            saa_scaler=saa_scaler_local,
            y_scaler=y_scaler_local,
            augment=config.get("augment", True),
            swath_slice=tuple(config["swath_slice"]),
            temporal_frames=config["temporal_frames"],
            filter_type=config.get("filter_type", "basic"),
            cbh_min=config.get("cbh_min"),
            cbh_max=config.get("cbh_max"),
            angles_mode=config.get("angles_mode", "both"),
        )
        val_dataset = prepare_streaming_data(
            hold_out_flight,
            sza_scaler=sza_scaler_local,
            saa_scaler=saa_scaler_local,
            y_scaler=y_scaler_local,
            augment=False,
            swath_slice=tuple(config["swath_slice"]),
            temporal_frames=config["temporal_frames"],
            filter_type=config.get("filter_type", "basic"),
            cbh_min=config.get("cbh_min"),
            cbh_max=config.get("cbh_max"),
            angles_mode=config.get("angles_mode", "both"),
        )

        # Apply HPC settings to DataLoader
        loader_settings = hpc_settings.copy()
        loader_settings.pop("hpc_mode", None)
        train_loader = DataLoader(
            train_dataset, shuffle=True, drop_last=True, **loader_settings
        )
        val_loader = DataLoader(
            val_dataset, shuffle=False, drop_last=True, **loader_settings
        )

        model_config = get_model_config(
            train_dataset.image_shape, config["temporal_frames"]
        )
        model_config["use_spatial_attention"] = config.get(
            "use_spatial_attention", True
        )
        model_config["use_temporal_attention"] = config.get(
            "use_temporal_attention", True
        )
        model_class = get_model_class(
            config.get("architecture", {}).get("name", "transformer")
        )
        model = model_class(model_config).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config.get("weight_decay", 0.01),
        )

        loo_save_name = f"loo_{hold_out_name}_{config['save_name']}"
        loo_save_path = os.path.join("models", "trained", f"{loo_save_name}.pth")
        loo_log_dir = os.path.join("logs", "tensorboard")

        fold_config = config.copy()
        fold_config["save_name"] = loo_save_name
        fold_config["epochs"] = config.get("loo_epochs", config["epochs"])

        best_model, _ = train_model(
            model,
            train_loader,
            val_loader,
            optimizer,
            fold_config,
            loo_save_path,
            loo_log_dir,
            scaler=scaler_info,
        )

        print(f"\n--- Evaluating Hold-out {hold_out_name} ---")
        metrics = evaluate_model_and_get_metrics(
            best_model, val_loader, device, scaler_info["y"], return_preds=True
        )

        print(
            f"Hold-out {hold_out_name} Results: MAE={metrics['mae']:.3f}, MSE={metrics['mse']:.3f}"
        )

        # Save per-fold metrics to JSON
        fold_metrics_summary = {
            k: v for k, v in metrics.items() if not isinstance(v, np.ndarray)
        }
        os.makedirs("results", exist_ok=True)
        fold_json_path = os.path.join(
            "results", f"loo_fold_{hold_out_name}_metrics.json"
        )
        with open(fold_json_path, "w") as f:
            json.dump(fold_metrics_summary, f, indent=4)
        print(f"→ Saved fold metrics to {fold_json_path}")

        loo_results.append(
            {
                "hold_out_flight": hold_out_name,
                "mae": metrics["mae"],
                "mse": metrics["mse"],
                "r2": metrics["r2"],
            }
        )

    results_df = pd.DataFrame(loo_results)
    results_save_path = os.path.join(
        "results", f"loo_summary_{config['save_name']}.csv"
    )
    os.makedirs(os.path.dirname(results_save_path), exist_ok=True)
    results_df.to_csv(results_save_path, index=False)
    print(f"\nLOO evaluation complete. Summary saved to {results_save_path}")

    return results_df
