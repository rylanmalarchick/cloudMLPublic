# in main.py

import argparse
import datetime
import os

import yaml
from sklearn.preprocessing import StandardScaler
from src.main_utils import load_all_flights_metadata_for_scalers, setup_environment
from src.pipeline import (
    run_final_training_and_evaluation,
    run_pretraining,
    run_loo_evaluation,
)


def main():
    parser = argparse.ArgumentParser(description="Cloud-ML Model Training")
    parser.add_argument(
        "--config",
        type=str,
        default="bestComboConfig.yaml",
        help="Path to the configuration file",
    )
    # Explicit boolean flags for skip options
    parser.add_argument(
        "--no_pretrain", action="store_true", help="Skip pretraining phase"
    )
    parser.add_argument(
        "--no_final", action="store_true", help="Skip final training/evaluation"
    )
    parser.add_argument("--no_plots", action="store_true", help="Skip plot generation")
    # --- CORRECTED ARGUMENT DICTIONARY ---
    extra_flags = {
        "flat_field_correction": bool,
        "clahe_clip_limit": float,
        "zscore_normalize": bool,
        "use_spatial_attention": bool,
        "use_temporal_attention": bool,
    }

    for key, value in {
        "learning_rate": float,  # Changed from "lr"
        "weight_decay": float,  # Changed from "wd"
        "epochs": int,
        "optimizer": str,
        "scheduler": str,
        "hpc_mode": bool,
        "temporal_frames": int,
        "loss_type": str,
        "loss_alpha": float,
        "huber_delta": float,
        "filter_type": str,
        "cbh_min": float,
        "cbh_max": float,
        "save_name": str,
        "loo": bool,
        "early_stopping_patience": int,
        "early_stopping_min_delta": float,
        "loo_epochs": int,
        **extra_flags,
    }.items():
        if value is bool:
            parser.add_argument(
                f"--{key}",
                action=argparse.BooleanOptionalAction,
                default=None,
                help=f"Enable or disable {key.replace('_', ' ')}",
            )
        else:
            parser.add_argument(f"--{key}", type=value, default=None)

    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    for key, value in vars(args).items():
        if value is not None:
            config[key] = value

    if not config.get("save_name"):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        config["save_name"] = f"{config['filter_type']}_{timestamp}"

    save_suffix = f"_{config['save_name']}"

    print(f"Experiment ID: {config['save_name']}")
    # Dry-run exit if skipping all phases
    if args.no_pretrain and args.no_final and args.no_plots:
        import sys

        sys.exit(0)

    hpc_mode = config.get("hpc_mode", False) or bool(os.getenv("HPC_MODE"))
    hpc_settings = {
        "hpc_mode": hpc_mode,
        "batch_size": 128 if hpc_mode else 4,
        "num_workers": 32 if hpc_mode else 2,
        "pin_memory": True if hpc_mode else False,
        "prefetch_factor": 4 if hpc_mode else 1,
    }
    print(f"--- {'HPC Mode Enabled' if hpc_mode else 'Development Mode'} ---")

    device = setup_environment(hpc_mode)

    log_dir = config.get("log_dir", "logs")
    os.makedirs(log_dir, exist_ok=True)

    data_dir = config["data_directory"]
    datasets_info = []
    for flight in config["flights"]:
        datasets_info.append(
            {
                "name": flight["name"],
                "iFileName": os.path.join(data_dir, flight["iFileName"]),
                "cFileName": os.path.join(data_dir, flight["cFileName"]),
                "nFileName": os.path.join(data_dir, flight["nFileName"]),
            }
        )

    print("\n--- Fitting Global Scalers ---")
    global_sza_data, global_saa_data, global_y_data = (
        load_all_flights_metadata_for_scalers(
            datasets_info,
            filter_type=config["filter_type"],
            cbh_min=config.get("cbh_min"),
            cbh_max=config.get("cbh_max"),
        )
    )
    global_sza_scaler = StandardScaler().fit(global_sza_data)
    global_saa_scaler = StandardScaler().fit(global_saa_data)
    global_y_scaler = StandardScaler().fit(global_y_data)
    scaler_info = {
        "sza": global_sza_scaler,
        "saa": global_saa_scaler,
        "y": global_y_scaler,
    }
    print("  Global scalers fitted successfully.")

    pre_ckpt = run_pretraining(
        config,
        datasets_info,
        device,
        hpc_settings,
        save_suffix,
        global_sza_scaler,
        global_saa_scaler,
        global_y_scaler,
        log_dir=log_dir,
    )

    run_final_training_and_evaluation(
        config,
        datasets_info,
        pre_ckpt,
        device,
        hpc_settings,
        save_suffix,
        global_sza_scaler,
        global_saa_scaler,
        global_y_scaler,
        log_dir=log_dir,
    )

    if config.get("loo", False):
        run_loo_evaluation(config, datasets_info, scaler_info, hpc_settings)


if __name__ == "__main__":
    main()
