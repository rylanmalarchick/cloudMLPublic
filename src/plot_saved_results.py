"""
Standalone script to load a saved checkpoint, evaluate on CPL picks only, and generate plots.
"""

import argparse
import datetime
import os

import h5py
import torch
from src.evaluate_model import evaluate_model
from src.main_utils import prepare_streaming_data, setup_environment
from src.visualization import plot_results


def main():
    parser = argparse.ArgumentParser(
        description="Load model and create evaluation plots on CPL-filtered data"
    )
    parser.add_argument(
        "-c", "--checkpoint", required=True, help="Path to model checkpoint (.pth)"
    )
    parser.add_argument(
        "-o", "--output_dir", default="plots", help="Directory to save plots"
    )
    parser.add_argument(
        "-f",
        "--flight",
        default=None,
        help="Flight name to evaluate (e.g. 30Oct24); omit for all",
    )
    parser.add_argument(
        "--hpc_mode", action="store_true", help="Enable HPC DataLoader settings"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for DataLoader"
    )
    parser.add_argument(
        "--num_workers", type=int, default=2, help="Number of DataLoader workers"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = setup_environment(args.hpc_mode)

    # Load checkpoint (allow full unpickle)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    # Instantiate model and load state
    model = torch.nn.Module()  # placeholder for correct model class
    # Replace with your model class import if needed
    from src.pytorchmodel import get_model_class, get_model_config

    # Use model selection based on checkpoint or default
    arch_name = ckpt.get("architecture", {}).get("name", "transformer")
    model_config = get_model_config(ckpt["image_shape"], ckpt.get("temporal_frames", 3))
    model_class = get_model_class(arch_name)
    model = model_class(model_config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Define flights_info same as in main.py
    datasets_info = [
        {
            "name": "10Feb25",
            "iFileName": "/home/rylan/Documents/NASA/cloudML/data/10Feb25/GLOVE2025_IRAI_L1B_Rev-_20250210.h5",
            "cFileName": "/home/rylan/Documents/NASA/cloudML/data/10Feb25/CPL_L2_V1-02_01kmLay_259015_10feb25.hdf5",
            "nFileName": "/home/rylan/Documents/NASA/cloudML/data/10Feb25/CRS_20250210_nav.hdf",
        },
        {
            "name": "30Oct24",
            "iFileName": "/home/rylan/Documents/NASA/cloudML/data/30Oct24/WHYMSIE2024_IRAI_L1B_Rev-_20241030.h5",
            "cFileName": "/home/rylan/Documents/NASA/cloudML/data/30Oct24/CPL_L2_V1-02_01kmLay_259006_30oct24.hdf5",
            "nFileName": "/home/rylan/Documents/NASA/cloudML/data/30Oct24/CRS_20241030_nav.hdf",
        },
        {
            "name": "04Nov24",
            "iFileName": "/home/rylan/Documents/NASA/cloudML/data/04Nov24/WHYMSIE2024_IRAI_L1B_Rev-_20241104.h5",
            "cFileName": "/home/rylan/Documents/NASA/cloudML/data/04Nov24/CPL_L2_V1-02_01kmLay_259008_04nov24.hdf5",
            "nFileName": "/home/rylan/Documents/NASA/cloudML/data/04Nov24/CRS_20241104_nav.hdf",
        },
    ]

    for info in datasets_info:
        if args.flight and info["name"] != args.flight:
            continue

        print(f"\nEvaluating flight: {info['name']}")
        # Prepare CPL-filtered data loader
        train_loader, test_loader, image_shape, y_scaler, ds = prepare_streaming_data(
            info["iFileName"],
            info["cFileName"],
            info["nFileName"],
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            temporal_frames=3,  # Default temporal frames
        )

        # Evaluate on CPL picks only
        loss, mae, mse, y_true, y_pred, indices = evaluate_model(
            model, test_loader, device, y_scaler, return_preds=True
        )
        print(f"Results: Loss={loss:.4f}, MAE={mae:.4f}, MSE={mse:.4f}")

        # Load navigation data
        with h5py.File(info["nFileName"], "r") as nf:
            nav_data = {"lat": nf["nav/IWG_lat"][:], "lon": nf["nav/IWG_lon"][:]}

        # Timestamp for plot directory
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        os.path.join(args.output_dir, f"{info['name']}_{ts}")

        # Generate plots (using CPL picks)
        plot_results(
            model=model,
            Y_test=y_true,
            Y_pred=y_pred,
            raw_indices=indices,
            nav_data=nav_data,
            model_name=info["name"],
            timestamp=ts,
            dataset=ds,
        )


if __name__ == "__main__":
    main()
