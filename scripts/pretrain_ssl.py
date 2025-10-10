import argparse
import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.unlabeled_dataset import UnlabeledCloudDataset
from src.mae_model import MaskedAutoencoder
from src.pytorchmodel import get_model_config


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Self-supervised pre-training for cloud model"
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Configuration file"
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of pre-training epochs"
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--output",
        type=str,
        default="ssl_pretrained_encoder.pth",
        help="Output weights file",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device()

    data_paths = [
        os.path.join(config["data_directory"], flight["iFileName"])
        for flight in config["flights"]
    ]

    dataset = UnlabeledCloudDataset(data_paths, config)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # Assuming image shape can be inferred from config or a default
    image_shape = (
        1,
        config.get("swath_slice", [0, 440])[1] - config.get("swath_slice", [0, 440])[0],
        64,
    )  # Example, adjust as needed
    model_config = get_model_config(image_shape, config["temporal_frames"])
    model = MaskedAutoencoder(model_config).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)

    print(f"Starting pre-training for {args.epochs} epochs on {device}...")
    model.train()

    for epoch in range(args.epochs):
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for images in pbar:
            images = images.to(device)

            reconstructed, mask = model(images)

            # Loss on masked patches
            loss = criterion(reconstructed[~mask], images[~mask])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{args.epochs}, Average Loss: {avg_loss:.6f}")

    print(f"Saving encoder weights to {args.output}")
    torch.save(model.encoder.state_dict(), args.output)

    print("Pre-training complete!")


if __name__ == "__main__":
    main()
