import os

import torch


def save_model(model, scaler_Y, save_path):
    """
    Save model checkpoint (state_dict + scaler + image shape) to the given path.
    Creates parent directories if needed.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    ckpt = {
        "model_state_dict": model.state_dict(),
        "y_scaler": scaler_Y,
        "image_shape": getattr(model, "image_shape", None),
    }
    torch.save(ckpt, save_path)
    print(f"→ Saved model + scaler to {save_path}")
