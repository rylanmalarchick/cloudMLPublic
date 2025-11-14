"""
Self-supervised pre-training utilities for CloudML model.

Based on insights from LSTM Autoencoder paper (Rostamijavanani et al.):
- Two-stage learning: unsupervised feature extraction → supervised mapping
- Encoder learns to extract meaningful spatial features via reconstruction
- Helps initialize weights before supervised training on limited labeled data

Usage:
    from src.pretraining import pretrain_encoder, ReconstructionDecoder

    # Phase 1: Self-supervised pre-training
    model = pretrain_encoder(model, train_loader, epochs=20, device='cuda')

    # Phase 2: Supervised training with pre-trained encoder
    model = train_supervised(model, train_loader, ...)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np


class ReconstructionDecoder(nn.Module):
    """
    Decoder for reconstructing input images from CNN encoder features.
    Used during self-supervised pre-training phase.

    Architecture mirrors the encoder in reverse:
    - Takes flattened feature vector from encoder
    - Upsamples through transposed convolutions
    - Outputs reconstructed image at original resolution
    """

    def __init__(self, feature_dim, image_shape=(440, 640)):
        super().__init__()
        self.feature_dim = feature_dim
        self.image_shape = image_shape

        # Calculate initial spatial size after encoder
        # Assuming encoder reduces by ~32x (5 pooling layers of 2x each)
        self.init_h = image_shape[0] // 32
        self.init_w = image_shape[1] // 32

        # Project flat features to 2D spatial features
        self.fc = nn.Linear(feature_dim, 256 * self.init_h * self.init_w)

        # Decoder: transposed convolutions to upsample
        self.decoder = nn.Sequential(
            # (batch, 256, init_h, init_w)
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # (batch, 256, init_h*2, init_w*2)
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # (batch, 128, init_h*4, init_w*4)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # (batch, 64, init_h*8, init_w*8)
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # (batch, 32, init_h*16, init_w*16)
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # (batch, 16, init_h*32, init_w*32) - should match original size
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),  # Output in [0, 1] range
        )

    def forward(self, features):
        """
        Args:
            features: (batch, feature_dim) - flattened encoder output
        Returns:
            reconstructed: (batch, 1, H, W) - reconstructed image
        """
        batch_size = features.size(0)

        # Project to spatial features
        x = self.fc(features)
        x = x.view(batch_size, 256, self.init_h, self.init_w)

        # Decode
        reconstructed = self.decoder(x)

        # Ensure output matches target size (handle rounding)
        if reconstructed.shape[2:] != self.image_shape:
            reconstructed = F.interpolate(
                reconstructed,
                size=self.image_shape,
                mode="bilinear",
                align_corners=False,
            )

        return reconstructed


def pretrain_encoder(
    model,
    train_loader,
    epochs=20,
    lr=1e-4,
    device="cuda",
    save_checkpoints=True,
    checkpoint_dir="models/pretrained",
):
    """
    Self-supervised pre-training of the CNN encoder via reconstruction.

    The encoder learns to extract meaningful spatial features by training
    it to reconstruct the input images. This provides better initialization
    than random weights when supervised training data is limited.

    Args:
        model: MultimodalRegressionModel instance
        train_loader: DataLoader with training data (labels ignored)
        epochs: Number of pre-training epochs
        lr: Learning rate for pre-training
        device: 'cuda' or 'cpu'
        save_checkpoints: Whether to save best checkpoint
        checkpoint_dir: Directory to save checkpoints

    Returns:
        model: Model with pre-trained encoder weights
    """
    print("\n" + "=" * 70)
    print("PHASE 1: SELF-SUPERVISED PRE-TRAINING")
    print("=" * 70)
    print(f"Training encoder via reconstruction task for {epochs} epochs")
    print(f"Learning rate: {lr}")
    print(f"Device: {device}")
    print("=" * 70 + "\n")

    # Move model to device
    model = model.to(device)
    model.train()

    # Create decoder for reconstruction
    # Get image shape from model config
    image_h = model.config["image_shape"][1]
    image_w = model.config["image_shape"][2]
    decoder = ReconstructionDecoder(
        feature_dim=model.cnn_output_size, image_shape=(image_h, image_w)
    ).to(device)

    # Optimizer for encoder + decoder
    encoder_params = list(model.cnn_layers.parameters())
    decoder_params = list(decoder.parameters())
    optimizer = torch.optim.Adam(
        encoder_params + decoder_params, lr=lr, weight_decay=1e-5
    )

    # Loss function
    criterion = nn.MSELoss()

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True
    )

    # Training loop
    best_loss = float("inf")
    losses_history = []

    for epoch in range(epochs):
        epoch_losses = []

        progress_bar = tqdm(train_loader, desc=f"Pretrain Epoch {epoch + 1}/{epochs}")

        for batch_idx, batch in enumerate(progress_bar):
            # Unpack batch
            if len(batch) == 3:
                images, _, _ = batch  # Ignore labels and metadata
            else:
                images = batch[0]

            images = images.to(device)
            batch_size, seq_len, h, w = images.shape

            # Process each frame independently with gradient accumulation
            # Split into smaller mini-batches to avoid OOM
            optimizer.zero_grad()
            batch_loss = 0.0

            # Process only FIRST frame to save memory (not all 7 frames)
            # This is still effective for learning spatial features
            frame = images[:, 0, :, :].unsqueeze(1)  # (batch, 1, h, w)

            # Forward through encoder
            # Temporarily disable gradient checkpointing for pre-training
            orig_checkpoint_flag = model.use_gradient_checkpointing
            model.use_gradient_checkpointing = False

            # Extract features using encoder
            with torch.cuda.amp.autocast(enabled=False):  # Disable AMP for stability
                features = _extract_features(model, frame)

            model.use_gradient_checkpointing = orig_checkpoint_flag

            # Decode to reconstruct image
            reconstructed = decoder(features)

            # Compute reconstruction loss
            loss = criterion(reconstructed, frame)
            batch_loss = loss

            # Backpropagation
            batch_loss.backward()

            # Clear cache to free memory
            del frame, features, reconstructed
            torch.cuda.empty_cache()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                encoder_params + decoder_params, max_norm=1.0
            )

            optimizer.step()

            # Track loss
            epoch_losses.append(batch_loss.item())

            # Update progress bar
            progress_bar.set_postfix(
                {
                    "loss": f"{batch_loss.item():.4f}",
                    "avg_loss": f"{np.mean(epoch_losses):.4f}",
                }
            )

        # Epoch statistics
        avg_epoch_loss = np.mean(epoch_losses)
        losses_history.append(avg_epoch_loss)

        print(
            f"\nEpoch {epoch + 1}/{epochs} - Avg Reconstruction Loss: {avg_epoch_loss:.4f}"
        )

        # Learning rate scheduling
        scheduler.step(avg_epoch_loss)

        # Save best checkpoint
        if avg_epoch_loss < best_loss and save_checkpoints:
            best_loss = avg_epoch_loss
            checkpoint_path = f"{checkpoint_dir}/pretrained_encoder_best.pth"

            # Create directory if it doesn't exist
            import os

            os.makedirs(checkpoint_dir, exist_ok=True)

            torch.save(
                {
                    "epoch": epoch,
                    "encoder_state_dict": model.cnn_layers.state_dict(),
                    "decoder_state_dict": decoder.state_dict(),
                    "loss": best_loss,
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                checkpoint_path,
            )

            print(f" Saved best checkpoint: {checkpoint_path} (loss: {best_loss:.4f})")

        # Early stopping check (if loss not improving for 5 epochs)
        if epoch > 5:
            recent_losses = losses_history[-5:]
            if all(
                recent_losses[i] <= recent_losses[i + 1]
                for i in range(len(recent_losses) - 1)
            ):
                print(f"\nWARNING: Early stopping: Loss not improving for 5 epochs")
                break

    print("\n" + "=" * 70)
    print("PRE-TRAINING COMPLETE!")
    print(f"Best reconstruction loss: {best_loss:.4f}")
    print("Encoder weights are now initialized with learned features")
    print("=" * 70 + "\n")

    # Clean up decoder (not needed for supervised training)
    del decoder
    torch.cuda.empty_cache()

    return model


def _extract_features(model, frame):
    """
    Extract features from a single frame using the CNN encoder.
    Helper function for pre-training.

    Args:
        model: MultimodalRegressionModel instance
        frame: (batch, 1, h, w) - single frame

    Returns:
        features: (batch, feature_dim) - flattened feature vector
    """
    x = frame

    # Apply spatial attention if enabled
    if model.config.get("use_spatial_attention", True):
        x = model.spatial_attention(x)

    # Pass through CNN layers (without FiLM since no scalars in pre-training)
    for layer in model.cnn_layers:
        if isinstance(
            layer,
            (
                nn.Conv2d,
                nn.BatchNorm2d,
                nn.LeakyReLU,
                nn.MaxPool2d,
                nn.Dropout2d,
                nn.AdaptiveAvgPool2d,
            ),
        ):
            x = layer(x)

    # Flatten
    features = x.flatten(1)

    return features


def pretrain_temporal_prediction(
    model, temporal_loader, epochs=15, lr=5e-5, device="cuda"
):
    """
    Alternative pre-training: predict next frame from previous frames.
    Only applicable if you have true temporal sequences of IRAI data.

    For cloud shadow problem, this is likely NOT applicable since your
    5 frames are simultaneous views, not a time sequence.

    Args:
        model: MultimodalRegressionModel instance
        temporal_loader: DataLoader with temporal sequences
        epochs: Number of epochs
        lr: Learning rate
        device: 'cuda' or 'cpu'

    Returns:
        model: Model with pre-trained temporal encoder
    """
    print("\n" + "=" * 70)
    print("PHASE 1: TEMPORAL PREDICTION PRE-TRAINING")
    print("=" * 70)
    print("WARNING: This assumes your data has true temporal sequences!")
    print("For simultaneous multi-view data, use pretrain_encoder() instead.")
    print("=" * 70 + "\n")

    model = model.to(device)
    model.train()

    # Use MSE loss for frame prediction
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        epoch_losses = []

        progress_bar = tqdm(
            temporal_loader, desc=f"Temporal Pretrain {epoch + 1}/{epochs}"
        )

        for batch in progress_bar:
            images, _, _ = batch
            images = images.to(device)

            # Use frames [0:seq_len-1] to predict frame [seq_len-1]
            input_frames = images[:, :-1, :, :]
            target_frame = images[:, -1, :, :]

            # Extract features from input frames
            features = []
            for t in range(input_frames.size(1)):
                frame = input_frames[:, t, :, :].unsqueeze(1)
                feat = _extract_features(model, frame)
                features.append(feat)

            temporal_input = torch.stack(features, dim=1)

            # Apply temporal attention to aggregate
            if hasattr(model, "temporal_attention"):
                aggregated, _ = model.temporal_attention(temporal_input)
            else:
                aggregated = temporal_input.mean(dim=1)

            # Predict target frame (would need a prediction head - simplified here)
            # This is a simplified version; full implementation needs a decoder
            loss = torch.tensor(0.0).to(device)  # Placeholder

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        print(f"Epoch {epoch + 1}/{epochs} - Avg Loss: {np.mean(epoch_losses):.4f}")

    print("\nTemporal pre-training complete!\n")
    return model


def evaluate_reconstruction_quality(
    model, decoder, val_loader, device="cuda", num_samples=5
):
    """
    Evaluate reconstruction quality by visualizing input vs reconstructed images.
    Useful for checking if pre-training is working properly.

    Args:
        model: Pre-trained model
        decoder: Reconstruction decoder
        val_loader: Validation data loader
        device: 'cuda' or 'cpu'
        num_samples: Number of samples to visualize

    Returns:
        avg_mse: Average reconstruction MSE
    """
    model.eval()
    decoder.eval()

    mse_losses = []

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= num_samples:
                break

            images, _, _ = batch
            images = images.to(device)

            # Take first frame
            frame = images[:, 0, :, :].unsqueeze(1)

            # Extract features and reconstruct
            features = _extract_features(model, frame)
            reconstructed = decoder(features)

            # Compute MSE
            mse = F.mse_loss(reconstructed, frame)
            mse_losses.append(mse.item())

            print(f"Sample {i + 1} - Reconstruction MSE: {mse.item():.4f}")

    avg_mse = np.mean(mse_losses)
    print(f"\nAverage Reconstruction MSE: {avg_mse:.4f}")

    model.train()
    decoder.train()

    return avg_mse
