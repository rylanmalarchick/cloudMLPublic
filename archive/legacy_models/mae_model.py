"""
Masked Autoencoder (MAE) for Self-Supervised Learning - Phase 2

A simplified MAE implementation optimized for cloud imagery SSL pre-training.
Based on "Masked Autoencoders Are Scalable Vision Learners" (He et al., 2021)

This implementation is designed for:
- Single-channel cloud images (440 width)
- GTX 1070 Ti memory constraints (8GB VRAM)
- Efficient training on ~60k images
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple


class PatchEmbed(nn.Module):
    """
    Split image into patches and embed them.

    For a 440-width image with patch_size=16:
    - Creates 27 patches horizontally (440 // 16 = 27.5, we'll pad/crop)
    - Embeds each patch into embed_dim dimensions
    """

    def __init__(
        self,
        img_width: int = 440,
        patch_size: int = 16,
        in_channels: int = 1,
        embed_dim: int = 192,
    ):
        super().__init__()
        self.img_width = img_width
        self.patch_size = patch_size

        # Calculate number of patches (we'll crop image to fit)
        self.n_patches = img_width // patch_size
        self.cropped_width = self.n_patches * patch_size

        # Projection: Conv1d to embed patches (for 1D signals)
        self.proj = nn.Conv1d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        # x: (B, C, W) - our images are 1D signals (B, 1, W)
        B, C, W = x.shape

        # Crop to fit patch size
        x = x[:, :, : self.cropped_width]  # (B, C, cropped_width)

        # Embed patches using Conv1d
        x = self.proj(x)  # (B, embed_dim, n_patches)

        # Transpose to (B, n_patches, embed_dim)
        x = x.transpose(1, 2)

        return x


class TransformerBlock(nn.Module):
    """
    Standard transformer encoder block with self-attention and FFN.
    """

    def __init__(
        self,
        embed_dim: int = 192,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # Self-attention with residual
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]

        # FFN with residual
        x = x + self.mlp(self.norm2(x))

        return x


class MAEEncoder(nn.Module):
    """
    MAE encoder: processes visible (unmasked) patches only.
    """

    def __init__(
        self,
        img_width: int = 440,
        patch_size: int = 16,
        embed_dim: int = 192,
        depth: int = 4,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.patch_embed = PatchEmbed(img_width, patch_size, 1, embed_dim)
        self.n_patches = self.patch_embed.n_patches

        # Positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, embed_dim))

        # CLS token (optional, for downstream tasks)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        # Initialize positional embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x, mask_indices=None):
        """
        Forward pass through encoder.

        Args:
            x: (B, 1, W) input images
            mask_indices: (B, n_visible) indices of visible patches
                         If None, use all patches (no masking)

        Returns:
            (B, n_visible, embed_dim) encoded visible patches
        """
        # Embed patches
        x = self.patch_embed(x)  # (B, n_patches, embed_dim)
        B, N, D = x.shape

        # Add positional embeddings
        x = x + self.pos_embed

        # Apply masking if provided
        if mask_indices is not None:
            # Select only visible patches
            x = torch.gather(
                x,
                dim=1,
                index=mask_indices.unsqueeze(-1).expand(-1, -1, D),
            )

        # Append CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        return x


class MAEDecoder(nn.Module):
    """
    MAE decoder: reconstructs original image from encoded patches.
    """

    def __init__(
        self,
        n_patches: int = 27,
        patch_size: int = 16,
        embed_dim: int = 192,
        decoder_embed_dim: int = 96,
        decoder_depth: int = 2,
        decoder_num_heads: int = 3,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()

        self.n_patches = n_patches
        self.patch_size = patch_size

        # Project encoder output to decoder dimension
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)

        # Mask token (learned embedding for masked patches)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        # Positional embeddings for decoder
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, n_patches, decoder_embed_dim)
        )

        # Transformer blocks
        self.decoder_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    decoder_embed_dim,
                    decoder_num_heads,
                    mlp_ratio,
                    dropout=0.0,
                )
                for _ in range(decoder_depth)
            ]
        )

        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)

        # Prediction head: project to patch pixels
        self.decoder_pred = nn.Linear(
            decoder_embed_dim,
            patch_size,  # Each patch is patch_size pixels
        )

        # Initialize weights
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)

    def forward(self, x, mask_indices, unmask_indices):
        """
        Decode visible patches and reconstruct full image.

        Args:
            x: (B, n_visible+1, embed_dim) encoded patches (with CLS)
            mask_indices: (B, n_masked) indices of masked patches
            unmask_indices: (B, n_visible) indices of visible patches

        Returns:
            (B, n_patches * patch_size) reconstructed image
        """
        B = x.shape[0]

        # Remove CLS token
        x = x[:, 1:, :]  # (B, n_visible, embed_dim)

        # Project to decoder dimension
        x = self.decoder_embed(x)  # (B, n_visible, decoder_embed_dim)

        # Create full sequence with mask tokens
        n_visible = x.shape[1]
        n_masked = self.n_patches - n_visible

        # Expand mask tokens
        mask_tokens = self.mask_token.expand(B, n_masked, -1)

        # Concatenate visible and mask tokens
        x_full = torch.cat([x, mask_tokens], dim=1)  # (B, n_patches, D)

        # Unshuffle: put patches back in original order
        # Create indices for gathering
        combined_indices = torch.cat([unmask_indices, mask_indices], dim=1)

        # Sort to get inverse permutation
        sorted_indices = torch.argsort(combined_indices, dim=1)

        # Apply inverse permutation
        x_full = torch.gather(
            x_full,
            dim=1,
            index=sorted_indices.unsqueeze(-1).expand(-1, -1, x_full.shape[-1]),
        )

        # Add positional embeddings
        x_full = x_full + self.decoder_pos_embed

        # Apply transformer blocks
        for block in self.decoder_blocks:
            x_full = block(x_full)

        x_full = self.decoder_norm(x_full)

        # Predict pixel values for each patch
        pred = self.decoder_pred(x_full)  # (B, n_patches, patch_size)

        # Reshape to image
        pred = pred.flatten(1)  # (B, n_patches * patch_size)

        return pred


class MaskedAutoencoder(nn.Module):
    """
    Complete MAE model for self-supervised pre-training.
    """

    def __init__(
        self,
        img_width: int = 440,
        patch_size: int = 16,
        embed_dim: int = 192,
        depth: int = 4,
        num_heads: int = 3,
        decoder_embed_dim: int = 96,
        decoder_depth: int = 2,
        decoder_num_heads: int = 3,
        mlp_ratio: float = 4.0,
        mask_ratio: float = 0.75,
    ):
        super().__init__()

        self.mask_ratio = mask_ratio
        self.patch_size = patch_size

        # Encoder
        self.encoder = MAEEncoder(
            img_width=img_width,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
        )

        # Decoder
        self.decoder = MAEDecoder(
            n_patches=self.encoder.n_patches,
            patch_size=patch_size,
            embed_dim=embed_dim,
            decoder_embed_dim=decoder_embed_dim,
            decoder_depth=decoder_depth,
            decoder_num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
        )

    def random_masking(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform random masking by shuffling patches.

        Args:
            x: (B, n_patches, D) patch embeddings

        Returns:
            x_masked: (B, n_visible, D) visible patches
            mask_indices: (B, n_masked) indices of masked patches
            unmask_indices: (B, n_visible) indices of visible patches
        """
        B, N, D = x.shape
        n_keep = int(N * (1 - self.mask_ratio))

        # Random shuffle
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)

        # Keep first n_keep patches
        unmask_indices = ids_shuffle[:, :n_keep]
        mask_indices = ids_shuffle[:, n_keep:]

        return mask_indices, unmask_indices

    def forward(
        self, imgs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through MAE.

        Args:
            imgs: (B, 1, W) input images

        Returns:
            pred: (B, n_patches * patch_size) reconstructed patches
            mask: (B, n_patches) binary mask (1 = masked, 0 = visible)
            target: (B, n_patches * patch_size) target pixel values
        """
        B, C, W = imgs.shape

        # Patchify
        n_patches = self.encoder.n_patches
        cropped_width = n_patches * self.patch_size
        imgs_cropped = imgs[:, :, :cropped_width]

        # Create target (patchified image)
        target = imgs_cropped.reshape(B, n_patches, self.patch_size)
        target = target.flatten(1)  # (B, n_patches * patch_size)

        # Random masking
        patch_embed = self.encoder.patch_embed(imgs)
        mask_indices, unmask_indices = self.random_masking(patch_embed)

        # Encode visible patches
        latent = self.encoder(imgs, unmask_indices)

        # Decode
        pred = self.decoder(latent, mask_indices, unmask_indices)

        # Create binary mask
        mask = torch.zeros(B, n_patches, device=imgs.device)
        mask.scatter_(1, mask_indices, 1)
        mask = mask.unsqueeze(-1).expand(-1, -1, self.patch_size)
        mask = mask.reshape(B, -1)  # (B, n_patches * patch_size)

        return pred, mask, target

    def forward_encoder(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Encode images without masking (for inference/fine-tuning).

        Args:
            imgs: (B, 1, W) input images

        Returns:
            (B, embed_dim) CLS token features
        """
        latent = self.encoder(imgs, mask_indices=None)
        return latent[:, 0, :]  # Return CLS token


def build_mae_small():
    """Build a small MAE model suitable for GTX 1070 Ti."""
    return MaskedAutoencoder(
        img_width=440,
        patch_size=16,
        embed_dim=192,
        depth=4,
        num_heads=3,
        decoder_embed_dim=96,
        decoder_depth=2,
        decoder_num_heads=3,
        mlp_ratio=4.0,
        mask_ratio=0.75,
    )


def build_mae_tiny():
    """Build a tiny MAE model for faster training."""
    return MaskedAutoencoder(
        img_width=440,
        patch_size=20,  # Larger patches = fewer patches = faster
        embed_dim=128,
        depth=3,
        num_heads=2,
        decoder_embed_dim=64,
        decoder_depth=1,
        decoder_num_heads=2,
        mlp_ratio=4.0,
        mask_ratio=0.75,
    )


if __name__ == "__main__":
    # Test the model
    print("Testing MAE model...")

    model = build_mae_small()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    batch_size = 4
    imgs = torch.randn(batch_size, 1, 440)

    pred, mask, target = model(imgs)
    print(f"\nInput shape: {imgs.shape}")
    print(f"Prediction shape: {pred.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Target shape: {target.shape}")

    # Test encoder only
    features = model.forward_encoder(imgs)
    print(f"Encoder output shape: {features.shape}")

    print("\n MAE model tests passed!")
