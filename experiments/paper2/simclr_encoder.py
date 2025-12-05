#!/usr/bin/env python3
"""SimCLR encoder for small cloud images (20x22 pixels).

This module implements a compact CNN encoder designed for SimCLR contrastive
learning on small (20x22) cloud radiance images. The architecture is optimized
for the image size while providing sufficient representational capacity.

Architecture:
    - 4-layer CNN backbone with progressive channel expansion
    - Global average pooling to fixed-size representation
    - Projection head (MLP) for contrastive learning
    - Separate output for downstream tasks (before projection)

Paper 2 Research Question:
    Can unsupervised domain adaptation via contrastive learning improve
    cross-flight generalization? (Baseline R² = -0.98 on cross-flight)

Author: Paper 2 Implementation
Date: 2025
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ConvBlock(nn.Module):
    """Convolutional block with BatchNorm and ReLU.
    
    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Size of the convolutional kernel.
        stride: Stride of the convolution.
        padding: Padding added to input.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through conv-bn-relu."""
        return self.relu(self.bn(self.conv(x)))


class SimCLREncoder(nn.Module):
    """SimCLR encoder for 20x22 cloud images.
    
    Compact CNN encoder optimized for small cloud radiance images.
    Produces fixed-size representations suitable for contrastive learning.
    
    Attributes:
        backbone: CNN feature extractor.
        projector: MLP projection head for contrastive loss.
        feature_dim: Dimension of backbone output features.
        projection_dim: Dimension of projection head output.
    
    Example:
        >>> encoder = SimCLREncoder(feature_dim=256, projection_dim=128)
        >>> x = torch.randn(32, 1, 20, 22)  # Batch of images
        >>> features, projections = encoder(x)
        >>> features.shape
        torch.Size([32, 256])
        >>> projections.shape
        torch.Size([32, 128])
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        feature_dim: int = 256,
        projection_dim: int = 128,
        hidden_dim: int = 256,
    ) -> None:
        """Initialize SimCLR encoder.
        
        Args:
            in_channels: Number of input channels (1 for grayscale).
            feature_dim: Output dimension of the backbone encoder.
            projection_dim: Output dimension of the projection head.
            hidden_dim: Hidden dimension of the projection MLP.
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.projection_dim = projection_dim
        
        # Backbone CNN
        # Input: (B, 1, 20, 22)
        self.backbone = nn.Sequential(
            # Layer 1: (B, 1, 20, 22) -> (B, 32, 20, 22)
            ConvBlock(in_channels, 32, kernel_size=3, stride=1, padding=1),
            
            # Layer 2: (B, 32, 20, 22) -> (B, 64, 10, 11)
            ConvBlock(32, 64, kernel_size=3, stride=2, padding=1),
            
            # Layer 3: (B, 64, 10, 11) -> (B, 128, 5, 6)
            ConvBlock(64, 128, kernel_size=3, stride=2, padding=1),
            
            # Layer 4: (B, 128, 5, 6) -> (B, 256, 3, 3)
            ConvBlock(128, 256, kernel_size=3, stride=2, padding=1),
            
            # Global average pooling: (B, 256, 3, 3) -> (B, 256, 1, 1)
            nn.AdaptiveAvgPool2d((1, 1)),
            
            # Flatten: (B, 256, 1, 1) -> (B, 256)
            nn.Flatten(),
            
            # Project to feature_dim if different from 256
            nn.Linear(256, feature_dim) if feature_dim != 256 else nn.Identity(),
        )
        
        # Projection head (MLP) for contrastive learning
        # Following SimCLR paper: 2-layer MLP with ReLU
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim),
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize model weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(
        self, x: Tensor, return_projection: bool = True
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Forward pass through encoder.
        
        Args:
            x: Input images of shape (B, 1, 20, 22).
            return_projection: Whether to return projection head output.
        
        Returns:
            Tuple of:
                - features: Backbone features of shape (B, feature_dim).
                - projections: Projection head output of shape (B, projection_dim),
                  or None if return_projection=False.
        """
        # Extract backbone features
        features = self.backbone(x)
        
        if return_projection:
            # Project for contrastive loss
            projections = self.projector(features)
            return features, projections
        else:
            return features, None
    
    def encode(self, x: Tensor) -> Tensor:
        """Extract features only (for downstream tasks).
        
        Args:
            x: Input images of shape (B, 1, 20, 22).
        
        Returns:
            Features of shape (B, feature_dim).
        """
        features, _ = self.forward(x, return_projection=False)
        return features


class NTXentLoss(nn.Module):
    """Normalized Temperature-scaled Cross-Entropy Loss (NT-Xent).
    
    Contrastive loss used in SimCLR. For a batch of N pairs (2N total views),
    each positive pair is contrasted against 2(N-1) negative pairs.
    
    Args:
        temperature: Temperature parameter for softmax scaling.
    
    Example:
        >>> loss_fn = NTXentLoss(temperature=0.5)
        >>> z1 = torch.randn(32, 128)  # Projections from view 1
        >>> z2 = torch.randn(32, 128)  # Projections from view 2
        >>> loss = loss_fn(z1, z2)
    """
    
    def __init__(self, temperature: float = 0.5) -> None:
        super().__init__()
        self.temperature = temperature
    
    def forward(self, z1: Tensor, z2: Tensor) -> Tensor:
        """Compute NT-Xent loss.
        
        Args:
            z1: Projections from first augmented view, shape (B, D).
            z2: Projections from second augmented view, shape (B, D).
        
        Returns:
            Scalar loss value.
        """
        batch_size = z1.size(0)
        device = z1.device
        
        # Normalize projections
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        # Concatenate projections: (2B, D)
        z = torch.cat([z1, z2], dim=0)
        
        # Compute similarity matrix: (2B, 2B)
        similarity = torch.mm(z, z.t()) / self.temperature
        
        # Create mask for positive pairs
        # Positive pairs are (i, i+B) and (i+B, i) for i in [0, B)
        mask = torch.eye(2 * batch_size, device=device, dtype=torch.bool)
        
        # Mask out self-similarity (diagonal)
        similarity.masked_fill_(mask, float('-inf'))
        
        # Create labels: positive pairs
        # For sample i in [0, B), positive is at i+B
        # For sample i in [B, 2B), positive is at i-B
        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size, device=device),
            torch.arange(0, batch_size, device=device),
        ])
        
        # Cross-entropy loss
        loss = F.cross_entropy(similarity, labels)
        
        return loss


class SimCLRModel(nn.Module):
    """Complete SimCLR model wrapping encoder and loss.
    
    Convenience wrapper that combines the encoder with NT-Xent loss
    for training.
    
    Args:
        feature_dim: Backbone output dimension.
        projection_dim: Projection head output dimension.
        temperature: NT-Xent temperature parameter.
    
    Example:
        >>> model = SimCLRModel(feature_dim=256, projection_dim=128)
        >>> x1 = torch.randn(32, 1, 20, 22)  # View 1
        >>> x2 = torch.randn(32, 1, 20, 22)  # View 2
        >>> loss, z1, z2 = model(x1, x2)
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        feature_dim: int = 256,
        projection_dim: int = 128,
        hidden_dim: int = 256,
        temperature: float = 0.5,
    ) -> None:
        super().__init__()
        
        self.encoder = SimCLREncoder(
            in_channels=in_channels,
            feature_dim=feature_dim,
            projection_dim=projection_dim,
            hidden_dim=hidden_dim,
        )
        self.criterion = NTXentLoss(temperature=temperature)
    
    def forward(
        self, x1: Tensor, x2: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward pass computing contrastive loss.
        
        Args:
            x1: First augmented view, shape (B, 1, 20, 22).
            x2: Second augmented view, shape (B, 1, 20, 22).
        
        Returns:
            Tuple of:
                - loss: Scalar NT-Xent loss.
                - z1: Projections from view 1.
                - z2: Projections from view 2.
        """
        _, z1 = self.encoder(x1)
        _, z2 = self.encoder(x2)
        
        loss = self.criterion(z1, z2)
        
        return loss, z1, z2
    
    def encode(self, x: Tensor) -> Tensor:
        """Extract features for downstream tasks.
        
        Args:
            x: Input images of shape (B, 1, 20, 22).
        
        Returns:
            Features of shape (B, feature_dim).
        """
        return self.encoder.encode(x)


def build_simclr_small() -> SimCLRModel:
    """Build small SimCLR model (default configuration).
    
    Returns:
        SimCLRModel with 256-dim features, 128-dim projections.
    """
    return SimCLRModel(
        feature_dim=256,
        projection_dim=128,
        hidden_dim=256,
        temperature=0.5,
    )


def build_simclr_tiny() -> SimCLRModel:
    """Build tiny SimCLR model for quick experiments.
    
    Returns:
        SimCLRModel with 128-dim features, 64-dim projections.
    """
    return SimCLRModel(
        feature_dim=128,
        projection_dim=64,
        hidden_dim=128,
        temperature=0.5,
    )


if __name__ == "__main__":
    # Quick test
    print("Testing SimCLR encoder...")
    
    model = build_simclr_small()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    x1 = torch.randn(8, 1, 20, 22)
    x2 = torch.randn(8, 1, 20, 22)
    
    loss, z1, z2 = model(x1, x2)
    print(f"Loss: {loss.item():.4f}")
    print(f"Projection shape: {z1.shape}")
    
    # Test feature extraction
    features = model.encode(x1)
    print(f"Feature shape: {features.shape}")
    
    print("All tests passed!")
