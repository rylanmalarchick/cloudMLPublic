#!/usr/bin/env python3
"""MoCo (Momentum Contrast) encoder for small cloud images (20x22 pixels).

This module implements MoCo v2 for contrastive learning on small cloud radiance
images. MoCo uses a momentum-updated key encoder and a queue of negatives to
enable large effective batch sizes without high memory requirements.

Key differences from SimCLR:
    - Momentum encoder: Slow-moving average of query encoder weights
    - Queue: Dictionary of past keys provides more negatives
    - Asymmetric architecture: Query and key encoders can differ

Paper 2 Research Question:
    Does MoCo's momentum encoder and queue mechanism improve cross-flight
    generalization compared to SimCLR?

References:
    - MoCo: https://arxiv.org/abs/1911.05722
    - MoCo v2: https://arxiv.org/abs/2003.04297

Author: Paper 2 Implementation
Date: December 2025
"""

from __future__ import annotations

import copy
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


class MoCoEncoder(nn.Module):
    """MoCo encoder backbone for 20x22 cloud images.
    
    Same architecture as SimCLR encoder for fair comparison.
    Produces fixed-size representations for contrastive learning.
    
    Attributes:
        backbone: CNN feature extractor.
        projector: MLP projection head for contrastive loss.
        feature_dim: Dimension of backbone output features.
        projection_dim: Dimension of projection head output.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        feature_dim: int = 256,
        projection_dim: int = 128,
        hidden_dim: int = 256,
    ) -> None:
        """Initialize MoCo encoder.
        
        Args:
            in_channels: Number of input channels (1 for grayscale).
            feature_dim: Output dimension of the backbone encoder.
            projection_dim: Output dimension of the projection head.
            hidden_dim: Hidden dimension of the projection MLP.
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.projection_dim = projection_dim
        
        # Backbone CNN (same as SimCLR)
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
        # MoCo v2 uses 2-layer MLP with ReLU (same as SimCLR)
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),  # Note: No BatchNorm in MoCo projector
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
    
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass through encoder.
        
        Args:
            x: Input images of shape (B, 1, 20, 22).
        
        Returns:
            Tuple of:
                - features: Backbone features of shape (B, feature_dim).
                - projections: L2-normalized projection head output, shape (B, projection_dim).
        """
        # Extract backbone features
        features = self.backbone(x)
        
        # Project and normalize for contrastive loss
        projections = self.projector(features)
        projections = F.normalize(projections, dim=1)
        
        return features, projections
    
    def encode(self, x: Tensor) -> Tensor:
        """Extract features only (for downstream tasks).
        
        Args:
            x: Input images of shape (B, 1, 20, 22).
        
        Returns:
            Features of shape (B, feature_dim).
        """
        return self.backbone(x)


class MoCoModel(nn.Module):
    """MoCo v2 model with momentum encoder and queue.
    
    Implements Momentum Contrast for unsupervised visual representation learning.
    Uses a momentum-updated key encoder and a queue of negative samples.
    
    Args:
        feature_dim: Backbone output dimension.
        projection_dim: Projection head output dimension.
        hidden_dim: Hidden dimension of projection MLP.
        queue_size: Number of keys to store in queue.
        momentum: Momentum coefficient for key encoder update.
        temperature: Temperature for softmax scaling.
    
    Example:
        >>> model = MoCoModel(feature_dim=256, projection_dim=128)
        >>> x_q = torch.randn(32, 1, 20, 22)  # Query images
        >>> x_k = torch.randn(32, 1, 20, 22)  # Key images (different augmentation)
        >>> loss, q, k = model(x_q, x_k)
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        feature_dim: int = 256,
        projection_dim: int = 128,
        hidden_dim: int = 256,
        queue_size: int = 4096,
        momentum: float = 0.999,
        temperature: float = 0.07,
    ) -> None:
        super().__init__()
        
        self.queue_size = queue_size
        self.momentum = momentum
        self.temperature = temperature
        
        # Query encoder
        self.encoder_q = MoCoEncoder(
            in_channels=in_channels,
            feature_dim=feature_dim,
            projection_dim=projection_dim,
            hidden_dim=hidden_dim,
        )
        
        # Key encoder (momentum-updated copy)
        self.encoder_k = MoCoEncoder(
            in_channels=in_channels,
            feature_dim=feature_dim,
            projection_dim=projection_dim,
            hidden_dim=hidden_dim,
        )
        
        # Initialize key encoder with query encoder weights
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False  # Key encoder is not trained by gradient
        
        # Create the queue
        self.register_buffer("queue", torch.randn(projection_dim, queue_size))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
    
    @torch.no_grad()
    def _momentum_update_key_encoder(self) -> None:
        """Update key encoder with momentum."""
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.momentum + param_q.data * (1.0 - self.momentum)
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: Tensor) -> None:
        """Update the queue with new keys.
        
        Args:
            keys: New key features to add to queue, shape (B, projection_dim).
        """
        batch_size = keys.shape[0]
        
        # Get queue buffer references
        queue_ptr: Tensor = self.queue_ptr  # type: ignore[assignment]
        queue: Tensor = self.queue  # type: ignore[assignment]
        
        ptr = int(queue_ptr.item())
        
        # Handle case where batch_size doesn't divide queue_size evenly
        if ptr + batch_size > self.queue_size:
            # Fill to end, then wrap around
            remaining = self.queue_size - ptr
            queue[:, ptr:] = keys[:remaining].T
            queue[:, :batch_size - remaining] = keys[remaining:].T
            ptr = batch_size - remaining
        else:
            # Replace the keys at ptr
            queue[:, ptr:ptr + batch_size] = keys.T
            ptr = (ptr + batch_size) % self.queue_size
        
        queue_ptr[0] = ptr
    
    def forward(
        self, x_q: Tensor, x_k: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward pass computing MoCo contrastive loss.
        
        Args:
            x_q: Query images (first augmentation), shape (B, 1, 20, 22).
            x_k: Key images (second augmentation), shape (B, 1, 20, 22).
        
        Returns:
            Tuple of:
                - loss: Scalar InfoNCE loss.
                - q: Query projections.
                - k: Key projections.
        """
        # Compute query features
        _, q = self.encoder_q(x_q)  # q: (B, projection_dim), normalized
        
        # Compute key features with no gradient
        with torch.no_grad():
            self._momentum_update_key_encoder()
            _, k = self.encoder_k(x_k)  # k: (B, projection_dim), normalized
        
        # Positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)  # (B, 1)
        
        # Negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])  # (B, K)
        
        # Logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1) / self.temperature
        
        # Labels: positive is always index 0
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        
        # InfoNCE loss
        loss = F.cross_entropy(logits, labels)
        
        # Update queue
        self._dequeue_and_enqueue(k)
        
        return loss, q, k
    
    def encode(self, x: Tensor) -> Tensor:
        """Extract features for downstream tasks.
        
        Uses the query encoder (trained encoder) for feature extraction.
        
        Args:
            x: Input images of shape (B, 1, 20, 22).
        
        Returns:
            Features of shape (B, feature_dim).
        """
        return self.encoder_q.encode(x)


def build_moco_small(queue_size: int = 4096) -> MoCoModel:
    """Build small MoCo model (default configuration).
    
    Args:
        queue_size: Size of the negative queue.
    
    Returns:
        MoCoModel with 256-dim features, 128-dim projections.
    """
    return MoCoModel(
        feature_dim=256,
        projection_dim=128,
        hidden_dim=256,
        queue_size=queue_size,
        momentum=0.999,
        temperature=0.07,
    )


def build_moco_tiny(queue_size: int = 2048) -> MoCoModel:
    """Build tiny MoCo model for quick experiments.
    
    Args:
        queue_size: Size of the negative queue.
    
    Returns:
        MoCoModel with 128-dim features, 64-dim projections.
    """
    return MoCoModel(
        feature_dim=128,
        projection_dim=64,
        hidden_dim=128,
        queue_size=queue_size,
        momentum=0.999,
        temperature=0.07,
    )


if __name__ == "__main__":
    # Quick test
    print("Testing MoCo encoder...")
    
    model = build_moco_small(queue_size=256)  # Small queue for testing
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  - Query encoder: {sum(p.numel() for p in model.encoder_q.parameters()):,}")
    print(f"  - Key encoder: {sum(p.numel() for p in model.encoder_k.parameters()):,} (frozen)")
    print(f"Queue size: {model.queue_size}")
    print(f"Queue shape: {model.queue.shape}")
    
    # Test forward pass
    x_q = torch.randn(8, 1, 20, 22)
    x_k = torch.randn(8, 1, 20, 22)
    
    loss, q, k = model(x_q, x_k)
    print(f"\nForward pass:")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Query shape: {q.shape}")
    print(f"  Key shape: {k.shape}")
    
    # Test feature extraction
    features = model.encode(x_q)
    print(f"\nFeature extraction:")
    print(f"  Feature shape: {features.shape}")
    
    # Test queue update
    queue_ptr_buf: Tensor = model.queue_ptr  # type: ignore[assignment]
    print(f"\nQueue pointer after forward: {queue_ptr_buf.item()}")
    
    # Run a few more forward passes to check queue cycling
    for i in range(5):
        loss, _, _ = model(x_q, x_k)
        print(f"  Step {i+1}: loss={loss.item():.4f}, queue_ptr={queue_ptr_buf.item()}")
    
    print("\nAll tests passed!")
