import torch
import torch.nn as nn
from src.pytorchmodel import MultimodalRegressionModel


class MaskedAutoencoder(nn.Module):
    def __init__(self, model_config, mask_ratio=0.75, patch_size=8):
        super(MaskedAutoencoder, self).__init__()

        base_model = MultimodalRegressionModel(model_config)
        self.encoder = self._extract_encoder(base_model)
        self.decoder = self._build_decoder(model_config, patch_size)

        self.mask_ratio = mask_ratio
        self.patch_size = patch_size

    def _extract_encoder(self, model):
        # Extract the CNN layers, excluding the final pooling and dense layers
        encoder_layers = [
            layer
            for layer in model.cnn_layers
            if not isinstance(layer, nn.AdaptiveAvgPool2d)
        ]
        return nn.Sequential(*encoder_layers)

    def _build_decoder(self, model_config, patch_size):
        # Build a decoder that mirrors the CNN encoder's structure
        decoder_layers = []
        # Example: if encoder is Conv -> Pool -> Conv -> Pool
        # Decoder should be TConv -> TConv
        # This needs to be carefully constructed based on your CNN architecture
        # For simplicity, a few transposed convolutions to get back to original size

        # Find the number of channels output by the encoder
        h, w = model_config["image_shape"][1], model_config["image_shape"][2]
        dummy_input = torch.zeros(1, 1, h, w)
        encoder_output_shape = self.encoder(dummy_input).shape
        in_channels = encoder_output_shape[1]

        # Simplified decoder
        decoder_layers.extend(
            [
                nn.ConvTranspose2d(
                    in_channels,
                    128,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                nn.ReLU(),
                nn.ConvTranspose2d(
                    128, 64, kernel_size=3, stride=2, padding=1, output_padding=1
                ),
                nn.ReLU(),
                nn.ConvTranspose2d(
                    64, 1, kernel_size=3, stride=2, padding=1, output_padding=1
                ),
                nn.Sigmoid(),  # To output pixel values between 0 and 1
            ]
        )

        return nn.Sequential(*decoder_layers)

    def random_masking(self, x):
        B, C, H, W = x.shape
        patches_h, patches_w = H // self.patch_size, W // self.patch_size
        num_patches = patches_h * patches_w

        num_mask = int(self.mask_ratio * num_patches)

        mask_indices = torch.rand(B, num_patches, device=x.device).argsort(dim=1)
        mask_idx = mask_indices[:, :num_mask]

        mask = torch.ones(B, num_patches, device=x.device)
        mask.scatter_(dim=1, index=mask_idx, value=0)

        mask = mask.reshape(B, 1, patches_h, patches_w)
        mask = mask.repeat_interleave(self.patch_size, dim=2).repeat_interleave(
            self.patch_size, dim=3
        )

        masked_x = x * mask

        return masked_x, mask.bool()  # Return boolean mask for loss calculation

    def forward(self, x):
        masked_x, mask = self.random_masking(x)

        features = self.encoder(masked_x)
        reconstructed = self.decoder(features)

        return reconstructed, mask
