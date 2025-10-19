# in src/pytorchmodel.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data, Batch

try:
    from mamba_ssm import Mamba

    MAMBA_AVAILABLE = True
except ImportError:
    Mamba = None
    MAMBA_AVAILABLE = False
    print("Mamba-SSM not available; SSMModel will use a fallback.")


# ... (all your other classes like SpatialAttention, TemporalAttention, FiLLMLayer remain the same) ...
class SpatialAttention(nn.Module):
    """
    Generates a spatial attention map to focus on relevant pixels within a single frame.
    """

    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(1, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attn_map = self.sigmoid(self.conv2(self.relu(self.conv1(x))))

        self.attention_weights = attn_map
        return x * attn_map


class TemporalAttention(nn.Module):
    """
    Computes temporal attention scores to weigh the importance of each frame in a sequence.
    """

    def __init__(self, in_features):
        super(TemporalAttention, self).__init__()
        self.fc1 = nn.Linear(in_features, in_features // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features // 2, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        scores = self.fc2(self.relu(self.fc1(x)))
        attn_weights = self.softmax(scores)
        weighted_features = x * attn_weights
        context_vector = torch.sum(weighted_features, dim=1)
        return context_vector, attn_weights.squeeze(-1)


class MultiScaleTemporalAttention(nn.Module):
    """
    Multi-scale attention across temporal/spatial frames.
    Processes frames at different temporal scales before attention.
    Based on Himawari-8 paper insights - captures features at multiple scales.

    For cloud shadow height: treats the 5 simultaneous camera views as a "temporal"
    sequence and applies multi-scale processing to capture cross-view relationships
    at different scales.
    """

    def __init__(self, feature_dim, num_heads=4):
        super().__init__()
        self.feature_dim = feature_dim

        # Multi-scale processing paths
        self.scale_1 = nn.Identity()  # Full resolution

        self.scale_2 = nn.Sequential(  # 2-frame average
            nn.Conv1d(feature_dim, feature_dim, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(),
        )

        self.scale_3 = nn.Sequential(  # 3-frame average
            nn.Conv1d(feature_dim, feature_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(),
        )

        # Attention over concatenated multi-scale features
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim * 3,  # Concatenated scales
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True,
        )

        # Project back to original dimension
        self.projection = nn.Linear(feature_dim * 3, feature_dim)
        self.layer_norm = nn.LayerNorm(feature_dim)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, feature_dim) - sequence of frame features
        Returns:
            attended: (batch, feature_dim) - aggregated feature vector
            weights: attention weights
        """
        batch_size, seq_len, feat_dim = x.shape

        # Transpose for Conv1d: (batch, feature_dim, seq_len)
        x_t = x.transpose(1, 2)

        # Multi-scale processing
        scale1_out = self.scale_1(x_t)  # (batch, feat, seq_len)
        scale2_out = self.scale_2(x_t)  # Conv1d with padding
        scale3_out = self.scale_3(x_t)  # Conv1d with padding

        # Ensure all scales have same seq_len by trimming if needed
        min_len = min(scale1_out.size(2), scale2_out.size(2), scale3_out.size(2))
        scale1_out = scale1_out[:, :, :min_len]
        scale2_out = scale2_out[:, :, :min_len]
        scale3_out = scale3_out[:, :, :min_len]

        # Concatenate scales: (batch, feat*3, seq_len)
        multi_scale = torch.cat([scale1_out, scale2_out, scale3_out], dim=1)

        # Transpose back: (batch, seq_len, feat*3)
        multi_scale = multi_scale.transpose(1, 2)

        # Apply attention
        attended, attn_weights = self.attention(multi_scale, multi_scale, multi_scale)

        # Pool over sequence dimension
        attended_pooled = attended.mean(dim=1)  # (batch, feat*3)

        # Project back to original feature dimension
        output = self.projection(attended_pooled)  # (batch, feat_dim)
        output = self.layer_norm(output)

        return output, attn_weights


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) layer.
    """

    def __init__(self, scalar_features, channels):
        super(FiLMLayer, self).__init__()
        self.channels = channels
        self.film_mlp = nn.Sequential(
            nn.Linear(scalar_features, 256), nn.ReLU(), nn.Linear(256, channels * 2)
        )

    def forward(self, x, scalars):
        batch_size = x.size(0)
        gamma_beta = self.film_mlp(scalars).view(batch_size, self.channels * 2, 1, 1)
        gamma = gamma_beta[:, : self.channels, :, :]
        beta = gamma_beta[:, self.channels :, :, :]
        return gamma * x + beta


class MultimodalRegressionModel(nn.Module):
    def __init__(self, model_config):
        super(MultimodalRegressionModel, self).__init__()
        self.config = model_config
        self.use_gradient_checkpointing = model_config.get(
            "gradient_checkpointing", False
        )
        self.cnn_layers, self.film_layers, self.cnn_output_size = (
            self._build_cnn_layers()
        )
        self.dense_layers = self._build_dense_layers()
        self.output = nn.Linear(self.config["dense_layers"][-1]["size"], 1)
        self.spatial_attention = SpatialAttention(1)

        # Use multi-scale temporal attention if specified in config
        if self.config.get("use_multiscale_temporal", False):
            self.temporal_attention = MultiScaleTemporalAttention(
                self.cnn_output_size, num_heads=self.config.get("attention_heads", 4)
            )
        else:
            self.temporal_attention = TemporalAttention(self.cnn_output_size)

        self._initialize_weights()

    def _build_cnn_layers(self):
        layers = nn.ModuleList()
        film_layers = nn.ModuleList()
        in_channels = 1
        h, w = self.config["image_shape"][1], self.config["image_shape"][2]

        for layer_config in self.config["cnn_layers"]:
            layers.append(
                nn.Conv2d(
                    in_channels, layer_config["out_channels"], **layer_config["params"]
                )
            )
            layers.append(nn.BatchNorm2d(layer_config["out_channels"]))
            layers.append(nn.LeakyReLU(self.config.get("leaky_relu_slope", 0.01)))

            if layer_config.get("film", False):
                film_layers.append(FiLMLayer(2, layer_config["out_channels"]))
            else:
                film_layers.append(None)

            if "pool" in layer_config:
                layers.append(nn.MaxPool2d(**layer_config["pool"]))

            if layer_config.get("dropout", 0) > 0:
                layers.append(nn.Dropout2d(layer_config["dropout"]))

            in_channels = layer_config["out_channels"]

        layers.append(nn.AdaptiveAvgPool2d((1, 1)))

        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, h, w)
            probe_model = nn.Sequential(
                *[layer for layer in layers if not isinstance(layer, FiLMLayer)]
            )
            cnn_output_size = probe_model(dummy_input).flatten(1).shape[1]

        return layers, film_layers, cnn_output_size

    def _build_dense_layers(self):
        layers = nn.ModuleList()
        in_features = self.cnn_output_size
        for layer_config in self.config["dense_layers"]:
            layers.append(nn.Linear(in_features, layer_config["size"]))
            if layer_config.get("batch_norm", False):
                layers.append(nn.BatchNorm1d(layer_config["size"]))
            layers.append(nn.LeakyReLU(self.config.get("leaky_relu_slope", 0.01)))
            if layer_config.get("dropout", 0) > 0:
                layers.append(nn.Dropout(layer_config["dropout"]))
            in_features = layer_config["size"]
        return layers

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="leaky_relu"
                )
        nn.init.normal_(self.output.weight, 0, 0.5)
        nn.init.constant_(self.output.bias, 0.1)

    def _process_frame(self, frame, scalars):
        """Process a single frame through CNN layers with optional gradient checkpointing."""
        x = frame
        if self.config.get("use_spatial_attention", True):
            x = self.spatial_attention(x)

        self.attention_maps.append(self.spatial_attention.attention_weights)

        film_idx_counter = 0
        for i, layer in enumerate(self.cnn_layers):
            x = layer(x)
            if (
                isinstance(layer, nn.LeakyReLU)
                and film_idx_counter < len(self.film_layers)
                and self.film_layers[film_idx_counter] is not None
            ):
                conv_layers_passed = sum(
                    1 for lyr in self.cnn_layers[: i + 1] if isinstance(lyr, nn.Conv2d)
                )
                if conv_layers_passed > film_idx_counter:
                    x = self.film_layers[film_idx_counter](x, scalars)
                    film_idx_counter += 1

        return x.flatten(1)

    def forward(self, image_input, param1_input, param2_input):
        image_input = torch.nan_to_num(image_input, 0.0)
        scalars = torch.cat(
            [
                torch.nan_to_num(param1_input.view(param1_input.size(0), -1), 0.0),
                torch.nan_to_num(param2_input.view(param2_input.size(0), -1), 0.0),
            ],
            dim=1,
        )

        batch_size, seq_len, h, w = image_input.shape
        frame_features = []

        self.attention_maps = []

        for t in range(seq_len):
            frame = image_input[:, t, :, :].unsqueeze(1)

            if self.use_gradient_checkpointing and self.training:
                # Use gradient checkpointing to save memory during training
                from torch.utils.checkpoint import checkpoint

                frame_feat = checkpoint(
                    self._process_frame, frame, scalars, use_reentrant=False
                )
            else:
                frame_feat = self._process_frame(frame, scalars)

            frame_features.append(frame_feat)

        temporal_input = torch.stack(frame_features, dim=1)
        if self.config.get("use_temporal_attention", True):
            x, _ = self.temporal_attention(temporal_input)
        else:
            x = torch.mean(temporal_input, dim=1)

        for layer in self.dense_layers:
            x = layer(x)

        cloud_height = self.output(x)
        return torch.nan_to_num(cloud_height, 0.0), None


# Alias for backward compatibility
TransformerModel = MultimodalRegressionModel


class GNNModel(MultimodalRegressionModel):
    def __init__(self, model_config):
        super(GNNModel, self).__init__(model_config)

        # Override temporal attention with GNN layers
        self.gnn_conv1 = GATConv(
            self.config["cnn_layers"][-1]["out_channels"], 256, heads=8, dropout=0.6
        )
        self.gnn_conv2 = GATConv(
            256 * 8, self.cnn_output_size, heads=1, concat=False, dropout=0.6
        )
        self.gnn_norm = torch.nn.LayerNorm(self.cnn_output_size)

        # Remove temporal attention if it exists from parent
        if hasattr(self, "temporal_attention"):
            del self.temporal_attention

        self._initialize_gnn_weights()

    def _initialize_gnn_weights(self):
        # Re-initialize GNN layers specifically
        for name, param in self.gnn_conv1.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param)
        for name, param in self.gnn_conv2.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param)

    def image_to_graph(self, feature_map):
        batch_size, channels, height, width = feature_map.shape
        node_features = feature_map.permute(0, 2, 3, 1).reshape(
            batch_size, -1, channels
        )

        data_list = []
        for i in range(batch_size):
            edge_index = []
            for r in range(height):
                for c in range(width):
                    node_idx = r * width + c
                    if c < width - 1:
                        edge_index.extend(
                            [[node_idx, node_idx + 1], [node_idx + 1, node_idx]]
                        )
                    if r < height - 1:
                        edge_index.extend(
                            [[node_idx, node_idx + width], [node_idx + width, node_idx]]
                        )

            edge_index = (
                torch.tensor(edge_index, dtype=torch.long, device=feature_map.device)
                .t()
                .contiguous()
            )
            graph_data = Data(x=node_features[i], edge_index=edge_index)
            data_list.append(graph_data)

        return Batch.from_data_list(data_list)

    def forward(self, image_input, param1_input, param2_input):
        image_input = torch.nan_to_num(image_input, 0.0)
        scalars = torch.cat(
            [
                torch.nan_to_num(param1_input.view(param1_input.size(0), -1), 0.0),
                torch.nan_to_num(param2_input.view(param2_input.size(0), -1), 0.0),
            ],
            dim=1,
        )

        batch_size, seq_len, h, w = image_input.shape
        frame_features = []

        self.attention_maps = []

        # Create a separate CNN forward pass that stops before pooling
        cnn_feature_extractor = nn.Sequential(
            *[
                layer
                for layer in self.cnn_layers
                if not isinstance(layer, nn.AdaptiveAvgPool2d)
            ]
        )

        for t in range(seq_len):
            x = image_input[:, t, :, :].unsqueeze(1)
            if self.config.get("use_spatial_attention", True):
                x = self.spatial_attention(x)

            self.attention_maps.append(self.spatial_attention.attention_weights)

            film_idx_counter = 0
            # Manually iterate through CNN layers to apply FiLM
            for i, layer in enumerate(cnn_feature_extractor):
                x = layer(x)
                if (
                    isinstance(layer, nn.LeakyReLU)
                    and film_idx_counter < len(self.film_layers)
                    and self.film_layers[film_idx_counter] is not None
                ):
                    conv_layers_passed = sum(
                        1
                        for lyr in cnn_feature_extractor[: i + 1]
                        if isinstance(lyr, nn.Conv2d)
                    )
                    if conv_layers_passed > film_idx_counter:
                        x = self.film_layers[film_idx_counter](x, scalars)
                        film_idx_counter += 1

            # Convert feature map to graph and process
            graph = self.image_to_graph(x)
            gnn_out = F.dropout(graph.x, p=0.6, training=self.training)
            gnn_out = F.elu(self.gnn_conv1(gnn_out, graph.edge_index))
            gnn_out = F.dropout(gnn_out, p=0.6, training=self.training)
            gnn_out = self.gnn_conv2(gnn_out, graph.edge_index)
            gnn_out = self.gnn_norm(gnn_out)

            # Global pooling to get a single vector per frame
            pooled_out = global_mean_pool(gnn_out, graph.batch)
            frame_features.append(pooled_out)

        # Temporal aggregation (simple averaging)
        x = torch.mean(torch.stack(frame_features, dim=1), dim=1)

        for layer in self.dense_layers:
            x = layer(x)

        cloud_height = self.output(x)
        return torch.nan_to_num(cloud_height, 0.0), None


class SSMModel(MultimodalRegressionModel):
    def __init__(self, model_config):
        super(SSMModel, self).__init__(model_config)

        # Override temporal attention with Mamba SSM
        if MAMBA_AVAILABLE:
            self.mamba = Mamba(
                d_model=self.cnn_output_size,
                d_state=16,
                d_conv=4,
                expand=2,
            )
        else:
            # Fallback: Use a simple LSTM or mean pooling
            self.mamba = nn.LSTM(
                input_size=self.cnn_output_size,
                hidden_size=self.cnn_output_size,
                num_layers=2,
                batch_first=True,
            )

        # Remove temporal attention if it exists from parent
        if hasattr(self, "temporal_attention"):
            del self.temporal_attention

    def forward(self, image_input, param1_input, param2_input):
        image_input = torch.nan_to_num(image_input, 0.0)
        scalars = torch.cat(
            [
                torch.nan_to_num(param1_input.view(param1_input.size(0), -1), 0.0),
                torch.nan_to_num(param2_input.view(param2_input.size(0), -1), 0.0),
            ],
            dim=1,
        )

        batch_size, seq_len, h, w = image_input.shape
        frame_features = []

        self.attention_maps = []

        for t in range(seq_len):
            x = image_input[:, t, :, :].unsqueeze(1)
            if self.config.get("use_spatial_attention", True):
                x = self.spatial_attention(x)

            self.attention_maps.append(self.spatial_attention.attention_weights)

            film_idx_counter = 0
            for i, layer in enumerate(self.cnn_layers):
                x = layer(x)
                if (
                    isinstance(layer, nn.LeakyReLU)
                    and film_idx_counter < len(self.film_layers)
                    and self.film_layers[film_idx_counter] is not None
                ):
                    conv_layers_passed = sum(
                        1
                        for lyr in self.cnn_layers[: i + 1]
                        if isinstance(lyr, nn.Conv2d)
                    )
                    if conv_layers_passed > film_idx_counter:
                        x = self.film_layers[film_idx_counter](x, scalars)
                        film_idx_counter += 1

            frame_features.append(x.flatten(1))

        # Reshape for Mamba
        temporal_input = torch.stack(frame_features, dim=1)

        if MAMBA_AVAILABLE:
            # Pass through Mamba
            mamba_out = self.mamba(temporal_input)
            # Pool the output from Mamba's sequence
            x = mamba_out.mean(dim=1)
        else:
            # Fallback: Use LSTM
            lstm_out, _ = self.mamba(temporal_input)
            x = lstm_out.mean(dim=1)

        for layer in self.dense_layers:
            x = layer(x)

        cloud_height = self.output(x)
        return torch.nan_to_num(cloud_height, 0.0), None


class SimpleCNNModel(nn.Module):
    """A simple CNN baseline without attention or complex features for ablation."""

    def __init__(self, model_config):
        super(SimpleCNNModel, self).__init__()
        self.config = model_config
        self.cnn_layers = self._build_simple_cnn()
        self.dense_layers = self._build_dense_layers()
        self.output = nn.Linear(self.config["dense_layers"][-1]["size"], 1)
        self._initialize_weights()

    def _build_simple_cnn(self):
        layers = nn.ModuleList()
        in_channels = 1
        for layer_config in self.config["cnn_layers"]:
            layers.append(
                nn.Conv2d(
                    in_channels, layer_config["out_channels"], **layer_config["params"]
                )
            )
            layers.append(nn.ReLU())
            if "pool" in layer_config:
                layers.append(nn.MaxPool2d(**layer_config["pool"]))
            in_channels = layer_config["out_channels"]
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        return layers

    def _build_dense_layers(self):
        layers = nn.ModuleList()
        in_features = self.config["cnn_layers"][-1]["out_channels"]
        for layer_config in self.config["dense_layers"]:
            layers.append(nn.Linear(in_features, layer_config["size"]))
            layers.append(nn.ReLU())
            in_features = layer_config["size"]
        return layers

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, image_input, param1_input, param2_input):
        # Simple CNN: average over temporal frames, ignore scalars for baseline
        x = torch.mean(image_input, dim=1)  # Average temporal frames
        for layer in self.cnn_layers:
            x = layer(x)
        x = x.flatten(1)
        for layer in self.dense_layers:
            x = layer(x)
        cloud_height = self.output(x)
        return cloud_height, None


class CustomLoss(nn.Module):
    """A custom loss function with multiple modes."""

    def __init__(
        self, loss_type="huber", alpha=0.5, huber_delta=1.0, gamma=2.0, under_weight=1.0
    ):
        super(CustomLoss, self).__init__()
        self.loss_type = loss_type
        self.alpha = alpha
        self.huber_delta = huber_delta
        self.gamma = gamma
        self.under_weight = under_weight

    def forward(self, y_pred, y_true, reduction="mean"):
        if y_pred.dim() > 1 and y_pred.shape[-1] == 1:
            y_pred = y_pred.squeeze(-1)
        if y_true.dim() > 1 and y_true.shape[-1] == 1:
            y_true = y_true.squeeze(-1)
        # --- NEW: Added reduction parameter ---
        if self.loss_type == "huber":
            return F.huber_loss(
                y_pred, y_true, delta=self.huber_delta, reduction=reduction
            )

        error = y_true - y_pred
        abs_error = torch.abs(error)

        if self.loss_type == "mse_mae":
            mse_loss = F.mse_loss(y_pred, y_true, reduction=reduction)
            mae_loss = F.l1_loss(y_pred, y_true, reduction=reduction)
            loss = self.alpha * mse_loss + (1 - self.alpha) * mae_loss
            return loss

        # ... other loss calculations ...
        quadratic = torch.clamp(abs_error, max=self.huber_delta)
        linear = abs_error - quadratic
        base_huber = 0.5 * quadratic.pow(2) + self.huber_delta * linear

        if self.loss_type == "huber_mae":
            focal_loss = abs_error**self.gamma
            loss = self.alpha * base_huber + (1 - self.alpha) * focal_loss
            if reduction == "mean":
                return loss.mean()
            return loss

        under_mask = (error > 0).float()
        weighted_huber = base_huber * (
            self.under_weight * under_mask + 1.0 * (1 - under_mask)
        )

        if self.loss_type == "weighted_huber":
            if reduction == "mean":
                return weighted_huber.mean()
            return weighted_huber

        if self.loss_type == "weighted_huber_mae":
            focal_loss = abs_error**self.gamma
            loss = self.alpha * weighted_huber + (1 - self.alpha) * focal_loss
            if reduction == "mean":
                return loss.mean()
            return loss

        raise ValueError(f"Unknown loss_type: {self.loss_type}")


def get_model_class(architecture_name):
    """Returns the model class based on architecture name."""
    if architecture_name == "transformer":
        return MultimodalRegressionModel
    elif architecture_name == "gnn":
        return GNNModel
    elif architecture_name == "ssm":
        return SSMModel
    elif architecture_name == "cnn":
        return SimpleCNNModel
    else:
        raise ValueError(f"Unknown architecture: {architecture_name}")


def get_model_config(
    image_shape, temporal_frames, scalar_features=3, memory_optimized=False
):
    """
    Returns model configuration.

    Args:
        image_shape: Shape of input images (temporal_frames, height, width)
        temporal_frames: Number of temporal frames
        scalar_features: Number of scalar features (SZA, SAA, etc.)
        memory_optimized: If True, use reduced channels for Colab T4 GPU (15GB)
    """
    if memory_optimized:
        # Memory-optimized config for Colab T4 - reduces channels by ~50%
        # This allows batch_size=16 to fit in ~8-9GB instead of 14GB+
        return {
            "image_shape": image_shape,
            "temporal_frames": temporal_frames,
            "scalar_features": scalar_features,
            "leaky_relu_slope": 0.01,
            "cnn_layers": [
                {
                    "out_channels": 32,  # Reduced from 64
                    "params": {"kernel_size": 3, "padding": 1},
                    "film": True,
                    "pool": {"kernel_size": 2, "stride": 2},
                    "dropout": 0.25,
                },
                {
                    "out_channels": 64,  # Reduced from 128
                    "params": {"kernel_size": 3, "padding": 1},
                    "pool": {"kernel_size": 2, "stride": 2},
                    "dropout": 0.25,
                },
                {
                    "out_channels": 128,  # Reduced from 256
                    "params": {"kernel_size": 3, "padding": 1},
                    "pool": {"kernel_size": 2, "stride": 2},
                    "dropout": 0.25,
                },
            ],
            "dense_layers": [
                {"size": 128, "batch_norm": True, "dropout": 0.5},  # Reduced from 256
                {"size": 64, "batch_norm": True, "dropout": 0.3},  # Reduced from 128
            ],
            "use_spatial_attention": True,
            "use_temporal_attention": True,
        }
    else:
        # Original full-size config for HPC systems with more memory
        return {
            "image_shape": image_shape,
            "temporal_frames": temporal_frames,
            "scalar_features": scalar_features,
            "leaky_relu_slope": 0.01,
            "cnn_layers": [
                {
                    "out_channels": 64,
                    "params": {"kernel_size": 3, "padding": 1},
                    "film": True,
                    "pool": {"kernel_size": 2, "stride": 2},
                    "dropout": 0.25,
                },
                {
                    "out_channels": 128,
                    "params": {"kernel_size": 3, "padding": 1},
                    "pool": {"kernel_size": 2, "stride": 2},
                    "dropout": 0.25,
                },
                {
                    "out_channels": 256,
                    "params": {"kernel_size": 3, "padding": 1},
                    "pool": {"kernel_size": 2, "stride": 2},
                    "dropout": 0.25,
                },
            ],
            "dense_layers": [
                {"size": 256, "batch_norm": True, "dropout": 0.5},
                {"size": 128, "batch_norm": True, "dropout": 0.3},
            ],
            "use_spatial_attention": True,
            "use_temporal_attention": True,
        }
