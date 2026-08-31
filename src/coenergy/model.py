"""Model definitions and factory for the coEnergy assessment."""

import torch
import torch.nn as nn


class SimpleMLP(nn.Module):
    """Basic Multi-Layer Perceptron: Linear layers + ReLU.

    Args:
        input_dim: Flattened input size (2 * T)
        n_outputs: Number of target outputs (n_hb + 3)
        hidden_dim: Size of every hidden layer
        nb_hidden: Number of hidden layers (excluding input and output)
    """

    def __init__(
        self,
        input_dim: int,
        n_outputs: int,
        hidden_dim: int = 256,
        nb_hidden: int = 2,
    ):
        super().__init__()

        layers = []
        in_dim = input_dim

        # Build hidden layers with constant width
        for _ in range(nb_hidden):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim

        # Final output layer
        layers.append(nn.Linear(in_dim, n_outputs))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, 2, T] -> flatten to [N, 2*T]
        x = x.flatten(start_dim=1)
        return self.net(x)


class ResidualBlock(nn.Module):
    """Two dilated convolutions with a residual connection.

    Padding is chosen so the temporal length is preserved, which keeps the
    residual add shape-compatible and lets every block see the full sequence.
    """

    def __init__(self, channels: int, kernel_size: int, dilation: int):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2

        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(x + out)


class DilatedCNN(nn.Module):
    """Dilated 1D CNN for time series regression.

    A strided stem first downsamples the sequence 4x (600 -> 150). The targets
    are a steady-state gain and time constants in hours, so per-timestep
    resolution carries no useful signal and the stack runs 4x cheaper without it.

    Dilations then double at every block, so n_blocks blocks reach a receptive
    field of ~4 * (2^n_blocks - 1) downsampled steps — 6 blocks covers the whole
    sequence. No recurrence, so the sequence is processed in parallel.

    The readout is a global mean+max pool over time rather than a final
    timestep: the targets are properties of the whole response, and mean
    pooling is what lets the network express an average gain.

    Args:
        n_channels: Number of input channels (2: temperature, solicitation)
        n_outputs: Number of target outputs (n_hb + 3)
        cnn_channels: Width of every convolution
        n_blocks: Number of residual blocks (dilation 2^i at block i)
        kernel_size: Convolution kernel size inside the blocks
        head_hidden: Width of the hidden layer in the regression head
    """

    def __init__(
        self,
        n_channels: int = 2,
        n_outputs: int = 7,
        cnn_channels: int = 64,
        n_blocks: int = 6,
        kernel_size: int = 3,
        head_hidden: int = 128,
    ):
        super().__init__()

        # Strided (not max-pooled) downsampling: a max pool over a temperature
        # trace keeps the upper envelope, a strided conv learns what to keep.
        self.stem = nn.Sequential(
            nn.Conv1d(n_channels, cnn_channels, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(
            *[ResidualBlock(cnn_channels, kernel_size, dilation=2 ** i) for i in range(n_blocks)]
        )
        # mean and max pooling are concatenated, hence 2 * cnn_channels
        self.head = nn.Sequential(
            nn.Linear(2 * cnn_channels, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, n_outputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, 2, T]
        x = self.stem(x)      # [N, C, T/4]
        x = self.blocks(x)

        pooled = torch.cat([x.mean(dim=-1), x.amax(dim=-1)], dim=1)
        return self.head(pooled)


def load_model(input_dim: int, n_outputs: int, config) -> nn.Module:
    """Factory function to build a model based on config."""
    model_type = config.model_type.lower()

    if model_type == "mlp":
        return SimpleMLP(
            input_dim=input_dim,
            n_outputs=n_outputs,
            hidden_dim=config.mlp_hidden_dim,
            nb_hidden=config.mlp_nb_hidden,
        )
    elif model_type == "cnn":
        return DilatedCNN(
            n_channels=2,
            n_outputs=n_outputs,
            cnn_channels=config.cnn_channels,
            n_blocks=config.cnn_blocks,
            kernel_size=config.cnn_kernel_size,
            head_hidden=config.cnn_head_hidden,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
