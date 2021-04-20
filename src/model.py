"""Model definitions."""
import torch
from torch.cuda.amp import autocast
from torch.nn import (
    AdaptiveAvgPool2d,
    BatchNorm2d,
    Conv2d,
    ConvTranspose2d,
    Dropout2d,
    LeakyReLU,
    MaxPool2d,
    Module,
    ModuleList,
    Sequential,
)
from torch.nn.utils import spectral_norm
from typing_extensions import Final

from .config import Config


def spectralize(module: Module) -> Module:
    """Recursively add spectral normalization to the module."""
    if "weight" in module._parameters:
        return spectral_norm(module)
    for key, value in module._modules.items():
        module._modules[key] = spectralize(value)
    return module


class ConvBlock(Module):
    """Block combining Conv2d with non-linearities.

    This block consists of:
        * Dropout2d
        * Conv2d
        * BatchNorm2d
        * LeakyReLU
    """

    LEAKY_RELU_SLOPE: Final = 0.2

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dropout: float = 0.0,
    ):
        """Initialize the layers.

        Args:
            in_channels: The no. of input channels
            out_channels: The no. of output channels
            kernel_size: The kernel size for Conv2d
            dropout: The probability of dropping out the inputs
        """
        super().__init__()
        # Appropriate padding to keep output and input sizes equal
        padding = (kernel_size - 1) // 2

        self.block = Sequential(
            Dropout2d(dropout),
            Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,  # no use of bias as BatchNorm2d will delete it
            ),
            BatchNorm2d(out_channels),
            LeakyReLU(self.LEAKY_RELU_SLOPE),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Get the block's outputs."""
        return self.block(inputs)


class ConvTBlock(Module):
    """Block combining ConvTranspose2d with non-linearities.

    This block consists of:
        * Dropout2d
        * ConvTranspose2d
        * BatchNorm2d
        * LeakyReLU
    """

    LEAKY_RELU_SLOPE: Final = 0.2

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dropout: float = 0.0,
    ):
        """Initialize the layers.

        Args:
            in_channels: The no. of input channels
            out_channels: The no. of output channels
            kernel_size: The kernel size for ConvTranspose2d
            dropout: The probability of dropping out the inputs
        """
        super().__init__()
        # Appropriate padding to keep output and input sizes equal
        padding = (kernel_size - 1) // 2
        self.block = Sequential(
            Dropout2d(dropout, inplace=True),
            ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=2,
                padding=padding,
                output_padding=1,
                bias=False,  # no use of bias as BatchNorm2d will delete it
            ),
            BatchNorm2d(out_channels),
            LeakyReLU(self.LEAKY_RELU_SLOPE, inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Get the block's outputs."""
        return self.block(inputs)


class ResBlock(Module):
    """Block with residual connections.

    This block consists of:
        * Skip connection start
        * ConvBlock
        * ConvBlock
        * Skip connection joins

    The skip connection simply consists of a 1x1 Conv2d.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dropout: float = 0.0,
    ):
        """Initialize the layers.

        Args:
            in_channels: The no. of input channels
            out_channels: The no. of output channels
            kernel_size: The kernel size for Conv2d
            dropout: The probability of dropping out the inputs
        """
        super().__init__()
        self.block = Sequential(
            ConvBlock(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                dropout=dropout,
            ),
            ConvBlock(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                dropout=dropout,
            ),
        )
        # No use of bias as the main block has a bias
        self.skip = Conv2d(
            in_channels, out_channels, kernel_size=1, bias=False
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Get the block's outputs."""
        return self.block(inputs) + self.skip(inputs)


class UNet(Module):
    """Class for the UNet architecture.

    This architecture is adapted from: https://arxiv.org/abs/1505.04597
    """

    def __init__(self, in_channels: int, out_channels: int, config: Config):
        """Initialize the model architecture.

        Args:
            in_channels: The no. of input channels
            out_channels: The no. of output channels
            config: The hyper-param config
        """
        super().__init__()
        self.config = config

        # Each block starts after the previous block, and terminates at a point
        # where either a skip connection starts or ends

        down_blocks = [
            ResBlock(in_channels, 64),
            Sequential(
                MaxPool2d(2), ResBlock(64, 128, dropout=config.gen_dropout)
            ),
            Sequential(
                MaxPool2d(2), ResBlock(128, 256, dropout=config.gen_dropout)
            ),
            Sequential(
                MaxPool2d(2), ResBlock(256, 512, dropout=config.gen_dropout)
            ),
        ]
        self.down_blocks = ModuleList(down_blocks)

        self.bottleneck = Sequential(
            MaxPool2d(2),
            ResBlock(512, 1024, dropout=config.gen_dropout),
            ConvTBlock(1024, 512, dropout=config.gen_dropout),
        )

        up_blocks = [
            Sequential(
                ResBlock(1024, 512, dropout=config.gen_dropout),
                ConvTBlock(512, 256, dropout=config.gen_dropout),
            ),
            Sequential(
                ResBlock(512, 256, dropout=config.gen_dropout),
                ConvTBlock(256, 128, dropout=config.gen_dropout),
            ),
            Sequential(
                ResBlock(256, 128, dropout=config.gen_dropout),
                ConvTBlock(128, 64, dropout=config.gen_dropout),
            ),
            Sequential(
                ResBlock(128, 64, dropout=config.gen_dropout),
                Conv2d(64, out_channels, kernel_size=1),
            ),
        ]
        self.up_blocks = ModuleList(up_blocks)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Segment the inputs."""
        # autocast is needed here since we're using DataParallel
        with autocast(enabled=self.config.mixed_precision):
            # Stores all intermediate block outputs for skip-connections
            down_outputs = [inputs]

            # Going down the U
            for block in self.down_blocks:
                down_outputs.append(block(down_outputs[-1]))

            output = self.bottleneck(down_outputs[-1])

            # Going up the U
            for i, block in enumerate(self.up_blocks, 1):
                combined = torch.cat([down_outputs[-i], output], -3)
                output = block(combined)

            return output


class Critic(Module):
    """Class for the patch critic architecture.

    Patch-discriminators are described in: https://arxiv.org/abs/1611.07004.
    Spectral normalization is described here: https://arxiv.org/abs/1802.05957.
    """

    def __init__(self, in_channels: int, config: Config):
        """Initialize the model architecture.

        Args:
            in_channels: The no. of input channels
            config: The hyper-param config
        """
        super().__init__()
        self.config = config

        # Architecture kept similar to the UNet's first half
        model = Sequential(
            ResBlock(in_channels, 64),
            MaxPool2d(2),
            ResBlock(64, 128, dropout=config.crit_dropout),
            MaxPool2d(2),
            ResBlock(128, 256, dropout=config.crit_dropout),
            MaxPool2d(2),
            ResBlock(256, 512, dropout=config.crit_dropout),
            MaxPool2d(2),
            ResBlock(512, 1024, dropout=config.crit_dropout),
            # Convert to a single output channel
            Dropout2d(config.crit_dropout),
            Conv2d(1024, 1, kernel_size=1),
            AdaptiveAvgPool2d(1),  # global average pooling
        )
        self.model = spectralize(model)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return the critic's output."""
        # autocast is needed here since we're using DataParallel
        with autocast(enabled=self.config.mixed_precision):
            return self.model(inputs).flatten(1)
