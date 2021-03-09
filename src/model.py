"""Model definitions."""
import torch
from torch.nn import (
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
from typing_extensions import Final

from .config import Config


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
        # Each block starts after the previous block, and terminates at a point
        # where either a skip connection starts or ends
        blocks = [
            ResBlock(in_channels, 64),
            Sequential(
                MaxPool2d(2), ResBlock(64, 128, dropout=config.dropout)
            ),
            Sequential(
                MaxPool2d(2), ResBlock(128, 256, dropout=config.dropout)
            ),
            Sequential(
                MaxPool2d(2), ResBlock(256, 512, dropout=config.dropout)
            ),
            Sequential(
                MaxPool2d(2),
                ResBlock(512, 1024, dropout=config.dropout),
                ConvTBlock(1024, 512, dropout=config.dropout),
            ),
            Sequential(
                ResBlock(1024, 512, dropout=config.dropout),
                ConvTBlock(512, 256, dropout=config.dropout),
            ),
            Sequential(
                ResBlock(512, 256, dropout=config.dropout),
                ConvTBlock(256, 128, dropout=config.dropout),
            ),
            Sequential(
                ResBlock(256, 128, dropout=config.dropout),
                ConvTBlock(128, 64, dropout=config.dropout),
            ),
            Sequential(
                ResBlock(128, 64, dropout=config.dropout),
                Conv2d(64, out_channels, kernel_size=1),
            ),
        ]
        self.blocks = ModuleList(blocks)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Segment the inputs."""
        # Stores all intermediate block outputs for skip-connections
        outputs = [inputs]

        # Going down the U
        for block in self.blocks[:5]:
            outputs.append(block(outputs[-1]))

        # Going up the U
        for i, block in enumerate(self.blocks[5:]):
            combined = torch.cat([outputs[4 - i], outputs[-1]], -3)
            outputs.append(block(combined))

        return outputs[-1]
