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


class ConvBlock(Module):
    """Block combining Conv2d with non-linearities.

    This block consists of:
        * Dropout2d
        * Conv2d
        * BatchNorm2d
        * LeakyReLU
    """

    DROPOUT: Final = 0.2
    LEAKY_RELU_SLOPE: Final = 0.2

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dropout: bool = True,
    ):
        """Initialize the layers.

        Args:
            in_channels: The no. of input channels
            out_channels: The no. of output channels
            kernel_size: The kernel size for Conv2d
            dropout: Whether to use dropout or not
        """
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.block = Sequential(
            Dropout2d(self.DROPOUT if dropout else 0.0),
            Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            ),
            BatchNorm2d(out_channels),
            LeakyReLU(self.LEAKY_RELU_SLOPE),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Get the block outputs."""
        return self.block(inputs)


class ConvTBlock(Module):
    """Block combining ConvTranspose2d with non-linearities.

    This block consists of:
        * Dropout2d
        * ConvTranspose2d
        * BatchNorm2d
        * LeakyReLU
    """

    DROPOUT: Final = 0.2
    LEAKY_RELU_SLOPE: Final = 0.2

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int = 3
    ):
        """Initialize the layers.

        Args:
            in_channels: The no. of input channels
            out_channels: The no. of output channels
            kernel_size: The kernel size for Conv2d
        """
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.block = Sequential(
            Dropout2d(self.DROPOUT, inplace=True),
            ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=2,
                padding=padding,
                output_padding=1,
                bias=False,
            ),
            BatchNorm2d(out_channels),
            LeakyReLU(self.LEAKY_RELU_SLOPE, inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Get the block outputs."""
        return self.block(inputs)


class UNet(Module):
    """Class for the UNet architecture."""

    def __init__(self, in_channels: int, out_channels: int):
        """Initialize the model architecture."""
        super().__init__()
        blocks = [
            Sequential(
                ConvBlock(in_channels, 64, dropout=False),
                ConvBlock(64, 64),
            ),
            Sequential(
                MaxPool2d(2),
                ConvBlock(64, 128),
                ConvBlock(128, 128),
            ),
            Sequential(
                MaxPool2d(2),
                ConvBlock(128, 256),
                ConvBlock(256, 256),
            ),
            Sequential(
                MaxPool2d(2),
                ConvBlock(256, 512),
                ConvBlock(512, 512),
            ),
            Sequential(
                MaxPool2d(2),
                ConvBlock(512, 1024),
                ConvBlock(1024, 1024),
                ConvTBlock(1024, 512),
            ),
            Sequential(
                ConvBlock(1024, 512),
                ConvBlock(512, 512),
                ConvTBlock(512, 256),
            ),
            Sequential(
                ConvBlock(512, 256),
                ConvBlock(256, 256),
                ConvTBlock(256, 128),
            ),
            Sequential(
                ConvBlock(256, 128),
                ConvBlock(128, 128),
                ConvTBlock(128, 64),
            ),
            Sequential(
                ConvBlock(128, 64),
                ConvBlock(64, 64),
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
