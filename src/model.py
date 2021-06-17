"""Model definitions."""
from typing import Optional, Tuple

import torch
from torch.cuda.amp import autocast
from torch.nn import (
    AdaptiveAvgPool2d,
    BatchNorm2d,
    Conv2d,
    Dropout2d,
    Identity,
    LeakyReLU,
    MaxPool2d,
    Module,
    ModuleList,
    Sequential,
)
from torch.nn.functional import interpolate
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


class CombineBlock(Module):
    """Block combining upsampling with non-linearities and concatenation.

    This block consists of:
        * Upsampling
        * ConvBlock
        * Concatenation
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
            out_channels: The no. of output channels after upsampling. Note
                that the final outputs will have double the channels, due to
                concatenation.
            kernel_size: The kernel size for Conv2d
            dropout: The probability of dropping out the inputs
        """
        super().__init__()
        self.conv = ConvBlock(
            in_channels, out_channels, kernel_size=kernel_size, dropout=dropout
        )

    def forward(
        self, inputs: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """Get the block's outputs.

        Args:
            inputs: The tuple that consists of:
                * The low-res inputs to be upsampled
                * The high-res inputs to be concatenated at the end
        """
        low_res, high_res = inputs
        upsampled = interpolate(low_res, high_res.shape[-2:])
        processed = self.conv(upsampled)
        return torch.cat([high_res, processed], -3)


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
        if in_channels != out_channels:
            self.skip = Conv2d(
                in_channels, out_channels, kernel_size=1, bias=False
            )
        else:
            self.skip = Identity()

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

        down_blocks = [ResBlock(in_channels, config.init_channels)]
        up_blocks_rev = []
        curr_channels = config.init_channels

        for i in range(config.unet_depth - 1):
            next_channels = min(curr_channels * 2, config.max_channels)

            if config.avgpool and i == config.unet_depth - 2:
                downsampler = AdaptiveAvgPool2d(1)
            else:
                downsampler = MaxPool2d(2)
            down_blocks.append(
                Sequential(
                    downsampler,
                    ResBlock(
                        curr_channels, next_channels, dropout=config.dropout
                    ),
                )
            )

            up_block_layers = [
                CombineBlock(
                    next_channels, curr_channels, dropout=config.dropout
                ),
                ResBlock(
                    2 * curr_channels,  # due to skip connection concatenation
                    curr_channels,
                    dropout=config.dropout,
                ),
            ]
            if i == 0:
                up_block_layers.append(
                    Conv2d(curr_channels, out_channels, kernel_size=1)
                )
            up_blocks_rev.append(Sequential(*up_block_layers))

            curr_channels = next_channels

        self.down_blocks = ModuleList(down_blocks[:-1])
        self.bottleneck = down_blocks[-1]
        self.up_blocks = ModuleList(up_blocks_rev[::-1])

    def forward(
        self, inputs: torch.Tensor, only_latent: bool = False
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """Segment the inputs."""
        # autocast is needed here since we're using DataParallel
        with autocast(enabled=self.config.mixed_precision):
            # Stores all intermediate block outputs for skip-connections
            down_outputs = [inputs]

            # Going down the U
            for block in self.down_blocks:
                down_outputs.append(block(down_outputs[-1]))

            latent = self.bottleneck(down_outputs[-1])
            if only_latent:
                return None, latent

            # Going up the U
            output = latent
            for i, block in enumerate(self.up_blocks, 1):
                output = block([output, down_outputs[-i]])

        return output, latent
