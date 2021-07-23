"""Ensembles multiple model predictions."""

import torch
from torch.nn import Conv2d, ReLU, Sequential


class EnsemblerCNN(torch.nn.Module):
    """The Ensembler CNN."""

    def __init__(self, input_image: bool, num_inputs: int):
        """Init."""
        super().__init__()

        HIDDEN_SIZE = 8

        # ResBlock(HIDDEN_SIZE, HIDDEN_SIZE, kernel_size=5, dropout=0.2)

        self.cnn_layers = Sequential(
            # Defining a 2D convolution layer
            Conv2d(
                (num_inputs + 3 if input_image else num_inputs),
                HIDDEN_SIZE,
                kernel_size=7,
                padding=3,
            ),
            ReLU(inplace=True),
            Conv2d(HIDDEN_SIZE, HIDDEN_SIZE, kernel_size=5, padding=2),
            ReLU(inplace=True),
            Conv2d(HIDDEN_SIZE, HIDDEN_SIZE, kernel_size=5, padding=2),
            ReLU(inplace=True),
            Conv2d(HIDDEN_SIZE, HIDDEN_SIZE, kernel_size=3, padding=1),
            ReLU(inplace=True),
            Conv2d(HIDDEN_SIZE, 1, kernel_size=3, padding=1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward."""
        return self.cnn_layers(inputs)
