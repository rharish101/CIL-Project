"""Ensembles multiple model predictions."""
import torch
from torch.nn import ReLU, Sequential

from .model import ResBlock


class EnsemblerCNN(torch.nn.Module):
    """The Ensembler CNN."""

    def __init__(self, input_image: bool):
        """Init."""
        super().__init__()

        HIDDEN_SIZE = 4

        self.cnn_layers = Sequential(
            # Defining a 2D convolution layer
            ResBlock(
                (6 if input_image else 3),
                HIDDEN_SIZE,
                kernel_size=7,
                dropout=0.2,
            ),
            ReLU(inplace=True),
            ResBlock(HIDDEN_SIZE, HIDDEN_SIZE, kernel_size=5, dropout=0.2),
            ReLU(inplace=True),
            ResBlock(HIDDEN_SIZE, HIDDEN_SIZE, kernel_size=3, dropout=0.2),
            ReLU(inplace=True),
            ResBlock(HIDDEN_SIZE, 1, kernel_size=3, dropout=0.2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward."""
        return self.cnn_layers(inputs)
