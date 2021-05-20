"""CRF post processing."""
import torch
import torch.nn as nn
from crfseg import CRF

from .model import UNet


def crf(prediction: torch.Tensor, nn_model: UNet):
    """CRF poset processing.

    Args:
        prediction: The initial prediction
        nn_model: The main neural network model
    """
    model = nn.Sequential(nn_model, CRF(n_spatial_dims=2))

    # x = torch.zeros(batch_size, n_channels, *spatial)
    return model(prediction)
