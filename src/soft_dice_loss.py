"""Module for soft dice loss."""
import torch

"""
Code taken from https://github.com/Paulymorphous/skeyenet
Original author: Jerin Paul (Dec, 2019)
"""


def dice_coef(
    y_true: torch.Tensor, y_pred: torch.Tensor, smooth=1
) -> torch.Tensor:
    """Calculates the dice coefficient.

    Args:
        y_true: true y
        y_pred: pred y
        smooth: smoothing factor to prevent division by zero
    """
    y_true_f = torch.flatten(y_true)
    y_pred_f = torch.flatten(y_pred)

    intersection = torch.sum(y_true_f * y_pred_f)
    dice = (2.0 * intersection + smooth) / (
        torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth
    )

    return dice


def soft_dice_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Calculates the dice loss.

    Had to add the sigmoid to map predictions into [0, 1]

    Args:
        y_true: true y
        y_pred: pred y
    """
    return 1 - dice_coef(y_true, torch.sigmoid(y_pred))
