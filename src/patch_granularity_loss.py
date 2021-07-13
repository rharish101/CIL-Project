"""Module that implements a loss that optimizes for the kaggle-objective."""
from typing import Callable

import torch
from typing_extensions import Final

# Percentage of pixels > 1 required to assign a foreground label to a patch
FOREGROUND_THRESHOLD: Final = 0.5
# Size of each patch as specified in the problem statement
PATCH_SIZE: Final = 16


def patchify_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Call this with the loss you intend to use.

    Had to add the sigmoid to map predictions into [0, 1]

    Args:
        loss: The loss that should be used
        y_true: true y
        y_pred: pred y
    """
    y_true_patches = []
    y_pred_patches = []
    for instance in range(y_pred.shape[0]):
        for row in range(0, y_pred.shape[2], PATCH_SIZE):
            for col in range(0, y_pred.shape[3], PATCH_SIZE):
                pred_patch = y_pred[
                    instance, 0, row : row + PATCH_SIZE, col : col + PATCH_SIZE
                ]
                true_patch = y_true[
                    instance, 0, row : row + PATCH_SIZE, col : col + PATCH_SIZE
                ]
                y_true_patches.append(true_patch.mean())
                y_pred_patches.append(pred_patch.mean())
    y_true_patches = torch.stack(y_true_patches)
    y_pred_patches = torch.stack(y_pred_patches)
    print("Previous loss:", loss(y_pred, y_true))
    print("Patchified loss: ", loss(y_pred_patches, y_true_patches))
    return loss(y_pred_patches, y_true_patches)
