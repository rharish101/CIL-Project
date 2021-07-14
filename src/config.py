"""Hyper-param config handling."""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import toml


@dataclass(frozen=True)
class Config:
    """Class to hold hyper-parameter configs.

    Attributes:
        learn_rate: The learning rate for the optimizer
        max_learn_rate: The maximum learning rate (needed by the scheduler)
        balanced_loss: Whether to balance the loss by inverse class frequency
        weight_decay: The L2 weight decay for the optimizer
        dropout: The probability of dropping out the inputs to a conv block
        train_batch_size: The global batch size for training (training uses
            random cropping, so a larger batch size can be used)
        test_batch_size: The global batch size for testing
        epochs: The no. of epochs to train the model
        val_split: The fraction of training data to use for validation
        mixed_precision: Whether to use mixed precision training
        seed: The random seed for reproducibility
        crop_size: The height/width of the randomly cropped training inputs
        rotation_range: The max absolute rotation in degrees for random
            rotation of training inputs
        threshold: None means greyscale, otherwise threshold for logit.
        loss: Which loss to train with. Possible values
            ["logit_bce", "soft_dice"] (8.4.21)
        prob_fg_thresh: The probable foreground threshold for GrabCut
        unet_depth: The total depth of the UNet architecture in the U
        avgpool: Whether to use a global average pooling path as the UNet
            bottleneck (at the lowest depth)
        init_channels: The number of channels for the first layer in the
            architecture
        max_channels: The maximum number of channels for any layer in the
            architecture
        lbl_fg_thresh: The threshold for the image in [0, 1] to be labelled as
            foreground
    """

    learn_rate: float = 1e-4
    max_learn_rate: float = 2e-4
    balanced_loss: bool = False
    dropout: float = 0.2
    weight_decay: float = 5e-6
    train_batch_size: int = 16
    test_batch_size: int = 6
    epochs: int = 150
    val_split: float = 0.2
    mixed_precision: bool = False
    seed: int = 0
    crop_size: int = 128
    threshold: Optional[float] = None
    loss: str = "logit_bce"
    prob_fg_thresh: int = 32
    unet_depth: int = 7
    avgpool: bool = True
    init_channels: int = 64
    max_channels: int = 1024
    lbl_fg_thresh: float = 0.5


def load_config(config_path: Optional[Path]) -> Config:
    """Load the hyper-param config at the given path.

    If the path doesn't exist, then an empty dict is returned.
    """
    if config_path is not None and config_path.exists():
        with open(config_path, "r") as f:
            args = toml.load(f)
    else:
        args = {}
    return Config(**args)
