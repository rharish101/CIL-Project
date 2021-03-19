"""Hyper-param config handling."""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import toml


@dataclass(frozen=True)
class Config:
    """Class to hold hyper-parameter configs.

    Attributes:
        gen_learn_rate: The learning rate for the generator's optimizer
        max_gen_learn_rate: The maximum learning rate for the generator (needed
            by the scheduler)
        crit_learn_rate: The learning rate for the critic's optimizer
        max_crit_learn_rate: The maximum learning rate for the critic (needed
            by the scheduler)
        balanced_loss: Whether to balance the loss by inverse class frequency
        wass_weight: The weight of the Wasserstein distance for the generator
        gen_dropout: The probability of dropping out the inputs to a conv block
            in the generator
        gen_weight_decay: The L2 weight decay for the generator's optimizer
        crit_dropout: The probability of dropping out the inputs to a conv
            block in the critic
        crit_weight_decay: The L2 weight decay for the critic's optimizer
        train_batch_size: The global batch size for training (training uses
            random cropping, so a larger batch size can be used)
        test_batch_size: The global batch size for testing
        epochs: The no. of epochs to train the model
        crit_steps: The no. of steps to train the critic per generator step
        val_split: The fraction of training data to use for validation
        mixed_precision: Whether to use mixed precision training
        seed: The random seed for reproducibility
        crop_size: The height/width of the randomly cropped training inputs
        rotation_range: The max absolute rotation in degrees for random
            rotation of training inputs
        threshold: Whether or not to threshold the image at 0.5
        loss: Which loss to train with. Possible values
            ["logit_bce", "soft_dice"] (8.4.21)
    """

    gen_learn_rate: float = 1e-4
    max_gen_learn_rate: float = 2e-4
    crit_learn_rate: float = 1e-4
    max_crit_learn_rate: float = 2e-4
    balanced_loss: bool = False
    wass_weight: float = 10.0
    gen_dropout: float = 0.2
    gen_weight_decay: float = 5e-6
    crit_dropout: float = 0.2
    crit_weight_decay: float = 5e-6
    train_batch_size: int = 16
    test_batch_size: int = 6
    epochs: int = 150
    crit_steps: int = 1
    val_split: float = 0.2
    mixed_precision: bool = False
    seed: int = 0
    crop_size: int = 128
    threshold: bool = True
    loss: str = "logit_bce"


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
