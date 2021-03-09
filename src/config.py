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
        batch_size: The global batch size
        epochs: The no. of epochs to train the model
        val_split: The fraction of training data to use for validation
        mixed_precision: Whether to use mixed precision training
        seed: The random seed for reproducibility
    """

    learn_rate: float = 1e-3
    max_learn_rate: float = 2e-3
    balanced_loss: bool = False
    weight_decay: float = 2e-5
    batch_size: int = 16
    epochs: int = 25
    val_split: float = 0.2
    mixed_precision: bool = False
    seed: int = 0


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
