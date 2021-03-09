"""Data loading utilities."""
from pathlib import Path
from typing import Callable, Tuple, TypeVar

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip,
    RandomRotation,
    RandomVerticalFlip,
)
from typing_extensions import Final

from .config import Config

INPUT_CHANNELS: Final = 3
OUTPUT_CHANNELS: Final = 1

_TransformArg = TypeVar("_TransformArg")
# Type for torchvision transforms
TransformType = Callable[[_TransformArg], _TransformArg]


class TrainDataset(Dataset):
    """Dataset for the training data."""

    def __init__(self, root_dir: Path):
        """Load the list of training images in the dataset.

        Args:
            root_dir: Path to the directory where the CIL data is extracted
        """
        train_dir = root_dir / "training/training"
        self.image_dir = train_dir / "images"
        self.ground_truth_dir = train_dir / "groundtruth"

        self.file_names = [path.name for path in self.image_dir.glob("*")]
        # Sort for reproducibility
        self.file_names.sort()

        self.transform = self.get_transform()

    def __len__(self) -> int:
        """Return the no. of images in the dataset."""
        return len(self.file_names)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the image and its ground truth at the given index."""
        file_name = self.file_names[idx]
        image = read_image(str(self.image_dir / file_name))
        ground_truth = read_image(str(self.ground_truth_dir / file_name))
        return self.transform((image, ground_truth))

    @staticmethod
    def get_transform() -> TransformType:
        """Get the transformation for the training data.

        This transform works on both the input and output simultaneously.
        """
        transforms = [
            # Scale from uint8 [0, 255] to float32 [0, 1]
            Lambda(lambda tup: tuple(i.float() / 255 for i in tup)),
            # Threshold the output image to get 0/1 labels
            Lambda(lambda tup: (tup[0], (tup[1] > 0.5).float())),
        ]
        return Compose(transforms)


class TestDataset(Dataset):
    """Dataset for the test data."""

    def __init__(self, root_dir: Path):
        """Load the list of test images in the dataset.

        Args:
            root_dir: Path to the directory where the CIL data is extracted
        """
        image_dir = root_dir / "test_images/test_images"
        # Sort for reproducibility
        self.image_paths = sorted(image_dir.glob("*"))

        self.transform = self.get_transform()

    def __len__(self) -> int:
        """Return the no. of images in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        """Get the image at the given index."""
        image_path = self.image_paths[idx]
        image = read_image(str(image_path))
        return self.transform(image), image_path.name

    @staticmethod
    def get_transform() -> TransformType:
        """Get the transformation for the test data."""
        # Scale from uint8 [0, 255] to float32 [0, 1]
        return Lambda(lambda x: x.float() / 255)


def get_randomizer(config: Config) -> TransformType:
    """Get the transformation for data augmentation.

    This performs random operations that implicitly "augment" the data, by
    creating completely new instances.
    """
    transforms = [
        # Combine them along channels so that random transforms do the same
        # rotation, crop, etc. for both batches of input and output
        Lambda(lambda tup: torch.cat(tup, 1)),
        RandomCrop(config.crop_size),
        RandomRotation(config.rotation_range),
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        # Split combined tensor into input and output
        Lambda(lambda x: (x[:, :3], x[:, 3].unsqueeze(1))),
    ]
    return Compose(transforms)
