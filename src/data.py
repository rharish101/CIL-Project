"""Data loading utilities."""

from functools import lru_cache
from pathlib import Path
from typing import Callable, List, Tuple, TypeVar

import albumentations as alb
import torch
from albumentations.core.composition import Compose as AlbCompose
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import Compose, Lambda
from typing_extensions import Final

from .config import Config

INPUT_CHANNELS: Final = 3
OUTPUT_CHANNELS: Final = 1

_TransformArg = TypeVar("_TransformArg")
# Type for torchvision transforms
TransformType = Callable[[_TransformArg], _TransformArg]


class TrainDataset(Dataset):
    """Dataset for the training data."""

    def __init__(
        self,
        config: Config,
        training_path_list: List[str],
        ground_truth_path_list: List[str],
        random_augmentation=True,
    ):
        """Load the list of training images in the dataset.

        Args:
            config: config file
            training_path_list: List of paths of training images
            ground_truth_path_list: List of paths of ground truth images
            random_augmentation: Set whether randomization transformations
                should be applied on the data
        """
        self.training_path_list = training_path_list
        self.ground_truth_path_list = ground_truth_path_list

        self.transform = self.get_transform()
        self.randomizer = get_randomizer(config)
        self.random_augmentation = random_augmentation

    def __len__(self) -> int:
        """Return the no. of images in the dataset."""
        return len(self.training_path_list)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the image and its ground truth at the given index."""
        image, ground_truth = self.load_images(idx)
        image, ground_truth = self.transform((image, ground_truth))

        # Do a random transform on each entry retrieval
        if self.random_augmentation:
            image_np = image.detach().numpy()
            label_np = ground_truth.detach().numpy()
            image_np = image_np.transpose((1, 2, 0))
            label_np = label_np.transpose((1, 2, 0))
            randomized = self.randomizer(image=image_np, label=label_np)
            image = randomized["image"]
            ground_truth = randomized["label"]

        return image, ground_truth

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

    @lru_cache(maxsize=None)
    def load_images(self, idx):
        """Load training and ground truth images from storage."""
        image = read_image(self.training_path_list[idx])
        ground_truth = read_image(self.ground_truth_path_list[idx])
        return image, ground_truth


class TestDataset(Dataset):
    """Dataset for the test data."""

    def __init__(self, image_dir: Path):
        """Load the list of test images in the dataset.

        Args:
            image_dir: Path to the directory containing the input images
        """
        # Sort for reproducibility
        self.image_paths = sorted(image_dir.expanduser().glob("*"))
        self.transform = self.get_transform()

    def __len__(self) -> int:
        """Return the no. of images in the dataset."""
        return len(self.image_paths)

    @lru_cache(maxsize=None)
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


def get_randomizer(config: Config) -> AlbCompose:
    """Get the transformation for data augmentation.

    This performs random operations that implicitly "augment" the data, by
    creating completely new instances.
    """
    transforms = [
        # Combine them along channels so that random transforms do the same
        # rotation, crop, etc. for both batches of input and output
        alb.RandomCrop(config.crop_size, config.crop_size),
        # Randomly rotate by 90 degrees.
        alb.RandomRotate90(),
        alb.HorizontalFlip(),
        alb.VerticalFlip(),
        alb.ElasticTransform(),
        ToTensorV2(),
    ]
    return alb.Compose(transforms, additional_targets={"label": "image"})


def get_file_paths(
    root_dir: Path,
) -> Tuple[List[str], List[str]]:
    """Load the list of training and ground truth image paths.

    Args:
        root_dir: Path to the directory where the CIL data is extracted

    Returns:
        List of training image paths
        List of ground truth image paths
    """
    train_dir = root_dir.expanduser() / "training/training"
    image_dir = train_dir / "images"
    ground_truth_dir = train_dir / "groundtruth"

    file_names = [path.name for path in image_dir.glob("*")]
    # Sort for reproducibility
    file_names.sort()

    training_path_list = [
        str(image_dir / file_name) for file_name in file_names
    ]
    ground_truth_path_list = [
        str(ground_truth_dir / file_name) for file_name in file_names
    ]

    return training_path_list, ground_truth_path_list
