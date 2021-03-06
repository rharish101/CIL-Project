"""Data loading utilities."""
from functools import lru_cache
from pathlib import Path
from typing import Callable, List, Tuple, TypeVar

import albumentations as alb
import numpy as np
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
        random_augmentation: bool = True,
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
        self.pair_randomizer, self.input_randomizer = get_randomizer(config)
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
            # input only transformations
            image_np = self.input_randomizer(image=image_np)["image"]
            # pair transformations
            randomized = self.pair_randomizer(image=image_np, label=label_np)
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


class EvalDataset(Dataset):
    """Dataset for the ground truth and predictions."""

    def __init__(
        self,
        config: Config,
        ground_truth_path_list: List[str],
        pred_dir: Path,
    ):
        """Load the list of ground truth and predictions in the dataset.

        Args:
            config: config file
            ground_truth_path_list: List of paths of ground truth images
            pred_dir: Directory containing the model's predictions for the
                input data
        """
        self.ground_truth_path_list = ground_truth_path_list
        self.pred_dir = pred_dir
        self.transform = self.get_transform()

    def __len__(self) -> int:
        """Return the no. of images in the dataset."""
        return len(self.ground_truth_path_list)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the image and its ground truth at the given index."""
        return self.transform(self.load_images(idx))

    @staticmethod
    def get_transform() -> TransformType:
        """Get the transformation for the data.

        This transform works on both the input and output simultaneously.
        """
        # Scale from uint8 [0, 255] to float32 [0, 1]
        return Lambda(lambda tup: tuple(i.float() / 255 for i in tup))

    @lru_cache(maxsize=None)
    def load_images(self, idx):
        """Load training and ground truth images from storage."""
        ground_truth_path = self.ground_truth_path_list[idx]
        ground_truth = read_image(ground_truth_path)

        file_name = Path(ground_truth_path).name
        prediction = read_image(str(self.pred_dir / file_name))

        return ground_truth, prediction


def get_randomizer(config: Config) -> Tuple[AlbCompose, AlbCompose]:
    """Get the transformations for data augmentation.

    This performs random operations that implicitly "augment" the data, by
    creating completely new instances.
    """
    pair_transforms = [alb.RandomCrop(config.crop_size, config.crop_size)]
    if config.extra_augmentations:
        pair_transforms += [
            alb.RandomRotate90(),
            alb.HorizontalFlip(),
            alb.VerticalFlip(),
            # Deformation
            alb.OneOf(
                [
                    alb.ElasticTransform(),
                    alb.GridDistortion(),
                    alb.OpticalDistortion(),
                ],
                p=0.5,
            ),
        ]
    else:
        pair_transforms.append(alb.ElasticTransform())
    pair_transforms.append(ToTensorV2())

    if config.extra_augmentations:
        input_transforms = [
            # Color transforms
            alb.RandomBrightnessContrast(),
            alb.ColorJitter(),
            alb.GaussianBlur(),
        ]
    else:
        input_transforms = []

    return (
        alb.Compose(pair_transforms, additional_targets={"label": "image"}),
        alb.Compose(input_transforms),
    )


def get_texture_transform(config: Config) -> TransformType[np.ndarray]:
    """Get a batch transform that changes textures but preserves shapes."""
    transforms = [
        alb.FromFloat("uint8"),
        alb.GaussianBlur(),
        alb.Downscale(
            scale_min=config.downscale_min, scale_max=config.downscale_max
        ),
        alb.ImageCompression(
            quality_lower=config.compress_quality_lower,
            quality_upper=config.compress_quality_upper,
        ),
        alb.GaussNoise(),
        alb.ISONoise(),
        alb.ToFloat(),
        ToTensorV2(),
    ]
    combined = alb.Compose(transforms)
    return lambda batch_x: np.stack(
        [combined(image=x)["image"] for x in batch_x]
    )


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
