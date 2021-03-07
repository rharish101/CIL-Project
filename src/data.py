"""Data loading utilities."""
from pathlib import Path
from typing import Any, Optional, Tuple

from torch import Tensor
from torch.utils.data import Dataset
from torchvision.io import read_image


class TrainDataset(Dataset):
    """Dataset for the training data."""

    def __init__(
        self,
        root_dir: Path,
        input_transform: Optional[Any] = None,
        output_transform: Optional[Any] = None,
    ):
        """Load the list of images in the dataset.

        Args:
            root_dir: Path to the directory where the CIL data is extracted
            input_transform: The transformation to be applied to the input
                images
            output_transform: The transformation to be applied to the output
                images
        """
        train_dir = root_dir / "training/training"
        self.image_dir = train_dir / "images"
        self.ground_truth_dir = train_dir / "groundtruth"

        self.file_names = [path.name for path in self.image_dir.glob("*")]
        # Sort for reproducibility
        self.file_names.sort()

        self.input_transform = input_transform
        self.output_transform = output_transform

    def __len__(self) -> int:
        """Return the no. of images in the dataset."""
        return len(self.file_names)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """Get the item at the given index."""
        file_name = self.file_names[idx]
        image = read_image(str(self.image_dir / file_name))
        if self.input_transform:
            image = self.input_transform(image)
        ground_truth = read_image(str(self.ground_truth_dir / file_name))
        if self.output_transform:
            ground_truth = self.output_transform(ground_truth)
        return image, ground_truth


class TestDataset(Dataset):
    """Dataset for the test data."""

    def __init__(self, root_dir: Path, transform: Optional[Any] = None):
        """Load the list of images in the dataset.

        Args:
            root_dir: Path to the directory where the CIL data is extracted
            transform: The transformation to be applied to the images
        """
        image_dir = root_dir / "test_images/test_images"
        # Sort for reproducibility
        self.image_paths = sorted(image_dir.glob("*"))

        self.transform = transform

    def __len__(self) -> int:
        """Return the no. of images in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """Get the item at the given index."""
        image = read_image(str(self.image_paths[idx]))
        if self.transform:
            image = self.transform(image)
        return image
