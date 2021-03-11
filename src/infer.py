"""Class to infer with the model."""
from pathlib import Path

import torch
from PIL import Image
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import Config
from .data import INPUT_CHANNELS, OUTPUT_CHANNELS, TestDataset
from .model import UNet
from .train import Trainer


class Inference:
    """Class to infer with the model."""

    def __init__(self, data_dir: Path, load_dir: Path, config: Config):
        """Store config and initialize everything.

        Args:
            data_dir: Path to the directory where the CIL data is extracted
            load_dir: Directory from where to load the model's weights
            config: The hyper-param config
        """
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.dataset = TestDataset(data_dir)
        self.loader = DataLoader(
            self.dataset,
            batch_size=config.test_batch_size,
            pin_memory=True,
        )

        self.model = UNet(INPUT_CHANNELS, OUTPUT_CHANNELS, config).to(
            self.device
        )
        Trainer.load_weights(self.model, load_dir)

        self.config = config

    def infer(self, output_dir: Path) -> None:
        """Infer with the model.

        Args:
            output_dir: Directory where to dump the model's outputs
        """
        output_dir = output_dir.expanduser()
        if not output_dir.exists():
            output_dir.mkdir(parents=True)

        with tqdm(total=len(self.dataset), desc="Inference") as progress_bar:
            for images, names in self.loader:
                images = images.to(self.device)

                with autocast(enabled=self.config.mixed_precision):
                    logits = self.model(images)

                # Get binary output of either 0.0 or 1.0
                # sigmoid(logits) > 0.5 => logits > 0
                predictions = (logits > 0).float()
                # Convert float32 in [0, 1] to uint8 in [0, 255]
                outputs = (predictions * 255).squeeze(1).byte()
                # Pillow needs numpy ndarrays; it fails with PyTorch tensors
                outputs_np = outputs.cpu().numpy()

                for img, name in zip(outputs_np, names):
                    path = output_dir / name
                    Image.fromarray(img).save(path)
                    progress_bar.update()
