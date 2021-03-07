"""Class to train the model."""
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import toml
import torch
from torch.nn import BCEWithLogitsLoss, Module
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from typing_extensions import Final

from .data import INPUT_CHANNELS, OUTPUT_CHANNELS, TrainDataset
from .model import UNet


class Trainer:
    """Class to train the model."""

    SAVE_NAME: Final = "model.pt"
    CONFIG_NAME: Final = "config.toml"

    def __init__(self, data_dir: Path, learn_rate: float, weight_decay: float):
        """Store config and initialize everything."""
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.dataset = TrainDataset(data_dir)

        self.model = UNet(INPUT_CHANNELS, OUTPUT_CHANNELS).to(self.device)
        self.optim = Adam(
            self.model.parameters(), lr=learn_rate, weight_decay=weight_decay
        )
        self.loss = BCEWithLogitsLoss()

        # Used when dumping hyper-params to a file
        self.config: Final = {
            "learn_rate": learn_rate,
            "weight_decay": weight_decay,
        }

    def train(
        self,
        batch_size: int,
        max_epochs: int,
        save_dir: Path,
        save_steps: int,
        log_dir: Path,
        log_steps: int,
    ) -> None:
        """Train the model."""
        loader = DataLoader(
            self.dataset, batch_size=batch_size, shuffle=True, pin_memory=True
        )

        writer = SummaryWriter(str(log_dir))

        # Save hyper-params as a TOML file for reference
        config: Dict[str, Any] = {
            **self.config,
            "batch_size": batch_size,
            "max_epochs": max_epochs,
            "date": datetime.now().astimezone(),
        }
        for dest in save_dir, log_dir:
            with open(dest / self.CONFIG_NAME, "w") as f:
                toml.dump(config, f)

        # Iterate step-by-step for a combined progress bar, and for automatic
        # step counting through enumerate.
        iterator = (data for epoch in range(max_epochs) for data in loader)
        iter_len = max_epochs * len(loader)

        for step, (image, ground_truth) in enumerate(
            tqdm(iterator, total=iter_len, desc="Training"), 1
        ):
            self.optim.zero_grad()

            image = image.to(self.device)
            ground_truth = ground_truth.to(self.device)

            prediction = self.model(image)
            loss = self.loss(prediction, ground_truth)
            loss.backward()
            self.optim.step()

            if step % save_steps == 0:
                self.save_weights(save_dir)

            if step % log_steps == 0:
                writer.add_scalar("losses/classification", loss, step)
                writer.add_scalar(
                    "losses/regularization", self._get_l2_reg(), step
                )
                for name, value in self.model.state_dict().items():
                    writer.add_histogram(name, value, step)

        self.save_weights(save_dir)

    def save_weights(self, save_dir: Path) -> None:
        """Save the model's weights."""
        if not save_dir.exists():
            save_dir.mkdir(parents=True)
        torch.save(self.model.state_dict(), save_dir / self.SAVE_NAME)

    @classmethod
    def load_weights(cls, model: Module, load_dir: Path) -> None:
        """Load the model's weights."""
        model.load_state_dict(torch.load(load_dir / cls.SAVE_NAME))

    def _get_l2_reg(self) -> torch.Tensor:
        """Get the L2 regularization value for the model."""
        loss = 0
        for param in self.model.parameters():
            loss += (param ** 2).sum()
        return loss