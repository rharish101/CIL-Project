"""Class to train the model."""
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import toml
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.nn import BCEWithLogitsLoss, Module
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from typing_extensions import Final

from .data import INPUT_CHANNELS, OUTPUT_CHANNELS, TrainDataset, get_randomizer
from .model import UNet


class Trainer:
    """Class to train the model."""

    SAVE_NAME: Final = "model.pt"
    CONFIG_NAME: Final = "config.toml"

    def __init__(
        self,
        data_dir: Path,
        learn_rate: float,
        weight_decay: float,
        val_split: float,
    ):
        """Store config and initialize everything."""
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        dataset = TrainDataset(data_dir)
        val_length = int(len(dataset) * val_split)
        self.train_dataset, self.val_dataset = random_split(
            dataset, [len(dataset) - val_length, val_length]
        )

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
        max_learn_rate: float,
        save_dir: Path,
        save_steps: int,
        log_dir: Path,
        log_steps: int,
        mixed_precision: bool = False,
    ) -> None:
        """Train the model."""
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
        )
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            pin_memory=True,
        )
        randomizer = get_randomizer()

        curr_date = datetime.now().astimezone()
        timestamped_log_dir = log_dir / curr_date.isoformat()
        timestamped_log_dir.mkdir(parents=True)

        # Save hyper-params as a TOML file for reference
        config: Dict[str, Any] = {
            **self.config,
            "batch_size": batch_size,
            "max_epochs": max_epochs,
            "max_learn_rate": max_learn_rate,
            "mixed_precision": mixed_precision,
            "date": datetime.now().astimezone(),
        }
        for dest in save_dir, timestamped_log_dir:
            with open(dest / self.CONFIG_NAME, "w") as f:
                toml.dump(config, f)

        train_writer = SummaryWriter(str(timestamped_log_dir / "training"))
        val_writer = SummaryWriter(str(timestamped_log_dir / "validation"))

        # Iterate step-by-step for a combined progress bar, and for automatic
        # step counting through enumerate.
        iterator = (
            data for epoch in range(max_epochs) for data in train_loader
        )
        max_steps = max_epochs * len(train_loader)

        scheduler = OneCycleLR(
            self.optim, max_lr=max_learn_rate, total_steps=max_steps
        )
        scaler = GradScaler(enabled=mixed_precision)

        for step, (image, ground_truth) in enumerate(
            tqdm(iterator, total=max_steps, desc="Training"), 1
        ):
            self.optim.zero_grad()

            image, ground_truth = randomizer((image, ground_truth))
            image = image.to(self.device)
            ground_truth = ground_truth.to(self.device)

            with autocast(enabled=mixed_precision):
                prediction = self.model(image)
                loss = self.loss(prediction, ground_truth)

            scaler.scale(loss).backward()
            scaler.step(self.optim)
            scaler.update()
            scheduler.step()

            if step % save_steps == 0:
                self.save_weights(save_dir)

            if step % log_steps == 0:
                with torch.no_grad():
                    val_loss = 0.0

                    for val_img, val_gt in tqdm(
                        val_loader, desc="Validating", leave=False
                    ):
                        val_img = val_img.to(self.device)
                        val_gt = val_gt.to(self.device)

                        with autocast():
                            val_pred = self.model(val_img)
                            val_loss += self.loss(val_pred, val_gt)

                    val_loss /= len(val_loader)

                train_writer.add_scalar("losses/classification", loss, step)
                val_writer.add_scalar("losses/classification", val_loss, step)
                train_writer.add_scalar(
                    "losses/regularization", self._get_l2_reg(), step
                )

                for name, value in self.model.state_dict().items():
                    train_writer.add_histogram(name, value, step)

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
