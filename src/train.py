"""Class to train the model."""
from datetime import datetime
from pathlib import Path
from typing import Tuple

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

from .config import Config
from .data import INPUT_CHANNELS, OUTPUT_CHANNELS, TrainDataset, get_randomizer
from .model import UNet


class Trainer:
    """Class to train the model."""

    SAVE_NAME: Final = "model.pt"  # for the model's weights
    CONFIG_NAME: Final = "config.toml"  # for info on hyper-params

    def __init__(self, data_dir: Path, config: Config):
        """Store config and initialize everything.

        Args:
            data_dir: Path to the directory where the CIL data is extracted
            config: The hyper-param config
        """
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        dataset = TrainDataset(data_dir)
        val_length = int(len(dataset) * config.val_split)
        train_dataset, val_dataset = random_split(
            dataset, [len(dataset) - val_length, val_length]
        )
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            pin_memory=True,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            pin_memory=True,
        )
        self.randomizer = get_randomizer()

        self.model = UNet(INPUT_CHANNELS, OUTPUT_CHANNELS).to(self.device)
        self.optim = Adam(
            self.model.parameters(),
            lr=config.learn_rate,
            weight_decay=config.weight_decay,
        )
        self.loss = BCEWithLogitsLoss()

        max_steps = config.epochs * len(self.train_loader)
        self.scheduler = OneCycleLR(
            self.optim,
            max_lr=config.max_learn_rate,
            total_steps=max_steps,
        )
        self.scaler = GradScaler(enabled=config.mixed_precision)

        # Used when dumping hyper-params to a file
        self.config = config

    def train(
        self,
        save_dir: Path,
        log_dir: Path,
        save_steps: int,
        log_steps: int,
    ) -> None:
        """Train the model.

        Args:
            save_dir: Directory where to save the model's weights
            log_dir: Directory where to log metrics
            save_steps: Step interval for saving the model's weights
            log_steps: Step interval for logging metrics
        """
        train_writer, val_writer = self._setup_dirs(save_dir, log_dir)

        # Iterate step-by-step for a combined progress bar, and for automatic
        # step counting through enumerate
        iterator = (
            data
            for epoch in range(self.config.epochs)
            for data in self.train_loader
        )
        max_steps = self.config.epochs * len(self.train_loader)

        for step, (image, ground_truth) in enumerate(
            tqdm(iterator, total=max_steps, desc="Training"), 1
        ):
            self.optim.zero_grad()

            image, ground_truth = self.randomizer((image, ground_truth))
            image = image.to(self.device)
            ground_truth = ground_truth.to(self.device)

            with autocast(enabled=self.config.mixed_precision):
                prediction = self.model(image)
                loss = self.loss(prediction, ground_truth)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
            self.scheduler.step()

            if step % save_steps == 0:
                self.save_weights(save_dir)

            if step % log_steps == 0:
                self._log_metrics(train_writer, val_writer, loss, step)

        self.save_weights(save_dir)

    def save_weights(self, save_dir: Path) -> None:
        """Save the model's weights.

        Args:
            save_dir: Directory where to save the model's weights
        """
        torch.save(self.model.state_dict(), save_dir / self.SAVE_NAME)

    @classmethod
    def load_weights(cls, model: Module, load_dir: Path) -> None:
        """Load the model's weights in-place.

        Args:
            model: The model whose weights are to be replaced in-place with the
                loaded weights
            load_dir: Directory from where to load the model's weights
        """
        model.load_state_dict(torch.load(load_dir / cls.SAVE_NAME))

    def _setup_dirs(
        self, save_dir: Path, log_dir: Path
    ) -> Tuple[SummaryWriter, SummaryWriter]:
        """Setup the save and log directories and return summary writers.

        This creates two summary writers, each for training and validation,
        that log to a timestamped directory.
        """
        if not save_dir.exists():
            save_dir.mkdir(parents=True)

        # Log to a timestamped directory, since we don't want to accidently
        # overwrite older logs
        curr_date = datetime.now().astimezone()
        timestamped_log_dir = log_dir / curr_date.isoformat()
        timestamped_log_dir.mkdir(parents=True)

        # Save hyper-params as a TOML file for reference
        config = {**vars(self.config), "date": curr_date}
        for dest in save_dir, timestamped_log_dir:
            with open(dest / self.CONFIG_NAME, "w") as f:
                toml.dump(config, f)

        # Use separate summary writers so that training and validation losses
        # can be viewed on the same graph in TensorBoard
        train_writer = SummaryWriter(str(timestamped_log_dir / "training"))
        val_writer = SummaryWriter(str(timestamped_log_dir / "validation"))

        return train_writer, val_writer

    def _get_l2_reg(self) -> torch.Tensor:
        """Get the L2 regularization value for the model."""
        loss = 0
        for param in self.model.parameters():
            loss += (param ** 2).sum()
        return loss

    def _get_val_loss(self) -> torch.Tensor:
        """Get the loss on the validation dataset."""
        with torch.no_grad():
            val_loss = 0.0

            for val_img, val_gt in tqdm(
                self.val_loader, desc="Validating", leave=False
            ):
                val_img = val_img.to(self.device)
                val_gt = val_gt.to(self.device)

                with autocast(enabled=self.config.mixed_precision):
                    val_pred = self.model(val_img)
                    val_loss += self.loss(val_pred, val_gt)

            return val_loss / len(self.val_loader)

    def _log_metrics(
        self,
        train_writer: SummaryWriter,
        val_writer: SummaryWriter,
        train_loss: torch.Tensor,
        step: int,
    ) -> None:
        """Log metrics for both training and validation."""
        train_writer.add_scalar("losses/classification", train_loss, step)
        val_writer.add_scalar(
            "losses/classification", self._get_val_loss(), step
        )
        train_writer.add_scalar(
            "losses/regularization", self._get_l2_reg(), step
        )

        # Log a histogram for each tensor parameter in the model, to
        # see if a parameter is training stably or not
        for name, value in self.model.state_dict().items():
            train_writer.add_histogram(name, value, step)
