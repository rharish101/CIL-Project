"""Class to train the model."""
from dataclasses import dataclass
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


@dataclass
class _Metrics:
    """Uniform structure to hold losses and metrics.

    Attributes:
        loss: The classification loss
        accuracy: The accuracy
    """

    loss: torch.Tensor = 0.0
    accuracy: torch.Tensor = 0.0
    f1_score: torch.Tensor = 0.0


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
            train_dataset,
            batch_size=config.train_batch_size,
            shuffle=True,
            pin_memory=True,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config.test_batch_size,
            # No shuffle as it won't make any difference
            pin_memory=True,
        )
        self.randomizer = get_randomizer()

        self.model = UNet(INPUT_CHANNELS, OUTPUT_CHANNELS, config).to(
            self.device
        )
        self.optim = Adam(
            self.model.parameters(),
            lr=config.learn_rate,
            weight_decay=config.weight_decay,
        )

        loss_weight = self._get_loss_weight() if config.balanced_loss else None
        # Using logits directly is numerically more stable and efficient
        self.loss = BCEWithLogitsLoss(pos_weight=loss_weight)

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

        # Pass a sample input to the model to log its computational graph
        with torch.no_grad():
            sample_images = next(iter(self.val_loader))[0].to(self.device)
            train_writer.add_graph(self.model, sample_images)

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
                with torch.no_grad():
                    acc = self._get_acc(prediction, ground_truth)
                    f1 = self._get_f1(prediction, ground_truth)
                    metrics = _Metrics(loss=loss, accuracy=acc, f1_score=f1)
                    self._log_metrics(train_writer, val_writer, metrics, step)

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
        # Map to CPU manually, as saved weights might prefer to be on the GPU
        # by default, which would crash if a GPU isn't available.
        state_dict = torch.load(load_dir / cls.SAVE_NAME, map_location="cpu")
        # Loading a GPU model's state dict from a CPU state dict works
        model.load_state_dict(state_dict)

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

    def _get_val_metrics(self) -> _Metrics:
        """Get the metrics on the validation dataset."""
        with torch.no_grad():
            metrics = _Metrics()

            for val_img, val_gt in tqdm(
                self.val_loader, desc="Validating", leave=False
            ):
                val_img = val_img.to(self.device)
                val_gt = val_gt.to(self.device)

                with autocast(enabled=self.config.mixed_precision):
                    val_pred = self.model(val_img)
                    metrics.loss += self.loss(val_pred, val_gt)

                metrics.accuracy += self._get_acc(val_pred, val_gt)
                metrics.f1_score += self._get_f1(val_pred, val_gt)

            metrics.loss /= len(self.val_loader)
            metrics.accuracy /= len(self.val_loader)
            metrics.f1_score /= len(self.val_loader)

        return metrics

    @staticmethod
    def _get_acc(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Get the mean of the accuracies for each item in the batch."""
        predictions = logits > 0
        target_bool = target > 0.5
        return (predictions == target_bool).float().mean()

    @staticmethod
    def _get_f1(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Get the mean of the F1 scores for each item in the batch.

        The F1 score is calculated per image, comparing the pixel
        classifications across the entire image. Then the sum of F1 scores of
        all images in the batch are returned.
        """
        predictions = logits > 0
        target_bool = target > 0.5

        true_pos = (predictions & target_bool).sum([1, 2, 3]).float()
        false_pos = (predictions & ~target_bool).sum([1, 2, 3]).float()
        false_neg = (~predictions & target_bool).sum([1, 2, 3]).float()

        eps = torch.finfo(true_pos.dtype).eps
        precision = true_pos / (true_pos + false_pos + eps)
        recall = true_pos / (true_pos + false_neg + eps)
        f1_score = 2 * precision * recall / (precision + recall + eps)

        return f1_score.mean()

    def _log_metrics(
        self,
        train_writer: SummaryWriter,
        val_writer: SummaryWriter,
        train_metrics: _Metrics,
        step: int,
    ) -> None:
        """Log metrics for both training and validation."""
        val_metrics = self._get_val_metrics()

        for key in vars(train_metrics):
            if key == "loss":
                tag = "losses/classification"
            else:
                tag = f"metrics/{key}"

            train_writer.add_scalar(tag, getattr(train_metrics, key), step)
            val_writer.add_scalar(tag, getattr(val_metrics, key), step)

        train_writer.add_scalar(
            "losses/regularization", self._get_l2_reg(), step
        )

        # Log a histogram for each tensor parameter in the model, to
        # see if a parameter is training stably or not
        for name, value in self.model.state_dict().items():
            train_writer.add_histogram(name, value, step)

    def _get_loss_weight(self) -> torch.Tensor:
        """Get the scalar weight for the positive class in the loss.

        This is calculated by the ratio of the negative class (class 0) to the
        positive class (class 1).
        """
        n_pos: torch.Tensor = 0.0
        n_neg: torch.Tensor = 0.0

        for _, ground_truth in self.train_loader:
            n_poss_curr = ground_truth.sum()
            n_pos += n_poss_curr
            n_neg += ground_truth.numel() - n_poss_curr

        eps = torch.finfo(n_pos.dtype).eps
        return n_neg / (n_pos + eps)
