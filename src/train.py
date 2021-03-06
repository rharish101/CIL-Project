"""Class to train the model."""
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import toml
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.nn import BCEWithLogitsLoss, DataParallel, LogSoftmax, Module
from torch.nn.functional import cosine_similarity
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from typing_extensions import Final

from .config import Config
from .data import (
    INPUT_CHANNELS,
    OUTPUT_CHANNELS,
    TrainDataset,
    get_file_paths,
    get_texture_transform,
)
from .model import UNet
from .soft_dice_loss import soft_dice_loss


@dataclass
class _Metrics:
    """Uniform structure to hold losses and metrics.

    Attributes:
        class_loss: The classification loss
        shape_loss: The shape constraint loss
        total_loss: The total weighted loss
        accuracy: The accuracy
        f1_score: The F1 score
    """

    class_loss: torch.Tensor = 0.0
    shape_loss: torch.Tensor = 0.0
    total_loss: torch.Tensor = 0.0
    accuracy: torch.Tensor = 0.0
    f1_score: torch.Tensor = 0.0


class ContrastiveLoss(Module):
    """A contrastive loss adapted from SimCLR.

    Link to SimCLR: https://arxiv.org/abs/2002.05709v3.
    """

    def __init__(self, temperature: float = 1.0):
        """Save hyper-params."""
        super().__init__()
        self.temperature = temperature
        self._log_softmax_fn = LogSoftmax(dim=-1)

    def forward(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Get the loss."""
        # BxCxHxW => HxWxBxC == ...xNxD
        inputs = inputs.permute(2, 3, 0, 1)
        targets = targets.permute(2, 3, 0, 1)

        batch_size = inputs.shape[-2]
        left = torch.cat([inputs, targets], -2).unsqueeze(-1)  # ...x2NxDx1
        right = left.permute(0, 1, 4, 3, 2)  # ...x1xDx2N

        # Get all-pairs cosine similarity, like ...x2NxD @ ...xDx2N
        similarity = cosine_similarity(
            left, right, dim=-2, eps=torch.finfo(left.dtype).eps
        )  # Now ...x2Nx2N

        # Mask out the self values
        mask = torch.eye(2 * batch_size, device=similarity.device).bool()
        mask_nd = (
            mask.unsqueeze(0).unsqueeze(0).tile(*similarity.shape[:2], 1, 1)
        )
        neg_inf = float("-inf") * torch.ones_like(similarity)
        similarity = torch.where(mask_nd, neg_inf, similarity)

        log_softmax = self._log_softmax_fn(similarity / self.temperature)

        # All positive pairs are (i, N+i mod 2N)
        # - - - x - -
        # - - - - x -
        # - - - - - x
        # x - - - - -
        # - x - - - -
        # - - x - - -
        positive_pairs = torch.cat(
            [
                torch.diagonal(
                    log_softmax, offset=batch_size, dim1=-2, dim2=-1
                ),  # ...xN
                torch.diagonal(
                    log_softmax, offset=-batch_size, dim1=-2, dim2=-1
                ),  # ...xN
            ],
            -1,
        )  # ...x2N
        return -(positive_pairs).mean()


class Trainer:
    """Class to train the model."""

    SAVE_NAME: Final = "model.pt"  # for the model's weights
    BEST_SAVE_NAME: Final = "best-model.pt"  # for the best-model (accuracy)
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

        training_path_list, ground_truth_path_list = get_file_paths(data_dir)

        X_train, X_test, y_train, y_test = self.train_test_split(
            training_path_list,
            ground_truth_path_list,
            test_portion=config.val_split,
        )

        train_dataset = TrainDataset(
            config, X_train, y_train, random_augmentation=True
        )
        val_dataset = TrainDataset(
            config, X_test, y_test, random_augmentation=False
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

        model = UNet(INPUT_CHANNELS, OUTPUT_CHANNELS, config)
        self.model = DataParallel(model).to(self.device)

        if config.loss == "logit_bce":
            loss_weight = (
                self._get_loss_weight() if config.balanced_loss else None
            )
            # Using logits directly is numerically more stable and efficient
            self.class_loss_fn = BCEWithLogitsLoss(pos_weight=loss_weight)
        elif config.loss == "soft_dice":
            self.class_loss_fn = soft_dice_loss

        self.texture_transform = get_texture_transform(config)
        self.shape_loss_fn = ContrastiveLoss(config.temperature)

        self.optim = Adam(
            self.model.parameters(),
            lr=config.learn_rate,
            weight_decay=config.weight_decay,
        )
        max_steps = config.epochs * len(self.train_loader)
        self.scheduler = OneCycleLR(
            self.optim,
            max_lr=config.max_learn_rate,
            total_steps=max_steps,
        )
        self.scaler = GradScaler(enabled=config.mixed_precision)

        # Used when dumping hyper-params to a file
        self.config = config

        # To store best acc achieved so far
        self.best_acc = 0.0

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
        train_writer, val_writer, timestamped_save_dir = self._setup_dirs(
            save_dir, log_dir
        )

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
            # Turn on batch-norm updates
            self.model.train()
            self.optim.zero_grad()

            image = image.to(self.device)
            ground_truth = ground_truth.to(self.device)

            with autocast(enabled=self.config.mixed_precision):
                prediction, latent = self.model(image)
                class_loss = self.class_loss_fn(prediction, ground_truth)

                if self.config.shape_loss_weight > 0:
                    image_np = np.moveaxis(image.cpu().numpy(), 1, 3)
                    other_image_np = self.texture_transform(image_np)
                    other_image = torch.from_numpy(other_image_np).to(
                        self.device
                    )
                    other_latent = self.model(other_image, only_latent=True)[1]
                    shape_loss = self.shape_loss_fn(latent, other_latent)
                else:
                    shape_loss = 0.0

                loss = class_loss + self.config.shape_loss_weight * shape_loss

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
            self.scheduler.step()

            if step % save_steps == 0:
                self.save_weights(timestamped_save_dir)

            if step % log_steps == 0:
                with torch.no_grad():
                    acc = self._get_acc(prediction, ground_truth)
                    f1 = self._get_f1(prediction, ground_truth)
                    metrics = _Metrics(
                        class_loss=class_loss,
                        shape_loss=shape_loss,
                        total_loss=loss,
                        accuracy=acc,
                        f1_score=f1,
                    )
                    self._log_metrics(
                        train_writer,
                        val_writer,
                        timestamped_save_dir,
                        metrics,
                        step,
                    )

        self.save_weights(timestamped_save_dir)

    def save_weights(self, save_dir: Path, is_best: bool = False) -> None:
        """Save the model's weights.

        Args:
            save_dir: Directory where to save the model's weights
            is_best: To check whether the weights should be saved
                for the best model (wrt accuracy)
        """
        if is_best:
            save_path = save_dir.expanduser() / self.BEST_SAVE_NAME
        else:
            save_path = save_dir.expanduser() / self.SAVE_NAME
        torch.save(self.model.state_dict(), save_path)

    @classmethod
    def load_weights(
        cls, model: Module, load_dir: Path, use_best_model: bool
    ) -> None:
        """Load the model's weights in-place.

        Args:
            model: The model whose weights are to be replaced in-place with the
                loaded weights
            load_dir: Directory from where to load the model's weights
            use_best_model: Whether to load best_model weights (wrt accuracy)
        """
        load_dir = load_dir.expanduser()
        if use_best_model:
            load_path = cls.BEST_SAVE_NAME
        else:
            load_path = cls.SAVE_NAME
        # Map to CPU manually, as saved weights might prefer to be on the GPU
        # by default, which would crash if a GPU isn't available.
        state_dict = torch.load(load_dir / load_path, map_location="cpu")
        # Loading a GPU model's state dict from a CPU state dict works
        model.load_state_dict(state_dict)

    def _setup_dirs(
        self, save_dir: Path, log_dir: Path
    ) -> Tuple[SummaryWriter, SummaryWriter, Path]:
        """Setup the save and log directories and return summary writers.

        This creates two summary writers, each for training and validation,
        that log to a timestamped directory. This also creates a timestamped
        directory (inside the given save directory) for saving the models.

        Args:
            save_dir: The directory where to save the models
            log_dir: The directory where to dump the logs

        Returns:
            The training summary writer
            The validation summary writer
            The path to the timestamped save directory
        """
        save_dir = save_dir.expanduser()
        log_dir = log_dir.expanduser()

        # Log and save to a timestamped directory, since we don't want to
        # accidently overwrite older logs and models
        curr_date = datetime.now().astimezone()

        timestamped_log_dir = log_dir / curr_date.isoformat()
        try:
            timestamped_log_dir.mkdir(parents=True)
        except Exception:
            # had to add this case because the original isoformat
            # can't be used as dir_name in Windows
            timestamped_log_dir = log_dir / curr_date.isoformat().replace(
                ":", "."
            )
            timestamped_log_dir.mkdir(parents=True)

        timestamped_save_dir = save_dir / curr_date.isoformat()
        timestamped_save_dir.mkdir(parents=True)

        # Save hyper-params as a TOML file for reference
        config = {**vars(self.config), "date": curr_date}
        for dest in timestamped_save_dir, timestamped_log_dir:
            with open(dest / self.CONFIG_NAME, "w") as f:
                toml.dump(config, f)

        # Use separate summary writers so that training and validation losses
        # can be viewed on the same graph in TensorBoard
        train_writer = SummaryWriter(str(timestamped_log_dir / "training"))
        val_writer = SummaryWriter(str(timestamped_log_dir / "validation"))

        return train_writer, val_writer, timestamped_save_dir

    def _get_l2_reg(self) -> torch.Tensor:
        """Get the L2 regularization value for the model."""
        loss = 0
        for param in self.model.parameters():
            loss += (param ** 2).sum()
        return loss

    def _get_val_metrics(
        self,
    ) -> Tuple[_Metrics, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get the metrics on the validation dataset.

        This also returns the inputs, ground-truth, and predictions for the
        last batch of images.
        """
        # Turn off batch-norm updates
        self.model.eval()

        with torch.no_grad():
            metrics = _Metrics()

            for val_img, val_gt in tqdm(
                self.val_loader, desc="Validating", leave=False
            ):
                val_img = val_img.to(self.device)
                val_gt = val_gt.to(self.device)

                with autocast(enabled=self.config.mixed_precision):
                    val_pred = self.model(val_img)[0]
                    metrics.class_loss += self.class_loss_fn(val_pred, val_gt)

                metrics.accuracy += self._get_acc(val_pred, val_gt)
                metrics.f1_score += self._get_f1(val_pred, val_gt)

            metrics.class_loss /= len(self.val_loader)
            metrics.accuracy /= len(self.val_loader)
            metrics.f1_score /= len(self.val_loader)

        return metrics, val_img, val_gt, torch.sigmoid(val_pred)

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
        timestamped_save_dir: Path,
        train_metrics: _Metrics,
        step: int,
    ) -> None:
        """Log metrics for both training and validation."""
        if len(self.val_loader) > 0:
            val_metrics, val_img, val_gt, val_pred = self._get_val_metrics()
            if val_metrics.accuracy > self.best_acc:
                self.best_acc = val_metrics.accuracy
                self.save_weights(timestamped_save_dir, True)

        for key in vars(train_metrics):
            if key == "class_loss":
                tag = "losses/classification"
            elif key in {"shape_loss", "total_loss"}:
                continue
            else:
                tag = f"metrics/{key}"

            train_writer.add_scalar(tag, getattr(train_metrics, key), step)
            if len(self.val_loader) > 0:
                val_writer.add_scalar(tag, getattr(val_metrics, key), step)

        reg_loss = self._get_l2_reg()
        train_writer.add_scalar("losses/regularization", reg_loss, step)
        train_writer.add_scalar("losses/shape", train_metrics.shape_loss, step)
        train_writer.add_scalar(
            "losses/total",
            train_metrics.total_loss + self.config.weight_decay * reg_loss,
            step,
        )

        # Log a histogram for each tensor parameter in the model, to
        # see if a parameter is training stably or not
        for name, value in self.model.state_dict().items():
            train_writer.add_histogram(name, value, step)

        # Log the validation images for easy visualization
        if len(self.val_loader) > 0:
            val_writer.add_images("input", val_img, step)
            val_writer.add_images("ground_truth", val_gt, step)
            val_writer.add_images("prediction", val_pred, step)

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

    @staticmethod
    def train_test_split(X, y, test_portion):
        """Splits training data into train test list wrt the test portion."""
        joint_list = list(zip(X, y))
        random.shuffle(joint_list)

        shuffled_X, shuffled_y = zip(*joint_list)
        pivot = int(len(X) * test_portion)

        return (
            shuffled_X[pivot:],
            shuffled_X[:pivot],
            shuffled_y[pivot:],
            shuffled_y[:pivot],
        )
