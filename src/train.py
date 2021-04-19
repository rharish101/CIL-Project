"""Class to train the model."""
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Tuple

import toml
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.nn import BCEWithLogitsLoss, DataParallel, Module
from torch.nn.functional import sigmoid
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from typing_extensions import Final

from .config import Config
from .data import INPUT_CHANNELS, OUTPUT_CHANNELS, TrainDataset, get_file_paths
from .model import Critic, UNet
from .soft_dice_loss import soft_dice_loss


@dataclass
class _Metrics:
    """Uniform structure to hold losses and metrics.

    Attributes:
        loss: The classification loss
        wass: The Wasserstein distance
        accuracy: The accuracy
        f1_score: The F1 score
    """

    loss: torch.Tensor = 0.0
    wass: torch.Tensor = 0.0
    accuracy: torch.Tensor = 0.0
    f1_score: torch.Tensor = 0.0


class Trainer:
    """Class to train the WGAN.

    Original paper: https://arxiv.org/abs/1701.07875.
    """

    GEN_SAVE_NAME: Final = "generator.pt"  # for the generator's weights
    CRIT_SAVE_NAME: Final = "critic.pt"  # for the critic's weights
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
            config, X_train, y_train, random_augmentation=False
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

        if config.loss == "logit_bce":
            loss_weight = (
                self._get_loss_weight() if config.balanced_loss else None
            )
            # Using logits directly is numerically more stable and efficient
            self.loss = BCEWithLogitsLoss(pos_weight=loss_weight)
        elif config.loss == "soft_dice":
            self.loss = soft_dice_loss

        generator = UNet(INPUT_CHANNELS, OUTPUT_CHANNELS, config)
        self.generator = DataParallel(generator).to(self.device)
        self.gen_optim = Adam(
            self.generator.parameters(),
            lr=config.gen_learn_rate,
            weight_decay=config.gen_weight_decay,
        )
        max_steps = config.epochs * len(self.train_loader)
        self.gen_scheduler = OneCycleLR(
            self.gen_optim,
            max_lr=config.max_gen_learn_rate,
            total_steps=max_steps,
        )

        critic = Critic(OUTPUT_CHANNELS, config)
        self.critic = DataParallel(critic).to(self.device)
        self.crit_optim = Adam(
            self.critic.parameters(),
            lr=config.crit_learn_rate,
            weight_decay=config.crit_weight_decay,
        )
        self.crit_scheduler = OneCycleLR(
            self.crit_optim,
            max_lr=config.max_crit_learn_rate,
            total_steps=max_steps * config.crit_steps,
        )

        self.scaler = GradScaler(enabled=config.mixed_precision)

        # Used when dumping hyper-params to a file
        self.config = config

    def _train_step_gan(
        self, image: torch.Tensor, ground_truth: torch.Tensor
    ) -> _Metrics:
        """Run a single training step for the GAN."""
        # Turn on batch-norm updates for the generator
        self.generator.train()
        self.gen_optim.zero_grad()

        with autocast(enabled=self.config.mixed_precision):
            prediction = self.generator(image)
            loss = self.loss(prediction, ground_truth)

        # Turn on batch-norm updates for the critic
        self.critic.train()
        for _ in range(self.config.crit_steps):
            self._train_step_critic(prediction, ground_truth)

        # Turn off batch-norm updates for the critic, as we're just using it
        # for getting the Wasserstein distance
        self.critic.eval()
        with autocast(enabled=self.config.mixed_precision):
            fake_out = self.critic(sigmoid(prediction))
            real_out = self.critic(ground_truth)
            wass_dist = (real_out - fake_out).mean()
            total_loss = loss + self.config.wass_weight * wass_dist

        self.scaler.scale(total_loss).backward()
        self.scaler.step(self.gen_optim)
        self.scaler.update()
        self.gen_scheduler.step()

        acc = self._get_acc(prediction, ground_truth)
        f1 = self._get_f1(prediction, ground_truth)
        return _Metrics(loss=loss, wass=wass_dist, accuracy=acc, f1_score=f1)

    def _train_step_critic(
        self, logits: torch.Tensor, ground_truth: torch.Tensor
    ) -> None:
        """Run a single training step for the critic."""
        self.crit_optim.zero_grad()

        with autocast(enabled=self.config.mixed_precision):
            # Detach inputs so that PyTorch doesn't propagate gradients behind
            # the critic's inputs. Otherwise, after optimizing the critic, the
            # generator's computational graph will be erased.
            fake_out = self.critic(sigmoid(logits.detach()))
            real_out = self.critic(ground_truth)

            wass_dist = real_out - fake_out
            loss = (-wass_dist).mean()

        self.scaler.scale(loss).backward()
        self.scaler.step(self.crit_optim)
        self.scaler.update()
        self.crit_scheduler.step()

    def train(
        self,
        save_dir: Path,
        log_dir: Path,
        save_steps: int,
        log_steps: int,
    ) -> None:
        """Train the model.

        Args:
            save_dir: Directory where to save all models' weights
            log_dir: Directory where to log metrics
            save_steps: Step interval for saving all models' weights
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
            image = image.to(self.device)
            ground_truth = ground_truth.to(self.device)

            metrics = self._train_step_gan(image, ground_truth)

            if step % save_steps == 0:
                self.save_weights(timestamped_save_dir)

            if step % log_steps == 0:
                with torch.no_grad():
                    self._log_metrics(train_writer, val_writer, metrics, step)

        self.save_weights(timestamped_save_dir)

    def save_weights(self, save_dir: Path) -> None:
        """Save all models' weights.

        Args:
            save_dir: Directory where to save the weights
        """
        save_dir = save_dir.expanduser()
        for model, name in [
            (self.generator, self.GEN_SAVE_NAME),
            (self.critic, self.CRIT_SAVE_NAME),
        ]:
            save_path = save_dir / name
            torch.save(model.state_dict(), save_path)

    @classmethod
    def load_weights(cls, generator: Module, load_dir: Path) -> None:
        """Load the generator's weights in-place.

        Args:
            generator: The generator whose weights are to be replaced in-place
                with the loaded weights
            load_dir: Directory from where to load the model's weights
        """
        load_dir = load_dir.expanduser()
        # Map to CPU manually, as saved weights might prefer to be on the GPU
        # by default, which would crash if a GPU isn't available.
        state_dict = torch.load(
            load_dir / cls.GEN_SAVE_NAME, map_location="cpu"
        )
        # Loading a GPU model's state dict from a CPU state dict works
        generator.load_state_dict(state_dict)

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

    @staticmethod
    def _get_l2_reg(model: Module) -> torch.Tensor:
        """Get the L2 regularization value for the model."""
        loss = 0
        for param in model.parameters():
            loss += (param ** 2).sum()
        return loss

    def _get_val_metrics(self) -> _Metrics:
        """Get the metrics on the validation dataset."""
        # Turn off batch-norm updates
        self.generator.eval()
        self.critic.eval()

        with torch.no_grad():
            metrics = _Metrics()

            for val_img, val_gt in tqdm(
                self.val_loader, desc="Validating", leave=False
            ):
                val_img = val_img.to(self.device)
                val_gt = val_gt.to(self.device)

                with autocast(enabled=self.config.mixed_precision):
                    val_pred = self.generator(val_img)
                    metrics.loss += self.loss(val_pred, val_gt)

                    fake_out = self.critic(sigmoid(val_pred))
                    real_out = self.critic(val_gt)
                    metrics.wass = (real_out - fake_out).mean()

                metrics.accuracy += self._get_acc(val_pred, val_gt)
                metrics.f1_score += self._get_f1(val_pred, val_gt)

            metrics.loss /= len(self.val_loader)
            metrics.wass /= len(self.val_loader)
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
            elif key == "wass":
                tag = "losses/wasserstein"
            else:
                tag = f"metrics/{key}"

            train_writer.add_scalar(tag, getattr(train_metrics, key), step)
            val_writer.add_scalar(tag, getattr(val_metrics, key), step)

        train_writer.add_scalar(
            "losses/generator_regularization",
            self._get_l2_reg(self.generator),
            step,
        )
        train_writer.add_scalar(
            "losses/critic_regularization", self._get_l2_reg(self.critic), step
        )

        # Log a histogram for each tensor parameter in the models, to
        # see if a parameter is training stably or not
        for model, tag in [
            (self.generator, "generator"),
            (self.critic, "critic"),
        ]:
            for name, value in model.state_dict().items():
                train_writer.add_histogram(f"{tag}/{name}", value, step)

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
