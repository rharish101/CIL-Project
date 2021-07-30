"""Class to evaluate the model."""
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import Config
from .data import EvalDataset, get_file_paths
from .train import Trainer


class Evaluator:
    """Class to evaluate the model."""

    def __init__(self, data_dir: Path, pred_dir: Path, config: Config):
        """Store config and initialize everything.

        Args:
            data_dir: Path to the directory where the CIL data is extracted
            pred_dir: Directory containing the model's predictions for the
                input data
            config: The hyper-param config
        """
        training_path_list, ground_truth_path_list = get_file_paths(data_dir)
        X_train, X_test, y_train, y_test = Trainer.train_test_split(
            training_path_list,
            ground_truth_path_list,
            test_portion=config.val_split,
        )

        train_dataset = EvalDataset(config, y_train, pred_dir)
        val_dataset = EvalDataset(config, y_test, pred_dir)

        train_loader = DataLoader(
            train_dataset, batch_size=config.test_batch_size, pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=config.test_batch_size, pin_memory=True
        )
        self.loaders = {
            "training": train_loader,
            "validation": val_loader,
        }

    def eval(self) -> None:
        """Evaluate with the model."""
        for name, loader in self.loaders.items():
            accuracy = 0.0
            f1_score = 0.0

            for targets, predictions in tqdm(
                loader, desc=f"Evaluating {name}"
            ):
                logits = torch.log(predictions / (1 - predictions))
                accuracy += Trainer._get_acc(logits, targets)
                f1_score += Trainer._get_acc(logits, targets)

            accuracy /= len(loader)
            f1_score /= len(loader)

            print(f"{name.capitalize()} accuracy: {accuracy:.5f}")
            print(f"{name.capitalize()} F1 score: {f1_score:.5f}")
