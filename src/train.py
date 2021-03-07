"""Class to train the model."""
from pathlib import Path

import torch
from torch.nn import BCEWithLogitsLoss, Module
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing_extensions import Final

from .data import INPUT_CHANNELS, OUTPUT_CHANNELS, TrainDataset
from .model import UNet


class Trainer:
    """Class to train the model."""

    SAVE_NAME: Final = "model.pt"

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

    def train(
        self, batch_size: int, max_epochs: int, save_dir: Path, save_steps: int
    ) -> None:
        """Train the model."""
        loader = DataLoader(
            self.dataset, batch_size=batch_size, shuffle=True, pin_memory=True
        )

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
