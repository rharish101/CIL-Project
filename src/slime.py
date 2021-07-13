"""Slime class."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import cat, cuda, float32, load, save, sigmoid
from torch.nn import BCELoss

DEFAULT_VISION_FILE = "slime_vision"
DEFAULT_RANGE_OF_SIGHT = 15
DEFAULT_LAYER_REDUCTION = 0.7


class SlimeVision(nn.Module):
    """Slime module."""

    def __init__(
        self, range_of_sight=9, hidden_size=3, layers=3, layer_reduction=1
    ):
        """Inits slime."""
        super(SlimeVision, self).__init__()
        self.input_layer = nn.Linear(
            5 * range_of_sight * range_of_sight,
            hidden_size * range_of_sight * range_of_sight,
        )
        self.layers = [
            nn.Linear(
                int(
                    hidden_size
                    * range_of_sight
                    * range_of_sight
                    * math.pow(layer_reduction, i)
                ),
                int(
                    hidden_size
                    * range_of_sight
                    * range_of_sight
                    * math.pow(layer_reduction, i + 1)
                ),
            )
            for i in range(layers)
        ]
        self.output_layer = nn.Linear(
            int(
                hidden_size
                * range_of_sight
                * range_of_sight
                * math.pow(layer_reduction, layers)
            ),
            1,
        )
        if cuda.is_available():
            self.input_layer.to(torch.device("cuda"))
            for layer in self.layers:
                layer.to(torch.device("cuda"))
            self.output_layer.to(torch.device("cuda"))
        self.range_of_sight = range_of_sight

    def forward(self, x):
        """Duh."""
        x = self.input_layer(x)
        for layer in self.layers:
            x = layer(x)
            x = F.relu(x)
        x = self.output_layer(x)
        return x


class Slime:
    """another one."""

    def __init__(
        self,
        state_file_path=None,
        save_path=DEFAULT_VISION_FILE,
        inference=False,
    ):
        """bruh."""
        self.vision = SlimeVision(
            range_of_sight=DEFAULT_RANGE_OF_SIGHT,
            layer_reduction=DEFAULT_LAYER_REDUCTION,
        )
        self.optimizer = optim.SGD(self.vision.parameters(), lr=0.0001)
        self.optimizer.zero_grad()
        if state_file_path is not None:
            self.vision.load_state_dict(load(state_file_path))
            if inference:
                self.vision.eval()
            print("Loaded pretrained model")
        if cuda.is_available():
            self.vision.to(torch.device("cuda"))
            print("Using Cuda")
        self.steps_backprop = 3000
        self.save_path = save_path
        self.iteration_count = 0
        self.loss_function = BCELoss()
        self.loss = 0
        self.accumulated_logits = 0
        self.inference = inference

    def classify(self, pos_x, pos_y, x, already_visited, y):
        """Takes a slice of x of size range_of_sight^2.

        centered around pos_x, pos_y
        """
        return x[3][pos_x][pos_y] >= 1
        # return True
        half_range = (self.vision.range_of_sight - 1) / 2
        x_offset = 0
        y_offset = 0
        if pos_x < half_range:
            x_offset = half_range - pos_x
        elif pos_x > len(x[0]) - half_range - 1:
            x_offset = len(x[0]) - half_range - 1 - pos_x
        if pos_y < half_range:
            y_offset = half_range - pos_y
        elif pos_y > len(x[0]) - half_range - 1:
            y_offset = len(x[0]) - half_range - 1 - pos_y
        x_small = cat((x, already_visited)).narrow(
            1,
            int(pos_x - half_range + max(0, x_offset)),
            int(2 * half_range + 1 - abs(x_offset)),
        )
        x_small = x_small.narrow(
            2,
            int(pos_y - half_range + max(0, y_offset)),
            int(2 * half_range + 1 - abs(y_offset)),
        )
        pad = (
            int(max(0, y_offset)),
            int(-min(0, y_offset)),
            int(max(0, x_offset)),
            int(-min(0, x_offset)),
            0,
            0,
        )
        x_input = F.pad(x_small, pad=pad, value=0)

        if cuda.is_available():
            x_input = x_input.cuda()
        x_input = x_input.view(
            5 * self.vision.range_of_sight * self.vision.range_of_sight
        )

        logit = sigmoid(self.vision(x_input)).float()

        if not self.inference:
            self.accumulated_logits += float(logit)
            target = torch.tensor(
                [min(int(y[0][pos_x][pos_y]), 1)], dtype=float32
            )
            if cuda.is_available():
                target = target.cuda()
            classification_loss = self.loss_function(logit, target)
            classification_loss.backward()
            self.loss += float(classification_loss)
            self.iteration_count += 1
            if self.iteration_count >= self.steps_backprop:
                self.optimizer.step()
                self.optimizer.zero_grad()
                print(
                    "Average loss:",
                    float(self.loss / self.iteration_count),
                    "Average logit:",
                    float(self.accumulated_logits / self.iteration_count),
                )
                self.loss = 0
                self.iteration_count = 0
                self.accumulated_logits = 0

        return logit >= 0.5

    def save(self):
        """bruh."""
        # print("Saving slime vision")
        save(self.vision.state_dict(), self.save_path)
