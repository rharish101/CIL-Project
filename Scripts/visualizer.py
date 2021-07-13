#!/usr/bin/env python3
"""GUI to visualize binary labels with the input images."""
import tkinter
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageTk

from src.config import Config, load_config
from submission import PATCH_SIZE, classify_patch


class Visualizer:
    """The class for the visualizer."""

    def __init__(
        self,
        config: Config,
        image_dir: Path,
        ground_truth_dir: Path,
        prediction_dir: Optional[Path] = None,
        down_sampled_output: bool = False,
        quantized_output: bool = False,
    ):
        """Initialize the GUI.

        Args:
            config: The hyper-param config
            image_dir: The path to the input images
            ground_truth_dir: The path to the binary outputs. This can be
                either ground truth (for the training data), or predictions.
            prediction_dir: The path to additional binary outputs. If not None,
                then these will be shown along with the "ground truth" for
                comparison.
            down_sampled_output: Whether to display down-sampled outputs (as
                submitted on Kaggle) instead of the high-resolution versions.
            quantized_output: Whether to display quantized (B/W) outputs
                instead of the grayscale versions.
        """
        self.config = config

        self.root = tkinter.Tk()
        self.files_index = 0
        self.files = sorted(image_dir.glob("*"))
        self.superimposed_images = dict()
        for png in self.files:
            im_1 = Image.open(png).convert("RGBA")
            im_2 = Image.open(ground_truth_dir / png.name).convert("RGBA")
            if down_sampled_output:
                im_2 = self._down_sample(im_2)
            elif quantized_output:
                im_2 = self._quantize(im_2)
            blended = Image.blend(im_1, im_2, 0.4)

            if prediction_dir is not None:
                back = Image.new("RGBA", im_1.size, color="black")
                front = Image.new("RGBA", im_1.size, color="green")
                mask = Image.open(prediction_dir / png.name)
                if down_sampled_output:
                    mask = self._down_sample(mask)
                elif quantized_output:
                    mask = self._quantize(mask)
                pred = Image.composite(front, back, mask)
                blended = Image.blend(blended, pred, 0.3)

            self.superimposed_images[png] = ImageTk.PhotoImage(blended)

        # Create a photoimage object of the image in the path
        initial_image = self.superimposed_images[self.files[self.files_index]]

        label1 = tkinter.Label(image=initial_image)
        # Position image
        label1.place(x=0, y=0)

        self.root.geometry(
            f"{initial_image.width() + 4}x{initial_image.height() + 4}"
        )
        self.root.title(str(self.files[self.files_index]))
        self.root.bind("<KeyPress>", self.navigate)

    @staticmethod
    def _pil_to_np(image: Image.Image) -> np.ndarray:
        """Convert PIL images to [0, 1] float images."""
        return np.array(image).astype(np.float32) / 255

    def _quantize(self, image_pil: Image.Image) -> Image.Image:
        """Quantize a grayscale image to black-and-white."""
        image = self._pil_to_np(image_pil)
        image_bw = np.uint8(image > self.config.lbl_fg_thresh) * 255
        return Image.fromarray(image_bw).convert(image_pil.mode)

    def _down_sample(self, image_pil: Image.Image) -> Image.Image:
        """Down sample as per the Kaggle submission."""
        image = self._pil_to_np(image_pil)
        output_img = np.empty(image.shape[:2], dtype=np.uint8)

        for row in range(0, image.shape[0], PATCH_SIZE):
            for col in range(0, image.shape[1], PATCH_SIZE):
                patch = image[row : row + PATCH_SIZE, col : col + PATCH_SIZE]
                label = classify_patch(patch, self.config)
                output_img[row : row + PATCH_SIZE, col : col + PATCH_SIZE] = (
                    label * 255
                )

        return Image.fromarray(output_img).convert(image_pil.mode)

    def navigate(self, event) -> None:
        """Handle keypresses for navigation."""
        if event.char == "d":
            index_change = 1
        elif event.char == "a":
            index_change = -1
        else:
            return

        self.files_index = (self.files_index + index_change) % len(self.files)
        next_im = self.superimposed_images[self.files[self.files_index]]
        label1_new = tkinter.Label(image=next_im)
        # Position image
        label1_new.place(x=0, y=0)

        self.root.title(str(self.files[self.files_index]))

    def run(self) -> None:
        """Start the GUI."""
        print("Press 'd' to go forward, and 'a' to go back")
        self.root.mainloop()


def main(args: Namespace) -> None:
    """Run the main program."""
    config = load_config(args.config)

    if args.mode == "train":
        train_dir = args.data_dir.expanduser() / "training/training"
        image_dir = train_dir / "images"
        ground_truth_dir = train_dir / "groundtruth"
        if args.pred_dir is not None:
            prediction_dir = args.pred_dir.expanduser()
        else:
            prediction_dir = None
    else:
        image_dir = args.data_dir.expanduser() / "test_images/test_images/"
        if args.pred_dir is None:
            raise ValueError(
                f"{args.mode} mode requires the prediction directory"
            )
        ground_truth_dir = args.pred_dir.expanduser()
        prediction_dir = None

    gui = Visualizer(
        config,
        image_dir,
        ground_truth_dir,
        prediction_dir,
        down_sampled_output=args.down_sample,
        quantized_output=args.quantize,
    )
    gui.run()


if __name__ == "__main__":
    parser = ArgumentParser(
        description="GUI to visualize binary labels with the input images",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "data_dir",
        metavar="DIR",
        type=Path,
        help="Path to the directory where the CIL data is extracted",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        help="Path to a TOML config containing hyper-parameter values",
    )
    parser.add_argument(
        "-m",
        "--mode",
        choices=["train", "test"],
        default="train",
        help="The choice of dataset to visualize",
    )
    parser.add_argument(
        "--pred-dir",
        type=Path,
        help="Directory containing the model's predictions for the input data "
        "(required for test mode to be used as ground truth)",
    )
    parser.add_argument(
        "--down-sample",
        action="store_true",
        help="Whether to show down-sampled outputs (used for Kaggle)",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Whether to show quantized (B/W) outputs instead of grayscale",
    )
    main(parser.parse_args())
