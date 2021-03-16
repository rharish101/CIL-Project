#!/usr/bin/env python3
"""GUI to visualize binary labels with the input images."""
import tkinter
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from pathlib import Path
from typing import Optional

from PIL import Image, ImageTk


class Visualizer:
    """The class for the visualizer."""

    def __init__(
        self,
        image_dir: Path,
        ground_truth_dir: Path,
        prediction_dir: Optional[Path] = None,
    ):
        """Initialize the GUI.

        Args:
            image_dir: The path to the input images
            ground_truth_dir: The path to the binary outputs. This can be
                either ground truth (for the training data), or predictions.
            prediction_dir: The path to additional binary outputs. If not None,
                then these will be shown along with the "ground truth" for
                comparison.
        """
        self.root = tkinter.Tk()
        self.files_index = 0

        self.files = sorted(image_dir.glob("*"))
        self.superimposed_images = dict()
        for png in self.files:
            im_1 = Image.open(png).convert("RGBA")
            im_2 = Image.open(ground_truth_dir / png.name).convert("RGBA")
            blended = Image.blend(im_1, im_2, 0.4)

            if prediction_dir is not None:
                back = Image.new("RGBA", im_1.size, color="black")
                front = Image.new("RGBA", im_1.size, color="green")
                mask = Image.open(prediction_dir / png.name)
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
        self.root.title(self.files[self.files_index])
        self.root.bind("<KeyPress>", self.navigate)

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

        self.root.title(self.files[self.files_index])

    def run(self) -> None:
        """Start the GUI."""
        print("Press 'd' to go forward, and 'a' to go back")
        self.root.mainloop()


def main(args: Namespace) -> None:
    """Run the main program."""
    if args.mode == "train":
        train_dir = args.data_dir / "training/training"
        image_dir = train_dir / "images"
        ground_truth_dir = train_dir / "groundtruth"
        prediction_dir = args.pred_dir
    else:
        image_dir = args.data_dir / "test_images/test_images/"
        if args.pred_dir is None:
            raise ValueError(
                f"{args.mode} mode requires the prediction directory"
            )
        ground_truth_dir = args.pred_dir
        prediction_dir = None

    gui = Visualizer(image_dir, ground_truth_dir, prediction_dir)
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
        "(required for test mode)",
    )
    main(parser.parse_args())
