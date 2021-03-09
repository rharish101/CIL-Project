#!/usr/bin/env python3
"""GUI to visualize binary labels with the input images."""
import tkinter
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from pathlib import Path

from PIL import Image, ImageTk


class Visualizer:
    """The class for the visualizer."""

    def __init__(self, image_dir: Path, ground_truth_dir: Path):
        """Initialize the GUI."""
        self.root = tkinter.Tk()
        self.files_index = 0

        self.files = sorted(image_dir.glob("*"))
        self.superimposed_images = dict()
        for png in self.files:
            im_1 = Image.open(png).convert("RGBA")
            im_2 = Image.open(ground_truth_dir / png.name).convert("RGBA")
            self.superimposed_images[png] = ImageTk.PhotoImage(
                Image.blend(im_1, im_2, 0.4)
            )

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
    else:
        image_dir = args.data_dir / "test_images/test_images/"
        ground_truth_dir = args.pred_dir

    gui = Visualizer(image_dir, ground_truth_dir)
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
        default="outputs",
        help="Directory containing the model's predictions for the test data "
        "(used only in the 'test' mode)",
    )
    main(parser.parse_args())
