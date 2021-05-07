"""This is a script to test out semantic thinning for different thresholds."""
import glob
import math
import tkinter
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from pathlib import Path

import imageio
import numpy
from PIL import Image, ImageTk
from skimage.morphology import skeletonize


def sigmoid(x):
    """Just normal sigmoid."""
    return 1 / (1 + math.exp(-x))


class SemThinTest:
    """Class that generates gui to look at semantic thinnings of outputs."""

    def __init__(self):
        """Init, for the types."""
        self.current_id: int
        self.threshold: float
        self.label1: tkinter.Label
        self.root: tkinter.Tk

    def navigate(self, event) -> None:
        """Handle keypresses for navigation."""
        if event.char == "d":
            index_change = 1
            self.current_id = (self.current_id + index_change) % len(
                self.images
            )
        elif event.char == "a":
            index_change = -1
            self.current_id = (self.current_id + index_change) % len(
                self.images
            )
        elif event.char == "w":
            self.threshold += 0.05
        elif event.char == "s":
            self.threshold -= 0.05
        elif event.char == "g":
            self.mode = "greyscale"
        elif event.char == "b":
            self.mode = "binary"
        elif event.char == "t":
            self.mode = "semantic_thinning"
        else:
            return

        if self.mode == "binary":
            next_im = (
                self.images[self.current_id] > sigmoid(self.threshold)
            ).astype(numpy.float) * 255
        elif self.mode == "greyscale":
            next_im = self.images[self.current_id] * 255
        elif self.mode == "semantic_thinning":
            next_im = (
                self.images[self.current_id] > sigmoid(self.threshold)
            ).astype(numpy.float)
            next_im = skeletonize(next_im)
        next_im = Image.fromarray(next_im)
        next_im = ImageTk.PhotoImage(next_im)
        self.label1.configure(image=next_im)
        self.label1.image = next_im  # type: ignore

        self.root.title(
            "Image: "
            + self.image_names[self.current_id]
            + " - Threshold: "
            + str(self.threshold)
            + " - Mode: "
            + self.mode
        )

    def main(self, args: Namespace) -> None:
        """Main, duh."""
        self.images = []
        self.image_names = []
        for im_path in glob.glob(str(args.image_dir.expanduser() / "*.png")):
            im = imageio.imread(im_path)
            im = im / 255
            self.images.append(im)
            self.image_names.append(im_path.split("\\")[-1][:-4])

        self.current_id = 0
        # this is the logit value, always needs to be put through sigmoid first
        self.threshold = 0
        self.mode = "binary"  # other options: "greyscale", "semantic_thinning"
        print(len(self.images), "images loaded")

        self.root = tkinter.Tk()
        initial_image = (
            self.images[self.current_id] > sigmoid(self.threshold)
        ).astype(numpy.float) * 255
        initial_image = Image.fromarray(initial_image)
        initial_image = ImageTk.PhotoImage(initial_image)

        self.label1 = tkinter.Label(image=initial_image)
        self.label1.place(x=0, y=0)

        self.root.geometry(
            f"{initial_image.width() + 4}x{initial_image.height() + 4}"
        )
        self.root.title(
            "Image: "
            + self.image_names[self.current_id]
            + " - Threshold: "
            + str(self.threshold)
            + " - Mode: "
            + self.mode
        )
        self.root.bind("<KeyPress>", self.navigate)

        print("Press 'd' to go forward, and 'a' to go back")
        print("Press 'w' to increase threshold, and 's' to go decrease")
        print(
            "Press 'b' to enter binary mode, 'g' for greyscale,"
            "'t' for semantic thinning"
        )
        self.root.mainloop()


if __name__ == "__main__":
    description = (
        "Script to test out different thresholds for semantic thinning."
    )
    formatter_class = ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description, formatter_class)  # type: ignore
    parser.add_argument(
        "--image_dir",
        type=Path,
        default="outputs",
        help="Directory where the model's (greyscale) outputs are stored",
    )
    SemThinTest().main(parser.parse_args())
