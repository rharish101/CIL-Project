#!/usr/bin/env python3
"""GUI to visualize the ground truth."""
import tkinter
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any

from PIL import Image, ImageTk


def main(args: Namespace) -> None:
    """Run the main program."""
    root: Any = tkinter.Tk()
    root.files_index = 0

    train_dir = args.data_dir / "training/training"
    files = list((train_dir / "images").glob("*"))
    superimposed_images = dict()
    for png in files:
        im_1 = Image.open(png, mode="r").convert("RGBA")
        im_2 = Image.open(
            train_dir / "groundtruth" / png.name, mode="r"
        ).convert("RGBA")
        superimposed_images[png] = ImageTk.PhotoImage(
            Image.blend(im_1, im_2, 0.4)
        )

    # Create a photoimage object of the image in the path
    test = superimposed_images[files[root.files_index]]

    label1: Any = tkinter.Label(image=test)
    label1.image = test

    # Position image
    label1.place(x=0, y=0)

    root.geometry(str(test.width() + 4) + "x" + str(test.height() + 4))
    root.title(files[root.files_index])

    def onKeyPress(event):
        if event.char == "d":
            root.files_index += 1
            if root.files_index == len(files):
                root.files_index = 0
        elif event.char == "a":
            root.files_index -= 1
            if root.files_index == 0:
                root.files_index = len(files) - 1

        next_im = superimposed_images[files[root.files_index]]

        label1_new = tkinter.Label(image=next_im)
        label1_new.image = next_im

        # Position image
        label1_new.place(x=0, y=0)

        root.title(files[root.files_index])

    root.bind("<KeyPress>", onKeyPress)
    print("Press 'd' to go forward, and 'a' to go back")
    root.mainloop()


if __name__ == "__main__":
    parser = ArgumentParser(description="GUI to visualize the ground truth")
    parser.add_argument(
        "data_dir",
        metavar="DIR",
        type=Path,
        help="Path to the directory where the CIL data is extracted",
    )
    main(parser.parse_args())
