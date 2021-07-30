#!/usr/bin/env python3
"""Script that runs slime post-proc by only copying the predicted value."""

import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from tkinter import Tk

from imageio import imwrite
from torchvision.io import read_image
from tqdm import tqdm

from run_slime import classify_image, get_classification_paths


class DummySlime:
    """Need a dummy for the function call."""

    def __init__(self):
        """Inits the dummy, you dummy."""
        self.inference = False


def main(args: Namespace) -> None:
    """Stupid sexy slimy main."""
    file_paths = get_classification_paths(args.input_dir, sort=False)
    try:
        os.mkdir(args.output_dir)
    except Exception:
        pass
    root = Tk()

    for idx in tqdm(range(len(file_paths))):
        pic = read_image(file_paths[idx]).float() / 255

        # get start positions
        sp = []
        for j in range(len(pic[0])):
            if pic[0][0][j] > 0:
                sp.append((0, j))
            if pic[0][1][j] > 0:
                sp.append((0, j))
            if pic[0][len(pic[0]) - 1][j] > 0:
                sp.append((len(pic[0]) - 1, j))
            if pic[0][len(pic[0]) - 2][j] > 0:
                sp.append((len(pic[0]) - 1, j))
            if pic[0][j][0] > 0:
                sp.append((j, 0))
            if pic[0][j][1] > 0:
                sp.append((j, 0))
            if pic[0][j][len(pic[0]) - 1] > 0:
                sp.append((j, len(pic[0]) - 1))
            if pic[0][j][len(pic[0]) - 2] > 0:
                sp.append((j, len(pic[0]) - 1))
        if len(sp) <= 30:
            print(
                "Pic:",
                file_paths[idx],
                "only has",
                len(sp),
                "start positions.",
            )

        res_pic = classify_image(
            file_paths[idx],
            pic.unsqueeze(0),
            None,
            sp,
            DummySlime(),
            root=root,
            visualize=args.visualize,
            stupid=True,
            args=args,
        )
        res_pic = (res_pic[0].numpy() * 255).astype(int)
        imwrite(
            args.output_dir + file_paths[idx][len(args.input_dir) :], res_pic
        )


if __name__ == "__main__":
    description = "Script to run the slime."
    formatter_class = ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description, formatter_class)  # type: ignore
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Dir with the stuff that should be inferred on.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Dir wthere the outputs should be put.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="At which probability treshold the slime should move over.",
    )
    parser.add_argument(
        "-v",
        "--visualize",
        nargs="?",
        const=True,
        help="Visualize process?",
    )
    main(parser.parse_args())
