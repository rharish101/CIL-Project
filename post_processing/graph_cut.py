#!/usr/bin/env python3
"""Script to apply graph cut post processing on outputs."""
import argparse
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

IMG_EXTENSION = "png"
PROBABLE_FOREGROUND_THRESHOLD = 32  # Out of 255


def apply_graph_cut(mask_path: str, iteration_count: int) -> np.ndarray:
    """Applies graph cut on an output mask generated by the model.

    Args:
        mask_path: path to the given mask file
        iteration_count: number of iteration in grabcut algorithm

    Returns:
        The graph cut applied mask as a numpy array
    """
    # Load mask in grayscale mode and mask image itself
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Assigning grabcut mask values
    cut_mask = mask > PROBABLE_FOREGROUND_THRESHOLD
    mask[cut_mask > 0] = cv2.GC_PR_FGD
    mask[cut_mask == 0] = cv2.GC_BGD

    # allocate memory for two arrays that the GrabCut algorithm internally
    # uses when segmenting the foreground from the background
    fgModel = np.zeros((1, 65), dtype="float")
    bgModel = np.zeros((1, 65), dtype="float")
    # apply GrabCut using the the mask segmentation method
    (mask, bgModel, fgModel) = cv2.grabCut(
        mask_image,
        mask,
        None,
        bgModel,
        fgModel,
        iterCount=iteration_count,
        mode=cv2.GC_INIT_WITH_MASK,
    )

    return mask


def main(args: argparse.Namespace) -> None:
    """Run the main program."""
    output_dir = args.mask_dir.expanduser() / "graph_cut"
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    for mask_path in tqdm(
        sorted(args.mask_dir.expanduser().glob(f"*.{IMG_EXTENSION}"))
    ):
        post_processed_mask = apply_graph_cut(
            str(mask_path.resolve()), args.iter
        )

        # Save the new mask
        im = Image.fromarray(
            (post_processed_mask == cv2.GC_PR_FGD).astype("uint8") * 255
        )
        im.save(str((output_dir / os.path.basename(mask_path)).resolve()))

    print(f"Graph cut applied masks saved in: {output_dir}")


if __name__ == "__main__":
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-m",
        "--mask-dir",
        type=Path,
        help="path to input mask files directory",
    )
    ap.add_argument(
        "-c",
        "--iter",
        type=int,
        default=10,
        help="# of GrabCut iterations (larger value => slower runtime)",
    )

    main(ap.parse_args())
