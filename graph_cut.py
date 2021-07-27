#!/usr/bin/env python3
"""Script to apply graph cut post processing on outputs."""
import argparse
import os
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing_extensions import Final

from src.config import Config, load_config

IMG_EXTENSION: Final = "png"


def apply_graph_cut(
    prob_mask: np.ndarray, iteration_count: int, config: Config
) -> np.ndarray:
    """Applies graph cut on an output mask generated by the model.

    Args:
        prob_mask: a numpy array containing the probability masks (3 channels)
        iteration_count: number of iteration in grabcut algorithm
        config: The hyper-param config

    Returns:
        The graph cut applied mask as a numpy array
    """
    # Initialize a unary mask
    mask = np.zeros(prob_mask.shape[:2], np.uint8)

    # Masking strategy: Assigning grabcut mask values
    cut_mask = prob_mask > config.prob_fg_thresh
    if (~cut_mask).all():
        return mask
    mask[np.any(cut_mask, axis=2)] = cv2.GC_PR_FGD

    # allocate memory for two arrays that the GrabCut algorithm internally
    # uses when segmenting the foreground from the background
    fgModel = np.zeros((1, 65), dtype="float")
    bgModel = np.zeros((1, 65), dtype="float")
    # apply GrabCut using the the mask segmentation method
    mask, bgModel, fgModel = cv2.grabCut(
        prob_mask,
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
    config = load_config(args.config)

    output_dir = args.output_dir.expanduser()
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    # Extract mask directory names for ensembler
    mask_dirs = [args.mask_dir.expanduser()]
    if args.ensemble:
        mask_dirs = list(args.mask_dir.expanduser().glob("[!.]*/"))

    for mask_path in tqdm(sorted(mask_dirs[0].glob(f"*.{IMG_EXTENSION}"))):
        # Load mask in the grayscale mode and mask image itself
        masks = []
        for mask_dir in mask_dirs:
            path = str(os.path.join(mask_dir, os.path.basename(mask_path)))
            masks.append(cv2.imread(path, cv2.IMREAD_GRAYSCALE))

        # Because of the OpenCV implementation, only ensembling of up to
        # 3 mask sets is supported.
        # Thus, we do the sampling to ensure this constraint
        if len(masks) < 3:
            masks = random.choices(masks, k=3)
        else:
            masks = random.sample(masks, k=3)
        mask = np.stack(masks, axis=2)

        # Apply Graph-cut on the input mask
        post_processed_mask = apply_graph_cut(mask, args.iter, config)

        # Save the new mask
        im = Image.fromarray(
            (post_processed_mask == cv2.GC_PR_FGD).astype("uint8") * 255
        )
        im.save(str((output_dir / os.path.basename(mask_path)).resolve()))

    print(f"Graph cut applied masks saved in: {output_dir}")


if __name__ == "__main__":
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser(
        description="Apply graph cut post processing on outputs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "-c",
        "--config",
        type=Path,
        help="Path to a TOML config containing hyper-parameter values",
    )
    ap.add_argument(
        "-m",
        "--mask-dir",
        default="outputs",
        type=Path,
        help="path to input mask files directory",
    )
    ap.add_argument(
        "-i",
        "--iter",
        type=int,
        default=10,
        help="# of GrabCut iterations (larger value => slower runtime)",
    )
    ap.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default="outputs/graph_cut",
        help="Directory where to dump the model's outputs",
    )
    ap.add_argument(
        "-e",
        "--ensemble",
        default=False,
        action="store_true",
        help="Set whether running graph-cut as the ensembling script",
    )

    main(ap.parse_args())
