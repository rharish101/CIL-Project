#!/usr/bin/env python3
"""Script to convert outputs into a submission."""
import csv
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
from PIL import Image
from typing_extensions import Final

# Percentage of pixels > 1 required to assign a foreground label to a patch
FOREGROUND_THRESHOLD: Final = 0.5
# Minimum count of pixels classified as road for patch to be predicted as road
PIXEL_COUNT_THRESHOLD: Final = 10
# Size of each patch as specified in the problem statement
PATCH_SIZE: Final = 16
CSV_NAME: Final = "submission.csv"  # name of the output CSV
IMG_EXTENSION: Final = "png"  # extension for the predicted outputs


def classify_patch(patch: np.ndarray) -> int:
    """Classify a patch into 0 or 1.

    Args:
        patch: A 16x16 ndarray for the patch

    Returns:
        The label as an int
    """
    road_pixel_count = np.sum(patch > FOREGROUND_THRESHOLD)
    return int(road_pixel_count > PIXEL_COUNT_THRESHOLD)


def get_image_output(image_path: Path) -> Iterable[Tuple[str, int]]:
    """Yield the predictions for an image as CSV rows."""
    image_num = int(image_path.stem.split("_")[-1])
    image = np.array(Image.open(image_path)) / 255

    for row in range(0, image.shape[0], PATCH_SIZE):
        for col in range(0, image.shape[1], PATCH_SIZE):
            patch = image[row : row + PATCH_SIZE, col : col + PATCH_SIZE]
            label = classify_patch(patch)
            yield (f"{image_num:03d}_{col}_{row}", label)


def main(args: Namespace) -> None:
    """Run the main program."""
    output_dir = args.output_dir.expanduser()
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    output_path = output_dir / CSV_NAME
    with open(output_path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["Id", "Prediction"])

        for path in sorted(
            args.image_dir.expanduser().glob(f"*.{IMG_EXTENSION}")
        ):
            for row in get_image_output(path):
                writer.writerow(row)

    print(f"Saved submission as: {output_path}")


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Script to convert outputs into a submission",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="outputs",
        help="Directory where to dump the submission CSV",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default="outputs",
        help="Directory where the model's outputs are stored",
    )
    main(parser.parse_args())
