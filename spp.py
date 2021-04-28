#!/usr/bin/env python3
"""Script that runs slime post-proc by only copying the predicted value."""
import random
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from graph_cut import IMG_EXTENSION
from src.config import Config, load_config


def classify_image(
    img: np.ndarray, start_positions: np.ndarray, config: Config
) -> np.ndarray:
    """Classify image."""
    stack = [start_positions]
    already_visited = np.zeros_like(img, dtype=np.float32)

    while not len(stack) == 0:
        next_posses = random.choice(stack)
        stack.remove(next_posses)
        random.shuffle(next_posses)
        for pos in next_posses:
            pos_x = pos[0]
            pos_y = pos[1]
            classified = img[pos_x][pos_y]
            if classified > config.spp_thresh:
                already_visited[pos_x][pos_y] = classified
                next_posses_list = []
                if (
                    pos_x < len(img) - 1
                    and already_visited[pos_x + 1][pos_y] == 0
                ):
                    next_posses_list.append((pos_x + 1, pos_y))
                if 0 < pos_x and already_visited[pos_x - 1][pos_y] == 0:
                    next_posses_list.append((pos_x - 1, pos_y))
                if (
                    pos_y < len(img) - 1
                    and already_visited[pos_x][pos_y + 1] == 0
                ):
                    next_posses_list.append((pos_x, pos_y + 1))
                if 0 < pos_y and already_visited[pos_x][pos_y - 1] == 0:
                    next_posses_list.append((pos_x, pos_y - 1))
                if not next_posses_list == []:
                    stack.append(next_posses_list)
            else:
                already_visited[pos_x][pos_y] = -1

    return np.maximum(already_visited, 0.0)


def main(args: Namespace) -> None:
    """Run the main program."""
    config = load_config(args.config)

    output_dir = args.output_dir.expanduser()
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    for img_path in tqdm(
        sorted(args.input_dir.expanduser().glob(f"*.{IMG_EXTENSION}"))
    ):
        pic = np.array(Image.open(img_path)).astype(np.float32) / 255

        # Get starting positions for the slime
        start_pos_mask = pic > 0
        start_pos_mask[1:-1, 1:-1] = False  # only keep the boundaries
        start_positions = np.stack(np.where(start_pos_mask), 1)

        res_pic = classify_image(pic, start_positions, config)
        res_pic = np.uint8(res_pic * 255)
        Image.fromarray(res_pic).save(args.output_dir / img_path.name)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Script to run the slime post-processor.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        help="Path to a TOML config containing hyper-parameter values",
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        type=Path,
        default="outputs",
        help="Directory with the raw predictions of the model",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default="outputs/slime",
        help="Directory where to dump the model's outputs",
    )
    main(parser.parse_args())
