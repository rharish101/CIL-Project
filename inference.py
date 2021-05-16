#!/usr/bin/env python3
"""Script to infer with the model."""
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from pathlib import Path

from src.config import load_config
from src.infer import Inference


def main(args: Namespace) -> None:
    """Run the main program."""
    config = load_config(args.config)
    inference = Inference(
        args.image_dir, args.load_dir, args.use_best_model, config
    )
    inference.infer(args.output_dir)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Script to infer with the model",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "image_dir",
        metavar="IMAGE_DIR",
        type=Path,
        help="Path to the directory containing the input images",
    )
    parser.add_argument(
        "load_dir",
        metavar="LOAD_DIR",
        type=Path,
        help="Directory from where to load the model's weights",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        help="Path to a TOML config containing hyper-parameter values",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="outputs",
        help="Directory where to dump the model's outputs",
    )
    parser.add_argument(
        "--use_best_model",
        type=bool,
        default=False,
        help="Whether to use the best model (w.r.t. accuracy)",
    )
    main(parser.parse_args())
