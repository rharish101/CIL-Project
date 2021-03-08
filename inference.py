#!/usr/bin/env python3
"""Script to infer with the model."""
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from pathlib import Path

from src.config import load_config
from src.infer import Inference


def main(args: Namespace) -> None:
    """Run the main program."""
    config = load_config(args.config)
    inference = Inference(args.data_dir, args.load_dir, config)
    inference.infer(args.output_dir)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Script to infer with the model",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "data_dir",
        metavar="DIR",
        type=Path,
        help="Path to the directory where the CIL data is extracted",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        help="Path to a TOML config containing hyper-parameter values",
    )
    parser.add_argument(
        "--load-dir",
        type=Path,
        default="saved_models",
        help="Directory from where to load the model's weights",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="outputs",
        help="Directory where to dump the model's outputs",
    )
    main(parser.parse_args())
