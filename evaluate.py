#!/usr/bin/env python3
"""Script to evaluate the model."""
import random
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from pathlib import Path

from src.config import load_config
from src.evaluate import Evaluator


def main(args: Namespace) -> None:
    """Run the main program."""
    config = load_config(args.config)
    random.seed(config.seed)

    evaluator = Evaluator(args.data_dir, args.pred_dir, config)
    evaluator.eval()


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Script to evaluate the model",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "data_dir",
        metavar="DIR",
        type=Path,
        help="Path to the directory where the CIL data is extracted",
    )
    parser.add_argument(
        "pred_dir",
        metavar="PRED_DIR",
        type=Path,
        help="Directory containing the model's predictions for the input data",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        help="Path to a TOML config containing hyper-parameter values",
    )
    main(parser.parse_args())
