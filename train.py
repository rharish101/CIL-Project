#!/usr/bin/env python3
"""Script to train the model."""
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from pathlib import Path

from src.config import load_config
from src.train import Trainer


def main(args: Namespace) -> None:
    """Run the main program."""
    config = load_config(args.config)
    trainer = Trainer(args.data_dir, config)
    trainer.train(
        args.save_dir,
        args.log_dir,
        save_steps=args.save_steps,
        log_steps=args.log_steps,
    )


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Script to train the model",
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
        "--save-dir",
        type=Path,
        default="saved_models",
        help="Directory where to save the model's weights",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default="logs",
        help="Directory where to log metrics",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=500,
        help="Step interval for saving the model's weights",
    )
    parser.add_argument(
        "--log-steps",
        type=int,
        default=100,
        help="Step interval for logging metrics",
    )
    main(parser.parse_args())
