import argparse
import logging
import random
from typing import Any

import numpy as np
import torch

from plot_times.plot_times import plot_times
from utils_torch import unique_with_backmap, unique_with_backmap_naive


def map_to_inputs(
    amount_rows: int, amount_cols: int
) -> tuple[tuple[torch.Tensor], dict[str, Any]]:
    # Set random seed for reproducibility.
    random.seed(69)

    # Decide on a tensor to calculate the unique values for.
    x = torch.randint(0, 9, (amount_rows, amount_cols))

    # Return the inputs.
    return (x,), {"return_inverse": True, "return_counts": True, "dim": 1}


def main(args: argparse.Namespace) -> None:
    """The entry point of the program.

    Args:
        args: The command line arguments.
    """
    amount_rows = 2 ** np.arange(0, 11)
    amount_cols = 2 ** np.arange(0, 11)

    plot_times(
        amount_rows,
        amount_cols,
        map_to_inputs,
        unique_with_backmap_naive,
        unique_with_backmap,
        "Unique with backmap",
        "Naive algorithm",
        "Optimized algorithm",
        "Amount of rows",
        "Amount of columns",
    )


if __name__ == "__main__":
    # Create the argument parser.
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Define command line arguments.
    parser.add_argument(
        "--logging_level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="The logging level to use.",
    )

    args = parser.parse_args()

    # Configure the logger.
    logging.basicConfig(
        level=args.logging_level,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    logging.debug(f"{args=}")

    main(args)
