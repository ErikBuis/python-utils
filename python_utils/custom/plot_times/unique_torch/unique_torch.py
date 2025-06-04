from __future__ import annotations

import argparse
import logging
import random
from typing import Any

import numpy as np
import torch

from python_utils.modules.torch import unique

from ..plot_times import plot_times


def approach_1(
    x: torch.Tensor, dim: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Approach 1: Use PyTorch's built-in unique function."""
    return torch.unique(x, return_inverse=True, return_counts=True, dim=dim)


def approach_2(
    x: torch.Tensor, dim: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Approach 2: Use custom unique function."""
    return unique(x, return_inverse=True, return_counts=True, dim=dim)


def map_to_inputs(
    amount_rows: int, amount_cols: int
) -> tuple[tuple[torch.Tensor], dict[str, Any]]:
    """Map input parameters to function arguments.

    Args:
        amount_rows: The number of rows in the tensor.
        amount_cols: The number of columns in the tensor.

    Returns:
        Tuple containing positional args and keyword args for the algorithms.
    """
    # Set random seed for reproducibility.
    random.seed(69)

    # Generate a random tensor with the specified dimensions.
    x = torch.randint(0, 10, size=(amount_rows, amount_cols))

    # Return the inputs.
    return (x,), {"dim": 0}


def main(args: argparse.Namespace) -> None:
    """The entry point of the program.

    Args:
        args: The command line arguments.
    """
    amount_rows = 2 ** np.arange(2, 21, 2)
    amount_cols = 2 ** np.arange(0, 8, 1)

    plot_times(
        amount_rows,
        amount_cols,
        map_to_inputs,
        approach_1,
        approach_2,
        "PyTorch unique comparison",
        "PyTorch's torch.unique",
        "Custom unique function",
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
