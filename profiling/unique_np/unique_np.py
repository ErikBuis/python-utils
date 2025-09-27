from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import numpy.typing as npt
from loguru import logger

from python_utils.modules.numpy import unique

from .. import configure_root_logger
from ..plot_times import plot_times


def approach_1(
    x: npt.NDArray, axis: int
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    """Approach 1: Use NumPy's built-in unique function."""
    return np.unique(
        x, return_index=True, return_inverse=True, return_counts=True, axis=axis
    )


def approach_2(
    x: npt.NDArray, axis: int
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    """Approach 2: Use custom unique function."""
    return unique(
        x,
        return_backmap=True,
        return_inverse=True,
        return_counts=True,
        axis=axis,
    )


def map_to_inputs(
    amount_rows: int, amount_cols: int
) -> tuple[tuple[npt.NDArray], dict[str, Any]]:
    """Map input parameters to function arguments.

    Args:
        amount_rows: The number of rows in the array.
        amount_cols: The number of columns in the array.

    Returns:
        Tuple containing positional args and keyword args for the algorithms.
    """
    # Set random seed for reproducibility.
    rng = np.random.default_rng(69)

    # Generate a random array with the specified dimensions.
    x = rng.integers(0, 10, size=(amount_rows, amount_cols))

    # Return the inputs.
    return (x,), {"axis": 0}


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
        "NumPy unique comparison",
        "NumPy's np.unique",
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

    # Parse the command line arguments.
    args = parser.parse_args()

    # Configure the root logger.
    configure_root_logger(args.logging_level)

    # Log the command line arguments for reproducibility.
    logger.debug(f"{args=}")

    # Run the program.
    with logger.catch():
        main(args)
