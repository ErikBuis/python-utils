from __future__ import annotations

import argparse
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
from loguru import logger

from python_utils.modules.numpy import groupby

from .. import configure_root_logger
from ..plot_times import plot_times


def approach_1(
    keys: npt.NDArray, values: npt.NDArray
) -> list[tuple[int, npt.NDArray]]:
    """Approach 1: Use Panda's groupby function."""
    # Create a DataFrame from the keys and values.
    df = pd.DataFrame({"keys": keys, "values": values})

    return [
        (cast(int, key), values_group["values"].to_numpy())
        for key, values_group in df.groupby("keys")
    ]


def approach_2(
    keys: npt.NDArray, values: npt.NDArray
) -> list[tuple[int, npt.NDArray]]:
    """Approach 2: Use custom groupby function."""
    return list(groupby(keys, values))  # type: ignore


def map_to_inputs(
    amount_elements: int, keys_range: int
) -> tuple[tuple[npt.NDArray, npt.NDArray], dict[str, Any]]:
    """Map input parameters to function arguments.

    Args:
        amount_elements: The number of elements in the array.
        keys_range: The range of values in the keys array.

    Returns:
        Tuple containing positional args and keyword args for the algorithms.
    """
    # Set random seed for reproducibility.
    rng = np.random.default_rng(69)

    # Generate a random keys and values array.
    keys = rng.integers(0, keys_range, size=(amount_elements,))
    values = rng.integers(0, 100, size=(amount_elements,))

    # Return the inputs.
    return (keys, values), {}


def main(args: argparse.Namespace) -> None:
    """The entry point of the program.

    Args:
        args: The command line arguments.
    """
    amount_elements = 2 ** np.arange(2, 29, 2)
    keys_range = 2 ** np.arange(2, 21, 2)

    plot_times(
        amount_elements,
        keys_range,
        map_to_inputs,
        approach_1,
        approach_2,
        "GroupBy comparison",
        "Pandas groupby",
        "Custom groupby function",
        "Amount of elements",
        "Range of keys",
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
