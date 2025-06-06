from __future__ import annotations

import argparse
import random
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from .. import configure_root_logger
from ..plot_times import plot_times


def approach_1(df: pd.DataFrame, col_name: str, row_idcs: list[int]) -> float:
    """Approach 1: Extract column as Series first."""
    series = df[col_name]
    result = 0
    for row_idx in row_idcs:
        result += series.iat[row_idx]
    return result


def approach_2(df: pd.DataFrame, col_name: str, row_idcs: list[int]) -> float:
    """Approach 2: Use .at for each access."""
    result = 0
    for row_idx in row_idcs:
        result += df.at[row_idx, col_name]
    return result


def map_to_inputs(
    amount_rows: int, amount_cols: int
) -> tuple[tuple[pd.DataFrame], dict[str, Any]]:
    # Set random seed for reproducibility.
    random.seed(69)

    # Construct the DataFrame.
    df = pd.DataFrame({
        f"col_{i}": [random.random() for _ in range(amount_rows)]
        for i in range(amount_cols)
    })

    # Return the inputs.
    return (df,), {
        "col_name": f"col_{random.randint(0, amount_cols - 1)}",
        "row_idcs": random.sample(
            list(range(amount_rows)), min(1000, amount_rows)
        ),
    }


def main(args: argparse.Namespace) -> None:
    """The entry point of the program.

    Args:
        args: The command line arguments.
    """
    amount_rows = 2 ** np.arange(0, 21, 2)
    amount_cols = 2 ** np.arange(3, 8, 1)

    plot_times(
        amount_rows,
        amount_cols,
        map_to_inputs,
        approach_1,
        approach_2,
        "Pandas indexing comparison",
        "First extract column as Series",
        "Use .at for each access",
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
