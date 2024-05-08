import argparse
import logging
import random
from collections.abc import Sequence
from typing import Any

import numpy as np
from plot_times.plot_times import plot_times

from custom.geometry import Interval, NumberSet


def map_to_inputs(
    amount_numbers: int, amount_intervals: int
) -> tuple[tuple[Sequence[NumberSet], Sequence[int | float]], dict[str, Any]]:
    # Set random seed for reproducibility.
    random.seed(69)

    # Decide on a set of possible bounds.
    possible_bounds = list(range(-1_0000_000, 1_000_000))
    possible_inclusions = [True, False]

    # Generate a list of random intervals. Do this by randomly choosing a
    # start and end bound for each interval and whether they are included
    # from the possible bounds and inclusions.
    intervals = []
    for _ in range(amount_intervals):
        start, end = 0, 0
        start_included, end_included = True, True
        while start == end:
            start = random.choice(possible_bounds)
            end = random.choice(possible_bounds)
            start_included = random.choice(possible_inclusions)
            end_included = random.choice(possible_inclusions)
        if start > end:
            start, end = end, start
            start_included, end_included = end_included, start_included
        intervals.append(Interval(start_included, start, end, end_included))
    numbersets = [NumberSet(interval) for interval in intervals]
    numbers = [random.choice(possible_bounds) for _ in range(amount_numbers)]
    return (numbersets, numbers), {}


def main(args: argparse.Namespace) -> None:
    """The entry point of the program.

    Args:
        args: The command line arguments.
    """
    amount_numbers = 2 ** np.arange(0, 11)
    amount_intervals = 2 ** np.arange(0, 11)

    plot_times(
        amount_numbers,
        amount_intervals,
        map_to_inputs,
        NumberSet.contains_parallel_naive,
        NumberSet.contains_parallel,
        "Find Numbers in Intervals",
        "Naive algorithm",
        "Lookup algorithm",
        "Amount of numbers",
        "Amount of intervals",
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
