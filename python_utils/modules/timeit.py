from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from timeit import timeit
from typing import Any

logger = logging.getLogger(__name__)


def auto_timeit(
    stmt: str | Callable = "pass", setup: str | Callable = "pass"
) -> float:
    """Automatically determine the number of runs for timeit.

    This function runs timeit a number of times and averages over the runs. It
    automatically tries to make the total run time of this function no longer
    than 2 seconds.

    Args:
        stmt: The statement to time.
        setup: The setup code.

    Returns:
        The average time per run.
    """
    n = 1
    total = n
    t = timeit(stmt, setup, number=n)

    while t < 0.2:
        n *= 10
        total += n
        t = timeit(stmt, setup, number=n)

    return t / total  # Normalise to time-per-run


def measure_times(
    times: dict[tuple[int, int], tuple[float, float]],
    ns: Sequence[int],
    ms: Sequence[int],
    map_to_inputs: Callable[[int, int], tuple[Sequence[Any], dict[str, Any]]],
    algorithm1: Callable[..., Any],
    algorithm2: Callable[..., Any],
) -> dict[tuple[int, int], tuple[float, float]]:
    """Measure the time it takes for two different algorithms to do a task.

    Warning: this function doesn't check whether the outputs of the two
    algorithms are actually equal. If you are comparing two algorithms, you
    should, of course, do this yourself.

    Args:
        times: A cache of the times it took for the algorithms to execute.
            This cache will be updated in place.
        ns: The values of n passed to the algorithms.
            Shape: [amount_ns].
        ms: The values of m passed to the algorithms.
            Shape: [amount_ms].
        map_to_inputs: A function mapping a pair of (n, m) to the inputs that
            should be passed to the algorithms. Should return a tuple of
            (args, kwargs).
        algorithm1: The function representing algorithm 1. It takes the args
            and kwargs returned by map_to_inputs().
        algorithm2: The function representing algorithm 2. It takes the args
            and kwargs returned by map_to_inputs().

    Returns:
        The updated cache of the times it took for the algorithms to execute,
        including the times for the new inputs.
    """
    logger.info(f"{'n':<9}{'m':<9}{'Algorithm 1':<24}{'Algorithm 2':<24}")

    for n in ns:
        for m in ms:
            # Get the times from the cache if they exist, otherwise measure
            # them and save them in the cache.
            if (n, m) in times:
                time_algorithm1, time_algorithm2 = times[(n, m)]
            else:
                args, kwargs = map_to_inputs(n, m)
                time_algorithm1 = auto_timeit(
                    lambda: algorithm1(*args, **kwargs)
                )
                time_algorithm2 = auto_timeit(
                    lambda: algorithm2(*args, **kwargs)
                )
                times[(n, m)] = (time_algorithm1, time_algorithm2)

            # Print the results for a fast overview.
            logger.info(
                f"{n:<9}{m:<9}"
                f"{time_algorithm1:<24}{time_algorithm2:<24}"
                f"{'<  ' if time_algorithm1 < time_algorithm2 else '  >'}"
            )

    return times
