"""Minimal integration test for tqdm_concurrent with processes and threads.

Run with: `python -m tests.modules.tqdm`

Run this test manually to visually check for issues with tqdm_concurrent when
used with multiple processes and threads. It creates multiple worker processes,
each running multiple threads, and each thread runs a tqdm_concurrent bar with
some simulated work. The test also includes some logging from the worker
threads to check that logs are properly routed through the listener queue
without interfering with the tqdm bars.
"""

import argparse
import random
import threading
import time
import unittest

from loguru import logger

from python_utils.custom.init import configure_root_logger, make_worker_init_fn
from python_utils.modules.concurrent import (
    parallelize_processes,
    parallelize_threads,
)
from python_utils.modules.tqdm import tqdm_concurrent

unittest.skip(
    "This is not a unit test and is meant to be run manually to visually check"
    " for issues with tqdm_concurrent when used with multiple processes and"
    " threads."
)

THREADS = 2
PROCESSES = 8


def _worker_thread(p: int, t: int) -> int:
    """Run a single tqdm_concurrent bar, simulating incremental work.

    Args:
        p: Index of the process (0 = main process).
        t: Index of this thread within its process.
    """
    desc = f"P{p}/T{t}"
    sleep = 0.02 + random.random() * 1.25
    for i in tqdm_concurrent(range(20), desc=desc):
        # Log a message at random times to check that logs are properly routed
        # through the listener queue without interfering with the tqdm bars.
        if random.random() < 0.5:
            logger.info(f"Test log from {desc} at {i} items")

        # Perform a high CPU load for a short time to make the bars more likely
        # to flicker if there are any issues with the listener implementation.
        start = time.time()
        while time.time() - start < sleep:
            time.sleep(0.001)
            pass

    return t


def _worker_process(p: int) -> int:
    """Run threads inside a worker process.

    Args:
        p: Index of this worker process (0 = main process).
    """
    tasks = [(_worker_thread, [p, t]) for t in range(THREADS)]
    for t in parallelize_threads(tasks, max_workers=THREADS):
        logger.success(f"Worker thread P{p}/T{t} finished.")

    return p


def main(args: argparse.Namespace) -> None:
    """The entry point of the program.

    Args:
        args: The command line arguments.
    """
    # Start worker processes, each running their own threads.
    process_tasks = [(_worker_process, [p]) for p in range(1, PROCESSES)]

    def _drain_processes() -> None:
        for p in parallelize_processes(
            process_tasks,
            max_workers=PROCESSES // 2,
            worker_init_fn=make_worker_init_fn(),
        ):
            logger.success(f"Worker process P{p} finished.")

    process_thread = threading.Thread(target=_drain_processes)
    process_thread.start()

    # Run threads within the main process.
    _worker_process(0)

    # Also run a test with multiple nested tqdm_concurrent bars in the main
    # process to stress test that the listener can handle updates from multiple
    # bars that appear and disappear quickly. We set leave=False on the
    # innermost bar to test that case as well.
    for _ in tqdm_concurrent(range(10), desc="Main process outer bar"):
        for _ in tqdm_concurrent(range(10), desc="Main process inner bar"):
            for _ in tqdm_concurrent(
                range(10), desc="Main process innermost bar", leave=False
            ):
                time.sleep(0.01)

    # Wait for worker processes to finish before exiting, so their logs don't
    # get cut off.
    process_thread.join()


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
