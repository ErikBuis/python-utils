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
from functools import partial
from multiprocessing.queues import Queue

from loguru import logger

from python_utils.modules.concurrent import (
    parallelize_processes,
    parallelize_threads,
)
from python_utils.modules.tqdm import (
    create_listener_sink,
    get_listener_queue,
    setup_listener_queue,
    tqdm_concurrent,
)

from .. import configure_root_logger

unittest.skip(
    "This is not a unit test and is meant to be run manually to visually check"
    " for issues with tqdm_concurrent when used with multiple processes and"
    " threads."
)

THREADS = 2
PROCESSES = 8


def _worker_init_fn(
    worker_id: int, listener_queue: Queue, logging_level: str | int
) -> None:
    """Initialize a worker process by connecting it to the listener.

    Args:
        worker_id: Index assigned by parallelize_processes.
        listener_queue: Queue created by the main process.
        logging_level: Log level to apply in this worker.
    """
    setup_listener_queue(listener_queue)
    configure_root_logger(
        logging_level, worker_id=worker_id, custom_sink=create_listener_sink()
    )


def _thread_worker(thread_id: int, process_id: int) -> None:
    """Run a single tqdm_concurrent bar, simulating incremental work.

    Args:
        thread_id: Index of this thread within its process.
        process_id: Index of the process (0 = main process).
    """
    desc = f"P{process_id}/T{thread_id}"
    logger.debug(f"Starting bar {desc}.")
    iterations = 10
    sleep = 0.02 + random.random() / 4 * iterations
    for i in tqdm_concurrent(range(iterations), desc=desc):
        if random.random() < 10 / iterations:
            logger.info(f"Test log from {desc} at {i} items")

        # Perform a high CPU load for a short time to make the bars more likely
        # to interfere with each other if there are any issues with the
        # listener queue.
        start = time.time()
        while time.time() - start < sleep:
            time.sleep(0.001)
            pass

    logger.debug(f"Finished bar {desc}.")


def _process_worker(process_id: int, threads: int = THREADS) -> int:
    """Run 2 threads via parallelize_threads inside a worker process.

    Args:
        process_id: Index of this worker process.
    """
    tasks = [(_thread_worker, [t, process_id]) for t in range(threads)]
    for _ in parallelize_threads(tasks, max_workers=threads):
        pass

    return process_id


def main(args: argparse.Namespace) -> None:
    """The entry point of the program.

    Args:
        args: The command line arguments.
    """
    listener_queue = get_listener_queue()
    worker_init_fn = partial(
        _worker_init_fn,
        listener_queue=listener_queue,
        logging_level=args.logging_level,
    )

    # Start worker processes, each running their own threads.
    process_tasks = [(_process_worker, [p]) for p in range(1, PROCESSES)]

    def _drain_processes() -> None:
        for p in parallelize_processes(
            process_tasks,
            max_workers=PROCESSES // 2,
            worker_init_fn=worker_init_fn,
        ):
            logger.success(f"Worker process {p} finished.")

    process_thread = threading.Thread(target=_drain_processes)
    process_thread.start()

    # Also run a test with multiple nested tqdm_concurrent bars in the main
    # process, to check that the listener queue doesn't interfere with normal
    # tqdm_concurrent usage outside.
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
    setup_listener_queue()
    configure_root_logger(
        args.logging_level, custom_sink=create_listener_sink()
    )

    # Log the command line arguments for reproducibility.
    logger.debug(f"{args=}")

    # Run the program.
    with logger.catch():
        main(args)
