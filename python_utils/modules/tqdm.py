"""Tqdm-compatible progress bars that work across multiple concurrent processes.

This module provides a drop-in replacement for tqdm progress bars that work
correctly when multiple tqdm bars are active at the same time, even across
multiple processes.

It achieves this by running a single listener thread in the main process that
consumes messages from all worker processes and renders the bars and log lines
in a coordinated way. New bars will be added at the bottom of the tqdm block,
and finished ones will automatically move up and become permanent lines, so the
block never has empty rows and the display is stable without reordering.

Notes:
- The module assumes the project uses loguru with a configure_root_logger()
  helper. Worker log lines are routed through a custom sink so the main process
  listener can print them above the bars without interference.
- In order for the module to work, you have to insert callbacks at several
  points in your code:
  1. The main process must call setup_listener_queue() to initialize the shared
     queue and start the listener thread.
  2. You must acquire the main process's queue using get_listener_queue() and
     pass it to your worker initializer function.
  3. Each worker process's initializer function must call setup_listener_queue()
     with the main process's queue to initialize the queue in that process.
  4. Each worker process must configure loguru to use the custom sink created
     by create_listener_sink() so log lines are routed to the listener.


Example usage:
```
from functools import partial
from multiprocessing.queue import Queue

from loguru import logger
from python_utils.modules.concurrent import (
    parallelize_processes, parallelize_threads
)  # optional to make worker management easier, not required for tqdm_concurrent
from python_utils.modules.tqdm import (
    create_listener_sink,
    get_listener_queue,
    setup_listener_queue,
    tqdm_concurrent,
)

from . import configure_root_logger


def your_worker_init_fn(
    worker_id: int,
    listener_queue: Queue,
    logging_level: str | int,
    **kwargs: Any,
) -> None:
    ...  # any worker setup

    setup_listener_queue(listener_queue)
    configure_root_logger(
        logging_level, worker_id=worker_id, custom_sink=create_listener_sink()
    )


def your_thread_fn(...) -> ...:
    ...  # do work

    for item in tqdm_concurrent(...):
        ...  # do work on item

    ...  # do work

    return thread_result


def your_process_fn(...) -> ...:
    ...  # do work

    your_thread_tasks = [
        (
            your_thread_fn,
            [...]  # args to pass to your_thread_fn
            {...}  # kwargs to pass to your_thread_fn
        ),
        ...  # more thread tasks
    ]

    for thread_result in parallelize_threads(your_thread_tasks):
        ...  # do work with thread_result

    ... # do work

    return process_result


def main(...) -> None:
    your_process_tasks = [
        (
            your_process_fn,
            [...]  # args to pass to your_process_fn
            {...}  # kwargs to pass to your_process_fn
        ),
        ...  # more process tasks
    ]

    for process_result in parallelize_processes(
        your_process_tasks,
        worker_init_fn=partial(
            your_worker_init_fn,
            logging_level=...,
            listener_queue=get_listener_queue(),
        ),
    ):
        ...  # do work with process_result


if __name__ == "__main__":
    setup_listener_queue()
    configure_root_logger(custom_sink=create_listener_sink())

    main(...)
```
"""

import atexit
import multiprocessing as mp
import threading
import uuid
from collections.abc import Callable, Iterable, Iterator
from multiprocessing.queues import Queue
from types import TracebackType
from typing import Any, Generic, TypeVar

from loguru._handler import Message
from tqdm import tqdm

T = TypeVar("T")

_queue: Queue | None = None
_listener: threading.Thread | None = None


def _run_listener(queue: Queue) -> None:
    """Consume messages from all processes and render them in the main process.

    Runs as a daemon thread in the main process. All tqdm bar objects live
    exclusively in this thread, so no external locking is needed for the active
    dict itself. However, tqdm's own class lock (an RLock) is still acquired
    when adjusting bar positions to prevent concurrent renders.

    Message types:
    - ("log", message: str): Print log message above the bars via tqdm.write().
    - ("bar_enter", bar_id: str, tqdm_kwargs: dict): Create a new bar, shifting
      existing bars up.
    - ("bar_update", bar_id: str, n: int): Advance bar by n steps.
    - ("bar_exit", bar_id: str): Close bar, shift remaining bars down, and
      print a permanent line.
    - ("shutdown",): Exit the listener loop and allow the thread to finish.

    Args:
        queue: Queue that all processes send messages to.
    """
    active: dict[str, tqdm] = {}

    while True:
        match queue.get():
            case ("log", message):
                tqdm.write(message, end="")

            case ("bar_enter", bar_id, tqdm_kwargs):
                active[bar_id] = tqdm(
                    position=len(active), leave=False, **tqdm_kwargs
                )

            case ("bar_update", bar_id, n) if bar_id in active:
                active[bar_id].update(n)

            case ("bar_exit", bar_id, leave) if bar_id in active:
                bar = active[bar_id]

                # Snapshot the display string at 100% before touching any state.
                final_str = str(bar)

                with tqdm.get_lock():
                    closing_pos = bar.pos  # negative int, e.g. -2 for pos=2
                    bar.close()  # erases the row
                    del active[bar_id]

                    # Increase the pos attribute for every bar that was below
                    # the closed one by one. tqdm's internal logic ensures that
                    # the bar with the most negative pos is always rendered at
                    # the bottom.
                    for other in active.values():
                        if other.pos < closing_pos:  # more negative = lower
                            other.pos += 1

                if leave:
                    tqdm.write(final_str)

            case ("shutdown",):
                return

            case _:
                pass  # ignore malformed messages


def _teardown_listener() -> None:
    """Gracefully shut down the listener thread in the main process.

    Sends a shutdown sentinel to the queue, waits for the listener thread to
    finish processing all remaining messages, and resets the module state.
    Called automatically via atexit when setup_listener_queue() is used in the
    main process, so manual calls are not required. Safe to call multiple times.
    """
    global _queue, _listener

    if _queue is None or _listener is None:
        return

    _queue.put(("shutdown",))
    _listener.join()
    _queue = None
    _listener = None


def setup_listener_queue(queue: Queue | None = None) -> None:
    """Set up the shared queue and start the listener if in the main process.

    Args:
        queue: If called in the main process, this argument must be None. If
            called in a worker process, this argument must be the queue created
            by the main process.
    """
    global _queue, _listener

    process_name = mp.current_process().name
    is_main = process_name == "MainProcess"
    if is_main and queue is not None or not is_main and queue is None:
        raise RuntimeError(
            "setup_listener_queue() must be called with queue=None in the main"
            " process, and with queue=<main_process_queue> in worker processes,"
            " but this was not the case. Please do the following: (1) Call"
            " setup_listener_queue() in the main process to initialize the"
            " queue, (2) then call get_listener_queue() in the main process and"
            " pass it to your worker initializer function, and (3) finally pass"
            " it to setup_listener_queue() within each worker process to"
            " initialize the queue there."
        )

    if _queue is not None:
        raise RuntimeError(
            f"Listener queue already initialized in {process_name}. Call"
            " get_listener_queue() to access it."
        )

    if is_main:
        _queue = mp.Queue()

        # Start the listener thread.
        _listener = threading.Thread(
            target=_run_listener, args=(_queue,), daemon=True
        )
        _listener.start()

        # Let the listener thread finish cleanly on program exit.
        atexit.register(_teardown_listener)
    else:
        _queue = queue


def get_listener_queue() -> Queue:
    """Get the shared queue for the listener.

    This function should only be called after the queue has been set up by
    setup_listener_queue().

    Returns:
        The shared queue for the listener.
    """
    global _queue

    if _queue is None:
        process_name = mp.current_process().name
        raise RuntimeError(
            f"Listener queue not initialized in {process_name}. Call"
            " setup_listener_queue() first."
        )

    return _queue


def create_listener_sink() -> Callable[[Message], None]:
    """Create a loguru sink that routes log messages to the listener.

    Returns:
        A function that takes a loguru Message and sends it to the listener.
    """
    global _queue

    queue = _queue
    if queue is None:
        process_name = mp.current_process().name
        raise RuntimeError(
            f"Listener queue not initialized in {process_name}. Call"
            " setup_listener_queue() first."
        )

    def listener_sink(msg: Message) -> None:
        queue.put(("log", msg))

    return listener_sink


class tqdm_concurrent(Generic[T]):
    """Drop-in replacement for tqdm that works across multiple processes.

    The interface is identical to tqdm.tqdm().

    Args:
        iterable: Optional iterable to wrap. If not provided, the bar must be
            manually updated by calling update(). If provided, the bar will
            automatically update on each iteration.
        **kwargs: Keyword arguments forwarded to tqdm.tqdm(). Note that the
            position argument is managed internally and will be silently ignored
            if supplied.
    """

    def __init__(
        self, iterable: Iterable[T] | None = None, **kwargs: Any
    ) -> None:
        self._id = str(uuid.uuid4())
        self._closed = False
        self._iterable = iterable

        if _queue is None:
            process_name = mp.current_process().name
            raise RuntimeError(
                f"Listener queue not initialized in {process_name}. Call"
                " setup_listener_queue() first."
            )

        # Strip position and leave, the listener manages it internally.
        kwargs.pop("position", None)
        self._leave = kwargs.pop("leave", True)

        # Infer total from the iterable when not explicitly provided.
        if "total" not in kwargs and iterable is not None:
            try:
                kwargs["total"] = len(iterable)  # type: ignore
            except TypeError:
                pass

        _queue.put(("bar_enter", self._id, kwargs))

    def update(self, n: int = 1) -> None:
        """Increment the bar by n iterations.

        Args:
            n: Number of iterations to add.
        """
        if self._closed:
            return

        if _queue is None:
            raise RuntimeError(
                "Listener queue not initialized. Call setup_listener_queue()"
                " first."
            )

        _queue.put(("bar_update", self._id, n))

    def close(self) -> None:
        """Mark the bar as complete. Safe to call multiple times."""
        if self._closed:
            return

        self._closed = True

        if _queue is None:
            raise RuntimeError(
                "Listener queue not initialized. Call setup_listener_queue()"
                " first."
            )

        _queue.put(("bar_exit", self._id, self._leave))

    def __iter__(self) -> Iterator[T]:
        """Iterate over the wrapped iterable, updating the bar after each item.

        Returns:
            The original iterable that was passed to the constructor.
        """
        if self._iterable is None:
            raise TypeError(
                "tqdm_concurrent object is not iterable (no iterable was"
                " provided)"
            )
        try:
            for item in self._iterable:
                yield item
                self.update(1)
        finally:
            self.close()

    def __enter__(self) -> "tqdm_concurrent":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()
