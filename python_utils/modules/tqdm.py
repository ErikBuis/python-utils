"""Tqdm-compatible progress bars that work across multiple concurrent processes.

This module provides a drop-in replacement for tqdm progress bars that work
correctly when multiple tqdm bars are active at the same time, even across
multiple processes.

It achieves this by running a single listener thread in the main process that
consumes messages from all worker processes and renders the bars and log lines
in a coordinated way. New bars will be added at the bottom of the tqdm block,
and finished ones will automatically move up and become permanent lines, so the
block never has empty rows and the display is stable without reordering.

Requirements and usage notes:
- The module assumes the project uses loguru with a configure_root_logger()
  helper. Worker log lines are routed through a custom sink so the main process
  listener can print them above the bars without interference.
- You MUST route all print statements through loguru, and can NOT use the native
  tqdm.tqdm() function any more when you are using this module.
- In order for the module to work, you have to insert callbacks at several
  points in your code:
  1. The main process must call setup_listener_queue() to initialize the shared
     queue and start the listener thread.
- If you are using single-processing, you are done! Otherwise, also follow the
  next steps:
  2. For each worker process, you must acquire the main process's queue using
     get_listener_queue() and pass it to your worker initializer function.
  3. Each worker process's initializer function must call setup_listener_queue()
     with that queue to set it up within that process.
  4. Each worker process must configure loguru to use the custom sink created
     by create_listener_sink() so log lines are routed to the listener.


Example usage:
```
from functools import partial
from multiprocessing.queues import Queue

from loguru import logger
from python_utils.modules.concurrent import (
    parallelize_processes, parallelize_threads
)  # optional to make worker management easier, not required for this module
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
import queue as _queue_module
import re
import shutil
import sys
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

_ANSI_RE = re.compile(r"\x1b\[[\d;]*[a-zA-Z]")


def _visual_len(s: str) -> int:
    """Calculate the visual length of a string, ignoring ANSI escape sequences.

    Args:
        s: The input string, which may contain ANSI escape sequences for
            coloring or formatting.

    Returns:
        The visual length of the string, ignoring ANSI escape sequences.
    """
    return len(_ANSI_RE.sub("", s))


def _visual_rows(ss: list[str], ncols: int) -> int:
    """Calculate the number of visual rows taken up by several strings.

    Args:
        ss: Strings to calculate the visual rows for.
        ncols: Number of columns in the terminal.

    Returns:
        The number of visual rows taken up by the log lines, accounting for
        line wrapping based on the terminal width.
    """
    rows = 0
    for log_msg in ss:
        lines = log_msg.split("\n")
        for line in lines:
            rows += (_visual_len(line) - 1) // ncols + 1
    return rows


class _NullSink:
    """File-like object that discards writes but reports stderr terminal info.

    Used as the `file` argument for tqdm bars so they track state and format
    correctly without writing anything to the terminal.

    The fileno() method is implemented to return the real stderr file
    descriptor, so that tqdm's `dynamic_ncols` can still detect the real
    terminal width.

    The `encoding` attribute is set to "utf-8" so that tqdm's internal
    _is_utf() check passes and Unicode block-fill characters are used
    instead of falling back to the ASCII `123456789#` set.
    """

    encoding: str = "utf-8"

    def write(self, s: str) -> int:
        return len(s)

    def flush(self) -> None:
        pass

    def isatty(self) -> bool:
        return sys.stderr.isatty()

    def fileno(self) -> int:
        return sys.stderr.fileno()


def _run_listener(queue: Queue) -> None:
    """Consume messages from all processes and render them in the main process.

    Instead of delegating to tqdm.write() (which causes flickering), this
    listener redirects all log messages and bar updates to a single render
    function. All actual terminal output is produced manually with ANSI escape
    sequences. For details on these helpers, see:
    https://en.wikipedia.org/wiki/ANSI_escape_code

    The listener drains the queue in batches, to merge multiple updates into a
    single render pass, which improves efficiency.

    We always reserve the bottom row for an empty line.

    Message types:
    - ("log", message: str): Print log message above the bars.
    - ("bar_enter", bar_id: str, tqdm_kwargs: dict): Create a new bar.
    - ("bar_update", bar_id: str, n: int): Advance bar by n steps.
    - ("bar_exit", bar_id: str, leave: bool): Close bar, optionally leaving a
      permanent line.
    - ("shutdown",): Exit the listener loop.

    Args:
        queue: Queue that all processes send messages to.
    """
    sink: _NullSink = _NullSink()
    active: dict[str, tqdm] = {}
    bar_order: list[str] = []
    prev_bar_rows = 0

    def render(log_msgs: list[str]) -> None:
        """Write log_msgs above the bars and redraw all active bars.

        Args:
            log_msgs: Log lines to print above the bar block.
        """
        nonlocal active, bar_order, prev_bar_rows

        if not log_msgs and not bar_order:
            return

        ncols, nrows = shutil.get_terminal_size()

        # Calculate how many rows the bars currently take up.
        curr_bar_rows = min(len(bar_order), nrows - 1)

        buf = []

        # Start atomic update.
        buf.append("\x1b[?2026h")

        # Pre-scroll as many rows up as needed to fit the new content.
        # This mitigates flickering due to autoscrolling (mostly).
        # I spent a lot of time on trying to find a solution that works without
        # any flickering at all, but it seems to be basically impossible due to
        # the way terminal rendering and autoscrolling works. A solution that
        # fully prevents flickering is "\x1b[{scroll}S", but this has the
        # downside of not preserving the scrollback buffer, so you lose the
        # ability to scroll up to see previous logs. The current solution of
        # pre-scrolling with newlines and then moving the cursor back up seems
        # to be the best compromise, as it preserves the scrollback buffer and
        # only causes slight flickering.
        scroll = curr_bar_rows + _visual_rows(log_msgs, ncols) - prev_bar_rows
        if scroll > 0:
            buf.append("\n" * scroll)
            buf.append(f"\x1b[{scroll}A")

        # Move cursor up to top of the bar block and clear to end of the screen.
        if prev_bar_rows > 0:
            buf.append(f"\x1b[{prev_bar_rows}A")
            buf.append("\x1b[J")

        # Print log messages.
        for log_msg in log_msgs:
            buf.append(log_msg)

        # Redraw bars.
        for bar_id in bar_order:
            active[bar_id].ncols = ncols
            bar_str = str(active[bar_id]) + "\n"
            buf.append(bar_str)

        # End atomic update.
        buf.append("\x1b[?2026l")

        sys.stderr.buffer.write("".join(buf).encode())
        sys.stderr.buffer.flush()

        prev_bar_rows = curr_bar_rows

    # Main loop.
    while True:
        # Block until at least one message arrives.
        batch = [queue.get()]

        # Drain all remaining pending messages without blocking.
        while True:
            try:
                batch.append(queue.get_nowait())
            except _queue_module.Empty:
                break

        # Accumulate log lines and state changes, then render once.
        log_msgs = []
        shutdown = False

        for msg in batch:
            match msg:
                case ("log", message):
                    log_msgs.append(message)

                case ("bar_enter", bar_id, tqdm_kwargs):
                    bar = tqdm(
                        leave=False,
                        file=sink,
                        dynamic_ncols=True,
                        position=0,
                        **tqdm_kwargs,
                    )
                    active[bar_id] = bar
                    bar_order.append(bar_id)

                case ("bar_update", bar_id, n) if bar_id in active:
                    active[bar_id].update(n)

                case ("bar_exit", bar_id, leave) if bar_id in active:
                    if leave:
                        active[bar_id].ncols = (
                            shutil.get_terminal_size().columns
                        )
                        log_msgs.append(str(active[bar_id]) + "\n")
                    active[bar_id].close()
                    del active[bar_id]
                    bar_order.remove(bar_id)

                case ("shutdown",):
                    shutdown = True
                    break

                case _:
                    pass

        render(log_msgs)

        if shutdown:
            break


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
        **kwargs: Keyword arguments forwarded to tqdm.tqdm(). The following
            arguments are reserved and will be managed internally:
            - leave: This argument is supported but will be managed internally
              to ensure correct behavior. Defaults to True. You can set this to
              False to have the bar not leave a permanent line when it finishes,
              but the listener will manage the actual terminal output to ensure
              the display remains stable and correctly ordered.
            - position: Silently ignored if supplied.
            - ncols: Silently ignored if supplied. The listener will handle
              dynamic column resizing automatically.
            - dynamic_ncols: Silently ignored if supplied. The listener will
              handle dynamic column resizing automatically.
            - file: Silently ignored if supplied. The output is always written
              to sys.stderr through a custom sink that the listener manages. If
              you need this feature, please submit a feature request with your
              use case.
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

        # Strip out reserved kwargs that are managed internally, and save the
        # leave value.
        self._leave = kwargs.pop("leave", True)
        kwargs.pop("file", None)
        kwargs.pop("ncols", None)
        kwargs.pop("dynamic_ncols", None)
        kwargs.pop("position", None)

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
