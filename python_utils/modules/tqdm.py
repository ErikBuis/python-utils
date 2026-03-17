"""Tqdm-compatible progress bars that work across multiple concurrent processes.

This module provides a drop-in replacement for tqdm progress bars that works
correctly when multiple tqdm bars are active at the same time, even across
multiple processes. If multiple bars are active at a time, they will be
"stacked" on top of each other in the terminal, and log messages will be
printed above the bars. New bars will be added at the bottom of the bar block,
while finished ones will automatically move up and become permanent lines, so
the block never has empty rows.

Requirements:
- You must call logger.add() at least once in the main process and every worker
  process after importing this module.

...that's it! Just call tqdm_concurrent() instead of tqdm.tqdm() and everything
should "magically" work in both the main process and worker processes. See the
example usage below for details.

---

Additional usage notes:
- You must route all print statements through the logger and all progress bars
  through tqdm_concurrent(). Do NOT use print() and the native tqdm.tqdm(), as
  those would bypass the module's listener and cause corrupted terminal output.
- This module monkey-patches logger.add() by replacing the user-supplied sink
  with our custom sink that forwards all log messages to the listener queue.
  This applies to the main process and all worker processes. Do NOT import this
  module after registering any sinks with logger.add(), as those sinks would
  bypass the listener and cause corrupted terminal output.

---

Example usage:
```
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial

from loguru import logger
from python_utils.modules.tqdm import tqdm_concurrent


def your_worker_init_fn(logging_level: str | int) -> None:
    logger.add(sys.stderr, level=logging_level)


def your_thread_fn(...) -> ...:
    for item in tqdm_concurrent(...):
        ...

    return thread_result


def your_process_fn(...) -> ...:
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(your_thread_fn, ...) for ...]
        thread_results = [f.result() for f in futures]

    return process_result


def main(...) -> None:
    worker_init_fn = partial(your_worker_init_fn, logging_level)
    with ProcessPoolExecutor(initializer=worker_init_fn) as executor:
        futures = [executor.submit(your_process_fn, ...) for ...]
        process_results = [f.result() for f in futures]


if __name__ == "__main__":
    logger.add(sys.stderr, level=logging_level)
    main(...)
```
"""

import atexit
import multiprocessing as mp
import os
import queue as _queue_module
import re
import shutil
import sys
import threading
import uuid
from collections.abc import Callable, Iterable, Iterator
from multiprocessing.managers import BaseManager
from multiprocessing.queues import Queue
from types import TracebackType
from typing import Any, Generic, TypeVar

from loguru import logger
from loguru._handler import Message
from tqdm import tqdm

T = TypeVar("T")

# Global state for the message queue and listener thread.
_queue: Queue | None = None
_listener: threading.Thread | None = None

# Lock to prevent multiple threads in the same worker process from
# simultaneously trying to connect to the listener server.
_auto_connect_lock = threading.Lock()

# Environment variable names used to advertise the manager's address and
# authkey so that worker processes can auto-connect to it.
_ENV_ADDR = "_TQDM_CONCURRENT_ADDR"
_ENV_AUTHKEY = "_TQDM_CONCURRENT_AUTHKEY"

# Pre-compile a regex to match ANSI escape sequences, which are used for e.g.
# coloring and formatting log messages.
_ANSI_RE = re.compile(r"\x1b\[[\d;]*[a-zA-Z]")


# ------------------------------------------------------------------------------
# Listener thread implementation
# ------------------------------------------------------------------------------


def _visual_rows(ss: list[str], ncols: int) -> int:
    """Calculate the number of visual rows taken up by several strings.

    Args:
        ss: Strings to calculate the visual rows for.
        ncols: Number of columns in the terminal.

    Returns:
        The number of visual rows taken up by the log lines, accounting for
        line wrapping based on the terminal width.
    """
    return sum(
        sum(
            (len(line) - 1) // ncols + 1
            for line in _ANSI_RE.sub("", s).split("\n")
        )
        for s in ss
    )


class _NullSink:
    """File-like object that discards writes but reports stderr terminal info.

    Used as the `file` argument for tqdm bars so they track state and format
    correctly without writing anything to the terminal.

    The fileno() method is implemented to return the real stderr file
    descriptor, so that tqdm's `dynamic_ncols` can still detect the real
    terminal width.

    The `encoding` attribute is set to "utf-8" so that tqdm's internal _is_utf()
    check passes and Unicode block-fill characters are used instead of falling
    back to the ASCII `123456789#` set.
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
    sequences. For details on ANSI escape sequences, see:
    https://en.wikipedia.org/wiki/ANSI_escape_code

    The listener drains the queue in batches to merge multiple updates into a
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
    sink = _NullSink()
    active = {}
    bar_order = []
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

    Sends a shutdown message to the queue, waits for the listener thread to
    finish processing all remaining messages, and resets the module state.
    Called automatically via atexit on program exit. Safe to call multiple
    times.
    """
    global _queue, _listener

    if _queue is None or _listener is None:
        return

    # If we would immediately send the shutdown message, a bug would occur where
    # the listener shuts down while the loguru enqueue thread is still running,
    # which would cause the remaining messages to be lost and never printed. The
    # remove() call prevents this by calling join() on the enqueue thread,
    # ensuring all pending messages are processed before we send the shutdown
    # signal to the listener.
    logger.remove()

    _queue.put(("shutdown",))
    _listener.join()
    _queue = None
    _listener = None


# ------------------------------------------------------------------------------
# Queue sharing between main process and worker processes
# ------------------------------------------------------------------------------


class _QueueManager(BaseManager):
    pass


def _start_manager_server(actual_queue: Queue) -> None:
    """Start the manager server that exposes the queue to worker processes.

    Registers a get_queue() callable on a random loopback port and stores the
    address and a random authkey in environment variables so that spawned (or
    forked) worker processes can connect without any explicit argument passing.

    Args:
        actual_queue: The main process's multiprocessing queue.
    """
    _QueueManager.register(
        "get_queue",
        callable=lambda: actual_queue,
        exposed=("put",),  # expose only the Queue.put() method for safety
    )
    authkey = os.urandom(32)
    manager = _QueueManager(address=("127.0.0.1", 0), authkey=authkey)
    server = manager.get_server()
    threading.Thread(target=server.serve_forever, daemon=True).start()
    _, port = server.address  # type: ignore
    os.environ[_ENV_ADDR] = str(port)
    os.environ[_ENV_AUTHKEY] = authkey.hex()


def _connect_to_manager_server() -> Queue | None:
    """Connect to the manager server started by the main process.

    Reads the server address and authkey from environment variables and returns
    a proxy object for the queue. Returns None if the env vars are not set,
    which means the main process did not start a manager server (e.g. when
    running without this module in the main process).

    Returns:
        A proxy for the queue, or None if no server address is available.
    """
    port_str = os.environ.get(_ENV_ADDR)
    authkey_str = os.environ.get(_ENV_AUTHKEY)
    if port_str is None or authkey_str is None:
        return None

    _QueueManager.register("get_queue")
    manager = _QueueManager(
        address=("127.0.0.1", int(port_str)), authkey=bytes.fromhex(authkey_str)
    )
    manager.connect()
    return manager.get_queue()  # type: ignore


# ------------------------------------------------------------------------------
# Loguru sink patching
# ------------------------------------------------------------------------------


def _make_listener_sink(queue: Queue) -> Callable[[Message], None]:
    """Build a loguru sink that routes messages to the listener queue.

    Args:
        queue: The queue (or queue proxy) to send log messages to.

    Returns:
        A callable that loguru calls with each log message.
    """

    def listener_sink(msg: Message) -> None:
        queue.put(("log", msg))

    return listener_sink


def _patch_loguru(queue: Queue) -> None:
    """Monkey-patch loguru so every sink added in this process is auto-wrapped.

    We do not want log messages written directly to stderr (that would bypass
    the listener and corrupt the terminal output). Instead, every call to
    logger.add() is intercepted and the user-supplied sink is replaced by one
    that forwards the formatted message to the listener queue.

    The patch is applied once per process. Any sinks already registered before
    the patch are removed first so there is no double-output during the brief
    window between process start and the user's logger.add() call.

    Args:
        queue: The queue (or queue proxy) to forward all log messages to.
    """
    original_add = logger.__class__.add

    listener_sink = _make_listener_sink(queue)

    def patched_add(self: Any, sink: Any, **kwargs: Any) -> int:
        # Regardless of what sink the caller provides, replace it with the
        # listener sink.
        return original_add(self, listener_sink, **kwargs)  # type: ignore

    logger.__class__.add = patched_add

    # Remove any sinks that were registered before the patch (e.g. the default
    # stderr sink that loguru adds at import time).
    logger.remove()


# ---------------------------------------------------------------------------
# Module-level initialization
# ---------------------------------------------------------------------------


def _init() -> None:
    """Initialize the module for the current process.

    If called in the main process: create the queue, start the listener thread,
    start the manager server, and register the teardown hook.

    If called in worker processes: if the queue is already populated (if fork
    was used by the OS), do nothing. Otherwise (if spawn was used by the OS),
    connect to the manager server and apply the loguru patch.
    """
    global _queue, _listener

    is_main = mp.current_process().name == "MainProcess"

    if is_main:
        actual_queue = mp.Queue()
        _queue = actual_queue

        # Start the listener thread.
        # Python's shutdown sequence is as follows:
        # 1. The main thread reaches the end of its code.
        # 2. The interpreter automatically performs a join() on all remaining
        #    non-daemon threads. It will block here until they exit.
        # 3. Only after all non-daemon threads have completed does Python call
        #    the functions registered via the atexit module.
        # 4. Any threads marked as daemon=True are then abruptly terminated
        #    without cleanup.
        # This means that if we would make the listener thread a non-daemon
        # thread, the atexit handler would never run because the interpreter is
        # still waiting for that thread to exit, causing a deadlock. Thus, we
        # have to make it a daemon thread!
        _listener = threading.Thread(
            target=_run_listener, args=(actual_queue,), daemon=True
        )
        _listener.start()
        atexit.register(_teardown_listener)

        _start_manager_server(actual_queue)
        _patch_loguru(actual_queue)

    else:
        # If this process was forked, _queue is already a valid Queue inherited
        # from the parent process. In this case, we should reset _listener to
        # None so _teardown_listener() is a no-op in this worker.
        if _queue is not None:
            _listener = None
            return

        # If this process was spawned, connect to the manager server via env
        # vars. We need to use a lock here to prevent multiple threads in the
        # same worker process from simultaneously trying to connect to the
        # manager server, which would cause log message duplication.
        with _auto_connect_lock:
            if _queue is not None:
                return  # another thread beat us to it

            proxy_queue = _connect_to_manager_server()
            if proxy_queue is None:
                raise RuntimeError(
                    "Failed to connect to tqdm_concurrent manager server. Make"
                    " sure you import python_utils.modules.tqdm before spawning"
                    " any worker processes."
                )

            _queue = proxy_queue
            _patch_loguru(proxy_queue)


_init()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


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
              you need to supply a custom sink, please submit a feature request
              with your use case.
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
                f"Listener queue not initialized in {process_name}. Make sure"
                " you import python_utils.modules.tqdm before spawning any"
                " worker processes."
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
                "Listener queue not initialized. This should never happen."
            )

        _queue.put(("bar_update", self._id, n))

    def close(self) -> None:
        """Mark the bar as complete. Safe to call multiple times."""
        if self._closed:
            return

        self._closed = True

        if _queue is None:
            raise RuntimeError(
                "Listener queue not initialized. This should never happen."
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
