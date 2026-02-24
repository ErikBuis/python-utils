import logging
import multiprocessing as mp
import multiprocessing.queues
import multiprocessing.synchronize
import queue
import threading
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import (
    FIRST_COMPLETED,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
    wait,
)
from typing import Any, TypeVar

from typing_extensions import overload

T = TypeVar("T")

# Module-level cancel event set by the pool initializer in each worker process.
_worker_cancel_event: multiprocessing.synchronize.Event | None = None


@overload
def __init_process(
    cancel_event: multiprocessing.synchronize.Event,
    worker_init_fn: Callable[[int], None],
    worker_id_queue: multiprocessing.queues.Queue,
) -> None:
    pass


@overload
def __init_process(
    cancel_event: multiprocessing.synchronize.Event,
    worker_init_fn: None = None,
    worker_id_queue: None = None,
) -> None:
    pass


def __init_process(
    cancel_event: multiprocessing.synchronize.Event,
    worker_init_fn: Callable[[int], None] | None = None,
    worker_id_queue: multiprocessing.queues.Queue | None = None,
) -> None:
    """Initialize each worker process with a shared cancel event.

    Args:
        cancel_event: A multiprocessing event used to signal cancellation.
        worker_init_fn: An optional function to call on each worker subprocess
            with the worker id as its only argument.
        worker_id_queue: An optional multiprocessing queue used to pass worker
            ids to the initializer if worker_init_fn is not None.
    """
    global _worker_cancel_event
    _worker_cancel_event = cancel_event

    if worker_init_fn is not None and worker_id_queue is not None:
        worker_init_fn(worker_id_queue.get())


def __process_wrapper(
    fn: Callable[..., T], args: list[Any] = [], kwargs: dict[str, Any] = {}
) -> T:
    """Run fn(*args, **kwargs) only if the cancellation event has not been set.

    Args:
        fn: The function to call.
        args: Positional arguments passed to fn.
        kwargs: Keyword arguments passed to fn.

    Returns:
        The return value of fn(*args, **kwargs) if the cancellation event is
        not set.
    """
    global _worker_cancel_event
    if _worker_cancel_event is None:
        raise RuntimeError(
            "Worker cancel event is not initialized. This should never happen"
            " because the initializer is required to set it up."
        )

    # Only run the function if the cancellation event has not been set.
    if _worker_cancel_event.is_set():
        raise KeyboardInterrupt

    try:
        return fn(*args, **kwargs)
    except KeyboardInterrupt:
        # Set the shared cancel event so that this worker (and all other
        # workers) will skip any further tasks that _process_worker pulls from
        # the call queue after this one. Without this, workers race to pick up
        # the next queued task before the main process has a chance to set the
        # cancel event, which can lead to multiple workers running tasks after
        # a KeyboardInterrupt has already been received.
        _worker_cancel_event.set()
        raise


def parallelize_processes(
    tasks: Sequence[
        tuple[Callable[..., T]]
        | tuple[Callable[..., T], list[Any]]
        | tuple[Callable[..., T], list[Any], dict[str, Any]]
    ],
    max_workers: int | None = None,
    worker_init_fn: Callable[[int], None] | None = None,
) -> Iterator[T]:
    """Submit tasks to a process pool and yield results as they complete.

    Args:
        tasks: List of tuples containing:
            - A callable to execute.
            - An optional list of positional arguments to pass to the callable.
            - An optional dict of keyword arguments to pass to the callable.
        max_workers: Maximum number of worker processes. Defaults to len(tasks).
        worker_init_fn: If not None, this will be called on each worker
            subprocess with the worker id as its only argument (an int in
            [0, max_workers - 1]).

    Yields:
        The return value of each completed task, in completion order.
    """
    max_workers = max_workers or len(tasks)

    # mp.Event() uses shared memory (semaphore), which can only be shared with
    # worker processes during spawning, not pickled through the work queue. We
    # therefore pass it via initargs so each worker receives it at start-up and
    # stores it as a global.
    cancel_event = mp.Event()
    if worker_init_fn is None:
        executor = ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=__init_process,
            initargs=(cancel_event,),
        )
    else:
        worker_id_queue = mp.Queue()
        for i in range(max_workers):
            worker_id_queue.put(i)
        executor = ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=__init_process,
            initargs=(cancel_event, worker_init_fn, worker_id_queue),
        )

    try:
        # Submit all tasks, handling cancellation in __process_wrapper().
        futures = []
        for task in tasks:
            futures.append(
                executor.submit(__process_wrapper, *task)  # type: ignore
            )

        # Yield results as they complete.
        for future in as_completed(futures):
            yield future.result()
    except (KeyboardInterrupt, GeneratorExit):
        logging.warning(
            "KeyboardInterrupt received, trying to cancel pending tasks"
            " gracefully..."
        )
        cancel_event.set()
        raise
    except Exception:
        cancel_event.set()
        raise
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


def __init_thread(
    worker_init_fn: Callable[[int], None], worker_id_queue: queue.Queue
) -> None:
    """Initialize each worker thread by calling worker_init_fn with a worker id.

    Args:
        worker_init_fn: A function to call on each worker thread with the worker
            id as its only argument.
        worker_id_queue: A thread-safe queue used to distribute worker ids.
    """
    worker_init_fn(worker_id_queue.get())


def __thread_wrapper(
    cancel_event: threading.Event,
    fn: Callable[..., T],
    args: list[Any] = [],
    kwargs: dict[str, Any] = {},
) -> T:
    """Run fn(*args, **kwargs) only if the cancellation event has not been set.

    Args:
        cancel_event: A threading event used to signal cancellation.
        fn: The function to call.
        args: Positional arguments passed to fn.
        kwargs: Keyword arguments passed to fn.

    Returns:
        The return value of fn(*args, **kwargs) if the cancellation event is
        not set.
    """
    if cancel_event.is_set():
        raise KeyboardInterrupt

    return fn(*args, **kwargs)


def parallelize_threads(
    tasks: Sequence[
        tuple[Callable[..., T]]
        | tuple[Callable[..., T], list[Any]]
        | tuple[Callable[..., T], list[Any], dict[str, Any]]
    ],
    max_workers: int | None = None,
    worker_init_fn: Callable[[int], None] | None = None,
) -> Iterator[T]:
    """Submit tasks to a thread pool and yield results as they complete.

    Unlike parallelize_processes, threads cannot be forcibly interrupted
    mid-execution. Cancellation only prevents tasks that have not yet started
    from running; tasks that are already running will complete normally even
    after a KeyboardInterrupt is received.

    Args:
        tasks: List of tuples containing:
            - A callable to execute.
            - An optional list of positional arguments to pass to the callable.
            - An optional dict of keyword arguments to pass to the callable.
        max_workers: Maximum number of worker threads. Defaults to len(tasks).
        worker_init_fn: If not None, this will be called on each worker thread
            with the worker id as its only argument (an int in
            [0, max_workers - 1]).

    Yields:
        The return value of each completed task, in completion order.
    """
    max_workers = max_workers or len(tasks)

    # ThreadPoolExecutor doesn't have a built-in way to signal cancellation to
    # worker threads, so we use a shared threading.Event() that each thread
    # checks before starting a new task.
    cancel_event = threading.Event()
    if worker_init_fn is None:
        executor = ThreadPoolExecutor(max_workers=max_workers)
    else:
        worker_id_queue = queue.Queue()
        for i in range(max_workers):
            worker_id_queue.put(i)
        executor = ThreadPoolExecutor(
            max_workers=max_workers,
            initializer=__init_thread,
            initargs=(worker_init_fn, worker_id_queue),
        )

    try:
        # Submit all tasks, handling cancellation in __thread_wrapper().
        futures = []
        for task in tasks:
            futures.append(
                executor.submit(
                    __thread_wrapper, cancel_event, *task  # type: ignore
                )
            )

        # Yield results as they complete. We use wait() with a timeout instead
        # of as_completed() here because with as_completed(), the main thread
        # would wait for a thread to finish before waking up, which would create
        # a race condition where a new thread starts before the main thread has
        # a chance to set the cancel event, leading to more tasks running after
        # a KeyboardInterrupt has already been received. With wait() and
        # timeout=1, the main thread wakes up every second to check for a
        # KeyboardInterrupt, so it has a higher chance to catch it before extra
        # tasks start.
        while futures:
            done, futures = wait(
                futures, timeout=1, return_when=FIRST_COMPLETED
            )
            for future in done:
                yield future.result()
    except (KeyboardInterrupt, GeneratorExit):
        logging.warning(
            "KeyboardInterrupt received, trying to cancel pending tasks"
            " gracefully..."
        )
        cancel_event.set()
        raise
    except Exception:
        cancel_event.set()
        raise
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
