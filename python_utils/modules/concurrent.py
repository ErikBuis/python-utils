import logging
import multiprocessing as mp
import multiprocessing.queues
import multiprocessing.synchronize
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, TypeVar

from typing_extensions import overload

T = TypeVar("T")

# Module-level cancel event set by the pool initializer in each worker process.
_worker_cancel_event: multiprocessing.synchronize.Event | None = None


@overload
def __init_worker(
    cancel_event: multiprocessing.synchronize.Event,
    worker_init_fn: Callable[[int], None],
    worker_id_queue: multiprocessing.queues.Queue,
) -> None:
    pass


@overload
def __init_worker(
    cancel_event: multiprocessing.synchronize.Event,
    worker_init_fn: None = None,
    worker_id_queue: None = None,
) -> None:
    pass


def __init_worker(
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


def __run_wrapper(
    fn: Callable[..., T], args: list[Any], kwargs: dict[str, Any]
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


def run_parallel(
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
            initializer=__init_worker,
            initargs=(cancel_event,),
        )
    else:
        worker_id_queue = mp.Queue()
        executor = ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=__init_worker,
            initargs=(cancel_event, worker_init_fn, worker_id_queue),
        )
        for i in range(max_workers):
            worker_id_queue.put(i)

    try:
        # Submit all tasks, wrapping in __run_wrapper() to handle cancellation.
        futures = []
        for task in tasks:
            fn = task[0]
            args = task[1] if len(task) > 1 else []
            kwargs = task[2] if len(task) > 2 else {}
            futures.append(executor.submit(__run_wrapper, fn, args, kwargs))

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
