import inspect
import logging
import multiprocessing as mp
from collections.abc import Callable
from functools import partial

from loguru import logger
from loguru._handler import Message
from tqdm import tqdm
from typing_extensions import override

_last_logging_level: str | int | None = None
_last_filter: dict[str | None, str | int | bool] | None = None


def _find_importing_package() -> str:
    """Find the package name of the __init__.py that imported this module.

    Returns:
        The package name of the __init__.py file that imported this module. This
        is equivalent to using __package__ in the __init__.py file itself.
    """
    frame = inspect.currentframe()
    assert frame is not None

    this_filename = frame.f_code.co_filename
    frame = frame.f_back
    filename = ""  # silence unbound variable warning

    # Walk up the stack, skipping frames in this file, frozen importlib
    # frames, and site-packages frames (e.g. pytest's assertion rewriter,
    # which inserts its exec_module() into the import chain even for modules
    # it does not rewrite).
    while frame:
        filename = frame.f_code.co_filename
        is_self = filename == this_filename
        is_frozen = "importlib" in filename and "_bootstrap" in filename
        is_import_hook = frame.f_code.co_name == "exec_module"
        if not is_self and not is_frozen and not is_import_hook:
            break
        frame = frame.f_back

    assert frame is not None
    if not filename.endswith("__init__.py"):
        raise ImportError(
            f"{__name__} must be imported from an __init__.py file, but was"
            f" imported from {filename!r}."
        )

    package = frame.f_globals.get("__package__")
    assert isinstance(package, str) and package
    return package


class _InterceptHandler(logging.Handler):
    """Intercept standard logging messages and redirect them to Loguru.

    This handler is used to redirect all logging messages to Loguru, including
    those from external libraries that use the standard logging module.

    Source:
    https://github.com/Delgan/loguru?tab=readme-ov-file
    """

    @override
    def emit(self, record: logging.LogRecord) -> None:
        # Get corresponding Loguru level if it exists.
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where the logged message originated.
        frame, depth = inspect.currentframe(), 0
        while frame:
            filename = frame.f_code.co_filename
            is_logging = filename == logging.__file__
            is_frozen = "importlib" in filename and "_bootstrap" in filename
            if depth > 0 and not (is_logging or is_frozen):
                break
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def configure_root_logger(
    logging_level: str | int,
    filter: dict[str | None, str | int | bool] = {},
    worker_id: int | None = None,
    custom_sink: Callable[[Message], None] | None = None,
) -> None:
    """Configure the root logger to redirect all logging messages to Loguru.

    This function should be called by the main process at the start of the
    program to ensure that all logging messages are properly redirected to
    Loguru.

    Args:
        logging_level: The logging level to use for the root logger. This can be
            a string (e.g. "INFO") or an integer (e.g. logging.INFO).
        filter: A dictionary mapping module names to logging levels. This allows
            you to set different minimal logging levels for different modules.
            The keys can be module names (e.g. "my_module") or None to specify
            the default logging level for all modules not explicitly listed in
            the dictionary. The values can be logging levels as strings (e.g.
            "DEBUG"), integers (e.g. logging.DEBUG), or booleans (e.g. False) to
            entirely enable/disable logging for the corresponding module.
        worker_id: The worker ID to include in the log messages. This is useful
            when running multiple worker processes to distinguish log messages
            from different workers. This argument should be None when called by
            the main process.
        custom_sink: A custom sink function to redirect log messages to. This
            allows you to use a different output destination, such as a file or
            a remote logging service, but can also serve as a callback to
            further customize the log messages before they are output. If None,
            uses tqdm.write(), which prevents writes to sys.stderr while
            preventing overlap with tqdm progress bars.
    """
    is_main = mp.current_process().name == "MainProcess"
    if worker_id is not None and is_main:
        raise ValueError(
            "worker_id should always be None when called by the main process."
        )

    global _last_logging_level, _last_filter, _package
    _last_logging_level = logging_level
    _last_filter = filter

    # Create a default sink that uses tqdm.write() when inside a tqdm progress
    # bar. This prevents overlap between the progress bar and the log messages.
    # Source: https://github.com/Delgan/loguru/issues/135
    def default_sink(msg: Message) -> None:
        tqdm.write(msg, end="")

    # Redirect all standard logging messages to the InterceptHandler, which
    # sends them to Loguru.
    for name in logging.root.manager.loggerDict.keys():
        logging.getLogger(name).handlers = []
        logging.getLogger(name).propagate = False
        logging.getLogger(name).addHandler(_InterceptHandler())

    # Configure the root logger.
    logger.add(
        (
            custom_sink if custom_sink is not None else default_sink
        ),  # type: ignore
        level=logging_level,
        format=(
            "<green>{time:HH:mm:ss.SSS}</green>"
            + " | <level>{level}</level>"
            + (
                " | <cyan>{name}:{line}</cyan>"
                if worker_id is None
                else " | <cyan>PID{process}:{name}:{line}</cyan>"
            )
            + " | <level>{message}</level>"
        ),
        filter={
            "": "INFO",  # external libraries
            "__main__": "TRACE",  # main file
            "__mp_main__": "TRACE",  # main file in spawned child processes
            _package: "TRACE",  # internal modules
            **filter,
        },
        colorize=True,
        enqueue=True,
    )


def _worker_init_fn(
    worker_id: int,
    logging_level: str | int,
    filter: dict[str | None, str | int | bool],
) -> None:
    """Initialization function for worker processes to configure their loggers.

    This function should be called by worker processes at the start of their
    execution to ensure that their logging messages are properly redirected to
    Loguru with the same configuration as the main process.

    Args:
        See configure_root_logger() for an explanation of the arguments.
    """
    configure_root_logger(logging_level, filter=filter, worker_id=worker_id)


def make_worker_init_fn() -> Callable[[int], None]:
    """Create a worker initialization function.

    The returned function automatically configures the logger with the same
    configuration as the main process. Additionally, it only requires the
    worker ID as an argument. This is useful when the worker initialization
    function needs to be passed to a library that does not allow passing
    additional arguments.

    The returned function is usually used as the `initializer` argument in
    `ProcessPoolExecutor` or the `worker_init_fn` argument in
    `torch.utils.data.DataLoader`.

    If you want to write a wrapper around the worker initialization function,
    please use the following pattern:
    >>> from collections.abc import Callable
    >>> from functools import partial

    >>> def _your_worker_init_fn(
    ...     _worker_init_fn: Callable[[int], None], worker_id: int
    ... ) -> None:
    ...     _worker_init_fn(worker_id)
    ...     # YOUR CODE HERE

    >>> def your_make_worker_init_fn() -> Callable[[int], None]:
    ...     return partial(_your_worker_init_fn, make_worker_init_fn())

    Returns:
        A function that takes a worker ID as its only argument.
    """
    if _last_logging_level is None or _last_filter is None:
        raise ValueError(
            "make_worker_init_fn() can only be called after"
            " configure_root_logger() has already been called by the main"
            " process."
        )

    return partial(
        _worker_init_fn, logging_level=_last_logging_level, filter=_last_filter
    )


# Determine the name of the importing package.
_package = _find_importing_package()

# Configure the root logger to redirect all logging messages to Loguru.
logging.basicConfig(handlers=[_InterceptHandler()], level=0, force=True)

# Disable logging messages while importing modules.
logger.remove()
