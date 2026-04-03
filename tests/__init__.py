import inspect
import logging
from collections.abc import Callable
from functools import partial

from loguru import logger
from loguru._handler import Message
from tqdm import tqdm
from typing_extensions import override

_last_logging_level: str | int | None = None
_last_filter: dict[str | None, str | int | bool] | None = None


# Intercept standard logging messages (also from external libraries) toward
# Loguru sinks. Source:
# https://github.com/Delgan/loguru?tab=readme-ov-file
class InterceptHandler(logging.Handler):
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
    global _last_logging_level, _last_filter
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
        logging.getLogger(name).addHandler(InterceptHandler())

    # Configure the root logger.
    logger.add(
        (
            custom_sink if custom_sink is not None else default_sink
        ),  # type: ignore
        level=logging_level,
        format=(
            "<green>{time:HH:mm:ss.SSS}</green>"
            + " | <level>{level:<8}</level>"
            + (
                " | <cyan>{name}:{line}</cyan>"
                if worker_id is None
                else " | <cyan>PID{process}:{name}:{line}</cyan>"
            )
            + " | <level>{message}</level>"
        ),
        filter={
            "": "INFO",  # default level for external libraries
            "__main__": "TRACE",  # all levels for the main file
            __package__: "TRACE",  # all levels for internal modules
            **filter,
        },
        colorize=True,
        enqueue=True,
    )


def worker_init_fn(
    worker_id: int,
    logging_level: str | int,
    filter: dict[str | None, str | int | bool],
) -> None:
    configure_root_logger(logging_level, filter=filter, worker_id=worker_id)


def make_worker_init_fn() -> Callable[[int], None]:
    if _last_logging_level is None or _last_filter is None:
        raise ValueError(
            "make_worker_init_fn() can only be called after"
            " configure_root_logger() has already been called by the main"
            " process."
        )
    return partial(
        worker_init_fn, logging_level=_last_logging_level, filter=_last_filter
    )


logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

# Disable logging messages while importing modules.
# The main module should configure the logger instead.
logger.remove()
