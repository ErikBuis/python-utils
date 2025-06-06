from __future__ import annotations

import inspect
import logging
import sys

from loguru import logger
from typing_extensions import override


# Intercept standard logging messages (also from external libraries) toward
# Loguru sinks. Source:
# https://github.com/Delgan/loguru?tab=readme-ov-file#entirely-compatible-with-standard-logging
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
) -> None:
    # Remove all existing handlers to avoid duplicate messages.
    for name in logging.root.manager.loggerDict.keys():
        logging.getLogger(name).handlers = []
        logging.getLogger(name).propagate = False
        logging.getLogger(name).addHandler(InterceptHandler())

    # Configure the root logger.
    logger.add(
        sys.stderr,
        level=logging_level,
        format=(
            "<green>{time:HH:mm:ss}</green>"
            + " | <level>{level:<8}</level>"
            + (
                " | <cyan>{name}:{line}</cyan>"
                if worker_id is None
                else " | <cyan>PID{process}:{name}:{line}</cyan>"
            )
            + " | <level>{message}</level>"
        ),
        filter={
            "": "INFO",  # Default level for external libraries.
            "__main__": "TRACE",  # All levels for the main file.
            __package__: "TRACE",  # All levels for internal modules.
            **filter,
        },
        enqueue=True,
    )


logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

# Disable logging messages while importing modules.
# The main module should configure the logger instead.
logger.remove()
