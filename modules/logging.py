import logging
from typing import Any, Literal


logger = logging.getLogger(__name__)

__is_logger_enabled_debug: bool | None = None
__is_logger_enabled_info: bool | None = None
__is_logger_enabled_warning: bool | None = None
__is_logger_enabled_error: bool | None = None
__is_logger_enabled_critical: bool | None = None


def configure_root_logger(
    logging_level: int,
    fmt: str | None = "%(asctime)s %(levelname)s %(message)s",
    datefmt: str | None = "%H:%M:%S",
    style: Literal["%", "{", "$"] = "%",
    validate: bool = True,
    *,
    defaults: dict[str, Any] | None = None,
) -> None:
    """Configure the root logger to log to the console.

    Args:
        logging_level: The logging level to set. One of the constants in the
            logging module.
    """
    # Remove all existing handlers.
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    for handler in root_logger.handlers:
        root_logger.removeHandler(handler)

    # Add a new handler that logs to the console.
    handler = logging.StreamHandler()
    handler.setLevel(logging_level)
    handler.setFormatter(
        logging.Formatter(
            fmt=fmt,
            datefmt=datefmt,
            style=style,
            validate=validate,
            defaults=defaults,
        )
    )
    root_logger.addHandler(handler)


def is_logger_enabled_debug() -> bool:
    """Check whether the logger is enabled for debug messages.

    Returns:
        Whether the logger is enabled for debug messages.
    """
    global __is_logger_enabled_debug
    if __is_logger_enabled_debug is None:
        __is_logger_enabled_debug = logger.isEnabledFor(logging.DEBUG)
    return __is_logger_enabled_debug


def is_logger_enabled_info() -> bool:
    """Check whether the logger is enabled for info messages.

    Returns:
        Whether the logger is enabled for info messages.
    """
    global __is_logger_enabled_info
    if __is_logger_enabled_info is None:
        __is_logger_enabled_info = logger.isEnabledFor(logging.INFO)
    return __is_logger_enabled_info


def is_logger_enabled_warning() -> bool:
    """Check whether the logger is enabled for warning messages.

    Returns:
        Whether the logger is enabled for warning messages.
    """
    global __is_logger_enabled_warning
    if __is_logger_enabled_warning is None:
        __is_logger_enabled_warning = logger.isEnabledFor(logging.WARNING)
    return __is_logger_enabled_warning


def is_logger_enabled_error() -> bool:
    """Check whether the logger is enabled for error messages.

    Returns:
        Whether the logger is enabled for error messages.
    """
    global __is_logger_enabled_error
    if __is_logger_enabled_error is None:
        __is_logger_enabled_error = logger.isEnabledFor(logging.ERROR)
    return __is_logger_enabled_error


def is_logger_enabled_critical() -> bool:
    """Check whether the logger is enabled for critical messages.

    Returns:
        Whether the logger is enabled for critical messages.
    """
    global __is_logger_enabled_critical
    if __is_logger_enabled_critical is None:
        __is_logger_enabled_critical = logger.isEnabledFor(logging.CRITICAL)
    return __is_logger_enabled_critical


def is_logger_enabled(logging_level: int) -> bool:
    """Check whether the logger is enabled for messages of a certain level.

    Args:
        logging_level: The logging level to check. One of the constants in the
            logging module.

    Returns:
        Whether the logger is enabled for messages of the given level.
    """
    if logging_level == logging.DEBUG:
        return is_logger_enabled_debug()
    if logging_level == logging.INFO:
        return is_logger_enabled_info()
    if logging_level == logging.WARNING:
        return is_logger_enabled_warning()
    if logging_level == logging.ERROR:
        return is_logger_enabled_error()
    if logging_level == logging.CRITICAL:
        return is_logger_enabled_critical()

    raise ValueError(
        f"Invalid logging level: {logging_level}. Please choose one of"
        " logging.DEBUG, logging.INFO, logging.WARNING, or logging.ERROR."
    )
