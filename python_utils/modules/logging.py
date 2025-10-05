from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def suppress_logging(name: str, level: str | int = "ERROR") -> Iterator[None]:
    """Context manager to temporarily suppress logging messages."""
    context_logger = logging.getLogger(name)
    original_level = context_logger.level
    context_logger.setLevel(logging.WARNING)
    try:
        yield
    finally:
        context_logger.setLevel(original_level)
