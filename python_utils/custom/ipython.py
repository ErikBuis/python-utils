# pyright: reportUnusedImport=false

from __future__ import annotations


def init_ipython() -> None:
    """Initialize the IPython environment."""
    import logging
    import math  # noqa: F401
    import os  # noqa: F401
    import random  # noqa: F401
    import sys  # noqa: F401
    import time  # noqa: F401
    from pathlib import Path  # noqa: F401

    global logger
    logger = logging.getLogger(__name__)

    logging.basicConfig(
        level="DEBUG",
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        force=True,  # necessary to make logging work in ipython
    )
    # In versions < 3.11, setting force=True causes the asyncio library within
    # ipython to output "DEBUG Using selector: EpollSelector". To suppress this
    # output, we set the logging level of the asyncio library to WARNING.
    logging.getLogger("asyncio").setLevel(logging.WARNING)
