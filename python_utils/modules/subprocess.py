from __future__ import annotations

import subprocess
from typing import Any


def subprocess_run_wrapper(
    *args: Any, **kwargs: Any
) -> subprocess.CompletedProcess:
    """Wrapper that raises an informative exception on error.

    Refer to the documentation of subprocess.run for explanation of the
    arguments and return values.
    """
    try:
        return subprocess.run(*args, **kwargs)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Command: {e.cmd} returned code {e.returncode}.\n"
            + (f"{e.stdout.decode('utf-8')}\n" if e.stdout else "")
            + (f"{e.stderr.decode('utf-8')}" if e.stderr else "")
        ) from e
