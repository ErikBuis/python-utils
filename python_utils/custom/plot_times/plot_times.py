"""
This file measures the time it takes for two algorithms to perform a given
task. The two algorithms must depend on two input parameters: n and m.
The results are cached in "plot_times_{task_name}.pkl", and the plot is saved
as "plot_times_{task_name}.png".
"""

from __future__ import annotations

import pickle
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from ...modules.matplotlib import get_scalar_mappable_middle_white
from ...modules.numpy import NDArrayGeneric
from ...modules.pathlib import slugify
from ...modules.timeit import measure_times


def plot_times(
    ns: npt.NDArray[np.integer],
    ms: npt.NDArray[np.integer],
    map_to_inputs: Callable[[int, int], Any],
    algorithm1: Callable[..., Any],
    algorithm2: Callable[..., Any],
    task_name: str,
    algorithm1_name: str = "Algorithm 1",
    algorithm2_name: str = "Algorithm 2",
    n_name: str = "$n$",
    m_name: str = "$m$",
    use_log_scale: bool = True,
) -> None:
    """Plot the time it takes to perform algorithm 1 vs algorithm 2.

    Args:
        ns: The values of n passed to the algorithms.
            Shape: [amount_ns].
        ms: The values of m passed to the algorithms.
            Shape: [amount_ms].
        map_to_inputs: A function mapping a pair of (n, m) to the inputs that
            should be passed to the algorithms. Should return a tuple of
            (args, kwargs).
        algorithm1: The function representing algorithm 1. It takes the values
            returned by map_to_inputs().
        algorithm2: The function representing algorithm 2. It takes the values
            returned by map_to_inputs().
        task_name: The name of the task which will be used in the plot's title.
            e.g. "Discretize a Bezier Curve", "Find numbers in Intervals", etc.
        algorithm1_name: The name of algorithm 1 which will be used in its
            plot's title.
        algorithm2_name: The name of algorithm 2 which will be used in its
            plot's title.
        n_name: Name of the variable n which will be used as the plot's x-axis
            label.
        m_name: Name of the variable m which will be used as the plot's y-axis
            label.
        use_log_scale: Whether to use a log scale for the color map.
    """
    path_to_parent = Path(sys.argv[0]).resolve().parent
    task_name_slugified = slugify(task_name)
    path_to_cache = path_to_parent / f"plot_times_{task_name_slugified}.pkl"
    path_to_image = path_to_parent / f"plot_times_{task_name_slugified}.png"
    color_algorithm1 = "blue"
    color_algorithm2 = "orange"
    magic_colorbar = {"fraction": 0.046, "pad": 0.04}

    cache = (
        pickle.load(open(path_to_cache, "rb")) if path_to_cache.exists() else {}
    )

    # The cache is formatted like this:
    # {
    #     (algorithm1_name, algorithm2_name): {
    #         (n1, m1): (time_algorithm1, time_algorithm2),
    #         (n2, m2): (time_algorithm1, time_algorithm2),
    #         ...
    #     },
    #     ...
    # }
    cache_key = (algorithm1_name, algorithm2_name)
    times = cache.get(cache_key, {})

    # Calculate the times the algorithms take to execute and save them.
    cache[cache_key] = measure_times(
        times, ns.tolist(), ms.tolist(), map_to_inputs, algorithm1, algorithm2
    )
    pickle.dump(cache, open(path_to_cache, "wb"))

    # Create matrices from the times.
    matrix_time_algorithm1 = np.zeros((len(ns), len(ms)))
    matrix_time_algorithm2 = np.zeros((len(ns), len(ms)))
    for i, n in enumerate(ns):
        for j, m in enumerate(ms):
            matrix_time_algorithm1[i, j] = times[(n, m)][0]
            matrix_time_algorithm2[i, j] = times[(n, m)][1]

    # Prepare the matrices for plotting.
    matrix_time_algorithm1 = matrix_time_algorithm1.T
    matrix_time_algorithm2 = matrix_time_algorithm2.T

    # Plot the matrices. Use a green-white-red colormap to show the difference
    # between the two methods.
    fig, axs = plt.subplots(
        1,
        3,
        figsize=(
            3.3
            + len(str(max(ms))) * 0.1 * 3  # width of m tick labels
            + 0.35 * 3 * len(ns),  # width of cells
            1.1
            + len(str(max(ns))) * 0.1  # height of n tick labels
            + 0.35 * len(ms),  # height of cells
        ),
    )
    axs = cast(NDArrayGeneric[matplotlib.axes.Axes], axs)

    fig.suptitle(f"{task_name} (time in seconds)")

    # Get color map for both matrices.
    scalar_mappable = get_scalar_mappable_middle_white(
        np.concat([matrix_time_algorithm1, matrix_time_algorithm2]).flatten(),
        from_color="green",
        to_color="red",
        use_log_scale=use_log_scale,
    )

    # Plot the matrices.
    axs[0].imshow(
        matrix_time_algorithm1,
        cmap=scalar_mappable.get_cmap(),
        norm=scalar_mappable.norm,
    )
    axs[0].set_title(algorithm1_name, color=color_algorithm1)
    axs[0].set_xlabel(n_name)
    axs[0].set_ylabel(m_name)
    axs[0].invert_yaxis()
    axs[0].set_xticks(np.arange(len(ns)))
    axs[0].set_yticks(np.arange(len(ms)))
    axs[0].set_xticklabels(ns, rotation=90)
    axs[0].set_yticklabels(ms)
    fig.colorbar(scalar_mappable, ax=axs[0], **magic_colorbar)  # type: ignore

    axs[1].imshow(
        matrix_time_algorithm2,
        cmap=scalar_mappable.get_cmap(),
        norm=scalar_mappable.norm,
    )
    axs[1].set_title(algorithm2_name, color=color_algorithm2)
    axs[1].set_xlabel(n_name)
    axs[1].set_ylabel(m_name)
    axs[1].invert_yaxis()
    axs[1].set_xticks(np.arange(len(ns)))
    axs[1].set_yticks(np.arange(len(ms)))
    axs[1].set_xticklabels(ns, rotation=90)
    axs[1].set_yticklabels(ms)
    fig.colorbar(scalar_mappable, ax=axs[1], **magic_colorbar)  # type: ignore

    # Get color map for the difference matrix.
    scalar_mappable = get_scalar_mappable_middle_white(
        (matrix_time_algorithm1 - matrix_time_algorithm2).flatten(),
        from_color=color_algorithm1,
        to_color=color_algorithm2,
        use_log_scale=True,
        zero_is_white=True,
    )

    # Plot the difference matrix.
    axs[2].imshow(
        matrix_time_algorithm1 - matrix_time_algorithm2,
        cmap=scalar_mappable.get_cmap(),
        norm=scalar_mappable.norm,
    )
    axs[2].set_title(
        f"Which is faster?\n({algorithm1_name} - {algorithm2_name})"
    )
    axs[2].set_xlabel(n_name)
    axs[2].set_ylabel(m_name)
    axs[2].invert_yaxis()
    axs[2].set_xticks(np.arange(len(ns)))
    axs[2].set_yticks(np.arange(len(ms)))
    axs[2].set_xticklabels(ns, rotation=90)
    axs[2].set_yticklabels(ms)
    fig.colorbar(scalar_mappable, ax=axs[2], **magic_colorbar)  # type: ignore

    plt.tight_layout()
    plt.savefig(path_to_image)
    plt.show()
