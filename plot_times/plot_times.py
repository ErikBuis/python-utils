"""
This file measures the time it takes for two algorithms to perform a given
task. The two algorithms must depend on two input parameters: n and m.
The results are cached in "plot_times_{task_name}.pkl", and the plot is saved
as "plot_times_{task_name}.png".
"""

import pickle
import re
import unicodedata
from collections.abc import Callable, Sequence
from pathlib import Path
from timeit import timeit
from typing import Any, Generic, TypeVar, cast

import matplotlib.axes
import matplotlib.cm
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt


T = TypeVar("T")


class NDArrayGeneric(np.ndarray, Generic[T]):
    """np.ndarray that allows for static type hinting of generics."""

    def __getitem__(self, key) -> T:
        return super().__getitem__(key)  # type: ignore


def auto_timeit(stmt: str | Callable = "pass", setup: str | Callable = "pass"):
    """Automatically determine the number of runs for timeit.

    This function runs timeit a number of times and averages over the runs. It
    automatically tries to make the total run time of this function no longer
    than 2 seconds.

    Args:
        stmt: The statement to time.
        setup: The setup code.

    Returns:
        The average time per run.
    """
    n = 1
    total = n
    t = timeit(stmt, setup, number=n)

    while t < 0.2:
        n *= 10
        total += n
        t = timeit(stmt, setup, number=n)

    return t / total  # Normalise to time-per-run


def measure_times(
    times: dict[tuple[int, int], tuple[float, float]],
    ns: npt.NDArray[np.int_],
    ms: npt.NDArray[np.int_],
    map_to_inputs: Callable[[int, int], tuple[Sequence[Any], dict[str, Any]]],
    algorithm1: Callable[..., Any],
    algorithm2: Callable[..., Any],
) -> dict[tuple[int, int], tuple[float, float]]:
    """Measure the time it takes for two different algorithms to do a task.

    Warning: this function doesn't check whether the outputs of the two
    algorithms are actually equal. If you are comparing two algorithms, you
    should, of course, do this yourself.

    Args:
        times: A cache of the times it took for the algorithms to execute.
            This cache will be updated in place.
        ns: The values of n passed to the algorithms.
            Shape: [amount_ns].
        ms: The values of m passed to the algorithms.
            Shape: [amount_ms].
        map_to_inputs: A function mapping a pair of (n, m) to the inputs that
            should be passed to the algorithms. Should return a tuple of
            (args, kwargs).
        algorithm1: The function representing algorithm 1. It takes the args
            and kwargs returned by map_to_inputs().
        algorithm2: The function representing algorithm 2. It takes the args
            and kwargs returned by map_to_inputs().

    Returns:
        times: The updated cache of the times it took for the algorithms to
            execute, including the times for the new inputs.
    """
    print(f"{'n':<9}{'m':<9}{'Algorithm 1':<24}{'Algorithm 2':<24}")

    for n in ns:
        for m in ms:
            # Get the times from the cache if they exist, otherwise measure
            # them and save them in the cache.
            if (n, m) in times:
                time_algorithm1, time_algorithm2 = times[(n, m)]
            else:
                args, kwargs = map_to_inputs(n, m)
                time_algorithm1 = auto_timeit(
                    lambda: algorithm1(*args, **kwargs)
                )
                time_algorithm2 = auto_timeit(
                    lambda: algorithm2(*args, **kwargs)
                )
                times[(n, m)] = (time_algorithm1, time_algorithm2)

            # Print the results for a fast overview.
            print(
                f"{n:<9}{m:<9}"
                f"{time_algorithm1:<24}{time_algorithm2:<24}"
                f"{'<  ' if time_algorithm1 < time_algorithm2 else '  >'}"
            )

    return times


def get_scalar_mappable(
    values: Sequence[float],
    from_color: str = "red",
    to_color: str = "green",
    use_log_scale: bool = False,
    zero_is_white: bool = False,
) -> matplotlib.cm.ScalarMappable:
    """Get a ScalarMappable with color map: from_color -> white -> to_color.

    Args:
        values: The values to map to colors.
        from_color: The color to map the smallest value(s) to.
        to_color: The color to map the largest value(s) to.
        use_log_scale: Whether to use a log scale.
        zero_is_white: If True, zero values will be mapped to white. Otherwise,
            the average between vmin and vmax will be mapped to white.

    Returns:
        scalar_mappable: ScalarMappable color map for the coefficients.
            Use scalar_mappable.get_cmap() to get the color map.
            Use scalar_mappable.norm to get the norm.
                The norm maps the range [vmin, vmax] to [0, 1].
                To get vmin and vmax, use scalar_mappable.norm.vmin and
                    scalar_mappable.norm.vmax respectively.
                To map a value to the range [0, 1], use
                    scalar_mappable.norm(value).
    """
    values_arr = np.array(values)
    vmin = values_arr.min()
    vmax = values_arr.max()

    # If zero values should be mapped to white, then we need to make sure that
    # there is at least one negative value and one positive value.
    if zero_is_white:
        if vmin > 0:
            vmin = -vmin
        elif vmax < 0:
            vmax = -vmax

    # Use a log scale if specified, otherwise use a linear scale.
    # norm is a function that maps the range [vmin, vmax] to [0, 1].
    if use_log_scale:
        values_arr_abs = np.abs(values_arr)
        if zero_is_white:
            # For the linear width, choose a value such that around 10% of the
            # values are mapped to (something close to) white.
            values_arr_abs.sort()
            linear_width = values_arr_abs[int(len(values_arr_abs) * 0.1)]
        else:
            # Make the linear width as small as possible.
            linear_width = values_arr_abs.min()
        norm = matplotlib.colors.AsinhNorm(
            vmin=vmin, vmax=vmax, linear_width=linear_width  # type: ignore
        )
    else:
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    # Map zero values to white if specified, otherwise map the average between
    # vmin and vmax to white.
    white_value = norm(0) if zero_is_white else 0.5

    # Create a from_color -> white -> to_color color map.
    color_list = [(0, from_color), (white_value, "white"), (1, to_color)]

    # Create the ScalarMappable color map.
    return matplotlib.cm.ScalarMappable(
        norm=norm,
        cmap=(
            matplotlib.colors.LinearSegmentedColormap.from_list(
                f"{from_color}_white_{to_color}", color_list
            )
        ),
    )


def slugify(value: str, convert_ascii: bool = True) -> str:
    """Make a string suitable for use in URLs and filenames.

    Convert to ASCII if convert_ascii True. Convert spaces or repeated dashes
    to single dashes. Remove characters that aren't alphanumerics, underscores,
    or hyphens. Convert to lowercase. Also strip leading and trailing
    whitespace, dashes, and underscores.

    Taken from:
    https://github.com/django/django/blob/master/django/utils/text.py
    """
    value = str(value)
    value = (
        unicodedata.normalize("NFKD", value)
        .encode("ascii", "ignore")
        .decode("ascii")
        if convert_ascii
        else unicodedata.normalize("NFKC", value)
    )
    value = re.sub(r"[^\w\s-]", "", value.lower())
    return re.sub(r"[-\s]+", "-", value).strip("-_")


def plot_times(
    ns: npt.NDArray[np.int_],
    ms: npt.NDArray[np.int_],
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
            should be passed to the algorithms.
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
    path_to_parent = Path(__file__).resolve().parent
    task_name_slugified = slugify(task_name)
    path_to_cache = path_to_parent / f"plot_times_{task_name_slugified}.pkl"
    path_to_image = path_to_parent / f"plot_times_{task_name_slugified}.png"
    color_algorithm1 = "blue"
    color_algorithm2 = "orange"
    magic_colorbar = {"fraction": 0.046, "pad": 0.04}

    cache = (
        pickle.load(open(path_to_cache, "rb"))
        if path_to_cache.exists()
        else {}
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
        times, ns, ms, map_to_inputs, algorithm1, algorithm2
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
    matrix_time_algorithm1 = matrix_time_algorithm1.T[::-1]
    matrix_time_algorithm2 = matrix_time_algorithm2.T[::-1]

    # Plot the matrices. Use a green-white-red colormap to show the difference
    # between the two methods.
    fig, axs = plt.subplots(
        1, 3, figsize=(4.5 + 0.35 * 3 * len(ns), 1.5 + 0.35 * len(ms))
    )
    axs = cast(NDArrayGeneric[matplotlib.axes.Axes], axs)

    fig.suptitle(f"Time to {task_name} (in seconds)")

    # Get color map for both matrices.
    scalar_mappable = get_scalar_mappable(
        np.concatenate((matrix_time_algorithm1, matrix_time_algorithm2))
        .flatten()
        .tolist(),
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
    scalar_mappable = get_scalar_mappable(
        (matrix_time_algorithm1 - matrix_time_algorithm2).flatten().tolist(),
        from_color=color_algorithm1,
        to_color=color_algorithm2,
        zero_is_white=True,
        use_log_scale=True,
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
