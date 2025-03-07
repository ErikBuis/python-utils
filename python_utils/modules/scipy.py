import logging
from collections.abc import Callable
from inspect import signature
from typing import Literal, cast

import matplotlib.axes
import numpy as np
import numpy.typing as npt
import scipy.optimize
from scipy.spatial import Voronoi


logger = logging.getLogger(__name__)


def plot_fitted_curve(
    ax: matplotlib.axes.Axes,
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    func: Callable[..., npt.NDArray[np.float64]],
    plot_confidence: bool = True,
    confidence: Literal["68", "95", "99.7"] = "95",
    sliding_window: float = 0.05,
    log_scale: bool = False,
    kwargs_optimize_curve_fit: dict | None = None,
    kwargs_plot: dict | None = None,
    kwargs_fill_between: dict | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Plot a fitted curve with varying confidence intervals.

    Note: The plotted data will be given labels. Make sure to call ax.legend()
    after this function if you want to display these labels.

    Args:
        ax: The axes to plot on.
        x: The x values.
            Shape: [P]
        y: The y values.
            Shape: [P]
        func: The function to fit the data to, called as f(x, ...). It must
            take the independent variable as the first argument and a variable
            number of parameters C to fit as its remaining arguments. Note that
            a high number of parameters may lead to overfitting (high variance)
            while a low number may lead to a poor fit (high bias).
        plot_confidence: Whether to plot the confidence interval as well as
            the fitted curve.
        confidence: The confidence level in percentage.
            Choose from "68", "95", or "99.7".
        sliding_window: The relative width of the sliding window compared to
            the range of x values. The sliding window will be used for
            calculating the standard deviation of the residuals. Based on the
            standard deviation, the confidence interval will be calculated.
            A smaller value will result in a more accurate but less smooth
            confidence interval.
        log_scale: Whether the x-axis is scaled logarithmically. Will be used
            to adjust the sliding window dynamically with the x values. This
            argument should be set to True if you are calling either
            ax.set_xscale("log") or ax.set_xscale("symlog"). Note that this
            function does not set/change the x-axis scale itself.
        kwargs_optimize_curve_fit: Keyword arguments for
            scipy.optimize.curve_fit().
        kwargs_plot: Keyword arguments for the fitted curve.
            Will be passed to ax.plot().
        kwargs_fill_between: Keyword arguments for the confidence interval
            area. Will be passed to ax.fill_between().

    Returns:
        Tuple containing:
        - Optimal values for the parameters so that the sum of the squared
            residuals of f(x, *popt) - y is minimized.
            Shape: [C]
        - The estimated approximate covariance of popt. The diagonals provide
            the variance of the parameter estimate. To compute one standard
            deviation errors on the parameters, use
            perr = np.sqrt(np.diag(pcov)). Note that the relationship between
            cov and parameter error estimates is derived based on a linear
            approximation to the model function around the optimum. When this
            approximation becomes inaccurate, cov may not provide an accurate
            measure of uncertainty.
            If the Jacobian matrix at the solution doesn't have a full rank,
            then this variable is a matrix filled with np.inf. Covariance
            matrices with large condition numbers (e.g. computed with
            numpy.linalg.cond) may indicate that results are unreliable.
            Shape: [C, C]
    """
    # Find with how much we need to multiply the stddev to get the desired
    # confidence interval.
    if confidence == "68":
        times_stddev = 1
    elif confidence == "95":
        times_stddev = 1.96
    elif confidence == "99.7":
        times_stddev = 3
    else:
        raise ValueError("Confidence must be one of '68', '95', or '99.7'.")

    # Catch any invalid input.
    x = np.array(x)
    y = np.array(y)

    if len(x) == 0 or len(y) == 0:
        raise ValueError("x and y must not be empty.")
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape.")
    if x.ndim != 1:
        raise ValueError("x and y must be 1D arrays.")

    # Give the kwargs a default value if they are not provided.
    if kwargs_optimize_curve_fit is None:
        kwargs_optimize_curve_fit = {}
    if kwargs_plot is None:
        kwargs_plot = {}
    if kwargs_fill_between is None:
        kwargs_fill_between = {}
    if "label" not in kwargs_plot:
        kwargs_plot["label"] = "Fitted curve"
    if "color" not in kwargs_plot:
        kwargs_plot["color"] = "green"
    if "zorder" not in kwargs_plot:
        kwargs_plot["zorder"] = 1.5
    if "label" not in kwargs_fill_between:
        kwargs_fill_between["label"] = f"{confidence}% interval"
    if "color" not in kwargs_fill_between:
        kwargs_fill_between["color"] = "green"
    if "alpha" not in kwargs_fill_between:
        kwargs_fill_between["alpha"] = 0.2
    if "zorder" not in kwargs_fill_between:
        kwargs_fill_between["zorder"] = 0.5

    # Sort the values if they are not sorted yet.
    if np.any(np.diff(x) < 0):
        sort_indices = np.argsort(x)
        x = x[sort_indices]
        y = y[sort_indices]

    # Fit a curve to the data.
    C = len(signature(func).parameters) - 1
    if len(x) < C:
        logger.warning(
            f"This function needs at least {C} data points to be able to fit a"
            f" curve through them, but only {len(x)} were provided. The curve"
            " will not be plotted."
        )
        return np.full(C, np.nan), np.full((C, C), np.nan)
    popt, pcov = scipy.optimize.curve_fit(
        func, x, y, **kwargs_optimize_curve_fit
    )

    # Create a linear/logarithmic space for plotting the curve.
    x_space = (
        np.logspace(np.log10(x[0]), np.log10(x[-1]), 100)
        if log_scale
        else np.linspace(x[0], x[-1], 100)
    )
    y_fitted = func(x_space, *popt)

    # Plot the fitted curve.
    ax.plot(x_space, y_fitted, **kwargs_plot)

    if not plot_confidence:
        return popt, pcov

    # Calculate confidence intervals with varying width.
    y_upper = np.full_like(y_fitted, np.nan)
    y_lower = np.full_like(y_fitted, np.nan)
    full_width = (
        np.log10(x[-1]) - np.log10(x[0]) if log_scale else x[-1] - x[0]
    )
    sliding_window_half_width = full_width * sliding_window / 2
    sliding_window_left = (
        10 ** (np.log10(x_space) - sliding_window_half_width)
        if log_scale
        else x_space - sliding_window_half_width
    )
    sliding_window_right = (
        10 ** (np.log10(x_space) + sliding_window_half_width)
        if log_scale
        else x_space + sliding_window_half_width
    )
    from_idx = 0  # inclusive
    to_idx = 0  # exclusive
    prev_non_nan_idx_upper = -1
    prev_non_nan_idx_lower = -1

    for i, x_val in enumerate(x_space):
        # Find the data points close to the current x_val. We do this in a way
        # such that the overall time complexity is O(n) instead if checking
        # all values at each iteration, because this would be O(n^2).
        left_bound = x_val - sliding_window_left[i]  # inclusive
        right_bound = x_val + sliding_window_right[i]  # exclusive
        while from_idx < len(x) and x[from_idx] < left_bound:
            from_idx += 1
        while to_idx < len(x) and x[to_idx] < right_bound:
            to_idx += 1

        # Calculate the residuals in the sliding window.
        residuals = y[from_idx:to_idx] - func(x[from_idx:to_idx], *popt)
        residuals_pos = residuals[residuals >= 0]
        residuals_neg = residuals[residuals < 0]

        if len(residuals_pos) > 1:
            # Calculate the stddev of the residuals.
            stddev_pos = np.sqrt(
                np.sum(residuals_pos**2) / (len(residuals_pos) - 1)
            )

            # Calculate the upper and lower bounds using the stddev.
            y_upper[i] = func(x_val, *popt) + times_stddev * stddev_pos

            # Substitute NaNs with the interpolated average between the
            # previous non-NaN value and this one.
            if prev_non_nan_idx_upper == -1:
                y_upper[:i] = y_upper[i]
            else:
                y_upper[prev_non_nan_idx_upper : i + 1] = np.linspace(
                    y_upper[prev_non_nan_idx_upper],
                    y_upper[i],
                    i + 1 - prev_non_nan_idx_upper,
                )
            prev_non_nan_idx_upper = i

        if len(residuals_neg) > 1:
            # Calculate the stddev of the residuals.
            stddev_neg = np.sqrt(
                np.sum(residuals_neg**2) / (len(residuals_neg) - 1)
            )

            # Calculate the upper and lower bounds using the stddev.
            y_lower[i] = func(x_val, *popt) - times_stddev * stddev_neg

            # Substitute NaNs with the interpolated average between the
            # previous non-NaN value and this one.
            if prev_non_nan_idx_lower == -1:
                y_lower[:i] = y_lower[i]
            else:
                y_lower[prev_non_nan_idx_lower : i + 1] = np.linspace(
                    y_lower[prev_non_nan_idx_lower],
                    y_lower[i],
                    i + 1 - prev_non_nan_idx_lower,
                )
            prev_non_nan_idx_lower = i

    # Substitute NaNs with the last non-NaN value.
    y_upper[prev_non_nan_idx_upper + 1 :] = y_upper[prev_non_nan_idx_upper]
    y_lower[prev_non_nan_idx_lower + 1 :] = y_lower[prev_non_nan_idx_lower]

    # Approximate the confidence interval's bounds.
    kwargs_optimize_curve_fit["p0"] = popt
    popt_upper, _ = scipy.optimize.curve_fit(
        func, x_space, y_upper, **kwargs_optimize_curve_fit
    )
    popt_lower, _ = scipy.optimize.curve_fit(
        func, x_space, y_lower, **kwargs_optimize_curve_fit
    )
    y_upper = func(x_space, *popt_upper)
    y_lower = func(x_space, *popt_lower)

    # Fill the confidence interval area.
    ax.fill_between(x_space, y_lower, y_upper, **kwargs_fill_between)

    return popt, pcov


def voronoi_constrain_to_rect(
    points: npt.NDArray[np.float64], rect: tuple[float, float, float, float]
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.intc],
    list[list[int]],
    npt.NDArray[np.signedinteger],
    list[list[int]],
]:
    """Compute Voronoi regions, but constrain all regions in a rectangle.

    The input must be 2D.

    The convex hull around the input points must be non-degenerate (i.e. the
    input points must not be colinear).

    The Vonorio diagram will be computed using a Euclidean plane.

    If the points in or on the rectangle were passed to the original Voronoi
    class, its attributes would have the following shapes:
    Shape of vor.points: [P', 2]
    Shape of vor.vertices: [V, 2]
    Shape of vor.ridge_points: [R, 2]
    Length of vor.ridge_vertices: R
    Shape of vor.point_region: [P']
    Length of vor.regions: P' + 1

    The core idea for this function (mirroring the points) has been taken from:
    https://stackoverflow.com/a/33602171/15636460
    However, the code itself was not taken from this answer because of a bug on
    rect edges and a different desired output format.

    Args:
        points: The points to compute the Voronoi diagram for.
            Shape: [P, 2]
        rect: The rectangle to clip the infinite regions in. The rectangle is
            represented by the tuple (x_min, y_min, x_max, y_max).

    Returns:
        Tuple containing:
        - The points in or on the rectangle.
            Shape: [P', 2]
        - The original vertices returned by Voronoi, plus vertices added on the
            given rectangle to constrain the infinite regions.
            Shape: [V + X, 2]
        - The original ridge_points returned by Voronoi, plus ridges added on
            the given rectangle to constrain the infinite regions.
            Shape: [R + X, 2]
        - The original ridge_vertices returned by Voronoi, plus ridges added on
            the given rectangle to constrain the infinite regions.
            Length: R + X
        - The original point_region returned by Voronoi.
            Shape: [P']
        - The original regions returned by Voronoi, where the infinite regions
            have been constrained to finite ones.
            Length: P' + 1
    """
    # This function mirrors the points in 4 directions, after which it computes
    # the Voronoi diagram. Theoretically, the runtime of this algorithm is
    # still O(n log n), where n is the number of original points. This is
    # because we are passing 5n points to the Voronoi class, which therefore
    # has a runtime of:
    # O(5n log(5n)) = O(5n log(5) + 5n log(n))
    #               = O(c_1 n log(n) + c_2 n)
    #               = O(n log(n))
    # However, in practice, n does not go to infinity, so the runtime of this
    # function is approximately 10 times slower:
    # 5n log(5n) = 5n log(5) + 5n log(n)
    #           ~= 5n log(n) + 5n log(n)
    #            = 10 * n log(n)

    # The input must be 2D.
    if points.shape[1] != 2:
        raise ValueError(
            f"Points must be 2D, but received {points.shape[1]}D coordinates."
        )

    eps = 1e-8

    # Check which points lie inside the rectangle.
    is_point_in_rect = (
        (rect[0] + eps < points[:, 0])
        & (points[:, 0] < rect[2] - eps)
        & (rect[1] + eps < points[:, 1])
        & (points[:, 1] < rect[3] - eps)
    )

    # Check which points lie on which boundary(s) of the rectangle.
    is_point_on_bound_left = (
        np.isclose(points[:, 0], rect[0], atol=eps)
        & (rect[1] - eps <= points[:, 1])
        & (points[:, 1] <= rect[3] + eps)
    )
    is_point_on_bound_right = (
        np.isclose(points[:, 0], rect[2], atol=eps)
        & (rect[1] - eps <= points[:, 1])
        & (points[:, 1] <= rect[3] + eps)
    )
    is_point_on_bound_lower = (
        np.isclose(points[:, 1], rect[1], atol=eps)
        & (rect[0] - eps <= points[:, 0])
        & (points[:, 0] <= rect[2] + eps)
    )
    is_point_on_bound_upper = (
        np.isclose(points[:, 1], rect[3], atol=eps)
        & (rect[0] - eps <= points[:, 0])
        & (points[:, 0] <= rect[2] + eps)
    )

    # Make sure to offset the points on the boundary slightly, so that they
    # are mirrored correctly.
    points_to_override = points.copy()
    points_to_override[is_point_on_bound_left, 0] += eps
    points_to_override[is_point_on_bound_right, 0] -= eps
    points_to_override[is_point_on_bound_lower, 1] += eps
    points_to_override[is_point_on_bound_upper, 1] -= eps

    # Only keep the points that lie inside or on the rectangle.
    is_point_in_or_on_rect = (
        is_point_in_rect
        | is_point_on_bound_left
        | is_point_on_bound_right
        | is_point_on_bound_lower
        | is_point_on_bound_upper
    )
    points_voronoi = points_to_override[is_point_in_or_on_rect]  # [P', 2]

    # Mirror the points in 4 directions.
    points_voronoi_left = points_voronoi.copy()
    points_voronoi_left[:, 0] = 2 * rect[0] - points_voronoi_left[:, 0]
    points_voronoi_right = points_voronoi.copy()
    points_voronoi_right[:, 0] = 2 * rect[2] - points_voronoi_right[:, 0]
    points_voronoi_lower = points_voronoi.copy()
    points_voronoi_lower[:, 1] = 2 * rect[1] - points_voronoi_lower[:, 1]
    points_voronoi_upper = points_voronoi.copy()
    points_voronoi_upper[:, 1] = 2 * rect[3] - points_voronoi_upper[:, 1]
    points_voronoi_all = np.concat(
        (
            points_voronoi,
            points_voronoi_left,
            points_voronoi_right,
            points_voronoi_lower,
            points_voronoi_upper,
        ),
        axis=0,
    )  # [5P', 2]

    # Compute the Voronoi diagram.
    vor = Voronoi(points_voronoi_all)

    # Filter the outputs so that only relevant parts remain.
    points = points[is_point_in_or_on_rect]  # [P', 2]

    regions = [[]]  # P' + 1
    vertex_idcs_old2new = np.full(len(vor.vertices), -1)
    vertex_idcs = []
    for pr in vor.point_region[: len(points)]:
        region = vor.regions[pr]
        for i, vertex_idx_old in enumerate(region):
            if vertex_idx_old == -1:
                raise RuntimeError(
                    "One of the center regions of the Voronoi diagram is"
                    " infinite, which should never happen. Please report this"
                    " bug to the developer."
                )

            vertex_idx_new = vertex_idcs_old2new[vertex_idx_old]
            if vertex_idx_new == -1:
                # We have not yet seen this vertex.
                vertex_idx_new = len(vertex_idcs)
                vertex_idcs_old2new[vertex_idx_old] = vertex_idx_new
                vertex_idcs.append(vertex_idx_old)

            # Update the region list in-place for efficiency.
            region[i] = vertex_idx_new

        regions.append(region)

    point_region = np.arange(1, len(points) + 1)  # [P']
    vertices = vor.vertices[vertex_idcs]  # [V + X, 2]

    # If a ridge contains one of the points with index 0 <= i < P', then it
    # should be kept. Otherwise, it should be discarded.
    is_ridge_point_selected = vor.ridge_points < len(points)
    is_ridge_selected = np.any(is_ridge_point_selected, axis=1)

    # If a ridge contains one of the points that won't be kept, then it should
    # be set to -1.
    ridge_points = vor.ridge_points
    ridge_points[~is_ridge_point_selected] = -1
    ridge_points = ridge_points[is_ridge_selected]  # [R + X, 2]

    # The ridge vertices need to be converted to the new vertex indices.
    ridge_vertices = np.array(vor.ridge_vertices)[is_ridge_selected]  # R + X
    ridge_vertices = cast(
        list[list[int]], vertex_idcs_old2new[ridge_vertices].tolist()
    )

    return (
        points,
        vertices,
        ridge_points,
        ridge_vertices,
        point_region,
        regions,
    )
