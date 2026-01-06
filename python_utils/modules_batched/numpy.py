from __future__ import annotations

import warnings
from typing import Any, Literal, cast, overload

import numpy as np
import numpy.typing as npt

from ..modules.numpy import (
    NpGeneric,
    NpGeneric1,
    NpGeneric2,
    NpInteger,
    NpNumber,
    apply_mask,
    counts_segments,
    lexsort,
    pack_padded,
    pack_padded_multidim,
    pad_packed,
    pad_packed_multidim,
    replace_padding,
    replace_padding_multidim,
)

# ################################### MATHS ####################################


def sum_batched(
    values: npt.NDArray[NpNumber],
    L_bs: npt.NDArray[np.integer],
    is_padding_zero: bool = False,
) -> npt.NDArray[NpNumber]:
    """Calculate the sum for each sample in the batch.

    Args:
        values: The values to calculate the sum for. Padded with zeros if
            is_padding_zero is True. Otherwise, padding could be arbitrary.
            Shape: [B, max(L_bs), *]
        L_bs: The number of valid values in each sample.
            Shape: [B]
        is_padding_zero: Whether the values are padded with zeros already.
            Setting this to True when the values are already padded with zeros
            can speed up the calculation.

    Returns:
        The sum value for each sample.
            Shape: [B, *]

    Examples:
    >>> values = np.array([
    ...     [1, 2, 3, -1],
    ...     [5, -1, -1, -1],
    ...     [-1, -1, -1, -1],
    ...     [13, 14, 15, 16],
    ... ])
    >>> L_bs = np.array([3, 1, 0, 4])
    >>> sum_batched(values, L_bs)
    array([ 6,  5,  0, 58])
    """
    if not is_padding_zero:
        values = replace_padding(values, L_bs)

    return values.sum(axis=1)  # [B, *]


def sum_batched_packed(
    values: npt.NDArray[NpNumber], L_bs: npt.NDArray[np.integer]
) -> npt.NDArray[NpNumber]:
    """Calculate the sum for each sample in the batch.

    Args:
        values: The values to calculate the sum for.
            Shape: [sum(L_bs), *]
        L_bs: The number of valid values in each sample.
            Shape: [B]

    Returns:
        The sum value for each sample.
            Shape: [B, *]

    Examples:
    >>> values = np.array([1, 2, 3, 5, 13, 14, 15, 16])
    >>> L_bs = np.array([3, 1, 0, 4])
    >>> sum_batched_packed(values, L_bs)
    array([ 6,  5,  0, 58])
    """
    start_idcs = L_bs.cumsum() - L_bs  # [B]
    values_summed = np.add.reduceat(values, start_idcs, axis=0)  # [B, *]

    # Handle empty segments (where L_b == 0). reduceat() returns values[idx] for
    # empty segments, but we want 0.
    values_summed[L_bs == 0] = 0

    return values_summed


def mean_batched(
    values: npt.NDArray[np.number],
    L_bs: npt.NDArray[np.integer],
    is_padding_zero: bool = False,
) -> npt.NDArray[np.floating]:
    """Calculate the mean for each sample in the batch.

    Args:
        values: The values to calculate the mean for. Padded with zeros if
            is_padding_zero is True. Otherwise, padding could be arbitrary.
            Shape: [B, max(L_bs), *]
        L_bs: The number of valid values in each sample.
            Shape: [B]
        is_padding_zero: Whether the values are padded with zeros already.
            Setting this to True when the values are already padded with zeros
            can speed up the calculation.

    Returns:
        The mean value for each sample.
            Shape: [B, *]

    Examples:
    >>> values = np.array([
    ...     [1, 2, 3, -1],
    ...     [5, -1, -1, -1],
    ...     [-1, -1, -1, -1],
    ...     [13, 14, 15, 16],
    ... ])
    >>> L_bs = np.array([3, 1, 0, 4])
    >>> with warnings.catch_warnings():
    ...     warnings.filterwarnings("ignore", category=RuntimeWarning)
    ...     mean_batched(values, L_bs)
    array([ 2. ,  5. ,  nan, 14.5])
    """
    return sum_batched(
        values, L_bs, is_padding_zero=is_padding_zero
    ) / L_bs.reshape(
        -1, *[1] * (values.ndim - 2)
    )  # [B, *]  # type: ignore


def mean_batched_packed(
    values: npt.NDArray[np.number], L_bs: npt.NDArray[np.integer]
) -> npt.NDArray[np.floating]:
    """Calculate the mean for each sample in the batch.

    Args:
        values: The values to calculate the mean for.
            Shape: [sum(L_bs), *]
        L_bs: The number of valid values in each sample.
            Shape: [B]

    Returns:
        The mean value for each sample.
            Shape: [B, *]

    Examples:
    >>> values = np.array([1, 2, 3, 5, 13, 14, 15, 16])
    >>> L_bs = np.array([3, 1, 0, 4])
    >>> with warnings.catch_warnings():
    ...     warnings.filterwarnings("ignore", category=RuntimeWarning)
    ...     mean_batched_packed(values, L_bs)
    array([ 2. ,  5. ,  nan, 14.5])
    """
    return sum_batched_packed(values, L_bs) / L_bs.reshape(
        -1, *[1] * (values.ndim - 2)
    )  # [B, *]  # type: ignore


def stddev_batched(
    values: npt.NDArray[np.number],
    L_bs: npt.NDArray[np.integer],
    is_padding_zero: bool = False,
) -> npt.NDArray[np.floating]:
    """Calculate the standard dev. for each sample in the batch.

    For a set of values x_1, ..., x_n, the standard deviation is defined as:
        sqrt(sum((x_i - mean(x))^2) / n)

    Args:
        values: The values to calculate the standard deviation for. Padded with
            zeros if is_padding_zero is True. Otherwise, padding could be
            arbitrary.
            Shape: [B, max(L_bs), *]
        L_bs: The number of valid values in each sample.
            Shape: [B]
        is_padding_zero: Whether the values are padded with zeros already.
            Setting this to True when the values are already padded with zeros
            can speed up the calculation.

    Returns:
        The standard deviation for each sample.
            Shape: [B, *]

    Examples:
    >>> values = np.array([
    ...     [1, 2, 3, -1],
    ...     [5, -1, -1, -1],
    ...     [-1, -1, -1, -1],
    ...     [13, 14, 15, 16],
    ... ])
    >>> L_bs = np.array([3, 1, 0, 4])
    >>> with warnings.catch_warnings():
    ...     warnings.filterwarnings("ignore", category=RuntimeWarning)
    ...     stddev_batched(values, L_bs)
    array([0.81649658, 0.        ,        nan, 1.11803399])
    """
    means = mean_batched(
        values, L_bs, is_padding_zero=is_padding_zero
    )  # [B, *]
    values_centered = values - np.expand_dims(means, 1)  # [B, max(L_bs), *]
    return np.sqrt(mean_batched(np.square(values_centered), L_bs))  # [B, *]


def stddev_batched_packed(
    values: npt.NDArray[np.number], L_bs: npt.NDArray[np.integer]
) -> npt.NDArray[np.floating]:
    """Calculate the standard dev. for each sample in the batch.

    For a set of values x_1, ..., x_n, the standard deviation is defined as:
        sqrt(sum((x_i - mean(x))^2) / n)

    Args:
        values: The values to calculate the standard deviation for.
            Shape: [sum(L_bs), *]
        L_bs: The number of valid values in each sample.
            Shape: [B]

    Returns:
        The standard deviation for each sample.
            Shape: [B, *]

    Examples:
    >>> values = np.array([1, 2, 3, 5, 13, 14, 15, 16])
    >>> L_bs = np.array([3, 1, 0, 4])
    >>> with warnings.catch_warnings():
    ...     warnings.filterwarnings("ignore", category=RuntimeWarning)
    ...     stddev_batched_packed(values, L_bs)
    array([0.81649658, 0.        ,        nan, 1.11803399])
    """
    means = mean_batched_packed(values, L_bs)  # [B, *]
    means_repeated = means.repeat(L_bs, axis=0)  # [sum(L_bs), *]
    values_centered = values - means_repeated  # [sum(L_bs), *]
    return np.sqrt(
        mean_batched_packed(np.square(values_centered), L_bs)
    )  # [B, *]


def min_batched(
    values: npt.NDArray[np.number],
    L_bs: npt.NDArray[np.integer],
    is_padding_inf: bool = False,
) -> npt.NDArray[np.float64]:
    """Calculate the minimum for each sample in the batch.

    Args:
        values: The values to calculate the minimum for. Padded with inf values
            if is_padding_inf is True. Otherwise, padding could be arbitrary.
            Shape: [B, max(L_bs), *]
        L_bs: The number of valid values in each sample.
            Shape: [B]
        is_padding_inf: Whether the values are padded with inf values already.
            Setting this to True when the values are already padded with inf
            values can speed up the calculation.

    Returns:
        The minimum value for each sample.
            Shape: [B, *]

    Examples:
    >>> values = np.array(
    ...     [
    ...         [1, 2, 3, -1],
    ...         [5, -1, -1, -1],
    ...         [-1, -1, -1, -1],
    ...         [13, 14, 15, 16],
    ...     ],
    ...     dtype=np.float64,
    ... )
    >>> L_bs = np.array([3, 1, 0, 4])
    >>> min_batched(values, L_bs)
    array([ 1.,  5., inf, 13.])
    """
    if not np.issubdtype(values.dtype, np.floating):
        warnings.warn(
            "Converting input values to float for proper handling of inf"
            " padding values.",
            RuntimeWarning,
        )
        values = values.astype(np.float64)

    if not is_padding_inf:
        values = replace_padding(values, L_bs, padding_value=float("inf"))

    return np.amin(values, axis=1)  # [B, *]


def min_batched_packed(
    values: npt.NDArray[np.number], L_bs: npt.NDArray[np.integer]
) -> npt.NDArray[np.float64]:
    """Calculate the minimum for each sample in the batch.

    Args:
        values: The values to calculate the minimum for.
            Shape: [sum(L_bs), *]
        L_bs: The number of valid values in each sample.
            Shape: [B]

    Returns:
        The minimum value for each sample.
            Shape: [B, *]

    Examples:
    >>> values = np.array([1, 2, 3, 5, 13, 14, 15, 16], dtype=np.float64)
    >>> L_bs = np.array([3, 1, 0, 4])
    >>> min_batched_packed(values, L_bs)
    array([ 1.,  5., inf, 13.])
    """
    if not np.issubdtype(values.dtype, np.floating):
        warnings.warn(
            "Converting input values to float for proper handling of inf"
            " padding values.",
            RuntimeWarning,
        )
        values = values.astype(np.float64)

    start_idcs = L_bs.cumsum() - L_bs  # [B]
    values_minimized = np.minimum.reduceat(values, start_idcs, axis=0)  # [B, *]

    # Handle empty segments (where L_b == 0). reduceat() returns values[idx] for
    # empty segments, but we want inf.
    values_minimized[L_bs == 0] = float("inf")

    return values_minimized


def max_batched(
    values: npt.NDArray[np.number],
    L_bs: npt.NDArray[np.integer],
    is_padding_minus_inf: bool = False,
) -> npt.NDArray[np.float64]:
    """Calculate the maximum for each sample in the batch.

    Args:
        values: The values to calculate the maximum for. Padded with -inf
            values if is_padding_minus_inf is True. Otherwise, padding could be
            arbitrary.
            Shape: [B, max(L_bs), *]
        L_bs: The number of valid values in each sample.
            Shape: [B]
        is_padding_minus_inf: Whether the values are padded with -inf values
            already. Setting this to True when the values are already padded
            with -inf values can speed up the calculation.

    Returns:
        The maximum value for each sample.
            Shape: [B, *]

    Examples:
    >>> values = np.array(
    ...     [
    ...         [1, 2, 3, -1],
    ...         [5, -1, -1, -1],
    ...         [-1, -1, -1, -1],
    ...         [13, 14, 15, 16],
    ...     ],
    ...     dtype=np.float64,
    ... )
    >>> L_bs = np.array([3, 1, 0, 4])
    >>> max_batched(values, L_bs)
    array([ 3.,  5., -inf, 16.])
    """
    if not np.issubdtype(values.dtype, np.floating):
        warnings.warn(
            "Converting input values to float for proper handling of inf"
            " padding values.",
            RuntimeWarning,
        )
        values = values.astype(np.float64)

    if not is_padding_minus_inf:
        values = replace_padding(values, L_bs, padding_value=float("-inf"))

    return np.amax(values, axis=1)  # [B, *]


def max_batched_packed(
    values: npt.NDArray[np.number], L_bs: npt.NDArray[np.integer]
) -> npt.NDArray[np.float64]:
    """Calculate the maximum for each sample in the batch.

    Args:
        values: The values to calculate the maximum for.
            Shape: [sum(L_bs), *]
        L_bs: The number of valid values in each sample.
            Shape: [B]

    Returns:
        The maximum value for each sample.
            Shape: [B, *]

    Examples:
    >>> values = np.array([1, 2, 3, 5, 13, 14, 15, 16], dtype=np.float64)
    >>> L_bs = np.array([3, 1, 0, 4])
    >>> max_batched_packed(values, L_bs)
    array([ 3.,  5., -inf, 16.])
    """
    if not np.issubdtype(values.dtype, np.floating):
        warnings.warn(
            "Converting input values to float for proper handling of inf"
            " padding values.",
            RuntimeWarning,
        )
        values = values.astype(np.float64)

    start_idcs = L_bs.cumsum() - L_bs  # [B]
    values_maximized = np.maximum.reduceat(values, start_idcs, axis=0)  # [B, *]

    # Handle empty segments (where L_b == 0). reduceat() returns values[idx] for
    # empty segments, but we want -inf.
    values_maximized[L_bs == 0] = float("-inf")

    return values_maximized


def any_batched(
    values: npt.NDArray[np.bool_],
    L_bs: npt.NDArray[np.integer],
    is_padding_false: bool = False,
) -> npt.NDArray[np.bool_]:
    """Determine whether any value is True for each sample in the batch.

    Args:
        values: The values to check. Padded with False values if
            is_padding_false is True. Otherwise, padding could be arbitrary.
            Shape: [B, max(L_bs), *]
        L_bs: The number of valid values in each sample.
            Shape: [B]
        is_padding_false: Whether the values are padded with False values
            already. Setting this to True when the values are already padded
            with False values can speed up the calculation.

    Returns:
        Whether any value is True for each sample.
            Shape: [B, *]

    Examples:
    >>> values = np.array([
    ...     [False, False, True, True],
    ...     [False, True, True, True],
    ...     [True, True, True, True],
    ...     [True, True, True, True],
    ... ])
    >>> L_bs = np.array([3, 1, 0, 4])
    >>> any_batched(values, L_bs)
    array([ True, False, False,  True])
    """
    if not is_padding_false:
        values = replace_padding(values, L_bs, padding_value=False)

    return values.any(axis=1)  # [B, *]  # type: ignore


def any_batched_packed(
    values: npt.NDArray[np.bool_], L_bs: npt.NDArray[np.integer]
) -> npt.NDArray[np.bool_]:
    """Determine whether any value is True for each sample in the batch.

    Args:
        values: The values to check.
            Shape: [sum(L_bs)]
        L_bs: The number of valid values in each sample.
            Shape: [B]

    Returns:
        Whether any value is True for each sample.
            Shape: [B]

    Examples:
    >>> values = np.array([False, False, True, False, True, True, True, True])
    >>> L_bs = np.array([3, 1, 0, 4])
    >>> any_batched_packed(values, L_bs)
    array([ True, False, False,  True])
    """
    start_idcs = L_bs.cumsum() - L_bs  # [B]
    values_anyd = np.logical_or.reduceat(values, start_idcs, axis=0)  # [B]

    # Handle empty segments (where L_b == 0). reduceat() returns values[idx] for
    # empty segments, but we want False.
    values_anyd[L_bs == 0] = False

    return values_anyd


def all_batched(
    values: npt.NDArray[np.bool_],
    L_bs: npt.NDArray[np.integer],
    is_padding_true: bool = False,
) -> npt.NDArray[np.bool_]:
    """Determine whether all values are True for each sample in the batch.

    Args:
        values: The values to check. Padded with True values if
            is_padding_true is True. Otherwise, padding could be arbitrary.
            Shape: [B, max(L_bs), *]
        L_bs: The number of valid values in each sample.
            Shape: [B]
        is_padding_true: Whether the values are padded with True values
            already. Setting this to True when the values are already padded
            with True values can speed up the calculation.

    Returns:
        Whether all values are True for each sample.
            Shape: [B, *]

    Examples:
    >>> values = np.array([
    ...     [True, True, True, False],
    ...     [True, False, False, False],
    ...     [False, False, False, False],
    ...     [False, True, True, True],
    ... ])
    >>> L_bs = np.array([3, 1, 0, 4])
    >>> all_batched(values, L_bs)
    array([ True,  True,  True, False])
    """
    if not is_padding_true:
        values = replace_padding(values, L_bs, padding_value=True)

    return values.all(axis=1)  # [B, *]  # type: ignore


def all_batched_packed(
    values: npt.NDArray[np.bool_], L_bs: npt.NDArray[np.integer]
) -> npt.NDArray[np.bool_]:
    """Determine whether all values are True for each sample in the batch.

    Args:
        values: The values to check.
            Shape: [sum(L_bs)]
        L_bs: The number of valid values in each sample.
            Shape: [B]

    Returns:
        Whether all values are True for each sample.
            Shape: [B]

    Examples:
    >>> values = np.array([True, True, True, True, False, True, True, True])
    >>> L_bs = np.array([3, 1, 0, 4])
    >>> all_batched_packed(values, L_bs)
    array([ True,  True,  True, False])
    """
    start_idcs = L_bs.cumsum() - L_bs  # [B]
    values_alld = np.logical_and.reduceat(values, start_idcs, axis=0)  # [B]

    # Handle empty segments (where L_b == 0). reduceat() returns values[idx] for
    # empty segments, but we want True.
    values_alld[L_bs == 0] = True

    return values_alld


def interp_batched(
    x: npt.NDArray[np.number],
    xp: npt.NDArray[np.number],
    fp: npt.NDArray[np.number],
    left: npt.NDArray[np.number] | None = None,
    right: npt.NDArray[np.number] | None = None,
    period: npt.NDArray[np.number] | None = None,
) -> npt.NDArray[np.number]:
    """Perform linear interpolation for a batch of 1D arrays.

    Warning: This function internally uses a for-loop over the batch dimension.
    This is because unlike torch.searchsorted(), np.searchsorted() does not
    support batched inputs, and there is no other low-level numpy function to
    vectorize this operation.

    Args:
        x: The x-coordinates at which to evaluate the interpolated values.
            Shape: [B, N]
        xp: The x-coordinates of the data points. Must be weakly monotonically
            increasing along the last dimension.
            Shape: [B, M]
        fp: The y-coordinates of the data points, same shape as xp.
            Shape: [B, M]
        left: Value to return for x < xp[0], default is fp[:, 0].
            Shape: [B]
        right: Value to return for x > xp[-1], default is fp[:, -1].
            Shape: [B]
        period: A period for the x-coordinates. This parameter allows the
            proper interpolation of angular x-coordinates. Parameters left and
            right are ignored if period is specified.
            Shape: [B]

    Returns:
        The interpolated values for each batch.
            Shape: [B, N]

    Examples:
    >>> x = np.array([
    ...     [10.5, 200.0, 40.0, 56.0],
    ...     [1.5, 2.5, 10.0, -1.0],
    ... ])
    >>> xp = np.array([
    ...     [0.0, 1.0, 20.0, 100.0],
    ...     [0.0, 1.0, 2.0, 3.0],
    ... ])
    >>> fp = np.array([
    ...     [0.0, 100.0, 200.0, 300.0],
    ...     [0.0, 10.0, 20.0, 30.0],
    ... ])
    >>> interp_batched(x, xp, fp)
    array([[150., 300., 225., 245.],
           [ 15.,  25.,  30.,   0.]])
    """
    # Handle periodic interpolation.
    if period is not None:
        if (period <= 0).any():
            raise ValueError("period must be positive.")

        # Normalize x and xp to [0, period).
        x %= np.expand_dims(period, 1)  # [B, N]  # type: ignore
        xp %= np.expand_dims(period, 1)  # [B, M]  # type: ignore

        # Re-sort xp and fp after the modulo operation.
        sorted_idcs = xp.argsort(axis=1)  # [B, M]
        xp = np.take_along_axis(xp, sorted_idcs, 1)  # [B, M]
        fp = np.take_along_axis(fp, sorted_idcs, 1)  # [B, M]

        # Extend xp and fp arrays to handle wrap-around interpolation. Add the
        # last point before the first, and the first point after the last.
        xp = np.concat(
            [
                np.expand_dims(xp[:, -1] - period, 1),
                xp,
                np.expand_dims(xp[:, 0] + period, 1),
            ],
            axis=1,
        )  # [B, M + 2]
        fp = np.concat(
            [np.expand_dims(fp[:, -1], 1), fp, np.expand_dims(fp[:, 0], 1)],
            axis=1,
        )  # [B, M + 2]

    B, M = xp.shape

    # Check if xp is weakly monotonically increasing.
    if not (np.diff(xp, axis=1) >= 0).all():
        raise ValueError(
            "xp must be weakly monotonically increasing along the last"
            " dimension."
        )

    # Find indices of neighbours in xp.
    right_idx = np.stack(
        [np.searchsorted(xp[b], x[b]) for b in range(B)]
    )  # [B, N]
    left_idx = right_idx - 1  # [B, N]

    # Clamp indices to valid range (we will handle the edges later).
    left_idx = left_idx.clip(min=0, max=M - 1)  # [B, N]
    right_idx = right_idx.clip(min=0, max=M - 1)  # [B, N]

    # Gather neighbour values.
    x_left = np.take_along_axis(xp, left_idx, 1)  # [B, N]
    x_right = np.take_along_axis(xp, right_idx, 1)  # [B, N]
    y_left = np.take_along_axis(fp, left_idx, 1)  # [B, N]
    y_right = np.take_along_axis(fp, right_idx, 1)  # [B, N]

    # Avoid division by zero for x_left == x_right.
    denom = x_right - x_left  # [B, N]
    denom[denom == 0] = 1
    p = (x - x_left) / denom  # [B, N]

    # Perform interpolation.
    y = y_left + p * (y_right - y_left)  # [B, N]

    # Handle edges only if period is not specified.
    if period is None:
        # Handle left edge.
        if left is None:
            left = fp[:, 0]  # [B]
        is_left = x < xp[:, [0]]  # [B, N]
        y[is_left] = left.repeat(is_left.sum(axis=1)).astype(y.dtype)

        # Handle right edge.
        if right is None:
            right = fp[:, -1]  # [B]
        is_right = x > xp[:, [-1]]  # [B, N]
        y[is_right] = right.repeat(is_right.sum(axis=1)).astype(y.dtype)

    return y


# ################################### RANDOM ###################################


def sample_unique_batched(
    L_bs: npt.NDArray[np.integer], max_L_bs: int, padding_value: Any = None
) -> npt.NDArray[np.int64]:
    """Sample unique indices i in [0, L_b) for each element in the batch.

    Args:
        L_bs: The number of valid values for each element in the batch.
            Shape: [B]
        max_L_bs: The maximum number of values of any element in the batch.
        padding_value: The value to use for padding the output indices. If None,
            the output indices are padded with random values. This is faster
            than padding with a specific value.

    Returns:
        The sampled unique indices. Padded with padding_value.
            Shape: [B, max(L_bs)]

    Examples:
    >>> L_bs = np.array([5, 3, 0, 4])
    >>> max_L_bs = 5
    >>> unique_idcs = sample_unique_batched(L_bs, max_L_bs, padding_value=0)
    >>> unique_idcs  # doctest: +SKIP
    array([[4, 2, 3, 1, 0],
           [2, 0, 1, 0, 0],
           [0, 0, 0, 0, 0],
           [1, 3, 2, 0, 0]])
    """
    B = len(L_bs)
    rng = np.random.default_rng()
    idcs = np.broadcast_to(
        np.expand_dims(np.arange(max_L_bs), 0), (B, max_L_bs)
    )  # [B, max(L_bs)]
    permuted_idcs = rng.permuted(idcs, axis=1)  # [B, max(L_bs)]
    mask = permuted_idcs < np.expand_dims(L_bs, 1)  # [B, max(L_bs)]
    random_idcs, _ = apply_mask(
        permuted_idcs,
        mask,
        np.full((B,), max_L_bs),
        padding_value=padding_value,
    )  # [B, max(L_bs)]
    return random_idcs


def sample_unique_pairs_batched(
    L_bs: npt.NDArray[np.integer], max_L_bs: int, padding_value: Any = None
) -> npt.NDArray[np.integer]:
    """Sample unique pairs of indices (i, j), where i and j are in [0, L_b).

    Note: The order of the indices in each pair is deemed irrelevant, so (i, j)
    is considered the same as (j, i). Therefore, there are a total of
    L_b * (L_b - 1) / 2 unique pairs for each element in the batch. For
    consistency, we ensure that i > j.

    Args:
        L_bs: The number of valid values for each element in the batch.
            Shape: [B]
        max_L_bs: The maximum number of valid values.
        padding_value: The value to use for padding the output indices. If None,
            the output indices are padded with random values. This is faster
            than padding with a specific value.

    Returns:
        The sampled unique pairs of indices. Padded with padding_value.
            Shape: [B, max(P_bs), 2]

    Examples:
    >>> L_bs = np.array([3, 1, 0, 4])
    >>> max_L_bs = 4
    >>> unique_pairs = sample_unique_pairs_batched(
    ...     L_bs, max_L_bs, padding_value=0
    ... )
    >>> unique_pairs  # doctest: +SKIP
    array([[[1, 0],
            [2, 1],
            [2, 0],
            [0, 0],
            [0, 0],
            [0, 0]],
           [[0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0]],
           [[0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0]],
           [[2, 0],
            [3, 1],
            [1, 0],
            [2, 1],
            [3, 0],
            [3, 2]]])
    """
    # Compute the number of unique pairs of indices.
    P_bs = L_bs * (L_bs - 1) // 2  # [B]
    max_P_bs = max_L_bs * (max_L_bs - 1) // 2

    # Select unique pairs of elements for each sample in the batch.
    idcs_pairs = sample_unique_batched(
        P_bs, max_P_bs, padding_value=0
    )  # [B, max(P_bs)]

    # Convert the pair indices to element indices.
    # np.tril_indices(max_L_bs, k=-1) returns the indices as follows:
    # i\j 0 1 2 3 4 ...
    #  0  x x x x x
    #  1  0 x x x x
    #  2  1 2 x x x
    #  3  3 4 5 x x
    #  4  6 7 8 9 x
    # ...
    tril_idcs = np.stack(np.tril_indices(max_L_bs, k=-1))  # [2, max(P_bs)]
    idcs_elements = tril_idcs[:, idcs_pairs]  # [2, B, max(P_bs)]
    idcs_elements = np.moveaxis(idcs_elements, 0, 2)  # [B, max(P_bs), 2]

    # Apply padding if requested.
    if padding_value is not None:
        replace_padding(
            idcs_elements, P_bs, padding_value=padding_value, in_place=True
        )

    return idcs_elements


# ########################## BASIC ARRAY MANIPULATION ##########################


def arange_batched(
    starts: npt.NDArray[np.number],
    stops: npt.NDArray[np.number] | None = None,
    steps: npt.NDArray[np.number] | None = None,
    padding_value: Any = None,
    dtype: np.dtype | None = None,
) -> tuple[npt.NDArray[np.number], npt.NDArray[np.intp]]:
    """Create a batch of arrays with values in the range [start, stop).

    Args:
        starts: The start value for each array in the batch. If stops is None,
            the range is [0, start).
            Shape: [B]
        stops: The end value for each array in the batch. The interval is
            half-open, so this end value is not included.
            Shape: [B]
        steps: The step value for each array in the batch. If None, the step
            value is set to 1.
            Shape: [B]
        padding_value: The value to pad the values with. If None, the values
            are padded with random values. This is faster than padding with
            a specific value.
        dtype: The data type of the output array.

    Returns:
        Tuple containing:
        - A batch of arrays with values in the range [start, stop).
            Padded with padding_value.
            Shape: [B, max(L_bs)]
        - The number of values of the arange sequences in the batch.
            Shape: [B]

    Examples:
    >>> starts = np.array([0, 5, 2, 3])
    >>> stops = np.array([3, 5, 8, -1])
    >>> steps = np.array([1, 1, 3, -1])
    >>> aranges, L_bs = arange_batched(starts, stops, steps, padding_value=-1)
    >>> aranges
    array([[ 0,  1,  2, -1],
           [-1, -1, -1, -1],
           [ 2,  5, -1, -1],
           [ 3,  2,  1,  0]])
    >>> L_bs
    array([3, 0, 2, 4])
    """
    B = len(starts)
    inferred_dtype = np.promote_types(
        starts.dtype,
        np.promote_types(
            stops.dtype if stops is not None else starts.dtype,
            steps.dtype if steps is not None else starts.dtype,
        ),
    )

    # Prepare the input arrays.
    if stops is None:
        stops = starts
        starts = np.zeros(B, dtype=inferred_dtype)
    if steps is None:
        steps = np.ones(B, dtype=inferred_dtype)

    # Compute the arange sequences in parallel.
    L_bs = np.ceil((stops - starts) / steps).astype(np.intp)  # [B]
    max_L_bs = int(L_bs.max())
    aranges = (
        np.expand_dims(starts, 1)  # [B, 1]
        + np.arange(max_L_bs, dtype=inferred_dtype)  # [max(L_bs)]
        * np.expand_dims(steps, 1)  # [B, 1]
    )  # [B, max(L_bs)]  # fmt: skip

    # Replace values that are out of bounds with the padding value.
    if padding_value is not None:
        replace_padding(
            aranges, L_bs, padding_value=padding_value, in_place=True
        )

    # Cast to the desired dtype.
    if dtype is not None:
        aranges = aranges.astype(dtype)

    return aranges, L_bs


def arange_batched_packed(
    starts: npt.NDArray[np.number],
    stops: npt.NDArray[np.number] | None = None,
    steps: npt.NDArray[np.number] | None = None,
    dtype: np.dtype | None = None,
) -> tuple[npt.NDArray[np.number], npt.NDArray[np.intp], int]:
    """Create a batch of arrays with values in the range [start, stop).

    Args:
        starts: The start value for each array in the batch. If stops is None,
            the range is [0, start).
            Shape: [B]
        stops: The end value for each array in the batch. The interval is
            half-open, so this end value is not included.
            Shape: [B]
        steps: The step value for each array in the batch. If None, the step
            value is set to 1.
            Shape: [B]
        dtype: The data type of the output array.

    Returns:
        Tuple containing:
        - A batch of arrays with values in the range [start, stop).
            Shape: [L]
        - The number of values of the arange sequences in the batch.
            Shape: [B]
        - The maximum length of the arange sequences in the batch.

    Examples:
    >>> starts = np.array([0, 5, 2, 3])
    >>> stops = np.array([3, 5, 8, -1])
    >>> steps = np.array([1, 1, 3, -1])
    >>> aranges, L_bs, max_L_bs = arange_batched_packed(starts, stops, steps)
    >>> aranges
    array([0, 1, 2, 2, 5, 3, 2, 1, 0])
    >>> L_bs
    array([3, 0, 2, 4])
    >>> max_L_bs
    4
    """
    B = len(starts)
    inferred_dtype = np.promote_types(
        starts.dtype,
        np.promote_types(
            stops.dtype if stops is not None else starts.dtype,
            steps.dtype if steps is not None else starts.dtype,
        ),
    )

    # Prepare the input arrays.
    if stops is None:
        stops = starts
        starts = np.zeros(B, dtype=inferred_dtype)
    if steps is None:
        steps = np.ones(B, dtype=inferred_dtype)

    # Compute the starts and steps of the arange sequences in parallel.
    L_bs = np.ceil((stops - starts) / steps).astype(np.intp)  # [B]
    max_L_bs = int(L_bs.max())
    starts_repeated = starts.repeat(L_bs)  # [L]
    steps_repeated = steps.repeat(L_bs)  # [L]

    # Compute the offsets for each arange sequence in parallel.
    L_bs_without_last = L_bs[:-1]  # [B - 1]
    transition_idcs = L_bs_without_last[L_bs_without_last != 0]  # [B']
    offsets_packed = np.ones_like(steps_repeated)  # [L]
    offsets_packed[0] = 0
    offsets_packed[transition_idcs.cumsum()] -= transition_idcs  # [B']
    offsets_packed = offsets_packed.cumsum()  # [L]

    # Compute the arange sequences in parallel.
    offsets_packed *= steps_repeated  # [L]
    aranges = starts_repeated + offsets_packed  # [L]

    # Cast to the desired dtype.
    if dtype is not None:
        aranges = aranges.astype(dtype)

    return aranges, L_bs, max_L_bs


def linspace_batched(
    starts: npt.NDArray[np.number],
    stops: npt.NDArray[np.number],
    nums: npt.NDArray[np.integer],
    padding_value: Any = None,
    dtype: np.dtype | None = None,
) -> tuple[npt.NDArray[np.number], npt.NDArray[np.intp]]:
    """Create a batch of arrays with values in the range [start, stop].

    Args:
        starts: The start value for each array in the batch.
            Shape: [B]
        stops: The end value for each array in the batch.
            Shape: [B]
        nums: The number of samples to generate for each array in the batch. If
            the number of samples is 1, only the start value is returned.
            Shape: [B]
        padding_value: The value to pad the values with. If None, the values
            are padded with random values. This is faster than padding with
            a specific value.
        dtype: The data type of the output array.

    Returns:
        Tuple containing:
        - A batch of arrays with values in the range [start, stop].
            Padded with padding_value.
            Shape: [B, max(L_bs)]
        - The number of values of the linspace sequences in the batch. This is
            the same as nums.
            Shape: [B]

    Examples:
    >>> starts = np.array([0.0, 5.0, 3.0, 2.0, 3.0])
    >>> stops = np.array([1.0, 6.0, 5.0, 8.0, -3.0])
    >>> nums = np.array([5, 1, 0, 3, 4])
    >>> linspaces, L_bs = linspace_batched(
    ...     starts, stops, nums, padding_value=-1.0
    ... )
    >>> linspaces
    array([[ 0.  ,  0.25,  0.5 ,  0.75,  1.  ],
           [ 5.  , -1.  , -1.  , -1.  , -1.  ],
           [-1.  , -1.  , -1.  , -1.  , -1.  ],
           [ 2.  ,  5.  ,  8.  , -1.  , -1.  ],
           [ 3.  ,  1.  , -1.  , -3.  , -1.  ]])
    >>> L_bs
    array([5, 1, 0, 3, 4])
    """
    inferred_dtype = np.promote_types(
        starts.dtype, np.promote_types(stops.dtype, nums.dtype)
    )

    # Compute the steps of the linspace sequences in parallel.
    L_bs = nums.astype(np.intp)  # [B]
    max_L_bs = int(L_bs.max())
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=RuntimeWarning
        )  # ignore division by zero since we already handle it in np.where()
        steps = np.where(L_bs != 1, (stops - starts) / (L_bs - 1), 0)  # [B]

    # Compute the linspace sequences in parallel.
    linspaces = (
        np.expand_dims(starts, 1)  # [B, 1]
        + np.arange(max_L_bs, dtype=inferred_dtype)  # [max(L_bs)]
        * np.expand_dims(steps, 1)  # [B, 1]
    )  # [B, max(L_bs)]  # fmt: skip

    # Set the last element of each linspace to the stop value manually to avoid
    # floating point issues.
    nonzero_idcs = np.nonzero(L_bs)[0]  # [B_nonzero]
    L_bs_nonzero = L_bs[nonzero_idcs]  # [B_nonzero]
    if len(nonzero_idcs) != 0:
        stop_idcs = L_bs_nonzero - 1  # [B_nonzero]

        # Only set the stop values for sequences with at least two elements.
        is_atleasttwo = L_bs_nonzero != 1  # [B_nonzero]
        atleasttwo_idcs = nonzero_idcs[is_atleasttwo]  # [B_atleasttwo]
        stop_idcs = stop_idcs[is_atleasttwo]  # [B_atleasttwo]

        linspaces[atleasttwo_idcs, stop_idcs] = stops[atleasttwo_idcs].astype(
            linspaces.dtype
        )

    # Replace values that are out of bounds with the padding value.
    if padding_value is not None:
        replace_padding(
            linspaces, L_bs, padding_value=padding_value, in_place=True
        )

    # Cast to the desired dtype.
    if dtype is not None:
        linspaces = linspaces.astype(dtype)

    return linspaces, L_bs


def linspace_batched_packed(
    starts: npt.NDArray[np.number],
    stops: npt.NDArray[np.number],
    nums: npt.NDArray[np.integer],
    dtype: np.dtype | None = None,
) -> tuple[npt.NDArray[np.number], npt.NDArray[np.intp], int]:
    """Create a batch of arrays with values in the range [start, stop].

    Args:
        starts: The start value for each array in the batch.
            Shape: [B]
        stops: The end value for each array in the batch.
            Shape: [B]
        nums: The number of samples to generate for each array in the batch. If
            the number of samples is 1, only the start value is returned.
            Shape: [B]
        dtype: The data type of the output array.

    Returns:
        Tuple containing:
        - A batch of arrays with values in the range [start, stop].
            Padded with padding_value.
            Shape: [B, max(L_bs)]
        - The number of values of the linspace sequences in the batch. This is
            the same as nums.
            Shape: [B]
        - The maximum length of the linspace sequences in the batch.

    Examples:
    >>> starts = np.array([0.0, 5.0, 3.0, 2.0, 3.0])
    >>> stops = np.array([1.0, 6.0, 5.0, 8.0, -3.0])
    >>> nums = np.array([5, 1, 0, 3, 4])
    >>> linspaces, L_bs, max_L_bs = linspace_batched_packed(starts, stops, nums)
    >>> linspaces
    array([ 0.  ,  0.25,  0.5 ,  0.75,  1.  ,  5.  ,  2.  ,  5.  ,  8.  ,
            3.  ,  1.  , -1.  , -3.  ])
    >>> L_bs
    array([5, 1, 0, 3, 4])
    >>> max_L_bs
    5
    """
    # Compute the steps of the linspace sequences in parallel.
    L_bs = nums.astype(np.intp)  # [B]
    max_L_bs = int(L_bs.max())
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=RuntimeWarning
        )  # ignore division by zero since we already handle it in np.where()
        steps = np.where(L_bs != 1, (stops - starts) / (L_bs - 1), 0)  # [B]

    # Compute the starts and steps of the linspace sequences in parallel.
    starts_repeated = starts.repeat(L_bs)  # [L]
    steps_repeated = steps.repeat(L_bs)  # [L]

    # Compute the offsets for each linspace sequence in parallel.
    L_bs_without_last = L_bs[:-1]  # [B - 1]
    transition_idcs = L_bs_without_last[L_bs_without_last != 0]  # [B']
    offsets_packed = np.ones_like(steps_repeated)  # [L]
    offsets_packed[0] = 0
    offsets_packed[transition_idcs.cumsum()] -= transition_idcs  # [B']
    offsets_packed = offsets_packed.cumsum()  # [L]

    # Compute the linspace sequences in parallel.
    offsets_packed *= steps_repeated  # [L]
    linspaces = starts_repeated + offsets_packed  # [L]

    # Set the last element of each linspace to the stop value manually to avoid
    # floating point issues.
    nonzero_idcs = np.nonzero(L_bs)[0]  # [B_nonzero]
    L_bs_nonzero = L_bs[nonzero_idcs]  # [B_nonzero]
    if len(nonzero_idcs) != 0:
        stop_idcs = L_bs_nonzero.cumsum() - 1  # [B_nonzero]

        # Only set the stop values for sequences with at least two elements.
        is_atleasttwo = L_bs_nonzero != 1  # [B_nonzero]
        atleasttwo_idcs = nonzero_idcs[is_atleasttwo]  # [B_atleasttwo]
        stop_idcs = stop_idcs[is_atleasttwo]  # [B_atleasttwo]

        linspaces[stop_idcs] = stops[atleasttwo_idcs].astype(linspaces.dtype)

    # Cast to the desired dtype.
    if dtype is not None:
        linspaces = linspaces.astype(dtype)

    return linspaces, L_bs, max_L_bs


def take_batched(
    values: npt.NDArray[NpGeneric], axis: int, indices: npt.NDArray[np.integer]
) -> npt.NDArray[NpGeneric]:
    """Select values from a batch of arrays using the given indices.

    Note that axis refers to the index of the dimension to sort along AFTER the
    batch dimension.

    Args:
        values: The values to select from.
            Shape: [B, N_0, ..., N_axis, ..., N_{D-1}]
        axis: The dimension to select along.
        indices: The indices to select.
            Shape: [B, N_select]

    Returns:
        The selected values.
            Shape: [B, N_0, ..., N_{axis-1}, N_select, N_{axis+1}, ..., N_{D-1}]

    Examples:
    >>> values = np.array([
    ...     [
    ...         [1, 2, 3],
    ...         [4, 5, 6],
    ...         [7, 8, 9],
    ...         [10, 11, 12],
    ...     ],
    ...     [
    ...         [13, 14, 15],
    ...         [16, 17, 18],
    ...         [19, 20, 21],
    ...         [22, 23, 24],
    ...     ],
    ... ])
    >>> indices = np.array([
    ...     [2, 0],
    ...     [1, 3],
    ... ])
    >>> selected_values = take_batched(values, 0, indices)
    >>> selected_values
    array([[[ 7,  8,  9],
            [ 1,  2,  3]],
           [[16, 17, 18],
            [22, 23, 24]]])
    """
    unsqueezed_shape = [1] * values.ndim
    unsqueezed_shape[0] = indices.shape[0]
    unsqueezed_shape[axis + 1] = indices.shape[1]
    indices_unsqueezed = indices.reshape(unsqueezed_shape)
    broadcasted_shape = list(values.shape)
    broadcasted_shape[axis + 1] = indices.shape[1]
    indices_broadcasted = np.broadcast_to(indices_unsqueezed, broadcasted_shape)
    return np.take_along_axis(values, indices_broadcasted, axis + 1)


def __duplicate_subarrays(
    values: npt.NDArray[NpGeneric],
    L_bs: npt.NDArray[np.integer],
    reps_bs: npt.NDArray[np.integer],
) -> tuple[npt.NDArray[NpGeneric], npt.NDArray[np.intp]]:
    """Duplicate subarrays according to the given reps.

    Args:
        values: Packed array with all subarrays concatenated.
            Shape: [sum(L_bs)]
        L_bs: Length of each subarray.
            Shape: [B]
        reps_bs: Number of times to duplicate each subarray.
            Shape: [B]

    Returns:
        Tuple containing:
        - Packed array with each subarray duplicated.
            Shape: [sum(L_bs * reps_bs)]
        - The lengths of each duplicated subarray.
            Shape: [B]

    Examples:
    >>> values = np.array([1, 2, 3, 4, 5, 6])
    >>> L_bs = np.array([2, 3, 1])
    >>> reps_bs = np.array([2, 0, 3])
    >>> values_duplicated, L_bs_duplicated = __duplicate_subarrays(
    ...     values, L_bs, reps_bs
    ... )
    >>> values_duplicated
    array([1, 2, 1, 2, 6, 6, 6])
    >>> L_bs_duplicated
    array([4, 0, 3])
    """
    # The key insight is to construct an index array that repeats the
    # appropriate indices for each subarray.

    # Compute the start and stop indices of each subarray.
    stops = L_bs.cumsum(dtype=np.intp)  # [B]
    starts = stops - L_bs  # [B]

    # Repeat the start and stop indices according to reps.
    starts = starts.repeat(reps_bs)  # [sum(reps_bs)]
    stops = stops.repeat(reps_bs)  # [sum(reps_bs)]

    # Create the index array for each subarray.
    idcs, _, _ = arange_batched_packed(starts, stops)  # [sum(L_bs * reps_bs)]

    # Gather the duplicated values.
    values_duplicated = values[idcs]  # [sum(L_bs * reps_bs)]  # type: ignore
    L_bs_duplicated = (L_bs * reps_bs).astype(np.intp)  # [B]

    return values_duplicated, L_bs_duplicated


def repeat_batched(
    values: npt.NDArray[NpGeneric],
    L_bsds: npt.NDArray[np.integer],
    repeats: npt.NDArray[np.integer],
    axis: int = 0,
    padding_value: Any = None,
) -> tuple[npt.NDArray[NpGeneric], npt.NDArray[np.intp]]:
    """Repeat values from a batch of arrays using the given repeats.

    Note that axis refers to the index of the dimension to sort along AFTER the
    batch dimension.

    Args:
        values: The values to repeat.
            Shape: [B, max(L_bs0), ..., max(L_bs{axis}), ..., max(L_bs{D-1})]
        L_bsds: Length of each sample along each dimension.
            Shape: [B, D]
        repeats: Number of times to repeat each value.
            Shape: [B, max(L_bs{axis})]
        axis: The dimension to repeat along.
        padding_value: The value to pad the values with. If None, the values
            are padded with random values. This is faster than padding with
            a specific value.

    Returns:
        Tuple containing:
        - The repeated values. Padded with padding_value.
            Shape: [B, max(L_bs0), ..., max(sum(repeats)), ..., max(L_bs{D-1})]
        - The lengths of each repeated sample along each dimension.
            Shape: [B, D]

    Examples:
    >>> values = np.array([
    ...     [
    ...         [1, 2],
    ...         [3, 4],
    ...         [5, 6],
    ...     ],
    ...     [
    ...         [7, 8],
    ...         [9, 10],
    ...         [11, 12],
    ...     ],
    ... ])
    >>> L_bsds = np.array([
    ...     [2, 2],
    ...     [3, 1],
    ... ])
    >>> repeats = np.array([
    ...     [1, 2, 0],
    ...     [3, 2, 1],
    ... ])
    >>> values_repeated, L_bsds_repeated = repeat_batched(
    ...     values, L_bsds, repeats, axis=0, padding_value=0
    ... )
    >>> values_repeated
    array([[[ 1,  2],
            [ 3,  4],
            [ 3,  4],
            [ 0,  0],
            [ 0,  0],
            [ 0,  0]],
           [[ 7,  0],
            [ 7,  0],
            [ 7,  0],
            [ 9,  0],
            [ 9,  0],
            [11,  0]]])
    >>> L_bsds_repeated
    array([[3, 2],
           [6, 1]])
    """
    max_L_bsds = np.array(values.shape[1:])  # [D]

    # Compute the new lengths after repeating.
    L_bsds_repeated = L_bsds.astype(np.intp, copy=True)  # [B, D]
    L_bsds_repeated[:, axis] = sum_batched(repeats, L_bsds[:, axis])
    max_L_bsds_repeated = max_L_bsds.copy()  # [D]
    max_L_bsds_repeated[axis] = int(L_bsds_repeated[:, axis].max())

    # Move axis to the front and merge it with the batch dimension.
    # This allows us to use np.repeat() directly.
    values = np.moveaxis(
        values, axis + 1, 1
    )  # [B, max(L_bs{axis}), max(L_bs0), ..., max(L_bs{D-1})]
    values = pack_padded(
        values, L_bsds[:, axis]
    )  # [sum(L_bs{axis}), max(L_bs0), ..., max(L_bs{D-1})]
    repeats = pack_padded(repeats, L_bsds[:, axis])  # [sum(L_bs{axis})]

    # Repeat the values.
    values_repeated = values.repeat(
        repeats, axis=0
    )  # [sum(L_bs{axis}_repeated), max(L_bs0), ..., max(L_bs{D-1})]

    # Un-merge the batch and axis dimensions and move axis back to its original
    # position.
    values_repeated = pad_packed(
        values_repeated,
        L_bsds_repeated[:, axis],
        int(max_L_bsds_repeated[axis]),
    )  # [B, max(L_bs{axis}_repeated), max(L_bs0), ..., max(L_bs{D-1})]
    values_repeated = np.moveaxis(
        values_repeated, 1, axis + 1
    )  # [B, max(L_bs0), ..., max(L_bs{axis}_repeated), ..., max(L_bs{D-1})]

    # Apply padding if requested.
    if padding_value is not None:
        replace_padding_multidim(
            values_repeated,
            L_bsds_repeated,
            padding_value=padding_value,
            in_place=True,
        )

    return values_repeated, L_bsds_repeated


def repeat_batched_packed(
    values: npt.NDArray[NpGeneric],
    L_bsds: npt.NDArray[np.integer],
    repeats_bs: npt.NDArray[np.integer],
    axis: int = 0,
) -> tuple[npt.NDArray[NpGeneric], npt.NDArray[np.intp]]:
    """Repeat values from a batch of packed arrays using the given repeats.

    Args:
        values: Packed array of all samples concatenated.
            Shape: [sum(L_bs0 * ... * L_bs{D-1})]
        L_bsds: Length of each sample along each dimension.
            Shape: [B, D]
        repeats_bs: Number of times to repeat each value.
            Shape: [B, max(L_bs{axis})]
        axis: The dimension to repeat along.

    Returns:
        Tuple containing:
        - Packed array with each sample repeated.
            Shape: [sum(L_bs0 * ... * L_bs{axis}_repeated * ... * L_bs{D-1})]
        - The lengths of each repeated sample along each dimension.
            Shape: [B, D]

    Examples:
    >>> values = np.array([1, 2, 3, 4, 7, 9, 11])
    >>> L_bsds = np.array([
    ...     [2, 2],
    ...     [3, 1],
    ... ])
    >>> repeats_bs = np.array([
    ...     [1, 2, 0],
    ...     [3, 2, 1],
    ... ])
    >>> values_repeated, L_bsds_repeated = repeat_batched_packed(
    ...     values, L_bsds, repeats_bs, axis=0
    ... )
    >>> values_repeated
    array([ 1,  2,  3,  4,  3,  4,  7,  7,  7,  9,  9, 11])
    >>> L_bsds_repeated
    array([[3, 2],
           [6, 1]])
    """
    # Compute the new lengths after repeating.
    L_bsds_repeated = L_bsds.astype(np.intp, copy=True)  # [B, D]
    L_bsds_repeated[:, axis] = sum_batched(repeats_bs, L_bsds[:, axis])

    # Calculate product of all lengths for all dimensions except axis.
    # We denote the product from dimension d0 (included) to d1 (excluded) as:
    # _bs_d0_to_d1 = _bs{d0} * _bs{d0+1} * ... * _bs{d1-2} * _bs{d1-1}
    L_bs_0_to_axisplus1 = L_bsds[:, : axis + 1].prod(axis=1)  # [B]
    L_bs_0_to_axis = L_bs_0_to_axisplus1 // L_bsds[:, axis]  # [B]
    L_bs_axisplus1_to_D = (
        L_bsds[:, axis:].prod(axis=1) // L_bsds[:, axis]
    )  # [B]

    # Pretend we are working with 1D subarrays, and construct the corresponding
    # L_bs and repeats_bs.
    L_bs = L_bs_axisplus1_to_D.repeat(
        L_bs_0_to_axisplus1
    )  # [sum(L_bs_0_to_axisplus1)]
    repeats_bs, _ = __duplicate_subarrays(
        pack_padded(repeats_bs, L_bsds[:, axis]),
        L_bsds[:, axis],
        L_bs_0_to_axis,
    )  # [sum(L_bs_0_to_axisplus1)], _

    # Duplicate the subarrays.
    values_repeated, _ = __duplicate_subarrays(
        values, L_bs, repeats_bs
    )  # [sum(L_bs0 * ... * L_bs{axis}_repeated * ... * L_bs{D-1})], _

    return values_repeated, L_bsds_repeated


def tile_batched(
    values: npt.NDArray[NpGeneric],
    L_bsds: npt.NDArray[np.integer],
    reps_bsds: npt.NDArray[np.integer],
    padding_value: Any = None,
) -> tuple[npt.NDArray[NpGeneric], npt.NDArray[np.intp]]:
    """Tile values from a batch of arrays using the given reps.

    Args:
        values: The values to tile.
            Shape: [B, max(L_bs0), ..., max(L_bs{D-1})]
        L_bsds: Length of each sample along each dimension.
            Shape: [B, D]
        reps_bsds: Number of times to tile each sample along each dimension.
            Shape: [B, D]
        padding_value: The value to pad the values with. If None, the values
            are padded with random values. This is faster than padding with
            a specific value.

    Returns:
        Tuple containing:
        - The tiled values. Padded with padding_value.
            Shape: [
                B, max(L_bs0 * reps_bs0), ..., max(L_bs{D-1} * reps_bs{D-1})
            ]
        - The lengths of each tiled sample along each dimension.
            Shape: [B, D]

    Examples:
    >>> values = np.array([
    ...     [
    ...         [1, 2],
    ...         [3, 4],
    ...         [5, 6],
    ...     ],
    ...     [
    ...         [7, 8],
    ...         [9, 10],
    ...         [11, 12],
    ...     ],
    ... ])
    >>> L_bsds = np.array([
    ...     [2, 2],
    ...     [3, 1],
    ... ])
    >>> reps_bsds = np.array([
    ...     [2, 3],
    ...     [1, 4],
    ... ])
    >>> values_tiled, L_bsds_tiled = tile_batched(
    ...     values, L_bsds, reps_bsds, padding_value=0
    ... )
    >>> values_tiled
    array([[[ 1,  2,  1,  2,  1,  2],
            [ 3,  4,  3,  4,  3,  4],
            [ 1,  2,  1,  2,  1,  2],
            [ 3,  4,  3,  4,  3,  4]],
           [[ 7,  7,  7,  7,  0,  0],
            [ 9,  9,  9,  9,  0,  0],
            [11, 11, 11, 11,  0,  0],
            [ 0,  0,  0,  0,  0,  0]]])
    >>> L_bsds_tiled
    array([[4, 6],
           [3, 4]])
    """
    # TODO Optimize this function by using a native padded solution instead.
    # Pack the values.
    values_packed = pack_padded_multidim(
        values, L_bsds
    )  # [sum(L_bs0 * ... * L_bs{D-1})]

    # Tile the values in their packed form.
    values_tiled_packed, L_bsds_tiled = tile_batched_packed(
        values_packed, L_bsds, reps_bsds
    )  # [sum(prod(L_bsds * reps_bsds, axis=1))], [B, D]
    max_L_bsds_tiled = L_bsds_tiled.max(axis=0)  # [D]

    # Pad the tiled values.
    values_tiled = pad_packed_multidim(
        values_tiled_packed,
        L_bsds_tiled,
        max_L_bsds_tiled,
        padding_value=padding_value,
    )  # [B, max(L_bs0 * reps_bs0), ..., max(L_bs{D-1} * reps_bs{D-1})]

    return values_tiled, L_bsds_tiled


def tile_batched_packed(
    values: npt.NDArray[NpGeneric],
    L_bsds: npt.NDArray[np.integer],
    reps_bsds: npt.NDArray[np.integer],
) -> tuple[npt.NDArray[NpGeneric], npt.NDArray[np.intp]]:
    """Tile values from a batch of packed arrays using the given reps.

    Args:
        values: Packed array of all samples concatenated.
            Shape: [sum(L_bs0 * ... * L_bs{D-1})]
        L_bsds: Length of each sample along each dimension.
            Shape: [B, D]
        reps_bsds: Number of times to tile each sample along each dimension.
            Shape: [B, D]

    Returns:
        Tuple containing:
        - Packed array with each sample tiled.
            Shape: [sum(prod(L_bsds * reps_bsds, axis=1))]
        - The lengths of each tiled sample along each dimension.
            Shape: [B, D]

    Examples:
    >>> values = np.array([1, 2, 3, 4, 7, 9, 11])
    >>> L_bsds = np.array([
    ...     [2, 2],
    ...     [3, 1],
    ... ])
    >>> reps_bsds = np.array([
    ...     [2, 3],
    ...     [1, 4],
    ... ])
    >>> values_tiled, L_bsds_tiled = tile_batched_packed(
    ...     values, L_bsds, reps_bsds
    ... )
    >>> values_tiled
    array([ 1,  2,  1,  2,  1,  2,  3,  4,  3,  4,  3,  4,  1,  2,  1,  2,  1,
            2,  3,  4,  3,  4,  3,  4,  7,  7,  7,  7,  9,  9,  9,  9, 11, 11,
           11, 11])
    >>> L_bsds_tiled
    array([[4, 6],
           [3, 4]])
    """
    # Compute the new lengths along each dimension.
    L_bsds_tiled = (L_bsds * reps_bsds).astype(np.intp)  # [B, D]

    # Calculate products of all lengths from dimension d0 to dimension d1.
    # We denote the product from dimension d0 (included) to d1 (excluded) as:
    # _bs_d0_to_d1 = _bs{d0} * _bs{d0+1} * ... * _bs{d1-2} * _bs{d1-1}
    L_bs_tiled_0_to_ds = L_bsds_tiled.cumprod(axis=1) // L_bsds_tiled  # [B, D]
    L_bs_d_to_Ds = np.flip(
        np.flip(L_bsds, axis=1).cumprod(axis=1), axis=1
    )  # [B, D]

    # Tile along each dimension iteratively. Only tile if the amount of reps
    # along that dimension is greater than 1.
    for d in np.nonzero(np.any(reps_bsds != 1, axis=0))[0]:
        # Pretend we are working with 1D subarrays, and construct the
        # corresponding L_bs and reps_bs.
        L_bs = L_bs_d_to_Ds[:, d].repeat(
            L_bs_tiled_0_to_ds[:, d]
        )  # [sum(L_bs_tiled_0_to_d)]
        reps_bs = reps_bsds[:, d].repeat(
            L_bs_tiled_0_to_ds[:, d]
        )  # [sum(L_bs_tiled_0_to_d)]

        # Duplicate the subarrays.
        values, _ = __duplicate_subarrays(
            values, L_bs, reps_bs
        )  # [sum(prod(L_bs_tiled_0_to_{d+1}s * L_bs_{d+1}_to_Ds))], _

    return values, L_bsds_tiled


def broadcast_to_batched(
    values: npt.NDArray[NpGeneric],
    L_bsds: npt.NDArray[np.integer],
    shape_bsds: npt.NDArray[np.integer],
    padding_value: Any = None,
) -> npt.NDArray[NpGeneric]:
    """Broadcast a batch of arrays to the given target shape.

    Args:
        values: The values to broadcast.
            Shape: [B, max(L_bs0), ..., max(L_bs{D-1})]
        L_bsds: Length of each sample along each dimension.
            Shape: [B, D]
        shape_bsds: The target shape to broadcast to for each sample.
            Shape: [B, D]
        padding_value: The value to pad the values with. If None, the values
            are padded with random values. This is faster than padding with
            a specific value.

    Returns:
        The broadcasted values.
            Shape: [B, max(shape_bs0), ..., max(shape_bs{D-1})]

    Examples:
    >>> values = np.array([
    ...     [
    ...         [1, 2],
    ...         [3, 4],
    ...         [5, 6],
    ...     ],
    ...     [
    ...         [7, 8],
    ...         [9, 10],
    ...         [11, 12],
    ...     ],
    ... ])
    >>> L_bsds = np.array([
    ...     [1, 2],
    ...     [3, 1],
    ... ])
    >>> shape_bsds = np.array([
    ...     [2, 2],
    ...     [3, 4],
    ... ])
    >>> values_broadcasted = broadcast_to_batched(
    ...     values, L_bsds, shape_bsds, padding_value=0
    ... )
    >>> values_broadcasted
    array([[[ 1,  2,  0,  0],
            [ 1,  2,  0,  0],
            [ 0,  0,  0,  0]],
           [[ 7,  7,  7,  7],
            [ 9,  9,  9,  9],
            [11, 11, 11, 11]]])
    """
    # TODO Optimize this function by using a native padded solution instead.
    # Pack the values.
    values_packed = pack_padded_multidim(
        values, L_bsds
    )  # [sum(L_bs0 * ... * L_bs{D-1})]

    # Broadcast the values in their packed form.
    values_broadcasted_packed = broadcast_to_batched_packed(
        values_packed, L_bsds, shape_bsds
    )  # [sum(prod(shape_bsds, axis=1))]
    max_shape_bsds = shape_bsds.max(axis=0)  # [D]

    # Pad the broadcasted values.
    values_broadcasted = pad_packed_multidim(
        values_broadcasted_packed,
        shape_bsds,
        max_shape_bsds,
        padding_value=padding_value,
    )  # [B, max(shape_bs0), ..., max(shape_bs{D-1})]

    return values_broadcasted


def broadcast_to_batched_packed(
    values: npt.NDArray[NpGeneric],
    L_bsds: npt.NDArray[np.integer],
    shape_bsds: npt.NDArray[np.integer],
) -> npt.NDArray[NpGeneric]:
    """Broadcast a batch of packed arrays to the given target shape.

    Warning: Unlike np.broadcast_to(), this function does not support adding
    new dimensions of size 1 to the left of the shape. The number of dimensions
    D must be the same in L_bsds and shape_bsds.

    Args:
        values: Packed array of all samples concatenated.
            Shape: [sum(L_bs0 * ... * L_bs{D-1})]
        L_bsds: Length of each sample along each dimension.
            Shape: [B, D]
        shape_bsds: The target shape to broadcast to for each sample.
            Shape: [B, D]

    Returns:
        Packed array with each sample broadcasted to the target shape.
            Shape: [sum(prod(shape_bsds, axis=1))]

    Examples:
    >>> values = np.array([1, 2, 7, 9, 11])
    >>> L_bsds = np.array([
    ...     [1, 2],
    ...     [3, 1],
    ... ])
    >>> shape_bsds = np.array([
    ...     [2, 2],
    ...     [3, 4],
    ... ])
    >>> values_broadcasted = broadcast_to_batched_packed(
    ...     values, L_bsds, shape_bsds
    ... )
    >>> values_broadcasted
    array([ 1,  2,  1,  2,  7,  7,  7,  7,  9,  9,  9,  9, 11, 11, 11, 11])
    """
    # For the values to be broadcastable to the new shape, each dimension must
    # either be equal or the original dimension must be 1.
    if not np.all((L_bsds == shape_bsds) | (L_bsds == 1)):
        raise ValueError(
            "L_bsds and shape_bsds must be broadcastable. Each dimension must"
            " either be equal or the original dimension must be 1."
        )

    # Compute the number of reps along each dimension.
    reps_bsds = np.where(L_bsds != shape_bsds, shape_bsds, 1)  # [B, D]

    # Calculate products of all lengths from dimension d0 to dimension d1.
    # We denote the product from dimension d0 (included) to d1 (excluded) as:
    # _bs_d0_to_d1 = _bs{d0} * _bs{d0+1} * ... * _bs{d1-2} * _bs{d1-1}
    shape_bs_0_to_ds = shape_bsds.cumprod(axis=1) // shape_bsds  # [B, D]
    L_bs_d_to_Ds = np.flip(
        np.flip(L_bsds, axis=1).cumprod(axis=1), axis=1
    )  # [B, D]

    # Broadcast along each dimension iteratively. Only broadcast if the amount
    # of reps along that dimension is greater than 1.
    for d in np.nonzero(np.any(reps_bsds != 1, axis=0))[0]:
        # Pretend we are working with 1D subarrays, and construct the
        # corresponding L_bs and reps_bs.
        L_bs = L_bs_d_to_Ds[:, d].repeat(
            shape_bs_0_to_ds[:, d]
        )  # [sum(shape_bs_0_to_d)]
        reps_bs = reps_bsds[:, d].repeat(
            shape_bs_0_to_ds[:, d]
        )  # [sum(shape_bs_0_to_d)]

        # Duplicate the subarrays.
        values, _ = __duplicate_subarrays(
            values, L_bs, reps_bs
        )  # [sum(prod(shape_bs_0_to_{d+1}s * L_bs_{d+1}_to_Ds))], _

    return values


def meshgrid_batched(
    *xi: tuple[npt.NDArray[NpGeneric], npt.NDArray[np.integer]],
    indexing: str = "xy",
    padding_value: Any = None,
) -> tuple[npt.NDArray[NpGeneric], ...]:
    """Create a meshgrid from a batch of arrays.

    Note: Compared to np.meshgrid(), this function does not support sparse
    outputs, since padding can only be applied along the dimension of the
    corresponding coordinate array. This would mean that any broadcasts you
    perform on the meshgrids this function returns may still contain arbitrary
    padding values in the other dimensions, which is undesirable. Thus, we have
    decided not to support sparse outputs here. Furthermore, the copy option is
    also not supported, since the meshgrid will always require a copy anyway
    due to padding.

    Args:
        *xi: List of tuples containing:
            - Padded input arrays representing the coordinates of a grid.
                Padding could be arbitrary.
                Shape: [B, max(L_bsd)]
            - The lengths of the input arrays.
                Shape: [B]
            Length: D
        indexing: The indexing convention used. 'ij' returns a meshgrid with
            matrix indexing, while 'xy' returns a meshgrid with Cartesian
            indexing.
        padding_value: The value to pad the outputs with. If None, the outputs
            are padded with random values. This is faster than padding with a
            specific value.

    Returns:
        Tuple of [B, max(L_bs0), ..., max(L_bs{D-1})] shaped arrays if indexing
        is 'ij' or tuple of [B, max(L_bs1), max(L_bs0), ..., max(L_bs{D-1})]
        shaped arrays if indexing is 'xy'. Each output array contains the
        D-dimensional meshgrid formed by the input arrays. Padded with
        padding_value.

    Examples:
    >>> x0 = np.array([
    ...     [1, 2],
    ...     [3, 0],
    ... ])
    >>> L_b0 = np.array([2, 1])
    >>> x1 = np.array([
    ...     [10, 20, 30, 0],
    ...     [50, 60, 70, 80],
    ... ])
    >>> L_b1 = np.array([3, 4])

    >>> meshgrid_0, meshgrid_1 = meshgrid_batched(
    ...     (x0, L_b0), (x1, L_b1), indexing="ij", padding_value=-1
    ... )
    >>> meshgrid_0
    array([[[ 1,  1,  1, -1],
            [ 2,  2,  2, -1]],
           [[ 3,  3,  3,  3],
            [-1, -1, -1, -1]]])
    >>> meshgrid_1
    array([[[10, 20, 30, -1],
            [10, 20, 30, -1]],
           [[50, 60, 70, 80],
            [-1, -1, -1, -1]]])

    >>> meshgrid_0, meshgrid_1 = meshgrid_batched(
    ...     (x0, L_b0), (x1, L_b1), indexing="xy", padding_value=-1
    ... )
    >>> meshgrid_0
    array([[[ 1,  2],
            [ 1,  2],
            [ 1,  2],
            [-1, -1]],
           [[ 3, -1],
            [ 3, -1],
            [ 3, -1],
            [ 3, -1]]])
    >>> meshgrid_1
    array([[[10, 10],
            [20, 20],
            [30, 30],
            [-1, -1]],
           [[50, -1],
            [60, -1],
            [70, -1],
            [80, -1]]])
    """
    xs, L_bsds = map(list, zip(*xi))  # D x [B, max(L_bsd)], D x [B]
    L_bsds = np.stack(L_bsds, axis=1)  # [B, D]
    max_L_bsds = np.array([x.shape[1] for x in xs])  # [D]

    B, D = L_bsds.shape

    # Prepare the shape of the output arrays.
    broadcasted_shape = [B, *max_L_bsds]
    broadcasted_bsds = L_bsds.copy()

    # Swap the first two axes if indexing is 'xy'.
    if indexing == "xy" and D >= 2:
        broadcasted_shape[1], broadcasted_shape[2] = (
            broadcasted_shape[2],
            broadcasted_shape[1],
        )
        broadcasted_bsds[:, 0], broadcasted_bsds[:, 1] = (
            broadcasted_bsds[:, 1],
            broadcasted_bsds[:, 0].copy(),
        )

    # Go through each input array and create the corresponding meshgrid.
    meshgrids = []
    for d in range(D):
        # Prepare the shape of the sparse output arrays.
        unsqueezed_shape = [B, *[1] * D]
        unsqueezed_shape[d + 1] = max_L_bsds[d]

        # Swap the first two axes if indexing is 'xy'.
        if indexing == "xy" and D >= 2 and d <= 1:
            unsqueezed_shape[1], unsqueezed_shape[2] = (
                unsqueezed_shape[2],
                unsqueezed_shape[1],
            )

        # Create the meshgrid.
        meshgrid = xs[d].reshape(
            unsqueezed_shape
        )  # [B, 1, ..., max(L_bsd), ..., 1]

        # Broadcast the meshgrid to the full shape.
        meshgrid = np.broadcast_to(
            meshgrid, broadcasted_shape
        ).copy()  # [B, max(L_bs0), ..., max(L_bs{D-1})]

        # Pad the outputs with the padding value.
        if padding_value is not None:
            meshgrid = replace_padding_multidim(
                meshgrid,
                broadcasted_bsds,
                padding_value=padding_value,
                in_place=True,
            )

        meshgrids.append(meshgrid)

    return tuple(meshgrids)


def meshgrid_batched_packed(
    *xi: tuple[npt.NDArray[NpGeneric], npt.NDArray[np.integer]],
    indexing: str = "xy",
) -> tuple[npt.NDArray[NpGeneric], ...]:
    """Create a meshgrid from a batch of packed arrays.

    Note: Compared to np.meshgrid(), this function does not support sparse
    outputs, since the packed format can not make use of broadcasting.
    Furthermore, the copy option is also not supported, since the packed output
    format is already memory efficient.

    Args:
        *xi: List of tuples containing:
            - Packed input arrays representing the coordinates of a grid.
                Shape: [L_d]
            - The lengths of the input arrays.
                Shape: [B]
            Length: D
        indexing: The indexing convention used. 'ij' returns a meshgrid with
            matrix indexing, while 'xy' returns a meshgrid with Cartesian
            indexing.

    Returns:
        Tuple of [sum(L_bs0 * ... * L_bs{D-1})] shaped arrays if indexing is
        'ij' or tuple of [sum(L_bs1 * L_bs0 * ... * L_bs{D-1})] shaped arrays
        if indexing is 'xy'. Each output array contains the D-dimensional
        meshgrid formed by the input arrays.

    Examples:
    >>> x0 = np.array([1, 2, 3])
    >>> L_b0 = np.array([2, 1])
    >>> x1 = np.array([10, 20, 30, 50, 60, 70, 80])
    >>> L_b1 = np.array([3, 4])

    >>> meshgrid_0, meshgrid_1 = meshgrid_batched_packed(
    ...     (x0, L_b0), (x1, L_b1), indexing="ij",
    ... )
    >>> meshgrid_0
    array([1, 1, 1, 2, 2, 2, 3, 3, 3, 3])
    >>> meshgrid_1
    array([10, 20, 30, 10, 20, 30, 50, 60, 70, 80])

    >>> meshgrid_0, meshgrid_1 = meshgrid_batched_packed(
    ...     (x0, L_b0), (x1, L_b1), indexing="xy",
    ... )
    >>> meshgrid_0
    array([1, 2, 1, 2, 1, 2, 3, 3, 3, 3])
    >>> meshgrid_1
    array([10, 10, 20, 20, 30, 30, 50, 60, 70, 80])
    """
    xs, L_bsds = map(list, zip(*xi))  # D x [L_d], D x [B]
    L_bsds = np.stack(L_bsds, axis=1)  # [B, D]

    B, D = L_bsds.shape

    # Prepare the shape of the output arrays.
    broadcasted_bsds = L_bsds.copy()

    # Swap the first two axes if indexing is 'xy'.
    if indexing == "xy" and D >= 2:
        broadcasted_bsds[:, 0], broadcasted_bsds[:, 1] = (
            broadcasted_bsds[:, 1],
            broadcasted_bsds[:, 0].copy(),
        )

    # Go through each input array and create the corresponding meshgrid.
    meshgrids = []
    for d in range(D):
        # Prepare the shape of the sparse output arrays.
        unsqueezed_bsds = np.ones((B, D), dtype=np.intp)
        unsqueezed_bsds[:, d] = L_bsds[:, d]

        # Swap the first two axes if indexing is 'xy'.
        if indexing == "xy" and D >= 2 and d <= 1:
            unsqueezed_bsds[:, 0], unsqueezed_bsds[:, 1] = (
                unsqueezed_bsds[:, 1],
                unsqueezed_bsds[:, 0].copy(),
            )

        # Create the meshgrid.
        meshgrid = xs[d]  # [L_d] = [sum(1 * ... * L_bsd * ... * 1)]

        # Broadcast the meshgrid to the full shape.
        meshgrid = broadcast_to_batched_packed(
            meshgrid, unsqueezed_bsds, broadcasted_bsds
        )  # [sum(L_bs0 * ... * L_bs{D-1})]

        meshgrids.append(meshgrid)

    return tuple(meshgrids)


# ######################## ADVANCED ARRAY MANIPULATION #########################


def swap_idcs_vals_batched(x: npt.NDArray[NpInteger]) -> npt.NDArray[NpInteger]:
    """Swap the indices and values of a batch of 1D arrays.

    Each row in the input array is assumed to contain exactly all integers from
    0 to N - 1, in any order.

    Warning: This function does not explicitly check if the input array
    contains no duplicates. If x contains duplicates, the behavior is
    non-deterministic (one of the values from x will be picked arbitrarily).

    Args:
        x: The array to swap.
            Shape: [B, N]

    Returns:
        The swapped array.
            Shape: [B, N]

    Examples:
    >>> x = np.array([
    ...     [2, 3, 0, 4, 1],
    ...     [1, 3, 2, 0, 4],
    ... ])
    >>> swap_idcs_vals_batched(x)
    array([[2, 4, 0, 1, 3],
           [3, 0, 2, 1, 4]])
    """
    if x.ndim != 2:
        raise ValueError("x must be a batch of 1D arrays.")

    B, N = x.shape
    dtype = x.dtype
    x_swapped = np.empty_like(x)
    x_swapped[np.expand_dims(np.arange(B, dtype=dtype), 1), x] = np.expand_dims(
        np.arange(N, dtype=dtype), 0
    )
    return x_swapped


def swap_idcs_vals_duplicates_batched(
    x: npt.NDArray[NpInteger], stable: bool = False
) -> npt.NDArray[NpInteger]:
    """Swap the indices and values of a batch of 1D arrays allowing duplicates.

    Each row in the input array is assumed to contain integers from 0 to
    M <= N, in any order, and may contain duplicates.

    Each row in the output array will contain exactly all integers from 0 to
    len(x) - 1, in any order.

    If the input doesn't contain duplicates, you should use
    swap_idcs_vals_batched() instead since it is faster (especially for large
    arrays).

    Args:
        x: The array to swap.
            Shape: [B, N]
        stable: Whether to preserve the relative order of equal elements. If
            False (default), an unstable sort is used, which is faster.

    Returns:
        The swapped array.
            Shape: [B, N]

    Examples:
    >>> x = np.array([
    ...     [1, 3, 0, 1, 3],
    ...     [5, 3, 3, 5, 2],
    ... ])
    >>> swap_idcs_vals_duplicates_batched(x, stable=True)
    array([[2, 0, 3, 1, 4],
           [4, 1, 2, 0, 3]])
    """
    if x.ndim != 2:
        raise ValueError("x must be a batch of 1D arrays.")

    dtype = x.dtype

    # For some reason, this O(n log n) algorithm is actually faster than a
    # native implementation that uses a Python for loop with complexity O(n).
    return x.argsort(axis=1, stable=stable).astype(dtype)


# ############################ CONSECUTIVE SEGMENTS ############################


def starts_segments_batched(
    x: npt.NDArray[np.generic], axis: int = 0, padding_value: Any = None
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
    """Find the start index of each consecutive segment in each batch array.

    Note that axis refers to the index of the dimension to sort along AFTER the
    batch dimension.

    Args:
        x: The input array. Consecutive equal values will be grouped.
            Shape: [B, N_0, ..., N_axis, ..., N_{D-1}]
        axis: The dimension along which the segments are lined up.
        padding_value: The value to pad the start indices with. If None, the
            start indices are padded with random values. This is faster than
            padding with a specific value.

    Returns:
        Tuple containing:
        - The start indices for each consecutive segment in x. Padded with
            padding_value.
            Shape: [B, max(S_bs)]
        - The number of consecutive segments in each array.
            Shape: [B]

    Examples:
    >>> x = np.array([
    ...     [4, 4, 4, 2, 2, 8, 3, 3, 3, 3],
    ...     [1, 1, 0, 0, 0, 0, 0, 2, 2, 2],
    ... ])
    >>> starts, S_bs = starts_segments_batched(x, padding_value=0)
    >>> starts
    array([[0, 3, 5, 6],
           [0, 2, 7, 0]])
    >>> S_bs
    array([4, 3])
    """
    B, N_axis = x.shape[0], x.shape[axis + 1]

    # Find the indices where the values change.
    is_change = (
        np.concat(
            [
                np.ones((B, 1), dtype=np.bool_),
                (
                    x.take(
                        np.arange(0, N_axis - 1), axis + 1
                    )  # [B, N_0, ..., N_axis - 1, ..., N_{D-1}]
                    != x.take(
                        np.arange(1, N_axis), axis + 1
                    )  # [B, N_0, ..., N_axis - 1, ..., N_{D-1}]
                ).any(
                    axis=tuple(
                        i for i in range(x.ndim) if i != axis + 1 and i != 0
                    )
                ),  # [B, N_axis - 1]
            ],
            axis=1,
        )  # [B, N_axis]
        if N_axis > 0
        else np.empty((B, 0), dtype=np.bool_)
    )  # [B, N_axis]

    # Find the start of each consecutive segment.
    batch_idcs, starts_idcs = is_change.nonzero()  # [S], [S]

    # Convert to padded representation.
    S_bs = counts_segments(batch_idcs)  # [B]
    max_S_bs = int(S_bs.max())
    starts = pad_packed(
        starts_idcs, S_bs, max_S_bs, padding_value=padding_value
    )  # [B, max(S_bs)]

    return starts, S_bs


@overload
def counts_segments_batched(  # type: ignore
    x: npt.NDArray[np.generic],
    axis: int = 0,
    return_starts: Literal[False] = ...,
    padding_value: Any = None,
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
    pass


@overload
def counts_segments_batched(
    x: npt.NDArray[np.generic],
    axis: int = 0,
    return_starts: Literal[True] = ...,
    padding_value: Any = None,
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp], npt.NDArray[np.intp]]:
    pass


def counts_segments_batched(
    x: npt.NDArray[np.generic],
    axis: int = 0,
    return_starts: bool = False,
    padding_value: Any = None,
) -> (
    tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]
    | tuple[npt.NDArray[np.intp], npt.NDArray[np.intp], npt.NDArray[np.intp]]
):
    """Count the length of each consecutive segment in each batch array.

    Note that axis refers to the index of the dimension to sort along AFTER the
    batch dimension.

    Args:
        x: The input array. Consecutive equal values will be grouped.
            Shape: [B, N_0, ..., N_axis, ..., N_{D-1}]
        axis: The dimension along which the segments are lined up.
        return_starts: Whether to also return the start indices of each
            consecutive segment.
        padding_value: The value to pad the counts with. If None, the counts
            are padded with random values. This is faster than padding with a
            specific value.

    Returns:
        Tuple containing:
        - The counts for each consecutive segment in x. Padded with
            padding_value.
            Shape: [B, max(S_bs)]
        - The number of consecutive segments in each array.
            Shape: [B]
        - (Optional) If return_starts is True, the start indices for each
            consecutive segment in x. Padded with padding_value.
            Shape: [B, max(S_bs)]

    Examples:
    >>> x = np.array([
    ...     [4, 4, 4, 2, 2, 8, 3, 3, 3, 3],
    ...     [1, 1, 0, 0, 0, 0, 0, 2, 2, 2],
    ... ])
    >>> counts, S_bs = counts_segments_batched(x, padding_value=0)
    >>> counts
    array([[3, 2, 1, 4],
           [2, 5, 3, 0]])
    >>> S_bs
    array([4, 3])
    """
    B, N_axis = x.shape[0], x.shape[axis + 1]

    # Find the start of each consecutive segment.
    starts, S_bs = starts_segments_batched(
        x, axis=axis, padding_value=padding_value
    )  # [B, max(S_bs)], [B]

    # Prepare starts for count calculation.
    starts_with_N_axis = np.concat(
        [starts, np.full((B, 1), N_axis, dtype=np.intp)], axis=1
    )  # [B, max(S_bs) + 1]
    starts_with_N_axis[np.arange(B), S_bs] = N_axis

    # Find the count of each consecutive segment.
    counts = (
        np.diff(starts_with_N_axis, axis=1)  # [B, max(S_bs)]
        if N_axis > 0
        else np.empty((B, 0), dtype=np.intp)
    )  # [S]

    # Replace the padding values if requested.
    if padding_value is not None:
        replace_padding(
            counts, S_bs, padding_value=padding_value, in_place=True
        )

    if return_starts:
        return counts, S_bs, starts
    return counts, S_bs


@overload
def outer_indices_segments_batched(  # type: ignore
    x: npt.NDArray[np.generic],
    axis: int = 0,
    return_counts: Literal[False] = ...,
    return_starts: Literal[False] = ...,
    padding_value: Any = None,
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
    pass


@overload
def outer_indices_segments_batched(
    x: npt.NDArray[np.generic],
    axis: int = 0,
    return_counts: Literal[True] = ...,
    return_starts: Literal[False] = ...,
    padding_value: Any = None,
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp], npt.NDArray[np.intp]]:
    pass


@overload
def outer_indices_segments_batched(
    x: npt.NDArray[np.generic],
    axis: int = 0,
    return_counts: Literal[False] = ...,
    return_starts: Literal[True] = ...,
    padding_value: Any = None,
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp], npt.NDArray[np.intp]]:
    pass


@overload
def outer_indices_segments_batched(
    x: npt.NDArray[np.generic],
    axis: int = 0,
    return_counts: Literal[True] = ...,
    return_starts: Literal[True] = ...,
    padding_value: Any = None,
) -> tuple[
    npt.NDArray[np.intp],
    npt.NDArray[np.intp],
    npt.NDArray[np.intp],
    npt.NDArray[np.intp],
]:
    pass


def outer_indices_segments_batched(
    x: npt.NDArray[np.generic],
    axis: int = 0,
    return_counts: bool = False,
    return_starts: bool = False,
    padding_value: Any = None,
) -> (
    tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]
    | tuple[npt.NDArray[np.intp], npt.NDArray[np.intp], npt.NDArray[np.intp]]
    | tuple[
        npt.NDArray[np.intp],
        npt.NDArray[np.intp],
        npt.NDArray[np.intp],
        npt.NDArray[np.intp],
    ]
):
    """Get the outer indices for each consecutive segment in each batch array.

    Args:
        x: The input array. Consecutive equal values will be grouped.
            Shape: [B, N_0, ..., N_axis, ..., N_{D-1}]
        axis: The dimension along which the segments are lined up.
        return_counts: Whether to also return the counts of each consecutive
            segment.
        return_starts: Whether to also return the start indices of each
            consecutive segment.
        padding_value: The value to pad the counts and/or starts with. If None,
            the counts and/or starts are padded with random values. This is
            faster than padding with a specific value.

    Returns:
        Tuple containing:
        - The outer indices for each consecutive segment in x.
            Shape: [B, N_axis]
        - The number of consecutive segments in each array.
            Shape: [B]
        - (Optional) If return_counts is True, the counts for each consecutive
            segment in x. Padded with padding_value.
            Shape: [B, max(S_bs)]
        - (Optional) If return_starts is True, the start indices for each
            consecutive segment in x. Padded with padding_value.
            Shape: [B, max(S_bs)]

    Examples:
    >>> x = np.array([
    ...     [4, 4, 4, 2, 2, 8, 3, 3, 3, 3],
    ...     [1, 1, 0, 0, 0, 0, 0, 2, 2, 2],
    ... ])
    >>> outer_idcs, S_bs = outer_indices_segments_batched(x)
    >>> outer_idcs
    array([[0, 0, 0, 1, 1, 2, 3, 3, 3, 3],
           [0, 0, 1, 1, 1, 1, 1, 2, 2, 2]])
    >>> S_bs
    array([4, 3])
    """
    # Find the start (optional) and count of each consecutive segment.
    if return_starts:
        counts, S_bs, starts = counts_segments_batched(
            x, axis=axis, return_starts=True, padding_value=padding_value
        )  # [B, max(S_bs)], [B], [B, max(S_bs)]
    else:
        counts, S_bs = counts_segments_batched(
            x, axis=axis, padding_value=padding_value
        )  # [B, max(S_bs)], [B]

    # Calculate the outer indices.
    outer_idcs, _ = repeat_batched(
        np.broadcast_to(
            np.expand_dims(np.arange(counts.shape[1], dtype=np.intp), 0),
            counts.shape,
        ),  # [B, max(S_bs)]
        np.expand_dims(S_bs, 1),  # [B, 1]
        counts,
        axis=0,
    )  # [B, N_axis], _

    if return_counts and return_starts:
        return outer_idcs, S_bs, counts, starts  # type: ignore
    if return_counts:
        return outer_idcs, S_bs, counts
    if return_starts:
        return outer_idcs, starts  # type: ignore
    return outer_idcs, S_bs


@overload
def inner_indices_segments_batched(  # type: ignore
    x: npt.NDArray[np.generic],
    axis: int = 0,
    return_counts: Literal[False] = ...,
    return_starts: Literal[False] = ...,
    padding_value: Any = None,
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
    pass


@overload
def inner_indices_segments_batched(
    x: npt.NDArray[np.generic],
    axis: int = 0,
    return_counts: Literal[True] = ...,
    return_starts: Literal[False] = ...,
    padding_value: Any = None,
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp], npt.NDArray[np.intp]]:
    pass


@overload
def inner_indices_segments_batched(
    x: npt.NDArray[np.generic],
    axis: int = 0,
    return_counts: Literal[False] = ...,
    return_starts: Literal[True] = ...,
    padding_value: Any = None,
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp], npt.NDArray[np.intp]]:
    pass


@overload
def inner_indices_segments_batched(
    x: npt.NDArray[np.generic],
    axis: int = 0,
    return_counts: Literal[True] = ...,
    return_starts: Literal[True] = ...,
    padding_value: Any = None,
) -> tuple[
    npt.NDArray[np.intp],
    npt.NDArray[np.intp],
    npt.NDArray[np.intp],
    npt.NDArray[np.intp],
]:
    pass


def inner_indices_segments_batched(
    x: npt.NDArray[np.generic],
    axis: int = 0,
    return_counts: bool = False,
    return_starts: bool = False,
    padding_value: Any = None,
) -> (
    tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]
    | tuple[npt.NDArray[np.intp], npt.NDArray[np.intp], npt.NDArray[np.intp]]
    | tuple[
        npt.NDArray[np.intp],
        npt.NDArray[np.intp],
        npt.NDArray[np.intp],
        npt.NDArray[np.intp],
    ]
):
    """Get the inner indices for each consecutive segment in each batch array.

    Note that axis refers to the index of the dimension to sort along AFTER the
    batch dimension.

    Args:
        x: The input array. Consecutive equal values will be grouped.
            Shape: [B, N_0, ..., N_axis, ..., N_{D-1}]
        axis: The dimension along which the segments are lined up.
        return_counts: Whether to also return the counts of each consecutive
            segment.
        return_starts: Whether to also return the start indices of each
            consecutive segment.
        padding_value: The value to pad the counts and/or starts with. If None,
            the counts and/or starts are padded with random values. This is
            faster than padding with a specific value.

    Returns:
        Tuple containing:
        - The inner indices for each consecutive segment in x.
            Shape: [B, N_axis]
        - The number of consecutive segments in each array.
            Shape: [B]
        - (Optional) If return_counts is True, the counts for each consecutive
            segment in x. Padded with padding_value.
            Shape: [B, max(S_bs)]
        - (Optional) If return_starts is True, the start indices for each
            consecutive segment in x. Padded with padding_value.
            Shape: [B, max(S_bs)]

    Examples:
    >>> x = np.array([
    ...     [4, 4, 4, 2, 2, 8, 3, 3, 3, 3],
    ...     [1, 1, 0, 0, 0, 0, 0, 2, 2, 2],
    ... ])
    >>> inner_idcs, S_bs = inner_indices_segments_batched(x)
    >>> inner_idcs
    array([[0, 1, 2, 0, 1, 0, 0, 1, 2, 3],
           [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]])
    >>> S_bs
    array([4, 3])
    """
    N_axis = x.shape[axis + 1]

    # Find the start and count of each consecutive segment.
    counts, S_bs, starts = counts_segments_batched(
        x, axis=axis, return_starts=True, padding_value=padding_value
    )  # [B, max(S_bs)], [B], [B, max(S_bs)]

    # Calculate the inner indices.
    inner_idcs = (
        np.expand_dims(np.arange(N_axis, dtype=np.intp), 0)  # [1, N_axis]
        - repeat_batched(
            starts, np.expand_dims(S_bs, 1), counts, axis=0
        )[0]  # [B, N_axis]
    )  # [B, N_axis]  # fmt: skip

    if return_counts and return_starts:
        return inner_idcs, S_bs, counts, starts
    if return_counts:
        return inner_idcs, S_bs, counts
    if return_starts:
        return inner_idcs, starts
    return inner_idcs, S_bs


# ################################## LEXSORT ###################################


def lexsort_along_batched(
    x: npt.NDArray[NpGeneric], axis: int = -1, stable: bool = False
) -> tuple[npt.NDArray[NpGeneric], npt.NDArray[np.intp]]:
    """Sort a batched array along axis, taking all others as constant tuples.

    Note that axis refers to the index of the dimension to sort along AFTER the
    batch dimension.

    This is like a batched version of np.sort(), but it doesn't sort along
    the other dimensions. As such, the other dimensions are treated as tuples.
    This function is roughly equivalent to the following Python code, but it
    is much faster.
    >>> np.stack([  # doctest: +SKIP
    ...     np.stack(
    ...         sorted(
    ...             np.unstack(x_b, axis=axis),
    ...             key=tuple,
    ...         ),
    ...         axis=axis,
    ...     )
    ...     for x_b in np.unstack(x, axis=0)
    ... ])

    Args:
        x: The input array.
            Shape: [B, N_0, ..., N_axis, ..., N_{D-1}]
        axis: The dimension to sort along.
        stable: Whether to preserve the relative order of equal elements. If
            False (default), an unstable sort is used, which is faster.

    Returns:
        Tuple containing:
        - Sorted version of x.
            Shape: [B, N_0, ..., N_axis, ..., N_{D-1}]
        - The backmap array, which contains the indices of the sorted values
            in the original input.
            The sorted version of x can be retrieved as follows:
            x_sorted = take_batched(x, axis, backmap)
            Shape: [B, N_axis]

    Examples:
    >>> x = np.array([
    ...     [
    ...         [2, 1],
    ...         [3, 0],
    ...         [1, 2],
    ...         [1, 3],
    ...     ],
    ...     [
    ...         [1, 2],
    ...         [1, 5],
    ...         [3, 4],
    ...         [2, 1],
    ...     ],
    ... ])
    >>> axis = 0

    >>> x_sorted, backmap = lexsort_along_batched(x, axis=axis)
    >>> x_sorted
    array([[[1, 2],
            [1, 3],
            [2, 1],
            [3, 0]],
           [[1, 2],
            [1, 5],
            [2, 1],
            [3, 4]]])
    >>> backmap
    array([[2, 3, 0, 1],
           [0, 1, 3, 2]])

    >>> # Get the lexicographically sorted version of x:
    >>> take_batched(x, axis, backmap)
    array([[[1, 2],
            [1, 3],
            [2, 1],
            [3, 0]],
           [[1, 2],
            [1, 5],
            [2, 1],
            [3, 4]]])
    """
    # See the non-batched version for an explanation of the algorithm.
    B, N_axis = x.shape[0], x.shape[axis + 1]

    if x.ndim == 2:
        y = np.expand_dims(x, 1)  # [B, 1, N_axis]
    else:
        y = np.moveaxis(
            x, axis + 1, -1
        )  # [B, N_0, ..., N_{axis-1}, N_{axis+1}, ..., N_{D-1}, N_axis]
        y = y.reshape(
            B, -1, N_axis
        )  # [B, N_0 * ... * N_{axis-1} * N_{axis+1} * ... * N_{D-1}, N_axis]
    y = np.moveaxis(
        y, 0, 1
    )  # [N_0 * ... * N_{axis-1} * N_{axis+1} * ... * N_{D-1}, B, N_axis]
    y = np.flip(
        y, axis=0
    )  # [N_0 * ... * N_{axis-1} * N_{axis+1} * ... * N_{D-1}, B, N_axis]
    backmap = lexsort(y, axis=-1, stable=stable)  # [B, N_axis]

    # Sort the array along the given dimension.
    x_sorted = take_batched(
        x, axis, backmap
    )  # [B, N_0, ..., N_axis, ..., N_{D-1}]

    # Finally, we return the sorted array and the backmap.
    return x_sorted, backmap


# ################################### UNIQUE ###################################


@overload
def unique_consecutive_batched(  # type: ignore
    x: npt.NDArray[NpGeneric],
    return_inverse: Literal[False] = ...,
    return_counts: Literal[False] = ...,
    axis: int | None = None,
    padding_value: Any = None,
) -> tuple[npt.NDArray[NpGeneric], npt.NDArray[np.intp]]:
    pass


@overload
def unique_consecutive_batched(
    x: npt.NDArray[NpGeneric],
    return_inverse: Literal[True] = ...,
    return_counts: Literal[False] = ...,
    axis: int | None = None,
    padding_value: Any = None,
) -> tuple[npt.NDArray[NpGeneric], npt.NDArray[np.intp], npt.NDArray[np.intp]]:
    pass


@overload
def unique_consecutive_batched(
    x: npt.NDArray[NpGeneric],
    return_inverse: Literal[False] = ...,
    return_counts: Literal[True] = ...,
    axis: int | None = None,
    padding_value: Any = None,
) -> tuple[npt.NDArray[NpGeneric], npt.NDArray[np.intp], npt.NDArray[np.intp]]:
    pass


@overload
def unique_consecutive_batched(
    x: npt.NDArray[NpGeneric],
    return_inverse: Literal[True] = ...,
    return_counts: Literal[True] = ...,
    axis: int | None = None,
    padding_value: Any = None,
) -> tuple[
    npt.NDArray[NpGeneric],
    npt.NDArray[np.intp],
    npt.NDArray[np.intp],
    npt.NDArray[np.intp],
]:
    pass


def unique_consecutive_batched(
    x: npt.NDArray[NpGeneric],
    return_inverse: bool = False,
    return_counts: bool = False,
    axis: int | None = None,
    padding_value: Any = None,
) -> (
    tuple[npt.NDArray[NpGeneric], npt.NDArray[np.intp]]
    | tuple[npt.NDArray[NpGeneric], npt.NDArray[np.intp], npt.NDArray[np.intp]]
    | tuple[
        npt.NDArray[NpGeneric],
        npt.NDArray[np.intp],
        npt.NDArray[np.intp],
        npt.NDArray[np.intp],
    ]
):
    """A batched version of np.unique_consecutive(), but WAY more effiecient.

    Note that axis refers to the index of the dimension to sort along AFTER the
    batch dimension.

    The returned unique elements are retrieved along the requested dimension,
    taking all the other dimensions apart from the batch dimension as constant
    tuples.

    Args:
        x: The input array. If it contains equal values, they must be
            consecutive along the given dimension.
            Shape: [B, N_0, ..., N_axis, ..., N_{D-1}]
        return_inverse: Whether to also return the inverse mapping array.
            This can be used to reconstruct the original array from the
            unique array.
        return_counts: Whether to also return the counts for each unique
            element.
        axis: The dimension to operate on. If None, the unique of the flattened
            input is returned. Otherwise, each of the arrays indexed by the
            given dimension is treated as one of the elements to apply the
            unique operation on. See examples for more details.
        padding_value: The value to pad the unique elements with. If None, the
            unique elements are padded with random values. This is faster than
            padding with a specific value.

    Returns:
        Tuple containing:
        - The unique elements. Padded with padding_value.
            Shape: [
                B, N_0, ..., N_{axis-1}, max(U_bs), N_{axis+1}, ..., N_{D-1}
            ]
        - The amount of unique elements per batch element.
            Shape: [B]
        - (Optional) If return_inverse is True, the indices where elements
            in the original input ended up in the returned unique values.
            The original array can be reconstructed as follows:
            x_reconstructed = take_batched(uniques, axis, inverse)
            Shape: [B, N_axis]
        - (Optional) If return_counts is True, the counts for each unique
            element. Padded with padding_value.
            Shape: [B, max(U_bs)]

    Examples:
    >>> # 1D example: -----------------------------------------------------
    >>> x = np.array([
    ...     [9, 9, 9, 9, 10, 10],
    ...     [8, 8, 7, 7, 9, 9],
    ... ])
    >>> axis = 0

    >>> uniques, U_bs, inverse, counts = unique_consecutive_batched(
    ...     x,
    ...     return_inverse=True,
    ...     return_counts=True,
    ...     axis=axis,
    ...     padding_value=0,
    ... )
    >>> uniques
    array([[ 9, 10,  0],
           [ 8,  7,  9]])
    >>> U_bs
    array([2, 3])
    >>> inverse
    array([[0, 0, 0, 0, 1, 1],
           [0, 0, 1, 1, 2, 2]])
    >>> counts
    array([[4, 2, 0],
           [2, 2, 2]])

    >>> # Reconstruct the original array:
    >>> take_batched(uniques, axis, inverse)
    array([[ 9,  9,  9,  9, 10, 10],
           [ 8,  8,  7,  7,  9,  9]])

    >>> # 2D example: -----------------------------------------------------
    >>> x = np.array([
    ...     [
    ...         [7, 9, 9, 10],
    ...         [8, 10, 10, 9],
    ...         [9, 8, 8, 7],
    ...         [9, 7, 7, 7],
    ...     ],
    ...     [
    ...         [7, 7, 7, 7],
    ...         [7, 7, 7, 10],
    ...         [9, 9, 9, 8],
    ...         [8, 8, 8, 8],
    ...     ],
    ... ])
    >>> axis = 1

    >>> uniques, U_bs, inverse, counts = unique_consecutive_batched(
    ...     x,
    ...     return_inverse=True,
    ...     return_counts=True,
    ...     axis=axis,
    ...     padding_value=0,
    ... )
    >>> uniques
    array([[[ 7,  9, 10],
            [ 8, 10,  9],
            [ 9,  8,  7],
            [ 9,  7,  7]],
           [[ 7,  7,  0],
            [ 7, 10,  0],
            [ 9,  8,  0],
            [ 8,  8,  0]]])
    >>> U_bs
    array([3, 2])
    >>> inverse
    array([[0, 1, 1, 2],
           [0, 0, 0, 1]])
    >>> counts
    array([[1, 2, 1],
           [3, 1, 0]])

    >>> # Reconstruct the original array:
    >>> take_batched(uniques, axis, inverse)
    array([[[ 7,  9,  9, 10],
            [ 8, 10, 10,  9],
            [ 9,  8,  8,  7],
            [ 9,  7,  7,  7]],
           [[ 7,  7,  7,  7],
            [ 7,  7,  7, 10],
            [ 9,  9,  9,  8],
            [ 8,  8,  8,  8]]])
    """
    if axis is None:
        raise NotImplementedError(
            "axis=None is not implemented yet. Please specify a dimension"
            " explicitly."
        )

    # Find each consecutive segment.
    if return_inverse and return_counts:
        outer_idcs, U_bs, counts, starts = outer_indices_segments_batched(
            x,
            axis=axis,
            return_counts=True,
            return_starts=True,
            padding_value=padding_value,
        )  # [B, N_axis], [B], [B, max(U_bs)], [B, max(U_bs)]
    elif return_inverse:
        outer_idcs, U_bs, starts = outer_indices_segments_batched(
            x, axis=axis, return_starts=True, padding_value=padding_value
        )  # [B, N_axis], [B], [B, max(U_bs)]
    elif return_counts:
        counts, U_bs, starts = counts_segments_batched(
            x, axis=axis, return_starts=True, padding_value=padding_value
        )  # [B, max(U_bs)], [B], [B, max(U_bs)]
    else:
        starts, U_bs = starts_segments_batched(
            x, axis=axis
        )  # [B, max(U_bs)], [B]

    # Find the unique values.
    replace_padding(starts, U_bs, in_place=True)
    uniques = take_batched(
        x, axis, starts
    )  # [B, N_0, ..., N_{axis-1}, max(U_bs), N_{axis+1}..., N_{D-1}]

    # Replace the padding values if requested.
    if padding_value is not None:
        uniques = np.moveaxis(
            uniques, axis + 1, 1
        )  # [B, max(U_bs), N_0, ..., N_{axis-1}, N_{axis+1}, ..., N_{D-1}]
        replace_padding(
            uniques, U_bs, padding_value=padding_value, in_place=True
        )
        uniques = np.moveaxis(
            uniques, 1, axis + 1
        )  # [B, N_0, ..., N_{axis-1}, max(U_bs), N_{axis+1}, ..., N_{D-1}]

    # Return the requested values.
    if return_inverse and return_counts:
        return uniques, U_bs, outer_idcs, counts  # type: ignore
    if return_inverse:
        return uniques, U_bs, outer_idcs  # type: ignore
    if return_counts:
        return uniques, U_bs, counts  # type: ignore
    return uniques, U_bs


@overload
def unique_batched(  # type: ignore
    x: npt.NDArray[NpGeneric],
    return_backmap: Literal[False] = ...,
    return_inverse: Literal[False] = ...,
    return_counts: Literal[False] = ...,
    axis: int | None = None,
    stable: bool = False,
    padding_value: Any = None,
) -> tuple[npt.NDArray[NpGeneric], npt.NDArray[np.intp]]:
    pass


@overload
def unique_batched(
    x: npt.NDArray[NpGeneric],
    return_backmap: Literal[True] = ...,
    return_inverse: Literal[False] = ...,
    return_counts: Literal[False] = ...,
    axis: int | None = None,
    stable: bool = False,
    padding_value: Any = None,
) -> tuple[npt.NDArray[NpGeneric], npt.NDArray[np.intp], npt.NDArray[np.intp]]:
    pass


@overload
def unique_batched(
    x: npt.NDArray[NpGeneric],
    return_backmap: Literal[False] = ...,
    return_inverse: Literal[True] = ...,
    return_counts: Literal[False] = ...,
    axis: int | None = None,
    stable: bool = False,
    padding_value: Any = None,
) -> tuple[npt.NDArray[NpGeneric], npt.NDArray[np.intp], npt.NDArray[np.intp]]:
    pass


@overload
def unique_batched(
    x: npt.NDArray[NpGeneric],
    return_backmap: Literal[False] = ...,
    return_inverse: Literal[False] = ...,
    return_counts: Literal[True] = ...,
    axis: int | None = None,
    stable: bool = False,
    padding_value: Any = None,
) -> tuple[npt.NDArray[NpGeneric], npt.NDArray[np.intp], npt.NDArray[np.intp]]:
    pass


@overload
def unique_batched(
    x: npt.NDArray[NpGeneric],
    return_backmap: Literal[True] = ...,
    return_inverse: Literal[True] = ...,
    return_counts: Literal[False] = ...,
    axis: int | None = None,
    stable: bool = False,
    padding_value: Any = None,
) -> tuple[
    npt.NDArray[NpGeneric],
    npt.NDArray[np.intp],
    npt.NDArray[np.intp],
    npt.NDArray[np.intp],
]:
    pass


@overload
def unique_batched(
    x: npt.NDArray[NpGeneric],
    return_backmap: Literal[True] = ...,
    return_inverse: Literal[False] = ...,
    return_counts: Literal[True] = ...,
    axis: int | None = None,
    stable: bool = False,
    padding_value: Any = None,
) -> tuple[
    npt.NDArray[NpGeneric],
    npt.NDArray[np.intp],
    npt.NDArray[np.intp],
    npt.NDArray[np.intp],
]:
    pass


@overload
def unique_batched(
    x: npt.NDArray[NpGeneric],
    return_backmap: Literal[False] = ...,
    return_inverse: Literal[True] = ...,
    return_counts: Literal[True] = ...,
    axis: int | None = None,
    stable: bool = False,
    padding_value: Any = None,
) -> tuple[
    npt.NDArray[NpGeneric],
    npt.NDArray[np.intp],
    npt.NDArray[np.intp],
    npt.NDArray[np.intp],
]:
    pass


@overload
def unique_batched(
    x: npt.NDArray[NpGeneric],
    return_backmap: Literal[True] = ...,
    return_inverse: Literal[True] = ...,
    return_counts: Literal[True] = ...,
    axis: int | None = None,
    stable: bool = False,
    padding_value: Any = None,
) -> tuple[
    npt.NDArray[NpGeneric],
    npt.NDArray[np.intp],
    npt.NDArray[np.intp],
    npt.NDArray[np.intp],
    npt.NDArray[np.intp],
]:
    pass


def unique_batched(
    x: npt.NDArray[NpGeneric],
    return_backmap: bool = False,
    return_inverse: bool = False,
    return_counts: bool = False,
    axis: int | None = None,
    stable: bool = False,
    padding_value: Any = None,
) -> (
    tuple[npt.NDArray[NpGeneric], npt.NDArray[np.intp]]
    | tuple[npt.NDArray[NpGeneric], npt.NDArray[np.intp], npt.NDArray[np.intp]]
    | tuple[
        npt.NDArray[NpGeneric],
        npt.NDArray[np.intp],
        npt.NDArray[np.intp],
        npt.NDArray[np.intp],
    ]
    | tuple[
        npt.NDArray[NpGeneric],
        npt.NDArray[np.intp],
        npt.NDArray[np.intp],
        npt.NDArray[np.intp],
        npt.NDArray[np.intp],
    ]
):
    """A batched version of np.unique(), but WAY more efficient.

    Note that axis refers to the index of the dimension to sort along AFTER the
    batch dimension.

    The returned unique elements are retrieved along the requested dimension,
    taking all the other dimensions apart from the batch dimension as constant
    tuples.

    Args:
        x: The input array.
            Shape: [B, N_0, ..., N_axis, ..., N_{D-1}]
        return_backmap: Whether to also return the backmap array.
            This can be used to sort the original array.
        return_inverse: Whether to also return the inverse mapping array.
            This can be used to reconstruct the original array from the
            unique array.
        return_counts: Whether to also return the counts of each unique
            element.
        axis: The dimension to operate on. If None, the unique of the flattened
            input is returned. Otherwise, each of the arrays indexed by the
            given dimension is treated as one of the elements to apply the
            unique operation on. See examples for more details.
        stable: Whether to preserve the relative order of equal elements. If
            False (default), an unstable sort is used, which is faster. Note
            that this only has an effect on the backmap array, so setting
            stable=True while return_backmap=False will have no effect. We will
            throw an error in this case to avoid degrading performance
            unnecessarily.
        padding_value: The value to pad the unique elements with. If None, the
            unique elements are padded with random values. This is faster than
            padding with a specific value.

    Returns:
        Tuple containing:
        - The unique elements, guaranteed to be sorted along the given
            dimension. Padded with padding_value.
            Shape: [
                B, N_0, ..., N_{axis-1}, max(U_bs), N_{axis+1}, ..., N_{D-1}
            ]
        - The amount of unique values per batch element.
            Shape: [B]
        - (Optional) If return_backmap is True, the backmap array, which
            contains the indices of the unique values in the original input.
            The sorted version of x can be retrieved as follows:
            x_sorted = take_batched(x, axis, backmap)
            Shape: [B, N_axis]
        - (Optional) If return_inverse is True, the indices where elements
            in the original input ended up in the returned unique values.
            The original array can be reconstructed as follows:
            x_reconstructed = take_batched(uniques, axis, inverse)
            Shape: [B, N_axis]
        - (Optional) If return_counts is True, the counts for each unique
            element. Padded with padding_value.
            Shape: [B, max(U_bs)]

    Examples:
    >>> # 1D example: -----------------------------------------------------
    >>> x = np.array([
    ...     [9, 10, 9, 9, 10, 9],
    ...     [8, 7, 9, 9, 8, 7],
    ... ])
    >>> axis = 0

    >>> uniques, U_bs, backmap, inverse, counts = unique_batched(
    ...     x,
    ...     return_backmap=True,
    ...     return_inverse=True,
    ...     return_counts=True,
    ...     axis=axis,
    ...     stable=True,
    ...     padding_value=0,
    ... )
    >>> uniques
    array([[ 9, 10,  0],
           [ 7,  8,  9]])
    >>> U_bs
    array([2, 3])
    >>> backmap
    array([[0, 2, 3, 5, 1, 4],
           [1, 5, 0, 4, 2, 3]])
    >>> inverse
    array([[0, 1, 0, 0, 1, 0],
           [1, 0, 2, 2, 1, 0]])
    >>> counts
    array([[4, 2, 0],
           [2, 2, 2]])

    >>> # Get the lexicographically sorted version of x:
    >>> take_batched(x, axis, backmap)
    array([[ 9,  9,  9,  9, 10, 10],
           [ 7,  7,  8,  8,  9,  9]])

    >>> # Reconstruct the original array:
    >>> take_batched(uniques, axis, inverse)
    array([[ 9, 10,  9,  9, 10,  9],
           [ 8,  7,  9,  9,  8,  7]])

    >>> # 2D example: -----------------------------------------------------
    >>> x = np.array([
    ...     [
    ...         [9, 10, 7, 9],
    ...         [10, 9, 8, 10],
    ...         [8, 7, 9, 8],
    ...         [7, 7, 9, 7],
    ...     ],
    ...     [
    ...         [7, 7, 7, 7],
    ...         [7, 10, 7, 7],
    ...         [9, 8, 9, 9],
    ...         [8, 8, 8, 8],
    ...     ],
    ... ])
    >>> axis = 1

    >>> uniques, U_bs, backmap, inverse, counts = unique_batched(
    ...     x,
    ...     return_backmap=True,
    ...     return_inverse=True,
    ...     return_counts=True,
    ...     axis=axis,
    ...     stable=True,
    ...     padding_value=0,
    ... )
    >>> uniques
    array([[[ 7,  9, 10],
            [ 8, 10,  9],
            [ 9,  8,  7],
            [ 9,  7,  7]],
           [[ 7,  7,  0],
            [ 7, 10,  0],
            [ 9,  8,  0],
            [ 8,  8,  0]]])
    >>> U_bs
    array([3, 2])
    >>> backmap
    array([[2, 0, 3, 1],
           [0, 2, 3, 1]])
    >>> inverse
    array([[1, 2, 0, 1],
           [0, 1, 0, 0]])
    >>> counts
    array([[1, 2, 1],
           [3, 1, 0]])

    >>> # Get the lexicographically sorted version of x:
    >>> take_batched(x, axis, backmap)
    array([[[ 7,  9,  9, 10],
            [ 8, 10, 10,  9],
            [ 9,  8,  8,  7],
            [ 9,  7,  7,  7]],
           [[ 7,  7,  7,  7],
            [ 7,  7,  7, 10],
            [ 9,  9,  9,  8],
            [ 8,  8,  8,  8]]])

    >>> # Reconstruct the original array:
    >>> take_batched(uniques, axis, inverse)
    array([[[ 9, 10,  7,  9],
            [10,  9,  8, 10],
            [ 8,  7,  9,  8],
            [ 7,  7,  9,  7]],
           [[ 7,  7,  7,  7],
            [ 7, 10,  7,  7],
            [ 9,  8,  9,  9],
            [ 8,  8,  8,  8]]])
    """
    if axis is None:
        raise NotImplementedError(
            "axis=None is not implemented yet. Please specify a dimension"
            " explicitly."
        )

    if stable and not return_backmap:
        raise ValueError(
            "stable=True has no effect when return_backmap=False, but it"
            " degrades performance. Please use either stable=False or"
            " return_backmap=True."
        )

    # Sort along the given dimension, taking all the other dimensions as
    # constant tuples. Torch's sort() doesn't work here since it will
    # sort the other dimensions independently.
    x_sorted, backmap = lexsort_along_batched(
        x, axis=axis, stable=stable
    )  # [B, N_0, ..., N_axis, ..., N_{D-1}], [B, N_axis]

    out = unique_consecutive_batched(
        x_sorted,
        return_inverse=return_inverse,
        return_counts=return_counts,
        axis=axis,
        padding_value=padding_value,
    )

    aux = []
    if return_backmap:
        aux.append(backmap)
    if return_inverse:
        # The backmap wasn't taken into account by unique_consecutive(), so we
        # have to apply it to the inverse mapping here.
        backmap_inv = swap_idcs_vals_batched(backmap)  # [B, N_axis]
        aux.append(
            np.take_along_axis(out[2], backmap_inv, axis=1)  # type: ignore
        )
    if return_counts:
        aux.append(out[-1])

    return out[0], out[1], *aux


# ############################ CONSECUTIVE SEGMENTS ############################


def counts_segments_ints_batched(
    x: npt.NDArray[np.integer], high: int
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
    """Count the frequency of each consecutive value, with values in [0, high).

    Args:
        x: The array for which to count the frequency of each integer value.
            Consecutive values in x are grouped together. It is assumed that
            every segment has a unique integer value that is not present in any
            other segment. The values in x must be in the range [0, high).
            Shape: [B, N]
        high: The highest value to include in the count (exclusive). May be
            higher than the maximum value in x, in which case the remaining
            values will be set to 0.

    Returns:
        Tuple containing:
        - The frequency of each element in x in range [0, high).
            Shape: [B, high]
        - The amount of unique elements per batch element.
            Shape: [B]

    Examples:
    >>> x = np.array([
    ...    [4, 4, 4, 2, 2, 8, 3, 3, 3, 3],
    ...    [1, 1, 0, 0, 0, 0, 0, 2, 2, 2],
    ... ])
    >>> freqs, U_bs = counts_segments_ints_batched(x, 10)
    >>> freqs
    array([[0, 0, 2, 4, 3, 0, 0, 0, 1, 0],
           [5, 2, 3, 0, 0, 0, 0, 0, 0, 0]])
    >>> U_bs
    array([4, 3])
    """
    if x.ndim != 2:
        raise ValueError("x must be a batch of 1D arrays.")

    B = x.shape[0]

    freqs = np.zeros((B, high), dtype=np.intp)
    uniques, U_bs, counts = unique_consecutive_batched(
        x, return_counts=True, axis=0
    )  # [B, max(U_bs)], [B], [B, max(U_bs)]
    freqs[
        np.arange(B).repeat(U_bs),  # [U]
        pack_padded(uniques, U_bs),  # [U]
    ] = pack_padded(counts, U_bs)  # [U]  # fmt: skip
    return freqs, U_bs


# ################################## GROUPBY ###################################


@overload
def groupby_batched(  # type: ignore
    keys: npt.NDArray[NpGeneric1],
    vals: npt.NDArray[NpGeneric2] | None = None,
    stable: bool = False,
    as_sequence: Literal[True] = ...,
    padding_value: Any = None,
) -> list[
    tuple[
        npt.NDArray[NpGeneric1],
        npt.NDArray[np.bool_],
        npt.NDArray[NpGeneric2],
        npt.NDArray[np.intp],
    ]
]:
    pass


@overload
def groupby_batched(
    keys: npt.NDArray[NpGeneric1],
    vals: npt.NDArray[NpGeneric2] | None = None,
    stable: bool = False,
    as_sequence: Literal[False] = ...,
    padding_value: Any = None,
) -> tuple[
    npt.NDArray[NpGeneric1],
    npt.NDArray[np.intp],
    npt.NDArray[NpGeneric2],
    npt.NDArray[np.intp],
]:
    pass


def groupby_batched(
    keys: npt.NDArray[NpGeneric1],
    vals: npt.NDArray[NpGeneric2] | None = None,
    stable: bool = False,
    as_sequence: bool = True,
    padding_value: Any = None,
) -> (
    list[
        tuple[
            npt.NDArray[NpGeneric1],
            npt.NDArray[np.bool_],
            npt.NDArray[NpGeneric2],
            npt.NDArray[np.intp],
        ]
    ]
    | tuple[
        npt.NDArray[NpGeneric1],
        npt.NDArray[np.intp],
        npt.NDArray[NpGeneric2],
        npt.NDArray[np.intp],
    ]
):
    """Group values by keys.

    Args:
        keys: The keys to group by.
            Shape: [B, N, *]
        vals: The values to group. If None, the values are set to the indices of
            the keys (i.e. vals = arange_batched(np.full((B,), N))[0]).
            Shape: [B, N, **]
        stable: Whether to preserve the order of vals that have the same key. If
            False (default), an unstable sort is used, which is faster.
        as_sequence: Whether to return the result as a sequence of (key, vals)
            tuples (True) or as packed arrays (False).
        padding_value: The value to pad the keys and vals with. If None, the
            keys and vals are padded with random values. This is faster than
            padding with a specific value.

    Returns:
        - If as_sequence is True (default), a list of tuples containing:
            - A unique key for every sample in the batch. Will be yielded in
                sorted order. If the unique keys for a specific sample are
                exhausted but not for others, its return value will be padding
                instead. Padded with padding_value.
                Shape: [B, *]
            - Whether the key is valid (True) or padding (False).
                Shape: [B]
            - The values that correspond to the keys for every sample in the
                batch. Sorted if stable is True. Padded with padding_value.
                Shape: [B, max(N_key_bs), **]
            - The number of values for each key, for every sample in the batch.
                If the unique keys for a specific sample are exhausted but not
                for others, its return value will be padding instead. Padded
                with padding_value.
                Shape: [B]
        - If as_sequence is False, a tuple containing:
            - Array of unique keys, sorted. Padded with padding_value.
                Shape: [B, max(U_bs), *]
            - Array with the amount of unique keys per batch element.
                Shape: [B]
            - Array of values stored a packed manner, grouped by key. Along
                every batch element, the first N_key0 values correspond to the
                first key, the next N_key1 values correspond to the second key,
                etc. Each group of values is sorted if stable is True. Padded
                with padding_value.
                Shape: [B, N, **]
            - Array containing the number of values for each unique key. Padded
                with padding_value.
                Shape: [B, max(U_bs)]

    Examples:
    >>> keys = np.array([
    ...     [4, 2, 4, 3, 2, 8, 4],
    ...     [1, 0, 1, 2, 0, 1, 0],
    ... ])
    >>> vals = np.array([
    ...     [
    ...         [0, 1],
    ...         [2, 3],
    ...         [4, 5],
    ...         [6, 7],
    ...         [8, 9],
    ...         [10, 11],
    ...         [12, 13],
    ...     ],
    ...     [
    ...         [14, 15],
    ...         [16, 17],
    ...         [18, 19],
    ...         [20, 21],
    ...         [22, 23],
    ...         [24, 25],
    ...         [26, 27],
    ...     ],
    ... ])

    >>> # Return as sequence of (key, vals) tuples:
    >>> grouped = groupby_batched(
    ...     keys, vals, stable=True, as_sequence=True, padding_value=0
    ... )
    >>> for key, mask, vals_group, counts in grouped:
    ...     print("Key:")
    ...     print(key)
    ...     print("Mask:")
    ...     print(mask)
    ...     print("Grouped Vals:")
    ...     print(vals_group)
    ...     print("Counts:")
    ...     print(counts)
    ...     print()
    Key:
    [2 0]
    Mask:
    [ True  True]
    Grouped Vals:
    [[[ 2  3]
      [ 8  9]
      [ 0  0]]
     [[16 17]
      [22 23]
      [26 27]]]
    Counts:
    [2 3]
    <BLANKLINE>
    Key:
    [3 1]
    Mask:
    [ True  True]
    Grouped Vals:
    [[[ 6  7]
      [ 0  0]
      [ 0  0]]
     [[14 15]
      [18 19]
      [24 25]]]
    Counts:
    [1 3]
    <BLANKLINE>
    Key:
    [4 2]
    Mask:
    [ True  True]
    Grouped Vals:
    [[[ 0  1]
      [ 4  5]
      [12 13]]
     [[20 21]
      [ 0  0]
      [ 0  0]]]
    Counts:
    [3 1]
    <BLANKLINE>
    Key:
    [8 0]
    Mask:
    [ True False]
    Grouped Vals:
    [[[10 11]]
     [[ 0  0]]]
    Counts:
    [1 0]

    >>> # Return as packed arrays:
    >>> keys_unique, U_bs, vals_grouped, counts = groupby_batched(
    ...     keys, vals, stable=True, as_sequence=False, padding_value=0
    ... )
    >>> keys_unique
    array([[2, 3, 4, 8],
           [0, 1, 2, 0]])
    >>> U_bs
    array([4, 3])
    >>> vals_grouped
    array([[[ 2,  3],
            [ 8,  9],
            [ 6,  7],
            [ 0,  1],
            [ 4,  5],
            [12, 13],
            [10, 11]],
           [[16, 17],
            [22, 23],
            [26, 27],
            [14, 15],
            [18, 19],
            [24, 25],
            [20, 21]]])
    >>> counts
    array([[2, 1, 3, 1],
           [3, 3, 1, 0]])
    """
    # Create a mapping from keys to values.
    keys_unique, U_bs, backmap, counts = unique_batched(
        keys,
        return_backmap=True,
        return_counts=True,
        axis=0,
        stable=stable,
        padding_value=padding_value,
    )  # [B, max(U_bs), *], [B], [B, N], [B, max(U_bs)]

    if vals is None:
        # Use the backmap directly as values.
        vals_grouped = cast(npt.NDArray[NpGeneric2], backmap)  # [B, N]
    else:
        # Rearrange values to match keys_unique.
        vals_grouped = take_batched(vals, 0, backmap)  # [B, N, **]

    # Return the results.
    if not as_sequence:
        return keys_unique, U_bs, vals_grouped, counts

    B, N = keys.shape[0], keys.shape[1]

    # Calculate outer indices.
    outer_idcs, _ = repeat_batched(
        np.broadcast_to(
            np.expand_dims(np.arange(counts.shape[1], dtype=np.intp), 0),
            counts.shape,
        ),  # [B, max(U_bs)]
        np.expand_dims(U_bs, 1),  # [B, 1]
        counts,
        axis=0,
    )  # [B, N], _

    # Create masks for every batch of unique keys.
    masks = (
        np.expand_dims(outer_idcs, 1)  # [B, 1, N]
        == np.expand_dims(
            np.arange(counts.shape[1], dtype=np.intp), (0, 2)
        )  # [1, max(U_bs), 1]
    )  # [B, max(U_bs), N]  # fmt: skip

    # Create the sequences of (key, vals_group) tuples.
    return [
        (
            keys_unique[:, u],  # [B, *]
            u < U_bs,  # [B]
            apply_mask(
                vals_grouped,
                masks[:, u],
                np.full((B,), N),
                padding_value=padding_value,
            )[0],  # [B, max(N_key_bs), **]
            counts[:, u],  # [B]
        )
        for u in range(keys_unique.shape[1])
    ]  # fmt: skip
