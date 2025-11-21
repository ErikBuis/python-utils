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
    apply_mask,
    counts_segments,
    lexsort,
    mask_padding,
    pack_padded,
    pad_packed,
    replace_padding,
)

# ################################### MATHS ####################################


def mean_padding_batched(
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
    """
    if not is_padding_zero:
        values = replace_padding(values, L_bs)

    return (
        values.sum(axis=1)  # [B, *]
        / L_bs.reshape(-1, *([1] * (values.ndim - 2)))  # [B, *]
    )  # [B, *]  # fmt: skip


def stddev_padding_batched(
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
    """
    means = mean_padding_batched(
        values, L_bs, is_padding_zero=is_padding_zero
    )  # [B, *]
    values_centered = values - np.expand_dims(means, 1)  # [B, max(L_bs), *]
    return np.sqrt(
        mean_padding_batched(np.square(values_centered), L_bs)
    )  # [B, *]


def min_padding_batched(
    values: npt.NDArray[np.number],
    L_bs: npt.NDArray[np.integer],
    is_padding_inf: bool = False,
) -> npt.NDArray[np.floating]:
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
    """
    if not is_padding_inf:
        values = replace_padding(values, L_bs, padding_value=float("inf"))

    return np.amin(values, axis=1)  # [B, *]


def max_padding_batched(
    values: npt.NDArray[np.number],
    L_bs: npt.NDArray[np.integer],
    is_padding_minus_inf: bool = False,
) -> npt.NDArray[np.floating]:
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
    """
    if not is_padding_minus_inf:
        values = replace_padding(values, L_bs, padding_value=float("-inf"))

    return np.amax(values, axis=1)  # [B, *]


def any_padding_batched(
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
    """
    if not is_padding_false:
        values = replace_padding(values, L_bs, padding_value=False)

    return values.any(axis=1)  # [B, *]


def all_padding_batched(
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
    """
    if not is_padding_true:
        values = replace_padding(values, L_bs, padding_value=True)

    return values.all(axis=1)  # [B, *]


def interp_batched(
    x: npt.NDArray[np.number],
    xp: npt.NDArray[np.number],
    fp: npt.NDArray[np.number],
    left: npt.NDArray[np.number] | None = None,
    right: npt.NDArray[np.number] | None = None,
    period: npt.NDArray[np.number] | None = None,
) -> npt.NDArray[np.number]:
    """Like np.interp(), but batched.

    This function performs linear interpolation on a batch of 1D arrays.

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
    """
    _, M = xp.shape

    # Handle periodic interpolation.
    if period is not None:
        if (period <= 0).any():
            raise ValueError("period must be positive.")

        xp_mod = xp % np.expand_dims(period, 1)  # [B, M]  # type: ignore
        sorted_idcs = xp_mod.argsort(axis=1)  # [B, M]
        xp = np.take_along_axis(xp_mod, sorted_idcs, 1)  # [B, M]
        fp = np.take_along_axis(fp, sorted_idcs, 1)  # [B, M]

    # Check if xp is weakly monotonically increasing.
    if not (np.diff(xp, axis=1) >= 0).all():
        raise ValueError(
            "xp must be weakly monotonically increasing along the last"
            " dimension."
        )

    # Find indices of neighbours in xp.
    right_idx = np.searchsorted(xp, x)  # [B, N]
    left_idx = right_idx - 1  # [B, N]

    # Clamp indices to valid range (we will handle the edges later).
    left_idx = left_idx.clip(a_min=0, a_max=M - 1)  # [B, N]
    right_idx = right_idx.clip(a_min=0, a_max=M - 1)  # [B, N]

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
    L_bs: npt.NDArray[np.integer], max_L_bs: int, num_samples: int
) -> npt.NDArray[np.int64]:
    """Sample unique indices i in [0, L_b-1] for each element in the batch.

    Warning: If the number of valid values in an element is less than the
    number of samples, then only the first L_b indices are unique. The
    remaining indices are sampled with replacement.

    Args:
        L_bs: The number of valid values for each element in the batch.
            Shape: [B]
        max_L_bs: The maximum number of values of any element in the batch.
        num_samples: The number of indices to sample for each element in the
            batch.

    Returns:
        The sampled unique indices.
            Shape: [B, num_samples]
    """
    # Select unique elements for each sample in the batch.
    # If the number of elements is less than the number of samples, we uniformly
    # sample with replacement. To do this, the np.clip(min=num_samples) and
    # % L_b operations are used.
    weights = mask_padding(
        L_bs.clip(amin=num_samples), max_L_bs
    ).astype(np.float64)  # [B, max(L_bs)]  # fmt: skip
    weights = weights / weights.sum(axis=1, keepdims=True)  # [B, max(L_bs)]
    rng = np.random.default_rng()
    return (
        rng.multinomial(num_samples, weights)  # [B, num_samples]
        % np.expand_dims(L_bs, 1)  # [B, num_samples]
    )  # [B, num_samples]  # fmt: skip


def sample_unique_pairs_batched(
    L_bs: npt.NDArray[np.integer], max_L_bs: int, num_samples: int
) -> npt.NDArray[np.integer]:
    """Sample unique pairs of indices (i, j), where i and j are in [0, L_b-1].

    Warning: If the number of valid values in an element is less than the
    number of samples, then only the first L_b * (L_b - 1) // 2 pairs are
    unique. The remaining pairs are sampled with replacement.

    Args:
        L_bs: The number of valid values for each element in the batch.
            Shape: [B]
        max_L_bs: The maximum number of valid values.
        num_samples: The number of pairs to sample.

    Returns:
        The sampled unique pairs of indices.
            Shape: [B, num_samples, 2]
    """
    # Compute the number of unique pairs of indices.
    P_bs = L_bs * (L_bs - 1) // 2  # [B]
    max_P_bs = max_L_bs * (max_L_bs - 1) // 2

    # Select unique pairs of elements for each sample in the batch.
    idcs_pairs = sample_unique_batched(
        P_bs, max_P_bs, num_samples
    )  # [B, num_samples]

    # Convert the pair indices to element indices.
    # np.triu_indices() returns the indices in the wrong order, e.g.:
    #    0 1 2 3 4
    # 0  x x x x x
    # 1  0 x x x x
    # 2  1 4 x x x
    # 3  2 5 7 x x
    # 4  3 6 8 9 x
    # This order is not suitable for all elements in the batch, as the number
    # of valid values L_b can change between elements. We need to change the
    # order to:
    #    0 1 2 3 4
    # 0  x x x x x
    # 1  0 x x x x
    # 2  1 2 x x x
    # 3  3 4 5 x x
    # 4  6 7 8 9 x
    # This is done using the max_P_bs - 1 - triu_idcs trick. However, the
    # order of the elements is still in reverse, so when indexing, we index at
    # -idcs_pairs - 1 instead of at idcs_pairs.
    triu_idcs = np.stack(
        np.triu_indices(max_L_bs, max_L_bs, 1)
    )  # [2, max(P_bs)]
    triu_idcs = max_L_bs - 1 - triu_idcs
    idcs_elements = triu_idcs[:, -idcs_pairs - 1]  # [2, B, num_samples]
    return np.transpose(idcs_elements, (1, 2, 0))  # [B, num_samples, 2]


# ########################## BASIC ARRAY MANIPULATION ##########################


def arange_batched(
    starts: npt.NDArray[np.number],
    stops: npt.NDArray[np.number] | None = None,
    steps: npt.NDArray[np.number] | None = None,
    padding_value: Any = None,
    dtype: np.dtype | None = None,
) -> tuple[npt.NDArray[np.number], npt.NDArray[np.intp]]:
    """Create a batch of arrays with values in the range [start, end).

    Args:
        starts: The start value for each array in the batch.
            Shape: [B]
        stops: The end value for each array in the batch. If None, the end
            value is set to the start value.
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
        - A batch of arrays with values in the range [start, end).
            Padded with padding_value.
            Shape: [B, max(L_bs)]
        - The number of values of the arange sequences in the batch.
            Shape: [B]
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
    """Create a batch of arrays with values in the range [start, end).

    Args:
        starts: The start value for each array in the batch.
            Shape: [B]
        stops: The end value for each array in the batch. If None, the end
            value is set to the start value.
            Shape: [B]
        steps: The step value for each array in the batch. If None, the step
            value is set to 1.
            Shape: [B]
        dtype: The data type of the output array.

    Returns:
        Tuple containing:
        - A batch of arrays with values in the range [start, end).
            Shape: [L]
        - The number of values of the arange sequences in the batch.
            Shape: [B]
        - The maximum length of the arange sequences in the batch.
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

    # Compute the starts and offsets of the arange sequences in parallel.
    L_bs = np.ceil((stops - starts) / steps).astype(np.intp)  # [B]
    max_L_bs = int(L_bs.max())
    starts_repeated = np.repeat(starts, L_bs)  # [L]
    steps_repeated = np.repeat(steps, L_bs)  # [L]
    offsets_packed = np.cumsum(steps_repeated)  # [L]

    # Correct the offsets to start from zero for each sequence.
    nonzero_idcs = np.nonzero(L_bs)[0]  # [B_nonzero]
    L_bs_nonzero = L_bs[nonzero_idcs]  # [B_nonzero]
    if len(nonzero_idcs) != 0:
        start_idcs = np.cumsum(L_bs_nonzero) - L_bs_nonzero  # [B_nonzero]
        corrections_packed = np.repeat(
            offsets_packed[start_idcs], L_bs_nonzero
        )  # [L]
        offsets_packed -= corrections_packed

    # Compute the arange sequences in parallel.
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
    """Create a batch of arrays with values in the range [start, end].

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
        - A batch of arrays with values in the range [start, end].
            Padded with padding_value.
            Shape: [B, max(L_bs)]
        - The number of values of the linspace sequences in the batch.
            Shape: [B]
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
        )  # ignore division by zero since we already handle it in np.where
        steps = np.where(L_bs != 1, (stops - starts) / (L_bs - 1), 0)  # [B]

    # Compute the linspace sequences in parallel.
    linspaces = (
        np.expand_dims(starts, 1)  # [B, 1]
        + np.arange(max_L_bs, dtype=inferred_dtype)  # [max(L_bs)]
        * np.expand_dims(steps, 1)  # [B, 1]
    )  # [B, max(L_bs)]  # fmt: skip

    # Set the last element of each linspace to the stop value manually to avoid
    # numerical issues.
    nonzero_idcs = np.nonzero(L_bs)[0]  # [B_nonzero]
    L_bs_nonzero = L_bs[nonzero_idcs]  # [B_nonzero]
    if len(nonzero_idcs) != 0:
        stop_idcs = L_bs_nonzero - 1  # [B_nonzero]

        # Only set the stop values for sequences with at least two elements.
        atleasttwo_idcs = nonzero_idcs[L_bs_nonzero != 1]  # [B_atleasttwo]
        stop_idcs = stop_idcs[L_bs_nonzero != 1]  # [B_atleasttwo]

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
    """Create a batch of arrays with values in the range [start, end].

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
        - A batch of arrays with values in the range [start, end].
            Padded with padding_value.
            Shape: [B, max(L_bs)]
        - The number of values of the linspace sequences in the batch.
            Shape: [B]
        - The maximum length of the linspace sequences in the batch.
    """
    # Compute the steps of the linspace sequences in parallel.
    L_bs = nums.astype(np.intp)  # [B]
    max_L_bs = int(L_bs.max())
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=RuntimeWarning
        )  # ignore division by zero since we already handle it in np.where
        steps = np.where(L_bs != 1, (stops - starts) / (L_bs - 1), 0)  # [B]

    # Compute the starts and offsets of the linspace sequences in parallel.
    starts_repeated = np.repeat(starts, L_bs)  # [L]
    steps_repeated = np.repeat(steps, L_bs)  # [L]
    offsets_packed = np.cumsum(steps_repeated)  # [L]

    # Correct the offsets to start from zero for each sequence.
    nonzero_idcs = np.nonzero(L_bs)[0]  # [B_nonzero]
    L_bs_nonzero = L_bs[nonzero_idcs]  # [B_nonzero]
    if len(nonzero_idcs) != 0:
        start_idcs = np.cumsum(L_bs_nonzero) - L_bs_nonzero  # [B_nonzero]
        corrections_packed = np.repeat(
            offsets_packed[start_idcs], L_bs_nonzero
        )  # [L]
        offsets_packed -= corrections_packed

    # Compute the linspace sequences in parallel.
    linspaces = starts_repeated + offsets_packed  # [L]

    # Set the last element of each linspace to the stop value manually to avoid
    # numerical issues.
    if len(nonzero_idcs) != 0:
        stop_idcs = np.cumsum(L_bs_nonzero) - 1  # [B_nonzero]

        # Only set the stop values for sequences with at least two elements.
        atleasttwo_idcs = nonzero_idcs[L_bs_nonzero != 1]  # [B_atleasttwo]
        stop_idcs = stop_idcs[L_bs_nonzero != 1]  # [B_atleasttwo]

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
    batch dimension. So e.g. if x has shape [B, N_0, N_1, N_2], then axis=0
    refers to N_0, axis=1 refers to N_1, etc.

    Args:
        values: The values to select from.
            Shape: [B, N_0, ..., N_axis, ..., N_{D-1}]
        axis: The dimension to select along.
        indices: The indices to select.
            Shape: [B, N_select]

    Returns:
        The selected values.
            Shape: [B, N_0, ..., N_{axis-1}, N_select, N_{axis+1}, ..., N_{D-1}]
    """
    idcs_reshape = [1] * values.ndim
    idcs_reshape[0] = indices.shape[0]
    idcs_reshape[axis + 1] = indices.shape[1]
    idcs_expand = list(values.shape)
    idcs_expand[axis + 1] = indices.shape[1]
    return np.take_along_axis(
        values,
        np.broadcast_to(indices.reshape(idcs_reshape), idcs_expand),
        axis + 1,
    )


def repeat_batched(
    values: npt.NDArray[NpGeneric],
    repeats: npt.NDArray[np.integer],
    sum_repeats: npt.NDArray[np.integer],
    max_sum_repeats: int,
    axis: int = 0,
    padding_value: Any = None,
) -> npt.NDArray[NpGeneric]:
    """Repeat values from a batch of arrays using the given repeats.

    Note that axis refers to the index of the dimension to sort along AFTER the
    batch dimension. So e.g. if x has shape [B, N_0, N_1, N_2], then axis=0
    refers to N_0, axis=1 refers to N_1, etc.

    Args:
        values: The values to repeat.
            Shape: [B, N_0, ..., N_axis, ..., N_{D-1}]
        repeats: The number of times to repeat each value.
            Shape: [B, N_axis]
        sum_repeats: The sum of repeats for each element in the batch.
            Must be equal to sum(repeats, axis=1).
            Shape: [B]
        max_sum_repeats: The maximum sum of repeats for any element in the
            batch. Must be equal to max(sum(repeats, axis=1)).
        axis: The dimension to repeat along.
        padding_value: The value to pad the values with. If None, the values
            are padded with random values. This is faster than padding with
            a specific value.

    Returns:
        The repeated values. Padded with padding_value.
            Shape: [B, N_0, ..., max_sum_repeats, ..., N_{D-1}]

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
    >>> repeats = np.array([
    ...     [1, 2, 3],
    ...     [0, 1, 0],
    ... ])
    >>> sum_repeats = np.array([6, 1])
    >>> max_sum_repeats = 6
    >>> repeat_batched(
    ...     values,
    ...     repeats,
    ...     sum_repeats,
    ...     max_sum_repeats,
    ...     axis=0,
    ...     padding_value=0,
    ... )
    array([[[ 1,  2],
            [ 3,  4],
            [ 3,  4],
            [ 5,  6],
            [ 5,  6],
            [ 5,  6]],
           [[ 9, 10],
            [ 0,  0],
            [ 0,  0],
            [ 0,  0],
            [ 0,  0],
            [ 0,  0]]])
    """
    # Move axis to the front and merge it with the batch dimension.
    # This allows us to use np.repeat() directly.
    values = np.moveaxis(
        values, axis + 1, 1
    )  # [B, N_axis, N_0, ..., N_{axis-1}, N_{axis+1}, ..., N_{D-1}]
    values = values.reshape(
        -1, *values.shape[2:]
    )  # [B * N_axis, N_0, ..., N_{axis-1}, N_{axis+1}, ..., N_{D-1}]
    repeats_reshaped = repeats.reshape(-1)  # [B * N_axis]

    # Repeat the values.
    values = values.repeat(
        repeats_reshaped, axis=0
    )  # [B * sum(repeats), N_0, ..., N_{axis-1}, N_{axis+1}, ..., N_{D-1}]

    # Un-merge the batch and axis dimensions and move axis back to its original
    # position.
    values = pad_packed(
        values, sum_repeats, max_sum_repeats, padding_value=padding_value
    )  # [B, max_sum_repeats, N_0, ..., N_{axis-1}, N_{axis+1}, ..., N_{D-1}]
    values = np.moveaxis(
        values, 1, axis + 1
    )  # [B, N_0, ..., max_sum_repeats, ..., N_{D-1}]
    return values


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
    x_swapped = np.empty_like(x)
    x_swapped[np.expand_dims(np.arange(B, dtype=x.dtype), 1), x] = (
        np.expand_dims(np.arange(N, dtype=x.dtype), 0)
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

    If the input doesn't contain duplicates, you should use swap_idcs_vals()
    instead since it is faster (especially for large arrays).

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

    # Believe it or not, this O(n log n) algorithm is actually faster than a
    # native implementation that uses a Python for loop with complexity O(n).
    return x.argsort(axis=1, stable=stable).astype(x.dtype)  # type: ignore


# ############################ CONSECUTIVE SEGMENTS ############################


def starts_segments_batched(
    x: npt.NDArray[np.generic], axis: int = 0, padding_value: Any = None
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
    """Find the start index of each consecutive segment in each batch array.

    Note that axis refers to the index of the dimension to sort along AFTER the
    batch dimension. So e.g. if x has shape [B, N_0, N_1, N_2], then axis=0
    refers to N_0, axis=1 refers to N_1, etc.

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
    array([
        [0, 3, 5, 6],
        [0, 2, 7, 0],
    ])
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
    batch_idcs = batch_idcs.astype(np.intp)
    starts_idcs = starts_idcs.astype(np.intp)
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
    batch dimension. So e.g. if x has shape [B, N_0, N_1, N_2], then axis=0
    refers to N_0, axis=1 refers to N_1, etc.

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
    array([
        [3, 2, 1, 4],
        [2, 5, 3, 0],
    ])
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

    Returns:
        Tuple containing:
        - The outer indices for each consecutive segment in x.
            Shape: [B, N_axis]
        - The number of consecutive segments in each array.
            Shape: [B]
        - (Optional) If return_counts is True, the counts for each consecutive
            segment in x.
            Shape: [B, max(S_bs)]
        - (Optional) If return_starts is True, the start indices for each
            consecutive segment in x.
            Shape: [B, max(S_bs)]

    Examples:
    >>> x = np.array([
    ...     [4, 4, 4, 2, 2, 8, 3, 3, 3, 3],
    ...     [1, 1, 0, 0, 0, 0, 0, 2, 2, 2],
    ... ])

    >>> outer_idcs, S_bs = outer_indices_segments_batched(x)
    >>> outer_idcs
    array([
        [0, 0, 0, 1, 1, 2, 3, 3, 3, 3],
        [0, 0, 1, 1, 1, 1, 1, 2, 2, 2],
    ])
    >>> S_bs
    array([4, 3])
    """
    B, N_axis = x.shape[0], x.shape[axis + 1]

    # Find the start (optional) and count of each consecutive segment.
    if return_starts:
        counts, S_bs, starts = counts_segments_batched(
            x, axis=axis, return_starts=True, padding_value=padding_value
        )  # [B, max(S_bs)], [B], [B, max(S_bs)]
    else:
        counts, S_bs = counts_segments_batched(
            x, axis=axis, padding_value=padding_value
        )  # [B, max(S_bs)], [B]

    # Prepare counts for outer index calculation.
    counts_with_zeros = replace_padding(counts, S_bs)
    sum_counts = np.full((B,), N_axis, dtype=np.intp)  # [B]
    max_sum_counts = N_axis

    # Calculate the outer indices.
    outer_idcs = repeat_batched(
        np.broadcast_to(
            np.expand_dims(np.arange(counts.shape[1], dtype=np.intp), 0),
            counts.shape,
        ),  # [B, max(S_bs)]
        counts_with_zeros,
        sum_counts,
        max_sum_counts,
        axis=0,
    )  # [B, N_axis]

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
    batch dimension. So e.g. if x has shape [B, N_0, N_1, N_2], then axis=0
    refers to N_0, axis=1 refers to N_1, etc.

    Args:
        x: The input array. Consecutive equal values will be grouped.
            Shape: [B, N_0, ..., N_axis, ..., N_{D-1}]
        axis: The dimension along which the segments are lined up.
        return_counts: Whether to also return the counts of each consecutive
            segment.
        return_starts: Whether to also return the start indices of each
            consecutive segment.
        padding_value: The value to pad the inner indices with. If None, the
            inner indices are padded with random values. This is faster than
            padding with a specific value.

    Returns:
        Tuple containing:
        - The indices for each consecutive segment in x.
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
    array([
        [0, 1, 2, 0, 1, 0, 0, 1, 2, 3],
        [0, 1, 0, 1, 2, 3, 4, 0, 1, 2],
    ])
    >>> S_bs
    array([4, 3])
    """
    B, N_axis = x.shape[0], x.shape[axis + 1]

    # Find the start and count of each consecutive segment.
    counts, S_bs, starts = counts_segments_batched(
        x, axis=axis, return_starts=True, padding_value=padding_value
    )  # [B, max(S_bs)], [B], [B, max(S_bs)]

    # Prepare counts for inner index calculation.
    counts_with_zeros = replace_padding(counts, S_bs)
    sum_counts = np.full((B,), N_axis, dtype=np.intp)  # [B]
    max_sum_counts = N_axis

    # Calculate the inner indices.
    inner_idcs = (
        np.expand_dims(np.arange(N_axis, dtype=np.intp), 0)  # [1, N_axis]
        - repeat_batched(
            starts, counts_with_zeros, sum_counts, max_sum_counts, axis=0
        )  # [B, N_axis]
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
    batch dimension. So e.g. if x has shape [B, N_0, N_1, N_2], then axis=0
    refers to N_0, axis=1 refers to N_1, etc.

    This is like a batched version of np.sort(), but it doesn't sort along
    the other dimensions. As such, the other dimensions are treated as tuples.
    This function is roughly equivalent to the following Python code, but it
    is much faster.
    >>> np.stack([
    ...     np.stack(
    ...         sorted(
    ...             map(np.squeeze, np.split(x_b, x_b.shape[axis], axis=axis)),
    ...             key=tuple,
    ...         ),
    ...         axis=axis,
    ...     )
    ...     for x_b in map(np.squeeze, np.split(x, x.shape[0], axis=0))
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
            >>> x_sorted = take_batched(x, axis, backmap)
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
    # We can use lexsort() to sort only the requested dimension.
    # First, we prepare the array for lexsort(). The input to this function
    # must be a tuple of array-like objects, that are evaluated from last to
    # first. This is quite confusing, so I'll put an example here. If we have:
    # >>> x = array([[[15, 13],
    # ...             [11,  4],
    # ...             [16,  2]],
    # ...            [[ 7, 21],
    # ...             [ 3, 20],
    # ...             [ 8, 22]],
    # ...            [[19, 14],
    # ...             [ 5, 12],
    # ...             [ 6,  0]],
    # ...            [[23,  1],
    # ...             [10, 17],
    # ...             [ 9, 18]]])
    # And axis=1, then the input to lexsort() must be:
    # >>> lexsort(array([[ 1, 17, 18],
    # ...                [23, 10,  9],
    # ...                [14, 12,  0],
    # ...                [19,  5,  6],
    # ...                [21, 20, 22],
    # ...                [ 7,  3,  8],
    # ...                [13,  4,  2],
    # ...                [15, 11, 16]]))
    # Note that the first row is evaluated last and the last row is evaluated
    # first. We can now see that the sorting order will be 11 < 15 < 16, so
    # lexsort() will return array([1, 0, 2]). I thouroughly tested what the
    # absolute fastest way is to perform this operation, and it turns out that
    # the following is the best way to do it:
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
    batch dimension. So e.g. if x has shape [B, N_0, N_1, N_2], then axis=0
    refers to N_0, axis=1 refers to N_1, etc.

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
        axis: The dimension to operate on. If None, the unique of the
            flattened input is returned. Otherwise, each of the arrays
            indexed by the given dimension is treated as one of the elements
            to apply the unique operation on. See examples for more details.
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
            >>> x_reconstructed = take_batched(uniques, axis, inverse)
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
    batch dimension. So e.g. if x has shape [B, N_0, N_1, N_2], then axis=0
    refers to N_0, axis=1 refers to N_1, etc.

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
        axis: The dimension to operate on. If None, the unique of the
            flattened input is returned. Otherwise, each of the arrays
            indexed by the given dimension is treated as one of the elements
            to apply the unique operation on. See examples for more details.
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
            >>> x_sorted = take_batched(x, axis, backmap)
            Shape: [B, N_axis]
        - (Optional) If return_inverse is True, the indices where elements
            in the original input ended up in the returned unique values.
            The original array can be reconstructed as follows:
            >>> x_reconstructed = take_batched(uniques, axis, inverse)
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
    # sort the other dimensions as well.
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
        aux.append(np.take_along_axis(out[2], backmap_inv, 1))  # type: ignore
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
    array([
        [0, 0, 2, 4, 3, 0, 0, 0, 1, 0],
        [5, 2, 3, 0, 0, 0, 0, 0, 0, 0],
    ])
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
            the keys (i.e. vals = arange_batched(N)).
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
                every batch element, the first N_key1 values correspond to the
                first key, the next N_key2 values correspond to the second key,
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
    ...     print(f"Key:\\n{key}")
    ...     print(f"Mask:\\n{mask}")
    ...     print(f"Grouped Vals:\\n{vals_group}")
    ...     print(f"Counts:\\n{counts}")
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

    Key:
    [3 1]
    Mask:
    [ True  True]
    Grouped Vals:
    [[[ 6  7]
      [ 0  0]
      [ 0  0]
     [[14 15]
      [18 19]
      [24 25]]]
    Counts:
    [1 3]

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

    # Rearrange values to match keys_unique.
    if vals is None:
        vals_grouped = cast(npt.NDArray[NpGeneric2], backmap)  # [B, N]
    else:
        vals_grouped = take_batched(vals, 0, backmap)  # [B, N, **]

    # Return the results.
    if not as_sequence:
        return keys_unique, U_bs, vals_grouped, counts

    B, N = keys.shape[:2]

    # Prepare counts for outer index calculation.
    counts_with_zeros = replace_padding(counts, U_bs)
    sum_counts = np.full((B,), N, dtype=np.intp)  # [B]
    max_sum_counts = N

    # Calculate outer indices.
    outer_idcs = repeat_batched(
        np.broadcast_to(
            np.expand_dims(np.arange(counts.shape[1], dtype=np.intp), 0),
            counts.shape,
        ),  # [B, max(U_bs)]
        counts_with_zeros,
        sum_counts,
        max_sum_counts,
        axis=0,
    )  # [B, N]

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
                sum_counts,
                padding_value=padding_value,
            )[0],  # [B, max(N_key_bs), **]
            counts[:, u],  # [B]
        )
        for u in range(keys_unique.shape[1])
    ]  # fmt: skip
