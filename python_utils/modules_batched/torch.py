from __future__ import annotations

import warnings
from typing import Any, Literal, overload

import torch

from ..modules.torch import (
    apply_mask,
    counts_segments,
    lexsort,
    pack_padded,
    pack_padded_multidim,
    pad_packed,
    pad_packed_multidim,
    permuted,
    replace_padding,
    replace_padding_multidim,
)

# ################################### MATHS ####################################


def sum_padding_batched(
    values: torch.Tensor, L_bs: torch.Tensor, is_padding_zero: bool = False
) -> torch.Tensor:
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
    >>> values = torch.tensor([
    ...     [1, 2, 3, -1],
    ...     [5, -1, -1, -1],
    ...     [-1, -1, -1, -1],
    ...     [13, 14, 15, 16],
    ... ])
    >>> L_bs = torch.tensor([3, 1, 0, 4])
    >>> sum_padding_batched(values, L_bs)
    tensor([ 6,  5,  0, 58])
    """
    if not is_padding_zero:
        values = replace_padding(values, L_bs)

    return values.sum(dim=1)  # [B, *]


def mean_padding_batched(
    values: torch.Tensor, L_bs: torch.Tensor, is_padding_zero: bool = False
) -> torch.Tensor:
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
    >>> values = torch.tensor([
    ...     [1, 2, 3, -1],
    ...     [5, -1, -1, -1],
    ...     [-1, -1, -1, -1],
    ...     [13, 14, 15, 16],
    ... ])
    >>> L_bs = torch.tensor([3, 1, 0, 4])
    >>> with warnings.catch_warnings():
    ...     warnings.filterwarnings("ignore", category=RuntimeWarning)
    ...     mean_padding_batched(values, L_bs)
    tensor([ 2.0000,  5.0000,     nan, 14.5000])
    """
    return sum_padding_batched(
        values, L_bs, is_padding_zero=is_padding_zero
    ) / L_bs.reshape(
        -1, *[1] * (values.ndim - 2)
    )  # [B, *]  # [B, *]


def stddev_padding_batched(
    values: torch.Tensor, L_bs: torch.Tensor, is_padding_zero: bool = False
) -> torch.Tensor:
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
    >>> values = torch.tensor([
    ...     [1, 2, 3, -1],
    ...     [5, -1, -1, -1],
    ...     [-1, -1, -1, -1],
    ...     [13, 14, 15, 16],
    ... ])
    >>> L_bs = torch.tensor([3, 1, 0, 4])
    >>> with warnings.catch_warnings():
    ...     warnings.filterwarnings("ignore", category=RuntimeWarning)
    ...     stddev_padding_batched(values, L_bs)
    tensor([0.8165, 0.0000,    nan, 1.1180])
    """
    means = mean_padding_batched(
        values, L_bs, is_padding_zero=is_padding_zero
    )  # [B, *]
    values_centered = values - means.unsqueeze(1)  # [B, max(L_bs), *]
    return mean_padding_batched(values_centered.square(), L_bs).sqrt()  # [B, *]


def min_padding_batched(
    values: torch.Tensor, L_bs: torch.Tensor, is_padding_inf: bool = False
) -> torch.Tensor:
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
    >>> values = torch.tensor([
    ...     [1, 2, 3, -1],
    ...     [5, -1, -1, -1],
    ...     [-1, -1, -1, -1],
    ...     [13, 14, 15, 16],
    ... ])
    >>> L_bs = torch.tensor([3, 1, 0, 4])
    >>> min_padding_batched(values, L_bs)
    tensor([ 1.,  5., inf, 13.])
    """
    if not is_padding_inf:
        values = replace_padding(
            values.float(), L_bs, padding_value=float("inf")
        )

    return values.amin(dim=1)  # [B, *]


def max_padding_batched(
    values: torch.Tensor, L_bs: torch.Tensor, is_padding_minus_inf: bool = False
) -> torch.Tensor:
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
    >>> values = torch.tensor([
    ...     [1, 2, 3, -1],
    ...     [5, -1, -1, -1],
    ...     [-1, -1, -1, -1],
    ...     [13, 14, 15, 16],
    ... ])
    >>> L_bs = torch.tensor([3, 1, 0, 4])
    >>> max_padding_batched(values, L_bs)
    tensor([ 3.,  5., -inf, 16.])
    """
    if not is_padding_minus_inf:
        values = replace_padding(
            values.float(), L_bs, padding_value=float("-inf")
        )

    return values.amax(dim=1)  # [B, *]


def any_padding_batched(
    values: torch.Tensor, L_bs: torch.Tensor, is_padding_false: bool = False
) -> torch.Tensor:
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
    >>> values = torch.tensor([
    ...     [False, False, True, False],
    ...     [False, False, False, False],
    ...     [False, False, False, False],
    ...     [True, True, True, True],
    ... ])
    >>> L_bs = torch.tensor([3, 2, 0, 4])
    >>> any_padding_batched(values, L_bs)
    tensor([ True, False, False,  True])
    """
    if not is_padding_false:
        values = replace_padding(values, L_bs, padding_value=False)

    return values.any(dim=1)  # [B, *]


def all_padding_batched(
    values: torch.Tensor, L_bs: torch.Tensor, is_padding_true: bool = False
) -> torch.Tensor:
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
    >>> values = torch.tensor([
    ...     [True, True, True, False],
    ...     [True, True, False, False],
    ...     [True, True, False, True],
    ...     [False, True, True, True],
    ... ])
    >>> L_bs = torch.tensor([3, 2, 0, 4])
    >>> all_padding_batched(values, L_bs)
    tensor([ True,  True,  True, False])
    """
    if not is_padding_true:
        values = replace_padding(values, L_bs, padding_value=True)

    return values.all(dim=1)  # [B, *]


def interp_batched(
    x: torch.Tensor,
    xp: torch.Tensor,
    fp: torch.Tensor,
    left: torch.Tensor | None = None,
    right: torch.Tensor | None = None,
    period: torch.Tensor | None = None,
) -> torch.Tensor:
    """Like np.interp(), but for PyTorch tensors and batched.

    Performs linear interpolation on a batch of 1D tensors.

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
    >>> x = torch.tensor([
    ...     [10.5, 200.0, 40.0, 56.0],
    ...     [1.5, 2.5, 10.0, -1.0],
    ... ])
    >>> xp = torch.tensor([
    ...     [0.0, 1.0, 20.0, 100.0],
    ...     [0.0, 1.0, 2.0, 3.0],
    ... ])
    >>> fp = torch.tensor([
    ...     [0.0, 100.0, 200.0, 300.0],
    ...     [0.0, 10.0, 20.0, 30.0],
    ... ])
    >>> interp_batched(x, xp, fp)
    tensor([[150., 300., 225., 245.],
            [ 15.,  25.,  30.,   0.]])
    """
    # Handle periodic interpolation.
    if period is not None:
        if (period <= 0).any():
            raise ValueError("period must be positive.")

        # Normalize x and xp to [0, period).
        x %= period.unsqueeze(1)  # [B, N]
        xp %= period.unsqueeze(1)  # [B, M]

        # Re-sort xp and fp after the modulo operation.
        sorted_idcs = xp.argsort(dim=1)  # [B, M]
        xp = xp.gather(1, sorted_idcs)  # [B, M]
        fp = fp.gather(1, sorted_idcs)  # [B, M]
        # Extend xp and fp tensors to handle wrap-around interpolation. Add the
        # last point before the first, and the first point after the last.
        xp = torch.concat(
            [
                (xp[:, -1] - period).unsqueeze(1),
                xp,
                (xp[:, 0] + period).unsqueeze(1),
            ],
            dim=1,
        )  # [B, M + 2]
        fp = torch.concat(
            [fp[:, -1].unsqueeze(1), fp, fp[:, 0].unsqueeze(1)], dim=1
        )  # [B, M + 2]

    M = xp.shape[1]

    # Check if xp is weakly monotonically increasing.
    if not (xp.diff(dim=1) >= 0).all():
        raise ValueError(
            "xp must be weakly monotonically increasing along the last"
            " dimension."
        )

    # Find indices of neighbours in xp.
    right_idx = torch.searchsorted(xp, x)  # [B, N]
    left_idx = right_idx - 1  # [B, N]

    # Clamp indices to valid range (we will handle the edges later).
    left_idx = left_idx.clamp(min=0, max=M - 1)  # [B, N]
    right_idx = right_idx.clamp(min=0, max=M - 1)  # [B, N]

    # Gather neighbour values.
    x_left = xp.gather(1, left_idx)  # [B, N]
    x_right = xp.gather(1, right_idx)  # [B, N]
    y_left = fp.gather(1, left_idx)  # [B, N]
    y_right = fp.gather(1, right_idx)  # [B, N]

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
        y[is_left] = left.repeat_interleave(is_left.sum(dim=1)).to(y.dtype)

        # Handle right edge.
        if right is None:
            right = fp[:, -1]  # [B]
        is_right = x > xp[:, [-1]]  # [B, N]
        y[is_right] = right.repeat_interleave(is_right.sum(dim=1)).to(y.dtype)

    return y


# ################################### RANDOM ###################################


def sample_unique_batched(
    L_bs: torch.Tensor, max_L_bs: int, padding_value: Any = None
) -> torch.Tensor:
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
    >>> L_bs = torch.tensor([5, 3, 0, 4])
    >>> max_L_bs = 5
    >>> unique_idcs = sample_unique_batched(L_bs, max_L_bs, padding_value=0)
    >>> unique_idcs  # doctest: +SKIP
    tensor([[4, 2, 3, 1, 0],
            [2, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [1, 3, 2, 0, 0]])
    """
    B = len(L_bs)
    device = L_bs.device
    idcs = (
        torch.arange(max_L_bs, device=device).unsqueeze(0).expand(B, max_L_bs)
    )  # [B, max(L_bs)]
    permuted_idcs = permuted(idcs, dim=1)  # [B, max(L_bs)]
    mask = permuted_idcs < L_bs.unsqueeze(1)  # [B, max(L_bs)]
    random_idcs, _ = apply_mask(
        permuted_idcs,
        mask,
        torch.full((B,), max_L_bs, device=device),
        padding_value=padding_value,
    )  # [B, max(L_bs)]
    return random_idcs


def sample_unique_pairs_batched(
    L_bs: torch.Tensor, max_L_bs: int, padding_value: Any = None
) -> torch.Tensor:
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
    >>> L_bs = torch.tensor([3, 1, 0, 4])
    >>> max_L_bs = 4
    >>> unique_pairs = sample_unique_pairs_batched(
    ...     L_bs, max_L_bs, padding_value=0
    ... )
    >>> unique_pairs  # doctest: +SKIP
    tensor([[[1, 0],
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
    device = L_bs.device

    # Compute the number of unique pairs of indices.
    P_bs = L_bs * (L_bs - 1) // 2  # [B]
    max_P_bs = max_L_bs * (max_L_bs - 1) // 2

    # Select unique pairs of elements for each sample in the batch.
    idcs_pairs = sample_unique_batched(
        P_bs, max_P_bs, padding_value=0
    )  # [B, max(P_bs)]

    # Convert the pair indices to element indices.
    # torch.tril_indices(max_L_bs, max_L_bs, offset=-1) returns the indices as
    # follows:
    # i\j 0 1 2 3 4 ...
    #  0  x x x x x
    #  1  0 x x x x
    #  2  1 2 x x x
    #  3  3 4 5 x x
    #  4  6 7 8 9 x
    # ...
    tril_idcs = torch.tril_indices(
        max_L_bs, max_L_bs, offset=-1, device=device
    )  # [2, max(P_bs)]
    idcs_elements = tril_idcs[:, idcs_pairs]  # [2, B, max(P_bs)]
    idcs_elements = idcs_elements.movedim(0, 2)  # [B, max(P_bs), 2]

    # Apply padding if requested.
    if padding_value is not None:
        replace_padding(
            idcs_elements, P_bs, padding_value=padding_value, in_place=True
        )

    return idcs_elements


# ######################### BASIC TENSOR MANIPULATION ##########################


def arange_batched(
    starts: torch.Tensor,
    stops: torch.Tensor | None = None,
    steps: torch.Tensor | None = None,
    padding_value: Any = None,
    device: torch.device | str | int | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a batch of tensors with values in the range [start, stop).

    Args:
        starts: The start value for each tensor in the batch. If stops is None,
            the range is [0, start).
            Shape: [B]
        stops: The end value for each tensor in the batch. The interval is
            half-open, so this end value is not included.
            Shape: [B]
        steps: The step value for each tensor in the batch. If None, the step
            value is set to 1.
            Shape: [B]
        padding_value: The value to pad the values with. If None, the values
            are padded with random values. This is faster than padding with
            a specific value.
        device: The device of the output tensor.
        dtype: The data type of the output tensor.

    Returns:
        Tuple containing:
        - A batch of tensors with values in the range [start, stop).
            Padded with padding_value.
            Shape: [B, max(L_bs)]
        - The number of values of the arange sequences in the batch.
            Shape: [B]

    Examples:
    >>> starts = torch.tensor([0, 5, 2, 3])
    >>> stops = torch.tensor([3, 5, 8, -1])
    >>> steps = torch.tensor([1, 1, 3, -1])
    >>> aranges, L_bs = arange_batched(starts, stops, steps, padding_value=-1)
    >>> aranges
    tensor([[ 0,  1,  2, -1],
            [-1, -1, -1, -1],
            [ 2,  5, -1, -1],
            [ 3,  2,  1,  0]])
    >>> L_bs
    tensor([3, 0, 2, 4])
    """
    B = len(starts)
    device = device if device is not None else starts.device
    inferred_dtype = torch.promote_types(
        starts.dtype,
        torch.promote_types(
            stops.dtype if stops is not None else starts.dtype,
            steps.dtype if steps is not None else starts.dtype,
        ),
    )

    # Prepare the input tensors.
    if stops is None:
        stops = starts
        starts = torch.zeros(B, device=device, dtype=inferred_dtype)
    if steps is None:
        steps = torch.ones(B, device=device, dtype=inferred_dtype)

    # Compute the arange sequences in parallel.
    L_bs = torch.ceil((stops - starts) / steps).long()  # [B]
    max_L_bs = int(L_bs.max())
    aranges = (
        starts.unsqueeze(1)  # [B, 1]
        + torch.arange(
            max_L_bs, device=device, dtype=inferred_dtype
        )  # [max(L_bs)]
        * steps.unsqueeze(1)  # [B, 1]
    )  # [B, max(L_bs)]  # fmt: skip

    # Replace values that are out of bounds with the padding value.
    if padding_value is not None:
        replace_padding(
            aranges, L_bs, padding_value=padding_value, in_place=True
        )

    # Cast to the desired device and dtype.
    aranges = aranges.to(device)
    if dtype is not None:
        aranges = aranges.to(dtype)

    return aranges, L_bs


def arange_batched_packed(
    starts: torch.Tensor,
    stops: torch.Tensor | None = None,
    steps: torch.Tensor | None = None,
    device: torch.device | str | int | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Create a batch of tensors with values in the range [start, stop).

    Args:
        starts: The start value for each tensor in the batch. If stops is None,
            the range is [0, start).
            Shape: [B]
        stops: The end value for each tensor in the batch. The interval is
            half-open, so this end value is not included.
            Shape: [B]
        steps: The step value for each tensor in the batch. If None, the step
            value is set to 1.
            Shape: [B]
        device: The device of the output tensor.
        dtype: The data type of the output tensor.

    Returns:
        Tuple containing:
        - A batch of tensors with values in the range [start, stop).
            Shape: [L]
        - The number of values of the arange sequences in the batch.
            Shape: [B]
        - The maximum length of the arange sequences in the batch.

    Examples:
    >>> starts = torch.tensor([0, 5, 2, 3])
    >>> stops = torch.tensor([3, 5, 8, -1])
    >>> steps = torch.tensor([1, 1, 3, -1])
    >>> aranges, L_bs, max_L_bs = arange_batched_packed(starts, stops, steps)
    >>> aranges
    tensor([0, 1, 2, 2, 5, 3, 2, 1, 0])
    >>> L_bs
    tensor([3, 0, 2, 4])
    >>> max_L_bs
    4
    """
    B = len(starts)
    device = device if device is not None else starts.device
    inferred_dtype = torch.promote_types(
        starts.dtype,
        torch.promote_types(
            stops.dtype if stops is not None else starts.dtype,
            steps.dtype if steps is not None else starts.dtype,
        ),
    )

    # Prepare the input tensors.
    if stops is None:
        stops = starts
        starts = torch.zeros(B, device=device, dtype=inferred_dtype)
    if steps is None:
        steps = torch.ones(B, device=device, dtype=inferred_dtype)

    # Compute the starts and steps of the arange sequences in parallel.
    L_bs = torch.ceil((stops - starts) / steps).long()  # [B]
    max_L_bs = int(L_bs.max())
    starts_repeated = starts.repeat_interleave(L_bs)  # [L]
    steps_repeated = steps.repeat_interleave(L_bs)  # [L]

    # Compute the offsets for each arange sequence in parallel.
    L_bs_without_last = L_bs[:-1]  # [B - 1]
    transition_idcs = L_bs_without_last[L_bs_without_last != 0]  # [B']
    offsets_packed = torch.ones_like(steps_repeated)  # [L]
    offsets_packed[0] = 0
    offsets_packed[transition_idcs.cumsum(0)] -= transition_idcs  # [B']
    offsets_packed = offsets_packed.cumsum(0)  # [L]

    # Compute the arange sequences in parallel.
    offsets_packed *= steps_repeated  # [L]
    aranges = starts_repeated + offsets_packed  # [L]

    # Cast to the desired device and dtype.
    aranges = aranges.to(device)
    if dtype is not None:
        aranges = aranges.to(dtype)

    return aranges, L_bs, max_L_bs


def linspace_batched(
    starts: torch.Tensor,
    stops: torch.Tensor,
    nums: torch.Tensor,
    padding_value: Any = None,
    device: torch.device | str | int | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a batch of tensors with values in the range [start, stop].

    Args:
        starts: The start value for each tensor in the batch.
            Shape: [B]
        stops: The end value for each tensor in the batch.
            Shape: [B]
        nums: The number of samples to generate for each tensor in the batch. If
            the number of samples is 1, only the start value is returned.
            Shape: [B]
        padding_value: The value to pad the values with. If None, the values
            are padded with random values. This is faster than padding with
            a specific value.
        device: The device of the output tensor.
        dtype: The data type of the output tensor.

    Returns:
        Tuple containing:
        - A batch of tensors with values in the range [start, stop].
            Padded with padding_value.
            Shape: [B, max(L_bs)]
        - The number of values of the linspace sequences in the batch. This is
            the same as nums.
            Shape: [B]

    Examples:
    >>> starts = torch.tensor([0.0, 5.0, 3.0, 2.0, 3.0])
    >>> stops = torch.tensor([1.0, 6.0, 5.0, 8.0, -3.0])
    >>> nums = torch.tensor([5, 1, 0, 3, 4])
    >>> linspaces, L_bs = linspace_batched(
    ...     starts, stops, nums, padding_value=-1.0
    ... )
    >>> linspaces
    tensor([[ 0.0000,  0.2500,  0.5000,  0.7500,  1.0000],
            [ 5.0000, -1.0000, -1.0000, -1.0000, -1.0000],
            [-1.0000, -1.0000, -1.0000, -1.0000, -1.0000],
            [ 2.0000,  5.0000,  8.0000, -1.0000, -1.0000],
            [ 3.0000,  1.0000, -1.0000, -3.0000, -1.0000]])
    >>> L_bs
    tensor([5, 1, 0, 3, 4])
    """
    device = device if device is not None else starts.device
    inferred_dtype = torch.promote_types(
        starts.dtype, torch.promote_types(stops.dtype, nums.dtype)
    )

    # Compute the steps of the linspace sequences in parallel.
    L_bs = nums.long()  # [B]
    max_L_bs = int(L_bs.max())
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=RuntimeWarning
        )  # ignore division by zero since we already handle it in torch.where()
        steps = torch.where(L_bs != 1, (stops - starts) / (L_bs - 1), 0)  # [B]

    # Compute the linspace sequences in parallel.
    linspaces = (
        starts.unsqueeze(1)  # [B, 1]
        + torch.arange(
            max_L_bs, device=device, dtype=inferred_dtype
        )  # [max(L_bs)]
        * steps.unsqueeze(1)  # [B, 1]
    )  # [B, max(L_bs)]  # fmt: skip

    # Set the last element of each linspace to the stop value manually to avoid
    # floating point issues.
    nonzero_idcs = L_bs.nonzero(as_tuple=True)[0]  # [B_nonzero]
    L_bs_nonzero = L_bs[nonzero_idcs]  # [B_nonzero]
    if len(nonzero_idcs) != 0:
        stop_idcs = L_bs_nonzero - 1  # [B_nonzero]

        # Only set the stop values for sequences with at least two elements.
        is_atleasttwo = L_bs_nonzero != 1  # [B_nonzero]
        atleasttwo_idcs = nonzero_idcs[is_atleasttwo]  # [B_atleasttwo]
        stop_idcs = stop_idcs[is_atleasttwo]  # [B_atleasttwo]

        linspaces[atleasttwo_idcs, stop_idcs] = stops[atleasttwo_idcs].to(
            linspaces.dtype
        )

    # Replace values that are out of bounds with the padding value.
    if padding_value is not None:
        replace_padding(
            linspaces, L_bs, padding_value=padding_value, in_place=True
        )

    # Cast to the desired device and dtype.
    linspaces = linspaces.to(device)
    if dtype is not None:
        linspaces = linspaces.to(dtype)

    return linspaces, L_bs


def linspace_batched_packed(
    starts: torch.Tensor,
    stops: torch.Tensor,
    nums: torch.Tensor,
    device: torch.device | str | int | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Create a batch of tensors with values in the range [start, stop].

    Args:
        starts: The start value for each tensor in the batch.
            Shape: [B]
        stops: The end value for each tensor in the batch.
            Shape: [B]
        nums: The number of samples to generate for each tensor in the batch. If
            the number of samples is 1, only the start value is returned.
            Shape: [B]
        device: The device of the output tensor.
        dtype: The data type of the output tensor.

    Returns:
        Tuple containing:
        - A batch of tensors with values in the range [start, stop].
            Padded with padding_value.
            Shape: [B, max(L_bs)]
        - The number of values of the linspace sequences in the batch. This is
            the same as nums.
            Shape: [B]
        - The maximum length of the linspace sequences in the batch.

    Examples:
    >>> starts = torch.tensor([0.0, 5.0, 3.0, 2.0, 3.0])
    >>> stops = torch.tensor([1.0, 6.0, 5.0, 8.0, -3.0])
    >>> nums = torch.tensor([5, 1, 0, 3, 4])
    >>> linspaces, L_bs, max_L_bs = linspace_batched_packed(starts, stops, nums)
    >>> linspaces
    tensor([ 0.0000,  0.2500,  0.5000,  0.7500,  1.0000,  5.0000,  2.0000,
             5.0000,  8.0000,  3.0000,  1.0000, -1.0000, -3.0000])
    >>> L_bs
    tensor([5, 1, 0, 3, 4])
    >>> max_L_bs
    5
    """
    device = device if device is not None else starts.device

    # Compute the steps of the linspace sequences in parallel.
    L_bs = nums.long()  # [B]
    max_L_bs = int(L_bs.max())
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=RuntimeWarning
        )  # ignore division by zero since we already handle it in torch.where()
        steps = torch.where(L_bs != 1, (stops - starts) / (L_bs - 1), 0)  # [B]

    # Compute the starts and steps of the linspace sequences in parallel.
    starts_repeated = starts.repeat_interleave(L_bs)  # [L]
    steps_repeated = steps.repeat_interleave(L_bs)  # [L]

    # Compute the offsets for each linspace sequence in parallel.
    L_bs_without_last = L_bs[:-1]  # [B - 1]
    transition_idcs = L_bs_without_last[L_bs_without_last != 0]  # [B']
    offsets_packed = torch.ones_like(steps_repeated)  # [L]
    offsets_packed[0] = 0
    offsets_packed[transition_idcs.cumsum(0)] -= transition_idcs  # [B']
    offsets_packed = offsets_packed.cumsum(0)  # [L]

    # Compute the linspace sequences in parallel.
    offsets_packed *= steps_repeated  # [L]
    linspaces = starts_repeated + offsets_packed  # [L]

    # Set the last element of each linspace to the stop value manually to avoid
    # floating point issues.
    nonzero_idcs = L_bs.nonzero(as_tuple=True)[0]  # [B_nonzero]
    L_bs_nonzero = L_bs[nonzero_idcs]  # [B_nonzero]
    if len(nonzero_idcs) != 0:
        stop_idcs = L_bs_nonzero.cumsum(0) - 1  # [B_nonzero]

        # Only set the stop values for sequences with at least two elements.
        is_atleasttwo = L_bs_nonzero != 1  # [B_nonzero]
        atleasttwo_idcs = nonzero_idcs[is_atleasttwo]  # [B_atleasttwo]
        stop_idcs = stop_idcs[is_atleasttwo]  # [B_atleasttwo]

        linspaces[stop_idcs] = stops[atleasttwo_idcs].to(linspaces.dtype)

    # Cast to the desired device and dtype.
    linspaces = linspaces.to(device)
    if dtype is not None:
        linspaces = linspaces.to(dtype)

    return linspaces, L_bs, max_L_bs


def index_select_batched(
    values: torch.Tensor, dim: int, indices: torch.Tensor
) -> torch.Tensor:
    """Select values from a batch of tensors using the given indices.

    Note that dim refers to the index of the dimension to sort along AFTER the
    batch dimension.

    Args:
        values: The values to select from.
            Shape: [B, N_0, ..., N_dim, ..., N_{D-1}]
        dim: The dimension to select along.
        indices: The indices to select.
            Shape: [B, N_select]

    Returns:
        The selected values.
            Shape: [B, N_0, ..., N_{dim-1}, N_select, N_{dim+1}, ..., N_{D-1}]

    Examples:
    >>> values = torch.tensor([
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
    >>> indices = torch.tensor([
    ...     [2, 0],
    ...     [1, 3],
    ... ])
    >>> selected_values = index_select_batched(values, 0, indices)
    >>> selected_values
    tensor([[[ 7,  8,  9],
             [ 1,  2,  3]],
            [[16, 17, 18],
             [22, 23, 24]]])
    """
    unsqueezed_shape = [1] * values.ndim
    unsqueezed_shape[0] = indices.shape[0]
    unsqueezed_shape[dim + 1] = indices.shape[1]
    indices_unsqueezed = indices.reshape(unsqueezed_shape)
    expanded_shape = list(values.shape)
    expanded_shape[dim + 1] = indices.shape[1]
    indices_expanded = indices_unsqueezed.expand(expanded_shape)
    return values.gather(dim + 1, indices_expanded.long())


def __duplicate_subtensors(
    values: torch.Tensor, L_bs: torch.Tensor, reps_bs: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Duplicate subtensors according to the given reps.

    Args:
        values: Packed tensor with all subtensors concatenated.
            Shape: [sum(L_bs)]
        L_bs: Length of each subtensor.
            Shape: [B]
        reps_bs: Number of times to duplicate each subtensor.
            Shape: [B]

    Returns:
        Tuple containing:
        - Packed tensor with each subtensor duplicated.
            Shape: [sum(L_bs * reps_bs)]
        - The lengths of each duplicated subtensor.
            Shape: [B]

    Examples:
    >>> values = torch.tensor([1, 2, 3, 4, 5, 6])
    >>> L_bs = torch.tensor([2, 3, 1])
    >>> reps_bs = torch.tensor([2, 0, 3])
    >>> values_duplicated, L_bs_duplicated = __duplicate_subtensors(
    ...     values, L_bs, reps_bs
    ... )
    >>> values_duplicated
    tensor([1, 2, 1, 2, 6, 6, 6])
    >>> L_bs_duplicated
    tensor([4, 0, 3])
    """
    # The key insight is to construct an index tensor that repeats the
    # appropriate indices for each subtensor.

    # Compute the start and stop indices of each subtensor.
    stops = L_bs.cumsum(0, dtype=torch.int64)  # [B]
    starts = stops - L_bs  # [B]

    # Repeat the start and stop indices according to reps.
    starts = starts.repeat_interleave(reps_bs)  # [sum(reps_bs)]
    stops = stops.repeat_interleave(reps_bs)  # [sum(reps_bs)]

    # Create the index tensor for each subtensor.
    idcs, _, _ = arange_batched_packed(starts, stops)  # [sum(L_bs * reps_bs)]

    # Gather the duplicated values.
    values_duplicated = values[idcs]  # [sum(L_bs * reps_bs)]
    L_bs_duplicated = (L_bs * reps_bs).long()  # [B]

    return values_duplicated, L_bs_duplicated


def repeat_interleave_batched(
    values: torch.Tensor,
    L_bsds: torch.Tensor,
    repeats: torch.Tensor,
    dim: int = 0,
    padding_value: Any = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Repeat values from a batch of tensors using the given repeats.

    Note that dim refers to the index of the dimension to sort along AFTER the
    batch dimension.

    Args:
        values: The values to repeat.
            Shape: [B, max(L_bs0), ..., max(L_bs{dim}), ..., max(L_bs{D-1})]
        L_bsds: Length of each sample along each dimension.
            Shape: [B, D]
        repeats: Number of times to repeat each value.
            Shape: [B, max(L_bs{dim})]
        dim: The dimension to repeat along.
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
    >>> values = torch.tensor([
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
    >>> L_bsds = torch.tensor([
    ...     [2, 2],
    ...     [3, 1],
    ... ])
    >>> repeats = torch.tensor([
    ...     [1, 2, 0],
    ...     [3, 2, 1],
    ... ])
    >>> values_repeated, L_bsds_repeated = repeat_interleave_batched(
    ...     values, L_bsds, repeats, dim=0, padding_value=0
    ... )
    >>> values_repeated
    tensor([[[ 1,  2],
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
    tensor([[3, 2],
            [6, 1]])
    """
    max_L_bsds = torch.tensor(values.shape[1:])  # [D]

    # Compute the new lengths after repeating.
    L_bsds_repeated = L_bsds.to(torch.int64, copy=True)  # [B, D]
    L_bsds_repeated[:, dim] = sum_padding_batched(repeats, L_bsds[:, dim])
    max_L_bsds_repeated = max_L_bsds.clone()  # [D]
    max_L_bsds_repeated[dim] = int(L_bsds_repeated[:, dim].max())

    # Move dim to the front and merge it with the batch dimension.
    # This allows us to use torch.repeat_interleave() directly.
    values = values.movedim(
        dim + 1, 1
    )  # [B, max(L_bs{dim}), max(L_bs0), ..., max(L_bs{D-1})]
    values = pack_padded(
        values, L_bsds[:, dim]
    )  # [sum(L_bs{dim}), max(L_bs0), ..., max(L_bs{D-1})]
    repeats = pack_padded(repeats, L_bsds[:, dim])  # [sum(L_bs{dim})]

    # Repeat the values.
    values_repeated = values.repeat_interleave(
        repeats, dim=0
    )  # [sum(L_bs{dim}_repeated), max(L_bs0), ..., max(L_bs{D-1})]

    # Un-merge the batch and dim dimensions and move dim back to its original
    # position.
    values_repeated = pad_packed(
        values_repeated, L_bsds_repeated[:, dim], int(max_L_bsds_repeated[dim])
    )  # [B, max(L_bs{dim}_repeated), max(L_bs0), ..., max(L_bs{D-1})]
    values_repeated = values_repeated.movedim(
        1, dim + 1
    )  # [B, max(L_bs0), ..., max(L_bs{dim}_repeated), ..., max(L_bs{D-1})]

    # Apply padding if requested.
    if padding_value is not None:
        replace_padding_multidim(
            values_repeated,
            L_bsds_repeated,
            padding_value=padding_value,
            in_place=True,
        )

    return values_repeated, L_bsds_repeated


def repeat_interleave_batched_packed(
    values: torch.Tensor,
    L_bsds: torch.Tensor,
    repeats_bs: torch.Tensor,
    dim: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Repeat values from a batch of packed tensors using the given repeats.

    Args:
        values: Packed tensor of all samples concatenated.
            Shape: [sum(L_bs0 * ... * L_bs{D-1})]
        L_bsds: Length of each sample along each dimension.
            Shape: [B, D]
        repeats_bs: Number of times to repeat each value.
            Shape: [B, max(L_bs{dim})]
        dim: The dimension to repeat along.

    Returns:
        Tuple containing:
        - Packed tensor with each sample repeated.
            Shape: [sum(L_bs0 * ... * L_bs{dim}_repeated * ... * L_bs{D-1})]
        - The lengths of each repeated sample along each dimension.
            Shape: [B, D]

    Examples:
    >>> values = torch.tensor([1, 2, 3, 4, 7, 9, 11])
    >>> L_bsds = torch.tensor([
    ...     [2, 2],
    ...     [3, 1],
    ... ])
    >>> repeats_bs = torch.tensor([
    ...     [1, 2, 0],
    ...     [3, 2, 1],
    ... ])
    >>> values_repeated, L_bsds_repeated = repeat_interleave_batched_packed(
    ...     values, L_bsds, repeats_bs, dim=0
    ... )
    >>> values_repeated
    tensor([ 1,  2,  3,  4,  3,  4,  7,  7,  7,  9,  9, 11])
    >>> L_bsds_repeated
    tensor([[3, 2],
            [6, 1]])
    """
    # Compute the new lengths after repeating.
    L_bsds_repeated = L_bsds.to(torch.int64, copy=True)  # [B, D]
    L_bsds_repeated[:, dim] = sum_padding_batched(repeats_bs, L_bsds[:, dim])

    # Calculate product of all lengths for all dimensions except dim.
    # We denote the product from dimension d0 (included) to d1 (excluded) as:
    # _bs_d0_to_d1 = _bs{d0} * _bs{d0+1} * ... * _bs{d1-2} * _bs{d1-1}
    L_bs_0_to_dimplus1 = L_bsds[:, : dim + 1].prod(dim=1)  # [B]
    L_bs_0_to_dim = L_bs_0_to_dimplus1 // L_bsds[:, dim]  # [B]
    L_bs_dimplus1_to_D = L_bsds[:, dim:].prod(dim=1) // L_bsds[:, dim]  # [B]

    # Pretend we are working with 1D subtensors, and construct the corresponding
    # L_bs and repeats_bs.
    L_bs = L_bs_dimplus1_to_D.repeat_interleave(
        L_bs_0_to_dimplus1
    )  # [sum(L_bs_0_to_dimplus1)]
    repeats_bs, _ = __duplicate_subtensors(
        pack_padded(repeats_bs, L_bsds[:, dim]), L_bsds[:, dim], L_bs_0_to_dim
    )  # [sum(L_bs_0_to_dimplus1)], _

    # Duplicate the subtensors.
    values_repeated, _ = __duplicate_subtensors(
        values, L_bs, repeats_bs
    )  # [sum(L_bs0 * ... * L_bs{dim}_repeated * ... * L_bs{D-1})], _

    return values_repeated, L_bsds_repeated


def tile_batched(
    values: torch.Tensor,
    L_bsds: torch.Tensor,
    reps_bsds: torch.Tensor,
    padding_value: Any = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tile values from a batch of tensors using the given reps.

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
    >>> values = torch.tensor([
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
    >>> L_bsds = torch.tensor([
    ...     [2, 2],
    ...     [3, 1],
    ... ])
    >>> reps_bsds = torch.tensor([
    ...     [2, 3],
    ...     [1, 4],
    ... ])
    >>> values_tiled, L_bsds_tiled = tile_batched(
    ...     values, L_bsds, reps_bsds, padding_value=0
    ... )
    >>> values_tiled
    tensor([[[ 1,  2,  1,  2,  1,  2],
             [ 3,  4,  3,  4,  3,  4],
             [ 1,  2,  1,  2,  1,  2],
             [ 3,  4,  3,  4,  3,  4]],
            [[ 7,  7,  7,  7,  0,  0],
             [ 9,  9,  9,  9,  0,  0],
             [11, 11, 11, 11,  0,  0],
             [ 0,  0,  0,  0,  0,  0]]])
    >>> L_bsds_tiled
    tensor([[4, 6],
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
    )  # [sum(prod(L_bsds * reps_bsds, dim=1))], [B, D]
    max_L_bsds_tiled = L_bsds_tiled.amax(dim=0)  # [D]

    # Pad the tiled values.
    values_tiled = pad_packed_multidim(
        values_tiled_packed,
        L_bsds_tiled,
        max_L_bsds_tiled,
        padding_value=padding_value,
    )  # [B, max(L_bs0 * reps_bs0), ..., max(L_bs{D-1} * reps_bs{D-1})]

    return values_tiled, L_bsds_tiled


def tile_batched_packed(
    values: torch.Tensor, L_bsds: torch.Tensor, reps_bsds: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tile values from a batch of packed tensors using the given reps.

    Args:
        values: Packed tensor of all samples concatenated.
            Shape: [sum(L_bs0 * ... * L_bs{D-1})]
        L_bsds: Length of each sample along each dimension.
            Shape: [B, D]
        reps_bsds: Number of times to tile each sample along each dimension.
            Shape: [B, D]

    Returns:
        Tuple containing:
        - Packed tensor with each sample tiled.
            Shape: [sum(prod(L_bsds * reps_bsds, dim=1))]
        - The lengths of each tiled sample along each dimension.
            Shape: [B, D]

    Examples:
    >>> values = torch.tensor([1, 2, 3, 4, 7, 9, 11])
    >>> L_bsds = torch.tensor([
    ...     [2, 2],
    ...     [3, 1],
    ... ])
    >>> reps_bsds = torch.tensor([
    ...     [2, 3],
    ...     [1, 4],
    ... ])
    >>> values_tiled, L_bsds_tiled = tile_batched_packed(
    ...     values, L_bsds, reps_bsds
    ... )
    >>> values_tiled
    tensor([ 1,  2,  1,  2,  1,  2,  3,  4,  3,  4,  3,  4,  1,  2,  1,  2,  1,
             2,  3,  4,  3,  4,  3,  4,  7,  7,  7,  7,  9,  9,  9,  9, 11, 11,
            11, 11])
    >>> L_bsds_tiled
    tensor([[4, 6],
            [3, 4]])
    """
    # Compute the new lengths along each dimension.
    L_bsds_tiled = (L_bsds * reps_bsds).long()  # [B, D]

    # Calculate products of all lengths from dimension d0 to dimension d1.
    # We denote the product from dimension d0 (included) to d1 (excluded) as:
    # _bs_d0_to_d1 = _bs{d0} * _bs{d0+1} * ... * _bs{d1-2} * _bs{d1-1}
    L_bs_tiled_0_to_ds = L_bsds_tiled.cumprod(dim=1) // L_bsds_tiled  # [B, D]
    L_bs_d_to_Ds = L_bsds.flip(1).cumprod(dim=1).flip(1)  # [B, D]

    # Tile along each dimension iteratively. Only tile if the amount of reps
    # along that dimension is greater than 1.
    for d in (reps_bsds != 1).any(dim=0).nonzero(as_tuple=True)[0]:
        # Pretend we are working with 1D subtensors, and construct the
        # corresponding L_bs and reps_bs.
        L_bs = L_bs_d_to_Ds[:, d].repeat_interleave(
            L_bs_tiled_0_to_ds[:, d]
        )  # [sum(L_bs_tiled_0_to_d)]
        reps_bs = reps_bsds[:, d].repeat_interleave(
            L_bs_tiled_0_to_ds[:, d]
        )  # [sum(L_bs_tiled_0_to_d)]

        # Duplicate the subtensors.
        values, _ = __duplicate_subtensors(
            values, L_bs, reps_bs
        )  # [sum(prod(L_bs_tiled_0_to_{d+1}s * L_bs_{d+1}_to_Ds))], _

    return values, L_bsds_tiled


def expand_batched(
    values: torch.Tensor,
    L_bsds: torch.Tensor,
    shape_bsds: torch.Tensor,
    padding_value: Any = None,
) -> torch.Tensor:
    """Expand a batch of tensors to the given target shape.

    Args:
        values: The values to expand.
            Shape: [B, max(L_bs0), ..., max(L_bs{D-1})]
        L_bsds: Length of each sample along each dimension.
            Shape: [B, D]
        shape_bsds: The target shape to expand to for each sample.
            Shape: [B, D]
        padding_value: The value to pad the values with. If None, the values
            are padded with random values. This is faster than padding with
            a specific value.

    Returns:
        The expanded values.
            Shape: [B, max(shape_bs0), ..., max(shape_bs{D-1})]

    Examples:
    >>> values = torch.tensor([
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
    >>> L_bsds = torch.tensor([
    ...     [1, 2],
    ...     [3, 1],
    ... ])
    >>> shape_bsds = torch.tensor([
    ...     [2, 2],
    ...     [3, 4],
    ... ])
    >>> values_expanded = expand_batched(
    ...     values, L_bsds, shape_bsds, padding_value=0
    ... )
    >>> values_expanded
    tensor([[[ 1,  2,  0,  0],
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

    # Expand the values in their packed form.
    values_expanded_packed = expand_batched_packed(
        values_packed, L_bsds, shape_bsds
    )  # [sum(prod(shape_bsds, dim=1))]
    max_shape_bsds = shape_bsds.amax(dim=0)  # [D]

    # Pad the expanded values.
    values_expanded = pad_packed_multidim(
        values_expanded_packed,
        shape_bsds,
        max_shape_bsds,
        padding_value=padding_value,
    )  # [B, max(shape_bs0), ..., max(shape_bs{D-1})]

    return values_expanded


def expand_batched_packed(
    values: torch.Tensor, L_bsds: torch.Tensor, shape_bsds: torch.Tensor
) -> torch.Tensor:
    """Expand a batch of packed tensors to the given target shape.

    Warning: Unlike torch.expand(), this function does not support adding
    new dimensions of size 1 to the left of the shape. The number of dimensions
    D must be the same in L_bsds and shape_bsds.

    Args:
        values: Packed tensor of all samples concatenated.
            Shape: [sum(L_bs0 * ... * L_bs{D-1})]
        L_bsds: Length of each sample along each dimension.
            Shape: [B, D]
        shape_bsds: The target shape to expand to for each sample.
            Shape: [B, D]

    Returns:
        Packed tensor with each sample expanded to the target shape.
            Shape: [sum(prod(shape_bsds, dim=1))]

    Examples:
    >>> values = torch.tensor([1, 2, 7, 9, 11])
    >>> L_bsds = torch.tensor([
    ...     [1, 2],
    ...     [3, 1],
    ... ])
    >>> shape_bsds = torch.tensor([
    ...     [2, 2],
    ...     [3, 4],
    ... ])
    >>> values_expanded = expand_batched_packed(values, L_bsds, shape_bsds)
    >>> values_expanded
    tensor([ 1,  2,  1,  2,  7,  7,  7,  7,  9,  9,  9,  9, 11, 11, 11, 11])
    """
    # For the values to be expandable to the new shape, each dimension must
    # either be equal or the original dimension must be 1.
    if not ((L_bsds == shape_bsds) | (L_bsds == 1)).all():
        raise ValueError(
            "L_bsds and shape_bsds must be expandable. Each dimension must"
            " either be equal or the original dimension must be 1."
        )

    # Compute the number of reps along each dimension.
    reps_bsds = torch.where(L_bsds != shape_bsds, shape_bsds, 1)  # [B, D]

    # Calculate products of all lengths from dimension d0 to dimension d1.
    # We denote the product from dimension d0 (included) to d1 (excluded) as:
    # _bs_d0_to_d1 = _bs{d0} * _bs{d0+1} * ... * _bs{d1-2} * _bs{d1-1}
    shape_bs_0_to_ds = shape_bsds.cumprod(dim=1) // shape_bsds  # [B, D]
    L_bs_d_to_Ds = L_bsds.flip(1).cumprod(dim=1).flip(1)  # [B, D]

    # Expand along each dimension iteratively. Only expand if the amount
    # of reps along that dimension is greater than 1.
    for d in (reps_bsds != 1).any(dim=0).nonzero(as_tuple=True)[0]:
        # Pretend we are working with 1D subtensors, and construct the
        # corresponding L_bs and reps_bs.
        L_bs = L_bs_d_to_Ds[:, d].repeat_interleave(
            shape_bs_0_to_ds[:, d]
        )  # [sum(shape_bs_0_to_d)]
        reps_bs = reps_bsds[:, d].repeat_interleave(
            shape_bs_0_to_ds[:, d]
        )  # [sum(shape_bs_0_to_d)]

        # Duplicate the subtensors.
        values, _ = __duplicate_subtensors(
            values, L_bs, reps_bs
        )  # [sum(prod(shape_bs_0_to_{d+1}s * L_bs_{d+1}_to_Ds))], _

    return values


def meshgrid_batched(
    *xi: tuple[torch.Tensor, torch.Tensor],
    indexing: str = "xy",
    padding_value: Any = None,
) -> tuple[torch.Tensor, ...]:
    """Create a meshgrid from a batch of tensors.

    This function is like torch.meshgrid(), but batched.

    Args:
        *xi: List of tuples containing:
            - Padded input tensors representing the coordinates of a grid.
                Padding could be arbitrary.
                Shape: [B, max(L_bsd)]
            - The lengths of the input tensors.
                Shape: [B]
            Length: D
        indexing: The indexing convention used. 'ij' returns a meshgrid with
            matrix indexing, while 'xy' returns a meshgrid with Cartesian
            indexing.
        padding_value: The value to pad the outputs with. If None, the outputs
            are padded with random values. This is faster than padding with a
            specific value.

    Returns:
        Tuple of [B, max(L_bs0), ..., max(L_bs{D-1})] shaped tensors if indexing
        is 'ij' or tuple of [B, max(L_bs1), max(L_bs0), ..., max(L_bs{D-1})]
        shaped tensors if indexing is 'xy'. Each output tensor contains the
        D-dimensional meshgrid formed by the input tensors. Padded with
        padding_value.

    Examples:
    >>> x0 = torch.tensor([
    ...     [1, 2],
    ...     [3, 0],
    ... ])
    >>> L_b0 = torch.tensor([2, 1])
    >>> x1 = torch.tensor([
    ...     [10, 20, 30, 0],
    ...     [50, 60, 70, 80],
    ... ])
    >>> L_b1 = torch.tensor([3, 4])

    >>> meshgrid_0, meshgrid_1 = meshgrid_batched(
    ...     (x0, L_b0), (x1, L_b1), indexing="ij", padding_value=-1
    ... )
    >>> meshgrid_0
    tensor([[[ 1,  1,  1, -1],
             [ 2,  2,  2, -1]],
            [[ 3,  3,  3,  3],
             [-1, -1, -1, -1]]])
    >>> meshgrid_1
    tensor([[[10, 20, 30, -1],
             [10, 20, 30, -1]],
            [[50, 60, 70, 80],
             [-1, -1, -1, -1]]])

    >>> meshgrid_0, meshgrid_1 = meshgrid_batched(
    ...     (x0, L_b0), (x1, L_b1), indexing="xy", padding_value=-1
    ... )
    >>> meshgrid_0
    tensor([[[ 1,  2],
             [ 1,  2],
             [ 1,  2],
             [-1, -1]],
            [[ 3, -1],
             [ 3, -1],
             [ 3, -1],
             [ 3, -1]]])
    >>> meshgrid_1
    tensor([[[10, 10],
             [20, 20],
             [30, 30],
             [-1, -1]],
            [[50, -1],
             [60, -1],
             [70, -1],
             [80, -1]]])
    """
    xs, L_bsds = map(list, zip(*xi))  # D x [B, max(L_bsd)], D x [B]
    L_bsds = torch.stack(L_bsds, dim=1)  # [B, D]
    max_L_bsds = torch.tensor([x.shape[1] for x in xs])  # [D]

    B, D = L_bsds.shape

    # Prepare the shape of the output tensors.
    broadcasted_shape = [B, *max_L_bsds]
    broadcasted_bsds = L_bsds.clone()

    # Swap the first two axes if indexing is 'xy'.
    if indexing == "xy" and D >= 2:
        broadcasted_shape[1], broadcasted_shape[2] = (
            broadcasted_shape[2],
            broadcasted_shape[1],
        )
        broadcasted_bsds[:, 0], broadcasted_bsds[:, 1] = (
            broadcasted_bsds[:, 1],
            broadcasted_bsds[:, 0].clone(),
        )

    # Go through each input tensor and create the corresponding meshgrid.
    meshgrids = []
    for d in range(D):
        # Prepare the shape of the sparse output tensors.
        unsqueezed_shape = [B, *[1] * D]
        unsqueezed_shape[d + 1] = max_L_bsds[d]  # type: ignore

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
        meshgrid = meshgrid.expand(
            broadcasted_shape
        ).clone()  # [B, max(L_bs0), ..., max(L_bs{D-1})]

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
    *xi: tuple[torch.Tensor, torch.Tensor], indexing: str = "xy"
) -> tuple[torch.Tensor, ...]:
    """Create a meshgrid from a batch of packed tensors.

    Like torch.meshgrid(), but batched and packed.

    Args:
        *xi: List of tuples containing:
            - Packed input tensors representing the coordinates of a grid.
                Shape: [L_d]
            - The lengths of the input tensors.
                Shape: [B]
            Length: D
        indexing: The indexing convention used. 'ij' returns a meshgrid with
            matrix indexing, while 'xy' returns a meshgrid with Cartesian
            indexing.

    Returns:
        Tuple of [sum(L_bs0 * ... * L_bs{D-1})] shaped tensors if indexing is
        'ij' or tuple of [sum(L_bs1 * L_bs0 * ... * L_bs{D-1})] shaped tensors
        if indexing is 'xy'. Each output tensor contains the D-dimensional
        meshgrid formed by the input tensors.

    Examples:
    >>> x0 = torch.tensor([1, 2, 3])
    >>> L_b0 = torch.tensor([2, 1])
    >>> x1 = torch.tensor([10, 20, 30, 50, 60, 70, 80])
    >>> L_b1 = torch.tensor([3, 4])

    >>> meshgrid_0, meshgrid_1 = meshgrid_batched_packed(
    ...     (x0, L_b0), (x1, L_b1), indexing="ij",
    ... )
    >>> meshgrid_0
    tensor([1, 1, 1, 2, 2, 2, 3, 3, 3, 3])
    >>> meshgrid_1
    tensor([10, 20, 30, 10, 20, 30, 50, 60, 70, 80])

    >>> meshgrid_0, meshgrid_1 = meshgrid_batched_packed(
    ...     (x0, L_b0), (x1, L_b1), indexing="xy",
    ... )
    >>> meshgrid_0
    tensor([1, 2, 1, 2, 1, 2, 3, 3, 3, 3])
    >>> meshgrid_1
    tensor([10, 10, 20, 20, 30, 30, 50, 60, 70, 80])
    """
    xs, L_bsds = map(list, zip(*xi))  # D x [L_d], D x [B]
    L_bsds = torch.stack(L_bsds, dim=1)  # [B, D]

    B, D = L_bsds.shape
    device = L_bsds.device

    # Prepare the shape of the output tensors.
    broadcasted_bsds = L_bsds.clone()

    # Swap the first two axes if indexing is 'xy'.
    if indexing == "xy" and D >= 2:
        broadcasted_bsds[:, 0], broadcasted_bsds[:, 1] = (
            broadcasted_bsds[:, 1],
            broadcasted_bsds[:, 0].clone(),
        )

    # Go through each input tensor and create the corresponding meshgrid.
    meshgrids = []
    for d in range(D):
        # Prepare the shape of the sparse output tensors.
        unsqueezed_bsds = torch.ones((B, D), dtype=torch.int64, device=device)
        unsqueezed_bsds[:, d] = L_bsds[:, d]

        # Swap the first two axes if indexing is 'xy'.
        if indexing == "xy" and D >= 2 and d <= 1:
            unsqueezed_bsds[:, 0], unsqueezed_bsds[:, 1] = (
                unsqueezed_bsds[:, 1],
                unsqueezed_bsds[:, 0].clone(),
            )

        # Create the meshgrid.
        meshgrid = xs[d]  # [L_d] = [sum(1 * ... * L_bsd * ... * 1)]

        # Broadcast the meshgrid to the full shape.
        meshgrid = expand_batched_packed(
            meshgrid, unsqueezed_bsds, broadcasted_bsds
        )  # [sum(L_bs0 * ... * L_bs{D-1})]

        meshgrids.append(meshgrid)

    return tuple(meshgrids)


# ######################## ADVANCED TENSOR MANIPULATION ########################


def swap_idcs_vals_batched(x: torch.Tensor) -> torch.Tensor:
    """Swap the indices and values of a batch of 1D tensors.

    Each row in the input tensor is assumed to contain exactly all integers from
    0 to N - 1, in any order.

    Warning: This function does not explicitly check if the input tensor
    contains no duplicates. If x contains duplicates, the behavior is
    non-deterministic (one of the values from x will be picked arbitrarily).

    Args:
        x: The tensor to swap.
            Shape: [B, N]

    Returns:
        The swapped tensor.
            Shape: [B, N]

    Examples:
    >>> x = torch.tensor([
    ...     [2, 3, 0, 4, 1],
    ...     [1, 3, 2, 0, 4],
    ... ])
    >>> swap_idcs_vals_batched(x)
    tensor([[2, 4, 0, 1, 3],
            [3, 0, 2, 1, 4]])
    """
    if x.ndim != 2:
        raise ValueError("x must be a batch of 1D tensors.")

    B, N = x.shape
    device = x.device
    dtype = x.dtype
    x_swapped = torch.empty_like(x)
    x_swapped[torch.arange(B, device=device, dtype=dtype).unsqueeze(1), x] = (
        torch.arange(N, device=device, dtype=dtype).unsqueeze(0)
    )
    return x_swapped


def swap_idcs_vals_duplicates_batched(
    x: torch.Tensor, stable: bool = False
) -> torch.Tensor:
    """Swap the indices and values of a batch of 1D tensors allowing duplicates.

    Each row in the input tensor is assumed to contain integers from 0 to
    M <= N, in any order, and may contain duplicates.

    Each row in the output tensor will contain exactly all integers from 0 to
    len(x) - 1, in any order.

    If the input doesn't contain duplicates, you should use
    swap_idcs_vals_batched() instead since it is faster (especially for large
    tensors).

    Args:
        x: The tensor to swap.
            Shape: [B, N]
        stable: Whether to preserve the relative order of equal elements. If
            False (default), an unstable sort is used, which is faster.

    Returns:
        The swapped tensor.
            Shape: [B, N]

    Examples:
    >>> x = torch.tensor([
    ...     [1, 3, 0, 1, 3],
    ...     [5, 3, 3, 5, 2],
    ... ])
    >>> swap_idcs_vals_duplicates_batched(x, stable=True)
    tensor([[2, 0, 3, 1, 4],
            [4, 1, 2, 0, 3]])
    """
    if x.ndim != 2:
        raise ValueError("x must be a batch of 1D tensors.")

    dtype = x.dtype

    # For some reason, this O(n log n) algorithm is actually faster than a
    # native implementation that uses a Python for loop with complexity O(n).
    return x.argsort(dim=1, stable=stable).to(dtype)


# ############################ CONSECUTIVE SEGMENTS ############################


def starts_segments_batched(
    x: torch.Tensor, dim: int = 0, padding_value: Any = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Find the start index of each consecutive segment in each batch tensor.

    Note that dim refers to the index of the dimension to sort along AFTER the
    batch dimension.

    Args:
        x: The input tensor. Consecutive equal values will be grouped.
            Shape: [B, N_0, ..., N_dim, ..., N_{D-1}]
        dim: The dimension along which the segments are lined up.
        padding_value: The value to pad the start indices with. If None, the
            start indices are padded with random values. This is faster than
            padding with a specific value.

    Returns:
        Tuple containing:
        - The start indices for each consecutive segment in x. Padded with
            padding_value.
            Shape: [B, max(S_bs)]
        - The number of consecutive segments in each tensor.
            Shape: [B]

    Examples:
    >>> x = torch.tensor([
    ...     [4, 4, 4, 2, 2, 8, 3, 3, 3, 3],
    ...     [1, 1, 0, 0, 0, 0, 0, 2, 2, 2],
    ... ])
    >>> starts, S_bs = starts_segments_batched(x, padding_value=0)
    >>> starts
    tensor([[0, 3, 5, 6],
            [0, 2, 7, 0]])
    >>> S_bs
    tensor([4, 3])
    """
    B, N_dim = x.shape[0], x.shape[dim + 1]
    device = x.device

    # Find the indices where the values change.
    is_change = (
        torch.concat(
            [
                torch.ones((B, 1), device=device, dtype=torch.bool),
                (
                    x.index_select(
                        dim + 1, torch.arange(0, N_dim - 1, device=device)
                    )  # [B, N_0, ..., N_dim - 1, ..., N_{D-1}]
                    != x.index_select(
                        dim + 1, torch.arange(1, N_dim, device=device)
                    )  # [B, N_0, ..., N_dim - 1, ..., N_{D-1}]
                ).any(
                    dim=tuple(
                        i for i in range(x.ndim) if i != dim + 1 and i != 0
                    )
                ),  # [B, N_dim - 1]
            ],
            dim=1,
        )  # [B, N_dim]
        if N_dim > 0
        else torch.empty((B, 0), device=device, dtype=torch.bool)
    )  # [B, N_dim]

    # Find the start of each consecutive segment.
    batch_idcs, starts_idcs = is_change.nonzero(as_tuple=True)  # [S], [S]

    # Convert to padded representation.
    S_bs = counts_segments(batch_idcs)  # [B]
    max_S_bs = int(S_bs.max())
    starts = pad_packed(
        starts_idcs, S_bs, max_S_bs, padding_value=padding_value
    )  # [B, max(S_bs)]

    return starts, S_bs


@overload
def counts_segments_batched(  # type: ignore
    x: torch.Tensor,
    dim: int = 0,
    return_starts: Literal[False] = ...,
    padding_value: Any = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    pass


@overload
def counts_segments_batched(
    x: torch.Tensor,
    dim: int = 0,
    return_starts: Literal[True] = ...,
    padding_value: Any = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pass


def counts_segments_batched(
    x: torch.Tensor,
    dim: int = 0,
    return_starts: bool = False,
    padding_value: Any = None,
) -> (
    tuple[torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
):
    """Count the length of each consecutive segment in each batch tensor.

    Note that dim refers to the index of the dimension to sort along AFTER the
    batch dimension.

    Args:
        x: The input tensor. Consecutive equal values will be grouped.
            Shape: [B, N_0, ..., N_dim, ..., N_{D-1}]
        dim: The dimension along which the segments are lined up.
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
        - The number of consecutive segments in each tensor.
            Shape: [B]
        - (Optional) If return_starts is True, the start indices for each
            consecutive segment in x. Padded with padding_value.
            Shape: [B, max(S_bs)]

    Examples:
    >>> x = torch.tensor([
    ...     [4, 4, 4, 2, 2, 8, 3, 3, 3, 3],
    ...     [1, 1, 0, 0, 0, 0, 0, 2, 2, 2],
    ... ])
    >>> counts, S_bs = counts_segments_batched(x, padding_value=0)
    >>> counts
    tensor([[3, 2, 1, 4],
            [2, 5, 3, 0]])
    >>> S_bs
    tensor([4, 3])
    """
    B, N_dim = x.shape[0], x.shape[dim + 1]
    device = x.device

    # Find the start of each consecutive segment.
    starts, S_bs = starts_segments_batched(
        x, dim=dim, padding_value=padding_value
    )  # [B, max(S_bs)], [B]

    # Prepare starts for count calculation.
    starts_with_N_dim = torch.concat(
        [starts, torch.full((B, 1), N_dim, device=device)], dim=1
    )  # [B, max(S_bs) + 1]
    starts_with_N_dim[torch.arange(B, device=device), S_bs] = N_dim

    # Find the count of each consecutive segment.
    counts = (
        starts_with_N_dim.diff(dim=1)  # [B, max(S_bs)]
        if N_dim > 0
        else torch.empty((B, 0), device=device, dtype=torch.int64)
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
    x: torch.Tensor,
    dim: int = 0,
    return_counts: Literal[False] = ...,
    return_starts: Literal[False] = ...,
    padding_value: Any = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    pass


@overload
def outer_indices_segments_batched(
    x: torch.Tensor,
    dim: int = 0,
    return_counts: Literal[True] = ...,
    return_starts: Literal[False] = ...,
    padding_value: Any = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pass


@overload
def outer_indices_segments_batched(
    x: torch.Tensor,
    dim: int = 0,
    return_counts: Literal[False] = ...,
    return_starts: Literal[True] = ...,
    padding_value: Any = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pass


@overload
def outer_indices_segments_batched(
    x: torch.Tensor,
    dim: int = 0,
    return_counts: Literal[True] = ...,
    return_starts: Literal[True] = ...,
    padding_value: Any = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pass


def outer_indices_segments_batched(
    x: torch.Tensor,
    dim: int = 0,
    return_counts: bool = False,
    return_starts: bool = False,
    padding_value: Any = None,
) -> (
    tuple[torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
):
    """Get the outer indices for each consecutive segment in each batch tensor.

    Args:
        x: The input tensor. Consecutive equal values will be grouped.
            Shape: [B, N_0, ..., N_dim, ..., N_{D-1}]
        dim: The dimension along which the segments are lined up.
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
            Shape: [B, N_dim]
        - The number of consecutive segments in each tensor.
            Shape: [B]
        - (Optional) If return_counts is True, the counts for each consecutive
            segment in x. Padded with padding_value.
            Shape: [B, max(S_bs)]
        - (Optional) If return_starts is True, the start indices for each
            consecutive segment in x. Padded with padding_value.
            Shape: [B, max(S_bs)]

    Examples:
    >>> x = torch.tensor([
    ...     [4, 4, 4, 2, 2, 8, 3, 3, 3, 3],
    ...     [1, 1, 0, 0, 0, 0, 0, 2, 2, 2],
    ... ])
    >>> outer_idcs, S_bs = outer_indices_segments_batched(x)
    >>> outer_idcs
    tensor([[0, 0, 0, 1, 1, 2, 3, 3, 3, 3],
            [0, 0, 1, 1, 1, 1, 1, 2, 2, 2]])
    >>> S_bs
    tensor([4, 3])
    """
    # Find the start (optional) and count of each consecutive segment.
    if return_starts:
        counts, S_bs, starts = counts_segments_batched(
            x, dim=dim, return_starts=True, padding_value=padding_value
        )  # [B, max(S_bs)], [B], [B, max(S_bs)]
    else:
        counts, S_bs = counts_segments_batched(
            x, dim=dim, padding_value=padding_value
        )  # [B, max(S_bs)], [B]

    # Calculate the outer indices.
    outer_idcs, _ = repeat_interleave_batched(
        torch.arange(counts.shape[1], dtype=torch.int64)
        .unsqueeze(0)
        .expand(counts.shape),  # [B, max(S_bs)]
        S_bs.unsqueeze(1),  # [B, 1]
        counts,
        dim=0,
    )  # [B, N_dim], _

    if return_counts and return_starts:
        return outer_idcs, S_bs, counts, starts  # type: ignore
    if return_counts:
        return outer_idcs, S_bs, counts
    if return_starts:
        return outer_idcs, starts  # type: ignore
    return outer_idcs, S_bs


@overload
def inner_indices_segments_batched(  # type: ignore
    x: torch.Tensor,
    dim: int = 0,
    return_counts: Literal[False] = ...,
    return_starts: Literal[False] = ...,
    padding_value: Any = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    pass


@overload
def inner_indices_segments_batched(
    x: torch.Tensor,
    dim: int = 0,
    return_counts: Literal[True] = ...,
    return_starts: Literal[False] = ...,
    padding_value: Any = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pass


@overload
def inner_indices_segments_batched(
    x: torch.Tensor,
    dim: int = 0,
    return_counts: Literal[False] = ...,
    return_starts: Literal[True] = ...,
    padding_value: Any = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pass


@overload
def inner_indices_segments_batched(
    x: torch.Tensor,
    dim: int = 0,
    return_counts: Literal[True] = ...,
    return_starts: Literal[True] = ...,
    padding_value: Any = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pass


def inner_indices_segments_batched(
    x: torch.Tensor,
    dim: int = 0,
    return_counts: bool = False,
    return_starts: bool = False,
    padding_value: Any = None,
) -> (
    tuple[torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
):
    """Get the inner indices for each consecutive segment in each batch tensor.

    Note that dim refers to the index of the dimension to sort along AFTER the
    batch dimension.

    Args:
        x: The input tensor. Consecutive equal values will be grouped.
            Shape: [B, N_0, ..., N_dim, ..., N_{D-1}]
        dim: The dimension along which the segments are lined up.
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
            Shape: [B, N_dim]
        - The number of consecutive segments in each tensor.
            Shape: [B]
        - (Optional) If return_counts is True, the counts for each consecutive
            segment in x. Padded with padding_value.
            Shape: [B, max(S_bs)]
        - (Optional) If return_starts is True, the start indices for each
            consecutive segment in x. Padded with padding_value.
            Shape: [B, max(S_bs)]

    Examples:
    >>> x = torch.tensor([
    ...     [4, 4, 4, 2, 2, 8, 3, 3, 3, 3],
    ...     [1, 1, 0, 0, 0, 0, 0, 2, 2, 2],
    ... ])
    >>> inner_idcs, S_bs = inner_indices_segments_batched(x)
    >>> inner_idcs
    tensor([[0, 1, 2, 0, 1, 0, 0, 1, 2, 3],
            [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]])
    >>> S_bs
    tensor([4, 3])
    """
    N_dim = x.shape[dim + 1]
    device = x.device

    # Find the start and count of each consecutive segment.
    counts, S_bs, starts = counts_segments_batched(
        x, dim=dim, return_starts=True, padding_value=padding_value
    )  # [B, max(S_bs)], [B], [B, max(S_bs)]

    # Calculate the inner indices.
    inner_idcs = (
        torch.arange(N_dim, device=device).unsqueeze(0)  # [1, N_dim]
        - repeat_interleave_batched(
            starts, S_bs.unsqueeze(1), counts, dim=0
        )[0]  # [B, N_dim]
    )  # [B, N_dim]  # fmt: skip

    if return_counts and return_starts:
        return inner_idcs, S_bs, counts, starts
    if return_counts:
        return inner_idcs, S_bs, counts
    if return_starts:
        return inner_idcs, starts
    return inner_idcs, S_bs


# ################################## LEXSORT ###################################


def lexsort_along_batched(
    x: torch.Tensor, dim: int = -1, stable: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sort a batched tensor along dim, taking all others as constant tuples.

    Note that dim refers to the index of the dimension to sort along AFTER the
    batch dimension.

    This is like a batched version of torch.sort(), but it doesn't sort along
    the other dimensions. As such, the other dimensions are treated as tuples.
    This function is roughly equivalent to the following Python code, but it
    is much faster.
    >>> torch.stack([  # doctest: +SKIP
    ...     torch.stack(
    ...         sorted(
    ...             x_b.unbind(dim),
    ...             key=tuple,
    ...         ),
    ...         dim=dim,
    ...     )
    ...     for x_b in x.unbind(0)
    ... ])

    Args:
        x: The input tensor.
            Shape: [B, N_0, ..., N_dim, ..., N_{D-1}]
        dim: The dimension to sort along.
        stable: Whether to preserve the relative order of equal elements. If
            False (default), an unstable sort is used, which is faster.

    Returns:
        Tuple containing:
        - Sorted version of x.
            Shape: [B, N_0, ..., N_dim, ..., N_{D-1}]
        - The backmap tensor, which contains the indices of the sorted values
            in the original input.
            The sorted version of x can be retrieved as follows:
            x_sorted = index_select_batched(x, dim, backmap)
            Shape: [B, N_dim]

    Examples:
    >>> x = torch.tensor([
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
    >>> dim = 0

    >>> x_sorted, backmap = lexsort_along_batched(x, dim=dim)
    >>> x_sorted
    tensor([[[1, 2],
             [1, 3],
             [2, 1],
             [3, 0]],
            [[1, 2],
             [1, 5],
             [2, 1],
             [3, 4]]])
    >>> backmap
    tensor([[2, 3, 0, 1],
            [0, 1, 3, 2]])

    >>> # Get the lexicographically sorted version of x:
    >>> index_select_batched(x, dim, backmap)
    tensor([[[1, 2],
             [1, 3],
             [2, 1],
             [3, 0]],
            [[1, 2],
             [1, 5],
             [2, 1],
             [3, 4]]])
    """
    # See the non-batched version for an explanation of the algorithm.
    B, N_dim = x.shape[0], x.shape[dim + 1]

    if x.ndim == 2:
        y = x.unsqueeze(1)  # [B, 1, N_dim]
    else:
        y = x.movedim(
            dim + 1, -1
        )  # [B, N_0, ..., N_{dim-1}, N_{dim+1}, ..., N_{D-1}, N_dim]
        y = y.reshape(
            B, -1, N_dim
        )  # [B, N_0 * ... * N_{dim-1} * N_{dim+1} * ... * N_{D-1}, N_dim]
    y = y.movedim(
        0, 1
    )  # [N_0 * ... * N_{dim-1} * N_{dim+1} * ... * N_{D-1}, B, N_dim]
    y = y.flip(
        dims=(0,)
    )  # [N_0 * ... * N_{dim-1} * N_{dim+1} * ... * N_{D-1}, B, N_dim]
    backmap = lexsort(y, dim=-1, stable=stable)  # [B, N_dim]

    # Sort the tensor along the given dimension.
    x_sorted = index_select_batched(
        x, dim, backmap
    )  # [B, N_0, ..., N_dim, ..., N_{D-1}]

    # Finally, we return the sorted tensor and the backmap.
    return x_sorted, backmap


# ################################### UNIQUE ###################################


@overload
def unique_consecutive_batched(  # type: ignore
    x: torch.Tensor,
    return_inverse: Literal[False] = ...,
    return_counts: Literal[False] = ...,
    dim: int | None = None,
    padding_value: Any = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    pass


@overload
def unique_consecutive_batched(
    x: torch.Tensor,
    return_inverse: Literal[True] = ...,
    return_counts: Literal[False] = ...,
    dim: int | None = None,
    padding_value: Any = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pass


@overload
def unique_consecutive_batched(
    x: torch.Tensor,
    return_inverse: Literal[False] = ...,
    return_counts: Literal[True] = ...,
    dim: int | None = None,
    padding_value: Any = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pass


@overload
def unique_consecutive_batched(
    x: torch.Tensor,
    return_inverse: Literal[True] = ...,
    return_counts: Literal[True] = ...,
    dim: int | None = None,
    padding_value: Any = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pass


def unique_consecutive_batched(
    x: torch.Tensor,
    return_inverse: bool = False,
    return_counts: bool = False,
    dim: int | None = None,
    padding_value: Any = None,
) -> (
    tuple[torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
):
    """A batched version of torch.unique_consecutive(), but WAY more effiecient.

    Note that dim refers to the index of the dimension to sort along AFTER the
    batch dimension.

    The returned unique elements are retrieved along the requested dimension,
    taking all the other dimensions apart from the batch dimension as constant
    tuples.

    Args:
        x: The input tensor. If it contains equal values, they must be
            consecutive along the given dimension.
            Shape: [B, N_0, ..., N_dim, ..., N_{D-1}]
        return_inverse: Whether to also return the inverse mapping tensor.
            This can be used to reconstruct the original tensor from the
            unique tensor.
        return_counts: Whether to also return the counts for each unique
            element.
        dim: The dimension to operate on. If None, the unique of the flattened
            input is returned. Otherwise, each of the tensors indexed by the
            given dimension is treated as one of the elements to apply the
            unique operation on. See examples for more details.
        padding_value: The value to pad the unique elements with. If None, the
            unique elements are padded with random values. This is faster than
            padding with a specific value.

    Returns:
        Tuple containing:
        - The unique elements. Padded with padding_value.
            Shape: [B, N_0, ..., N_{dim-1}, max(U_bs), N_{dim+1}, ..., N_{D-1}]
        - The amount of unique elements per batch element.
            Shape: [B]
        - (Optional) If return_inverse is True, the indices where elements
            in the original input ended up in the returned unique values.
            The original tensor can be reconstructed as follows:
            x_reconstructed = index_select_batched(uniques, dim, inverse)
            Shape: [B, N_dim]
        - (Optional) If return_counts is True, the counts for each unique
            element. Padded with padding_value.
            Shape: [B, max(U_bs)]

    Examples:
    >>> # 1D example: -----------------------------------------------------
    >>> x = torch.tensor([
    ...     [9, 9, 9, 9, 10, 10],
    ...     [8, 8, 7, 7, 9, 9],
    ... ])
    >>> dim = 0

    >>> uniques, U_bs, inverse, counts = unique_consecutive_batched(
    ...     x, return_inverse=True, return_counts=True, dim=dim, padding_value=0
    ... )
    >>> uniques
    tensor([[ 9, 10,  0],
            [ 8,  7,  9]])
    >>> U_bs
    tensor([2, 3])
    >>> inverse
    tensor([[0, 0, 0, 0, 1, 1],
            [0, 0, 1, 1, 2, 2]])
    >>> counts
    tensor([[4, 2, 0],
            [2, 2, 2]])

    >>> # Reconstruct the original tensor:
    >>> index_select_batched(uniques, dim, inverse)
    tensor([[ 9,  9,  9,  9, 10, 10],
            [ 8,  8,  7,  7,  9,  9]])

    >>> # 2D example: -----------------------------------------------------
    >>> x = torch.tensor([
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
    >>> dim = 1

    >>> uniques, U_bs, inverse, counts = unique_consecutive_batched(
    ...     x, return_inverse=True, return_counts=True, dim=dim, padding_value=0
    ... )
    >>> uniques
    tensor([[[ 7,  9, 10],
             [ 8, 10,  9],
             [ 9,  8,  7],
             [ 9,  7,  7]],
            [[ 7,  7,  0],
             [ 7, 10,  0],
             [ 9,  8,  0],
             [ 8,  8,  0]]])
    >>> U_bs
    tensor([3, 2])
    >>> inverse
    tensor([[0, 1, 1, 2],
            [0, 0, 0, 1]])
    >>> counts
    tensor([[1, 2, 1],
            [3, 1, 0]])

    >>> # Reconstruct the original tensor:
    >>> index_select_batched(uniques, dim, inverse)
    tensor([[[ 7,  9,  9, 10],
             [ 8, 10, 10,  9],
             [ 9,  8,  8,  7],
             [ 9,  7,  7,  7]],
            [[ 7,  7,  7,  7],
             [ 7,  7,  7, 10],
             [ 9,  9,  9,  8],
             [ 8,  8,  8,  8]]])
    """
    if dim is None:
        raise NotImplementedError(
            "dim=None is not implemented yet. Please specify a dimension"
            " explicitly."
        )

    # Find each consecutive segment.
    if return_inverse and return_counts:
        outer_idcs, U_bs, counts, starts = outer_indices_segments_batched(
            x,
            dim=dim,
            return_counts=True,
            return_starts=True,
            padding_value=padding_value,
        )  # [B, N_dim], [B], [B, max(U_bs)], [B, max(U_bs)]
    elif return_inverse:
        outer_idcs, U_bs, starts = outer_indices_segments_batched(
            x, dim=dim, return_starts=True, padding_value=padding_value
        )  # [B, N_dim], [B], [B, max(U_bs)]
    elif return_counts:
        counts, U_bs, starts = counts_segments_batched(
            x, dim=dim, return_starts=True, padding_value=padding_value
        )  # [B, max(U_bs)], [B], [B, max(U_bs)]
    else:
        starts, U_bs = starts_segments_batched(
            x, dim=dim
        )  # [B, max(U_bs)], [B]

    # Find the unique values.
    replace_padding(starts, U_bs, in_place=True)
    uniques = index_select_batched(
        x, dim, starts
    )  # [B, N_0, ..., N_{dim-1}, max(U_bs), N_{dim+1}..., N_{D-1}]

    # Replace the padding values if requested.
    if padding_value is not None:
        uniques = uniques.movedim(
            dim + 1, 1
        )  # [B, max(U_bs), N_0, ..., N_{dim-1}, N_{dim+1}, ..., N_{D-1}]
        replace_padding(
            uniques, U_bs, padding_value=padding_value, in_place=True
        )
        uniques = uniques.movedim(
            1, dim + 1
        )  # [B, N_0, ..., N_{dim-1}, max(U_bs), N_{dim+1}, ..., N_{D-1}]

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
    x: torch.Tensor,
    return_backmap: Literal[False] = ...,
    return_inverse: Literal[False] = ...,
    return_counts: Literal[False] = ...,
    dim: int | None = None,
    stable: bool = False,
    padding_value: Any = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    pass


@overload
def unique_batched(
    x: torch.Tensor,
    return_backmap: Literal[True] = ...,
    return_inverse: Literal[False] = ...,
    return_counts: Literal[False] = ...,
    dim: int | None = None,
    stable: bool = False,
    padding_value: Any = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pass


@overload
def unique_batched(
    x: torch.Tensor,
    return_backmap: Literal[False] = ...,
    return_inverse: Literal[True] = ...,
    return_counts: Literal[False] = ...,
    dim: int | None = None,
    stable: bool = False,
    padding_value: Any = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pass


@overload
def unique_batched(
    x: torch.Tensor,
    return_backmap: Literal[False] = ...,
    return_inverse: Literal[False] = ...,
    return_counts: Literal[True] = ...,
    dim: int | None = None,
    stable: bool = False,
    padding_value: Any = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pass


@overload
def unique_batched(
    x: torch.Tensor,
    return_backmap: Literal[True] = ...,
    return_inverse: Literal[True] = ...,
    return_counts: Literal[False] = ...,
    dim: int | None = None,
    stable: bool = False,
    padding_value: Any = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pass


@overload
def unique_batched(
    x: torch.Tensor,
    return_backmap: Literal[True] = ...,
    return_inverse: Literal[False] = ...,
    return_counts: Literal[True] = ...,
    dim: int | None = None,
    stable: bool = False,
    padding_value: Any = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pass


@overload
def unique_batched(
    x: torch.Tensor,
    return_backmap: Literal[False] = ...,
    return_inverse: Literal[True] = ...,
    return_counts: Literal[True] = ...,
    dim: int | None = None,
    stable: bool = False,
    padding_value: Any = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pass


@overload
def unique_batched(
    x: torch.Tensor,
    return_backmap: Literal[True] = ...,
    return_inverse: Literal[True] = ...,
    return_counts: Literal[True] = ...,
    dim: int | None = None,
    stable: bool = False,
    padding_value: Any = None,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    pass


def unique_batched(
    x: torch.Tensor,
    return_backmap: bool = False,
    return_inverse: bool = False,
    return_counts: bool = False,
    dim: int | None = None,
    stable: bool = False,
    padding_value: Any = None,
) -> (
    tuple[torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    | tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]
):
    """A batched version of torch.unique(), but WAY more efficient.

    Note that dim refers to the index of the dimension to sort along AFTER the
    batch dimension.

    The returned unique elements are retrieved along the requested dimension,
    taking all the other dimensions apart from the batch dimension as constant
    tuples.

    Args:
        x: The input tensor.
            Shape: [B, N_0, ..., N_dim, ..., N_{D-1}]
        return_backmap: Whether to also return the backmap tensor.
            This can be used to sort the original tensor.
        return_inverse: Whether to also return the inverse mapping tensor.
            This can be used to reconstruct the original tensor from the
            unique tensor.
        return_counts: Whether to also return the counts of each unique
            element.
        dim: The dimension to operate on. If None, the unique of the flattened
            input is returned. Otherwise, each of the tensors indexed by the
            given dimension is treated as one of the elements to apply the
            unique operation on. See examples for more details.
        stable: Whether to preserve the relative order of equal elements. If
            False (default), an unstable sort is used, which is faster. Note
            that this only has an effect on the backmap tensor, so setting
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
            Shape: [B, N_0, ..., N_{dim-1}, max(U_bs), N_{dim+1}, ..., N_{D-1}]
        - The amount of unique values per batch element.
            Shape: [B]
        - (Optional) If return_backmap is True, the backmap tensor, which
            contains the indices of the unique values in the original input.
            The sorted version of x can be retrieved as follows:
            x_sorted = index_select_batched(x, dim, backmap)
            Shape: [B, N_dim]
        - (Optional) If return_inverse is True, the indices where elements
            in the original input ended up in the returned unique values.
            The original tensor can be reconstructed as follows:
            x_reconstructed = index_select_batched(uniques, dim, inverse)
            Shape: [B, N_dim]
        - (Optional) If return_counts is True, the counts for each unique
            element. Padded with padding_value.
            Shape: [B, max(U_bs)]

    Examples:
    >>> # 1D example: -----------------------------------------------------
    >>> x = torch.tensor([
    ...     [9, 10, 9, 9, 10, 9],
    ...     [8, 7, 9, 9, 8, 7],
    ... ])
    >>> dim = 0

    >>> uniques, U_bs, backmap, inverse, counts = unique_batched(
    ...     x,
    ...     return_backmap=True,
    ...     return_inverse=True,
    ...     return_counts=True,
    ...     dim=dim,
    ...     stable=True,
    ...     padding_value=0,
    ... )
    >>> uniques
    tensor([[ 9, 10,  0],
            [ 7,  8,  9]])
    >>> U_bs
    tensor([2, 3])
    >>> backmap
    tensor([[0, 2, 3, 5, 1, 4],
            [1, 5, 0, 4, 2, 3]])
    >>> inverse
    tensor([[0, 1, 0, 0, 1, 0],
            [1, 0, 2, 2, 1, 0]])
    >>> counts
    tensor([[4, 2, 0],
            [2, 2, 2]])

    >>> # Get the lexicographically sorted version of x:
    >>> index_select_batched(x, dim, backmap)
    tensor([[ 9,  9,  9,  9, 10, 10],
            [ 7,  7,  8,  8,  9,  9]])

    >>> # Reconstruct the original tensor:
    >>> index_select_batched(uniques, dim, inverse)
    tensor([[ 9, 10,  9,  9, 10,  9],
            [ 8,  7,  9,  9,  8,  7]])

    >>> # 2D example: -----------------------------------------------------
    >>> x = torch.tensor([
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
    >>> dim = 1

    >>> uniques, U_bs, backmap, inverse, counts = unique_batched(
    ...     x,
    ...     return_backmap=True,
    ...     return_inverse=True,
    ...     return_counts=True,
    ...     dim=dim,
    ...     stable=True,
    ...     padding_value=0,
    ... )
    >>> uniques
    tensor([[[ 7,  9, 10],
             [ 8, 10,  9],
             [ 9,  8,  7],
             [ 9,  7,  7]],
            [[ 7,  7,  0],
             [ 7, 10,  0],
             [ 9,  8,  0],
             [ 8,  8,  0]]])
    >>> U_bs
    tensor([3, 2])
    >>> backmap
    tensor([[2, 0, 3, 1],
            [0, 2, 3, 1]])
    >>> inverse
    tensor([[1, 2, 0, 1],
            [0, 1, 0, 0]])
    >>> counts
    tensor([[1, 2, 1],
            [3, 1, 0]])

    >>> # Get the lexicographically sorted version of x:
    >>> index_select_batched(x, dim, backmap)
    tensor([[[ 7,  9,  9, 10],
             [ 8, 10, 10,  9],
             [ 9,  8,  8,  7],
             [ 9,  7,  7,  7]],
            [[ 7,  7,  7,  7],
             [ 7,  7,  7, 10],
             [ 9,  9,  9,  8],
             [ 8,  8,  8,  8]]])

    >>> # Reconstruct the original tensor:
    >>> index_select_batched(uniques, dim, inverse)
    tensor([[[ 9, 10,  7,  9],
             [10,  9,  8, 10],
             [ 8,  7,  9,  8],
             [ 7,  7,  9,  7]],
            [[ 7,  7,  7,  7],
             [ 7, 10,  7,  7],
             [ 9,  8,  9,  9],
             [ 8,  8,  8,  8]]])
    """
    if dim is None:
        raise NotImplementedError(
            "dim=None is not implemented yet. Please specify a dimension"
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
        x, dim=dim, stable=stable
    )  # [B, N_0, ..., N_dim, ..., N_{D-1}], [B, N_dim]

    out = unique_consecutive_batched(
        x_sorted,
        return_inverse=return_inverse,
        return_counts=return_counts,
        dim=dim,
        padding_value=padding_value,
    )

    aux = []
    if return_backmap:
        aux.append(backmap)
    if return_inverse:
        # The backmap wasn't taken into account by unique_consecutive(), so we
        # have to apply it to the inverse mapping here.
        backmap_inv = swap_idcs_vals_batched(backmap)  # [B, N_dim]
        aux.append(out[2].gather(1, backmap_inv))  # type: ignore
    if return_counts:
        aux.append(out[-1])

    return out[0], out[1], *aux


# ############################ CONSECUTIVE SEGMENTS ############################


def counts_segments_ints_batched(
    x: torch.Tensor, high: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Count the frequency of each consecutive value, with values in [0, high).

    Args:
        x: The tensor for which to count the frequency of each integer value.
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
    >>> x = torch.tensor([
    ...    [4, 4, 4, 2, 2, 8, 3, 3, 3, 3],
    ...    [1, 1, 0, 0, 0, 0, 0, 2, 2, 2],
    ... ])
    >>> freqs, U_bs = counts_segments_ints_batched(x, 10)
    >>> freqs
    tensor([[0, 0, 2, 4, 3, 0, 0, 0, 1, 0],
            [5, 2, 3, 0, 0, 0, 0, 0, 0, 0]])
    >>> U_bs
    tensor([4, 3])
    """
    if x.ndim != 2:
        raise ValueError("x must be a batch of 1D tensors.")

    B = x.shape[0]
    device = x.device

    freqs = torch.zeros((B, high), device=device, dtype=torch.int64)
    uniques, U_bs, counts = unique_consecutive_batched(
        x, return_counts=True, dim=0
    )  # [B, max(U_bs)], [B], [B, max(U_bs)]
    freqs[
        torch.arange(B, device=device).repeat_interleave(U_bs),  # [U]
        pack_padded(uniques, U_bs),  # [U]
    ] = pack_padded(counts, U_bs)  # [U]  # fmt: skip
    return freqs, U_bs


# ################################## GROUPBY ###################################


@overload
def groupby_batched(  # type: ignore
    keys: torch.Tensor,
    vals: torch.Tensor | None = None,
    stable: bool = False,
    as_sequence: Literal[True] = ...,
    padding_value: Any = None,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    pass


@overload
def groupby_batched(
    keys: torch.Tensor,
    vals: torch.Tensor | None = None,
    stable: bool = False,
    as_sequence: Literal[False] = ...,
    padding_value: Any = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pass


def groupby_batched(
    keys: torch.Tensor,
    vals: torch.Tensor | None = None,
    stable: bool = False,
    as_sequence: bool = True,
    padding_value: Any = None,
) -> (
    list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
):
    """Group values by keys.

    Args:
        keys: The keys to group by.
            Shape: [B, N, *]
        vals: The values to group. If None, the values are set to the indices of
            the keys (i.e. vals = arange_batched(torch.full((B,), N))[0]).
            Shape: [B, N, **]
        stable: Whether to preserve the order of vals that have the same key. If
            False (default), an unstable sort is used, which is faster.
        as_sequence: Whether to return the result as a sequence of (key, vals)
            tuples (True) or as packed tensors (False).
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
            - Tensor of unique keys, sorted. Padded with padding_value.
                Shape: [B, max(U_bs), *]
            - Tensor with the amount of unique keys per batch element.
                Shape: [B]
            - Tensor of values stored a packed manner, grouped by key. Along
                every batch element, the first N_key1 values correspond to the
                first key, the next N_key2 values correspond to the second key,
                etc. Each group of values is sorted if stable is True. Padded
                with padding_value.
                Shape: [B, N, **]
            - Tensor containing the number of values for each unique key. Padded
                with padding_value.
                Shape: [B, max(U_bs)]

    Examples:
    >>> keys = torch.tensor([
    ...     [4, 2, 4, 3, 2, 8, 4],
    ...     [1, 0, 1, 2, 0, 1, 0],
    ... ])
    >>> vals = torch.tensor([
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
    tensor([2, 0])
    Mask:
    tensor([True, True])
    Grouped Vals:
    tensor([[[ 2,  3],
             [ 8,  9],
             [ 0,  0]],
            [[16, 17],
             [22, 23],
             [26, 27]]])
    Counts:
    tensor([2, 3])
    <BLANKLINE>
    Key:
    tensor([3, 1])
    Mask:
    tensor([True, True])
    Grouped Vals:
    tensor([[[ 6,  7],
             [ 0,  0],
             [ 0,  0]],
            [[14, 15],
             [18, 19],
             [24, 25]]])
    Counts:
    tensor([1, 3])
    <BLANKLINE>
    Key:
    tensor([4, 2])
    Mask:
    tensor([True, True])
    Grouped Vals:
    tensor([[[ 0,  1],
             [ 4,  5],
             [12, 13]],
            [[20, 21],
             [ 0,  0],
             [ 0,  0]]])
    Counts:
    tensor([3, 1])
    <BLANKLINE>
    Key:
    tensor([8, 0])
    Mask:
    tensor([ True, False])
    Grouped Vals:
    tensor([[[10, 11]],
            [[ 0,  0]]])
    Counts:
    tensor([1, 0])

    >>> # Return as packed tensors:
    >>> keys_unique, U_bs, vals_grouped, counts = groupby_batched(
    ...     keys, vals, stable=True, as_sequence=False, padding_value=0
    ... )
    >>> keys_unique
    tensor([[2, 3, 4, 8],
            [0, 1, 2, 0]])
    >>> U_bs
    tensor([4, 3])
    >>> vals_grouped
    tensor([[[ 2,  3],
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
    tensor([[2, 1, 3, 1],
            [3, 3, 1, 0]])
    """
    device = keys.device

    # Create a mapping from keys to values.
    keys_unique, U_bs, backmap, counts = unique_batched(
        keys,
        return_backmap=True,
        return_counts=True,
        dim=0,
        stable=stable,
        padding_value=padding_value,
    )  # [B, max(U_bs), *], [B], [B, N], [B, max(U_bs)]

    # Rearrange values to match keys_unique.
    if vals is None:
        vals_grouped = backmap  # [B, N]
    else:
        vals_grouped = index_select_batched(vals, 0, backmap)  # [B, N, **]

    # Return the results.
    if not as_sequence:
        return keys_unique, U_bs, vals_grouped, counts

    B, N = keys.shape[0], keys.shape[1]

    # Calculate outer indices.
    outer_idcs, _ = repeat_interleave_batched(
        torch.arange(counts.shape[1], device=device)
        .unsqueeze(0)
        .expand(counts.shape),  # [B, max(U_bs)]
        U_bs.unsqueeze(1),  # [B, 1]
        counts,
        dim=0,
    )  # [B, N], _

    # Create masks for every batch of unique keys.
    masks = (
        outer_idcs.unsqueeze(1)  # [B, 1, N]
        == torch.arange(
            counts.shape[1], device=device
        ).unsqueeze(0).unsqueeze(2)  # [1, max(U_bs), 1]
    )  # [B, max(U_bs), N]  # fmt: skip

    # Create the sequences of (key, vals_group) tuples.
    return [
        (
            keys_unique[:, u],  # [B, *]
            u < U_bs,  # [B]
            apply_mask(
                vals_grouped,
                masks[:, u],
                torch.full((B,), N, device=device),
                padding_value=padding_value,
            )[0],  # [B, max(N_key_bs), **]
            counts[:, u],  # [B]
        )
        for u in range(keys_unique.shape[1])
    ]  # fmt: skip
