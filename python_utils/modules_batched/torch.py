from __future__ import annotations

from typing import Any, Literal, overload

import torch

from ..modules.torch import (
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
    """
    if not is_padding_zero:
        values = replace_padding(values, L_bs)

    return (
        values.sum(dim=1)  # [B, *]
        / L_bs.reshape(-1, *([1] * (values.ndim - 2)))  # [B, *]
    )  # [B, *]  # fmt: skip


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
    """
    if not is_padding_inf:
        values = replace_padding(values, L_bs, padding_value=float("inf"))

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
    """
    if not is_padding_minus_inf:
        values = replace_padding(values, L_bs, padding_value=float("-inf"))

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

    This function is a direct translation of np.interp() to PyTorch tensors.
    It performs linear interpolation on a batch of 1D tensors.

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

        xp_mod = xp % period.unsqueeze(1)  # [B, M]
        sorted_idcs = xp_mod.argsort(dim=1)  # [B, M]
        xp = xp_mod.gather(1, sorted_idcs)  # [B, M]
        fp = fp.gather(1, sorted_idcs)  # [B, M]

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
    L_bs: torch.Tensor, max_L_bs: int, num_samples: int
) -> torch.Tensor:
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
    # sample with replacement. To do this, the torch.clamp(min=num_samples) and
    # % L_b operations are used.
    weights = mask_padding(
        L_bs.clamp(min=num_samples), max_L_bs
    ).double()  # [B, max(L_bs)]
    return (
        weights.multinomial(num_samples)  # [B, num_samples]
        % L_bs.unsqueeze(1)  # [B, num_samples]
    )  # [B, num_samples]  # fmt: skip


def sample_unique_pairs_batched(
    L_bs: torch.Tensor, max_L_bs: int, num_samples: int
) -> torch.Tensor:
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
    device = L_bs.device

    # Compute the number of unique pairs of indices.
    P_bs = L_bs * (L_bs - 1) // 2  # [B]
    max_P_bs = max_L_bs * (max_L_bs - 1) // 2

    # Select unique pairs of elements for each sample in the batch.
    idcs_pairs = sample_unique_batched(
        P_bs, max_P_bs, num_samples
    )  # [B, num_samples]

    # Convert the pair indices to element indices.
    # torch.triu_indices() returns the indices in the wrong order, e.g.:
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
    triu_idcs = torch.triu_indices(
        max_L_bs, max_L_bs, 1, device=device
    )  # [2, max(P_bs)]
    triu_idcs = max_L_bs - 1 - triu_idcs
    idcs_elements = triu_idcs[:, -idcs_pairs - 1]  # [2, B, num_samples]
    return idcs_elements.permute(1, 2, 0)  # [B, num_samples, 2]


# ######################### BASIC TENSOR MANIPULATION ##########################


def arange_batched(
    starts: torch.Tensor,
    ends: torch.Tensor | None = None,
    steps: torch.Tensor | None = None,
    padding_value: Any = None,
    device: torch.device | str | int | None = None,
    dtype: torch.dtype | None = None,
    requires_grad: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a batch of tensors with values in the range [start, end).

    Args:
        starts: The start value for each tensor in the batch.
            Shape: [B]
        ends: The end value for each tensor in the batch. If None, the end
            value is set to the start value.
            Shape: [B]
        steps: The step value for each tensor in the batch. If None, the step
            value is set to 1.
            Shape: [B]
        padding_value: The value to pad the values with. If None, the values
            are padded with random values. This is faster than padding with
            a specific value.
        device: The device of the output tensor.
        dtype: The data type of the output tensor.
        requires_grad: Whether to enable gradients for the output tensor.

    Returns:
        Tuple containing:
        - A batch of tensors with values in the range [start, end).
            Padded with padding_value.
            Shape: [B, max(L_bs)]
        - The number of values of any arange sequence in the batch.
            Shape: [B]
    """
    device = device if device is not None else starts.device
    B = len(starts)

    # Prepare the input tensors.
    if ends is None:
        ends = starts
        starts = torch.zeros(B, device=device)
    if steps is None:
        steps = torch.ones(B, device=device)

    # Compute the arange sequences in parallel.
    L_bs = ((ends - starts) // steps).int()  # [B]
    max_L_bs = int(L_bs.max())
    aranges = (
        starts.unsqueeze(1)  # [B, 1]
        + torch.arange(max_L_bs, device=device)  # [max(L_bs)]
        * steps.unsqueeze(1)  # [B, 1]
    )  # [B, max(L_bs)]  # fmt: skip

    # Replace values that are out of bounds with the padding value.
    if padding_value is not None:
        replace_padding(
            aranges, L_bs, padding_value=padding_value, in_place=True
        )

    # Perform final adjustments.
    aranges = aranges.to(dtype if dtype is not None else aranges.dtype)
    if requires_grad:
        aranges.requires_grad_()

    return aranges, L_bs


def linspace_batched(
    starts: torch.Tensor,
    ends: torch.Tensor,
    steps: torch.Tensor,
    padding_value: Any = None,
    device: torch.device | str | int | None = None,
    dtype: torch.dtype | None = None,
    requires_grad: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a batch of tensors with values in the range [start, end].

    Args:
        starts: The start value for each tensor in the batch.
            Shape: [B]
        ends: The end value for each tensor in the batch.
            Shape: [B]
        steps: The number of steps for each tensor in the batch.
            Shape: [B]
        padding_value: The value to pad the values with. If None, the values
            are padded with random values. This is faster than padding with
            a specific value.
        device: The device of the output tensor.
        dtype: The data type of the output tensor.
        requires_grad: Whether to enable gradients for the output tensor.

    Returns:
        Tuple containing:
        - A batch of tensors with values in the range [start, end].
            Padded with padding_value.
            Shape: [B, max(L_bs)]
        - The number of values of any linspace sequence in the batch.
            Shape: [B]
    """
    device = device if device is not None else starts.device
    B = len(starts)

    # Compute the linspace sequences in parallel.
    L_bs = steps.int()  # [B]
    max_L_bs = int(L_bs.max())
    L_bs_minus_1 = L_bs - 1  # [B]
    linspaces = (
        starts.unsqueeze(1)  # [B, 1]
        + torch.arange(max_L_bs, device=device)  # [max(L_bs)]
        / L_bs_minus_1.unsqueeze(1)  # [B, 1]
        * (ends - starts).unsqueeze(1)  # [B, 1]
    )  # [B, max(L_bs)]  # fmt: skip

    # Prevent floating point errors.
    linspaces[torch.arange(B, device=device), L_bs_minus_1] = ends.to(
        linspaces.dtype
    )

    # Replace values that are out of bounds with the padding value.
    if padding_value is not None:
        replace_padding(
            linspaces, L_bs, padding_value=padding_value, in_place=True
        )

    # Perform final adjustments.
    linspaces = linspaces.to(dtype if dtype is not None else linspaces.dtype)
    if requires_grad:
        linspaces.requires_grad_()

    return linspaces, L_bs


def index_select_batched(
    values: torch.Tensor, dim: int, indices: torch.Tensor
) -> torch.Tensor:
    """Select values from a batch of tensors using the given indices.

    Note that dim refers to the index of the dimension to sort along AFTER the
    batch dimension. So e.g. if x has shape [B, N_0, N_1, N_2], then dim=0
    refers to N_0, dim=1 refers to N_1, etc.

    Args:
        values: The values to select from.
            Shape: [B, N_0, ..., N_dim, ..., N_{D-1}]
        dim: The dimension to select along.
        indices: The indices to select.
            Shape: [B, N_select]

    Returns:
        The selected values.
            Shape: [B, N_0, ..., N_{dim-1}, N_select, N_{dim+1}, ..., N_{D-1}]
    """
    idcs_reshape = [1] * values.ndim
    idcs_reshape[0] = indices.shape[0]
    idcs_reshape[dim + 1] = indices.shape[1]
    idcs_expand = list(values.shape)
    idcs_expand[dim + 1] = indices.shape[1]
    return values.gather(
        dim + 1,
        indices.reshape(idcs_reshape).expand(idcs_expand).to(torch.int64),
    )


def repeat_interleave_batched(
    values: torch.Tensor,
    repeats: torch.Tensor,
    sum_repeats: torch.Tensor,
    max_sum_repeats: int,
    dim: int = 0,
    padding_value: Any = None,
) -> torch.Tensor:
    """Repeat values from a batch of tensors using the given repeats.

    Note that dim refers to the index of the dimension to sort along AFTER the
    batch dimension. So e.g. if x has shape [B, N_0, N_1, N_2], then dim=0
    refers to N_0, dim=1 refers to N_1, etc.

    Args:
        values: The values to repeat.
            Shape: [B, N_0, ..., N_dim, ..., N_{D-1}]
        repeats: The number of times to repeat each value.
            Shape: [B, N_dim]
        sum_repeats: The sum of repeats for each element in the batch.
            Must be equal to sum(repeats, dim=1).
            Shape: [B]
        max_sum_repeats: The maximum sum of repeats for any element in the
            batch. Must be equal to max(sum(repeats, dim=1)).
        dim: The dimension to repeat along.
        padding_value: The value to pad the values with. If None, the values
            are padded with random values. This is faster than padding with
            a specific value.

    Returns:
        The repeated values. Padded with padding_value.
            Shape: [B, N_0, ..., max_sum_repeats, ..., N_{D-1}]

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
    >>> repeats = torch.tensor([
    ...     [1, 2, 3],
    ...     [0, 1, 0],
    ... ])
    >>> sum_repeats = torch.tensor([6, 1])
    >>> max_sum_repeats = 6
    >>> repeat_interleave_batched(
    ...     values,
    ...     repeats,
    ...     sum_repeats,
    ...     max_sum_repeats,
    ...     dim=0,
    ...     padding_value=0,
    ... )
    tensor([[[ 1,  2],
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
    # Move dim to the front and merge it with the batch dimension.
    # This allows us to use torch.repeat_interleave() directly.
    values = values.movedim(
        dim + 1, 1
    )  # [B, N_dim, N_0, ..., N_{dim-1}, N_{dim+1}, ..., N_{D-1}]
    values = values.reshape(
        -1, *values.shape[2:]
    )  # [B * N_dim, N_0, ..., N_{dim-1}, N_{dim+1}, ..., N_{D-1}]
    repeats_reshaped = repeats.reshape(-1)  # [B * N_dim]

    # Repeat the values.
    values = values.repeat_interleave(
        repeats_reshaped, dim=0
    )  # [B * sum(repeats), N_0, ..., N_{dim-1}, N_{dim+1}, ..., N_{D-1}]

    # Un-merge the batch and dim dimensions and move dim back to its original
    # position.
    values = pad_packed(
        values, sum_repeats, max_sum_repeats, padding_value=padding_value
    )  # [B, max_sum_repeats, N_0, ..., N_{dim-1}, N_{dim+1}, ..., N_{D-1}]
    values = values.movedim(
        1, dim + 1
    )  # [B, N_0, ..., max_sum_repeats, ..., N_{D-1}]
    return values


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
    x_swapped = torch.empty_like(x)
    x_swapped[
        torch.arange(B, device=x.device, dtype=x.dtype).unsqueeze(1), x
    ] = torch.arange(N, device=x.device, dtype=x.dtype).unsqueeze(0)
    return x_swapped


def swap_idcs_vals_duplicates_batched(
    x: torch.Tensor, stable: bool = False
) -> torch.Tensor:
    """Swap the indices and values of a batch of 1D tensors allowing duplicates.

    Each row in the input tensor is assumed to contain integers from 0 to
    M <= N, in any order, and may contain duplicates.

    Each row in the output tensor will contain exactly all integers from 0 to
    len(x) - 1, in any order.

    If the input doesn't contain duplicates, you should use swap_idcs_vals()
    instead since it is faster (especially for large tensors).

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

    # Believe it or not, this O(n log n) algorithm is actually faster than a
    # native implementation that uses a Python for loop with complexity O(n).
    return x.argsort(dim=1, stable=stable).to(x.dtype)


# ############################ CONSECUTIVE SEGMENTS ############################


def starts_segments_batched(
    x: torch.Tensor, dim: int = 0, padding_value: Any = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Find the start index of each consecutive segment in each batch tensor.

    Note that dim refers to the index of the dimension to sort along AFTER the
    batch dimension. So e.g. if x has shape [B, N_0, N_1, N_2], then dim=0
    refers to N_0, dim=1 refers to N_1, etc.

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
    tensor([
        [0, 3, 5, 6],
        [0, 2, 7, 0],
    ])
    >>> S_bs
    tensor([4, 3])
    """
    B, N_dim = x.shape[0], x.shape[dim + 1]

    # Find the indices where the values change.
    is_change = (
        torch.concat(
            [
                torch.ones((B, 1), device=x.device, dtype=torch.bool),
                (
                    x.index_select(
                        dim + 1, torch.arange(0, N_dim - 1, device=x.device)
                    )  # [B, N_0, ..., N_dim - 1, ..., N_{D-1}]
                    != x.index_select(
                        dim + 1, torch.arange(1, N_dim, device=x.device)
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
        else torch.empty((B, 0), device=x.device, dtype=torch.bool)
    )  # [B, N_dim]

    # Find the start of each consecutive segment.
    batch_idcs, starts_idcs = is_change.nonzero(as_tuple=True)  # [S], [S]

    # Convert to padded representation.
    batch_idcs = batch_idcs.int()
    starts_idcs = starts_idcs.int()
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
    batch dimension. So e.g. if x has shape [B, N_0, N_1, N_2], then dim=0
    refers to N_0, dim=1 refers to N_1, etc.

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
    tensor([
        [3, 2, 1, 4],
        [2, 5, 3, 0],
    ])
    >>> S_bs
    tensor([4, 3])
    """
    B, N_dim = x.shape[0], x.shape[dim + 1]

    # Find the start of each consecutive segment.
    starts, S_bs = starts_segments_batched(
        x, dim=dim, padding_value=padding_value
    )  # [B, max(S_bs)], [B]

    # Prepare starts for count calculation.
    starts_with_N_dim = torch.concat(
        [starts, torch.full((B, 1), N_dim, device=x.device, dtype=torch.int32)],
        dim=1,
    )  # [B, max(S_bs) + 1]
    starts_with_N_dim[torch.arange(B, device=x.device), S_bs] = N_dim

    # Find the count of each consecutive segment.
    counts = (
        starts_with_N_dim.diff(dim=1)  # [B, max(S_bs)]
        if N_dim > 0
        else torch.empty((B, 0), device=x.device, dtype=torch.int32)
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

    Returns:
        Tuple containing:
        - The outer indices for each consecutive segment in x.
            Shape: [B, N_dim]
        - The number of consecutive segments in each tensor.
            Shape: [B]
        - (Optional) If return_counts is True, the counts for each consecutive
            segment in x.
            Shape: [B, max(S_bs)]
        - (Optional) If return_starts is True, the start indices for each
            consecutive segment in x.
            Shape: [B, max(S_bs)]

    Examples:
    >>> x = torch.tensor([
    ...     [4, 4, 4, 2, 2, 8, 3, 3, 3, 3],
    ...     [1, 1, 0, 0, 0, 0, 0, 2, 2, 2],
    ... ])

    >>> outer_idcs, S_bs = outer_indices_segments_batched(x)
    >>> outer_idcs
    tensor([
        [0, 0, 0, 1, 1, 2, 3, 3, 3, 3],
        [0, 0, 1, 1, 1, 1, 1, 2, 2, 2],
    ])
    >>> S_bs
    tensor([4, 3])
    """
    B, N_dim = x.shape[0], x.shape[dim + 1]

    # Find the start (optional) and count of each consecutive segment.
    if return_starts:
        counts, S_bs, starts = counts_segments_batched(
            x, dim=dim, return_starts=True, padding_value=padding_value
        )  # [B, max(S_bs)], [B], [B, max(S_bs)]
    else:
        counts, S_bs = counts_segments_batched(
            x, dim=dim, padding_value=padding_value
        )  # [B, max(S_bs)], [B]

    # Prepare counts for outer index calculation.
    counts_with_zeros = replace_padding(counts, S_bs)
    sum_counts = torch.full(
        (B,), N_dim, device=x.device, dtype=torch.int32
    )  # [B]
    max_sum_counts = N_dim

    # Calculate the outer indices.
    outer_idcs = repeat_interleave_batched(
        torch.arange(
            counts.shape[1], device=x.device, dtype=torch.int32
        ).unsqueeze(0).expand_as(counts),  # [B, max(S_bs)]  # fmt: skip
        counts_with_zeros,
        sum_counts,
        max_sum_counts,
        dim=0,
    )  # [B, N_dim]

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
    batch dimension. So e.g. if x has shape [B, N_0, N_1, N_2], then dim=0
    refers to N_0, dim=1 refers to N_1, etc.

    Args:
        x: The input tensor. Consecutive equal values will be grouped.
            Shape: [B, N_0, ..., N_dim, ..., N_{D-1}]
        dim: The dimension along which the segments are lined up.
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
    tensor([
        [0, 1, 2, 0, 1, 0, 0, 1, 2, 3],
        [0, 1, 0, 1, 2, 3, 4, 0, 1, 2],
    ])
    >>> S_bs
    tensor([4, 3])
    """
    B, N_dim = x.shape[0], x.shape[dim + 1]

    # Find the start and count of each consecutive segment.
    counts, S_bs, starts = counts_segments_batched(
        x, dim=dim, return_starts=True, padding_value=padding_value
    )  # [B, max(S_bs)], [B], [B, max(S_bs)]

    # Prepare counts for inner index calculation.
    counts_with_zeros = replace_padding(counts, S_bs)
    sum_counts = torch.full(
        (B,), N_dim, device=x.device, dtype=torch.int32
    )  # [B]
    max_sum_counts = N_dim

    # Calculate the inner indices.
    inner_idcs = (
        torch.arange(
            N_dim, device=x.device, dtype=torch.int32
        ).unsqueeze(0)  # [1, N_dim]
        - repeat_interleave_batched(
            starts, counts_with_zeros, sum_counts, max_sum_counts, dim=0
        )  # [B, N_dim]
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
    batch dimension. So e.g. if x has shape [B, N_0, N_1, N_2], then dim=0
    refers to N_0, dim=1 refers to N_1, etc.

    This is like a batched version of torch.sort(), but it doesn't sort along
    the other dimensions. As such, the other dimensions are treated as tuples.
    This function is roughly equivalent to the following Python code, but it
    is much faster.
    >>> torch.stack([
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
            >>> x_sorted = index_select_batched(x, dim, backmap)
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
    # We can use lexsort() to sort only the requested dimension.
    # First, we prepare the tensor for lexsort(). The input to this function
    # must be a tuple of tensor-like objects, that are evaluated from last to
    # first. This is quite confusing, so I'll put an example here. If we have:
    # >>> x = tensor([[[15, 13],
    # ...              [11,  4],
    # ...              [16,  2]],
    # ...             [[ 7, 21],
    # ...              [ 3, 20],
    # ...              [ 8, 22]],
    # ...             [[19, 14],
    # ...              [ 5, 12],
    # ...              [ 6,  0]],
    # ...             [[23,  1],
    # ...              [10, 17],
    # ...              [ 9, 18]]])
    # And dim=1, then the input to lexsort() must be:
    # >>> lexsort(tensor([[ 1, 17, 18],
    # ...                 [23, 10,  9],
    # ...                 [14, 12,  0],
    # ...                 [19,  5,  6],
    # ...                 [21, 20, 22],
    # ...                 [ 7,  3,  8],
    # ...                 [13,  4,  2],
    # ...                 [15, 11, 16]]))
    # Note that the first row is evaluated last and the last row is evaluated
    # first. We can now see that the sorting order will be 11 < 15 < 16, so
    # lexsort() will return tensor([1, 0, 2]). I thouroughly tested what the
    # absolute fastest way is to perform this operation, and it turns out that
    # the following is the best way to do it:
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
    batch dimension. So e.g. if x has shape [B, N_0, N_1, N_2], then dim=0
    refers to N_0, dim=1 refers to N_1, etc.

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
        dim: The dimension to operate on. If None, the unique of the
            flattened input is returned. Otherwise, each of the tensors
            indexed by the given dimension is treated as one of the elements
            to apply the unique operation on. See examples for more details.
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
            >>> x_reconstructed = index_select_batched(uniques, dim, inverse)
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
    batch dimension. So e.g. if x has shape [B, N_0, N_1, N_2], then dim=0
    refers to N_0, dim=1 refers to N_1, etc.

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
        dim: The dimension to operate on. If None, the unique of the
            flattened input is returned. Otherwise, each of the tensors
            indexed by the given dimension is treated as one of the elements
            to apply the unique operation on. See examples for more details.
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
            >>> x_sorted = index_select_batched(x, dim, backmap)
            Shape: [B, N_dim]
        - (Optional) If return_inverse is True, the indices where elements
            in the original input ended up in the returned unique values.
            The original tensor can be reconstructed as follows:
            >>> x_reconstructed = index_select_batched(uniques, dim, inverse)
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
    # sort the other dimensions as well.
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
    tensor([
        [0, 0, 2, 4, 3, 0, 0, 0, 1, 0],
        [5, 2, 3, 0, 0, 0, 0, 0, 0, 0],
    ])
    >>> U_bs
    tensor([4, 3])
    """
    if x.ndim != 2:
        raise ValueError("x must be a batch of 1D tensors.")

    B = x.shape[0]

    freqs = torch.zeros((B, high), device=x.device, dtype=torch.int32)
    uniques, U_bs, counts = unique_consecutive_batched(
        x, return_counts=True, dim=0
    )  # [B, max(U_bs)], [B], [B, max(U_bs)]
    freqs[
        torch.arange(B, device=x.device).repeat_interleave(U_bs),  # [U]
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
            the keys (i.e. vals = arange_batched(N)).
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
    ...     print(f"Key:\\n{key}")
    ...     print(f"Mask:\\n{mask}")
    ...     print(f"Grouped Vals:\\n{vals_group}")
    ...     print(f"Counts:\\n{counts}")
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
    tensor([2, 3], dtype=torch.int32)

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
    tensor([1, 3], dtype=torch.int32)

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
    tensor([3, 1], dtype=torch.int32)

    Key:
    tensor([8, 0])
    Mask:
    tensor([ True, False])
    Grouped Vals:
    tensor([[[10, 11]],
            [[ 0,  0]]])
    Counts:
    tensor([1, 0], dtype=torch.int32)

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
            [3, 3, 1, 0]], dtype=torch.int32)
    """
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
        vals = backmap  # [B, N]
    else:
        vals = index_select_batched(vals, 0, backmap)  # [B, N, **]

    # Return the results.
    if not as_sequence:
        return keys_unique, U_bs, vals, counts

    B, N = keys.shape[:2]

    # Prepare counts for outer index calculation.
    counts_with_zeros = replace_padding(counts, U_bs)
    sum_counts = torch.full(
        (B,), N, device=keys.device, dtype=torch.int32
    )  # [B]
    max_sum_counts = N

    # Calculate outer indices.
    outer_idcs = repeat_interleave_batched(
        torch.arange(
            counts.shape[1], device=keys.device, dtype=torch.int32
        ).unsqueeze(0).expand_as(counts),  # [B, max(U_bs)]  # fmt: skip
        counts_with_zeros,
        sum_counts,
        max_sum_counts,
        dim=0,
    )  # [B, N]

    # Create masks for every batch of unique keys.
    masks = (
        outer_idcs.unsqueeze(1)  # [B, 1, N]
        == torch.arange(
            counts.shape[1], device=keys.device, dtype=torch.int32
        ).unsqueeze(0).unsqueeze(2)  # [1, max(U_bs), 1]
    )  # [B, max(U_bs), N]  # fmt: skip

    # Create the sequences of (key, vals) tuples.
    return [
        (
            keys_unique[:, u],  # [B, *]
            u < U_bs,  # [B]
            apply_mask(
                vals, masks[:, u], sum_counts, padding_value=padding_value
            )[0],  # [B, max(N_key_bs), **]
            counts[:, u],  # [B]
        )
        for u in range(keys_unique.shape[1])
    ]  # fmt: skip
