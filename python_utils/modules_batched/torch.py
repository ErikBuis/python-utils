from functools import lru_cache
from typing import Any

import torch


@lru_cache(maxsize=8)
def mask_padding_batched(L_bs: torch.Tensor, max_L_b: int) -> torch.Tensor:
    """Create a mask that indicates which values are valid in each sample.

    Args:
        L_bs: The number of valid values in each sample.
            Shape: [B]
        max_L_b: The maximum number of values of any element in the batch.

    Returns:
        A mask that indicates which values are valid in each sample.
            Shape: [B, max(L_bs)]
    """
    dtype = L_bs.dtype
    device = L_bs.device

    return (
        torch.arange(max_L_b, dtype=dtype, device=device)  # [max(L_bs)]
        < L_bs.unsqueeze(1)  # [B, 1]
    )  # [B, max(L_bs)]  # fmt: skip


def pack_padded_batched(
    values: torch.Tensor, L_bs: torch.Tensor
) -> torch.Tensor:
    """Pack a batch of padded values into a single tensor.

    Args:
        values: The values to pack. The values must be padded for heterogenous
            batch sizes.
            Shape: [B, max(L_bs), *]
        L_bs: The number of valid values in each sample.
            Shape: [B]

    Returns:
        The packed values.
            Shape: [sum(L_bs), *]
    """
    max_L_bs = values.shape[1]
    mask = mask_padding_batched(L_bs, max_L_bs)
    return values[mask]


def pad_packed_batched(
    values: torch.Tensor,
    L_bs: torch.Tensor,
    max_L_bs: int,
    padding_value: Any | None = 0,
) -> torch.Tensor:
    """Pad a batch of packed values to create a tensor with a fixed size.

    Args:
        values: The values to pad with zeros for heterogenous batch sizes.
            Shape: [sum(L_bs), *]
        L_bs: The number of valid values in each sample.
            Shape: [B]
        max_L_b: The maximum number of values of any element in the batch.
        padding_value: The value to pad the values with. If None, the values
            are padded with random values. This is faster than padding with
            a specific value.

    Returns:
        The values padded with zeros for heterogenous batch sizes.
            Shape: [B, max(L_bs), *]
    """
    B = len(L_bs)
    dtype = values.dtype
    device = values.device

    padded_shape = (B, max_L_bs, *values.shape[1:])
    if padding_value is None:
        values_padded = torch.empty(padded_shape, dtype=dtype, device=device)
    elif padding_value == 0:
        values_padded = torch.zeros(padded_shape, dtype=dtype, device=device)
    elif padding_value == 1:
        values_padded = torch.ones(padded_shape, dtype=dtype, device=device)
    else:
        values_padded = torch.full(
            padded_shape, padding_value, dtype=dtype, device=device
        )

    mask = mask_padding_batched(L_bs, max_L_bs)
    values_padded[mask] = values

    return values_padded


def replace_padding_batched(
    values: torch.Tensor,
    L_bs: torch.Tensor,
    padding_value: Any = 0,
    in_place: bool = False,
) -> torch.Tensor:
    """Pad the values with padding_value to create a tensor with a fixed size.

    Args:
        values: The values to pad with padding_value for heterogenous batch
            sizes. Will be modified in-place if in_place is True.
            Shape: [B, max(L_bs), *]
        L_bs: The number of valid values in each sample.
            Shape: [B]
        padding_value: The value to pad the values with.
        in_place: Whether to perform the operation in-place.

    Returns:
        The values padded with padding_value for heterogenous batch sizes.
            Shape: [B, max(L_bs), *]
    """
    max_L_bs = values.shape[1]
    mask = mask_padding_batched(L_bs, max_L_bs)
    values_padded = values if in_place else values.clone()
    values_padded[~mask] = padding_value
    return values_padded


def mean_padding_batched(
    values: torch.Tensor, L_bs: torch.Tensor, is_padding_zero: bool = False
) -> torch.Tensor:
    """Calculate the mean per dimension for each sample in the batch.

    Args:
        values: The values to calculate the mean for. The values must be padded
            for heterogenous batch sizes.
            Shape: [B, max(L_bs), *]
        L_bs: The number of valid values in each sample.
            Shape: [B]
        is_padding_zero: Whether the values are padded with zeros already.

    Returns:
        The mean value per dimension for each sample.
            Shape: [B, *]
    """
    if not is_padding_zero:
        values = replace_padding_batched(values, L_bs)

    return (
        torch.sum(values, dim=1)  # [B, *]
        / L_bs.reshape(-1, *([1] * (values.ndim - 2)))  # [B, *]
    )  # [B, *]  # fmt: skip


def stddev_padding_batched(
    values: torch.Tensor, L_bs: torch.Tensor, is_padding_zero: bool = False
) -> torch.Tensor:
    """Calculate the standard dev. per dimension for each sample in the batch.

    For a set of values x_1, ..., x_n, the standard deviation is defined as:
        sqrt(sum((x_i - mean(x))^2) / n)

    Args:
        values: The values to calculate the standard deviation for. The values
            must be padded for heterogenous batch sizes.
            Shape: [B, max(L_bs), *]
        L_bs: The number of valid values in each sample.
            Shape: [B]
        is_padding_zero: Whether the values are padded with zeros already.

    Returns:
        The standard deviation per dimension for each sample.
            Shape: [B, *]
    """
    means = mean_padding_batched(
        values, L_bs, is_padding_zero=is_padding_zero
    )  # [B, *]
    values_centered = values - means.unsqueeze(1)  # [B, max(L_bs), *]
    replace_padding_batched(values_centered, L_bs, in_place=True)
    return mean_padding_batched(
        values_centered.square(), L_bs, is_padding_zero=True
    ).sqrt()  # [B, *]


def min_padding_batched(
    values: torch.Tensor, L_bs: torch.Tensor, is_padding_inf: bool = False
) -> torch.Tensor:
    """Calculate the minimum per dimension for each sample in the batch.

    Args:
        values: The values to calculate the minimum for. The values must be
            padded for heterogenous batch sizes.
            Shape: [B, max(L_bs), *]
        L_bs: The number of valid values in each sample.
            Shape: [B]
        is_padding_inf: Whether the values are padded with inf values already.
            already.

    Returns:
        The minimum value per dimension for each sample.
            Shape: [B, *]
    """
    if not is_padding_inf:
        values = replace_padding_batched(
            values, L_bs, padding_value=float("inf")
        )

    return values.amin(dim=1)  # [B, *]


def max_padding_batched(
    values: torch.Tensor,
    L_bs: torch.Tensor,
    is_padding_minus_inf: bool = False,
) -> torch.Tensor:
    """Calculate the maximum per dimension for each sample in the batch.

    Args:
        values: The values to calculate the maximum for. The values must be
            padded for heterogenous batch sizes.
            Shape: [B, max(L_bs), *]
        L_bs: The number of valid values in each sample.
            Shape: [B]
        is_padding_minus_inf: Whether the values are padded with -inf values
            already.

    Returns:
        The maximum value per dimension for each sample.
            Shape: [B, *]
    """
    if not is_padding_minus_inf:
        values = replace_padding_batched(
            values, L_bs, padding_value=float("-inf")
        )

    return values.amax(dim=1)  # [B, *]


def arange_batched(
    starts: torch.Tensor,
    ends: torch.Tensor | None = None,
    steps: torch.Tensor | None = None,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
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
        dtype: The data type of the tensor.
        device: The device of the tensor.
        requires_grad: Whether to require gradients for the tensor.

    Returns:
        Tuple containing:
        - A batch of tensors with values in the range [start, end),
            padded with zeros for heterogenous batch sizes.
            Shape: [B, max(L_bs)]
        - The number of values of any arange sequence in the batch.
            Shape: [B]
    """
    B = len(starts)
    if ends is None:
        ends = starts
        starts = torch.zeros(B, dtype=dtype, device=device)
    if steps is None:
        steps = torch.ones(B, dtype=dtype, device=device)

    L_bs = ((ends - starts) // steps).long()
    max_L_b = int(L_bs.max())
    aranges = torch.arange(max_L_b, dtype=dtype, device=device)  # [max(L_bs)]
    aranges = starts.unsqueeze(1) + aranges * steps.unsqueeze(1)
    aranges[aranges >= ends.unsqueeze(1)] = 0
    if requires_grad:
        aranges.requires_grad_()

    return aranges, L_bs


def interp_batched(
    x: torch.Tensor,
    xp: torch.Tensor,
    fp: torch.Tensor,
    left: torch.Tensor | None = None,
    right: torch.Tensor | None = None,
    period: torch.Tensor | None = None,
) -> torch.Tensor:
    """Like numpy.interp, but for PyTorch tensors and batched.

    This function is a direct translation of numpy.interp to PyTorch tensors.
    It performs linear interpolation on a batch of 1D tensors.

    Args:
        x: The x-coordinates at which to evaluate the interpolated values.
            Shape: [B, N]
        xp: The x-coordinates of the data points, must be increasing along the
            last dimension.
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
        if period <= 0:
            raise ValueError("period must be positive.")

        xp, sorted_idcs = torch.sort(xp % period, dim=1)
        fp = torch.gather(fp, 1, sorted_idcs)

    # Check if xp is strictly increasing.
    if not torch.all(torch.diff(xp, dim=1) > 0):
        raise ValueError(
            "xp must be strictly increasing along the last dimension."
        )

    # Find indices of neighbours in xp.
    right_idx = torch.searchsorted(xp, x)  # [B, N]
    left_idx = right_idx - 1  # [B, N]

    # Clamp indices to valid range (we will handle the edges later).
    left_idx = torch.clamp(left_idx, min=0, max=M - 1)  # [B, N]
    right_idx = torch.clamp(right_idx, min=0, max=M - 1)  # [B, N]

    # Gather neighbour values.
    x_left = torch.gather(xp, 1, left_idx)  # [B, N]
    x_right = torch.gather(xp, 1, right_idx)  # [B, N]
    y_left = torch.gather(fp, 1, left_idx)  # [B, N]
    y_right = torch.gather(fp, 1, right_idx)  # [B, N]

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


def sample_unique_batched(
    L_bs: torch.Tensor, max_L_b: int, num_samples: int
) -> torch.Tensor:
    """Sample unique indices i in [0, L_b-1] for each element in the batch.

    Warning: If the number of valid values in an element is less than the
    number of samples, then only the first L_b indices are unique. The
    remaining indices are sampled with replacement.

    Args:
        L_bs: The number of valid values for each element in the batch.
            Shape: [B]
        max_L_b: The maximum number of values of any element in the batch.
        num_samples: The number of indices to sample for each element in the
            batch.

    Returns:
        The sampled unique indices.
            Shape: [B, num_samples]
    """
    # Select unique elements for each sample in the batch.
    # If the number of elements is less than the number of samples, we
    # uniformly# sample with replacement. To do this, the
    # .clamp(min=num_samples) and % L_b operations are used.
    weights = mask_padding_batched(
        L_bs.clamp(min=num_samples), max_L_b
    ).double()  # [B, max(L_bs)]
    return (
        torch.multinomial(weights, num_samples)  # [B, num_samples]
        % L_bs.unsqueeze(1)  # [B, num_samples]
    )  # [B, num_samples]  # fmt: skip


def sample_unique_pairs_batched(
    L_bs: torch.Tensor, max_L_b: int, num_samples: int
) -> torch.Tensor:
    """Sample unique pairs of indices (i, j), where i and j are in [0, L_b-1].

    Warning: If the number of valid values in an element is less than the
    number of samples, then only the first L_b * (L_b - 1) // 2 pairs are
    unique. The remaining pairs are sampled with replacement.

    Args:
        L_bs: The number of valid values for each element in the batch.
            Shape: [B]
        max_L_b: The maximum number of valid values.
        num_samples: The number of pairs to sample.

    Returns:
        The sampled unique pairs of indices.
            Shape: [B, num_samples, 2]
    """
    device = L_bs.device

    # Compute the number of unique pairs of indices.
    P_bs = L_bs * (L_bs - 1) // 2  # [B]
    max_P_b = max_L_b * (max_L_b - 1) // 2

    # Select unique pairs of elements for each sample in the batch.
    idcs_pairs = sample_unique_batched(
        P_bs, max_P_b, num_samples
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
    # This is done using the `max_P_b - 1 - triu_idcs` trick. However, the
    # order of the elements is still in reverse, so when indexing, we index at
    # `-idcs_pairs - 1` instead of `idcs_pairs`.
    triu_idcs = torch.triu_indices(
        max_L_b, max_L_b, 1, device=device
    )  # [2, max(P_bs)]
    triu_idcs = max_L_b - 1 - triu_idcs
    idcs_elements = triu_idcs[:, -idcs_pairs - 1]  # [2, B, num_samples]
    return idcs_elements.permute(1, 2, 0)  # [B, num_samples, 2]
