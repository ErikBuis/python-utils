from functools import lru_cache
from typing import Any, Literal, overload

import torch

from ..modules.torch import count_freqs_until, lexsort


@lru_cache(maxsize=8)
def mask_padding_batched(L_bs: torch.Tensor, max_L_b: int) -> torch.Tensor:
    """Create a mask that indicates which values are valid in each sample.

    Args:
        L_bs: The number of valid values in each sample.
            Shape: [B]
        max_L_b: The maximum number of values of any element in the batch.

    Returns:
        A mask that indicates which values are valid in each sample.
        mask[b, i] is True if i < L_bs[b] and False otherwise.
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
        values: The values to pack. Padding could be arbitrary.
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
        values: The values to pad.
            Shape: [sum(L_bs), *]
        L_bs: The number of valid values in each sample.
            Shape: [B]
        max_L_b: The maximum number of values of any element in the batch.
        padding_value: The value to pad the values with. If None, the values
            are padded with random values. This is faster than padding with
            a specific value.

    Returns:
        The padded values. Padded with padding_value.
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


@overload
def replace_padding_batched(  # type: ignore
    values: torch.Tensor,
    L_bs: torch.Tensor,
    padding_value: Any = 0,
    in_place: Literal[False] = ...,
) -> torch.Tensor:
    pass


@overload
def replace_padding_batched(
    values: torch.Tensor,
    L_bs: torch.Tensor,
    padding_value: Any = 0,
    in_place: Literal[True] = ...,
) -> None:
    pass


def replace_padding_batched(
    values: torch.Tensor,
    L_bs: torch.Tensor,
    padding_value: Any = 0,
    in_place: bool = False,
) -> torch.Tensor | None:
    """Pad the values with padding_value to create a tensor with a fixed size.

    Args:
        values: The values to pad. Padding could be arbitrary.
            Shape: [B, max(L_bs), *]
        L_bs: The number of valid values in each sample.
            Shape: [B]
        padding_value: The value to pad the values with.
        in_place: Whether to perform the operation in-place.

    Returns:
        The padded values. Padded with padding_value.
        If in_place is True, this is None.
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
        values: The values to calculate the mean for. Padded with zeros if
            is_padding_zero is True. Otherwise, padding could be arbitrary.
            Shape: [B, max(L_bs), *]
        L_bs: The number of valid values in each sample.
            Shape: [B]
        is_padding_zero: Whether the values are padded with zeros already.
            Setting this to True when the values are already padded with zeros
            can speed up the calculation.

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
        The standard deviation per dimension for each sample.
            Shape: [B, *]
    """
    means = mean_padding_batched(
        values, L_bs, is_padding_zero=is_padding_zero
    )  # [B, *]
    values_centered = values - means.unsqueeze(1)  # [B, max(L_bs), *]
    return mean_padding_batched(
        values_centered.square(), L_bs
    ).sqrt()  # [B, *]


def min_padding_batched(
    values: torch.Tensor, L_bs: torch.Tensor, is_padding_inf: bool = False
) -> torch.Tensor:
    """Calculate the minimum per dimension for each sample in the batch.

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
    device: torch.device | str | int | None = None,
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
        - A batch of tensors with values in the range [start, end).
            Padded with zeros.
            Shape: [B, max(L_bs)]
        - The number of values of any arange sequence in the batch.
            Shape: [B]
    """
    dtype = dtype if dtype is not None else starts.dtype
    device = device if device is not None else starts.device

    B = len(starts)
    if ends is None:
        ends = starts
        starts = torch.zeros(B, dtype=dtype, device=device)
    if steps is None:
        steps = torch.ones(B, dtype=dtype, device=device)

    L_bs = ((ends - starts) // steps).long()
    max_L_b = int(L_bs.max())
    aranges = torch.arange(max_L_b, dtype=dtype, device=device)
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
    # This is done using the max_P_b - 1 - triu_idcs trick. However, the
    # order of the elements is still in reverse, so when indexing, we index at
    # -idcs_pairs - 1 instead of at idcs_pairs.
    triu_idcs = torch.triu_indices(
        max_L_b, max_L_b, 1, device=device
    )  # [2, max(P_bs)]
    triu_idcs = max_L_b - 1 - triu_idcs
    idcs_elements = triu_idcs[:, -idcs_pairs - 1]  # [2, B, num_samples]
    return idcs_elements.permute(1, 2, 0)  # [B, num_samples, 2]


def swap_idcs_vals_batched(x: torch.Tensor) -> torch.Tensor:
    """Swap the indices and values of a batched of 1D tensors.

    Each row in the input tensor is assumed to contain exactly all integers
    from 0 to x.shape[1] - 1, in any order.

    Warning: This function does not explicitly check if the input tensor
    contains no duplicates. If x contains duplicates, no error will be raised
    and undefined behaviour will occur!

    Args:
        x: The tensor to swap.
            Shape: [B, N]

    Returns:
        The swapped tensor.
            Shape: [B, N]

    Examples:
        >>> x = torch.tensor([
        >>>     [2, 3, 0, 4, 1],
        >>>     [1, 3, 2, 0, 4],
        >>> ])
        >>> swap_idcs_vals_batched(x)
        tensor([[2, 4, 0, 1, 3],
                [3, 0, 2, 1, 4]])
    """
    if x.ndim != 2:
        raise ValueError("x must be a batch of 1D tensors.")

    # TODO I'm pretty sure this scatter_ can be replaced by a gather, and that
    # TODO it would be faster. I should test this.
    x_swapped = torch.empty_like(x)
    x_swapped.scatter_(
        1, x, torch.arange(x.shape[1], device=x.device).repeat(x.shape[0], 1)
    )
    return x_swapped


def index_select_batched(
    values: torch.Tensor, dim: int, idcs: torch.Tensor
) -> torch.Tensor:
    """Select values from a batch of tensors using the given indices.

    Note that dim refers to the index of the dimension to sort along AFTER the
    batch dimension. So e.g. if x has shape [B, N_0, N_1, N_2], then dim=0
    refers to N_0, dim=1 refers to N_1, etc.

    Args:
        values: The values to select from.
            Shape: [B, N_0, ..., N_dim, ..., N_{D-1}]
        dim: The dimension to select along.
        idcs: The indices to select.
            Shape: [B, N_select]

    Returns:
        The selected values.
            Shape: [B, N_0, ..., N_{dim-1}, N_select, N_{dim+1}, ..., N_{D-1}]
    """
    idcs_reshape = [1] * values.ndim
    idcs_reshape[0] = idcs.shape[0]
    idcs_reshape[dim + 1] = idcs.shape[1]
    idcs_expand = list(values.shape)
    idcs_expand[dim + 1] = idcs.shape[1]
    return torch.gather(
        values, dim + 1, idcs.reshape(idcs_reshape).expand(idcs_expand)
    )


def lexsort_along_batched(
    x: torch.Tensor, dim: int = -1
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
    >>>     torch.stack(
    >>>         sorted(
    >>>             x_b.unbind(dim),
    >>>             key=lambda t: t.tolist(),
    >>>         ),
    >>>         dim=dim,
    >>>     )
    >>>     for x_b in x.unbind(0)
    >>> ])

    The sort is always stable, meaning that the order of equal elements is
    preserved.

    Args:
        x: The input tensor.
            Shape: [B, N_0, ..., N_dim, ..., N_{D-1}]
        dim: The dimension to sort along.

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
        >>>     [
        >>>         [2, 1],
        >>>         [3, 0],
        >>>         [1, 2],
        >>>         [1, 3],
        >>>     ],
        >>>     [
        >>>         [1, 2],
        >>>         [1, 5],
        >>>         [3, 4],
        >>>         [2, 1],
        >>>     ],
        >>> ])
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
    # must be a tuple of array-like objects, that are evaluated from last to
    # first. This is quite confusing, so I'll put an example here. If we have:
    # >>> x = tensor([[[15, 13],
    # >>>              [11,  4],
    # >>>              [16,  2]],
    # >>>             [[ 7, 21],
    # >>>              [ 3, 20],
    # >>>              [ 8, 22]],
    # >>>             [[19, 14],
    # >>>              [ 5, 12],
    # >>>              [ 6,  0]],
    # >>>             [[23,  1],
    # >>>              [10, 17],
    # >>>              [ 9, 18]]])
    # And dim=1, then the input to lexsort() must be:
    # >>> lexsort(tensor([[ 1, 17, 18],
    # >>>                 [23, 10,  9],
    # >>>                 [14, 12,  0],
    # >>>                 [19,  5,  6],
    # >>>                 [21, 20, 22],
    # >>>                 [ 7,  3,  8],
    # >>>                 [13,  4,  2],
    # >>>                 [15, 11, 16]]))
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
    backmap = lexsort(y, dim=-1)  # [B, N_dim]

    # Sort the tensor along the given dimension.
    x_sorted = index_select_batched(
        x, dim, backmap
    )  # [B, N_0, ..., N_dim, ..., N_{D-1}]

    # Finally, we return the sorted tensor and the backmap.
    return x_sorted, backmap


@overload
def unique_consecutive_batched(  # type: ignore
    x: torch.Tensor,
    return_inverse: Literal[False] = ...,
    return_counts: Literal[False] = ...,
    dim: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    ...  # fmt: skip


@overload
def unique_consecutive_batched(
    x: torch.Tensor,
    return_inverse: Literal[True] = ...,
    return_counts: Literal[False] = ...,
    dim: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ...  # fmt: skip


@overload
def unique_consecutive_batched(
    x: torch.Tensor,
    return_inverse: Literal[False] = ...,
    return_counts: Literal[True] = ...,
    dim: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ...  # fmt: skip


@overload
def unique_consecutive_batched(
    x: torch.Tensor,
    return_inverse: Literal[True] = ...,
    return_counts: Literal[True] = ...,
    dim: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ...  # fmt: skip


def unique_consecutive_batched(
    x: torch.Tensor,
    return_inverse: bool = False,
    return_counts: bool = False,
    dim: int | None = None,
) -> (
    tuple[torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
):
    """A batched version of torch.unique_consecutive, but WAY more effiecient.

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
        dim: The dimension to operate upon. If None, the unique of the
            flattened input is returned. Otherwise, each of the tensors
            indexed by the given dimension is treated as one of the elements
            to apply the unique operation upon. See examples for more details.

    Returns:
        Tuple containing:
        - The unique elements. Padded with zeros.
            Shape: [B, N_0, ..., N_{dim-1}, max(U_bs), N_{dim+1}, ..., N_{D-1}]
        - The amount of unique elements per batch element.
            Shape: [B]
        - (Optional) If return_inverse is True, the indices where elements
            in the original input ended up in the returned unique values.
            The original tensor can be reconstructed as follows:
            >>> x_reconstructed = index_select_batched(uniques, dim, inverse)
            Shape: [B, N_dim]
        - (Optional) If return_counts is True, the counts for each unique
            element. Padded with zeros.
            Shape: [B, max(U_bs)]

    Examples:
        >>> # 1D example: -----------------------------------------------------
        >>> x = torch.tensor([
        >>>     [9, 9, 9, 9, 10, 10],
        >>>     [8, 8, 7, 7, 9, 9],
        >>> ])
        >>> dim = 0

        >>> uniques, U_bs, inverse, counts = unique_consecutive_batched(
        >>>     x, return_inverse=True, return_counts=True, dim=dim
        >>> )
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
        >>>     [
        >>>         [7,  9,  9, 10],
        >>>         [8, 10, 10,  9],
        >>>         [9,  8,  8,  7],
        >>>         [9,  7,  7,  7],
        >>>     ],
        >>>     [
        >>>         [7,  7,  7,  7],
        >>>         [7,  7,  7, 10],
        >>>         [9,  9,  9,  8],
        >>>         [8,  8,  8,  8],
        >>>     ],
        >>> ])
        >>> dim = 1

        >>> uniques, U_bs, inverse, counts = unique_consecutive_batched(
        >>>     x, return_inverse=True, return_counts=True, dim=dim
        >>> )
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

    B, N_dim = x.shape[0], x.shape[dim + 1]

    # Flatten all dimensions except the one we want to operate on.
    if x.ndim == 2:
        y = x.unsqueeze(1)  # [B, 1, N_dim]
    else:
        y = x.movedim(
            dim + 1, -1
        )  # [B, N_0, ..., N_{dim-1}, N_{dim+1}, ..., N_{D-1}, N_dim]
        y = y.reshape(
            B, -1, N_dim
        )  # [B, N_0 * ... * N_{dim-1} * N_{dim+1} * ... * N_{D-1}, N_dim]

    # Find the indices where the values change.
    is_change = torch.concatenate(
        [
            (
                torch.ones((B, 1), dtype=torch.bool, device=x.device)
                if N_dim > 0
                else torch.empty((B, 0), dtype=torch.bool, device=x.device)
            ),  # [B, 1] or [B, 0]
            (y[:, :, :-1] != y[:, :, 1:]).any(
                dim=1
            ),  # [B, N_dim - 1] or [B, 0]
        ],
        dim=1,
    )  # [B, N_dim]

    # Find the unique values.
    batch_idcs, dim_idcs = is_change.nonzero(
        as_tuple=True
    )  # [sum(U_bs)], [sum(U_bs)]
    U_bs = count_freqs_until(batch_idcs, B)  # [B]
    max_U_bs = int(U_bs.max())
    dim_idcs_padded = pad_packed_batched(
        dim_idcs, U_bs, max_U_bs
    )  # [B, max(U_bs)]
    uniques = index_select_batched(
        x, dim, dim_idcs_padded
    )  # [B, N_0, ..., N_{dim-1}, max(U_bs), N_{dim+1}..., N_{D-1}]
    uniques = uniques.movedim(
        dim + 1, 1
    )  # [B, max(U_bs), N_0, ..., N_{dim-1}, N_{dim+1}, ..., N_{D-1}]
    replace_padding_batched(uniques, U_bs, in_place=True)
    uniques = uniques.movedim(
        1, dim + 1
    )  # [B, N_0, ..., N_{dim-1}, max(U_bs), N_{dim+1}, ..., N_{D-1}]

    # Calculate auxiliary values.
    aux = []
    if return_inverse:
        # Find the indices where the elements in the original input ended up
        # in the returned unique values.
        inverse = is_change.cumsum(dim=1) - 1  # [B, N_dim]
        aux.append(inverse)
    if return_counts:
        # Find the counts for each unique element.
        counts = torch.diff(
            torch.concatenate(
                [
                    dim_idcs_padded,  # [B, max(U_bs)]
                    (
                        torch.full(
                            (B, 1), N_dim, dtype=torch.int64, device=x.device
                        )
                        if N_dim > 0
                        else torch.empty(
                            (B, 0), dtype=torch.int64, device=x.device
                        )
                    ),  # [B, 1] or [B, 0]
                ],
                dim=1,
            ),  # [B, max(U_bs) + 1]
            dim=1,
        )  # [B, max(U_bs)]
        replace_padding_batched(counts, U_bs, in_place=True)
        aux.append(counts)

    return uniques, U_bs, *aux


@overload
def unique_batched(  # type: ignore
    x: torch.Tensor,
    return_backmap: Literal[False] = ...,
    return_inverse: Literal[False] = ...,
    return_counts: Literal[False] = ...,
    dim: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    pass


@overload
def unique_batched(
    x: torch.Tensor,
    return_backmap: Literal[True] = ...,
    return_inverse: Literal[False] = ...,
    return_counts: Literal[False] = ...,
    dim: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pass


@overload
def unique_batched(
    x: torch.Tensor,
    return_backmap: Literal[False] = ...,
    return_inverse: Literal[True] = ...,
    return_counts: Literal[False] = ...,
    dim: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pass


@overload
def unique_batched(
    x: torch.Tensor,
    return_backmap: Literal[False] = ...,
    return_inverse: Literal[False] = ...,
    return_counts: Literal[True] = ...,
    dim: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pass


@overload
def unique_batched(
    x: torch.Tensor,
    return_backmap: Literal[True] = ...,
    return_inverse: Literal[True] = ...,
    return_counts: Literal[False] = ...,
    dim: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pass


@overload
def unique_batched(
    x: torch.Tensor,
    return_backmap: Literal[True] = ...,
    return_inverse: Literal[False] = ...,
    return_counts: Literal[True] = ...,
    dim: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pass


@overload
def unique_batched(
    x: torch.Tensor,
    return_backmap: Literal[False] = ...,
    return_inverse: Literal[True] = ...,
    return_counts: Literal[True] = ...,
    dim: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pass


@overload
def unique_batched(
    x: torch.Tensor,
    return_backmap: Literal[True] = ...,
    return_inverse: Literal[True] = ...,
    return_counts: Literal[True] = ...,
    dim: int | None = None,
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
) -> (
    tuple[torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    | tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]
):
    """A batched version of torch.unique, but WAY more efficient.

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
        dim: The dimension to operate upon. If None, the unique of the
            flattened input is returned. Otherwise, each of the tensors
            indexed by the given dimension is treated as one of the elements
            to apply the unique operation upon. See examples for more details.

    Returns:
        Tuple containing:
        - The unique elements, guaranteed to be sorted along the given
            dimension. Padded with zeros.
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
            element. Padded with zeros.
            Shape: [B, max(U_bs)]

    Examples:
        >>> # 1D example: -----------------------------------------------------
        >>> x = torch.tensor([
        >>>     [9, 10, 9, 9, 10, 9],
        >>>     [8, 7, 9, 9, 8, 7],
        >>> ])
        >>> dim = 0

        >>> uniques, U_bs, backmap, inverse, counts = unique_batched(
        >>>     x,
        >>>     return_backmap=True,
        >>>     return_inverse=True,
        >>>     return_counts=True,
        >>>     dim=dim,
        >>> )
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
        >>>     [
        >>>         [9, 10, 7, 9],
        >>>         [10, 9, 8, 10],
        >>>         [8, 7, 9, 8],
        >>>         [7, 7, 9, 7],
        >>>     ],
        >>>     [
        >>>         [7, 7, 7, 7],
        >>>         [7, 10, 7, 7],
        >>>         [9, 8, 9, 9],
        >>>         [8, 8, 8, 8],
        >>>     ],
        >>> ])
        >>> dim = 1

        >>> uniques, U_bs, backmap, inverse, counts = unique_batched(
        >>>     x,
        >>>     return_backmap=True,
        >>>     return_inverse=True,
        >>>     return_counts=True,
        >>>     dim=dim,
        >>> )
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

    # Sort along the given dimension, taking all the other dimensions as
    # constant tuples. Torch's sort() doesn't work here since it will
    # sort the other dimensions as well.
    x_sorted, backmap = lexsort_along_batched(
        x, dim=dim
    )  # [B, N_0, ..., N_dim, ..., N_{D-1}], [B, N_dim]

    out = unique_consecutive_batched(
        x_sorted,
        return_inverse=return_inverse,  # type: ignore
        return_counts=return_counts,  # type: ignore
        dim=dim,
    )

    aux = []
    if return_backmap:
        aux.append(backmap)
    if return_inverse:
        # The backmap wasn't taken into account by unique_consecutive(), so we
        # have to do it ourselves.
        backmap_inv = swap_idcs_vals_batched(backmap)  # [B, N_dim]
        aux.append(out[2].gather(1, backmap_inv))
    if return_counts:
        aux.append(out[-1])

    return out[0], out[1], *aux
