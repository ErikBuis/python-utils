from __future__ import annotations

import random
import warnings
from typing import Any, Callable, Literal, overload

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import (
    collate_str_fn,
    default_collate_fn_map,
)
from torch.utils.data.dataloader import default_collate

# Allow the dataset to return None when a sample is corrupted. When it does,
# make torch's default collate function replace it with another sample.
default_collate_fn_map.update({type(None): collate_str_fn})


def interp(
    x: torch.Tensor,
    xp: torch.Tensor,
    fp: torch.Tensor,
    left: float | None = None,
    right: float | None = None,
    period: float | None = None,
) -> torch.Tensor:
    """Like numpy.interp(), but for PyTorch tensors.

    This function is a direct translation of numpy.interp() to PyTorch tensors.
    It performs linear interpolation on a 1D tensor.

    Args:
        x: The x-coordinates at which to evaluate the interpolated values.
            Shape: [N]
        xp: The x-coordinates of the data points, must be increasing.
            Shape: [M]
        fp: The y-coordinates of the data points, same shape as xp.
            Shape: [M]
        left: Value to return for x < xp[0], default is fp[0].
        right: Value to return for x > xp[-1], default is fp[-1].
        period: A period for the x-coordinates. This parameter allows the
            proper interpolation of angular x-coordinates. Parameters left and
            right are ignored if period is specified.

    Returns:
        The interpolated values for each batch.
            Shape: [N]
    """
    M = len(xp)

    # Handle periodic interpolation.
    if period is not None:
        if period <= 0:
            raise ValueError("period must be positive.")

        xp, sorted_idcs = torch.sort(xp % period)
        fp = fp[sorted_idcs]

    # Check if xp is strictly increasing.
    if not torch.all(torch.diff(xp) > 0):
        raise ValueError(
            "xp must be strictly increasing along the last dimension."
        )

    # Find indices of neighbours in xp.
    right_idx = torch.searchsorted(xp, x)  # [N]
    left_idx = right_idx - 1  # [N]

    # Clamp indices to valid range (we will handle the edges later).
    left_idx = torch.clamp(left_idx, min=0, max=M - 1)  # [N]
    right_idx = torch.clamp(right_idx, min=0, max=M - 1)  # [N]

    # Gather neighbour values.
    x_left = xp[left_idx]  # [N]
    x_right = xp[right_idx]  # [N]
    y_left = fp[left_idx]  # [N]
    y_right = fp[right_idx]  # [N]

    # Avoid division by zero for x_left == x_right.
    denom = x_right - x_left  # [N]
    denom[denom == 0] = 1
    p = (x - x_left) / denom  # [N]

    # Perform interpolation.
    y = y_left + p * (y_right - y_left)  # [N]

    # Handle left edge.
    if left is None:
        left = fp[0].item()
    y[x < xp[[0]]] = left

    # Handle right edge.
    if right is None:
        right = fp[-1].item()
    y[x > xp[[-1]]] = right

    return y


def cumsum_start_0(
    t: torch.Tensor,
    dim: int | None = None,
    dtype: torch.dtype | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Like torch.cumsum, but adds a zero at the start of the tensor.

    Args:
        a: Input tensor.
            Shape: [N_0, ..., N_dim, ..., N_{D-1}]
        dim: Dimension along which the cumulative sum is computed. The default
            (None) is to compute the cumsum over the flattened tensor.
        dtype: Type of the returned tensor and of the accumulator in which the
            elements are summed. If dtype is not specified, it defaults to the
            dtype of a.
        out: Alternative output tensor in which to place the result. It must
            have the same shape and buffer length as the expected output but
            the type will be cast if necessary.
            Shape: [N_0, ..., N_dim + 1, ..., N_{D-1}]

    Returns:
        A new tensor holding the result returned unless out is specified, in
        which case a reference to out is returned. The result has the same
        size as a except along the requested dimension.
            Shape: [N_0, ..., N_dim + 1, ..., N_{D-1}]
    """
    device = t.device

    if dim is None:
        t = t.flatten()
        dim = 0

    if dtype is None:
        dtype = t.dtype

    if out is not None:
        idx = [slice(None)] * t.ndim
        idx[dim] = 0  # type: ignore
        out[tuple(idx)] = 0
        idx[dim] = slice(1, None)
        torch.cumsum(t, dim=dim, dtype=dtype, out=out[tuple(idx)])
        return out

    shape = list(t.shape)
    shape[dim] = 1
    zeros = torch.zeros(shape, device=device, dtype=dtype)
    cumsum = torch.cumsum(t, dim=dim, dtype=dtype)
    return torch.concat([zeros, cumsum], dim=dim)


def to_tensor(
    object: object,
    device: torch.device | str | int | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Convert an object to a tensor.

    Warning: Does not copy the data if possible. Thus, the returned tensor
    could share memory with the original object.

    Args:
        object: The object to convert to a tensor.
        device: The desired device of the returned tensor. If None, the
            device of the object will be used.
        dtype: The desired data type of the returned tensor. If None, the
            dtype of the object will be used.
    """
    # Torch supported types: bool, uint8, int8, int16, int32, int64, float16,
    # float32, float64, complex64, and complex128.
    if isinstance(object, torch.Tensor):
        return object.to(device=device, dtype=dtype)

    if isinstance(object, np.ndarray):
        try:
            return torch.from_numpy(object).to(device=device, dtype=dtype)
        except TypeError as e:
            # Try to convert to a supported type before calling from_numpy().
            np2torch_fallbacks = {
                "uint16": np.int16,  # torch.int16
                "uint32": np.int32,  # torch.int32
                "uint64": np.int64,  # torch.int64
                "float128": np.float64,  # torch.float64
                "complex256": np.complex128,  # torch.complex128
            }
            fallback = np2torch_fallbacks.get(object.dtype.name)
            # Re-raise the original error if there is no fallback.
            if fallback is None:
                raise e
            # Warn the user that we are converting to a different type.
            warnings.warn(
                f"Can't convert np.ndarray of type {object.dtype} to tensor."
                f" Falling back to type {fallback}."
            )
            return torch.from_numpy(object.astype(fallback)).to(
                device=device, dtype=dtype
            )

    # Last resort: because numpy recognizes more array-like types than torch,
    # we try to convert to numpy first.
    try:
        return to_tensor(np.array(object), device=device, dtype=dtype)
    except Exception as e:
        raise TypeError(
            f"Could not convert object of type {type(object)} to tensor."
        ) from e


def swap_idcs_vals(x: torch.Tensor) -> torch.Tensor:
    """Swap the indices and values of a 1D tensor.

    The input tensor is assumed to contain exactly all integers from 0 to
    x.shape[0] - 1, in any order.

    Warning: This function does not explicitly check if the input tensor
    contains no duplicates. If x contains duplicates, no error will be raised
    and undefined behaviour will occur!

    Args:
        x: The tensor to swap.
            Shape: [N]

    Returns:
        The swapped tensor.
            Shape: [N]

    Examples:
        >>> x = torch.tensor([2, 3, 0, 4, 1])
        >>> swap_idcs_vals(x)
        tensor([2, 4, 0, 1, 3])
    """
    if x.ndim != 1:
        raise ValueError("x must be 1D.")

    x_swapped = torch.empty_like(x)
    x_swapped[x] = torch.arange(len(x), device=x.device)
    return x_swapped


def swap_idcs_vals_duplicates(x: torch.Tensor) -> torch.Tensor:
    """Swap the indices and values of a 1D tensor, allowing duplicates.

    The input tensor is assumed to contain integers from 0 to M <= N, in any
    order, and may contain duplicates.

    The output tensor will contain exactly all integers from 0 to len(x) - 1,
    in any order.

    If the input doesn't contain duplicates, you should use swap_idcs_vals()
    instead since it is faster (especially for large tensors).

    Args:
        x: The tensor to swap.
            Shape: [N]

    Returns:
        The swapped tensor.
            Shape: [N]

    Examples:
        >>> x = torch.tensor([1, 2, 0, 1, 2])
        >>> swap_idcs_vals_duplicates(x)
        tensor([2, 0, 3, 1, 4])
    """
    if x.ndim != 1:
        raise ValueError("x must be 1D.")

    # Believe it or not, this O(n log n) algorithm is actually faster than a
    # native implementation that uses a Python for loop with complexity O(n).
    return torch.sort(x).indices


def ravel_multi_index(
    multi_index: torch.Tensor,
    dims: torch.Tensor,
    mode: Literal["raise", "wrap", "clip"] = "raise",
    order: Literal["C", "F"] = "C",
) -> torch.Tensor:
    """Like numpy.ravel_multi_index(), but for PyTorch tensors.

    Converts a tuple of index arrays into an array of flat indices, applying
    boundary modes to the multi-index.

    Warning: Unlike numpy.ravel_multi_index(), this function does not raise an
    error if integer overflow occurs. Instead, it will silently wrap around
    and return incorrect results. Be careful when using this function with
    large tensors.

    Warning: Unlike numpy.ravel_multi_index(), this function does not support
    the mode being a tuple of modes.

    Args:
        multi_index: A tensor of shape [K, N_0, ..., N_{D-1}] or a tuple
            containing K [N_0, ..., N_{D-1}]-shaped sequences. K refers to the
            amount of elements in the tuples.
        dims: The shape of the array into which the indices point.
            Shape: [K]
        mode: Specifies how out-of-bounds indices are handled. Can specify
            either one mode or a tuple of modes, one mode per index.
            - "raise": Raise an error (default).
            - "wrap": Wrap around.
            - "clip": Clip to the range.
            In "clip" mode, a negative index which would normally wrap will
            clip to 0 instead.
        order: Determines whether the multi-index should be viewed as indexing
            in row-major (C-style) or column-major (Fortran-style) order.

    Returns:
        Tensor of indices into the flattened version of an array of shape dims.
            Shape: [N_0, ..., N_{D-1}]
    """
    if isinstance(multi_index, tuple):
        multi_index = torch.stack(multi_index)  # [K, N_0, ..., N_{D-1}]

    if order == "C":
        factors = torch.concat([
            dims[1:].flip(0).cumprod(0, dtype=torch.int64).flip(0),
            torch.ones(1, device=dims.device, dtype=torch.int64),
        ])  # [K]
    elif order == "F":
        factors = torch.concat([
            torch.ones(1, device=dims.device, dtype=torch.int64),
            dims[:-1].cumprod(0, dtype=torch.int64),
        ])  # [K]
    else:
        raise ValueError("only 'C' or 'F' order is permitted")

    dims = dims.reshape(
        -1, *(1 for _ in range(multi_index.ndim - 1))
    )  # [K, 1, ..., 1]
    factors = factors.reshape(
        -1, *([1] * (multi_index.ndim - 1))
    )  # [K, 1, ..., 1]

    if mode == "raise":
        # Raise an error if the multi-index is out of bounds.
        if not torch.all(multi_index.abs() < dims):
            raise ValueError(
                "invalid dims: array size defined by dims is larger than the"
                " maximum possible size."
            )
    elif mode == "wrap":
        # Wrap around the multi-index.
        multi_index = multi_index % dims
    elif mode == "clip":
        # Clip the multi-index to the range.
        multi_index = torch.clamp(multi_index, torch.as_tensor(0), dims - 1)
    else:
        raise ValueError(
            f"clipmode must be one of 'clip', 'raise', or 'wrap' (got '{mode}')"
        )

    return torch.sum(multi_index * factors, dim=0)  # [N_0, ..., N_{D-1}]


def lexsort(
    keys: torch.Tensor | tuple[torch.Tensor, ...], dim: int = -1
) -> torch.Tensor:
    """Like np.lexsort(), but for PyTorch tensors.

    Perform an indirect stable sort using a sequence of keys.

    Given multiple sorting keys, which can be interpreted as elements of a
    tuple, lexsort returns an array of integer indices that describes the sort
    order of the given tuples. The last key in the tuple is used for the
    primary sort order, the second-to-last key for the secondary sort order,
    and so on. The first dimension is always interpreted as the dimension
    along which the tuples lie. Sorting is done according to the last row,
    second last row etc.

    Args:
        keys: Tensor of shape [K, N_0, ..., N_dim, ..., N_{D-1}] or a tuple
            containing K [N_0, ..., N_dim, ..., N_{D-1}]-shaped sequences. K
            refers to the amount of elements in the tuples. The last element
            is the primary sort key.
        dim: Dimension to be indirectly sorted. By default, sort over the last
            dimension.

    Returns:
        Tensor of indices that sort the keys along the specified axis.
            Shape: [N_0, ..., N_dim, ..., N_{D-1}]

    Examples:
        >>> lexsort((torch.tensor([ 1, 17, 18]),
        >>>          torch.tensor([23, 10,  9]),
        >>>          torch.tensor([14, 12,  0]),
        >>>          torch.tensor([19,  5,  6]),
        >>>          torch.tensor([21, 20, 22]),
        >>>          torch.tensor([ 7,  3,  8]),
        >>>          torch.tensor([13,  4,  2]),
        >>>          torch.tensor([15, 11, 16])))
        tensor([1, 0, 2])

        >>> lexsort(torch.tensor([[4, 8, 2, 8, 3, 7, 3],
        >>>                       [9, 4, 0, 4, 0, 4, 1],
        >>>                       [1, 5, 1, 4, 3, 4, 4]]))
        tensor([2, 0, 4, 6, 5, 3, 1])
    """
    if isinstance(keys, tuple):
        keys = torch.stack(keys)  # [K, N_0, ..., N_dim, ..., N_{D-1}]

    # If the tensor is an integer tensor, first try sorting by representing
    # each of the "tuples" as a single integer. This is much faster than
    # lexsorting along the given dimension.
    if keys.dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
        # Compute the minimum and maximum values for each key.
        dims_flat = tuple(range(1, keys.ndim))
        maxs = torch.amax(keys, dim=dims_flat, keepdim=True)  # [K, 1, ..., 1]
        mins = torch.amin(keys, dim=dims_flat, keepdim=True)  # [K, 1, ..., 1]
        extents = (maxs - mins + 1).squeeze(dim=dims_flat)  # [K]
        keys_dense = keys - mins  # [K, N_0, ..., N_dim, ..., N_{D-1}]

        # Convert the tuples to single integers.
        idcs = ravel_multi_index(
            keys_dense, extents, mode="raise", order="F"
        )  # [N_0, ..., N_dim, ..., N_{D-1}]

        # Sort the integers.
        return torch.argsort(idcs, dim=dim)  # [N_0, ..., N_dim, ..., N_{D-1}]

    # If the tensor is not an integer tensor or if overflow would occur when
    # converting to integers, we have to use np.lexsort(). Unfortunately,
    # torch doesn't have a lexsort() equivalent, so we have to use numpy here.
    return torch.from_numpy(
        np.lexsort(keys.detach().cpu().numpy(), axis=dim)
    ).to(keys.device)


def lexsort_along(
    x: torch.Tensor, dim: int = -1
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sort a tensor along dim, taking all others as constant tuples.

    This is like torch.sort(), but it doesn't sort along the other dimensions.
    As such, the other dimensions are treated as tuples. This function is
    roughly equivalent to the following Python code, but it is much faster.
    >>> torch.stack(
    >>>     sorted(
    >>>         x.unbind(dim),
    >>>         key=lambda t: t.tolist(),
    >>>     ),
    >>>     dim=dim,
    >>> )

    The sort is always stable, meaning that the order of equal elements is
    preserved.

    Args:
        x: The input tensor.
            Shape: [N_0, ..., N_dim, ..., N_{D-1}]
        dim: The dimension to sort along.

    Returns:
        Tuple containing:
        - Sorted version of x.
            Shape: [N_0, ..., N_dim, ..., N_{D-1}]
        - The backmap tensor, which contains the indices of the sorted values
            in the original input.
            The sorted version of x can be retrieved as follows:
            >>> x_sorted = x.index_select(dim, backmap)
            Shape: [N_dim]

    Examples:
        >>> x = torch.tensor([
        >>>     [2, 1],
        >>>     [3, 0],
        >>>     [1, 2],
        >>>     [1, 3],
        >>> ])
        >>> dim = 0

        >>> x_sorted, backmap = lexsort_along(x, dim=dim)
        >>> x_sorted
        tensor([[1, 2],
                [1, 3],
                [2, 1],
                [3, 0]])
        >>> backmap
        tensor([2, 3, 0, 1]))

        >>> # Get the lexicographically sorted version of x:
        >>> x.index_select(dim, backmap)
        tensor([[1, 2],
                [1, 3],
                [2, 1],
                [3, 0]])
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
    N_dim = x.shape[dim]

    if x.ndim == 1:
        y = x.unsqueeze(0)  # [1, N_dim]
    else:
        y = x.movedim(
            dim, -1
        )  # [N_0, ..., N_{dim-1}, N_{dim+1}, ..., N_{D-1}, N_dim]
        y = y.reshape(
            -1, N_dim
        )  # [N_0 * ... * N_{dim-1} * N_{dim+1} * ... * N_{D-1}, N_dim]
    y = y.flip(
        dims=(0,)
    )  # [N_0 * ... * N_{dim-1} * N_{dim+1} * ... * N_{D-1}, N_dim]
    backmap = lexsort(y, dim=-1)  # [N_dim]

    # Sort the tensor along the given dimension.
    x_sorted = x.index_select(dim, backmap)  # [N_0, ..., N_dim, ..., N_{D-1}]

    # Finally, we return the sorted tensor and the backmap.
    return x_sorted, backmap


@overload
def unique_consecutive(  # type: ignore
    x: torch.Tensor,
    return_inverse: Literal[False] = ...,
    return_counts: Literal[False] = ...,
    dim: int | None = None,
) -> torch.Tensor:
    pass


@overload
def unique_consecutive(
    x: torch.Tensor,
    return_inverse: Literal[True] = ...,
    return_counts: Literal[False] = ...,
    dim: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    pass


@overload
def unique_consecutive(
    x: torch.Tensor,
    return_inverse: Literal[False] = ...,
    return_counts: Literal[True] = ...,
    dim: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    pass


@overload
def unique_consecutive(
    x: torch.Tensor,
    return_inverse: Literal[True] = ...,
    return_counts: Literal[True] = ...,
    dim: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pass


def unique_consecutive(
    x: torch.Tensor,
    return_inverse: bool = False,
    return_counts: bool = False,
    dim: int | None = None,
) -> (
    torch.Tensor
    | tuple[torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
):
    """Like torch.unique_consecutive, but WAY more efficient.

    The returned unique elements are retrieved along the requested dimension,
    taking all the other dimensions as constant tuples.

    Args:
        x: The input tensor. If it contains equal values, they must be
            consecutive along the given dimension.
            Shape: [N_0, ..., N_dim, ..., N_{D-1}]
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
        - The unique elements.
            Shape: [N_0, ..., N_{dim-1}, U, N_{dim+1}, ..., N_{D-1}]
        - (Optional) If return_inverse is True, the indices where elements
            in the original input ended up in the returned unique values.
            The original tensor can be reconstructed as follows:
            >>> x_reconstructed = uniques.index_select(dim, inverse)
            Shape: [N_dim]
        - (Optional) If return_counts is True, the counts for each unique
            element.
            Shape: [U]

    Examples:
        >>> # 1D example: -----------------------------------------------------
        >>> x = torch.tensor([9, 9, 9, 9, 10, 10])
        >>> dim = 0

        >>> uniques, inverse, counts = unique_consecutive(
        >>>     x, return_inverse=True, return_counts=True, dim=dim
        >>> )
        >>> uniques
        tensor([ 9, 10])
        >>> inverse
        tensor([0, 0, 0, 0, 1, 1])
        >>> counts
        tensor([4, 2])

        >>> # Reconstruct the original tensor:
        >>> uniques.index_select(dim, inverse)
        tensor([ 9,  9,  9,  9, 10, 10])

        >>> # 2D example: -----------------------------------------------------
        >>> x = torch.tensor([
        >>>     [7,  9,  9, 10],
        >>>     [8, 10, 10,  9],
        >>>     [9,  8,  8,  7],
        >>>     [9,  7,  7,  7],
        >>> ])
        >>> dim = 1

        >>> uniques, inverse, counts = unique_consecutive(
        >>>     x, return_inverse=True, return_counts=True, dim=dim
        >>> )
        >>> uniques
        tensor([[ 7,  9, 10],
                [ 8, 10,  9],
                [ 9,  8,  7],
                [ 9,  7,  7]])
        >>> inverse
        tensor([0, 1, 1, 2])
        >>> counts
        tensor([1, 2, 1])

        >>> # Reconstruct the original tensor:
        >>> uniques.index_select(dim, inverse)
        tensor([[ 7,  9,  9, 10],
                [ 8, 10, 10,  9],
                [ 9,  8,  8,  7],
                [ 9,  7,  7,  7]])

        >>> # 3D example: -----------------------------------------------------
        >>> x = torch.tensor([
        >>>     [
        >>>         [0, 1, 2, 2],
        >>>         [4, 6, 5, 5],
        >>>         [9, 8, 7, 7],
        >>>     ],
        >>>     [
        >>>         [4, 2, 8, 8],
        >>>         [3, 3, 7, 7],
        >>>         [0, 2, 1, 1]
        >>>     ],
        >>> ])
        >>> dim = 2

        >>> uniques, inverse, counts = unique_consecutive(
        >>>     x, return_inverse=True, return_counts=True, dim=dim
        >>> )
        >>> uniques
        tensor([[[0, 1, 2],
                 [4, 6, 5],
                 [9, 8, 7]],
                [[4, 2, 8],
                 [3, 3, 7],
                 [0, 2, 1]]])
        >>> inverse
        tensor([0, 1, 2, 2])
        >>> counts
        tensor([1, 1, 2])

        >>> # Reconstruct the original tensor:
        >>> uniques.index_select(dim, inverse)
        tensor([[[0, 1, 2, 2],
                 [4, 6, 5, 5],
                 [9, 8, 7, 7]],
                [[4, 2, 8, 8],
                 [3, 3, 7, 7],
                 [0, 2, 1, 1]])
    """
    if dim is None:
        raise NotImplementedError(
            "dim=None is not implemented yet. Please specify a dimension"
            " explicitly."
        )

    N_dim = x.shape[dim]

    # Flatten all dimensions except the one we want to operate on.
    if x.ndim == 1:
        y = x.unsqueeze(0)  # [1, N_dim]
    else:
        y = x.movedim(
            dim, -1
        )  # [N_0, ..., N_{dim-1}, N_{dim+1}, ..., N_{D-1}, N_dim]
        y = y.reshape(
            -1, N_dim
        )  # [N_0 * ... * N_{dim-1} * N_{dim+1} * ... * N_{D-1}, N_dim]

    # Find the indices where the values change.
    is_change = torch.concat([
        (
            torch.ones(1, device=x.device, dtype=torch.bool)
            if N_dim > 0
            else torch.empty(0, device=x.device, dtype=torch.bool)
        ),  # [1] or [0]
        (y[:, :-1] != y[:, 1:]).any(dim=0),  # [N_dim - 1] or [0]
    ])  # [N_dim]

    # Find the unique values.
    idcs = is_change.nonzero(as_tuple=True)[0]  # [U]
    uniques = x.index_select(
        dim, idcs
    )  # [N_0, ..., N_{dim-1}, U, N_{dim+1}, ..., N_{D-1}]

    # Calculate auxiliary values.
    aux = []
    if return_inverse:
        # Find the indices where the elements in the original input ended up
        # in the returned unique values.
        inverse = is_change.cumsum(dim=0) - 1  # [N_dim]
        aux.append(inverse)
    if return_counts:
        # Find the counts for each unique element.
        counts = torch.diff(
            torch.concat([
                idcs,  # [U]
                (
                    torch.full((1,), N_dim, device=x.device, dtype=torch.int64)
                    if N_dim > 0
                    else torch.empty(0, device=x.device, dtype=torch.int64)
                ),  # [1] or [0]
            ])
        )  # [U]
        aux.append(counts)

    if aux:
        return uniques, *aux
    else:
        return uniques


@overload
def unique(  # type: ignore
    x: torch.Tensor,
    return_backmap: Literal[False] = ...,
    return_inverse: Literal[False] = ...,
    return_counts: Literal[False] = ...,
    dim: int | None = None,
) -> torch.Tensor:
    pass


@overload
def unique(
    x: torch.Tensor,
    return_backmap: Literal[True] = ...,
    return_inverse: Literal[False] = ...,
    return_counts: Literal[False] = ...,
    dim: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    pass


@overload
def unique(
    x: torch.Tensor,
    return_backmap: Literal[False] = ...,
    return_inverse: Literal[True] = ...,
    return_counts: Literal[False] = ...,
    dim: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    pass


@overload
def unique(
    x: torch.Tensor,
    return_backmap: Literal[False] = ...,
    return_inverse: Literal[False] = ...,
    return_counts: Literal[True] = ...,
    dim: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    pass


@overload
def unique(
    x: torch.Tensor,
    return_backmap: Literal[True] = ...,
    return_inverse: Literal[True] = ...,
    return_counts: Literal[False] = ...,
    dim: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pass


@overload
def unique(
    x: torch.Tensor,
    return_backmap: Literal[True] = ...,
    return_inverse: Literal[False] = ...,
    return_counts: Literal[True] = ...,
    dim: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pass


@overload
def unique(
    x: torch.Tensor,
    return_backmap: Literal[False] = ...,
    return_inverse: Literal[True] = ...,
    return_counts: Literal[True] = ...,
    dim: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pass


@overload
def unique(
    x: torch.Tensor,
    return_backmap: Literal[True] = ...,
    return_inverse: Literal[True] = ...,
    return_counts: Literal[True] = ...,
    dim: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pass


def unique(
    x: torch.Tensor,
    return_backmap: bool = False,
    return_inverse: bool = False,
    return_counts: bool = False,
    dim: int | None = None,
) -> (
    torch.Tensor
    | tuple[torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
):
    """Like torch.unique, but WAY more efficient.

    The returned unique elements are retrieved along the requested dimension,
    taking all the other dimensions as constant tuples.

    Args:
        x: The input tensor.
            Shape: [N_0, ..., N_dim, ..., N_{D-1}]
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
            dimension.
            Shape: [N_0, ..., N_{dim-1}, U, N_{dim+1}, ..., N_{D-1}]
        - (Optional) If return_backmap is True, the backmap tensor, which
            contains the indices of the unique values in the original input.
            The sorted version of x can be retrieved as follows:
            >>> x_sorted = x.index_select(dim, backmap)
            Shape: [N_dim]
        - (Optional) If return_inverse is True, the indices where elements
            in the original input ended up in the returned unique values.
            The original tensor can be reconstructed as follows:
            >>> x_reconstructed = uniques.index_select(dim, inverse)
            Shape: [N_dim]
        - (Optional) If return_counts is True, the counts for each unique
            element.
            Shape: [U]

    Examples:
        >>> # 1D example: -----------------------------------------------------
        >>> x = torch.tensor([9, 10, 9, 9, 10, 9])
        >>> dim = 0

        >>> uniques, backmap, inverse, counts = unique(
        >>>     x,
        >>>     return_backmap=True,
        >>>     return_inverse=True,
        >>>     return_counts=True,
        >>>     dim=dim,
        >>> )
        >>> uniques
        tensor([ 9, 10])
        >>> backmap
        tensor([0, 2, 3, 5, 1, 4])
        >>> inverse
        tensor([0, 1, 0, 0, 1, 0])
        >>> counts
        tensor([4, 2])

        >>> # Get the lexicographically sorted version of x:
        >>> x.index_select(dim, backmap)
        tensor([ 9,  9,  9,  9, 10, 10])

        >>> # Reconstruct the original tensor:
        >>> uniques.index_select(dim, inverse)
        tensor([ 9, 10,  9,  9, 10,  9])

        >>> # 2D example: -----------------------------------------------------
        >>> x = torch.tensor([
        >>>     [9, 10, 7, 9],
        >>>     [10, 9, 8, 10],
        >>>     [8, 7, 9, 8],
        >>>     [7, 7, 9, 7],
        >>> ])
        >>> dim = 1

        >>> uniques, backmap, inverse, counts = unique(
        >>>     x,
        >>>     return_backmap=True,
        >>>     return_inverse=True,
        >>>     return_counts=True,
        >>>     dim=dim,
        >>> )
        >>> uniques
        tensor([[ 7,  9, 10],
                [ 8, 10,  9],
                [ 9,  8,  7],
                [ 9,  7,  7]])
        >>> backmap
        tensor([2, 0, 3, 1])
        >>> inverse
        tensor([1, 2, 0, 1])
        >>> counts
        tensor([1, 2, 1])

        >>> # Get the lexicographically sorted version of x:
        >>> x.index_select(dim, backmap)
        tensor([[ 7,  9,  9, 10],
                [ 8, 10, 10,  9],
                [ 9,  8,  8,  7],
                [ 9,  7,  7,  7]])

        >>> # Reconstruct the original tensor:
        >>> uniques.index_select(dim, inverse)
        tensor([[ 9, 10,  7,  9],
                [10,  9,  8, 10],
                [ 8,  7,  9,  8],
                [ 7,  7,  9,  7]])

        >>> # 3D example: -----------------------------------------------------
        >>> x = torch.tensor([
        >>>     [
        >>>         [0, 2, 1, 2],
        >>>         [4, 5, 6, 5],
        >>>         [9, 7, 8, 7],
        >>>     ],
        >>>     [
        >>>         [4, 8, 2, 8],
        >>>         [3, 7, 3, 7],
        >>>         [0, 1, 2, 1],
        >>>     ],
        >>> ])
        >>> dim = 2

        >>> uniques, backmap, inverse, counts = unique(
        >>>     x,
        >>>     return_backmap=True,
        >>>     return_inverse=True,
        >>>     return_counts=True,
        >>>     dim=dim,
        >>> )
        >>> uniques
        tensor([[[0, 1, 2],
                 [4, 6, 5],
                 [9, 8, 7]],
                [[4, 2, 8],
                 [3, 3, 7],
                 [0, 2, 1]]])
        >>> backmap
        tensor([0, 2, 1, 3])
        >>> inverse
        tensor([0, 2, 1, 2])
        >>> counts
        tensor([1, 1, 2])

        >>> # Get the lexicographically sorted version of x:
        >>> x.index_select(dim, backmap)
        tensor([[[0, 1, 2, 2],
                 [4, 6, 5, 5],
                 [9, 8, 7, 7]],
                [[4, 2, 8, 8],
                 [3, 3, 7, 7],
                 [0, 2, 1, 1]]])

        >>> # Reconstruct the original tensor:
        >>> uniques.index_select(dim, inverse)
        tensor([[[0, 2, 1, 2],
                 [4, 5, 6, 5],
                 [9, 7, 8, 7]],
                [[4, 8, 2, 8],
                 [3, 7, 3, 7],
                 [0, 1, 2, 1]])
    """
    if dim is None:
        raise NotImplementedError(
            "dim=None is not implemented yet. Please specify a dimension"
            " explicitly."
        )

    # Sort along the given dimension, taking all the other dimensions as
    # constant tuples. Torch's sort() doesn't work here since it will
    # sort the other dimensions as well.
    x_sorted, backmap = lexsort_along(
        x, dim=dim
    )  # [N_0, ..., N_dim, ..., N_{D-1}], [N_dim]

    out = unique_consecutive(
        x_sorted,
        return_inverse=return_inverse,
        return_counts=return_counts,
        dim=dim,
    )

    aux = []
    if return_backmap:
        aux.append(backmap)
    if return_inverse:
        # The backmap wasn't taken into account by unique_consecutive(), so we
        # have to do it ourselves.
        backmap_inv = swap_idcs_vals(backmap)  # [N_dim]
        aux.append(out[1][backmap_inv])
    if return_counts:
        aux.append(out[-1])

    if aux:
        return out[0], *aux
    return out


def count_freqs_until(x: torch.Tensor, high: int) -> torch.Tensor:
    """Count the frequency of each integer value in range(0, high).

    This differs from torch.unique_consecutive() in that it also counts the
    frequency of elements not present in x (i.e. with a frequency of 0).

    Warning: This function assumes that x is sorted.

    Warning: If x contains values outside the range(0, high), the function will
    crash with an index out of bounds error.

    Args:
        x: The tensor for which to count the frequency of each element.
            Shape: [N]
        high: The value until which to count the frequency of each element.

    Returns:
        The frequency of each element in x in range(0, high).
            Shape: [high]
    """
    freqs = torch.zeros(high, device=x.device, dtype=torch.int64)
    unique, counts = unique_consecutive(x, return_counts=True, dim=0)
    freqs[unique] = counts
    return freqs


def collate_replace_corrupted(
    batch_list: list[Any],
    dataset: Dataset,
    default_collate_fn: Callable | None = None,
) -> Any:
    """Collate function that allows to replace corrupted samples in the batch.

    The given dataset should return None when a sample is corrupted. This
    function will then replace such a sample in the batch with another
    randomly-selected sample from the dataset.

    Warning: Since corrupted samples are replaced with random other samples
    from the dataset, a sample might be sampled multiple times in one pass
    through the dataloader. This implies that a model might be trained on the
    same sample multiple times in one epoch.

    Note: As a DataLoader only accepts collate functions with a single
    argument, you should use functools.partial() to pass your dataset object
    and your default collate function to this function. For example:
    >>> from functools import partial
    >>> dataset = MyDataset()
    >>> collate_fn = partial(collate_replace_corrupted, dataset=dataset)
    >>> dataloader = DataLoader(dataset, collate_fn=collate_fn)

    This function was based on:
    https://stackoverflow.com/a/69578320/15636460

    Args:
        batch_list: List of samples from the DataLoader.
        dataset: Dataset that the DataLoader is passing through.
        default_collate_fn: The collate function to call once the batch has no
            corrupted samples any more. If None,
            torch.utils.data.dataloader.default_collate is called.

    Returns:
        Batch with new samples instead of corrupted ones.
    """
    # Use torch.utils.data.dataloader.default_collate if no other default
    # collate function is specified.
    default_collate_fn = (
        default_collate_fn
        if default_collate_fn is not None
        else default_collate
    )

    # Filter out all corrupted samples.
    B = len(batch_list)
    batch_list = [sample for sample in batch_list if sample is not None]

    # Replace the corrupted samples with other randomly selected samples.
    while len(batch_list) < B:
        sample = dataset[random.randint(0, len(dataset) - 1)]  # type: ignore
        if sample is not None:
            batch_list.append(sample)

    # When the whole batch is fine, apply the default collate function.
    return default_collate_fn(batch_list)
