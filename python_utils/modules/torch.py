import itertools
import operator
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

# Indexing object for the output of unique_with_backmap().
BackMap = tuple[slice | torch.Tensor, ...]


def interp(
    x: torch.Tensor,
    xp: torch.Tensor,
    fp: torch.Tensor,
    left: float | None = None,
    right: float | None = None,
    period: float | None = None,
) -> torch.Tensor:
    """Like numpy.interp, but for PyTorch tensors.

    This function is a direct translation of numpy.interp to PyTorch tensors.
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
            Shape: [N_1, ..., N_dim, ..., N_D]
        dim: Dimension along which the cumulative sum is computed. The default
            (None) is to compute the cumsum over the flattened tensor.
        dtype: Type of the returned tensor and of the accumulator in which the
            elements are summed. If dtype is not specified, it defaults to the
            dtype of a.
        out: Alternative output tensor in which to place the result. It must
            have the same shape and buffer length as the expected output but
            the type will be cast if necessary.
            Shape: [N_1, ..., N_dim + 1, ..., N_D]

    Returns:
        A new tensor holding the result returned unless out is specified, in
        which case a reference to out is returned. The result has the same
        size as a except along the requested dimension.
            Shape: [N_1, ..., N_dim + 1, ..., N_D]
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
    zeros = torch.zeros(shape, dtype=dtype, device=device)
    cumsum = torch.cumsum(t, dim=dim, dtype=dtype)
    return torch.concatenate([zeros, cumsum], dim=dim)


def to_tensor(
    object: object,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Convert an object to a tensor.

    Warning: Does not copy the data if possible. Thus, the returned tensor
    could share memory with the original object.

    Args:
        object: The object to convert to a tensor.
        dtype: The desired data type of the returned tensor. If None, the
            dtype of the object will be used.
        device: The desired device of the returned tensor. If None, the
            device of the object will be used.
    """
    # Torch supported types: bool, uint8, int8, int16, int32, int64, float16,
    # float32, float64, complex64, and complex128.
    if isinstance(object, torch.Tensor):
        return object.to(device, dtype)

    if isinstance(object, np.ndarray):
        try:
            return torch.from_numpy(object).to(device, dtype)
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
            return torch.from_numpy(object.astype(fallback)).to(device, dtype)

    # Last resort: because numpy recognizes more array-like types than torch,
    # we try to convert to numpy first.
    try:
        return to_tensor(np.array(object), dtype=dtype, device=device)
    except Exception as e:
        raise TypeError(
            f"Could not convert object of type {type(object)} to tensor."
        ) from e


def swap_idcs_vals(x: torch.Tensor) -> torch.Tensor:
    """Swap the indices and values of a 1D tensor.

    The input tensor is assumed to contain exactly all integers from 0 to
    len(x) - 1, in any order.

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
    if len(x.shape) != 1:
        raise ValueError("x must be 1D.")

    x_swapped = torch.empty_like(x)
    x_swapped.scatter_(0, x, torch.arange(len(x), device=x.device))
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
    if len(x.shape) != 1:
        raise ValueError("x must be 1D.")

    # Believe it or not, this O(n log n) algorithm is actually faster than a
    # native implementation that uses a Python for loop with complexity O(n).
    return torch.sort(x).indices


def multidim_indexing(
    vals: torch.Tensor, idcs: torch.Tensor, dim: int
) -> torch.Tensor:
    """Perform multidimensional indexing on a tensor.

    This function is useful if you have the result of e.g. torch.argsort(vals)
    and you want to find the sorted values.

    Args:
        vals: The tensor to index.
            Shape: [N_1, ..., N_dim, ..., N_D]
        idcs: The indices to use for indexing.
            Shape: [N_1, ..., N_dim, ..., N_D]
        dim: The dimension to index along.

    Returns:
        The indexed tensor.
            Shape: [N_1, ..., N_dim, ..., N_D]

    Example:
        # >>> vals = torch.tensor([[[5, 6, 2, 8, 3],
        # >>>                       [2, 2, 0, 8, 7],
        # >>>                       [4, 5, 4, 1, 3],
        # >>>                       [5, 5, 4, 4, 4]],
        # >>>                      [[2, 4, 4, 3, 5],
        # >>>                       [2, 7, 6, 1, 2],
        # >>>                       [3, 1, 7, 3, 5],
        # >>>                       [0, 0, 6, 2, 8]],
        # >>>                      [[0, 0, 7, 0, 1],
        # >>>                       [6, 3, 1, 3, 8],
        # >>>                       [3, 5, 0, 0, 6],
        # >>>                       [7, 4, 0, 7, 6]]])
        >>> idcs = torch.argsort(vals, dim=2)
        >>> multidim_indexing(vals, idcs, dim=2)
        tensor([[[2, 2, 0, 1, 3],
                 [4, 5, 2, 4, 3],
                 [5, 5, 4, 8, 4],
                 [5, 6, 4, 8, 7]],
                [[0, 0, 4, 1, 2],
                 [2, 1, 6, 2, 5],
                 [2, 4, 6, 3, 5],
                 [3, 7, 7, 3, 8]],
                [[0, 0, 0, 0, 1],
                 [3, 3, 0, 0, 6],
                 [6, 4, 1, 3, 6],
                 [7, 5, 7, 7, 8]]])
    """
    # The code below is equivalent to the following for ndim=4 and dim=1:
    # >>> vals[
    # >>>     torch.arange(vals.shape[0]).reshape(-1, 1, 1, 1),
    # >>>     idcs,
    # >>>     torch.arange(vals.shape[2]).reshape(1, 1, -1, 1),
    # >>>     torch.arange(vals.shape[3]).reshape(1, 1, 1, -1),
    # >>> ]
    return vals[
        tuple(
            (
                torch.arange(vals.shape[d]).reshape(
                    [1 if d != dim else -1 for d in range(vals.ndim)]
                )
                if d != dim
                else idcs
            )
            for d in range(vals.ndim)
        )
    ]


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
        keys: Tensor of shape [K, N_1, ..., N_dim, ..., N_D] or a tuple
            containing K [N_1, ..., N_dim, ..., N_D]-shaped sequences. K
            refers to the amount of elements in the tuples. The last element
            is the primary sort key.
        dim: Dimension to be indirectly sorted. By default, sort over the last
            dimension.

    Returns:
        Tensor of indices that sort the keys along the specified axis.
            Shape: [N_1, ..., N_dim, ..., N_D]

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
        keys = torch.stack(keys, dim=0)

    K = len(keys)
    ndim = len(keys.shape)
    device = keys.device

    # If the tensor is an integer tensor, first try sorting by representing
    # each of the "tuples" as a single integer, let's call it the "t-value".
    # This is much faster than lexsorting along the given dimension.
    if keys.dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
        # This method can't be used if the largest t-value would cause integer
        # overflow. Thus, we have to check if the maximum t-value would be
        # outside the int64 range.
        dims_not_zero = tuple(range(1, ndim))
        dims_expanded = tuple(1 if d != 0 else K for d in range(ndim))
        mins = torch.amin(keys, dim=dims_not_zero)  # [K]
        maxs = torch.amax(keys, dim=dims_not_zero)  # [K]
        extents = maxs - mins + 1  # [K]
        # Python ints can't overflow, so we can safely do cumprod here.
        extents_cumprod = list(
            itertools.accumulate(extents.tolist(), operator.mul)
        )  # [K]
        if extents_cumprod[-1] <= 2**63 - 1:
            extents_cumprod = torch.tensor(
                [1] + extents_cumprod[:-1], device=device, dtype=torch.int64
            )  # [K]

            # Convert to t-values.
            t_values = torch.sum(
                (keys - mins.reshape(dims_expanded))
                * extents_cumprod.reshape(dims_expanded),
                dim=0,
            )  # [N_1, ..., N_dim, ..., N_D]

            # Sort the t-values.
            return torch.argsort(t_values, dim=dim)

    # If the tensor is not an integer tensor, we have to use np.lexsort().
    # Unfortunately, torch doesn't have a lexsort() equivalent, so we have to
    # convert to numpy.
    return torch.from_numpy(np.lexsort(keys.detach().cpu().numpy())).to(device)


def lexsort_along(
    x: torch.Tensor, dim: int = -1
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sort a tensor along a specific dimension, taking all others as constant.

    This is like torch.sort(), but it doesn't sort along the other dimensions.
    As such, the other dimensions are treated as tuples. This function is
    roughly equivalent to the following Python code, but it is much faster.
    >>> torch.stack(
    >>>     sorted(
    >>>         x.unbind(dim),
    >>>         key=lambda t: t.tolist(),
    >>>         reverse=descending,
    >>>     ),
    >>>     dim=dim,
    >>> )

    The sort is always stable, meaning that the order of equal elements is
    preserved.

    Args:
        x: The tensor to sort.
            Shape: [N_1, ..., N_dim, ..., N_D]
        dim: The dimension to sort along.

    Returns:
        Tuple containing:
        - Sorted version of x.
            Shape: [N_1, ..., N_dim, ..., N_D]
        - The indices where the elements in the original input ended up in the
            returned sorted values.
            Shape: [N_dim]

    Examples:
        >>> x = torch.tensor([[2, 1], [3, 0], [1, 2], [1, 3]])
        >>> lexsort_along(x, dim=0)
        (tensor([[1, 2],
                 [1, 3],
                 [2, 1],
                 [3, 0]]),
         tensor([2, 3, 0, 1]))
        >>> torch.sort(x, dim=0)
        (tensor([[1, 0],
                 [1, 1],
                 [2, 2],
                 [3, 3]]),
         tensor([[2, 1],
                 [3, 0],
                 [0, 2],
                 [1, 3]])
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
    if x.ndim == 1:
        y = x.unsqueeze(0)  # [1, N_dim]
    else:
        y = x.movedim(dim, -1)  # [N_1, ..., N_dim-1, N_dim+1, ..., N_D, N_dim]
        y = y.reshape(
            -1, y.shape[-1]
        )  # [N_1 * ... * N_dim-1 * N_dim+1 * ... * N_D, N_dim]
    y = y.flip(dims=(0,))  # [N_1 * ... * N_dim-1 * N_dim+1 * ... * N_D, N_dim]
    idcs = lexsort(y)  # [N_dim]

    # Now we have to convert the output back to a tensor. This is a bit tricky,
    # because we must be able to select indices from any given dimension. To do
    # this, we perform:
    x_sorted = x.index_select(dim, idcs)  # [N_1, ..., N_dim, ..., N_D]

    # Finally, we return the sorted tensor and the indices.
    return x_sorted, idcs


@overload
def unique_consecutive(  # type: ignore
    x: torch.Tensor,
    return_inverse: Literal[False] = False,
    return_counts: Literal[False] = False,
    dim: int | None = None,
) -> torch.Tensor:
    ...  # fmt: skip


@overload
def unique_consecutive(
    x: torch.Tensor,
    return_inverse: Literal[True] = True,
    return_counts: Literal[False] = False,
    dim: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    ...  # fmt: skip


@overload
def unique_consecutive(
    x: torch.Tensor,
    return_inverse: Literal[False] = False,
    return_counts: Literal[True] = True,
    dim: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    ...  # fmt: skip


@overload
def unique_consecutive(
    x: torch.Tensor,
    return_inverse: Literal[True] = True,
    return_counts: Literal[True] = True,
    dim: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ...  # fmt: skip


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
        x: The input tensor. Must be sorted along the given dimension.
            Shape: [N_1, ..., N_dim, ..., N_D]
        return_inverse: Whether to also return the indices for where elements
            in the original input ended up in the returned unique list.
        return_counts: Whether to also return the counts for each unique
            element.
        dim: The dimension to operate upon. If None, the unique of the
            flattened input is returned. Otherwise, each of the tensors
            indexed by the given dimension is treated as one of the elements
            to apply the unique operation upon. See examples for more details.

    Returns:
        Tuple containing:
        - The unique elements.
            Shape: [N_1, ..., N_dim-1, N_unique, N_dim+1, ..., N_D]
        - (optional) if return_inverse is True, the indices where elements
            in the original input ended up in the returned unique values.
            Shape: [N_dim]
        - (optional) if return_counts is True, the counts for each unique
            element.
            Shape: [N_unique]

    Examples:
        >>> # 1D example: -----------------------------------------------------
        >>> x = torch.tensor([9, 9, 9, 9, 10, 10])
        >>> dim = 0

        >>> uniques, inverse, counts = unique_consecutive(
        >>>     x, return_inverse=True, return_counts=True, dim=dim
        >>> )
        >>> uniques
        tensor([9, 10])
        >>> inverse
        tensor([0, 0, 0, 0, 1, 1])
        >>> counts
        tensor([4, 2])

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
        tensor([[7, 9, 10],
                [8, 10, 9],
                [9, 8, 7],
                [9, 7, 7]])
        >>> inverse
        tensor([1, 2, 0, 1])
        >>> counts
        tensor([1, 2, 1])

        >>> # 3D example: -----------------------------------------------------
        >>> x = torch.tensor([
        >>>     [[0, 1, 2, 2], [4, 6, 5, 5], [9, 8, 7, 7]],
        >>>     [[4, 2, 8, 8], [3, 3, 7, 7], [0, 2, 1, 1]],
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
        tensor([0, 2, 1, 2])
        >>> counts
        tensor([1, 1, 2])
    """
    if dim is None:
        raise NotImplementedError(
            "dim=None is not implemented yet. Please specify a dimension"
            " explicitly."
        )

    device = x.device

    # Flatten all dimensions except the one we want to operate on.
    if x.ndim == 1:
        y = x.unsqueeze(0)  # [1, N_dim]
    else:
        y = x.movedim(dim, -1)  # [N_1, ..., N_dim-1, N_dim+1, ..., N_D, N_dim]
        y = y.reshape(
            -1, y.shape[-1]
        )  # [N_1 * ... * N_dim-1 * N_dim+1 * ... * N_D, N_dim]

    # Find the indices where the values change.
    is_change = torch.concatenate(
        [
            (
                torch.tensor([True], device=device)
                if x.shape[dim] > 0
                else torch.tensor([], dtype=torch.bool, device=device)
            ),
            (y[:, :-1] != y[:, 1:]).any(dim=0),
        ],
        dim=0,
    )  # [N_dim]

    # Find the unique values.
    idcs = is_change.nonzero(as_tuple=True)[0]  # [N_unique]
    unique = x.index_select(
        dim, idcs
    )  # [N_1, ..., N_dim-1, N_unique, N_dim+1, ..., N_D]

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
            torch.concatenate(
                [
                    idcs,
                    (
                        torch.tensor([x.shape[dim]], device=device)
                        if x.shape[dim] > 0
                        else torch.tensor([], dtype=torch.int64, device=device)
                    ),
                ],
                dim=0,
            )
        )  # [N_unique]
        aux.append(counts)

    if aux:
        return unique, *aux
    else:
        return unique


@overload
def unique(  # type: ignore
    x: torch.Tensor,
    return_inverse: Literal[False] = False,
    return_counts: Literal[False] = False,
    dim: int | None = None,
) -> torch.Tensor:
    # Overload for the case where:
    # - return_inverse is False
    # - return_counts is False
    ...


@overload
def unique(
    x: torch.Tensor,
    return_inverse: Literal[True] = True,
    return_counts: Literal[False] = False,
    dim: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Overload for the case where:
    # - return_inverse is True
    # - return_counts is False
    ...


@overload
def unique(
    x: torch.Tensor,
    return_inverse: Literal[False] = False,
    return_counts: Literal[True] = True,
    dim: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Overload for the case where:
    # - return_inverse is False
    # - return_counts is True
    ...


@overload
def unique(
    x: torch.Tensor,
    return_inverse: Literal[True] = True,
    return_counts: Literal[True] = True,
    dim: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Overload for the case where:
    # - return_inverse is True
    # - return_counts is True
    ...


def unique(
    x: torch.Tensor,
    return_inverse: bool = False,
    return_counts: bool = False,
    dim: int | None = None,
) -> (
    torch.Tensor
    | tuple[torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
):
    """Like torch.unique, but WAY more efficient.

    The returned unique elements are retrieved along the requested dimension,
    taking all the other dimensions as constant tuples.

    Args:
        x: The input tensor.
            Shape: [N_1, ..., N_dim, ..., N_D]
        return_inverse: Whether to also return the indices for where elements
            in the original input ended up in the returned unique list.
        return_counts: Whether to also return the counts for each unique
            element.
        dim: The dimension to operate upon. If None, the unique of the
            flattened input is returned. Otherwise, each of the tensors
            indexed by the given dimension is treated as one of the elements
            to apply the unique operation upon. See examples for more details.

    Returns:
        Tuple containing:
        - The unique elements.
            Shape: [N_1, ..., N_dim-1, N_unique, N_dim+1, ..., N_D]
        - (optional) if return_inverse is True, the indices where elements
            in the original input ended up in the returned unique values.
            Shape: [N_dim]
        - (optional) if return_counts is True, the counts for each unique
            element.
            Shape: [N_unique]

    Examples:
        >>> # 1D example: -----------------------------------------------------
        >>> x = torch.tensor([9, 10, 9, 9, 10, 9])
        >>> dim = 0

        >>> uniques, inverse, counts = unique(
        >>>     x, return_inverse=True, return_counts=True, dim=dim
        >>> )
        >>> uniques
        tensor([9, 10])
        >>> inverse
        tensor([0, 1, 0, 0, 1, 0])
        >>> counts
        tensor([4, 2])

        >>> # 2D example: -----------------------------------------------------
        >>> x = torch.tensor([
        >>>     [9, 10, 7, 9],
        >>>     [10, 9, 8, 10],
        >>>     [8, 7, 9, 8],
        >>>     [7, 7, 9, 7],
        >>> ])
        >>> dim = 1

        >>> uniques, inverse, counts = unique_with_backmap(
        >>>     x, return_inverse=True, return_counts=True, dim=dim
        >>> )
        >>> uniques
        tensor([[7, 9, 10],
                [8, 10, 9],
                [9, 8, 7],
                [9, 7, 7]])
        >>> inverse
        tensor([1, 2, 0, 1])
        >>> counts
        tensor([1, 2, 1])

        >>> # 3D example: -----------------------------------------------------
        >>> x = torch.tensor([
        >>>     [[0, 2, 1, 2], [4, 5, 6, 5], [9, 7, 8, 7]],
        >>>     [[4, 8, 2, 8], [3, 7, 3, 7], [0, 1, 2, 1]],
        >>> ])
        >>> dim = 2

        >>> uniques, inverse, counts = unique_with_backmap(
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
        tensor([0, 2, 1, 2])
        >>> counts
        tensor([1, 1, 2])
    """
    if dim is None:
        raise NotImplementedError(
            "dim=None is not implemented yet. Please specify a dimension"
            " explicitly."
        )

    # Sort along the given dimension, taking all the other dimensions as
    # constant tuples. Torch's sort() doesn't work here since it will
    # sort the other dimensions as well.
    x_sorted, backmap_along_dim = lexsort_along(x, dim=dim)

    out = unique_consecutive(
        x_sorted,
        return_inverse=return_inverse,  # type: ignore
        return_counts=return_counts,  # type: ignore
        dim=dim,
    )
    if isinstance(out, tuple):
        unique, *aux = out
    else:
        unique = out
        aux = []

    if return_inverse:
        # The backmap wasn't taken into account by unique_consecutive(), so we
        # have to do it ourselves.
        backmap_along_dim_inv = swap_idcs_vals(backmap_along_dim)
        aux[0] = aux[0][backmap_along_dim_inv]

    if aux:
        return unique, *aux
    else:
        return unique


@overload
def unique_with_backmap(  # type: ignore
    x: torch.Tensor,
    return_inverse: Literal[False] = False,
    return_counts: Literal[False] = False,
    dim: int | None = None,
) -> tuple[torch.Tensor, BackMap]:
    ...  # fmt: skip


@overload
def unique_with_backmap(
    x: torch.Tensor,
    return_inverse: Literal[True] = True,
    return_counts: Literal[False] = False,
    dim: int | None = None,
) -> tuple[torch.Tensor, BackMap, torch.Tensor]:
    ...  # fmt: skip


@overload
def unique_with_backmap(
    x: torch.Tensor,
    return_inverse: Literal[False] = False,
    return_counts: Literal[True] = True,
    dim: int | None = None,
) -> tuple[torch.Tensor, BackMap, torch.Tensor]:
    ...  # fmt: skip


@overload
def unique_with_backmap(
    x: torch.Tensor,
    return_inverse: Literal[True] = True,
    return_counts: Literal[True] = True,
    dim: int | None = None,
) -> tuple[torch.Tensor, BackMap, torch.Tensor, torch.Tensor]:
    ...  # fmt: skip


def unique_with_backmap(
    x: torch.Tensor,
    return_inverse: bool = False,
    return_counts: bool = False,
    dim: int | None = None,
) -> (
    tuple[torch.Tensor, BackMap]
    | tuple[torch.Tensor, BackMap, torch.Tensor]
    | tuple[torch.Tensor, BackMap, torch.Tensor, torch.Tensor]
):
    """Like torch.unique, but also returns a back map.

    The returned unique elements are retrieved along the requested dimension,
    taking all the other dimensions as constant tuples.

    Args:
        x: The input tensor.
            Shape: [N_1, ..., N_dim, ..., N_D]
        return_inverse: Whether to also return the indices for where elements
            in the original input ended up in the returned unique list.
        return_counts: Whether to also return the counts for each unique
            element.
        dim: The dimension to operate upon. If None, the unique of the
            flattened input is returned. Otherwise, each of the tensors
            indexed by the given dimension is treated as one of the elements
            to apply the unique operation upon. See examples for more details.

    Returns:
        Tuple containing:
        - The unique elements.
            Shape: [N_1, ..., N_dim-1, N_unique, N_dim+1, ..., N_D]
        - The back map as an indexing object, which maps the original input
            to the unique values. The first counts[0] indices in backmap[dim]
            correspond to the first unique value, the next counts[1] indices
            correspond to the second unique value, etc.
            Note: The element at index dim is always a Tensor.
            Length: D
            Shape of inner tensors: [N_dim]
        - (optional) if return_inverse is True, the indices where elements
            in the original input ended up in the returned unique values.
            Shape: [N_dim]
        - (optional) if return_counts is True, the counts for each unique
            element.
            Shape: [N_unique]

    Examples:
        >>> # 1D example: -----------------------------------------------------
        >>> x = torch.tensor([9, 10, 9, 9, 10, 9])
        >>> dim = 0

        >>> uniques, backmap, inverse, counts = unique_with_backmap(
        >>>     x, return_inverse=True, return_counts=True, dim=dim
        >>> )
        >>> uniques
        tensor([9, 10])
        >>> backmap
        (torch.tensor([0, 2, 3, 5, 1, 4]),)
        >>> inverse
        tensor([0, 1, 0, 0, 1, 0])
        >>> counts
        tensor([4, 2])

        >>> # Get the lexicographically sorted version of x:
        >>> x[backmap]
        tensor([9, 9, 9, 9, 10, 10])

        >>> # Get all indices of the i'th unique value:
        >>> cumcounts = cumsum_start_0(counts)
        >>> get_idcs = lambda i: backmap[dim][cumcounts[i] : cumcounts[i + 1]]
        >>> get_idcs(1)
        tensor([1, 4])

        >>> # 2D example: -----------------------------------------------------
        >>> x = torch.tensor([
        >>>     [9, 10, 7, 9],
        >>>     [10, 9, 8, 10],
        >>>     [8, 7, 9, 8],
        >>>     [7, 7, 9, 7],
        >>> ])
        >>> dim = 1

        >>> uniques, backmap, inverse, counts = unique_with_backmap(
        >>>     x, return_inverse=True, return_counts=True, dim=dim
        >>> )
        >>> uniques
        tensor([[7, 9, 10],
                [8, 10, 9],
                [9, 8, 7],
                [9, 7, 7]])
        >>> backmap
        (slice(None, None, None), tensor([2, 0, 3, 1]))
        >>> inverse
        tensor([1, 2, 0, 1])
        >>> counts
        tensor([1, 2, 1])

        >>> # Get the lexicographically sorted version of x:
        >>> x[backmap]
        tensor([[7, 9, 9, 10],
                [8, 10, 10, 9],
                [9, 8, 8, 7],
                [9, 7, 7, 7]])

        >>> # Get all indices of the i'th unique value:
        >>> cumcounts = cumsum_start_0(counts)
        >>> get_idcs = lambda i: backmap[dim][cumcounts[i] : cumcounts[i + 1]]
        >>> get_idcs(1)
        tensor([0, 3])

        >>> # 3D example: -----------------------------------------------------
        >>> x = torch.tensor([
        >>>     [[0, 2, 1, 2], [4, 5, 6, 5], [9, 7, 8, 7]],
        >>>     [[4, 8, 2, 8], [3, 7, 3, 7], [0, 1, 2, 1]],
        >>> ])
        >>> dim = 2

        >>> uniques, backmap, inverse, counts = unique_with_backmap(
        >>>     x, return_inverse=True, return_counts=True, dim=dim
        >>> )
        >>> uniques
        tensor([[[0, 1, 2],
                 [4, 6, 5],
                 [9, 8, 7]],
                [[4, 2, 8],
                 [3, 3, 7],
                 [0, 2, 1]]])
        >>> backmap
        (slice(None, None, None),
         slice(None, None, None),
         tensor([0, 2, 1, 3]))
        >>> inverse
        tensor([0, 2, 1, 2])
        >>> counts
        tensor([1, 1, 2])

        >>> # Get the lexicographically sorted version of x:
        >>> x[backmap]
        tensor([[[0, 1, 2, 2],
                 [4, 6, 5, 5],
                 [9, 8, 7, 7]],
                [[4, 2, 8, 8],
                 [3, 3, 7, 7],
                 [0, 2, 1, 1]]])

        >>> # Get all indices of the i'th unique value:
        >>> cumcounts = cumsum_start_0(counts)
        >>> get_idcs = lambda i: backmap[dim][cumcounts[i] : cumcounts[i + 1]]
        >>> get_idcs(1)
        tensor([2])
    """
    if dim is None:
        raise NotImplementedError(
            "dim=None is not implemented yet. Please specify a dimension"
            " explicitly."
        )

    # Sort along the given dimension, taking all the other dimensions as
    # constant tuples. Torch's sort() doesn't work here since it will
    # sort the other dimensions as well.
    x_sorted, backmap_along_dim = lexsort_along(x, dim=dim)

    out = unique_consecutive(
        x_sorted,
        return_inverse=return_inverse,  # type: ignore
        return_counts=return_counts,  # type: ignore
        dim=dim,
    )
    if isinstance(out, tuple):
        unique, *aux = out
    else:
        unique = out
        aux = []

    if return_inverse:
        # The backmap wasn't taken into account by unique_consecutive(), so we
        # have to do it ourselves.
        backmap_along_dim_inv = swap_idcs_vals(backmap_along_dim)
        aux[0] = aux[0][backmap_along_dim_inv]

    backmap = tuple(
        slice(None) if d != dim else backmap_along_dim
        for d in range(len(x.shape))
    )

    return unique, backmap, *aux


@overload
def unique_with_backmap_naive(  # type: ignore
    x: torch.Tensor,
    return_inverse: Literal[False] = False,
    return_counts: Literal[False] = False,
    dim: int | None = None,
) -> tuple[torch.Tensor, BackMap]:
    # Overload for the case where:
    # - return_inverse is False
    # - return_counts is False
    ...


@overload
def unique_with_backmap_naive(
    x: torch.Tensor,
    return_inverse: Literal[True] = True,
    return_counts: Literal[False] = False,
    dim: int | None = None,
) -> tuple[torch.Tensor, BackMap, torch.Tensor]:
    # Overload for the case where:
    # - return_inverse is True
    # - return_counts is False
    ...


@overload
def unique_with_backmap_naive(
    x: torch.Tensor,
    return_inverse: Literal[False] = False,
    return_counts: Literal[True] = True,
    dim: int | None = None,
) -> tuple[torch.Tensor, BackMap, torch.Tensor]:
    # Overload for the case where:
    # - return_inverse is False
    # - return_counts is True
    ...


@overload
def unique_with_backmap_naive(
    x: torch.Tensor,
    return_inverse: Literal[True] = True,
    return_counts: Literal[True] = True,
    dim: int | None = None,
) -> tuple[torch.Tensor, BackMap, torch.Tensor, torch.Tensor]:
    # Overload for the case where:
    # - return_inverse is True
    # - return_counts is True
    ...


def unique_with_backmap_naive(
    x: torch.Tensor,
    return_inverse: bool = False,
    return_counts: bool = False,
    dim: int | None = None,
) -> (
    tuple[torch.Tensor, BackMap]
    | tuple[torch.Tensor, BackMap, torch.Tensor]
    | tuple[torch.Tensor, BackMap, torch.Tensor, torch.Tensor]
):
    if dim is None:
        raise NotImplementedError(
            "dim=None is not implemented yet. Please specify a dimension"
            " explicitly."
        )

    out = unique(
        x,
        return_inverse=True,
        return_counts=return_counts,  # type: ignore
        dim=dim,
    )
    if isinstance(out, tuple):
        unique_, *aux = out
    else:
        unique_ = out
        aux = []
    backmap_along_dim = swap_idcs_vals_duplicates(aux[0])

    backmap = tuple(
        slice(None) if d != dim else backmap_along_dim
        for d in range(len(x.shape))
    )

    if return_inverse:
        return unique_, backmap, *aux
    return unique_, backmap, *aux[1:]


def collate_replace_corrupted(
    batch: list[Any],
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
        batch: Batch from the DataLoader.
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
    B = len(batch)
    batch = [sample for sample in batch if sample is not None]

    # Replace the corrupted samples with other randomly selected samples.
    while len(batch) < B:
        sample = dataset[random.randint(0, len(dataset) - 1)]  # type: ignore
        if sample is not None:
            batch.append(sample)

    # When the whole batch is fine, apply the default collate function.
    return default_collate_fn(batch)
