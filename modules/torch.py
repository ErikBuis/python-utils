import random
import warnings
from typing import Any, Callable, Literal, NamedTuple, overload

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import (
    collate_str_fn,
    default_collate_fn_map,
)
from torch.utils.data.dataloader import default_collate


# Allow the dataset to return `None` when an example is corrupted. When it
# does, make torch's default collate function replace it with another example.
default_collate_fn_map.update({type(None): collate_str_fn})

# NamedTuple for the output of lexsort_along().
SortOnlyDim = NamedTuple(
    "SortOnlyDim", [("values", torch.Tensor), ("indices", torch.Tensor)]
)


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
        size as a except along the axis dimension where the size is one more.
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
        return object.to(dtype=dtype, device=device)

    if isinstance(object, np.ndarray):
        try:
            return torch.from_numpy(object).to(dtype=dtype, device=device)
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
                dtype=dtype, device=device
            )

    # Last resort: because numpy recognizes more array-like types than torch,
    # we try to convert to numpy first.
    try:
        return to_tensor(np.array(object), dtype=dtype, device=device)
    except Exception as e:
        raise TypeError(
            f"Could not convert object of type {type(object)} to tensor."
        ) from e


def lexsort_along(
    x: torch.Tensor, dim: int = -1, descending: bool = False
) -> SortOnlyDim:
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
        descending: Whether to sort in descending order.

    Returns:
        A namedtuple of (values, indices):
            values: Sorted version of x.
                Shape: [N_1, ..., N_dim, ..., N_D]
            indices: The indices where the elements in the original input ended
                up in the returned sorted values.
                Shape: [N_dim]

    Examples:
        >>> x = torch.tensor([[2, 1], [3, 0], [1, 2], [1, 3]])
        >>> lexsort_along(x, dim=0)
        SortOnlyDim(
            values=tensor([[1, 2],
                           [1, 3],
                           [2, 1],
                           [3, 0]]),
            indices=tensor([2, 3, 0, 1])
        )
        >>> torch.sort(x, dim=0)
        torch.return_types.sort(
            values=tensor([[1, 0],
                           [1, 1],
                           [2, 2],
                           [3, 3]]),
            indices=tensor([[2, 1],
                            [3, 0],
                            [0, 2],
                            [1, 3]])
        )
    """
    # If x is 1D, we can just do a normal sort.
    if len(x.shape) == 1:
        torch_return_types_sort = torch.sort(x, dim=dim, descending=descending)
        return SortOnlyDim(
            torch_return_types_sort.values, torch_return_types_sort.indices
        )

    # We can use np.lexsort() to sort only the requested dimension.
    # Unfortunately, torch doesn't have a lexsort() equivalent, so we have to
    # convert to numpy.

    # First, we prepare the tensor for np.lexsort(). The input to this function
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
    # Then the input to np.lexsort() must be:
    # >>> np.lexsort((tensor([ 1, 17, 18]),
    # >>>             tensor([23, 10,  9]),
    # >>>             tensor([14, 12,  0]),
    # >>>             tensor([19,  5,  6]),
    # >>>             tensor([21, 20, 22]),
    # >>>             tensor([7, 3, 8]),
    # >>>             tensor([13,  4,  2]),
    # >>>             tensor([15, 11, 16])))
    # Note that the first tensor is evaluated last and the last tensor is
    # evaluated first; we can see that the sorting order will be 11 < 15 < 16,
    # so np.lexsort() will return array([1, 0, 2]). I thouroughly tested what
    # the absolute fastest way is to perform this operation, and it turns out
    # that the following is the fastest:
    in_lexsort = x.transpose(0, dim).flatten(1).flip(dims=(1,)).cpu().unbind(1)
    out_lexsort = torch.from_numpy(np.lexsort(in_lexsort)).to(x.device)

    # Now we have to convert the output back to a tensor. This is a bit tricky,
    # because we must be able to select indices from any given dimension. To do
    # this, we perform:
    x_sorted = x.index_select(dim, out_lexsort)

    # Finally, we return the sorted tensor and the indices.
    return SortOnlyDim(x_sorted, out_lexsort)


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


@overload
def unique_with_backmap(
    x: torch.Tensor,
    return_inverse: Literal[False] = False,
    return_counts: Literal[False] = False,
    dim: int | None = None,
) -> tuple[torch.Tensor, list[slice | torch.Tensor]]:
    # Overload for the case where:
    # - return_inverse is False
    # - return_counts is False
    ...


@overload
def unique_with_backmap(
    x: torch.Tensor,
    return_inverse: Literal[True] = True,
    return_counts: Literal[False] = False,
    dim: int | None = None,
) -> tuple[torch.Tensor, list[slice | torch.Tensor], torch.Tensor]:
    # Overload for the case where:
    # - return_inverse is True
    # - return_counts is False
    ...


@overload
def unique_with_backmap(
    x: torch.Tensor,
    return_inverse: Literal[False] = False,
    return_counts: Literal[True] = True,
    dim: int | None = None,
) -> tuple[torch.Tensor, list[slice | torch.Tensor], torch.Tensor]:
    # Overload for the case where:
    # - return_inverse is False
    # - return_counts is True
    ...


@overload
def unique_with_backmap(
    x: torch.Tensor,
    return_inverse: Literal[True] = True,
    return_counts: Literal[True] = True,
    dim: int | None = None,
) -> tuple[
    torch.Tensor, list[slice | torch.Tensor], torch.Tensor, torch.Tensor
]:
    # Overload for the case where:
    # - return_inverse is True
    # - return_counts is True
    ...


def unique_with_backmap(
    x: torch.Tensor,
    return_inverse: bool = False,
    return_counts: bool = False,
    dim: int | None = None,
) -> (
    tuple[torch.Tensor, list[slice | torch.Tensor]]
    | tuple[torch.Tensor, list[slice | torch.Tensor], torch.Tensor]
    | tuple[
        torch.Tensor, list[slice | torch.Tensor], torch.Tensor, torch.Tensor
    ]
):
    """Like torch.unique, but also returns a back map.

    The returned unique elements are sorted along the given dimension, taking
    all the other dimensions as constant tuples.

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
            Shape of inner Tensor objects: [N_dim]
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
        [torch.tensor([0, 2, 3, 5, 1, 4])]
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
        [slice(None, None, None), tensor([2, 0, 3, 1])]
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
        [slice(None, None, None),
         slice(None, None, None),
         tensor([0, 2, 1, 3])]
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
    # constant tuples. torch's sort() doesn't work here since it will
    # sort the other dimensions as well.
    x_sorted, backmap_along_dim = lexsort_along(x, dim=dim)

    unique, *aux = torch.unique_consecutive(
        x_sorted,
        return_inverse=return_inverse,
        return_counts=return_counts,
        dim=dim,
    )

    if return_inverse:
        # The backmap wasn't taken into account by torch.unique_consecutive(),
        # so we have to do it ourselves.
        backmap_along_dim_inv = swap_idcs_vals(backmap_along_dim)
        aux[0] = aux[0][backmap_along_dim_inv]

    backmap = [
        slice(None) if d != dim else backmap_along_dim
        for d in range(len(x.shape))
    ]

    return unique, backmap, *aux


def unique_with_backmap_naive(
    x: torch.Tensor,
    return_inverse: bool = False,
    return_counts: bool = False,
    dim: int | None = None,
) -> (
    tuple[torch.Tensor, list[slice | torch.Tensor]]
    | tuple[torch.Tensor, list[slice | torch.Tensor], torch.Tensor]
    | tuple[
        torch.Tensor, list[slice | torch.Tensor], torch.Tensor, torch.Tensor
    ]
):
    if dim is None:
        raise NotImplementedError(
            "dim=None is not implemented yet. Please specify a dimension"
            " explicitly."
        )

    unique, inverse, *aux = torch.unique(
        x, return_inverse=True, return_counts=return_counts, dim=dim
    )
    backmap_along_dim = swap_idcs_vals_duplicates(inverse)

    backmap = [
        slice(None) if d != dim else backmap_along_dim
        for d in range(len(x.shape))
    ]

    if return_inverse:
        return unique, backmap, inverse, *aux
    return unique, backmap, *aux


def collate_replace_corrupted(
    batch: list[Any],
    dataset: Dataset,
    default_collate_fn: Callable | None = None,
) -> Any:
    """Collate function that allows to replace corrupted examples in the batch.

    The dataloader should return None when an example is corrupted. This
    function will then replace all None's in the batch with other
    randomly-selected examples from the Dataset.

    Warning: Since corrupted examples are replaced with random other examples
    from the dataset, a sample might be sampled multiple times in one pass
    through the dataloader. This implies that a model might be trained on the
    same example multiple times in one epoch.

    Note: As a DataLoader only accepts collate functions with a single
    argument, you should use functools.partial to create a partial function
    first. For example:
    >>> from functools import partial
    >>> collate_fn = partial(collate_replace_corrupted, dataset=dataset)

    This function was based on:
    https://stackoverflow.com/a/69578320/15636460

    Args:
        batch: Batch from the DataLoader.
        dataset: Dataset that the DataLoader is passing through.
        default_collate_fn: The collate function to call once the batch has no
            corrupted examples any more. If None,
            torch.utils.data.dataloader.default_collate is called.

    Returns:
        Batch with new examples instead of corrupted ones.
    """
    # Use torch.utils.data.dataloader.default_collate if no other default
    # collate function is specified.
    default_collate_fn = (
        default_collate_fn
        if default_collate_fn is not None
        else default_collate
    )

    # Filter out all corrupted examples.
    B = len(batch)
    batch = [example for example in batch if example is not None]

    # Replace the corrupted examples with other randomly selected examples.
    while len(batch) < B:
        sample = dataset[random.randint(0, len(dataset) - 1)]  # type: ignore
        if sample is not None:
            batch.append(sample)

    # When the whole batch is fine, apply the default collate function.
    return default_collate_fn(batch)
