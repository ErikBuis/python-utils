from typing import Any, Generic, Literal, TypeVar, overload

import numpy as np
import numpy.typing as npt
from typing_extensions import override


T = TypeVar("T")


class NDArrayGeneric(np.ndarray, Generic[T]):
    """np.ndarray that allows for static type hinting of generics."""

    @override
    def __getitem__(self, key: Any) -> T:
        return super().__getitem__(key)  # type: ignore


def cumsum_start_0(
    a: npt.ArrayLike,
    axis: int | None = None,
    dtype: np.dtype | None = None,
    out: npt.NDArray | None = None,
) -> npt.NDArray:
    """Like np.cumsum, but adds a zero at the start of the array.

    Args:
        a: Input array.
            Shape: [N_0, ..., N_axis, ..., N_{D-1}]
        axis: Axis along which the cumulative sum is computed. The default
            (None) is to compute the cumsum over the flattened array.
        dtype: Type of the returned array and of the accumulator in which the
            elements are summed. If dtype is not specified, it defaults to the
            dtype of a.
        out: Alternative output array in which to place the result. It must
            have the same shape and buffer length as the expected output but
            the type will be cast if necessary.
            Shape: [N_0, ..., N_axis + 1, ..., N_{D-1}]

    Returns:
        A new array holding the result returned unless out is specified, in
        which case a reference to out is returned. The result has the same
        size as a except along the requested axis.
            Shape: [N_0, ..., N_axis + 1, ..., N_{D-1}]
    """
    a = np.array(a)

    if axis is None:
        a = a.flatten()
        axis = 0

    if dtype is None:
        dtype = a.dtype

    if out is not None:
        idx = [slice(None)] * a.ndim
        idx[axis] = 0  # type: ignore
        out[tuple(idx)] = 0
        idx[axis] = slice(1, None)
        np.cumsum(a, axis=axis, dtype=dtype, out=out[tuple(idx)])
        return out

    shape = list(a.shape)
    shape[axis] = 1
    zeros = np.zeros(shape, dtype=dtype)
    cumsum = np.cumsum(a, axis=axis, dtype=dtype)
    return np.concatenate([zeros, cumsum], axis=axis)


def pad_sequence(
    sequences: list[npt.ArrayLike],
    batch_first: bool = False,
    padding_value: Any = 0,
) -> npt.NDArray:
    """Pad a list of variable length arrays with padding_value.

    Note: This function is the numpy equivalent of
    torch.nn.utils.rnn.pad_sequence(). It is slower than the torch
    implementation, so please use the latter if you are working with PyTorch
    tensors.

    Args:
        sequences: A sequence of variable length arrays.
            Length: B
            Inner shape: [L_b, *]
        batch_first: Whether to return the batch dimension as the first
            dimension. If False, the output will have shape [max(L_bs), B, *].
            If True, the output will have shape [B, max(L_bs), *].
        padding_value: The value to use for padding the inner sequences.

    Returns:
        Array of shape [max(L_bs), B, *] if batch_first is False, otherwise
        array of shape [B, max(L_bs), *]. Padded with padding_value.
    """
    sequences_arr = [np.array(seq) for seq in sequences]
    star_shape = sequences_arr[0].shape[1:]
    assert all(
        (arr.shape[1:] == star_shape) for arr in sequences_arr[1:]
    ), "All arrays must have the same shape after the first dimension."
    B = len(sequences_arr)
    max_L_bs = max(len(arr) for arr in sequences_arr)
    shape = (
        (B, max_L_bs, *star_shape)
        if batch_first
        else (max_L_bs, B, *star_shape)
    )
    dtype = np.result_type(*sequences_arr)
    padded = np.full(shape, padding_value, dtype=dtype)
    for b, arr in enumerate(sequences_arr):
        if batch_first:
            padded[b, : len(arr)] = arr
        else:
            padded[: len(arr), b] = arr
    return padded


def lexsort_along(
    x: npt.NDArray, axis: int = -1
) -> tuple[npt.NDArray, npt.NDArray]:
    """Sort an array along a specific dimension, taking all others as constant.

    This is like np.sort(), but it doesn't sort along the other dimensions.
    As such, the other dimensions are treated as tuples. This function is
    roughly equivalent to the following Python code, but it is much faster.
    >>> np.stack(
    >>>     sorted(
    >>>         [np.take(x, i, axis=axis) for i in range(x.shape[axis])],
    >>>         key=lambda n: n.tolist(),
    >>>         reverse=descending,
    >>>     ),
    >>>     axis=axis,
    >>> )

    The sort is always stable, meaning that the order of equal elements is
    preserved.

    Args:
        x: The array to sort.
            Shape: [N_0, ..., N_axis, ..., N_{D-1}]
        axis: The dimension to sort along.

    Returns:
        Tuple containing:
        - Sorted version of x.
            Shape: [N_0, ..., N_axis, ..., N_{D-1}]
        - The indices where the elements in the original input ended up in the
            returned sorted values.
            Shape: [N_axis]

    Examples:
        >>> x = np.array([[2, 1], [3, 0], [1, 2], [1, 3]])
        >>> lexsort_along(x, axis=0)
        (array([[1, 2],
                [1, 3],
                [2, 1],
                [3, 0]]),
         array([2, 3, 0, 1]))
        >>> torch.sort(x, axis=0)
        (array([[1, 0],
                [1, 1],
                [2, 2],
                [3, 3]]),
         array([[2, 1],
                [3, 0],
                [0, 2],
                [1, 3]]))
    """
    # We can use np.lexsort() to sort only the requested dimension.
    # First, we prepare the array for np.lexsort(). The input to this function
    # must be a tuple of array-like objects, that are evaluated from last to
    # first. This is quite confusing, so I'll put an example here. If we have:
    # >>> x = array([[[15, 13],
    # >>>             [11,  4],
    # >>>             [16,  2]],
    # >>>            [[ 7, 21],
    # >>>             [ 3, 20],
    # >>>             [ 8, 22]],
    # >>>            [[19, 14],
    # >>>             [ 5, 12],
    # >>>             [ 6,  0]],
    # >>>            [[23,  1],
    # >>>             [10, 17],
    # >>>             [ 9, 18]]])
    # And axis=1, then the input to lexsort() must be:
    # >>> lexsort(array([[ 1, 17, 18],
    # >>>                [23, 10,  9],
    # >>>                [14, 12,  0],
    # >>>                [19,  5,  6],
    # >>>                [21, 20, 22],
    # >>>                [ 7,  3,  8],
    # >>>                [13,  4,  2],
    # >>>                [15, 11, 16]]))
    # Note that the first row is evaluated last and the last row is evaluated
    # first. We can now see that the sorting order will be 11 < 15 < 16, so
    # lexsort() will return array([1, 0, 2]). I thouroughly tested what the
    # absolute fastest way is to perform this operation, and it turns out that
    # the following is the best way to do it:
    if x.ndim == 1:
        y = np.expand_dims(x, 0)  # [1, N_axis]
    else:
        y = np.moveaxis(
            x, axis, -1
        )  # [N_0, ..., N_{axis-1}, N_{axis+1}, ..., N_{D-1}, N_axis]
        y = y.reshape(
            -1, y.shape[-1]
        )  # [N_0 * ... * N_{axis-1} * N_{axis+1} * ... * N_{D-1}, N_axis]
    y = np.flip(
        y, axis=(0,)
    )  # [N_0 * ... * N_{axis-1} * N_{axis+1} * ... * N_{D-1}, N_axis]
    idcs = np.lexsort(y)  # [N_axis]

    # Now we have to convert the output back to a array. This is a bit tricky,
    # because we must be able to select indices from any given dimension. To do
    # this, we perform:
    x_sorted = x.take(idcs, axis)  # [N_0, ..., N_axis, ..., N_{D-1}]

    # Finally, we return the sorted array and the indices.
    return x_sorted, idcs


@overload
def unique_consecutive(  # type: ignore
    x: npt.NDArray,
    return_inverse: Literal[False] = ...,
    return_counts: Literal[False] = ...,
    axis: int | None = None,
) -> npt.NDArray:
    pass


@overload
def unique_consecutive(
    x: npt.NDArray,
    return_inverse: Literal[True] = ...,
    return_counts: Literal[False] = ...,
    axis: int | None = None,
) -> tuple[npt.NDArray, npt.NDArray]:
    pass


@overload
def unique_consecutive(
    x: npt.NDArray,
    return_inverse: Literal[False] = ...,
    return_counts: Literal[True] = ...,
    axis: int | None = None,
) -> tuple[npt.NDArray, npt.NDArray]:
    pass


@overload
def unique_consecutive(
    x: npt.NDArray,
    return_inverse: Literal[True] = ...,
    return_counts: Literal[True] = ...,
    axis: int | None = None,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    pass


def unique_consecutive(
    x: npt.NDArray,
    return_inverse: bool = False,
    return_counts: bool = False,
    axis: int | None = None,
) -> (
    npt.NDArray
    | tuple[npt.NDArray, npt.NDArray]
    | tuple[npt.NDArray, npt.NDArray, npt.NDArray]
):
    """A consecutive version of np.unique.

    The returned unique elements are retrieved along the requested dimension,
    taking all the other dimensions as constant tuples.

    Args:
        x: The input array. Must be sorted along the given dimension.
            Shape: [N_0, ..., N_axis, ..., N_{D-1}]
        return_inverse: Whether to also return the indices for where elements
            in the original input ended up in the returned unique list.
        return_counts: Whether to also return the counts for each unique
            element.
        axis: The dimension to operate upon. If None, the unique of the
            flattened input is returned. Otherwise, each of the arrays
            indexed by the given dimension is treated as one of the elements
            to apply the unique operation upon. See examples for more details.

    Returns:
        Tuple containing:
        - The unique elements.
            Shape: [N_0, ..., N_{axis-1}, N_unique, N_{axis+1}, ..., N_{D-1}]
        - (Optional) If return_inverse is True, the indices where elements
            in the original input ended up in the returned unique values.
            Shape: [N_axis]
        - (Optional) If return_counts is True, the counts for each unique
            element.
            Shape: [N_unique]

    Examples:
        >>> # 1D example: -----------------------------------------------------
        >>> x = np.array([9, 9, 9, 9, 10, 10])
        >>> dim = 0

        >>> uniques, inverse, counts = unique_consecutive(
        >>>     x, return_inverse=True, return_counts=True, dim=dim
        >>> )
        >>> uniques
        array([9, 10])
        >>> inverse
        array([0, 0, 0, 0, 1, 1])
        >>> counts
        array([4, 2])

        >>> # 2D example: -----------------------------------------------------
        >>> x = np.array([
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
        array([[7, 9, 10],
                [8, 10, 9],
                [9, 8, 7],
                [9, 7, 7]])
        >>> inverse
        array([1, 2, 0, 1])
        >>> counts
        array([1, 2, 1])

        >>> # 3D example: -----------------------------------------------------
        >>> x = np.array([
        >>>     [[0, 1, 2, 2], [4, 6, 5, 5], [9, 8, 7, 7]],
        >>>     [[4, 2, 8, 8], [3, 3, 7, 7], [0, 2, 1, 1]],
        >>> ])
        >>> dim = 2

        >>> uniques, inverse, counts = unique_consecutive(
        >>>     x, return_inverse=True, return_counts=True, dim=dim
        >>> )
        >>> uniques
        array([[[0, 1, 2],
                 [4, 6, 5],
                 [9, 8, 7]],
                [[4, 2, 8],
                 [3, 3, 7],
                 [0, 2, 1]]])
        >>> inverse
        array([0, 2, 1, 2])
        >>> counts
        array([1, 1, 2])
    """
    if axis is None:
        raise NotImplementedError(
            "axis=None is not implemented yet. Please specify a dimension"
            " explicitly."
        )

    # Flatten all dimensions except the one we want to operate on.
    if x.ndim == 1:
        y = np.expand_dims(x, 0)  # [1, N_axis]
    else:
        y = np.moveaxis(
            x, axis, -1
        )  # [N_0, ..., N_{axis-1}, N_{axis+1}, ..., N_{D-1}, N_axis]
        y = y.reshape(
            -1, y.shape[-1]
        )  # [N_0 * ... * N_{axis-1} * N_{axis+1} * ... * N_{D-1}, N_axis]

    # Find the indices where the values change.
    is_change = np.concatenate(
        [np.array([True]), np.any(y[:, :-1] != y[:, 1:], axis=0)], axis=0
    )  # [N_axis]

    # Find the unique values.
    idcs = is_change.nonzero()[0]  # [N_unique]
    unique = x.take(
        idcs, axis
    )  # [N_0, ..., N_{axis-1}, N_unique, N_{axis+1}, ..., N_{D-1}]

    # Calculate auxiliary values.
    aux = []
    if return_inverse:
        # Find the indices where the elements in the original input ended up
        # in the returned unique values.
        inverse = is_change.cumsum(axis=0) - 1  # [N_axis]
        aux.append(inverse)
    if return_counts:
        # Find the counts for each unique element.
        counts = np.diff(
            np.concatenate([idcs, np.array([x.shape[axis]])], axis=0)
        )  # [N_unique]
        aux.append(counts)

    if aux:
        return unique, *aux
    else:
        return unique


def unequal_seqs_add(a: npt.NDArray, b: npt.NDArray) -> npt.NDArray:
    """Add two arrays, and adjust the size of the result if necessary.

    Args:
        a: The first array.
            Shape: [N]
        b: The second array.
            Shape: [M]

    Returns:
        The sum of the two arrays.
            Shape: [max(N, M)]
    """
    if len(a) < len(b):
        a = np.pad(a, (0, len(b) - len(a)))
    elif len(a) > len(b):
        b = np.pad(b, (0, len(a) - len(b)))
    return a + b


def init_normalized_histogram(
    bin_edges: npt.ArrayLike,
) -> tuple[float, npt.NDArray[np.float64]]:
    return 0, np.zeros(len(bin_edges) - 1)  # type: ignore


def update_normalized_histogram(
    old_value_avg: float,
    old_histogram: npt.NDArray[np.float64],
    amount_old_values: int,
    new_values: npt.ArrayLike,
    bin_edges: npt.ArrayLike,
) -> tuple[float, npt.NDArray[np.float64]]:
    """Update a normalized histogram with new values.

    Args:
        old_value_avg: The average of the old values in the histogram.
        old_histogram: The old normalized histogram.
            Shape: [bins]
        amount_old_values: The amount of old values in the histogram.
        new_values: The new values to add to the histogram.
            Shape: [amount_new_values]
        bin_edges: The bin edges of the histogram.
            Length: [bins + 1]

    Returns:
        Tuple containing:
        - The updated average of the histogram.
        - The updated normalized histogram.
            Shape: [bins]
    """
    new_values = np.array(new_values)

    # Update the average of the histogram.
    updated_value_avg = (
        old_value_avg * amount_old_values + new_values.mean() * len(new_values)
    ) / (amount_old_values + len(new_values))

    # Create a normalized histogram of the new values.
    new_histogram = np.histogram(new_values, bin_edges)[0] / len(new_values)

    # Weigh both histograms by the amount of values they contain.
    old_histogram *= amount_old_values
    new_histogram *= len(new_values)
    updated_histogram = (old_histogram + new_histogram) / (
        amount_old_values + len(new_values)
    )

    # Add the histograms and renormalize them.
    return updated_value_avg, updated_histogram
