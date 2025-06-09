from __future__ import annotations

from collections.abc import Generator
from typing import Any, Generic, Literal, TypeVar, overload

import numpy as np
import numpy.typing as npt
from typing_extensions import override

T = TypeVar("T")
NpGeneric1 = TypeVar("NpGeneric1", bound=np.generic)
NpGeneric2 = TypeVar("NpGeneric2", bound=np.generic)


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
    """Like np.cumsum(), but adds a zero at the start of the array.

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
    return np.concat([zeros, cumsum], axis=axis)


def swap_idcs_vals(x: npt.NDArray) -> npt.NDArray:
    """Swap the indices and values of a 1D array.

    The input array is assumed to contain exactly all integers from 0 to
    x.shape[0] - 1, in any order.

    Warning: This function does not explicitly check if the input array
    contains no duplicates. If x contains duplicates, no error will be raised
    and undefined behaviour will occur!

    Args:
        x: The array to swap.
            Shape: [N]

    Returns:
        The swapped array.
            Shape: [N]

    Examples:
        >>> x = np.array([2, 3, 0, 4, 1])
        >>> swap_idcs_vals(x)
        array([2, 4, 0, 1, 3])
    """
    if x.ndim != 1:
        raise ValueError("x must be 1D.")

    x_swapped = np.empty_like(x)
    x_swapped[x] = np.arange(len(x))
    return x_swapped


def swap_idcs_vals_duplicates(x: npt.NDArray) -> npt.NDArray:
    """Swap the indices and values of a 1D array, allowing duplicates.

    The input array is assumed to contain integers from 0 to M <= N, in any
    order, and may contain duplicates.

    The output array will contain exactly all integers from 0 to len(x) - 1,
    in any order.

    If the input doesn't contain duplicates, you should use swap_idcs_vals()
    instead since it is faster (especially for large arrays).

    Args:
        x: The array to swap.
            Shape: [N]

    Returns:
        The swapped array.
            Shape: [N]

    Examples:
        >>> x = np.array([1, 3, 0, 1, 3])
        >>> swap_idcs_vals_duplicates(x)
        array([2, 0, 3, 1, 4])
    """
    if x.ndim != 1:
        raise ValueError("x must be 1D.")

    # Believe it or not, this O(n log n) algorithm is actually faster than a
    # native implementation that uses a Python for loop with complexity O(n).
    return np.argsort(x)


def lexsort(
    keys: npt.NDArray | tuple[npt.NDArray, ...], axis: int = -1
) -> npt.NDArray:
    """Like np.lexsort(), but MUCH faster.

    Perform an indirect sort using a sequence of keys.

    Given multiple sorting keys, which can be interpreted as elements of a
    tuple, lexsort returns an array of integer indices that describes the sort
    order of the given tuples. The last key in the tuple is used for the
    primary sort order, the second-to-last key for the secondary sort order,
    and so on. The first dimension is always interpreted as the dimension
    along which the tuples lie.

    Warning: Unlike np.lexsort(), this function does not perform a stable sort.
    This means that the order of equal elements is not preserved. If you need a
    stable sort, you should use np.lexsort() instead.

    Args:
        keys: Array of shape [K, N_0, ..., N_axis, ..., N_{D-1}] or a tuple
            containing K [N_0, ..., N_axis, ..., N_{D-1}]-shaped sequences. K
            refers to the amount of elements in the tuples. The last element
            is the primary sort key.
        axis: Dimension to be indirectly sorted. By default, sort over the last
            dimension.

    Returns:
        Array of indices that sort the keys along the specified axis.
            Shape: [N_0, ..., N_axis, ..., N_{D-1}]

    Examples:
        >>> lexsort((np.array([ 1, 17, 18]),
        >>>          np.array([23, 10,  9]),
        >>>          np.array([14, 12,  0]),
        >>>          np.array([19,  5,  6]),
        >>>          np.array([21, 20, 22]),
        >>>          np.array([ 7,  3,  8]),
        >>>          np.array([13,  4,  2]),
        >>>          np.array([15, 11, 16])))
        array([1, 0, 2])

        >>> lexsort(np.array([[4, 8, 2, 8, 3, 7, 3],
        >>>                       [9, 4, 0, 4, 0, 4, 1],
        >>>                       [1, 5, 1, 4, 3, 4, 4]]))
        array([2, 0, 4, 6, 5, 3, 1])
    """
    if isinstance(keys, tuple):
        keys = np.stack(keys)  # [K, N_0, ..., N_axis, ..., N_{D-1}]

    # If the array is an integer array, first try sorting by representing
    # each of the "tuples" as a single integer. This is much faster than
    # lexsorting along the given dimension.
    if np.issubdtype(keys.dtype, np.integer) and keys.size != 0:
        # Compute the minimum and maximum values for each key.
        axes_flat = tuple(range(1, keys.ndim))
        maxs = np.amax(keys, axis=axes_flat, keepdims=True)  # [K, 1, ..., 1]
        mins = np.amin(keys, axis=axes_flat, keepdims=True)  # [K, 1, ..., 1]
        extents = np.squeeze(maxs - mins + 1, axis=axes_flat)  # [K]
        keys_dense = keys - mins  # [K, N_0, ..., N_axis, ..., N_{D-1}]

        try:
            # Convert the tuples to single integers.
            idcs = np.ravel_multi_index(
                keys_dense.astype(np.int64), extents, mode="raise", order="F"
            )  # [N_0, ..., N_axis, ..., N_{D-1}]

            # Sort the integers.
            return np.argsort(
                idcs, axis=axis
            )  # [N_0, ..., N_axis, ..., N_{D-1}]
        except ValueError:
            # Overflow would occur when converting to integers.
            pass

    # If the array is not an integer array or if overflow would occur when
    # converting to integers, we have to use np.lexsort().
    return np.lexsort(keys, axis=axis)


def lexsort_along(
    x: npt.NDArray, axis: int = -1
) -> tuple[npt.NDArray, npt.NDArray]:
    """Sort an array along axis, taking all others as tuples.

    This is like np.sort(), but the other dimensions are treated as tuples.
    This function is roughly equivalent to the following Python code, but it is
    much faster.
    >>> np.stack(
    >>>     sorted(
    >>>         [np.take(x, i, axis=axis) for i in range(x.shape[axis])],
    >>>         key=tuple,
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
        - The backmap array, which contains the indices of the sorted values
            in the original input.
            The sorted version of x can be retrieved as follows:
            >>> x_sorted = x.take(backmap, axis)
            Shape: [N_axis]

    Examples:
        >>> x = np.array([
        >>>     [2, 1],
        >>>     [3, 0],
        >>>     [1, 2],
        >>>     [1, 3],
        >>> ])
        >>> axis = 0

        >>> x_sorted, backmap = lexsort_along(x, axis=axis)
        >>> x_sorted
        array([[1, 2],
               [1, 3],
               [2, 1],
               [3, 0]])
        >>> backmap
        array([2, 3, 0, 1]))

        >>> # Get the lexicographically sorted version of x:
        >>> x.take(backmap)
        array([[1, 2],
               [1, 3],
               [2, 1],
               [3, 0]])
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
    # And axis=1, then the input to np.lexsort() must be:
    # >>> np.lexsort(array([[ 1, 17, 18],
    # >>>                   [23, 10,  9],
    # >>>                   [14, 12,  0],
    # >>>                   [19,  5,  6],
    # >>>                   [21, 20, 22],
    # >>>                   [ 7,  3,  8],
    # >>>                   [13,  4,  2],
    # >>>                   [15, 11, 16]]))
    # Note that the first row is evaluated last and the last row is evaluated
    # first. We can now see that the sorting order will be 11 < 15 < 16, so
    # np.lexsort() will return array([1, 0, 2]). I thouroughly tested what the
    # absolute fastest way is to perform this operation, and it turns out that
    # the following is the best way to do it:
    N_axis = x.shape[axis]

    if x.ndim == 1:
        y = np.expand_dims(x, 0)  # [1, N_axis]
    else:
        y = np.moveaxis(
            x, axis, -1
        )  # [N_0, ..., N_{axis-1}, N_{axis+1}, ..., N_{D-1}, N_axis]
        y = y.reshape(
            -1, N_axis
        )  # [N_0 * ... * N_{axis-1} * N_{axis+1} * ... * N_{D-1}, N_axis]
    y = np.flip(
        y, axis=(0,)
    )  # [N_0 * ... * N_{axis-1} * N_{axis+1} * ... * N_{D-1}, N_axis]
    backmap = lexsort(y, axis=-1)  # [N_axis]

    # Sort the array along the given axis.
    x_sorted = x.take(backmap, axis)  # [N_0, ..., N_axis, ..., N_{D-1}]

    # Finally, we return the sorted array and the backmap.
    return x_sorted, backmap


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
    """A consecutive version of np.unique().

    The returned unique elements are retrieved along the requested dimension,
    taking all the other dimensions as constant tuples.

    Args:
        x: The input array. Must be sorted along the given dimension.
            Shape: [N_0, ..., N_axis, ..., N_{D-1}]
        return_inverse: Whether to also return the inverse mapping array.
            This can be used to reconstruct the original array from the unique
            array.
        return_counts: Whether to also return the number of times each unique
            element occurred in the original array.
        axis: The dimension to operate upon. If None, the unique of the
            flattened input is returned. Otherwise, each of the arrays indexed
            by the given dimension is treated as one of the elements to apply
            the unique operation upon. See examples for more details.

    Returns:
        Tuple containing:
        - The unique elements.
            Shape: [N_0, ..., N_{axis-1}, U, N_{axis+1}, ..., N_{D-1}]
        - (Optional) If return_inverse is True, the indices where elements
            in the original array ended up in the returned unique values.
            The original array can be reconstructed as follows:
            >>> x_reconstructed = uniques.take(inverse, axis)
            Shape: [N_axis]
        - (Optional) If return_counts is True, the counts for each unique
            element.
            Shape: [U]

    Examples:
        >>> # 1D example: -----------------------------------------------------
        >>> x = np.array([9, 9, 9, 9, 10, 10])
        >>> axis = 0

        >>> uniques, inverse, counts = unique_consecutive(
        >>>     x, return_inverse=True, return_counts=True, axis=axis
        >>> )
        >>> uniques
        array([9, 10])
        >>> inverse
        array([0, 0, 0, 0, 1, 1])
        >>> counts
        array([4, 2])

        >>> # Reconstruct the original array:
        >>> uniques.take(inverse, axis)
        array([ 9,  9,  9,  9, 10, 10])

        >>> # 2D example: -----------------------------------------------------
        >>> x = np.array([
        >>>     [7,  9,  9, 10],
        >>>     [8, 10, 10,  9],
        >>>     [9,  8,  8,  7],
        >>>     [9,  7,  7,  7],
        >>> ])
        >>> axis = 1

        >>> uniques, inverse, counts = unique_consecutive(
        >>>     x, return_inverse=True, return_counts=True, axis=axis
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

        >>> # Reconstruct the original array:
        >>> uniques.take(inverse, axis)
        array([[ 7,  9,  9, 10],
               [ 8, 10, 10,  9],
               [ 9,  8,  8,  7],
               [ 9,  7,  7,  7]])

        >>> # 3D example: -----------------------------------------------------
        >>> x = np.array([
        >>>     [
        >>>         [0, 1, 2, 2],
        >>>         [4, 6, 5, 5],
        >>>         [9, 8, 7, 7],
        >>>     ],
        >>>     [
        >>>         [4, 2, 8, 8],
        >>>         [3, 3, 7, 7],
        >>>         [0, 2, 1, 1],
        >>>     ],
        >>> ])
        >>> axis = 2

        >>> uniques, inverse, counts = unique_consecutive(
        >>>     x, return_inverse=True, return_counts=True, axis=axis
        >>> )
        >>> uniques
        array([[[0, 1, 2],
                [4, 6, 5],
                [9, 8, 7]],
               [[4, 2, 8],
                [3, 3, 7],
                [0, 2, 1]]])
        >>> inverse
        array([0, 1, 2, 2])
        >>> counts
        array([1, 1, 2])

    >>> # Reconstruct the original array:
    >>> uniques.take(inverse, axis)
    array([[[0, 1, 2, 2],
            [4, 6, 5, 5],
            [9, 8, 7, 7]],
           [[4, 2, 8, 8],
            [3, 3, 7, 7],
            [0, 2, 1, 1]]])
    """
    if axis is None:
        raise NotImplementedError(
            "axis=None is not implemented yet. Please specify a dimension"
            " explicitly."
        )

    N_axis = x.shape[axis]

    # Flatten all dimensions except the one we want to operate on.
    if x.ndim == 1:
        y = np.expand_dims(x, 0)  # [1, N_axis]
    else:
        y = np.moveaxis(
            x, axis, -1
        )  # [N_0, ..., N_{axis-1}, N_{axis+1}, ..., N_{D-1}, N_axis]
        y = y.reshape(
            -1, N_axis
        )  # [N_0 * ... * N_{axis-1} * N_{axis+1} * ... * N_{D-1}, N_axis]

    # Find the indices where the values change.
    is_change = np.concat(
        [np.array([True]), np.any(y[:, :-1] != y[:, 1:], axis=0)], axis=0
    )  # [N_axis]
    is_change = np.concat([
        (
            np.ones(1, dtype=bool) if N_axis > 0 else np.empty(0, dtype=bool)
        ),  # [1] or [0]
        np.any(y[:, :-1] != y[:, 1:], axis=0),  # [N_axis - 1] or [0]
    ])  # [N_axis]

    # Find the unique values.
    idcs = is_change.nonzero()[0]  # [U]
    uniques = x.take(
        idcs, axis
    )  # [N_0, ..., N_{axis-1}, U, N_{axis+1}, ..., N_{D-1}]

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
            np.concat([
                idcs,  # [U]
                (
                    np.full((1,), N_axis, dtype=np.int64)
                    if N_axis > 0
                    else np.empty(0, dtype=np.int64)
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
    x: npt.NDArray,
    return_backmap: Literal[False] = ...,
    return_inverse: Literal[False] = ...,
    return_counts: Literal[False] = ...,
    axis: int | None = None,
) -> npt.NDArray:
    pass


@overload
def unique(
    x: npt.NDArray,
    return_backmap: Literal[True] = ...,
    return_inverse: Literal[False] = ...,
    return_counts: Literal[False] = ...,
    axis: int | None = None,
) -> tuple[npt.NDArray, npt.NDArray]:
    pass


@overload
def unique(
    x: npt.NDArray,
    return_backmap: Literal[False] = ...,
    return_inverse: Literal[True] = ...,
    return_counts: Literal[False] = ...,
    axis: int | None = None,
) -> tuple[npt.NDArray, npt.NDArray]:
    pass


@overload
def unique(
    x: npt.NDArray,
    return_backmap: Literal[False] = ...,
    return_inverse: Literal[False] = ...,
    return_counts: Literal[True] = ...,
    axis: int | None = None,
) -> tuple[npt.NDArray, npt.NDArray]:
    pass


@overload
def unique(
    x: npt.NDArray,
    return_backmap: Literal[True] = ...,
    return_inverse: Literal[True] = ...,
    return_counts: Literal[False] = ...,
    axis: int | None = None,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    pass


@overload
def unique(
    x: npt.NDArray,
    return_backmap: Literal[True] = ...,
    return_inverse: Literal[False] = ...,
    return_counts: Literal[True] = ...,
    axis: int | None = None,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    pass


@overload
def unique(
    x: npt.NDArray,
    return_backmap: Literal[False] = ...,
    return_inverse: Literal[True] = ...,
    return_counts: Literal[True] = ...,
    axis: int | None = None,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    pass


@overload
def unique(
    x: npt.NDArray,
    return_backmap: Literal[True] = ...,
    return_inverse: Literal[True] = ...,
    return_counts: Literal[True] = ...,
    axis: int | None = None,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    pass


def unique(
    x: npt.NDArray,
    return_backmap: bool = False,
    return_inverse: bool = False,
    return_counts: bool = False,
    axis: int | None = None,
) -> (
    npt.NDArray
    | tuple[npt.NDArray, npt.NDArray]
    | tuple[npt.NDArray, npt.NDArray, npt.NDArray]
    | tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]
):
    """Like np.unique(), but can also return a backmap array.

    The returned unique elements are retrieved along the requested dimension,
    taking all the other dimensions as constant tuples.

    This function is faster than numpy's version for small arrays (especially
    when the amount of elements in the tuples is small), but slower for large
    arrays. However, the main advantage of this function is that it can return
    a backmap array that can be used in further processing.

    Args:
        x: The input array.
            Shape: [N_0, ..., N_axis, ..., N_{D-1}]
        return_backmap: Whether to also return the backmap array.
            This can be used to sort the original array.
        return_inverse: Whether to also return the inverse mapping array.
            This can be used to reconstruct the original array from the unique
            array.
        return_counts: Whether to also return the counts of each unique
            element.
        axis: The dimension to operate upon. If None, the unique of the
            flattened input is returned. Otherwise, each of the arrays
            indexed by the given dimension is treated as one of the elements
            to apply the unique operation upon. See examples for more details.

    Returns:
        Tuple containing:
        - The unique elements, guaranteed to be sorted along the given
            dimension.
            Shape: [N_0, ..., N_{axis-1}, U, N_{axis+1}, ..., N_{D-1}]
        - (Optional) If return_backmap is True, the backmap array, which
            contains the indices of the unique values in the original input.
            The sorted version of x can be retrieved as follows:
            >>> x_sorted = x.take(backmap, axis)
            Shape: [N_axis]
        - (Optional) If return_inverse is True, the indices where elements
            in the original input ended up in the returned unique values.
            The original array can be reconstructed as follows:
            >>> x_reconstructed = uniques.take(inverse, axis)
            Shape: [N_axis]
        - (Optional) If return_counts is True, the counts for each unique
            element.
            Shape: [U]

    Examples:
        >>> # 1D example: -----------------------------------------------------
        >>> x = np.array([9, 10, 9, 9, 10, 9])
        >>> axis = 0

        >>> uniques, backmap, inverse, counts = unique(
        >>>     x,
        >>>     return_backmap=True,
        >>>     return_inverse=True,
        >>>     return_counts=True,
        >>>     axis=axis,
        >>> )
        >>> uniques
        array([ 9, 10])
        >>> backmap
        array([0, 2, 3, 5, 1, 4])
        >>> inverse
        array([0, 1, 0, 0, 1, 0])
        >>> counts
        array([4, 2])

        >>> # Get the lexicographically sorted version of x:
        >>> x.take(backmap, axis)
        array([ 9,  9,  9,  9, 10, 10])

        >>> # Reconstruct the original array:
        >>> uniques.take(inverse, axis)
        array([ 9, 10,  9,  9, 10,  9])

        >>> # 2D example: -----------------------------------------------------
        >>> x = np.array([
        >>>     [9, 10, 7, 9],
        >>>     [10, 9, 8, 10],
        >>>     [8, 7, 9, 8],
        >>>     [7, 7, 9, 7],
        >>> ])
        >>> axis = 1

        >>> uniques, backmap, inverse, counts = unique(
        >>>     x,
        >>>     return_backmap=True,
        >>>     return_inverse=True,
        >>>     return_counts=True,
        >>>     axis=axis,
        >>> )
        >>> uniques
        array([[ 7,  9, 10],
                [ 8, 10,  9],
                [ 9,  8,  7],
                [ 9,  7,  7]])
        >>> backmap
        array([2, 0, 3, 1])
        >>> inverse
        array([1, 2, 0, 1])
        >>> counts
        array([1, 2, 1])

        >>> # Get the lexicographically sorted version of x:
        >>> x.take(backmap, axis)
        array([[ 7,  9,  9, 10],
                [ 8, 10, 10,  9],
                [ 9,  8,  8,  7],
                [ 9,  7,  7,  7]])

        >>> # Reconstruct the original array:
        >>> uniques.take(inverse, axis)
        array([[ 9, 10,  7,  9],
                [10,  9,  8, 10],
                [ 8,  7,  9,  8],
                [ 7,  7,  9,  7]])

        >>> # 3D example: -----------------------------------------------------
        >>> x = np.array([
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
        >>> axis = 2

        >>> uniques, backmap, inverse, counts = unique(
        >>>     x,
        >>>     return_backmap=True,
        >>>     return_inverse=True,
        >>>     return_counts=True,
        >>>     axis=axis,
        >>> )
        >>> uniques
        array([[[0, 1, 2],
                 [4, 6, 5],
                 [9, 8, 7]],
                [[4, 2, 8],
                 [3, 3, 7],
                 [0, 2, 1]]])
        >>> backmap
        array([0, 2, 1, 3])
        >>> inverse
        array([0, 2, 1, 2])
        >>> counts
        array([1, 1, 2])

        >>> # Get the lexicographically sorted version of x:
        >>> x.take(backmap, axis)
        array([[[0, 1, 2, 2],
                 [4, 6, 5, 5],
                 [9, 8, 7, 7]],
                [[4, 2, 8, 8],
                 [3, 3, 7, 7],
                 [0, 2, 1, 1]]])

        >>> # Reconstruct the original array:
        >>> uniques.take(inverse, axis)
        array([[[0, 2, 1, 2],
                 [4, 5, 6, 5],
                 [9, 7, 8, 7]],
                [[4, 8, 2, 8],
                 [3, 7, 3, 7],
                 [0, 1, 2, 1]])
    """
    if axis is None:
        raise NotImplementedError(
            "axis=None is not implemented yet. Please specify a dimension"
            " explicitly."
        )

    # Sort along the given dimension, taking all the other dimensions as
    # constant tuples. NumPy's sort() doesn't work here since it will sort the
    # other dimensions as well.
    x_sorted, backmap = lexsort_along(
        x, axis=axis
    )  # [N_0, ..., N_axis, ..., N_{D-1}], [N_axis]

    out = unique_consecutive(
        x_sorted,
        return_inverse=return_inverse,
        return_counts=return_counts,
        axis=axis,
    )

    aux = []
    if return_backmap:
        aux.append(backmap)
    if return_inverse:
        # The backmap wasn't taken into account by unique_consecutive(), so we
        # have to do it ourselves.
        backmap_inv = swap_idcs_vals(backmap)  # [N_axis]
        aux.append(out[1][backmap_inv])
    if return_counts:
        aux.append(out[-1])

    if aux:
        return out[0], *aux
    return out


def groupby(
    keys: npt.NDArray[NpGeneric1], vals: npt.NDArray[NpGeneric2]
) -> Generator[tuple[NpGeneric1, npt.NDArray[NpGeneric2]]]:
    """Group values by keys.

    Args:
        keys: The keys to group by.
            Shape: [N]
        vals: The values to group.
            Shape: [N]

    Yields:
        Tuples containing:
        - A unique key. Will be yielded in sorted order.
        - The values that correspond to the key. Not guaranteed to be sorted.
            Shape: [N_key]
    """
    # Create a mapping from keys to values.
    keys_unique, backmap, counts = unique(
        keys, return_backmap=True, return_counts=True, axis=0
    )  # [E_unique], [E], [E_unique]
    vals = vals[backmap]  # sort values to match keys_unique
    end_slices = np.cumsum(counts)  # [E_unique]
    start_slices = end_slices - counts  # [E_unique]

    # Map each key to its corresponding values.
    for key, start_slice, end_slice in zip(
        keys_unique, start_slices, end_slices
    ):
        yield key, vals[start_slice:end_slice]


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
        old_histogram: The old normalized histogram. Will be updated in-place.
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
