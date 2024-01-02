import unittest
from typing import Literal, NamedTuple, overload

import numpy as np
import torch


SortOnlyDim = NamedTuple(
    "SortOnlyDim", [("values", torch.Tensor), ("indices", torch.Tensor)]
)


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
        >>> sort_only_dim(x, dim=0)
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

    # We can use np.lexsort() to only the given dimension. Unfortunately, torch
    # doesn't have a lexsort() equivalent, so we have to convert to numpy.

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
    inp_lexsort = x.transpose(0, dim).flatten(1).flip(dims=(1,)).unbind(1)
    out_lexsort = torch.from_numpy(np.lexsort(inp_lexsort)).to(x.device)

    # Now we have to convert the output back to a tensor. This is a bit tricky,
    # because we must be able to select indices from any given dimension. To do
    # this, we perform:
    x_sorted = x.index_select(dim, out_lexsort)

    # Finally, we return the sorted tensor and the indices.
    return SortOnlyDim(x_sorted, out_lexsort)


class TestLexsortAlong(unittest.TestCase):
    def test_lexsort_along_1D_dim0(self) -> None:
        x = torch.tensor([4, 6, 2, 7, 0, 5, 1, 3])
        values, indices = lexsort_along(x, dim=0)
        self.assertTrue(
            torch.equal(values, torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]))
        )
        self.assertTrue(
            torch.equal(indices, torch.tensor([4, 6, 2, 7, 0, 5, 1, 3]))
        )

    def test_lexsort_along_2D_dim0(self) -> None:
        x = torch.tensor([[2, 1], [3, 0], [1, 2], [1, 3]])
        values, indices = lexsort_along(x, dim=0)
        self.assertTrue(
            torch.equal(values, torch.tensor([[1, 2], [1, 3], [2, 1], [3, 0]]))
        )
        self.assertTrue(torch.equal(indices, torch.tensor([2, 3, 0, 1])))

    def test_lexsort_along_3D_dim1(self) -> None:
        x = torch.tensor(
            [
                [[15, 13], [11, 4], [16, 2]],
                [[7, 21], [3, 20], [8, 22]],
                [[19, 14], [5, 12], [6, 0]],
                [[23, 1], [10, 17], [9, 18]],
            ]
        )
        values, indices = lexsort_along(x, dim=1)
        self.assertTrue(
            torch.equal(
                values,
                torch.tensor(
                    [
                        [[11, 4], [15, 13], [16, 2]],
                        [[3, 20], [7, 21], [8, 22]],
                        [[5, 12], [19, 14], [6, 0]],
                        [[10, 17], [23, 1], [9, 18]],
                    ]
                ),
            )
        )
        self.assertTrue(torch.equal(indices, torch.tensor([1, 0, 2])))

    def test_lexsort_along_3D_dimminus1(self) -> None:
        x = torch.tensor(
            [
                [[15, 13], [11, 4], [16, 2]],
                [[7, 21], [3, 20], [8, 22]],
                [[19, 14], [5, 12], [6, 0]],
                [[23, 1], [10, 17], [9, 18]],
            ]
        )
        values, indices = lexsort_along(x, dim=-1)
        self.assertTrue(
            torch.equal(
                values,
                torch.tensor(
                    [
                        [[13, 15], [4, 11], [2, 16]],
                        [[21, 7], [20, 3], [22, 8]],
                        [[14, 19], [12, 5], [0, 6]],
                        [[1, 23], [17, 10], [18, 9]],
                    ]
                ),
            )
        )
        self.assertTrue(torch.equal(indices, torch.tensor([1, 0])))


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


class TestSwapIdcsVals(unittest.TestCase):
    def test_swap_idcs_vals_len5(self) -> None:
        x = torch.tensor([2, 3, 0, 4, 1])
        self.assertTrue(
            torch.equal(swap_idcs_vals(x), torch.tensor([2, 4, 0, 1, 3]))
        )

    def test_swap_idcs_vals_len10(self) -> None:
        x = torch.tensor([6, 3, 0, 1, 4, 7, 2, 8, 9, 5])
        self.assertTrue(
            torch.equal(
                swap_idcs_vals(x), torch.tensor([2, 3, 6, 1, 4, 9, 0, 5, 7, 8])
            )
        )

    def test_swap_idcs_vals_2D(self) -> None:
        x = torch.tensor([[2, 3], [0, 4], [1, 5]])
        with self.assertRaises(ValueError):
            swap_idcs_vals(x)


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


class TestSwapIdcsValsDuplicates(unittest.TestCase):
    def test_swap_idcs_vals_duplicates_len5(self) -> None:
        x = torch.tensor([1, 2, 0, 1, 2])
        self.assertTrue(
            torch.equal(
                swap_idcs_vals_duplicates(x), torch.tensor([2, 0, 3, 1, 4])
            )
        )

    def test_swap_idcs_vals_duplicates_len10(self) -> None:
        x = torch.tensor([3, 3, 0, 3, 4, 2, 1, 1, 2, 0])
        self.assertTrue(
            torch.equal(
                swap_idcs_vals_duplicates(x),
                torch.tensor([2, 9, 6, 7, 5, 8, 0, 1, 3, 4]),
            )
        )

    def test_swap_idcs_vals_duplicates_2D(self) -> None:
        x = torch.tensor([[2, 3], [0, 4], [1, 5]])
        with self.assertRaises(ValueError):
            swap_idcs_vals_duplicates(x)

    def test_swap_idcs_vals_duplicates_no_duplicates(self) -> None:
        x = torch.tensor([2, 3, 0, 4, 1])
        self.assertTrue(
            torch.equal(
                swap_idcs_vals_duplicates(x), torch.tensor([2, 4, 0, 1, 3])
            )
        )


@overload
def unique_with_backmap(
    x: torch.Tensor,
    return_inverse: Literal[False] = False,
    return_counts: Literal[False] = False,
    dim: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
    tuple[torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
):
    """Like torch.unique, but also returns a back map.

    Note that x is always sorted in the returned unique tensor.

    Args:
        x: The input tensor.
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
            - The back map. This is a tensor of tensors that maps the original
              input indices to the unique values.
            - (optional) if return_inverse is True, the indices where elements
              in the original input ended up in the returned unique values.
            - (optional) if return_counts is True, the counts for each unique
              element.

    Examples:
        >>> x = torch.tensor([9, 10, 9, 9, 10, 9])
        >>> uniques, backmap, inverse, counts = torch.unique_backmap(
        >>>     x, return_inverse=True, return_counts=True
        >>> )
        >>> uniques
        tensor([9, 10])
        >>> inverse
        tensor([0, 1, 0, 0, 1, 0])
        >>> counts
        tensor([4, 2])
        >>> backmap
        tensor([[0, 2, 3, 5, 1, 4]])
        >>> x[backmap.unbind()]
        tensor([9, 9, 9, 9, 10, 10])

        >>> # Get all indices of the first unique value:
        >>> backmap[:, :counts[0]]
        tensor([[0, 2, 3, 5]])

        >>> # Get all indices of the i'th unique value (for i > 0):
        >>> cumcounts = counts.cumsum(dim=0)
        >>> get_idcs = lambda i: backmap[:, cumcounts[i - 1] : cumcounts[i]]
        >>> get_idcs(1)
        tensor([[1, 4]])
    """
    if dim is None:
        raise NotImplementedError(
            "dim=None is not implemented yet. Please explicitly specify a"
            " dimension."
        )

    # Sort along the given dimension, taking all the other dimensions as
    # constant tuples. torch's sort() doesn't work here since it will
    # sort the other dimensions as well.
    x_sorted, backmap = lexsort_along(x, dim=dim)

    unique, *aux = torch.unique_consecutive(
        x_sorted,
        return_inverse=return_inverse,
        return_counts=return_counts,
        dim=dim,
    )

    if return_inverse:
        # The backmap wasn't taken into account by torch.unique_consecutive(),
        # so we have to do it ourselves.
        backmap_inverse = swap_idcs_vals(backmap)
        aux[0] = aux[0][backmap_inverse]

    backmap = backmap.unsqueeze(dim)

    return unique, backmap, *aux  # type: ignore


class TestUniqueWithBackmap(unittest.TestCase):
    def test_unique_with_backmap_1D_dim0(self) -> None:
        # Should be the same as dim=None in the 1D case.
        x = torch.tensor([9, 10, 9, 9, 10, 9])
        uniques, backmap, inverse, counts = unique_with_backmap(
            x, return_inverse=True, return_counts=True, dim=0
        )
        self.assertTrue(torch.equal(uniques, torch.tensor([9, 10])))
        self.assertTrue(torch.equal(inverse, torch.tensor([0, 1, 0, 0, 1, 0])))
        self.assertTrue(torch.equal(counts, torch.tensor([4, 2])))
        self.assertTrue(
            torch.equal(backmap, torch.tensor([[0, 2, 3, 5, 1, 4]]))
        )
        self.assertTrue(
            torch.equal(
                x[backmap.unbind()], torch.tensor([9, 9, 9, 9, 10, 10])
            )
        )
        self.assertTrue(
            torch.equal(backmap[:, : counts[0]], torch.tensor([[0, 2, 3, 5]]))
        )
        cumcounts = counts.cumsum(dim=0)
        get_idcs = lambda i: backmap[:, cumcounts[i - 1] : cumcounts[i]]
        self.assertTrue(torch.equal(get_idcs(1), torch.tensor([[1, 4]])))

    def test_unique_with_backmap_1D_dimNone(self) -> None:
        # Not implemented, skip this test for now.
        return

        x = torch.tensor([9, 10, 9, 9, 10, 9])
        uniques, backmap, inverse, counts = unique_with_backmap(
            x, return_inverse=True, return_counts=True, dim=None
        )
        self.assertTrue(torch.equal(uniques, torch.tensor([9, 10])))
        self.assertTrue(torch.equal(inverse, torch.tensor([0, 1, 0, 0, 1, 0])))
        self.assertTrue(torch.equal(counts, torch.tensor([4, 2])))
        self.assertTrue(
            torch.equal(backmap, torch.tensor([[0, 2, 3, 5, 1, 4]]))
        )
        self.assertTrue(
            torch.equal(
                x[backmap.unbind()], torch.tensor([9, 9, 9, 9, 10, 10])
            )
        )
        self.assertTrue(
            torch.equal(backmap[:, : counts[0]], torch.tensor([[0, 2, 3, 5]]))
        )
        cumcounts = counts.cumsum(dim=0)
        get_idcs = lambda i: backmap[:, cumcounts[i - 1] : cumcounts[i]]
        self.assertTrue(torch.equal(get_idcs(1), torch.tensor([[1, 4]])))

    def test_unique_with_backmap_2D_dim0(self) -> None:
        x = torch.tensor(
            [[9, 10, 8, 7], [10, 9, 7, 7], [7, 8, 9, 9], [9, 10, 8, 7]]
        )
        uniques, backmap, inverse, counts = unique_with_backmap(
            x, return_inverse=True, return_counts=True, dim=0
        )
        self.assertTrue(
            torch.equal(
                uniques,
                torch.tensor([[7, 8, 9, 9], [9, 10, 8, 7], [10, 9, 7, 7]]),
            )
        )
        self.assertTrue(torch.equal(inverse, torch.tensor([1, 2, 0, 1])))
        self.assertTrue(torch.equal(counts, torch.tensor([1, 2, 1])))
        self.assertTrue(torch.equal(backmap, torch.tensor([[2, 0, 3, 1]])))
        self.assertTrue(
            torch.equal(
                x[backmap.unbind()],
                torch.tensor(
                    [[7, 8, 9, 9], [9, 10, 8, 7], [9, 10, 8, 7], [10, 9, 7, 7]]
                ),
            )
        )
        self.assertTrue(
            torch.equal(backmap[:, : counts[0]], torch.tensor([[2]]))
        )
        cumcounts = counts.cumsum(dim=0)
        get_idcs = lambda i: backmap[:, cumcounts[i - 1] : cumcounts[i]]
        self.assertTrue(torch.equal(get_idcs(1), torch.tensor([[0, 3]])))

    def test_unique_with_backmap_2D_dimNone(self) -> None:
        # Not implenmented, skip this test for now.
        return

        x = torch.tensor(
            [[9, 10, 8, 7], [10, 9, 7, 7], [7, 8, 9, 9], [9, 10, 8, 7]]
        )
        uniques, backmap, inverse, counts = unique_with_backmap(
            x, return_inverse=True, return_counts=True, dim=None
        )
        self.assertTrue(torch.equal(uniques, torch.tensor([7, 8, 9, 10])))
        self.assertTrue(
            torch.equal(
                inverse,
                torch.tensor(
                    [[2, 3, 1, 0], [3, 2, 0, 0], [0, 1, 2, 2], [2, 3, 1, 0]]
                ),
            )
        )
        self.assertTrue(torch.equal(counts, torch.tensor([5, 3, 5, 3])))
        self.assertTrue(
            torch.equal(
                backmap,
                torch.tensor(
                    [
                        [0, 1, 1, 2, 3, 0, 2, 3, 0, 1, 2, 2, 3, 0, 1, 3],
                        [3, 2, 3, 0, 3, 2, 1, 2, 0, 1, 2, 3, 0, 1, 0, 1],
                    ]
                ),
            )
        )
        self.assertTrue(
            torch.equal(
                x[backmap.unbind()],
                torch.tensor(
                    [7, 7, 7, 7, 7, 8, 8, 8, 9, 9, 9, 9, 9, 10, 10, 10]
                ),
            )
        )
        self.assertTrue(
            torch.equal(
                backmap[:, : counts[0]],
                torch.tensor([[0, 1, 1, 2, 3], [3, 2, 3, 0, 3]]),
            )
        )
        cumcounts = counts.cumsum(dim=0)
        get_idcs = lambda i: backmap[:, cumcounts[i - 1] : cumcounts[i]]
        self.assertTrue(
            torch.equal(get_idcs(1), torch.tensor([[0, 2, 3], [2, 1, 2]]))
        )


def unique_with_backmap_naive(
    x: torch.Tensor,
    return_inverse: bool = False,
    return_counts: bool = False,
    dim: int | None = None,
) -> (
    tuple[torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
):
    if dim is None:
        raise NotImplementedError(
            "dim=None is not implemented yet. Please explicitly specify a"
            " dimension."
        )

    unique, inverse, *aux = torch.unique(
        x, return_inverse=True, return_counts=return_counts, dim=dim
    )
    backmap = swap_idcs_vals_duplicates(inverse).unsqueeze(0)
    if return_inverse:
        return unique, backmap, inverse, *aux  # type: ignore
    return unique, backmap, *aux  # type: ignore
