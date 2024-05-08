import unittest
from typing import cast

import torch

from modules.torch import (
    lexsort_along,
    swap_idcs_vals,
    swap_idcs_vals_duplicates,
    unique_with_backmap,
)


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
        x = torch.tensor([
            [[15, 13], [11, 4], [16, 2]],
            [[7, 21], [3, 20], [8, 22]],
            [[19, 14], [5, 12], [6, 0]],
            [[23, 1], [10, 17], [9, 18]],
        ])
        values, indices = lexsort_along(x, dim=1)
        self.assertTrue(
            torch.equal(
                values,
                torch.tensor([
                    [[11, 4], [15, 13], [16, 2]],
                    [[3, 20], [7, 21], [8, 22]],
                    [[5, 12], [19, 14], [6, 0]],
                    [[10, 17], [23, 1], [9, 18]],
                ]),
            )
        )
        self.assertTrue(torch.equal(indices, torch.tensor([1, 0, 2])))

    def test_lexsort_along_3D_dimminus1(self) -> None:
        x = torch.tensor([
            [[15, 13], [11, 4], [16, 2]],
            [[7, 21], [3, 20], [8, 22]],
            [[19, 14], [5, 12], [6, 0]],
            [[23, 1], [10, 17], [9, 18]],
        ])
        values, indices = lexsort_along(x, dim=-1)
        self.assertTrue(
            torch.equal(
                values,
                torch.tensor([
                    [[13, 15], [4, 11], [2, 16]],
                    [[21, 7], [20, 3], [22, 8]],
                    [[14, 19], [12, 5], [0, 6]],
                    [[1, 23], [17, 10], [18, 9]],
                ]),
            )
        )
        self.assertTrue(torch.equal(indices, torch.tensor([1, 0])))


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


class TestUniqueWithBackmap(unittest.TestCase):
    def test_unique_with_backmap_1D_dim0(self) -> None:
        # Should be the same as dim=None in the 1D case.
        x = torch.tensor([9, 10, 9, 9, 10, 9])
        dim = 0
        uniques, backmap, inverse, counts = unique_with_backmap(
            x, return_inverse=True, return_counts=True, dim=dim
        )
        backmap_along_dim = cast(torch.Tensor, backmap[dim])
        self.assertTrue(torch.equal(uniques, torch.tensor([9, 10])))
        for d in range(len(x.shape)):
            if d == dim:
                self.assertTrue(
                    torch.equal(
                        backmap_along_dim, torch.tensor([0, 2, 3, 5, 1, 4])
                    )
                )
            else:
                self.assertEqual(backmap[d], slice(None))
        self.assertTrue(torch.equal(inverse, torch.tensor([0, 1, 0, 0, 1, 0])))
        self.assertTrue(torch.equal(counts, torch.tensor([4, 2])))

        self.assertTrue(
            torch.equal(x[backmap], torch.tensor([9, 9, 9, 9, 10, 10]))
        )

        self.assertTrue(
            torch.equal(
                backmap_along_dim[: counts[0]], torch.tensor([0, 2, 3, 5])
            )
        )

        cumcounts = counts.cumsum(dim=0)
        get_idcs = lambda i: backmap_along_dim[cumcounts[i - 1] : cumcounts[i]]
        self.assertTrue(torch.equal(get_idcs(1), torch.tensor([1, 4])))

    def test_unique_with_backmap_1D_dimNone(self) -> None:
        # Not implemented, skip this test for now.
        return

    def test_unique_with_backmap_2D_dim1(self) -> None:
        x = torch.tensor(
            [[9, 10, 7, 9], [10, 9, 8, 10], [8, 7, 9, 8], [7, 7, 9, 7]]
        )
        dim = 1
        uniques, backmap, inverse, counts = unique_with_backmap(
            x, return_inverse=True, return_counts=True, dim=dim
        )
        backmap_along_dim = cast(torch.Tensor, backmap[dim])
        self.assertTrue(
            torch.equal(
                uniques,
                torch.tensor([[7, 9, 10], [8, 10, 9], [9, 8, 7], [9, 7, 7]]),
            )
        )
        for d in range(len(x.shape)):
            if d == dim:
                self.assertTrue(
                    torch.equal(backmap_along_dim, torch.tensor([2, 0, 3, 1]))
                )
            else:
                self.assertEqual(backmap[d], slice(None))
        self.assertTrue(torch.equal(inverse, torch.tensor([1, 2, 0, 1])))
        self.assertTrue(torch.equal(counts, torch.tensor([1, 2, 1])))

        self.assertTrue(
            torch.equal(
                x[backmap],
                torch.tensor(
                    [[7, 9, 9, 10], [8, 10, 10, 9], [9, 8, 8, 7], [9, 7, 7, 7]]
                ),
            )
        )

        self.assertTrue(
            torch.equal(backmap_along_dim[: counts[0]], torch.tensor([2]))
        )

        cumcounts = counts.cumsum(dim=0)
        get_idcs = lambda i: backmap_along_dim[cumcounts[i - 1] : cumcounts[i]]
        self.assertTrue(torch.equal(get_idcs(1), torch.tensor([0, 3])))

    def test_unique_with_backmap_2D_dimNone(self) -> None:
        # Not implenmented, skip this test for now.
        return

    def test_unique_with_backmap_3D_dim2(self) -> None:
        x = torch.tensor([
            [[0, 2, 1, 2], [4, 5, 6, 5], [9, 7, 8, 7]],
            [[4, 8, 2, 8], [3, 7, 3, 7], [0, 1, 2, 1]],
        ])
        dim = 2
        uniques, backmap, inverse, counts = unique_with_backmap(
            x, return_inverse=True, return_counts=True, dim=dim
        )
        backmap_along_dim = cast(torch.Tensor, backmap[dim])
        self.assertTrue(
            torch.equal(
                uniques,
                torch.tensor([
                    [[0, 1, 2], [4, 6, 5], [9, 8, 7]],
                    [[4, 2, 8], [3, 3, 7], [0, 2, 1]],
                ]),
            )
        )
        for d in range(len(x.shape)):
            if d == dim:
                self.assertTrue(
                    torch.equal(backmap_along_dim, torch.tensor([0, 2, 1, 3]))
                )
            else:
                self.assertEqual(backmap[d], slice(None))
        self.assertTrue(torch.equal(inverse, torch.tensor([0, 2, 1, 2])))
        self.assertTrue(torch.equal(counts, torch.tensor([1, 1, 2])))

        self.assertTrue(
            torch.equal(
                x[backmap],
                torch.tensor([
                    [[0, 1, 2, 2], [4, 6, 5, 5], [9, 8, 7, 7]],
                    [[4, 2, 8, 8], [3, 3, 7, 7], [0, 2, 1, 1]],
                ]),
            )
        )

        self.assertTrue(
            torch.equal(backmap_along_dim[: counts[0]], torch.tensor([0]))
        )

        cumcounts = counts.cumsum(dim=0)
        get_idcs = lambda i: backmap_along_dim[cumcounts[i - 1] : cumcounts[i]]
        self.assertTrue(torch.equal(get_idcs(1), torch.tensor([2])))
