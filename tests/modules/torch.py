from __future__ import annotations

import unittest

import numpy as np
import torch

from python_utils.modules.torch import (
    interp,
    lexsort_along,
    ravel_multi_index,
    swap_idcs_vals,
    swap_idcs_vals_duplicates,
    unique,
)


class TestInterp(unittest.TestCase):
    def test_interp_equivalent_np(self) -> None:
        x = torch.rand(100) * 102 - 1  # in [-1, 101)
        xp = torch.sort(torch.rand(100)).values * 100  # in [0, 100)
        fp = torch.rand(100)  # in [0, 1)
        left = -1
        right = 101
        self.assertTrue(
            torch.allclose(
                interp(x, xp, fp, left, right),
                torch.from_numpy(
                    np.interp(x, xp, fp, left, right).astype(np.float32)
                ),
            )
        )


class TestRavelMultiIndex(unittest.TestCase):
    def test_ravel_multi_index_equivalent_np(self) -> None:
        dims = torch.arange(10, 20)  # [10]
        multi_index = torch.stack(
            [torch.randint(0, int(dim), (10, 10)) for dim in dims]
        )  # [10, 10, 10]
        self.assertTrue(
            torch.equal(
                ravel_multi_index(multi_index, dims),
                torch.from_numpy(
                    np.ravel_multi_index(
                        multi_index.numpy(), dims.numpy()  # type: ignore
                    )
                ),
            )
        )
        self.assertTrue(
            torch.equal(
                ravel_multi_index(multi_index, dims, order="F"),
                torch.from_numpy(
                    np.ravel_multi_index(
                        multi_index.numpy(),  # type: ignore
                        dims.numpy(),  # type: ignore
                        order="F",
                    )
                ),
            )
        )
        multi_index = torch.stack([
            torch.concat([
                torch.randint(-2 * int(dim), -int(dim), (5, 10)),
                torch.randint(int(dim), 2 * int(dim), (5, 10)),
            ])
            for dim in dims
        ])  # [10, 10, 10]
        self.assertRaises(ValueError, ravel_multi_index, multi_index, dims)
        self.assertTrue(
            torch.equal(
                ravel_multi_index(multi_index, dims, mode="wrap"),
                torch.from_numpy(
                    np.ravel_multi_index(
                        multi_index.numpy(),  # type: ignore
                        dims.numpy(),  # type: ignore
                        mode="wrap",
                    )
                ),
            )
        )
        self.assertTrue(
            torch.equal(
                ravel_multi_index(multi_index, dims, mode="clip"),
                torch.from_numpy(
                    np.ravel_multi_index(
                        multi_index.numpy(),  # type: ignore
                        dims.numpy(),  # type: ignore
                        mode="clip",
                    )
                ),
            )
        )


class TestLexsortAlong(unittest.TestCase):
    def test_lexsort_along_1D_dim0(self) -> None:
        x = torch.as_tensor([4, 6, 2, 7, 0, 5, 1, 3])
        values, indices = lexsort_along(x, dim=0)
        self.assertTrue(
            torch.equal(values, torch.as_tensor([0, 1, 2, 3, 4, 5, 6, 7]))
        )
        self.assertTrue(
            torch.equal(indices, torch.as_tensor([4, 6, 2, 7, 0, 5, 1, 3]))
        )

    def test_lexsort_along_2D_dim0(self) -> None:
        x = torch.as_tensor([[2, 1], [3, 0], [1, 2], [1, 3]])
        values, indices = lexsort_along(x, dim=0)
        self.assertTrue(
            torch.equal(
                values, torch.as_tensor([[1, 2], [1, 3], [2, 1], [3, 0]])
            )
        )
        self.assertTrue(torch.equal(indices, torch.as_tensor([2, 3, 0, 1])))

    def test_lexsort_along_3D_dim1(self) -> None:
        x = torch.as_tensor([
            [[15, 13], [11, 4], [16, 2]],
            [[7, 21], [3, 20], [8, 22]],
            [[19, 14], [5, 12], [6, 0]],
            [[23, 1], [10, 17], [9, 18]],
        ])
        values, indices = lexsort_along(x, dim=1)
        self.assertTrue(
            torch.equal(
                values,
                torch.as_tensor([
                    [[11, 4], [15, 13], [16, 2]],
                    [[3, 20], [7, 21], [8, 22]],
                    [[5, 12], [19, 14], [6, 0]],
                    [[10, 17], [23, 1], [9, 18]],
                ]),
            )
        )
        self.assertTrue(torch.equal(indices, torch.as_tensor([1, 0, 2])))

    def test_lexsort_along_3D_dimminus1(self) -> None:
        x = torch.as_tensor([
            [[15, 13], [11, 4], [16, 2]],
            [[7, 21], [3, 20], [8, 22]],
            [[19, 14], [5, 12], [6, 0]],
            [[23, 1], [10, 17], [9, 18]],
        ])
        values, indices = lexsort_along(x, dim=-1)
        self.assertTrue(
            torch.equal(
                values,
                torch.as_tensor([
                    [[13, 15], [4, 11], [2, 16]],
                    [[21, 7], [20, 3], [22, 8]],
                    [[14, 19], [12, 5], [0, 6]],
                    [[1, 23], [17, 10], [18, 9]],
                ]),
            )
        )
        self.assertTrue(torch.equal(indices, torch.as_tensor([1, 0])))


class TestSwapIdcsVals(unittest.TestCase):
    def test_swap_idcs_vals_len5(self) -> None:
        x = torch.as_tensor([2, 3, 0, 4, 1])
        self.assertTrue(
            torch.equal(swap_idcs_vals(x), torch.as_tensor([2, 4, 0, 1, 3]))
        )

    def test_swap_idcs_vals_len10(self) -> None:
        x = torch.as_tensor([6, 3, 0, 1, 4, 7, 2, 8, 9, 5])
        self.assertTrue(
            torch.equal(
                swap_idcs_vals(x),
                torch.as_tensor([2, 3, 6, 1, 4, 9, 0, 5, 7, 8]),
            )
        )

    def test_swap_idcs_vals_2D(self) -> None:
        x = torch.as_tensor([[2, 3], [0, 4], [1, 5]])
        with self.assertRaises(ValueError):
            swap_idcs_vals(x)


class TestSwapIdcsValsDuplicates(unittest.TestCase):
    def test_swap_idcs_vals_duplicates_len5(self) -> None:
        x = torch.as_tensor([1, 2, 0, 1, 2])
        self.assertTrue(
            torch.equal(
                swap_idcs_vals_duplicates(x), torch.as_tensor([2, 0, 3, 1, 4])
            )
        )

    def test_swap_idcs_vals_duplicates_len10(self) -> None:
        x = torch.as_tensor([3, 3, 0, 3, 4, 2, 1, 1, 2, 0])
        self.assertTrue(
            torch.equal(
                swap_idcs_vals_duplicates(x),
                torch.as_tensor([2, 9, 6, 7, 5, 8, 0, 1, 3, 4]),
            )
        )

    def test_swap_idcs_vals_duplicates_2D(self) -> None:
        x = torch.as_tensor([[2, 3], [0, 4], [1, 5]])
        with self.assertRaises(ValueError):
            swap_idcs_vals_duplicates(x)

    def test_swap_idcs_vals_duplicates_no_duplicates(self) -> None:
        x = torch.as_tensor([2, 3, 0, 4, 1])
        self.assertTrue(
            torch.equal(
                swap_idcs_vals_duplicates(x), torch.as_tensor([2, 4, 0, 1, 3])
            )
        )


class TestUnique(unittest.TestCase):
    def test_unique_1D_dim0(self) -> None:
        # Should be the same as dim=None in the 1D case.
        x = torch.as_tensor([9, 10, 9, 9, 10, 9])
        dim = 0
        uniques, backmap, inverse, counts = unique(
            x,
            return_backmap=True,
            return_inverse=True,
            return_counts=True,
            dim=dim,
        )
        self.assertTrue(torch.equal(uniques, torch.as_tensor([9, 10])))
        self.assertTrue(
            torch.equal(backmap, torch.as_tensor([0, 2, 3, 5, 1, 4]))
        )
        self.assertTrue(
            torch.equal(inverse, torch.as_tensor([0, 1, 0, 0, 1, 0]))
        )
        self.assertTrue(torch.equal(counts, torch.as_tensor([4, 2])))

        self.assertTrue(
            torch.equal(x[backmap], torch.as_tensor([9, 9, 9, 9, 10, 10]))
        )

        self.assertTrue(
            torch.equal(backmap[: counts[0]], torch.as_tensor([0, 2, 3, 5]))
        )

        cumcounts = counts.cumsum(dim=0)
        get_idcs = lambda i: backmap[
            cumcounts[i - 1] : cumcounts[i]
        ]  # noqa: E731
        self.assertTrue(torch.equal(get_idcs(1), torch.as_tensor([1, 4])))

    def test_unique_1D_dimNone(self) -> None:
        # Not implemented, skip this test for now.
        return

    def test_unique_2D_dim1(self) -> None:
        x = torch.as_tensor(
            [[9, 10, 7, 9], [10, 9, 8, 10], [8, 7, 9, 8], [7, 7, 9, 7]]
        )
        dim = 1
        uniques, backmap, inverse, counts = unique(
            x,
            return_backmap=True,
            return_inverse=True,
            return_counts=True,
            dim=dim,
        )
        self.assertTrue(
            torch.equal(
                uniques,
                torch.as_tensor(
                    [[7, 9, 10], [8, 10, 9], [9, 8, 7], [9, 7, 7]]
                ),
            )
        )
        self.assertTrue(torch.equal(backmap, torch.as_tensor([2, 0, 3, 1])))
        self.assertTrue(torch.equal(inverse, torch.as_tensor([1, 2, 0, 1])))
        self.assertTrue(torch.equal(counts, torch.as_tensor([1, 2, 1])))

        self.assertTrue(
            torch.equal(
                x[:, backmap],
                torch.as_tensor(
                    [[7, 9, 9, 10], [8, 10, 10, 9], [9, 8, 8, 7], [9, 7, 7, 7]]
                ),
            )
        )

        self.assertTrue(
            torch.equal(backmap[: counts[0]], torch.as_tensor([2]))
        )

        cumcounts = counts.cumsum(dim=0)
        get_idcs = lambda i: backmap[
            cumcounts[i - 1] : cumcounts[i]
        ]  # noqa: E731
        self.assertTrue(torch.equal(get_idcs(1), torch.as_tensor([0, 3])))

    def test_unique_2D_dimNone(self) -> None:
        # Not implenmented, skip this test for now.
        return

    def test_unique_3D_dim2(self) -> None:
        x = torch.as_tensor([
            [[0, 2, 1, 2], [4, 5, 6, 5], [9, 7, 8, 7]],
            [[4, 8, 2, 8], [3, 7, 3, 7], [0, 1, 2, 1]],
        ])
        dim = 2
        uniques, backmap, inverse, counts = unique(
            x,
            return_backmap=True,
            return_inverse=True,
            return_counts=True,
            dim=dim,
        )
        self.assertTrue(
            torch.equal(
                uniques,
                torch.as_tensor([
                    [[0, 1, 2], [4, 6, 5], [9, 8, 7]],
                    [[4, 2, 8], [3, 3, 7], [0, 2, 1]],
                ]),
            )
        )
        self.assertTrue(torch.equal(backmap, torch.as_tensor([0, 2, 1, 3])))
        self.assertTrue(torch.equal(inverse, torch.as_tensor([0, 2, 1, 2])))
        self.assertTrue(torch.equal(counts, torch.as_tensor([1, 1, 2])))

        self.assertTrue(
            torch.equal(
                x[:, :, backmap],
                torch.as_tensor([
                    [[0, 1, 2, 2], [4, 6, 5, 5], [9, 8, 7, 7]],
                    [[4, 2, 8, 8], [3, 3, 7, 7], [0, 2, 1, 1]],
                ]),
            )
        )

        self.assertTrue(
            torch.equal(backmap[: counts[0]], torch.as_tensor([0]))
        )

        cumcounts = counts.cumsum(dim=0)
        get_idcs = lambda i: backmap[
            cumcounts[i - 1] : cumcounts[i]
        ]  # noqa: E731
        self.assertTrue(torch.equal(get_idcs(1), torch.as_tensor([2])))
