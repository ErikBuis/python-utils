from __future__ import annotations

import unittest

import numpy as np

from python_utils.modules.numpy import lexsort_along, unique


class TestLexsortAlong(unittest.TestCase):
    def test_lexsort_along_1D_axis0(self) -> None:
        x = np.array([4, 6, 2, 7, 0, 5, 1, 3])
        values, backmap = lexsort_along(x, axis=0)
        self.assertTrue(
            np.array_equal(values, np.array([0, 1, 2, 3, 4, 5, 6, 7]))
        )
        self.assertTrue(
            np.array_equal(backmap, np.array([4, 6, 2, 7, 0, 5, 1, 3]))
        )

    def test_lexsort_along_2D_axis0(self) -> None:
        x = np.array([[2, 1], [3, 0], [1, 2], [1, 3]])
        values, backmap = lexsort_along(x, axis=0)
        self.assertTrue(
            np.array_equal(values, np.array([[1, 2], [1, 3], [2, 1], [3, 0]]))
        )
        self.assertTrue(np.array_equal(backmap, np.array([2, 3, 0, 1])))

    def test_lexsort_along_3D_axis1(self) -> None:
        x = np.array([
            [[15, 13], [11, 4], [16, 2]],
            [[7, 21], [3, 20], [8, 22]],
            [[19, 14], [5, 12], [6, 0]],
            [[23, 1], [10, 17], [9, 18]],
        ])
        values, backmap = lexsort_along(x, axis=1)
        self.assertTrue(
            np.array_equal(
                values,
                np.array([
                    [[11, 4], [15, 13], [16, 2]],
                    [[3, 20], [7, 21], [8, 22]],
                    [[5, 12], [19, 14], [6, 0]],
                    [[10, 17], [23, 1], [9, 18]],
                ]),
            )
        )
        self.assertTrue(np.array_equal(backmap, np.array([1, 0, 2])))

    def test_lexsort_along_3D_axisminus1(self) -> None:
        x = np.array([
            [[15, 13], [11, 4], [16, 2]],
            [[7, 21], [3, 20], [8, 22]],
            [[19, 14], [5, 12], [6, 0]],
            [[23, 1], [10, 17], [9, 18]],
        ])
        values, backmap = lexsort_along(x, axis=-1)
        self.assertTrue(
            np.array_equal(
                values,
                np.array([
                    [[13, 15], [4, 11], [2, 16]],
                    [[21, 7], [20, 3], [22, 8]],
                    [[14, 19], [12, 5], [0, 6]],
                    [[1, 23], [17, 10], [18, 9]],
                ]),
            )
        )
        self.assertTrue(np.array_equal(backmap, np.array([1, 0])))


class TestUnique(unittest.TestCase):
    def test_unique_1D_axis0(self) -> None:
        # Should be the same as axis=None in the 1D case.
        x = np.array([9, 10, 9, 9, 10, 9])
        axis = 0
        uniques, backmap, inverse, counts = unique(
            x,
            return_backmap=True,
            return_inverse=True,
            return_counts=True,
            axis=axis,
            stable=True,
        )
        self.assertTrue(np.array_equal(uniques, np.array([9, 10])))
        self.assertTrue(np.array_equal(backmap, np.array([0, 2, 3, 5, 1, 4])))
        self.assertTrue(np.array_equal(inverse, np.array([0, 1, 0, 0, 1, 0])))
        self.assertTrue(np.array_equal(counts, np.array([4, 2])))

        self.assertTrue(
            np.array_equal(x[backmap], np.array([9, 9, 9, 9, 10, 10]))
        )

        self.assertTrue(
            np.array_equal(backmap[: counts[0]], np.array([0, 2, 3, 5]))
        )

        cumcounts = counts.cumsum(axis=0)
        get_idcs = lambda i: backmap[
            cumcounts[i - 1] : cumcounts[i]
        ]  # noqa: E731
        self.assertTrue(np.array_equal(get_idcs(1), np.array([1, 4])))

    def test_unique_1D_axisNone(self) -> None:
        # Not implemented, skip this test for now.
        return

    def test_unique_2D_axis1(self) -> None:
        x = np.array(
            [[9, 10, 7, 9], [10, 9, 8, 10], [8, 7, 9, 8], [7, 7, 9, 7]]
        )
        axis = 1
        uniques, backmap, inverse, counts = unique(
            x,
            return_backmap=True,
            return_inverse=True,
            return_counts=True,
            axis=axis,
            stable=True,
        )
        self.assertTrue(
            np.array_equal(
                uniques,
                np.array([[7, 9, 10], [8, 10, 9], [9, 8, 7], [9, 7, 7]]),
            )
        )
        self.assertTrue(np.array_equal(backmap, np.array([2, 0, 3, 1])))
        self.assertTrue(np.array_equal(inverse, np.array([1, 2, 0, 1])))
        self.assertTrue(np.array_equal(counts, np.array([1, 2, 1])))

        self.assertTrue(
            np.array_equal(
                x[:, backmap],
                np.array(
                    [[7, 9, 9, 10], [8, 10, 10, 9], [9, 8, 8, 7], [9, 7, 7, 7]]
                ),
            )
        )

        self.assertTrue(np.array_equal(backmap[: counts[0]], np.array([2])))

        cumcounts = counts.cumsum(axis=0)
        get_idcs = lambda i: backmap[
            cumcounts[i - 1] : cumcounts[i]
        ]  # noqa: E731
        self.assertTrue(np.array_equal(get_idcs(1), np.array([0, 3])))

    def test_unique_2D_axisNone(self) -> None:
        # Not implenmented, skip this test for now.
        return

    def test_unique_3D_axis2(self) -> None:
        x = np.array([
            [[0, 2, 1, 2], [4, 5, 6, 5], [9, 7, 8, 7]],
            [[4, 8, 2, 8], [3, 7, 3, 7], [0, 1, 2, 1]],
        ])
        axis = 2
        uniques, backmap, inverse, counts = unique(
            x,
            return_backmap=True,
            return_inverse=True,
            return_counts=True,
            axis=axis,
            stable=True,
        )
        self.assertTrue(
            np.array_equal(
                uniques,
                np.array([
                    [[0, 1, 2], [4, 6, 5], [9, 8, 7]],
                    [[4, 2, 8], [3, 3, 7], [0, 2, 1]],
                ]),
            )
        )
        self.assertTrue(np.array_equal(backmap, np.array([0, 2, 1, 3])))
        self.assertTrue(np.array_equal(inverse, np.array([0, 2, 1, 2])))
        self.assertTrue(np.array_equal(counts, np.array([1, 1, 2])))

        self.assertTrue(
            np.array_equal(
                x[:, :, backmap],
                np.array([
                    [[0, 1, 2, 2], [4, 6, 5, 5], [9, 8, 7, 7]],
                    [[4, 2, 8, 8], [3, 3, 7, 7], [0, 2, 1, 1]],
                ]),
            )
        )

        self.assertTrue(np.array_equal(backmap[: counts[0]], np.array([0])))

        cumcounts = counts.cumsum(axis=0)
        get_idcs = lambda i: backmap[
            cumcounts[i - 1] : cumcounts[i]
        ]  # noqa: E731
        self.assertTrue(np.array_equal(get_idcs(1), np.array([2])))
