import unittest

import numpy as np

from utils.modules.numpy import lexsort_along


class TestLexsortAlong(unittest.TestCase):
    def test_lexsort_along_1D_axis0(self) -> None:
        x = np.array([4, 6, 2, 7, 0, 5, 1, 3])
        values, indices = lexsort_along(x, axis=0)
        self.assertTrue(
            np.array_equal(values, np.array([0, 1, 2, 3, 4, 5, 6, 7]))
        )
        self.assertTrue(
            np.array_equal(indices, np.array([4, 6, 2, 7, 0, 5, 1, 3]))
        )

    def test_lexsort_along_2D_axis0(self) -> None:
        x = np.array([[2, 1], [3, 0], [1, 2], [1, 3]])
        values, indices = lexsort_along(x, axis=0)
        self.assertTrue(
            np.array_equal(values, np.array([[1, 2], [1, 3], [2, 1], [3, 0]]))
        )
        self.assertTrue(np.array_equal(indices, np.array([2, 3, 0, 1])))

    def test_lexsort_along_3D_axis1(self) -> None:
        x = np.array([
            [[15, 13], [11, 4], [16, 2]],
            [[7, 21], [3, 20], [8, 22]],
            [[19, 14], [5, 12], [6, 0]],
            [[23, 1], [10, 17], [9, 18]],
        ])
        values, indices = lexsort_along(x, axis=1)
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
        self.assertTrue(np.array_equal(indices, np.array([1, 0, 2])))

    def test_lexsort_along_3D_axisminus1(self) -> None:
        x = np.array([
            [[15, 13], [11, 4], [16, 2]],
            [[7, 21], [3, 20], [8, 22]],
            [[19, 14], [5, 12], [6, 0]],
            [[23, 1], [10, 17], [9, 18]],
        ])
        values, indices = lexsort_along(x, axis=-1)
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
        self.assertTrue(np.array_equal(indices, np.array([1, 0])))
