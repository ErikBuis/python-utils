from __future__ import annotations

import random
import unittest

import numpy as np

from python_utils.custom import graphics
from python_utils.modules.numpy import lexsort_along


class XiaolinWuAntiAliasing(unittest.TestCase):
    # xiaolin_wu_anti_aliasing should return the same values as the
    # xiaolin_wu_anti_aliasing_naive function.
    def test_xialin_wu_anti_aliasing(self) -> None:
        for i in range(100):
            if i < 50:
                x0 = random.randint(-100, 100) / 2
                y0 = random.randint(-100, 100) / 2
                x1 = random.randint(-100, 100) / 2
                y1 = random.randint(-100, 100) / 2
            else:
                x0 = random.random() * 100 - 50
                y0 = random.random() * 100 - 50
                x1 = random.random() * 100 - 50
                y1 = random.random() * 100 - 50

            # First get the values from the optimized function.
            pixels_x_1, pixels_y_1, vals_1 = graphics.xiaolin_wu_anti_aliasing(
                x0, y0, x1, y1
            )

            # Remove empty pixels and sort to ensure the order is the same.
            ret = np.stack([pixels_x_1, pixels_y_1, vals_1], axis=1)
            ret = ret[~np.isclose(vals_1, 0)]
            ret, _ = lexsort_along(ret, axis=0)
            pixels_x_1, pixels_y_1, vals_1 = ret[:, 0], ret[:, 1], ret[:, 2]

            # Now get the values from the naive function.
            pixels_x_2, pixels_y_2, vals_2 = (
                graphics.xiaolin_wu_anti_aliasing_naive(x0, y0, x1, y1)
            )

            # Remove empty pixels and sort to ensure the order is the same.
            ret = np.stack([pixels_x_2, pixels_y_2, vals_2], axis=1)
            ret = ret[~np.isclose(vals_2, 0)]
            ret, _ = lexsort_along(ret, axis=0)
            pixels_x_2, pixels_y_2, vals_2 = ret[:, 0], ret[:, 1], ret[:, 2]

            # Check if the values are the same.
            self.assertEqual(len(pixels_x_1), len(pixels_x_2))
            self.assertEqual(len(pixels_y_1), len(pixels_y_2))
            self.assertEqual(len(vals_1), len(vals_2))
            self.assertTrue(
                np.allclose(pixels_x_1, pixels_x_2, rtol=1e-5, atol=1e-8)
            )
            self.assertTrue(
                np.allclose(pixels_y_1, pixels_y_2, rtol=1e-5, atol=1e-8)
            )
            self.assertTrue(np.allclose(vals_1, vals_2, rtol=1e-5, atol=1e-8))
