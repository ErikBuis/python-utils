import unittest

import torch
import torch.nn as nn

from python_utils.custom.geometry import xiaolin_wu_anti_aliasing
from python_utils.custom_batched.geometry import (
    xiaolin_wu_anti_aliasing_batched,
)


class XiaolinWuAntiAliasingBatched(unittest.TestCase):
    # xiaolin_wu_anti_aliasing_batched should return the same values as the
    # xiaolin_wu_anti_aliasing function, but in a batched manner.
    def test_xiaolin_wu_anti_aliasing_batched(self) -> None:
        # Generate random line segments.
        x0s = torch.concat([
            torch.randint(-100, 101, (50,)) / 2,
            torch.rand(50, dtype=torch.float64) * 100 - 50,
        ])
        y0s = torch.concat([
            torch.randint(-100, 101, (50,)) / 2,
            torch.rand(50, dtype=torch.float64) * 100 - 50,
        ])
        x1s = torch.concat([
            torch.randint(-100, 101, (50,)) / 2,
            torch.rand(50, dtype=torch.float64) * 100 - 50,
        ])
        y1s = torch.concat([
            torch.randint(-100, 101, (50,)) / 2,
            torch.rand(50, dtype=torch.float64) * 100 - 50,
        ])

        # Get the values from the sequential function.
        pixels_x_seq = []
        pixels_y_seq = []
        vals_seq = []
        for x0, y0, x1, y1 in zip(x0s, y0s, x1s, y1s):
            pixels_x, pixels_y, vals = xiaolin_wu_anti_aliasing(
                x0.item(), y0.item(), x1.item(), y1.item()
            )
            pixels_x_seq.append(pixels_x)
            pixels_y_seq.append(pixels_y)
            vals_seq.append(vals)
        S_bs_seq = torch.as_tensor(list(map(len, vals_seq)))
        pixels_x_seq = nn.utils.rnn.pad_sequence(
            pixels_x_seq, batch_first=True
        )
        pixels_y_seq = nn.utils.rnn.pad_sequence(
            pixels_y_seq, batch_first=True
        )
        vals_seq = nn.utils.rnn.pad_sequence(vals_seq, batch_first=True)

        # Get the values from the batched function.
        pixels_x_bat, pixels_y_bat, vals_bat, S_bs_bat = (
            xiaolin_wu_anti_aliasing_batched(x0s, y0s, x1s, y1s)
        )

        # Check if the values are the same.
        self.assertEqual(pixels_x_seq.shape, pixels_x_bat.shape)
        self.assertEqual(pixels_y_seq.shape, pixels_y_bat.shape)
        self.assertEqual(vals_seq.shape, vals_bat.shape)
        self.assertTrue(torch.allclose(S_bs_seq, S_bs_bat))
        self.assertTrue(torch.allclose(pixels_x_seq, pixels_x_bat))
        self.assertTrue(torch.allclose(pixels_y_seq, pixels_y_bat))
        self.assertTrue(torch.allclose(vals_seq, vals_bat))
