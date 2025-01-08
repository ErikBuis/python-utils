import unittest

import numpy as np
import torch

from python_utils.modules_batched.torch import interp_batched


class TestInterpBatched(unittest.TestCase):
    def test_interp_batched_equivalent_np(self) -> None:
        B = 10
        x = torch.rand((B, 100)) * 102 - 1  # in [-1, 101)
        xp = (
            torch.sort(torch.rand((B, 100)), dim=1).values * 100
        )  # in [0, 100)
        fp = torch.rand((B, 100))  # in [0, 1)
        left = torch.full((B,), -1)
        right = torch.full((B,), 101)
        self.assertTrue(
            torch.allclose(
                interp_batched(x, xp, fp, left, right),
                torch.stack([
                    torch.from_numpy(
                        np.interp(
                            x_b, xp_b, fp_b, left_b.item(), right_b.item()
                        ).astype(np.float32)
                    )
                    for x_b, xp_b, fp_b, left_b, right_b in zip(
                        x, xp, fp, left, right
                    )
                ]),
            )
        )
