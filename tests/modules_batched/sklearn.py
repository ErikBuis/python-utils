import unittest
from math import ceil, isqrt, sqrt
from typing import cast

import matplotlib.axes
import torch
from matplotlib import pyplot as plt

from python_utils.modules.math import optimal_grid_layout, optimal_size
from python_utils.modules_batched.sklearn import ransac_batched


@unittest.skipUnless(
    False,
    "Running this test will open a window. It has been hardcoded to be skipped"
    " for now. If you want to run it, change the condition to True.",
)
class TestRansacBatched(unittest.TestCase):
    def test_visually(self) -> None:
        # Set input parameters.
        B = 12
        max_L_bs = 10
        max_r = 20

        # Generate random lines.
        rs = torch.rand((B, max_L_bs)) * max_r
        thetas = torch.rand((B, max_L_bs)) * 2 * torch.pi
        L_bs = torch.randint(2, max_L_bs + 1, (B,))
        lines = (rs, thetas)
        max_view = max_r * 2
        max_t = ceil(max_view * sqrt(2))

        # Calculate a good intersection point using RANSAC.
        intersections, _ = ransac_batched(
            lines,
            L_bs,
            max_iterations=L_bs,
            inlier_threshold=max_r / 5,
            min_inliers=L_bs * 2 // 3,
        )  # [B, 2]

        # Plot the lines and intersection points.
        ncols, nrows = optimal_grid_layout(
            B, isqrt(B - 1) + 2, isqrt(B - 1) + 2
        )
        figsize = optimal_size(ncols / nrows, 12, 8)
        fig, axs = plt.subplots(
            nrows, ncols, sharex=True, sharey=True, figsize=figsize
        )
        for b, ax in enumerate(axs.flatten()):
            ax = cast(matplotlib.axes.Axes, ax)

            r_sample = rs[b, : L_bs[b]]  # [L_b]
            theta_sample = thetas[b, : L_bs[b]]  # [L_b]

            # Normal vector of the line.
            n_lines = torch.stack(
                [torch.cos(theta_sample), torch.sin(theta_sample)], dim=1
            )  # [L_b, 2]
            # Point on the line closest to the origin.
            v_lines = r_sample.unsqueeze(1) * n_lines  # [L_b, 2]
            # Unit vector pointing along the line.
            u_lines = torch.stack(
                [n_lines[:, 1], -n_lines[:, 0]], dim=1
            )  # [L_b, 2]
            # Points along the line.
            t = torch.arange(-max_t, max_t + 1)  # [T]
            points_on_lines = (
                v_lines.unsqueeze(1) + t.unsqueeze(1) * u_lines.unsqueeze(1)
            )  # [L_b, T, 2]  # fmt: skip

            # Plot the lines.
            ax.plot(
                points_on_lines[:, :, 0].T,  # [T, L_b]
                points_on_lines[:, :, 1].T,  # [T, L_b]
                alpha=0.5,
            )

            # Plot the RANSAC intersection point.
            ax.scatter(intersections[b, 0], intersections[b, 1], color="red")
            ax.set_xlim(-max_view, max_view)
            ax.set_ylim(-max_view, max_view)

        fig.suptitle("Please assess the quality of the intersection points.")
        fig.supxlabel("x")
        fig.supylabel("y")
        fig.tight_layout()
        plt.show()
