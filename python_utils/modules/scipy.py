from __future__ import annotations

import logging
from typing import cast

import numpy as np
import numpy.typing as npt
from scipy.spatial import Voronoi

logger = logging.getLogger(__name__)


def voronoi_constrain_to_rect(
    points: npt.NDArray[np.float64], rect: tuple[float, float, float, float]
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.int32],
    list[list[int]],
    npt.NDArray[np.signedinteger],
    list[list[int]],
]:
    """Compute Voronoi regions, but constrain all regions in a rectangle.

    The input must be 2D.

    The convex hull around the input points must be non-degenerate (i.e. the
    input points must not be colinear).

    The Vonorio diagram will be computed using a Euclidean plane.

    If the points in or on the rectangle were passed to the original Voronoi
    class, its attributes would have the following shapes:
    Shape of vor.points: [P', 2]
    Shape of vor.vertices: [V, 2]
    Shape of vor.ridge_points: [R, 2]
    Length of vor.ridge_vertices: R
    Shape of vor.point_region: [P']
    Length of vor.regions: P' + 1

    The core idea for this function (mirroring the points) has been taken from:
    https://stackoverflow.com/a/33602171/15636460
    However, the code itself was not taken from this answer because of a bug on
    rect edges and a different desired output format.

    Args:
        points: The points to compute the Voronoi diagram for.
            Shape: [P, 2]
        rect: The rectangle to clip the infinite regions in. The rectangle is
            represented by the tuple (x_min, y_min, x_max, y_max).

    Returns:
        Tuple containing:
        - The points in or on the rectangle.
            Shape: [P', 2]
        - The original vertices returned by Voronoi, plus vertices added on the
            given rectangle to constrain the infinite regions.
            Shape: [V + X, 2]
        - The original ridge_points returned by Voronoi, plus ridges added on
            the given rectangle to constrain the infinite regions.
            Shape: [R + X, 2]
        - The original ridge_vertices returned by Voronoi, plus ridges added on
            the given rectangle to constrain the infinite regions.
            Length: R + X
        - The original point_region returned by Voronoi.
            Shape: [P']
        - The original regions returned by Voronoi, where the infinite regions
            have been constrained to finite ones.
            Length: P' + 1
    """
    # This function mirrors the points in 4 directions, after which it computes
    # the Voronoi diagram. Theoretically, the runtime of this algorithm is
    # still O(n log n), where n is the number of original points. This is
    # because we are passing 5n points to the Voronoi class, which therefore
    # has a runtime of:
    # O(5n log(5n)) = O(5n log(5) + 5n log(n))
    #               = O(c_1 n log(n) + c_2 n)
    #               = O(n log(n))
    # However, in practice, n does not go to infinity, so the runtime of this
    # function is approximately 10 times slower:
    # 5n log(5n) = 5n log(5) + 5n log(n)
    #           ~= 5n log(n) + 5n log(n)
    #            = 10 * n log(n)

    # The input must be 2D.
    if points.shape[1] != 2:
        raise ValueError(
            f"Points must be 2D, but received {points.shape[1]}D coordinates."
        )

    eps = 1e-8

    # Check which points lie inside the rectangle.
    is_point_in_rect = (
        (rect[0] + eps < points[:, 0])
        & (points[:, 0] < rect[2] - eps)
        & (rect[1] + eps < points[:, 1])
        & (points[:, 1] < rect[3] - eps)
    )

    # Check which points lie on which boundary(s) of the rectangle.
    is_point_on_bound_left = (
        np.isclose(points[:, 0], rect[0], atol=eps)
        & (rect[1] - eps <= points[:, 1])
        & (points[:, 1] <= rect[3] + eps)
    )
    is_point_on_bound_right = (
        np.isclose(points[:, 0], rect[2], atol=eps)
        & (rect[1] - eps <= points[:, 1])
        & (points[:, 1] <= rect[3] + eps)
    )
    is_point_on_bound_lower = (
        np.isclose(points[:, 1], rect[1], atol=eps)
        & (rect[0] - eps <= points[:, 0])
        & (points[:, 0] <= rect[2] + eps)
    )
    is_point_on_bound_upper = (
        np.isclose(points[:, 1], rect[3], atol=eps)
        & (rect[0] - eps <= points[:, 0])
        & (points[:, 0] <= rect[2] + eps)
    )

    # Make sure to offset the points on the boundary slightly, so that they
    # are mirrored correctly.
    points_to_override = points.copy()
    points_to_override[is_point_on_bound_left, 0] += eps
    points_to_override[is_point_on_bound_right, 0] -= eps
    points_to_override[is_point_on_bound_lower, 1] += eps
    points_to_override[is_point_on_bound_upper, 1] -= eps

    # Only keep the points that lie inside or on the rectangle.
    is_point_in_or_on_rect = (
        is_point_in_rect
        | is_point_on_bound_left
        | is_point_on_bound_right
        | is_point_on_bound_lower
        | is_point_on_bound_upper
    )
    points_voronoi = points_to_override[is_point_in_or_on_rect]  # [P', 2]

    # Mirror the points in 4 directions.
    points_voronoi_left = points_voronoi.copy()
    points_voronoi_left[:, 0] = 2 * rect[0] - points_voronoi_left[:, 0]
    points_voronoi_right = points_voronoi.copy()
    points_voronoi_right[:, 0] = 2 * rect[2] - points_voronoi_right[:, 0]
    points_voronoi_lower = points_voronoi.copy()
    points_voronoi_lower[:, 1] = 2 * rect[1] - points_voronoi_lower[:, 1]
    points_voronoi_upper = points_voronoi.copy()
    points_voronoi_upper[:, 1] = 2 * rect[3] - points_voronoi_upper[:, 1]
    points_voronoi_all = np.concat(
        (
            points_voronoi,
            points_voronoi_left,
            points_voronoi_right,
            points_voronoi_lower,
            points_voronoi_upper,
        ),
        axis=0,
    )  # [5P', 2]

    # Compute the Voronoi diagram.
    vor = Voronoi(points_voronoi_all)

    # Filter the outputs so that only relevant parts remain.
    points = points[is_point_in_or_on_rect]  # [P', 2]

    regions = [[]]  # P' + 1
    vertex_idcs_old2new = np.full(len(vor.vertices), -1)
    vertex_idcs = []
    for pr in vor.point_region[: len(points)]:
        region = vor.regions[pr]
        for i, vertex_idx_old in enumerate(region):
            if vertex_idx_old == -1:
                raise RuntimeError(
                    "One of the center regions of the Voronoi diagram is"
                    " infinite, which should never happen. Please report this"
                    " bug to the developer."
                )

            vertex_idx_new = vertex_idcs_old2new[vertex_idx_old]
            if vertex_idx_new == -1:
                # We have not yet seen this vertex.
                vertex_idx_new = len(vertex_idcs)
                vertex_idcs_old2new[vertex_idx_old] = vertex_idx_new
                vertex_idcs.append(vertex_idx_old)

            # Update the region list in-place for efficiency.
            region[i] = vertex_idx_new

        regions.append(region)

    point_region = np.arange(1, len(points) + 1)  # [P']
    vertices = vor.vertices[vertex_idcs]  # [V + X, 2]

    # If a ridge contains one of the points with index 0 <= i < P', then it
    # should be kept. Otherwise, it should be discarded.
    is_ridge_point_selected = vor.ridge_points < len(points)
    is_ridge_selected = np.any(is_ridge_point_selected, axis=1)

    # If a ridge contains one of the points that won't be kept, then it should
    # be set to -1.
    ridge_points = vor.ridge_points
    ridge_points[~is_ridge_point_selected] = -1
    ridge_points = ridge_points[is_ridge_selected]  # [R + X, 2]

    # The ridge vertices need to be converted to the new vertex indices.
    ridge_vertices = np.array(vor.ridge_vertices)[is_ridge_selected]  # R + X
    ridge_vertices = cast(
        list[list[int]], vertex_idcs_old2new[ridge_vertices].tolist()
    )

    return (
        points,
        vertices,
        ridge_points,
        ridge_vertices,
        point_region,
        regions,
    )
