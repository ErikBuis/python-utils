from __future__ import annotations

from functools import lru_cache

import numpy as np
import numpy.typing as npt

from ..modules_batched.numpy import (
    arange_batched_packed,
    meshgrid_batched_packed,
)


def line_intersection_batched(
    lines1: tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]],
    lines2: tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]],
) -> npt.NDArray[np.floating]:
    """Calculate the intersection point of two lines.

    Warning: If the lines are parallel, the intersection point is set to
    (nan, nan).

    The intersection point is calculated by solving the following system of
    equations:
        r1 = x * cos(theta1) + y * sin(theta1)
        r2 = x * cos(theta2) + y * sin(theta2)
    This can be written as:
        [cos(theta1)  sin(theta1)] [x] = [r1]
        [cos(theta2)  sin(theta2)] [y] = [r2]
    The solution is:
        [x] = [cos(theta1)  sin(theta1)]^-1 [r1]
        [y]   [cos(theta2)  sin(theta2)]    [r2]
            = csc(theta1 - theta2) [-sin(theta2)  sin(theta1)] [r1]
                                   [ cos(theta2) -cos(theta1)] [r2]
            = csc(theta1 - theta2) [-r1 * sin(theta2) + r2 * sin(theta1)]
                                   [ r1 * cos(theta2) - r2 * cos(theta1)]

    Args:
        lines1: The first batch of lines. Each line is represented by a pair
            (r, theta) in Hough space as a tuple containing:
            - The values of r.
                Shape: [B]
            - The values of theta.
                Shape: [B]
        lines2: The second batch of lines. Each line is represented by a pair
            (r, theta) in Hough space as a tuple containing:
            - The values of r.
                Shape: [B]
            - The values of theta.
                Shape: [B]

    Returns:
        The intersection points of the lines. If the lines are parallel, the
        intersection point is set to (nan, nan).
            Shape: [B, 2]
    """
    r1, theta1 = lines1
    r2, theta2 = lines2
    s1 = np.sin(theta1)  # [B]
    s2 = np.sin(theta2)  # [B]
    c1 = np.cos(theta1)  # [B]
    c2 = np.cos(theta2)  # [B]
    csc = 1 / (c2 * s1 - c1 * s2)  # [B]
    return np.expand_dims(csc, 1) * np.stack(
        [-r1 * s2 + r2 * s1, r1 * c2 - r2 * c1], axis=1
    )  # [B, 2]


def distance_line_to_point_batched(
    lines: tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]],
    points: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """Calculate the shortest distance from a line to a point.

    The distance is calculated by solving the following system of equations:
        n_line.dot(point - v_line)
    where:
        n_line is the normal vector of the line.
        v_line is the point on the line closest to the origin.
        point is the point to calculate the distance to.

    Args:
        lines: The lines to calculate the distance to. Each line is represented
            by a pair (r, theta) in Hough space as a tuple containing:
            - The values of r.
                Shape: [B]
            - The values of theta.
                Shape: [B]
        points: The points to calculate the distance to the lines.
            Shape: [B, 2]

    Returns:
        The shortest distance from the lines to the points.
            Shape: [B]
    """
    r, theta = lines
    n_line = np.stack([np.cos(theta), np.sin(theta)], axis=1)  # [B, 2]
    v_line = np.expand_dims(r, 1) * n_line  # [B, 2]
    return np.abs((n_line * (points - v_line)).sum(axis=1))  # [B]


@lru_cache(maxsize=2)
def __get_theory_cells(
    cell_bounds: tuple[float, float, float, float],
    cell_size: tuple[float, float],
) -> tuple[int, int]:
    """Calculate the maximum number of cells per dimension.

    Args:
        cell_bounds: The total area covered by the cells, in the form of
            (min_x, min_y, max_x, max_y).
        cell_size: The size of each cell, in the form of
            (cell_size_x, cell_size_y).

    Returns:
        The maximum number of cells per dimension.
    """
    theory_cells_x = int((cell_bounds[2] - cell_bounds[0]) // cell_size[0])
    theory_cells_y = int((cell_bounds[3] - cell_bounds[1]) // cell_size[1])
    return theory_cells_x, theory_cells_y


def theory_cells_max(
    cell_bounds: tuple[float, float, float, float],
    cell_size: tuple[float, float],
) -> int:
    """Calculate the maximum number of cells that can fit in the given bounds.

    Args:
        cell_bounds: The total area covered by the cells, in the form of
            (min_x, min_y, max_x, max_y).
        cell_size: The size of each cell, in the form of
            (cell_size_x, cell_size_y).

    Returns:
        The maximum number of cells that can theoretically fit in the given
        bounds.
    """
    theory_cells_x, theory_cells_y = __get_theory_cells(cell_bounds, cell_size)
    return theory_cells_x * theory_cells_y


def compress_coords_to_ids(
    coords: npt.NDArray[np.floating],
    cell_bounds: tuple[float, float, float, float],
    cell_size: tuple[float, float],
) -> npt.NDArray[np.intp]:
    """Compress (x, y) cell coordinates to single integer IDs.

    Args:
        coords: (x, y) coordinates of the lowerleft corner of the cells to
            compress. If a coordinate is not on an exact cell corner, it will be
            floored to the nearest lowerleft cell corner automatically.
            Shape: [*, 2]
        cell_bounds: The total area covered by the cells, in the form of
            (min_x, min_y, max_x, max_y).
        cell_size: The size of each cell, in the form of
            (cell_size_x, cell_size_y).

    Returns:
        Integer IDs representing the cells uniquely
            Shape: [*]
    """
    theory_cells = __get_theory_cells(cell_bounds, cell_size)

    # Map the (x, y) coordinates to cell indices (i, j). The lowerleft corner of
    # the total cell area is considered the origin (0, 0), with each cell to the
    # right and above increasing the index by 1.
    cells = ((coords - cell_bounds[:2]) // cell_size).astype(np.intp)  # [*, 2]
    cells = np.moveaxis(cells, -1, 0)  # [2, *]

    # Compress the cell indices (i, j) to cell IDs.
    # We use C order (default) to have y vary fastest (aka 'ij' indexing). This
    # way, neighboring cells in the y direction have consecutive IDs. Example
    # for theory_cells_x = 4, theory_cells_y = 3:
    # y   ^
    #   2 | 2  5  8  11
    #   1 | 1  4  7  10
    #   0 | 0  3  6   9
    #     +-+--+--+--+--> x
    #       0  1  2  3
    cell_ids = np.ravel_multi_index(cells, theory_cells)  # [*]
    return cell_ids  # type: ignore


def decompress_coords_from_ids(
    cell_ids: npt.NDArray[np.intp],
    cell_bounds: tuple[float, float, float, float],
    cell_size: tuple[float, float],
) -> npt.NDArray[np.floating]:
    """Decompress integer IDs back to (x, y) cell coordinates.

    Args:
        cells_ids: Integer IDs representing the cells uniquely.
            Shape: [*]
        cell_bounds: The total area covered by the cells, in the form of
            (min_x, min_y, max_x, max_y).
        cell_size: The size of each cell, in the form of
            (cell_size_x, cell_size_y).

    Returns:
        (x, y) coordinates of the lowerleft corner of the decompressed cells.
            Shape: [*, 2]
    """
    theory_cells = __get_theory_cells(cell_bounds, cell_size)

    # Decompress the cell IDs back to cell indices (i, j). The lowerleft corner
    # of the total cell area is considered the origin (0, 0), with each cell to
    # the right and above increasing the index by 1.
    cells = np.array(np.unravel_index(cell_ids, theory_cells))  # [2, *]

    # Map the cell indices (i, j) back to (x, y) coordinates.
    cells = np.moveaxis(cells, 0, -1)  # [*, 2]
    coords = cells * cell_size + cell_bounds[:2]  # [*, 2]
    return coords


def __generate_cell_ids_between(
    cell_ids_lowerleft: npt.NDArray[np.intp],
    cell_ids_upperright: npt.NDArray[np.intp],
    cell_bounds: tuple[float, float, float, float],
    cell_size: tuple[float, float],
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
    """Generate all cell IDs in the rectangular area between two cell IDs.

    Warning: This function assumes that the input cell IDs correspond to
    rectangular areas aligned with the grid (lowerleft corner must always have
    X and Y coordinates less than or equal to those of the upperright corner).

    Args:
        cell_ids_lowerleft: Cell IDs of the lowerleft corners.
            Shape: [P]
        cell_ids_upperright: Cell IDs of the upperright corners.
            Shape: [P]
        cell_bounds: The total area covered by the cells, in the form of
            (min_x, min_y, max_x, max_y).
        cell_size: The size of each cell, in the form of
            (cell_size_x, cell_size_y).

    Returns:
        Tuple containing:
        - Packed array of all cell IDs in the rectangular area between the given
            corners.
            Shape: [sum(O_ps)]
        - The number of cells covered by the rectangular area for each pair of
            corners.
            Shape: [P]

    Examples:
    >>> # Assume a grid with 7 cells in the X direction and 10 cells in the Y
    >>> # direction. Graphically, if we have the following lowerleft and
    >>> # upperright corners:
    >>> #   +--------------------- (5, 5)
    >>> #   |                        |
    >>> # (2, 3) --------------------+
    >>> #          +--------------------- 5*10+5=55
    >>> #  -->     |                          |
    >>> #      2*10+3=23 ---------------------+
    >>> # And:
    >>> #   +---- (2, 8)
    >>> # (1, 7) ---+
    >>> #  -->     +---- 2*10+8=28
    >>> #      1*10+7=17 ----+
    >>> #
    >>> # Then the function should return all cell IDs in these areas, which
    >>> # should look like this:
    >>> # (2, 5), (3, 5), (4, 5), (5, 5),
    >>> # (2, 4), (3, 4), (4, 4), (5, 4),
    >>> # (2, 3), (3, 3), (4, 3), (5, 3).
    >>> #      2*10+5=25, 3*10+5=35, 4*10+5=45, 5*10+5=55,
    >>> #  --> 2*10+4=24, 3*10+4=34, 4*10+4=44, 5*10+4=54,
    >>> #      2*10+3=23, 3*10+3=33, 4*10+3=43, 5*10+3=53,
    >>> # And:
    >>> # (1, 8), (2, 8),
    >>> # (1, 7), (2, 7).
    >>> #  --> 1*10+8=18, 2*10+8=28,
    >>> #      1*10+7=17, 2*10+7=27.
    >>> #
    >>> # Now for the code:
    >>> cell_bounds = (0.0, 0.0, 7.0, 10.0)
    >>> cell_size = (1.0, 1.0)
    >>> cell_ids_lowerleft = np.array([23, 17])
    >>> cell_ids_upperright = np.array([55, 28])
    >>> cell_ids_packed, O_ps = __generate_cell_ids_between(
    ...     cell_ids_lowerleft, cell_ids_upperright, cell_bounds, cell_size
    ... )
    >>> cell_ids_packed
    array([23, 24, 25, 33, 34, 35, 43, 44, 45, 53, 54, 55, 17, 18, 27, 28])
    >>> O_ps
    array([12,  4])
    """
    theory_cells = __get_theory_cells(cell_bounds, cell_size)

    # First, we convert the cell IDs to (i, j) coordinates on the cell grid.
    i_lowerleft, j_lowerleft = np.unravel_index(
        cell_ids_lowerleft, theory_cells
    )  # [P], [P]
    i_upperright, j_upperright = np.unravel_index(
        cell_ids_upperright, theory_cells
    )  # [P], [P]
    O_ps = (i_upperright - i_lowerleft + 1) * (
        j_upperright - j_lowerleft + 1
    )  # [P]

    # Generate all (i, j) coordinates in the rectangular area.
    i_coords_packed, I_ps, _ = arange_batched_packed(
        i_lowerleft, i_upperright + 1
    )  # [sum(I_ps)], [P], int
    j_coords_packed, J_ps, _ = arange_batched_packed(
        j_lowerleft, j_upperright + 1
    )  # [sum(J_ps)], [P], int

    # Create the meshgrid of (i, j) coordinates for each rectangular area.
    i_coords_mesh_packed, j_coords_mesh_packed = meshgrid_batched_packed(
        (i_coords_packed, I_ps), (j_coords_packed, J_ps), indexing="ij"
    )  # [sum(O_ps)], [sum(O_ps)]

    # Compress the (i, j) coordinates back to cell IDs.
    cell_ids_packed = np.ravel_multi_index(
        (i_coords_mesh_packed, j_coords_mesh_packed),  # type: ignore
        theory_cells,
    )  # [sum(O_ps)]

    return cell_ids_packed, O_ps


def rects_cells_overlap(
    rects: npt.NDArray[np.floating],
    cell_bounds: tuple[float, float, float, float],
    cell_size: tuple[float, float],
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
    """Find all cells that overlap with the given rectangles.

    Note: This function returns cell IDs, not (x, y) coordinates. Use
    decompress_coords_from_ids() to convert the returned cell IDs back to the
    (x, y) coordinates of their lowerleft corners.

    Args:
        rects: Rectangles in the form of (min_x, min_y, max_x, max_y).
            Shape: [P, 4]
        cell_bounds: The total area covered by the cells, in the form of
            (min_x, min_y, max_x, max_y).
        cell_size: The size of each cell, in the form of
            (cell_size_x, cell_size_y).

    Returns:
        Tuple containing:
        - Packed array of all cell IDs that overlap with the rectangles.
            Shape: [sum(O_ps)]
        - Array with the number of cells that overlap with each rectangle.
            Shape: [P]
    """
    # Compress the cell coordinates to single ints representing cell IDs.
    rects_cell_ids_lowerleft = compress_coords_to_ids(
        rects[:, [0, 1]], cell_bounds, cell_size
    )  # [P]
    rects_cell_ids_upperright = compress_coords_to_ids(
        rects[:, [2, 3]], cell_bounds, cell_size
    )  # [P]

    # Generate all cell IDs that each rectangle overlaps with.
    cell_ids_packed, O_ps = __generate_cell_ids_between(
        rects_cell_ids_lowerleft,
        rects_cell_ids_upperright,
        cell_bounds,
        cell_size,
    )  # [sum(O_ps)], [P]

    return cell_ids_packed, O_ps
