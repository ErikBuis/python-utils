from __future__ import annotations

from functools import lru_cache

import geopandas as gpd
import numpy as np
import numpy.typing as npt
import torch
from shapely import GeometryCollection, MultiPolygon, Polygon

from ..modules.scipy import voronoi_constrain_to_rect
from ..modules_batched.numpy import (
    arange_batched_packed,
    meshgrid_batched_packed,
)


def line_intersection_batched(
    lines1: tuple[torch.Tensor, torch.Tensor],
    lines2: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
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
    s1 = theta1.sin()  # [B]
    s2 = theta2.sin()  # [B]
    c1 = theta1.cos()  # [B]
    c2 = theta2.cos()  # [B]
    csc = 1 / (c2 * s1 - c1 * s2)  # [B]
    return csc.unsqueeze(1) * torch.stack(
        [-r1 * s2 + r2 * s1, r1 * c2 - r2 * c1], dim=1
    )  # [B, 2]


def distance_line_to_point_batched(
    lines: tuple[torch.Tensor, torch.Tensor], points: torch.Tensor
) -> torch.Tensor:
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
    n_line = torch.stack([theta.cos(), theta.sin()], dim=1)  # [B, 2]
    v_line = r.unsqueeze(1) * n_line  # [B, 2]
    return (n_line * (points - v_line)).sum(dim=1).abs()  # [B]


def is_point_in_bbox_batched(
    bboxes: torch.Tensor, points: torch.Tensor
) -> torch.Tensor:
    """Check if points are in bounding boxes.

    Args:
        bboxes: The bounding box(es). Each bounding box is represented by
            the coordinates of its bottom left and top right corners.
            Shape: [4] or [B, 4]
        points: The point(s) to be checked for intersection with the bounding
            box(es).
            Shape: [2] or [B, 2]

    Returns:
        A tensor of booleans. True if the corresponding point is in the
        bounding box, False otherwise.
            Shape: [1] or [B]
    """
    if bboxes.ndim == 1:
        bboxes = bboxes.unsqueeze(0)
    if points.ndim == 1:
        points = points.unsqueeze(0)

    return (
        (bboxes[:, 0] <= points[:, 0])
        & (points[:, 0] < bboxes[:, 2])
        & (bboxes[:, 1] <= points[:, 1])
        & (points[:, 1] < bboxes[:, 3])
    )


def __is_point_in_polygon_simple_batched(
    exterior: torch.Tensor, points: torch.Tensor
) -> torch.Tensor:
    """Check if points are in a polygon without holes.

    Args:
        exterior: The vertices on the outside of the polygon. The first vertex
            must be repeated at the end to close the polygon.
            Shape: [V, 2]
        points: The points to be checked for intersection with the polygon.
            Shape: [B, 2]

    Returns:
        A tensor of booleans. True if the point is in the polygon, False
        otherwise.
            Shape: [B]
    """
    device = points.device
    B = len(points)
    V = len(exterior)
    x, y = points.unbind(dim=1)

    # This algorithm is based on the PNPOLY algorithm by W. Randolph Franklin.
    # It is described here:
    # https://wrfranklin.org/Research/Short_Notes/pnpoly.html
    # In short, it works by drawing a horizontal ray from the point to the
    # right and counting the number of times the ray intersects with an edge
    # of the polygon. At each intersection, the ray switches between being
    # inside and outside the polygon. This is called the Jordan curve theorem.
    in_polygon = torch.zeros(B, device=device, dtype=torch.bool)
    # Iterate over the edges of the polygon...
    for i, j in zip(range(0, V - 1), range(1, V)):
        xi, yi = exterior[i]
        xj, yj = exterior[j]
        # For each edge between vertex i and vertex j:
        # If one vertex is above and one is below the point, and the point is
        # to the left of the intersection, then the ray intersects the edge.
        in_polygon ^= (
            # Why is the below formula correct?
            # A != B is essentially an XOR operation. In this case, it is used
            # to check if vertex i and vertex j are on different sides of the
            # point.
            (yi > y) != (yj > y)  # fmt: skip
        ) & (
            # Why is the below formula correct?
            # Formula for edge:
            #     yi = m * xi + b      m = (yj - yi) / (xj - xi)
            #     yj = m * xj + b  =>  b = yi - m * xi
            # Intersection with horizontal ray at y is at:
            #     y = m * x + b
            #     x = (y - b) / m
            #       = (y - yi + m * xi) / m
            #       = (y - yi) / m + xi
            #       = (y - yi) * (xj - xi) / (yj - yi) + xi
            x
            < (xj - xi) / (yj - yi) * (y - yi) + xi  # swap order for efficiency
        )
    return in_polygon


def __is_point_in_polygon_complex_batched(
    exterior: torch.Tensor, interiors: list[torch.Tensor], points: torch.Tensor
) -> torch.Tensor:
    """Check if points are in a polygon with holes.

    Args:
        exterior: The vertices on the outside of the polygon. The first vertex
            must be repeated at the end to close the polygon.
            Shape: [V, 2]
        interiors: A list of holes in the polygon as tensors of vertices.
            The first vertex must be repeated at the end to close each hole.
            Length: H
            Inner shape: [V_h, 2]
        points: The points to be checked for intersection with the polygon.
            Shape: [B, 2]

    Returns:
        A tensor of booleans. True if the point is in the polygon, False
        otherwise.
            Shape: [B]
    """
    dtype = points.dtype
    device = points.device

    # This algorithm is based on the PNPOLY algorithm by W. Randolph Franklin.
    # It is described here:
    # https://wrfranklin.org/Research/Short_Notes/pnpoly.html
    # In short, it works by adding a (0, 0) point between the outside and each
    # hole's points, on the start and end of the sequence, and by appending the
    # first point of each sequence to the end of the sequence. This is
    # described in the section "Concave Components, Multiple Components, and
    # Holes" of the above link.
    zero_point = torch.zeros((1, 2), device=device, dtype=dtype)
    point_tensors = [zero_point, exterior]
    for hole in interiors:
        point_tensors.extend([zero_point, hole])
    point_tensors.append(zero_point)
    return __is_point_in_polygon_simple_batched(
        torch.concat(point_tensors), points
    )


def is_point_in_polygon_batched(
    polygon: Polygon, points: torch.Tensor
) -> torch.Tensor:
    """Check if points are in a polygon.

    Args:
        polygon: The polygon as a Shapely Polygon object.
        points: The point(s) to be checked for intersection with the polygon.
            Shape: [2] or [B, 2]

    Returns:
        A tensor of booleans. True if the point is in the polygon, False
        otherwise.
            Shape: [B]
    """
    dtype = points.dtype
    device = points.device

    if points.ndim == 1:
        points = points.unsqueeze(0)

    exterior = torch.as_tensor(
        polygon.exterior.coords, device=device, dtype=dtype
    )
    if not polygon.interiors:
        return __is_point_in_polygon_simple_batched(exterior, points)
    interiors = [
        torch.as_tensor(interior.coords, device=device, dtype=dtype)
        for interior in polygon.interiors
    ]
    return __is_point_in_polygon_complex_batched(exterior, interiors, points)


def is_point_in_polygon_like_batched(
    polygon_like: Polygon | MultiPolygon, points: torch.Tensor
) -> torch.Tensor:
    """Check if points are in a polygon or a multipolygon.

    Args:
        polygon_like: The polygon as a Shapely Polygon or MultiPolygon object.
        points: The point(s) to be checked for intersection with the polygon or
            multipolygon.
            Shape: [2] or [B, 2]

    Returns:
        A tensor of booleans. True if the point is in the polygon or
        multipolygon, False otherwise.
            Shape: [B]
    """
    device = points.device

    if isinstance(polygon_like, Polygon):
        return is_point_in_polygon_batched(polygon_like, points)
    in_polygon = torch.zeros(len(points), device=device, dtype=torch.bool)
    for subpolygon in polygon_like.geoms:
        in_polygon |= is_point_in_polygon_batched(subpolygon, points)

    return in_polygon


def cut_polygon_around_points(
    polygon: Polygon, points: torch.Tensor
) -> gpd.GeoSeries:
    """Split a polygon into multiple polygons, each corresponding to a point.

    Each coordinate (x, y) in the polygon is assigned to the point that is
    closest to it. Therefore, this problem corresponds to a Voronoi diagram
    that is clipped by the original polygon.

    Warning: This function assumes (and does not check) that all points are
    inside the polygon. If a point is outside the polygon, undefined behavior
    can occur. Do note, however, that points on the boundary or very close to
    it (to prevent floating point errors) are fine.

    Args:
        polygon: The polygon.
        points: The points to split the polygon around.
            Shape: [B, 2]

    Returns:
        A GeoSeries containing the split polygons as Polygon or MultiPolygon
        objects.
            Shape: [B]
    """
    if len(points) == 1:
        return gpd.GeoSeries([polygon])

    _, vertices, _, _, point_region, regions = voronoi_constrain_to_rect(
        points.numpy(force=True), polygon.bounds
    )

    # Convert the regions to polygons.
    polygons = []
    for pr in point_region:
        polygons.append(Polygon(vertices[regions[pr]]))
    polygons = gpd.GeoSeries(polygons)

    # Perform an intersection with the original polygon to remove the parts
    # that lie outside of it.
    polygon_pieces = polygons.intersection(polygon)

    # Ensure all pieces only consist of Polygon objects.
    for i, polygon_piece in enumerate(polygon_pieces):
        if isinstance(polygon_piece, GeometryCollection):
            polygon_pieces[i] = MultiPolygon([
                subgeometry
                for subgeometry in polygon_piece.geoms
                if isinstance(subgeometry, Polygon)
            ])

    return polygon_pieces


@lru_cache(maxsize=2)
def __get_theory_cells(
    cell_bounds: tuple[float, float, float, float],
    cell_size: tuple[float, float],
) -> tuple[
    npt.NDArray[np.intp], npt.NDArray[np.float64], npt.NDArray[np.float64]
]:
    """Calculate the maximum number of cells per dimension.

    Args:
        cell_bounds: The total area covered by the cells, in the form of
            (min_x, min_y, max_x, max_y).
        cell_size: The size of each cell, in the form of
            (cell_size_x, cell_size_y).

    Returns:
        Tuple containing:
        - The maximum number of cells per dimension.
            Shape: [2]
        - The cell bounds as a numpy array.
            Shape: [4]
        - The cell size as a numpy array.
            Shape: [2]
    """
    cell_bounds_np = np.array(cell_bounds)  # [4]
    cell_size_np = np.array(cell_size)
    theory_cells = (
        (cell_bounds_np[[2, 3]] - cell_bounds_np[[0, 1]]) // cell_size_np
    ).astype(np.intp)  # [2]  # fmt: skip
    return theory_cells, cell_bounds_np, cell_size_np


def theory_cells_max(
    cell_bounds: tuple[float, float, float, float],
    cell_size: tuple[float, float],
) -> np.int64:
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
    theory_cells, _, _ = __get_theory_cells(cell_bounds, cell_size)
    return np.prod(theory_cells)


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
    theory_cells, cell_bounds_np, cell_size_np = __get_theory_cells(
        cell_bounds, cell_size
    )

    # Map the (x, y) coordinates to cell indices (i, j). The lowerleft corner of
    # the total cell area is considered the origin (0, 0), with each cell to the
    # right and above increasing the index by 1.
    cells = (
        (coords - cell_bounds_np[[0, 1]]) // cell_size_np
    ).astype(np.intp)  # [*, 2]  # fmt: skip
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
    theory_cells, cell_bounds_np, cell_size_np = __get_theory_cells(
        cell_bounds, cell_size
    )

    # Decompress the cell IDs back to cell indices (i, j). The lowerleft corner
    # of the total cell area is considered the origin (0, 0), with each cell to
    # the right and above increasing the index by 1.
    cells = np.array(np.unravel_index(cell_ids, theory_cells))  # [2, *]

    # Map the cell indices (i, j) back to (x, y) coordinates.
    cells = np.moveaxis(cells, 0, -1)
    coords = cells * cell_size_np + cell_bounds_np[[0, 1]]  # [*, 2]
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
    >>> cell_ids_lowerleft = np.array([23, 17])
    >>> cell_ids_upperright = np.array([55, 28])
    >>> cell_ids_packed, O_ps = __generate_cell_ids_between(
    ...     cell_ids_lowerleft, cell_ids_upperright
    ... )
    >>> cell_ids_packed
    array([23, 24, 25, 33, 34, 35, 43, 44, 45, 53, 54, 55, 17, 18, 27, 28])
    >>> O_ps
    array([12,  4])
    """
    theory_cells, _, _ = __get_theory_cells(cell_bounds, cell_size)

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
