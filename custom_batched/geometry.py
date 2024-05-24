import geopandas as gpd
import torch
from shapely import Polygon

from ..modules.scipy import voronoi_constrain_to_rect


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
            (r, theta) in Hough space. Tuple containing:
            - The values of r.
                Shape: [B]
            - The values of theta.
                Shape: [B]
        lines2: The second batch of lines. Each line is represented by a pair
            (r, theta) in Hough space. Tuple containing:
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
    s1 = torch.sin(theta1)  # [B]
    s2 = torch.sin(theta2)  # [B]
    c1 = torch.cos(theta1)  # [B]
    c2 = torch.cos(theta2)  # [B]
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
            by a pair (r, theta) in Hough space. Tuple containing:
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
    n_line = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)  # [B, 2]
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
    in_polygon = torch.zeros(B, dtype=torch.bool, device=device)
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
            < (xj - xi) / (yj - yi) * (y - yi)  # swap order for efficiency
            + xi
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
    zero_point = torch.zeros(1, 2, dtype=dtype, device=device)
    point_tensors = [zero_point, exterior]
    for hole in interiors:
        point_tensors.extend([zero_point, hole])
    point_tensors.append(zero_point)
    return __is_point_in_polygon_simple_batched(
        torch.concatenate(point_tensors), points
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

    exterior = torch.tensor(
        polygon.exterior.coords, dtype=dtype, device=device
    )
    if not polygon.interiors:
        return __is_point_in_polygon_simple_batched(exterior, points)
    interiors = [
        torch.tensor(interior.coords, dtype=dtype, device=device)
        for interior in polygon.interiors
    ]
    return __is_point_in_polygon_complex_batched(exterior, interiors, points)


def cut_polygon_around_points(
    polygon: Polygon, points: torch.Tensor
) -> gpd.GeoSeries:
    """Split a polygon into multiple polygons, each corresponding to a point.

    Each coordinate (x, y) in the polygon is assigned to the point that is
    closest to it. Therefore, this problem corresponds to a Voronoi diagram
    that is clipped by the original polygon.

    Warning: This function assumes (and does not check) that all points are
    inside the polygon. If a point is outside the polygon, the function will
    perform undefined behavior. Do note, however, that points on the boundary
    or very close to it (to prevent floating point errors) are fine.

    Args:
        polygon: The polygon.
        points: The points to split the polygon around.
            Shape: [B, 2]

    Returns:
        A GeoSeries containing the split polygons.
            Shape: [B]
    """
    if len(points) == 1:
        return gpd.GeoSeries([polygon])  # type: ignore

    (_, vertices, _, _, point_region, regions) = voronoi_constrain_to_rect(
        points.cpu().numpy(), polygon.bounds
    )

    # Convert the regions to polygons.
    polygons = []
    for pr in point_region:
        polygons.append(Polygon(vertices[regions[pr]]))
    polygons = gpd.GeoSeries(polygons)

    # Perform an intersection with the original polygon to remove the parts
    # that lie outside of it.
    return polygons.intersection(polygon)  # type: ignore
