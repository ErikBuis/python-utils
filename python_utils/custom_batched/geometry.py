import geopandas as gpd
import numpy as np
import torch
import torch.nn as nn
from shapely import GeometryCollection, MultiPolygon, Polygon

from ..modules.numpy import unique_consecutive
from ..modules.scipy import voronoi_constrain_to_rect
from ..modules.torch import cumsum_start_0
from ..modules_batched.torch import (
    arange_batched,
    pad_packed_batched,
    replace_padding_batched,
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
        points.cpu().numpy(), polygon.bounds
    )

    # Convert the regions to polygons.
    polygons = []
    for pr in point_region:
        polygons.append(Polygon(vertices[regions[pr]]))
    polygons = gpd.GeoSeries(polygons)

    # Perform an intersection with the original polygon to remove the parts
    # that lie outside of it.
    polygon_pieces = polygons.intersection(polygon)  # type: ignore

    # Ensure all pieces only consist of Polygon objects.
    for i, polygon_piece in enumerate(polygon_pieces):
        if isinstance(polygon_piece, GeometryCollection):
            polygon_pieces[i] = MultiPolygon([
                subgeometry
                for subgeometry in polygon_piece.geoms
                if isinstance(subgeometry, Polygon)
            ])

    return polygon_pieces


def polygon_exterior_vertices(
    polygon: Polygon, device: torch.device | str = "cpu"
) -> torch.Tensor:
    """Get the vertices of a polygon.

    Args:
        polygon: The Polygon object to get the vertices of.
        device: The device to use.

    Returns:
        The vertices of the polygon.
            Shape: [V, 2]
    """
    return torch.tensor(
        polygon.exterior.coords[:-1], dtype=torch.float32, device=device
    )


def multipolygon_exterior_vertices(
    polygon: MultiPolygon, device: torch.device | str = "cpu"
) -> torch.Tensor:
    """Get the vertices of a multipolygon.

    Args:
        polygon: The MultiPolygon object to get the vertices of.
        device: The device to use.

    Returns:
        The vertices of the multipolygon.
            Shape: [V, 2]
    """
    return torch.concatenate([
        torch.tensor(
            polygon_i.exterior.coords[:-1], dtype=torch.float32, device=device
        )
        for polygon_i in polygon.geoms
    ])


def polygon_like_exterior_vertices(
    polygon: Polygon | MultiPolygon, device: torch.device | str = "cpu"
) -> torch.Tensor:
    """Get the vertices of a polygon-like object.

    Args:
        polygon: The Polygon or MultiPolygon object to get the vertices of.
        device: The device to use.

    Returns:
        The vertices of the polygon.
            Shape: [V, 2]
    """
    if isinstance(polygon, Polygon):
        return polygon_exterior_vertices(polygon, device)
    else:  # polygon is a MultiPolygon
        return multipolygon_exterior_vertices(polygon, device)


def polygons_exterior_vertices(
    polygons: gpd.GeoSeries, device: torch.device = torch.device("cpu")
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get the vertices of multiple polygons.

    Note: You should use this function if your GeoSeries only contains Polygon
    objects, as it is faster than using polygon_exterior_vertices()
    sequentially, or using polygon_likes_exterior_vertices().

    Args:
        polygons: A GeoSeries of Polygon objects to get the vertices of.
            Shape: [B]
        device: The device to use.

    Returns:
        Tuple containing:
        - The vertices of the polygons, padded with zeros for heterogeneous
            batch sizes.
            Shape: [B, max(V_b), 2]
        - The number of vertices in each polygon.
            Shape: [B]
    """
    # I timed multiple different approaches against each other, among which a
    # sequential version that just called polygon_exterior_vertices()
    # repeatedly. Literally all conceivable variations of the below code were
    # timed. This turned out to be the fastest one.
    coords = polygons.exterior.get_coordinates()
    coords_tensor = torch.from_numpy(coords.to_numpy(dtype=np.float32)).to(
        device
    )
    V_bs = torch.from_numpy(
        unique_consecutive(
            coords.index.to_numpy(), axis=0, return_counts=True
        )[1]
    ).to(device)
    vertices_padded = pad_packed_batched(coords_tensor, V_bs, int(V_bs.max()))
    replace_padding_batched(
        vertices_padded, V_bs - 1, padding_value=0, in_place=True
    )
    return vertices_padded[:, :-1], V_bs - 1


def multipolygons_exterior_vertices(
    polygons: gpd.GeoSeries, device: torch.device = torch.device("cpu")
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get the vertices of multiple multipolygons.

    Note: You should use this function if your GeoSeries only contains
    MultiPolygon objects, as it is faster than using
    multipolygon_exterior_vertices() sequentially, or using
    polygon_likes_exterior_vertices().

    Args:
        polygons: A GeoSeries of MultiPolygon objects to get the vertices of.
            Shape: [B]
        device: The device to use.

    Returns:
        Tuple containing:
        - The vertices of the multipolygons, padded with zeros for
            heterogeneous batch sizes.
            Shape: [B, max(V_b), 2]
        - The number of vertices in each multipolygon.
            Shape: [B]
    """
    # Unfortunately, there is no way to vectorize this operation, as the
    # exterior vertices of each polygon in the MultiPolygon can't be requested
    # seperately from the interior vertices in a batched manner. Therefore, we
    # have to iterate over the MultiPolygons.
    vertices_list = [
        multipolygon_exterior_vertices(polygon, device) for polygon in polygons
    ]
    vertices_padded = nn.utils.rnn.pad_sequence(
        vertices_list, batch_first=True
    )
    V_bs = torch.tensor(
        [len(vertices) for vertices in vertices_list], device=device
    )
    return vertices_padded, V_bs


def polygon_likes_exterior_vertices(
    polygons: gpd.GeoSeries, device: torch.device = torch.device("cpu")
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get the vertices of multiple polygon-like objects.

    Note: You should use this function if your GeoSeries contains both Polygon
    and MultiPolygon objects, as it is faster than using
    polygon_exterior_vertices() and multipolygon_exterior_vertices()
    sequentially.

    Args:
        polygons: A GeoSeries of Polygon or MultiPolygon objects to get the
            vertices of.
            Shape: [B]
        device: The device to use.

    Returns:
        Tuple containing:
        - The vertices of the polygon-like objects, padded with zeros for
            heterogeneous batch sizes.
            Shape: [B, max(V_b), 2]
        - The number of vertices in each polygon-like object.
            Shape: [B]
    """
    # I timed multiple different approaches against each other, among which a
    # sequential version that just called polygon_like_exterior_vertices()
    # repeatedly. Literally all conceivable variations of the below code were
    # timed. This turned out to be the fastest one.

    # Perform some preparatory operations for the Polygon objects.
    coords = polygons.exterior.get_coordinates()
    if len(coords) != 0:
        coords_tensor = torch.from_numpy(coords.to_numpy(dtype=np.float32)).to(
            device
        )
        V_bs_polygons = torch.from_numpy(
            unique_consecutive(
                coords.index.to_numpy(), axis=0, return_counts=True
            )[1]
        ).to(device)
        V_bs_cumsum = cumsum_start_0(V_bs_polygons, dim=0)

    # Retrieve the exterior coordinates depending on the type of the polygon.
    i = 0
    vertices_list = [
        (
            coords_tensor[  # type: ignore
                V_bs_cumsum[i]  # type: ignore
                : V_bs_cumsum[i := i + 1] - 1  # noqa: F841  # type: ignore
            ]  # fmt: skip
            if isinstance(polygon, Polygon)
            else multipolygon_exterior_vertices(polygon, device)
        )
        for polygon in polygons
    ]

    # Merge the results into a single tensor.
    vertices_padded = nn.utils.rnn.pad_sequence(
        vertices_list, batch_first=True
    )
    V_bs = torch.tensor(
        [len(vertices) for vertices in vertices_list], device=device
    )

    return vertices_padded, V_bs


def polygon_likes_exterior_vertices_naive(
    polygons: gpd.GeoSeries, device: torch.device = torch.device("cpu")
) -> tuple[torch.Tensor, torch.Tensor]:
    vertices_list = []
    for polygon in polygons:
        vertices_list.append(polygon_like_exterior_vertices(polygon, device))
    vertices_padded = nn.utils.rnn.pad_sequence(
        vertices_list, batch_first=True
    )
    V_bs = torch.tensor(
        [len(vertices) for vertices in vertices_list], device=device
    )
    return vertices_padded, V_bs


def xiaolin_wu_anti_aliasing_batched(
    x0: torch.Tensor, y0: torch.Tensor, x1: torch.Tensor, y1: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Xiaolin Wu's line algorithm for drawing anti-aliased lines.

    Args:
        x0: X-coordinate of the first endpoint of the line segment.
            Shape: [B]
        y1: Y-coordinate of the first endpoint of the line segment.
            Shape: [B]
        x1: X-coordinate of the second endpoint of the line segment.
            Shape: [B]
        y1: Y-coordinate of the second endpoint of the line segment.
            Shape: [B]

    Returns:
        Tuple containing:
        - Pixel x-coordinates, padded with zeros.
            Shape: [B, max(S_b)]
        - Pixel y-coordinates, padded with zeros.
            Shape: [B, max(S_b)]
        - Pixel values between 0 and 1, padded with zeros.
            Shape: [B, max(S_b)]
        - The number of pixels in each line segment.
            Shape: [B]
    """
    steep = torch.abs(y1 - y0) > torch.abs(x1 - x0)  # [B]

    # Swap the x and y coordinates to ensure the line is not steep.
    x0, y0, x1, y1 = torch.where(
        steep, torch.stack([y0, x0, y1, x1]), torch.stack([x0, y0, x1, y1])
    )

    # Swap the start and end to ensure the line goes from left to right.
    x0, y0, x1, y1 = torch.where(
        x0 > x1, torch.stack([x1, y1, x0, y0]), torch.stack([x0, y0, x1, y1])
    )

    # Calculate the gradient of the line segments.
    dx, dy = x1 - x0, y1 - y0  # [B], [B]
    gradient = torch.where(dx != 0, dy / dx, 1)  # [B]

    # Pre-process the beginning of the line segments.
    xpxl_begin = x0.round().long()  # [B]
    xgap_begin = 1 - (x0 + 0.5 - xpxl_begin)  # [B]

    # Pre-process the end of the line segments.
    xpxl_end = x1.round().long()  # [B]
    xgap_end = x1 + 0.5 - xpxl_end  # [B]

    # Initialize the return values.
    S_bs = 2 * (xpxl_end - xpxl_begin + 1)  # [B]
    max_S_b = int(S_bs.max())
    B = len(S_bs)
    pixels_x = torch.empty((B, max_S_b), dtype=torch.int64)
    pixels_y = torch.empty((B, max_S_b), dtype=torch.int64)
    vals = torch.empty((B, max_S_b), dtype=torch.float64)

    # Calculate values used in the main loop.
    x, _ = arange_batched(
        xpxl_begin, xpxl_end + 1, dtype=torch.int64
    )  # [B, max(S_b) // 2]
    intery = y0.unsqueeze(1) + gradient.unsqueeze(1) * (
        x.double() - x0.unsqueeze(1)
    )  # [B, max(S_b) // 2]
    ipart_intery = intery.floor().long()  # [B, max(S_b) // 2]
    fpart_intery = intery - ipart_intery  # [B, max(S_b) // 2]
    rfpart_intery = 1 - fpart_intery  # [B, max(S_b) // 2]

    # Fill the return values.
    pixels_x[:, ::2] = torch.where(steep.unsqueeze(1), ipart_intery, x)
    pixels_y[:, ::2] = torch.where(steep.unsqueeze(1), x, ipart_intery)
    pixels_x[:, 1::2] = torch.where(steep.unsqueeze(1), ipart_intery + 1, x)
    pixels_y[:, 1::2] = torch.where(steep.unsqueeze(1), x, ipart_intery + 1)
    vals[:, ::2] = rfpart_intery
    vals[:, 1::2] = fpart_intery

    # Handle the beginning and end of the line segments.
    vals[:, :2] *= xgap_begin.unsqueeze(1)
    vals[torch.arange(B), S_bs - 2] *= xgap_end
    vals[torch.arange(B), S_bs - 1] *= xgap_end

    # Pad the return values.
    replace_padding_batched(pixels_x, S_bs, in_place=True)
    replace_padding_batched(pixels_y, S_bs, in_place=True)
    replace_padding_batched(vals, S_bs, in_place=True)

    return pixels_x, pixels_y, vals, S_bs
