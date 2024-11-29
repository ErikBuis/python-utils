from typing import NamedTuple, cast

import geopandas as gpd
import numpy as np
import pandas as pd
import torch
from shapely import LinearRing, MultiPolygon, Polygon

from ..modules.torch import unique_consecutive
from ..modules_batched.torch import pad_packed_batched


LinearRingVertices = NamedTuple(
    "LinearRingVertices", [("vertices", torch.Tensor)]
)
"""The vertices of a LinearRing object.

Attributes:
- vertices: The vertices of the linear ring. The first vertex is repeated at
    the end to close the ring.
    Shape: [V, 2]
"""

LinearRingsVertices = NamedTuple(
    "LinearRingsVertices", [("vertices", torch.Tensor), ("V_bs", torch.Tensor)]
)
"""The vertices of a batch of LinearRing objects.

Attributes:
- vertices: The vertices of the linear rings. The first vertex is repeated at
    the end to close the ring. Padded with zeros.
    Shape: [B, max(V_bs), 2]
- V_bs: The number of vertices in each linear ring.
    Shape: [B]
"""

PolygonExterior = NamedTuple("PolygonExterior", [("vertices", torch.Tensor)])
"""The exterior of a Polygon object.

Attributes:
- vertices: The exterior vertices of the polygon. The first vertex is repeated
    at the end to close the polygon.
    Shape: [V, 2]
"""

PolygonsExterior = NamedTuple(
    "PolygonsExterior", [("vertices", torch.Tensor), ("V_bs", torch.Tensor)]
)
"""The exterior of a batch of Polygon objects.

Attributes:
- vertices: The exterior vertices of the polygons. The first vertex is repeated
    at the end to close the polygon. Padded with zeros.
    Shape: [B, max(V_bs), 2]
- V_bs: The number of vertices in each polygon.
    Shape: [B]
"""

PolygonInteriors = NamedTuple(
    "PolygonInteriors", [("vertices", torch.Tensor), ("V_is", torch.Tensor)]
)
"""The interiors of a Polygon object.

Attributes:
- vertices: The interior vertices of the polygon. The first vertex is repeated
    at the end to close the polygon. Padded with zeros.
    Shape: [I, max(V_is), 2]
- V_is: The number of vertices in each interior.
    Shape: [I]
"""

PolygonsInteriors = NamedTuple(
    "PolygonsInteriors",
    [
        ("vertices", torch.Tensor),
        ("I_bs", torch.Tensor),
        ("V_is", torch.Tensor),
    ],
)
"""The interiors of a batch of Polygon objects.

Attributes:
- vertices: The interior vertices of the polygons. The first vertex is
    repeated at the end to close the polygon. Padded with zeros.
    Shape: [B, max(I_bs), max(V_is), 2]
- I_bs: The number of interiors in each polygon.
    Shape: [B]
- V_is: The number of vertices in each interior. Padded with zeros.
    Shape: [B, max(I_bs)]
"""

PolygonVertices = NamedTuple(
    "PolygonVertices",
    [("exterior", PolygonExterior), ("interiors", PolygonInteriors)],
)
"""The vertices of a Polygon object.

Attributes:
- exterior: The exterior vertices of the polygon as a PolygonExterior object.
- interiors: The interior vertices of the polygon as a PolygonInteriors object.
"""

PolygonsVertices = NamedTuple(
    "PolygonsVertices",
    [("exterior", PolygonsExterior), ("interiors", PolygonsInteriors)],
)
"""The vertices of a batch of Polygon objects.

Attributes:
- exterior: The exterior vertices of the polygons as a PolygonsExterior object.
- interiors: The interior vertices of the polygons as a PolygonsInteriors
    object.
"""

MultiPolygonExterior = NamedTuple(
    "MultiPolygonExterior",
    [("vertices", torch.Tensor), ("V_ps", torch.Tensor)],
)
"""The exterior of a MultiPolygon object.

Attributes:
- vertices: The exterior vertices of the multipolygon. The first vertex is
    repeated at the end to close the polygon. Padded with zeros.
    Shape: [P, max(V_ps), 2]
- V_ps: The number of vertices in each polygon.
    Shape: [P]
"""

MultiPolygonsExterior = NamedTuple(
    "MultiPolygonsExterior",
    [
        ("vertices", torch.Tensor),
        ("P_bs", torch.Tensor),
        ("V_ps", torch.Tensor),
    ],
)
"""The exterior of a batch of MultiPolygon objects.

Attributes:
- vertices: The exterior vertices of the multipolygons. The first vertex is
    repeated at the end to close the polygon. Padded with zeros.
    Shape: [B, max(P_bs), max(V_ps), 2]
- P_bs: The number of polygons in each multipolygon.
    Shape: [B]
- V_ps: The number of vertices in each polygon. Padded with zeros.
    Shape: [B, max(P_bs)]
"""

MultiPolygonInteriors = NamedTuple(
    "MultiPolygonInteriors",
    [
        ("vertices", torch.Tensor),
        ("I_ps", torch.Tensor),
        ("V_is", torch.Tensor),
    ],
)
"""The interiors of a MultiPolygon object.

Attributes:
- vertices: The interior vertices of the multipolygon. The first vertex is
    repeated at the end to close the polygon. Padded with zeros.
    Shape: [P, max(I_ps), max(V_is), 2]
- I_ps: The number of interiors in each polygon.
    Shape: [P]
- V_is: The number of vertices in each interior. Padded with zeros.
    Shape: [P, max(I_ps)]
"""

MultiPolygonsInteriors = NamedTuple(
    "MultiPolygonsInteriors",
    [
        ("vertices", torch.Tensor),
        ("P_bs", torch.Tensor),
        ("I_ps", torch.Tensor),
        ("V_is", torch.Tensor),
    ],
)
"""The interiors of a batch of MultiPolygon objects.

Attributes:
- vertices: The interior vertices of the multipolygons. The first vertex is
    repeated at the end to close the polygon. Padded with zeros.
    Shape: [B, max(P_bs), max(I_ps), max(V_is), 2]
- P_bs: The number of polygons in each multipolygon.
    Shape: [B]
- I_ps: The number of interiors in each polygon. Padded with zeros.
    Shape: [B, max(P_bs)]
- V_is: The number of vertices in each interior. Padded with zeros.
    Shape: [B, max(P_bs), max(I_ps)]
"""

MultiPolygonVertices = NamedTuple(
    "MultiPolygonVertices",
    [
        ("exteriors", MultiPolygonExterior),
        ("interiors", MultiPolygonInteriors),
    ],
)
"""The vertices of a MultiPolygon object.

Attributes:
- exteriors: The exterior vertices of the multipolygon as a
    MultiPolygonExterior object.
- interiors: The interior vertices of the multipolygon as a
    MultiPolygonInteriors object.
"""

MultiPolygonsVertices = NamedTuple(
    "MultiPolygonsVertices",
    [
        ("exteriors", MultiPolygonsExterior),
        ("interiors", MultiPolygonsInteriors),
    ],
)
"""The vertices of a batch of MultiPolygon objects.

Attributes:
- exteriors: The exterior vertices of the multipolygons as a
    MultiPolygonsExterior object.
- interiors: The interior vertices of the multipolygons as a
    MultiPolygonsInteriors object.
"""

PolygonLikeExterior = NamedTuple(
    "PolygonLikeExterior", [("vertices", torch.Tensor), ("V_ps", torch.Tensor)]
)
"""The exterior of a Polygon or MultiPolygon object.

Attributes:
- vertices: The exterior vertices of the polygon or multipolygon. The first
    vertex is repeated at the end to close the polygon. Padded with zeros.
    Shape: [P, max(V_ps), 2]
- V_ps: The number of vertices in each polygon.
    Shape: [P]
"""

PolygonLikesExterior = NamedTuple(
    "PolygonLikeExteriors",
    [
        ("vertices", torch.Tensor),
        ("P_bs", torch.Tensor),
        ("V_ps", torch.Tensor),
    ],
)
"""The exterior of a batch of Polygon or MultiPolygon objects.

Attributes:
- vertices: The exterior vertices of the polygons or multipolygons. The first
    vertex is repeated at the end to close the polygon. Padded with zeros.
    Shape: [B, max(P_bs), max(V_ps), 2]
- P_bs: The number of polygons in each polygon or multipolygon.
    Shape: [B]
- V_ps: The number of vertices in each polygon. Padded with zeros.
    Shape: [B, max(P_bs)]
"""

PolygonLikeInteriors = NamedTuple(
    "PolygonLikeInteriors",
    [
        ("vertices", torch.Tensor),
        ("I_ps", torch.Tensor),
        ("V_is", torch.Tensor),
    ],
)
"""The interiors of a Polygon or MultiPolygon object.

Attributes:
- vertices: The interior vertices of the polygon or multipolygon. The first
    vertex is repeated at the end to close the polygon. Padded with zeros.
    Shape: [P, max(I_ps), max(V_is), 2]
- I_ps: The number of interiors in each polygon.
    Shape: [P]
- V_is: The number of vertices in each interior. Padded with zeros.
    Shape: [P, max(I_ps)]
"""

PolygonLikesInteriors = NamedTuple(
    "PolygonLikesInteriors",
    [
        ("vertices", torch.Tensor),
        ("P_bs", torch.Tensor),
        ("I_ps", torch.Tensor),
        ("V_is", torch.Tensor),
    ],
)
"""The interiors of a batch of Polygon or MultiPolygon objects.

Attributes:
- vertices: The interior vertices of the polygons or multipolygons. The first
    vertex is repeated at the end to close the polygon. Padded with zeros.
    Shape: [B, max(P_bs), max(I_ps), max(V_is), 2]
- P_bs: The number of polygons in each polygon or multipolygon.
    Shape: [B]
- I_ps: The number of interiors in each polygon. Padded with zeros.
    Shape: [B, max(P_bs)]
- V_is: The number of vertices in each interior. Padded with zeros.
    Shape: [B, max(P_bs), max(I_ps)]
"""

PolygonLikeVertices = NamedTuple(
    "PolygonLikeVertices",
    [("exterior", PolygonLikeExterior), ("interiors", PolygonLikeInteriors)],
)
"""The vertices of a Polygon or MultiPolygon object.

Attributes:
- exterior: The exterior vertices of the polygon or multipolygon as a
    PolygonLikeExterior object.
- interiors: The interior vertices of the polygon or multipolygon as a
    PolygonLikeInteriors object.
"""

PolygonLikesVertices = NamedTuple(
    "PolygonLikesVertices",
    [("exterior", PolygonLikesExterior), ("interiors", PolygonLikesInteriors)],
)
"""The vertices of a batch of Polygon or MultiPolygon objects.

Attributes:
- exterior: The exterior vertices of the polygons or multipolygons as a
    PolygonLikesExterior object.
- interiors: The interior vertices of the polygons or multipolygons as a
    PolygonLikesInteriors object.
"""


# ########### CONVERT BETWEEN SHAPELY/GEOPANDAS OBJECTS AND TENSORS ###########


def __count_freqs_until(
    obj_with_index: pd.Series | pd.DataFrame,
    high: int,
    device: torch.device | str,
) -> torch.Tensor:
    """Count the frequency of each index value with in range(0, high).

    This differs from torch.unique_consecutive() in that it also counts the
    frequency of elements not present in x (i.e. with a frequency of 0).

    Warning: This function assumes that x is sorted.

    Warning: If x contains values outside the range(0, high), the function will
    crash with an index out of bounds error.

    Args:
        obj_with_index: The pd.Series or pd.DataFrame for which to count the
            frequency of each element in its index.
            Shape: [N]
        high: The value until which to count the frequency of each element.
        device: The device to use.

    Returns:
        The frequency of each element in x in range(0, high).
            Shape: [high]
    """
    index = torch.from_numpy(obj_with_index.index.to_numpy()).to(device=device)
    freqs = torch.zeros(high, dtype=torch.int64, device=device)
    unique, counts = unique_consecutive(index, return_counts=True, dim=0)
    freqs[unique] = counts
    return freqs


def LinearRing2LinearRingVertices(
    linearring: LinearRing, device: torch.device | str = "cpu"
) -> LinearRingVertices:
    """Convert a LinearRing object to a LinearRingVertices object.

    Args:
        linearring: The LinearRing object.
        device: The device to use.

    Returns:
        The vertices of the LinearRing object as a LinearRingVertices object.
    """
    vertices = torch.tensor(
        linearring.coords, dtype=torch.float32, device=device
    )  # [V, 2]
    return LinearRingVertices(vertices)


def LinearRings2LinearRingsVertices(
    linearrings: gpd.GeoSeries, device: torch.device | str = "cpu"
) -> LinearRingsVertices:
    """Convert a batch of LinearRing objects to a LinearRingsVertices object.

    Args:
        linearrings: The GeoSeries of LinearRing objects. Can also contain
            NaN/None values, these will be handled correctly by inserting
            an empty row into the output.
        device: The device to use.

    Returns:
        The vertices of a batch of the LinearRing objects as a
        LinearRingsVertices object.
    """
    # I timed multiple different approaches against each other, so almost all
    # conceivable variations of the below code were compared. This one turned
    # out to be the fastest.
    vertices_df = linearrings.get_coordinates()  # [sum(V_bs), 2]
    vertices_packed = torch.from_numpy(
        vertices_df.to_numpy(dtype=np.float32)
    ).to(device=device)  # [sum(V_bs), 2]  # fmt: skip
    V_bs = __count_freqs_until(vertices_df, len(linearrings), device)  # [B]
    vertices = pad_packed_batched(
        vertices_packed, V_bs, int(V_bs.max())
    )  # [B, max(V_bs), 2]
    return LinearRingsVertices(vertices, V_bs)


def Polygon2PolygonExterior(
    polygon: Polygon, device: torch.device | str = "cpu"
) -> PolygonExterior:
    """Convert a Polygon object to a PolygonExterior object.

    Args:
        polygon: The Polygon object.
        device: The device to use.

    Returns:
        The exterior of the Polygon as a PolygonExterior object.
    """
    (exterior,) = LinearRing2LinearRingVertices(
        polygon.exterior, device
    )  # [V, 2]
    return PolygonExterior(exterior)


def Polygons2PolygonsExterior(
    polygons: gpd.GeoSeries, device: torch.device | str = "cpu"
) -> PolygonsExterior:
    """Convert a batch of Polygon objects to a PolygonsExterior object.

    Args:
        polygons: The GeoSeries of Polygon objects.
        device: The device to use.

    Returns:
        The exteriors of the batch of Polygon objects as a PolygonsExterior
        object.
    """
    exterior, V_bs = LinearRings2LinearRingsVertices(
        polygons.exterior, device
    )  # [B, max(V_bs), 2], [B]
    return PolygonsExterior(exterior, V_bs)


def Polygon2PolygonInteriors(
    polygon: Polygon, device: torch.device | str = "cpu"
) -> PolygonInteriors:
    """Convert a Polygon object to a PolygonInteriors object.

    Args:
        polygon: The Polygon object.
        device: The device to use.

    Returns:
        The interiors of the Polygon as a PolygonInteriors object.
    """
    interiors, V_is = LinearRings2LinearRingsVertices(
        gpd.GeoSeries(polygon.interiors), device
    )  # [I, max(V_is), 2], [I]
    return PolygonInteriors(interiors, V_is)


def Polygons2PolygonsInteriors(
    polygons: gpd.GeoSeries, device: torch.device | str = "cpu"
) -> PolygonsInteriors:
    """Convert a batch of Polygon objects to a PolygonsInteriors object.

    Args:
        polygons: The GeoSeries of Polygon objects.
        device: The device to use.

    Returns:
        The interiors of the batch of Polygon objects as a PolygonsInteriors
        object.
    """
    interiors_series = (
        cast(pd.Series, polygons.interiors).explode().dropna()
    )  # [sum(I_bs)]
    I_bs = __count_freqs_until(interiors_series, len(polygons), device)  # [B]
    interiors_packed, V_is_packed = LinearRings2LinearRingsVertices(
        gpd.GeoSeries(interiors_series.reset_index(drop=True)), device
    )  # [sum(I_bs), max(V_is), 2], [sum(I_bs)]
    I_bs_max = int(I_bs.max())
    interiors = pad_packed_batched(
        interiors_packed, I_bs, I_bs_max
    )  # [B, max(I_bs), max(V_is), 2]
    V_is = pad_packed_batched(V_is_packed, I_bs, I_bs_max)  # [B, max(I_bs)]
    return PolygonsInteriors(interiors, I_bs, V_is)


def Polygon2PolygonVertices(
    polygon: Polygon, device: torch.device | str = "cpu"
) -> PolygonVertices:
    """Convert a Polygon object to a PolygonVertices object.

    Args:
        polygon: The Polygon object.
        device: The device to use.

    Returns:
        The vertices of the Polygon as a PolygonVertices object.
    """
    return PolygonVertices(
        Polygon2PolygonExterior(polygon, device),
        Polygon2PolygonInteriors(polygon, device),
    )


def Polygons2PolygonsVertices(
    polygons: gpd.GeoSeries, device: torch.device | str = "cpu"
) -> PolygonsVertices:
    """Convert a batch of Polygon objects to a PolygonsVertices object.

    Args:
        polygons: The GeoSeries of Polygon objects.
        device: The device to use.

    Returns:
        The vertices of the batch of Polygon objects as a PolygonsVertices
        object.
    """
    return PolygonsVertices(
        Polygons2PolygonsExterior(polygons, device),
        Polygons2PolygonsInteriors(polygons, device),
    )


def MultiPolygon2MultiPolygonExterior(
    multipolygon: MultiPolygon, device: torch.device | str = "cpu"
) -> MultiPolygonExterior:
    """Convert a MultiPolygon object to a MultiPolygonExterior object.

    Args:
        multipolygon: The MultiPolygon object.
        device: The device to use.

    Returns:
        The exterior of the MultiPolygon as a MultiPolygonExterior object.
    """
    exterior, V_ps = Polygons2PolygonsExterior(
        gpd.GeoSeries(multipolygon.geoms), device
    )  # [P, max(V_ps), 2], [P]
    return MultiPolygonExterior(exterior, V_ps)


def MultiPolygons2MultiPolygonsExterior(
    multipolygons: gpd.GeoSeries, device: torch.device | str = "cpu"
) -> MultiPolygonsExterior:
    """Convert a batch of MultiPolygon objects to a MultiPolygonsExterior
    object.

    Args:
        multipolygons: The GeoSeries of MultiPolygon objects.
        device: The device to use.

    Returns:
        The exterior of the batch of MultiPolygon objects as a
        MultiPolygonsExterior object.
    """
    polygons_geoseries = multipolygons.explode()  # [sum(P_bs)]
    P_bs = __count_freqs_until(
        polygons_geoseries, len(multipolygons), device
    )  # [B]
    exterior_packed, V_ps_packed = Polygons2PolygonsExterior(
        cast(gpd.GeoSeries, polygons_geoseries.reset_index(drop=True)), device
    )  # [sum(P_bs), max(V_ps), 2], [sum(P_bs)]
    P_bs_max = int(P_bs.max())
    exteriors = pad_packed_batched(
        exterior_packed, P_bs, P_bs_max
    )  # [B, max(P_bs), max(V_ps), 2]
    V_ps = pad_packed_batched(V_ps_packed, P_bs, P_bs_max)  # [B, max(P_bs)]
    return MultiPolygonsExterior(exteriors, P_bs, V_ps)


def MultiPolygon2MultiPolygonInteriors(
    multipolygon: MultiPolygon, device: torch.device | str = "cpu"
) -> MultiPolygonInteriors:
    """Convert a MultiPolygon object to a MultiPolygonInteriors object.

    Args:
        multipolygon: The MultiPolygon object.
        device: The device to use.

    Returns:
        The interiors of the MultiPolygon as a MultiPolygonInteriors object.
    """
    interiors, I_ps, V_is = Polygons2PolygonsInteriors(
        gpd.GeoSeries(multipolygon.geoms), device
    )  # [P, max(I_ps), max(V_is), 2], [P], [P, max(I_ps)]
    return MultiPolygonInteriors(interiors, I_ps, V_is)


def MultiPolygons2MultiPolygonsInteriors(
    multipolygons: gpd.GeoSeries, device: torch.device | str = "cpu"
) -> MultiPolygonsInteriors:
    """Convert a batch of MultiPolygon objects to a MultiPolygonsInteriors
    object.

    Args:
        multipolygons: The GeoSeries of MultiPolygon objects.
        device: The device to use.

    Returns:
        The interiors of the batch of MultiPolygon objects as a
        MultiPolygonsInteriors object.
    """
    polygons_geoseries = multipolygons.explode()  # [sum(P_bs)]
    P_bs = __count_freqs_until(
        polygons_geoseries, len(multipolygons), device
    )  # [B]
    (
        interiors_packed,  # [sum(P_bs), max(I_ps), max(V_is), 2]
        I_ps_packed,  # [sum(P_bs)]
        V_is_packed,  # [sum(P_bs), max(I_ps)]
    ) = Polygons2PolygonsInteriors(
        cast(gpd.GeoSeries, polygons_geoseries.reset_index(drop=True)), device
    )
    P_bs_max = int(P_bs.max())
    interiors = pad_packed_batched(
        interiors_packed, P_bs, P_bs_max
    )  # [B, max(P_bs), max(I_ps), max(V_is), 2]
    I_ps = pad_packed_batched(I_ps_packed, P_bs, P_bs_max)  # [B, max(P_bs)]
    V_is = pad_packed_batched(
        V_is_packed, P_bs, P_bs_max
    )  # [B, max(P_bs), max(I_ps)]
    return MultiPolygonsInteriors(interiors, P_bs, I_ps, V_is)


def MultiPolygon2MultiPolygonVertices(
    multipolygon: MultiPolygon, device: torch.device | str = "cpu"
) -> MultiPolygonVertices:
    """Convert a MultiPolygon object to a MultiPolygonVertices object.

    Args:
        multipolygon: The MultiPolygon object.
        device: The device to use.

    Returns:
        The vertices of the MultiPolygon as a MultiPolygonVertices object.
    """
    return MultiPolygonVertices(
        MultiPolygon2MultiPolygonExterior(multipolygon, device),
        MultiPolygon2MultiPolygonInteriors(multipolygon, device),
    )


def MultiPolygons2MultiPolygonsVertices(
    multipolygons: gpd.GeoSeries, device: torch.device | str = "cpu"
) -> MultiPolygonsVertices:
    """Convert a batch of MultiPolygon objects to a MultiPolygonsVertices
    object.

    Args:
        multipolygons: The GeoSeries of MultiPolygon objects.
        device: The device to use.

    Returns:
        The vertices of the batch of MultiPolygon objects as a
        MultiPolygonsVertices object.
    """
    return MultiPolygonsVertices(
        MultiPolygons2MultiPolygonsExterior(multipolygons, device),
        MultiPolygons2MultiPolygonsInteriors(multipolygons, device),
    )


def PolygonLike2PolygonLikeExterior(
    polygonlike: Polygon | MultiPolygon, device: torch.device | str = "cpu"
) -> PolygonLikeExterior:
    """Convert a Polygon or MultiPolygon object to a PolygonLikeExterior
    object.

    Args:
        polygonlike: The Polygon or MultiPolygon object.
        device: The device to use.

    Returns:
        The exterior of the Polygon or MultiPolygon as a PolygonLikeExterior
        object.
    """
    if isinstance(polygonlike, Polygon):
        (exterior,) = Polygon2PolygonExterior(polygonlike, device)
        exterior = exterior.unsqueeze(0)
        V_ps = torch.tensor([1], device=device)
    else:
        exterior, V_ps = MultiPolygon2MultiPolygonExterior(polygonlike, device)
    return PolygonLikeExterior(exterior, V_ps)


def PolygonLikes2PolygonLikesExterior(
    polygonlikes: gpd.GeoSeries, device: torch.device | str = "cpu"
) -> PolygonLikesExterior:
    """Convert a batch of Polygon or MultiPolygon objects to a
    PolygonLikesExterior object.

    Args:
        polygonlikes: The GeoSeries of Polygon or MultiPolygon objects.
        device: The device to use.

    Returns:
        The exterior of the batch of Polygon or MultiPolygon objects as a
        PolygonLikesExterior object.
    """
    exterior, P_bs, V_ps = MultiPolygons2MultiPolygonsExterior(
        polygonlikes, device
    )
    return PolygonLikesExterior(exterior, P_bs, V_ps)


def PolygonLike2PolygonLikeInteriors(
    polygonlike: Polygon | MultiPolygon, device: torch.device | str = "cpu"
) -> PolygonLikeInteriors:
    """Convert a Polygon or MultiPolygon object to a PolygonLikeInteriors
    object.

    Args:
        polygonlike: The Polygon or MultiPolygon object.
        device: The device to use.

    Returns:
        The interiors of the Polygon or MultiPolygon as a PolygonLikeInteriors
        object.
    """
    if isinstance(polygonlike, Polygon):
        interiors, V_is = Polygon2PolygonInteriors(polygonlike, device)
        interiors = interiors.unsqueeze(0)
        I_ps = torch.tensor([1], device=device)
        V_is = V_is.unsqueeze(0)
    else:
        interiors, I_ps, V_is = MultiPolygon2MultiPolygonInteriors(
            polygonlike, device
        )
    return PolygonLikeInteriors(interiors, I_ps, V_is)


def PolygonLikes2PolygonLikesInteriors(
    polygonlikes: gpd.GeoSeries, device: torch.device | str = "cpu"
) -> PolygonLikesInteriors:
    """Convert a batch of Polygon or MultiPolygon objects to a
    PolygonLikesInteriors object.

    Args:
        polygonlikes: The GeoSeries of Polygon or MultiPolygon objects.
        device: The device to use.

    Returns:
        The interiors of the batch of Polygon or MultiPolygon objects as a
        PolygonLikesInteriors object.
    """
    interiors, P_bs, I_ps, V_is = MultiPolygons2MultiPolygonsInteriors(
        polygonlikes, device
    )
    return PolygonLikesInteriors(interiors, P_bs, I_ps, V_is)


def PolygonLike2PolygonLikeVertices(
    polygonlike: Polygon | MultiPolygon, device: torch.device | str = "cpu"
) -> PolygonLikeVertices:
    """Convert a Polygon or MultiPolygon object to a PolygonLikeVertices
    object.

    Args:
        polygonlike: The Polygon or MultiPolygon object.
        device: The device to use.

    Returns:
        The vertices of the Polygon or MultiPolygon as a PolygonLikeVertices
        object.
    """
    return PolygonLikeVertices(
        PolygonLike2PolygonLikeExterior(polygonlike, device),
        PolygonLike2PolygonLikeInteriors(polygonlike, device),
    )


def PolygonLikes2PolygonLikesVertices(
    polygonlikes: gpd.GeoSeries, device: torch.device | str = "cpu"
) -> PolygonLikesVertices:
    """Convert a batch of Polygon or MultiPolygon objects to a
    PolygonLikesVertices object.

    Args:
        polygonlikes: The GeoSeries of Polygon or MultiPolygon objects.
        device: The device to use.

    Returns:
        The vertices of the batch of Polygon or MultiPolygon objects as a
        PolygonLikesVertices object.
    """
    return PolygonLikesVertices(
        PolygonLikes2PolygonLikesExterior(polygonlikes, device),
        PolygonLikes2PolygonLikesInteriors(polygonlikes, device),
    )


def PolygonVertices2Polygon(polygon_vertices: PolygonVertices) -> Polygon:
    """Convert a PolygonVertices object to a Polygon object.

    Args:
        polygon_vertices: The PolygonVertices object.

    Returns:
        The Polygon object.
    """
    exterior = polygon_vertices.exterior
    exterior_vertices = exterior.vertices  # [V, 2]
    interiors = polygon_vertices.interiors
    interiors_vertices = [
        interiors.vertices[i, : interiors.V_is[i]]
        for i in range(len(interiors.vertices))
    ]  # I x [V_i, 2]
    return Polygon(exterior_vertices, interiors_vertices)


def PolygonsVertices2Polygons(
    polygons_vertices: PolygonsVertices,
) -> gpd.GeoSeries:
    """Convert a PolygonsVertices object to a batch of Polygon objects.

    Args:
        polygons_vertices: The PolygonsVertices object.

    Returns:
        The GeoSeries of Polygon objects.
    """
    exterior = polygons_vertices.exterior
    exterior_vertices = [
        exterior.vertices[b, : exterior.V_bs[b]]
        for b in range(len(exterior.vertices))
    ]  # B x [V_b, 2]
    interiors = polygons_vertices.interiors
    interiors_vertices = [
        [
            interiors.vertices[b, i, : interiors.V_is[b, i]]
            for i in range(interiors.I_bs[b])
        ]  # I_b x [V_i, 2]
        for b in range(len(interiors.vertices))
    ]  # B x I_b x [V_i, 2]
    return gpd.GeoSeries([
        Polygon(exterior_vertices[b], interiors_vertices[b])
        for b in range(len(exterior_vertices))
    ])


def MultiPolygonVertices2MultiPolygon(
    multipolygon_vertices: MultiPolygonVertices,
) -> MultiPolygon:
    """Convert a MultiPolygonVertices object to a MultiPolygon object.

    Args:
        multipolygon_vertices: The MultiPolygonVertices object.

    Returns:
        The MultiPolygon object.
    """
    exteriors = multipolygon_vertices.exteriors
    exterior_vertices = [
        exteriors.vertices[p, : exteriors.V_ps[p]]
        for p in range(len(exteriors.vertices))
    ]
    interiors = multipolygon_vertices.interiors
    interior_vertices = [
        [
            interiors.vertices[p, i, : interiors.V_is[p, i]]
            for i in range(interiors.I_ps[p])
        ]
        for p in range(len(interiors.vertices))
    ]
    return MultiPolygon(tuple(zip(exterior_vertices, interior_vertices)))


def MultiPolygonsVertices2MultiPolygons(
    multipolygons_vertices: MultiPolygonsVertices,
) -> gpd.GeoSeries:
    """Convert a MultiPolygonsVertices object to a batch of MultiPolygon
    objects.

    Args:
        multipolygons_vertices: The MultiPolygonsVertices object.

    Returns:
        The GeoSeries of MultiPolygon objects.
    """
    exteriors = multipolygons_vertices.exteriors
    exterior_vertices = [
        [
            exteriors.vertices[b, p, : exteriors.V_ps[b, p]]
            for p in range(exteriors.P_bs[b])
        ]  # P_b x [V_p, 2]
        for b in range(len(exteriors.vertices))
    ]  # B x P_b x [V_p, 2]
    interiors = multipolygons_vertices.interiors
    interior_vertices = [
        [
            [
                interiors.vertices[b, p, i, : interiors.V_is[b, p, i]]
                for i in range(interiors.I_ps[b, p])
            ]  # I_p x [V_i, 2]
            for p in range(interiors.P_bs[b])
        ]  # P_b x I_p x [V_i, 2]
        for b in range(len(interiors.vertices))
    ]  # B x P_b x I_p x [V_i, 2]
    return gpd.GeoSeries([
        MultiPolygon(tuple(zip(exterior_vertices[b], interior_vertices[b])))
        for b in range(len(exterior_vertices))
    ])


def PolygonLikeVertices2PolygonLike(
    polygonlike_vertices: PolygonLikeVertices,
) -> Polygon | MultiPolygon:
    """Convert a PolygonLikeVertices object to a Polygon or MultiPolygon
    object.

    Args:
        polygonlike_vertices: The PolygonLikeVertices object.

    Returns:
        The Polygon or MultiPolygon object.
    """
    multipolygon = MultiPolygonVertices2MultiPolygon(
        MultiPolygonVertices(
            MultiPolygonExterior(
                polygonlike_vertices.exterior.vertices,
                polygonlike_vertices.exterior.V_ps,
            ),
            MultiPolygonInteriors(
                polygonlike_vertices.interiors.vertices,
                polygonlike_vertices.interiors.I_ps,
                polygonlike_vertices.interiors.V_is,
            ),
        )
    )
    if len(multipolygon.geoms) == 1:
        return multipolygon.geoms[0]
    return multipolygon


def PolygonLikesVertices2PolygonLikes(
    polygonlikes_vertices: PolygonLikesVertices,
) -> gpd.GeoSeries:
    """Convert a PolygonLikesVertices object to a batch of Polygon or
    MultiPolygon objects.

    Args:
        polygonlikes_vertices: The PolygonLikesVertices object.

    Returns:
        The GeoSeries of Polygon or MultiPolygon objects.
    """
    multipolygons = MultiPolygonsVertices2MultiPolygons(
        MultiPolygonsVertices(
            MultiPolygonsExterior(
                polygonlikes_vertices.exterior.vertices,
                polygonlikes_vertices.exterior.P_bs,
                polygonlikes_vertices.exterior.V_ps,
            ),
            MultiPolygonsInteriors(
                polygonlikes_vertices.interiors.vertices,
                polygonlikes_vertices.interiors.P_bs,
                polygonlikes_vertices.interiors.I_ps,
                polygonlikes_vertices.interiors.V_is,
            ),
        )
    )
    for i, multipolygon in enumerate(multipolygons):
        if len(multipolygon.geoms) == 1:
            multipolygons[i] = multipolygon.geoms[0]
    return multipolygons
