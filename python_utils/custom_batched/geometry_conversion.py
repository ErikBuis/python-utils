from typing import NamedTuple, cast

import geopandas as gpd
import pandas as pd
import torch
from shapely import LinearRing, MultiPolygon, Polygon

from ..modules.torch import count_freqs_until
from ..modules_batched.random import (
    rand_float_decreasingly_likely,
    rand_int_decreasingly_likely,
)
from ..modules_batched.torch import pad_packed_batched


# ############################### TENSOR TYPES ################################

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
    "PolygonLikesExterior",
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


# ################ CONVERT SHAPELY/GEOPANDAS OBJECT TO TENSOR #################


def __count_freqs_until(
    obj_with_index: pd.Series | pd.DataFrame,
    high: int,
    device: torch.device | str,
) -> torch.Tensor:
    """Count the frequency of each integer index value in range(0, high).

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
    return count_freqs_until(index, high)


def LinearRing2LinearRingVertices(
    linearring: LinearRing,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> LinearRingVertices:
    """Convert a LinearRing object to a LinearRingVertices object.

    Args:
        linearring: The LinearRing object.
        dtype: The data type of the output vertices.
        device: The device to use.

    Returns:
        The vertices of the LinearRing object as a LinearRingVertices object.
    """
    vertices = torch.tensor(
        linearring.coords, dtype=dtype, device=device
    )  # [V, 2]
    return LinearRingVertices(vertices)


def LinearRings2LinearRingsVertices(
    linearrings: gpd.GeoSeries,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> LinearRingsVertices:
    """Convert a batch of LinearRing objects to a LinearRingsVertices object.

    Args:
        linearrings: The GeoSeries of LinearRing objects. Can also contain
            NaN/None values, these will be handled correctly by inserting
            an empty row into the output.
        dtype: The data type of the output vertices.
        device: The device to use.

    Returns:
        The vertices of a batch of the LinearRing objects as a
        LinearRingsVertices object.
    """
    # I timed multiple different approaches against each other, so almost all
    # conceivable variations of the below code were compared. This one turned
    # out to be the fastest.
    vertices_df = linearrings.get_coordinates()  # [sum(V_bs), 2]
    V_bs = __count_freqs_until(vertices_df, len(linearrings), device)  # [B]
    vertices_packed = torch.from_numpy(
        vertices_df.to_numpy()
    ).to(dtype=dtype, device=device)  # [sum(V_bs), 2]  # fmt: skip
    vertices = pad_packed_batched(
        vertices_packed, V_bs, int(V_bs.max()) if len(V_bs) > 0 else 0
    )  # [B, max(V_bs), 2]
    return LinearRingsVertices(vertices, V_bs)


def Polygon2PolygonExterior(
    polygon: Polygon,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> PolygonExterior:
    """Convert a Polygon object to a PolygonExterior object.

    Args:
        polygon: The Polygon object.
        dtype: The data type of the output vertices.
        device: The device to use.

    Returns:
        The exterior of the Polygon as a PolygonExterior object.
    """
    (exterior,) = LinearRing2LinearRingVertices(
        polygon.exterior, dtype, device
    )  # [V, 2]
    return PolygonExterior(exterior)


def Polygons2PolygonsExterior(
    polygons: gpd.GeoSeries,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> PolygonsExterior:
    """Convert a batch of Polygon objects to a PolygonsExterior object.

    Args:
        polygons: The GeoSeries of Polygon objects.
        dtype: The data type of the output vertices.
        device: The device to use.

    Returns:
        The exteriors of the batch of Polygon objects as a PolygonsExterior
        object.
    """
    exterior, V_bs = LinearRings2LinearRingsVertices(
        polygons.exterior, dtype, device
    )  # [B, max(V_bs), 2], [B]
    return PolygonsExterior(exterior, V_bs)


def Polygon2PolygonInteriors(
    polygon: Polygon,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> PolygonInteriors:
    """Convert a Polygon object to a PolygonInteriors object.

    Args:
        polygon: The Polygon object.
        dtype: The data type of the output vertices.
        device: The device to use.

    Returns:
        The interiors of the Polygon as a PolygonInteriors object.
    """
    interiors, V_is = LinearRings2LinearRingsVertices(
        gpd.GeoSeries(polygon.interiors), dtype, device
    )  # [I, max(V_is), 2], [I]
    return PolygonInteriors(interiors, V_is)


def Polygons2PolygonsInteriors(
    polygons: gpd.GeoSeries,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> PolygonsInteriors:
    """Convert a batch of Polygon objects to a PolygonsInteriors object.

    Args:
        polygons: The GeoSeries of Polygon objects.
        dtype: The data type of the output vertices.
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
        gpd.GeoSeries(interiors_series.reset_index(drop=True)), dtype, device
    )  # [sum(I_bs), max(V_is), 2], [sum(I_bs)]
    I_bs_max = int(I_bs.max())
    interiors = pad_packed_batched(
        interiors_packed, I_bs, I_bs_max
    )  # [B, max(I_bs), max(V_is), 2]
    V_is = pad_packed_batched(V_is_packed, I_bs, I_bs_max)  # [B, max(I_bs)]
    return PolygonsInteriors(interiors, I_bs, V_is)


def Polygon2PolygonVertices(
    polygon: Polygon,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> PolygonVertices:
    """Convert a Polygon object to a PolygonVertices object.

    Args:
        polygon: The Polygon object.
        dtype: The data type of the output vertices.
        device: The device to use.

    Returns:
        The vertices of the Polygon as a PolygonVertices object.
    """
    return PolygonVertices(
        Polygon2PolygonExterior(polygon, dtype, device),
        Polygon2PolygonInteriors(polygon, dtype, device),
    )


def Polygons2PolygonsVertices(
    polygons: gpd.GeoSeries,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> PolygonsVertices:
    """Convert a batch of Polygon objects to a PolygonsVertices object.

    Args:
        polygons: The GeoSeries of Polygon objects.
        dtype: The data type of the output vertices.
        device: The device to use.

    Returns:
        The vertices of the batch of Polygon objects as a PolygonsVertices
        object.
    """
    return PolygonsVertices(
        Polygons2PolygonsExterior(polygons, dtype, device),
        Polygons2PolygonsInteriors(polygons, dtype, device),
    )


def MultiPolygon2MultiPolygonExterior(
    multipolygon: MultiPolygon,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> MultiPolygonExterior:
    """Convert a MultiPolygon object to a MultiPolygonExterior object.

    Args:
        multipolygon: The MultiPolygon object.
        dtype: The data type of the output vertices.
        device: The device to use.

    Returns:
        The exterior of the MultiPolygon as a MultiPolygonExterior object.
    """
    exterior, V_ps = Polygons2PolygonsExterior(
        gpd.GeoSeries(multipolygon.geoms), dtype, device
    )  # [P, max(V_ps), 2], [P]
    return MultiPolygonExterior(exterior, V_ps)


def MultiPolygons2MultiPolygonsExterior(
    multipolygons: gpd.GeoSeries,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> MultiPolygonsExterior:
    """Convert a batch of MultiPolygon objects to a MultiPolygonsExterior
    object.

    Args:
        multipolygons: The GeoSeries of MultiPolygon objects.
        dtype: The data type of the output vertices.
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
        cast(gpd.GeoSeries, polygons_geoseries.reset_index(drop=True)),
        dtype,
        device,
    )  # [sum(P_bs), max(V_ps), 2], [sum(P_bs)]
    P_bs_max = int(P_bs.max())
    exteriors = pad_packed_batched(
        exterior_packed, P_bs, P_bs_max
    )  # [B, max(P_bs), max(V_ps), 2]
    V_ps = pad_packed_batched(V_ps_packed, P_bs, P_bs_max)  # [B, max(P_bs)]
    return MultiPolygonsExterior(exteriors, P_bs, V_ps)


def MultiPolygon2MultiPolygonInteriors(
    multipolygon: MultiPolygon,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> MultiPolygonInteriors:
    """Convert a MultiPolygon object to a MultiPolygonInteriors object.

    Args:
        multipolygon: The MultiPolygon object.
        dtype: The data type of the output vertices.
        device: The device to use.

    Returns:
        The interiors of the MultiPolygon as a MultiPolygonInteriors object.
    """
    interiors, I_ps, V_is = Polygons2PolygonsInteriors(
        gpd.GeoSeries(multipolygon.geoms), dtype, device
    )  # [P, max(I_ps), max(V_is), 2], [P], [P, max(I_ps)]
    return MultiPolygonInteriors(interiors, I_ps, V_is)


def MultiPolygons2MultiPolygonsInteriors(
    multipolygons: gpd.GeoSeries,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> MultiPolygonsInteriors:
    """Convert a batch of MultiPolygon objects to a MultiPolygonsInteriors
    object.

    Args:
        multipolygons: The GeoSeries of MultiPolygon objects.
        dtype: The data type of the output vertices.
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
        cast(gpd.GeoSeries, polygons_geoseries.reset_index(drop=True)),
        dtype,
        device,
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
    multipolygon: MultiPolygon,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> MultiPolygonVertices:
    """Convert a MultiPolygon object to a MultiPolygonVertices object.

    Args:
        multipolygon: The MultiPolygon object.
        dtype: The data type of the output vertices.
        device: The device to use.

    Returns:
        The vertices of the MultiPolygon as a MultiPolygonVertices object.
    """
    return MultiPolygonVertices(
        MultiPolygon2MultiPolygonExterior(multipolygon, dtype, device),
        MultiPolygon2MultiPolygonInteriors(multipolygon, dtype, device),
    )


def MultiPolygons2MultiPolygonsVertices(
    multipolygons: gpd.GeoSeries,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> MultiPolygonsVertices:
    """Convert a batch of MultiPolygon objects to a MultiPolygonsVertices
    object.

    Args:
        multipolygons: The GeoSeries of MultiPolygon objects.
        dtype: The data type of the output vertices.
        device: The device to use.

    Returns:
        The vertices of the batch of MultiPolygon objects as a
        MultiPolygonsVertices object.
    """
    return MultiPolygonsVertices(
        MultiPolygons2MultiPolygonsExterior(multipolygons, dtype, device),
        MultiPolygons2MultiPolygonsInteriors(multipolygons, dtype, device),
    )


def PolygonLike2PolygonLikeExterior(
    polygon_like: Polygon | MultiPolygon,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> PolygonLikeExterior:
    """Convert a Polygon or MultiPolygon object to a PolygonLikeExterior
    object.

    Args:
        polygon_like: The Polygon or MultiPolygon object.
        dtype: The data type of the output vertices.
        device: The device to use.

    Returns:
        The exterior of the Polygon or MultiPolygon as a PolygonLikeExterior
        object.
    """
    if isinstance(polygon_like, Polygon):
        (exterior,) = Polygon2PolygonExterior(polygon_like, dtype, device)
        V_ps = torch.tensor([len(exterior)], device=device)
        exterior = exterior.unsqueeze(0)
    else:
        exterior, V_ps = MultiPolygon2MultiPolygonExterior(
            polygon_like, dtype, device
        )
    return PolygonLikeExterior(exterior, V_ps)


def PolygonLikes2PolygonLikesExterior(
    polygon_likes: gpd.GeoSeries,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> PolygonLikesExterior:
    """Convert a batch of Polygon or MultiPolygon objects to a
    PolygonLikesExterior object.

    Args:
        polygon_likes: The GeoSeries of Polygon or MultiPolygon objects.
        dtype: The data type of the output vertices.
        device: The device to use.

    Returns:
        The exterior of the batch of Polygon or MultiPolygon objects as a
        PolygonLikesExterior object.
    """
    exterior, P_bs, V_ps = MultiPolygons2MultiPolygonsExterior(
        polygon_likes, dtype, device
    )
    return PolygonLikesExterior(exterior, P_bs, V_ps)


def PolygonLike2PolygonLikeInteriors(
    polygon_like: Polygon | MultiPolygon,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> PolygonLikeInteriors:
    """Convert a Polygon or MultiPolygon object to a PolygonLikeInteriors
    object.

    Args:
        polygon_like: The Polygon or MultiPolygon object.
        dtype: The data type of the output vertices.
        device: The device to use.

    Returns:
        The interiors of the Polygon or MultiPolygon as a PolygonLikeInteriors
        object.
    """
    if isinstance(polygon_like, Polygon):
        interiors, V_is = Polygon2PolygonInteriors(polygon_like, dtype, device)
        I_ps = torch.tensor([len(interiors)], device=device)
        V_is = V_is.unsqueeze(0)
        interiors = interiors.unsqueeze(0)
    else:
        interiors, I_ps, V_is = MultiPolygon2MultiPolygonInteriors(
            polygon_like, dtype, device
        )
    return PolygonLikeInteriors(interiors, I_ps, V_is)


def PolygonLikes2PolygonLikesInteriors(
    polygon_likes: gpd.GeoSeries,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> PolygonLikesInteriors:
    """Convert a batch of Polygon or MultiPolygon objects to a
    PolygonLikesInteriors object.

    Args:
        polygon_likes: The GeoSeries of Polygon or MultiPolygon objects.
        dtype: The data type of the output vertices.
        device: The device to use.

    Returns:
        The interiors of the batch of Polygon or MultiPolygon objects as a
        PolygonLikesInteriors object.
    """
    interiors, P_bs, I_ps, V_is = MultiPolygons2MultiPolygonsInteriors(
        polygon_likes, dtype, device
    )
    return PolygonLikesInteriors(interiors, P_bs, I_ps, V_is)


def PolygonLike2PolygonLikeVertices(
    polygon_like: Polygon | MultiPolygon,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> PolygonLikeVertices:
    """Convert a Polygon or MultiPolygon object to a PolygonLikeVertices
    object.

    Args:
        polygon_like: The Polygon or MultiPolygon object.
        dtype: The data type of the output vertices.
        device: The device to use.

    Returns:
        The vertices of the Polygon or MultiPolygon as a PolygonLikeVertices
        object.
    """
    return PolygonLikeVertices(
        PolygonLike2PolygonLikeExterior(polygon_like, dtype, device),
        PolygonLike2PolygonLikeInteriors(polygon_like, dtype, device),
    )


def PolygonLikes2PolygonLikesVertices(
    polygon_likes: gpd.GeoSeries,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> PolygonLikesVertices:
    """Convert a batch of Polygon or MultiPolygon objects to a
    PolygonLikesVertices object.

    Args:
        polygon_likes: The GeoSeries of Polygon or MultiPolygon objects.
        dtype: The data type of the output vertices.
        device: The device to use.

    Returns:
        The vertices of the batch of Polygon or MultiPolygon objects as a
        PolygonLikesVertices object.
    """
    return PolygonLikesVertices(
        PolygonLikes2PolygonLikesExterior(polygon_likes, dtype, device),
        PolygonLikes2PolygonLikesInteriors(polygon_likes, dtype, device),
    )


# ################ CONVERT TENSOR TO SHAPELY/GEOPANDAS OBJECT #################


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
    polygon_like_vertices: PolygonLikeVertices,
) -> Polygon | MultiPolygon:
    """Convert a PolygonLikeVertices object to a Polygon or MultiPolygon
    object.

    Args:
        polygon_like_vertices: The PolygonLikeVertices object.

    Returns:
        The Polygon or MultiPolygon object.
    """
    multipolygon = MultiPolygonVertices2MultiPolygon(
        MultiPolygonVertices(
            MultiPolygonExterior(
                polygon_like_vertices.exterior.vertices,
                polygon_like_vertices.exterior.V_ps,
            ),
            MultiPolygonInteriors(
                polygon_like_vertices.interiors.vertices,
                polygon_like_vertices.interiors.I_ps,
                polygon_like_vertices.interiors.V_is,
            ),
        )
    )
    if len(multipolygon.geoms) == 1:
        return multipolygon.geoms[0]
    return multipolygon


def PolygonLikesVertices2PolygonLikes(
    polygon_likes_vertices: PolygonLikesVertices,
) -> gpd.GeoSeries:
    """Convert a PolygonLikesVertices object to a batch of Polygon or
    MultiPolygon objects.

    Args:
        polygon_likes_vertices: The PolygonLikesVertices object.

    Returns:
        The GeoSeries of Polygon or MultiPolygon objects.
    """
    multipolygons = MultiPolygonsVertices2MultiPolygons(
        MultiPolygonsVertices(
            MultiPolygonsExterior(
                polygon_likes_vertices.exterior.vertices,
                polygon_likes_vertices.exterior.P_bs,
                polygon_likes_vertices.exterior.V_ps,
            ),
            MultiPolygonsInteriors(
                polygon_likes_vertices.interiors.vertices,
                polygon_likes_vertices.interiors.P_bs,
                polygon_likes_vertices.interiors.I_ps,
                polygon_likes_vertices.interiors.V_is,
            ),
        )
    )
    for i, multipolygon in enumerate(multipolygons):
        if len(multipolygon.geoms) == 1:
            multipolygons[i] = multipolygon.geoms[0]
    return multipolygons


# ######################## GENERATE RANDOM GEOMETRIES #########################


def generate_random_polygon_like() -> Polygon | MultiPolygon:
    """Generate a random valid Polygon or MultiPolygon object.

    The amount of Polygons, the amount of vertices in the exteriors and
    interiors, the amount of interiors and the coordinates of the vertices are
    all randomly generated.

    In theory, this function could generate any Polygon or MultiPolygon object.

    Returns:
        A random Polygon or MultiPolygon object.
    """
    scaling_factor = (
        rand_float_decreasingly_likely(1) * 11 + 1
    )  # in [1, inf), E(X) = 12

    # Generate the exterior of the Polygon.
    V = int(rand_int_decreasingly_likely(1)) * 7 + 3  # in [3, inf), E(X) = 10
    vertices = (
        torch.rand(V, 2) - 0.5
    ) * scaling_factor  # in (-inf, inf), E(abs(X)) = 3
    exterior = PolygonExterior(vertices)

    # Generate the interiors of the Polygon.
    I = int(
        rand_int_decreasingly_likely(1)
    )  # in [0, inf), E(X) = 1  # noqa: E741
    V_is = rand_int_decreasingly_likely(I) * 7 + 3  # in [3, inf), E(X) = 10
    vertices = (
        torch.rand(I, 0 if I == 0 else int(V_is.max()), 2) - 0.5
    ) * scaling_factor  # in (-inf, inf), E(abs(X)) = 3
    interiors = PolygonInteriors(vertices, V_is)

    # Create the Polygon object.
    polygon_invalid = PolygonVertices2Polygon(
        PolygonVertices(exterior, interiors)
    )

    # Make the Polygon valid.
    polygon_like = polygon_invalid.buffer(0)

    if polygon_like.is_empty:
        # If the Polygon is empty, try again. This should be very rare, as it
        # would only happen in cases where e.g. the interior fully covers the
        # exterior.
        return generate_random_polygon_like()

    return polygon_like


def generate_random_polygon() -> Polygon:
    """Generate a random valid Polygon object.

    The amount of vertices in the exteriors and interiors, the amount of
    interiors and the coordinates of the vertices are all randomly generated.

    In theory, this function could generate any Polygon object.

    Returns:
        A random Polygon object.
    """
    polygon_like = generate_random_polygon_like()
    if isinstance(polygon_like, Polygon):
        return polygon_like
    return polygon_like.geoms[0]


def generate_random_multipolygon() -> MultiPolygon:
    """Generate a random valid MultiPolygon object.

    The amount of Polygons, the amount of vertices in the exteriors and
    interiors, the amount of interiors and the coordinates of the vertices are
    all randomly generated.

    In theory, this function could generate any MultiPolygon object.

    Returns:
        A random MultiPolygon object.
    """
    polygon_like = generate_random_polygon_like()
    if isinstance(polygon_like, MultiPolygon):
        return polygon_like
    return MultiPolygon([polygon_like])


def generate_random_polygon_likes(amount: int = 64) -> gpd.GeoSeries:
    """Generate a random GeoSeries of Polygon and MultiPolygon objects.

    The amount of Polygons, the amount of vertices in the exteriors and
    interiors, the amount of interiors and the coordinates of the vertices are
    all randomly generated.

    In theory, this function could generate any Polygon or MultiPolygon object.

    Args:
        amount: The amount of Polygon and MultiPolygon objects to generate.

    Returns:
        A random GeoSeries of Polygon and MultiPolygon objects.
    """
    return gpd.GeoSeries(
        [generate_random_polygon_like() for _ in range(amount)]
    )


def generate_random_polygons(amount: int = 64) -> gpd.GeoSeries:
    """Generate a random GeoSeries of Polygon objects.

    The amount of vertices in the exteriors and interiors, the amount of
    interiors and the coordinates of the vertices are all randomly generated.

    In theory, this function could generate any Polygon object.

    Args:
        amount: The amount of Polygon objects to generate.

    Returns:
        A random GeoSeries of Polygon objects.
    """
    return gpd.GeoSeries([generate_random_polygon() for _ in range(amount)])


def generate_random_multipolygons(amount: int = 64) -> gpd.GeoSeries:
    """Generate a random GeoSeries of MultiPolygon objects.

    The amount of Polygons, the amount of vertices in the exteriors and
    interiors, the amount of interiors and the coordinates of the vertices are
    all randomly generated.

    In theory, this function could generate any MultiPolygon object.

    Args:
        amount: The amount of MultiPolygon objects to generate.

    Returns:
        A random GeoSeries of MultiPolygon objects.
    """
    return gpd.GeoSeries(
        [generate_random_multipolygon() for _ in range(amount)]
    )
