# pyright: reportAttributeAccessIssue=false, reportMissingImports=false

from typing import Annotated

from pydantic import GetPydanticSchema
from pydantic_core import core_schema


try:
    import torch

    TensorAnn = Annotated[
        torch.Tensor,
        GetPydanticSchema(
            lambda tp, handler: core_schema.is_instance_schema(torch.Tensor)
        ),
    ]
except ImportError:
    pass

try:
    import shapely

    PointAnn = Annotated[
        shapely.geometry.Point,
        GetPydanticSchema(
            lambda tp, handler: core_schema.is_instance_schema(
                shapely.geometry.Point
            )
        ),
    ]
    LineStringAnn = Annotated[
        shapely.geometry.LineString,
        GetPydanticSchema(
            lambda tp, handler: core_schema.is_instance_schema(
                shapely.geometry.LineString
            )
        ),
    ]
    LinearRingAnn = Annotated[
        shapely.geometry.LinearRing,
        GetPydanticSchema(
            lambda tp, handler: core_schema.is_instance_schema(
                shapely.geometry.LinearRing
            )
        ),
    ]
    PolygonAnn = Annotated[
        shapely.geometry.Polygon,
        GetPydanticSchema(
            lambda tp, handler: core_schema.is_instance_schema(
                shapely.geometry.Polygon
            )
        ),
    ]
    MultiPointAnn = Annotated[
        shapely.geometry.MultiPoint,
        GetPydanticSchema(
            lambda tp, handler: core_schema.is_instance_schema(
                shapely.geometry.MultiPoint
            )
        ),
    ]
    MultiLineStringAnn = Annotated[
        shapely.geometry.MultiLineString,
        GetPydanticSchema(
            lambda tp, handler: core_schema.is_instance_schema(
                shapely.geometry.MultiLineString
            )
        ),
    ]
    MultiPolygonAnn = Annotated[
        shapely.geometry.MultiPolygon,
        GetPydanticSchema(
            lambda tp, handler: core_schema.is_instance_schema(
                shapely.geometry.MultiPolygon
            )
        ),
    ]
    MultiGeometryCollectionAnn = Annotated[
        shapely.geometry.GeometryCollection,
        GetPydanticSchema(
            lambda tp, handler: core_schema.is_instance_schema(
                shapely.geometry.GeometryCollection
            )
        ),
    ]
except ImportError:
    pass

try:
    import pytorch3d

    PointcloudsAnn = Annotated[
        pytorch3d.structures.Pointclouds,
        GetPydanticSchema(
            lambda tp, handler: core_schema.is_instance_schema(
                pytorch3d.structures.Pointclouds
            )
        ),
    ]
except ImportError:
    pass
