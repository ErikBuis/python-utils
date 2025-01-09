# pyright: reportMissingImports=false, reportAttributeAccessIssue=false

from typing import Annotated

from pydantic import BaseModel, GetPydanticSchema
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
    import torch
    from lightning.pytorch.utilities import move_data_to_device

    class ToDevice(BaseModel):
        """A custom type for data that can be moved to a device using to().

        This is useful for e.g. letting Pytorch Lightning be able to move the
        data to the correct device automatically when training on a GPU.

        Warning: The to() does not deep copy the object, it only moves inner
        attributes that themselves have a to() method to the specified device.
        Some inner attributes might still be shared between the original and
        moved object.

        Note: This class is only available if Pytorch Lightning is installed.

        Note: This class will automatically recognize which attributes are
        movable using to() and which are not. If an attribute is not movable,
        it will be left as is. The class will also automatically infer the
        model fields and correctly handle alias fields defined using
        Field(..., alias=...).
        """

        def to(self, device: torch.device | str | int) -> "ToDevice":
            """Move the data to the specified device.

            Args:
                device: The device to move the data to.

            Returns:
                Data with the inner attributes moved to the specified device.
            """
            return self.model_copy(
                update={
                    k: move_data_to_device(getattr(self, k), device)
                    for k in self.model_fields
                }
            )

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
    import pytorch3d.structures

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

try:
    import geopandas

    GeoSeriesAnn = Annotated[
        geopandas.GeoSeries,
        GetPydanticSchema(
            lambda tp, handler: core_schema.is_instance_schema(
                geopandas.GeoSeries
            )
        ),
    ]
    GeoDataFrameAnn = Annotated[
        geopandas.GeoDataFrame,
        GetPydanticSchema(
            lambda tp, handler: core_schema.is_instance_schema(
                geopandas.GeoDataFrame
            )
        ),
    ]
except ImportError:
    pass
