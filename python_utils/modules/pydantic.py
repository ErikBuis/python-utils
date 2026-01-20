# pyright: reportMissingImports=false, reportAttributeAccessIssue=false

from __future__ import annotations

from abc import ABC
from collections.abc import Callable
from typing import Annotated, Any, TypeVar, get_args

from pydantic import BaseModel, GetPydanticSchema, PlainSerializer
from pydantic_core import core_schema

try:
    import numpy as np
    import numpy.typing as npt

    def _ndarray_schema(
        tp: type[npt.NDArray[Any]],
        handler: Callable[[Any], core_schema.CoreSchema],
    ) -> core_schema.CoreSchema:
        """Schema for typed numpy arrays with deserialization support.

        Validates both that the value is an np.ndarray and that its dtype
        matches the type parameter (if specified).

        Notes:
        - Uses np.issubdtype() to check dtype compatibility at runtime.
        - For NDArrayAnnTyped without type args, only validates np.ndarray.
        - Deserialization is handled via the
          core_schema.with_info_before_validator_function().

        Args:
            tp: The annotated type (e.g. npt.NDArray[np.float64]).
            handler: Pydantic schema handler (not used here).

        Returns:
            CoreSchema for validating the typed numpy array.

        Examples:
        >>> NDArrayAnnTyped = Annotated[
        ...     npt.NDArray[T], GetPydanticSchema(_ndarray_schema)
        ... ]

        >>> class M(BaseModel):
        ...     attribute1: NDArrayAnnTyped[np.float64]
        ...     attribute2: NDArrayAnnTyped

        >>> model = M(
        ...     attribute1=np.array([1.0, 2.0], dtype=np.float64),
        ...     attribute2=np.array([1, 2, 3])
        ... )  # succeeds

        >>> model = M(  # doctest: +IGNORE_EXCEPTION_DETAIL
        ...     attribute1=np.array([1, 2, 3], dtype=np.int32),
        ...     attribute2=np.array([1, 2, 3])
        ... )  # fails validation for attribute1
        Traceback (most recent call last):
            ...
        pydantic_core._pydantic_core.ValidationError: 1 validation error for M
        """
        # For npt.NDArray[T], get_args returns (Any, dtype[T]).
        ndarray_args = get_args(tp)
        assert len(ndarray_args) == 2, "Unexpected NDArray structure"
        dtype_arg = ndarray_args[1]

        # For dtype[T], get_args returns (T,).
        dtype_args = get_args(dtype_arg)
        assert len(dtype_args) == 1, "Unexpected dtype structure"
        type_arg = dtype_args[0]

        def _deserialize_ndarray(value: Any) -> np.ndarray:
            """Deserialize input to a numpy ndarray.

            Args:
                value: The input value to deserialize.

            Returns:
                The deserialized numpy ndarray.
            """
            if isinstance(value, np.ndarray):
                return value
            return np.array(value)

        def _validate_ndarray(value: Any) -> np.ndarray:
            """Validate that value is an ndarray with a correct dtype.

            Args:
                value: The value to validate.

            Returns:
                The validated ndarray if validation passes.
            """
            if not isinstance(value, np.ndarray):
                raise ValueError(f"Expected np.ndarray, but got {type(value)}")

            # Check if dtype matches the expected type.
            if not np.issubdtype(value.dtype, type_arg):
                raise ValueError(
                    "Expected array with dtype compatible with"
                    f" {type_arg.__name__}, but got {value.dtype}"
                )

            return value

        # Check if the given value is a TypeVar (this will also be the case if
        # the user did not specify a type argument). If so, we fall through to
        # the basic ndarray check.
        if isinstance(type_arg, TypeVar):
            # Deserialization + basic instance check.
            return core_schema.chain_schema([
                core_schema.no_info_plain_validator_function(
                    _deserialize_ndarray
                ),
                core_schema.is_instance_schema(np.ndarray),
            ])

        # Check if it's an actual numpy dtype (not an unrelated class such as
        # str or dict).
        try:
            assert issubclass(type_arg, np.generic)
        except (TypeError, AssertionError):
            raise TypeError(
                f"Invalid numpy dtype for {tp.__name__}: {type_arg} is not a"
                " subclass of np.generic"
            )

        # Deserialization + dtype validation.
        return core_schema.chain_schema([
            core_schema.no_info_plain_validator_function(_deserialize_ndarray),
            core_schema.no_info_plain_validator_function(_validate_ndarray),
        ])

    NDArrayAnn = Annotated[
        npt.NDArray,
        GetPydanticSchema(_ndarray_schema),
        PlainSerializer(lambda x: x.tolist(), return_type=list),
    ]
except ImportError:
    pass


try:
    import pandas as pd

    DataFrameAnn = Annotated[
        pd.DataFrame,
        GetPydanticSchema(
            lambda tp, handler: core_schema.is_instance_schema(pd.DataFrame)
        ),
    ]
    SeriesAnn = Annotated[
        pd.Series,
        GetPydanticSchema(
            lambda tp, handler: core_schema.is_instance_schema(pd.Series)
        ),
    ]
except ImportError:
    pass


try:
    import geopandas as gpd

    GeoSeriesAnn = Annotated[
        gpd.GeoSeries,
        GetPydanticSchema(
            lambda tp, handler: core_schema.is_instance_schema(gpd.GeoSeries)
        ),
    ]
    GeoDataFrameAnn = Annotated[
        gpd.GeoDataFrame,
        GetPydanticSchema(
            lambda tp, handler: core_schema.is_instance_schema(gpd.GeoDataFrame)
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
    from .pytorch3d import Transform3D

    Transform3DAnn = Annotated[
        Transform3D,
        GetPydanticSchema(
            lambda tp, handler: core_schema.is_instance_schema(Transform3D)
        ),
    ]
except ImportError:
    pass


try:
    import torch
    from lightning.pytorch.utilities import move_data_to_device

    try:
        from typing import Self
    except ImportError:
        from typing_extensions import Self

    class ToDevice(BaseModel, ABC):
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

        def to(self, device: torch.device | str | int) -> Self:
            """Move the data to the specified device.

            Args:
                device: The device to move the data to.

            Returns:
                Data with the inner attributes moved to the specified device.
            """
            return self.model_copy(
                update={
                    k: move_data_to_device(getattr(self, k), device)
                    for k in self.__class__.model_fields
                }
            )

except ImportError:
    pass
