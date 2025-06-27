from __future__ import annotations

from collections.abc import Callable
from typing import Literal, TypeVar, cast, overload

import numpy as np
import numpy.typing as npt
import pandas as pd

NPGeneric = TypeVar("NPGeneric", bound=np.generic)


@overload
def create_func_values2idcs(
    values_unique: npt.NDArray[NPGeneric],
    handle_missing_values: Literal[False] = ...,
) -> Callable[[npt.NDArray[NPGeneric]], npt.NDArray[np.int64]]:
    pass


@overload
def create_func_values2idcs(
    values_unique: npt.NDArray[NPGeneric],
    handle_missing_values: Literal[True] = ...,
) -> Callable[[npt.NDArray[NPGeneric]], npt.NDArray[np.int64 | np.float64]]:
    pass


def create_func_values2idcs(
    values_unique: npt.NDArray[NPGeneric], handle_missing_values: bool = False
) -> Callable[[npt.NDArray[NPGeneric]], npt.NDArray[np.int64 | np.float64]]:
    """Create a function that maps each value to its index in the given array.

    Args:
        values_unique: The values to map to indices. Must be unique.
            Shape: [U]
        handle_missing_values: Whether to map a value NOT present in the given
            unique values to np.nan (True) or perform undefined behaviour
            (False). In the former case, note that the function will return a
            float array instead of an integer array. The latter option is
            enabled by default since it is a lot faster, but if a value is
            missing anyway, the function will do one of the following things
            arbitrarily:
            - Raise an IndexError or a KeyError.
            - Return an arbitrary integer.
            So please make sure that if you set this argument to False, the
            values you pass to the function are always present in the original
            unique values array.

    Returns:
        A function that maps each value to its index in the given array.
            Args:
                x: The values to map. Does not need to be unique.
                    Shape: [N]
            Returns:
                The indices of the values in the original values_unique array.
                    Will always range from 0 to U - 1. If a value is not
                    present in the given unique values, the function will
                    either return np.nan or perform undefined behaviour,
                    depending on whether missing values are handled or not.
                    Shape: [N]
    """
    if (
        not handle_missing_values
        and np.issubdtype(values_unique.dtype, np.integer)
        and len(values_unique) > 0
    ):
        values_unique_int = cast(npt.NDArray[np.integer], values_unique)
        # If the values are integers, we can use a more efficient method for
        # remapping. However, this will cost more memory if the values are not
        # densely packed. Therefore, we only use this method if the values are
        # not larger than 100M (which will cost 800MB of RAM).
        min_value = np.min(values_unique_int)
        max_value = np.max(values_unique_int)
        if max_value - min_value + 1 <= 100_000_000:
            value2idx = np.empty(max_value - min_value + 1, dtype=np.int64)
            value2idx[values_unique_int - min_value] = np.arange(
                len(values_unique_int)
            )
            return lambda x: value2idx[
                cast(npt.NDArray[np.integer], x) - min_value
            ]

    # If the values are not integers or missing values should be handled, we
    # will use a dictionary for remapping. Note that pd.Series.map() will
    # return np.nan automatically if a value is not present in the dictionary.
    value2idx = {value: idx for idx, value in enumerate(values_unique)}
    return lambda x: pd.Series(x).map(value2idx).to_numpy()


@overload
def remap_series_to_idcs(
    series: pd.Series,
    values_unique: npt.NDArray[NPGeneric] = ...,  # type: ignore
) -> tuple[
    npt.NDArray[NPGeneric],
    Callable[[npt.NDArray[NPGeneric]], npt.NDArray[np.int64]],
    npt.NDArray[np.int64],
]:
    pass


@overload
def remap_series_to_idcs(
    series: pd.Series, values_unique: None = ...
) -> tuple[
    npt.NDArray[NPGeneric],
    Callable[[npt.NDArray[NPGeneric]], npt.NDArray[np.int64 | np.float64]],
    npt.NDArray[np.int64 | np.float64],
]:
    pass


def remap_series_to_idcs(
    series: pd.Series, values_unique: npt.NDArray[NPGeneric] | None = None
) -> tuple[
    npt.NDArray[NPGeneric],
    Callable[[npt.NDArray[NPGeneric]], npt.NDArray[np.int64 | np.float64]],
    npt.NDArray[np.int64 | np.float64],
]:
    """Map each unique value in a series to an index.

    Args:
        series: The series to remap.
            Length: N
        values_unique: The unique values in the series.
            If None, the unique values will be calculated from the series.
            This argument is useful for reusing the same mapping.
            Shape: [U]

    Returns:
        Tuple containing:
        - The unique values in the series.
            Shape: [U]
        - A function that maps each unique value to an index.
        - The series as an array with its values replaced by their
            corresponding indices. If values_unique is given but there are
            values in the series that are not present in the unique values,
            the function will return a float array instead of an integer array,
            where np.nan will be used to represent missing values. If
            values_unique is not given, the function will always return an
            integer array since the unique values will always be calculated
            from the series internally.
            Shape: [N]
    """
    if values_unique is None:
        values_unique = series.unique()
        value2idx = create_func_values2idcs(values_unique)
    else:
        value2idx = create_func_values2idcs(values_unique, True)
    values_remapped = value2idx(series.to_numpy())
    return values_unique, value2idx, values_remapped
