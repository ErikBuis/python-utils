from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from ..modules.numpy import NpGeneric, NpInteger

# ################################## PADDING ###################################


# Unlike with torch, in numpy we can't cache this function because numpy arrays
# are not hashable.
def mask_padding_batched(
    L_bs: npt.NDArray[np.integer], max_L_bs: int
) -> npt.NDArray[np.bool_]:
    """Create a mask that indicates which values are valid in each sample.

    Args:
        L_bs: The number of valid values in each sample.
            Shape: [B]
        max_L_bs: The maximum number of values of any element in the batch.
            Must be equal to max(L_bs).

    Returns:
        A mask that indicates which values are valid in each sample.
        mask[b, i] is True if i < L_bs[b] and False otherwise.
            Shape: [B, max(L_bs)]
    """
    dtype = L_bs.dtype

    return (
        np.arange(max_L_bs, dtype=dtype)  # [max(L_bs)]
        < np.expand_dims(L_bs, 1)  # [B, 1]
    )  # [B, max(L_bs)]  # fmt: skip


def pack_padded_batched(
    values: npt.NDArray[NpGeneric], L_bs: npt.NDArray[np.integer]
) -> npt.NDArray[NpGeneric]:
    """Pack a batch of padded values into a single array.

    Args:
        values: The values to pack. Padding could be arbitrary.
            Shape: [B, max(L_bs), *]
        L_bs: The number of valid values in each sample.
            Shape: [B]

    Returns:
        The packed values.
            Shape: [L, *]
    """
    max_L_bs = values.shape[1]
    mask = mask_padding_batched(L_bs, max_L_bs)  # [B, max(L_bs)]
    return values[mask]


def pad_packed_batched(
    values: npt.NDArray[NpGeneric],
    L_bs: npt.NDArray[np.integer],
    max_L_bs: int,
    padding_value: Any = None,
) -> npt.NDArray[NpGeneric]:
    """Pad a batch of packed values to create an array with a fixed size.

    Args:
        values: The values to pad.
            Shape: [L, *]
        L_bs: The number of valid values in each sample.
            Shape: [B]
        max_L_bs: The maximum number of values of any element in the batch.
            Must be equal to max(L_bs).
        padding_value: The value to pad the values with. If None, the values
            are padded with random values. This is faster than padding with
            a specific value.

    Returns:
        The padded values. Padded with padding_value.
            Shape: [B, max(L_bs), *]
    """
    B = len(L_bs)
    dtype = values.dtype

    padded_shape = (B, max_L_bs, *values.shape[1:])
    if padding_value is None:
        values_padded = np.empty(padded_shape, dtype=dtype)
    elif padding_value == 0:
        values_padded = np.zeros(padded_shape, dtype=dtype)
    elif padding_value == 1:
        values_padded = np.ones(padded_shape, dtype=dtype)
    else:
        values_padded = np.full(padded_shape, padding_value, dtype=dtype)

    mask = mask_padding_batched(L_bs, max_L_bs)  # [B, max(L_bs)]
    values_padded[mask] = values

    return values_padded


def pad_sequence_batched(
    values: Sequence[npt.NDArray[NpGeneric]],
    L_bs: npt.NDArray[np.integer],
    max_L_bs: int,
    padding_value: Any = None,
) -> npt.NDArray[NpGeneric]:
    """Pad a batch of sequences to create an array with a fixed size.

    This function is equivalent to torch.nn.utils.rnn.pad_sequence(), but
    surprisingly it is a bit faster, even if the padding value is not set to
    None! And if the padding value is set to None, the function will be even
    faster, since it will not need to overwrite the allocated memory. The former
    is because torch.nn.utils.rnn.pad_sequence() performs some extra checks that
    we skip here. It is also a bit more flexible, since it allows for the batch
    dimension to be at axis=1 instead of axis=0.

    In conclusion, you should almost always use this function instead.

    Args:
        values: The sequence values to pad. Padding could be arbitrary.
            Length: B
            Shape of inner arrays: [L_b, *]
        L_bs: The number of valid values in each sample.
            Shape: [B]
        max_L_bs: The maximum number of values of any element in the batch.
            Must be equal to max(L_bs).
        padding_value: The value to pad the values with. If None, the values
            are padded with random values. This is faster than padding with
            a specific value.

    Returns:
        The padded values. Padded with padding_value.
            Shape: [B, max(L_bs), *]
    """
    return pad_packed_batched(
        np.concat(values), L_bs, max_L_bs, padding_value=padding_value
    )  # [B, max(L_bs), *]


def replace_padding_batched(
    values: npt.NDArray[NpGeneric],
    L_bs: npt.NDArray[np.integer],
    padding_value: Any = 0,
    in_place: bool = False,
) -> npt.NDArray[NpGeneric]:
    """Pad the values with padding_value to create an array with a fixed size.

    Args:
        values: The values to pad. Padding could be arbitrary.
            Shape: [B, max(L_bs), *]
        L_bs: The number of valid values in each sample.
            Shape: [B]
        padding_value: The value to pad the values with. Can be one of:
            - A scalar value, or an array of shape [] or [*], containing the
              value to pad all elements with.
            - An array of shape [B, max(L_bs)] or [B, max(L_bs), *], containing
              the value to pad each element with.
            - An array of shape [B, 1] or [B, 1, *], containing the value to
              pad each row with.
            - An array of shape [1, max(L_bs)] or [1, max(L_bs), *], containing
              the value to pad each column with.
            - An array of shape [B * max(L_bs) - L] or [B * max(L_bs) - L, *],
              containing the value to pad each element in the padding mask with.
        in_place: Whether to perform the operation in-place.

    Returns:
        The padded values. Padded with padding_value.
            Shape: [B, max(L_bs), *]
    """
    B, max_L_bs, *star = values.shape
    mask = mask_padding_batched(L_bs, max_L_bs)  # [B, max(L_bs)]
    values_padded = values if in_place else values.copy()

    # Handle the case where padding_value is a scalar.
    if not isinstance(padding_value, np.ndarray):
        values_padded[~mask] = padding_value
        return values_padded

    # Handle shapes that are missing the star dimensions.
    if (
        padding_value.shape == ()
        or padding_value.shape == (B, max_L_bs)
        or padding_value.shape == (B, 1)
        or padding_value.shape == (1, max_L_bs)
        or padding_value.shape == (B * max_L_bs - L_bs.sum(),)
    ):
        padding_value = padding_value.reshape(B, max_L_bs, *[1] * len(star))

    # Now handle all expected shapes.
    # Note that the shape of the value to be set (values_padded[~mask]) is
    # always [B * max_L_bs - L_bs.sum(), *].
    if padding_value.shape == tuple(star):
        values_padded[~mask] = padding_value
    elif padding_value.shape == (B, max_L_bs, *star):
        values_padded[~mask] = padding_value[~mask]
    elif padding_value.shape == (B, 1, *star):
        values_padded[~mask] = padding_value.squeeze(1).repeat(
            max_L_bs - L_bs, axis=0
        )
    elif padding_value.shape == (1, max_L_bs, *star):
        values_padded[~mask] = np.broadcast_to(
            padding_value, (B, max_L_bs, *star)
        )[~mask]
    elif padding_value.shape == (B * max_L_bs - L_bs.sum(), *star):
        values_padded[~mask] = padding_value
    else:
        raise ValueError(
            "Shape of padding_value did not match any of the expected shapes."
            f" Got {list(padding_value.shape)}, but expected one of: [],"
            f" {star}, {[B, max_L_bs]}, {[B, max_L_bs, *star]}, {[B, 1]},"
            f" {[B, 1, *star]}, {[1, max_L_bs]},  {[1, max_L_bs, *star]},"
            f" {[B * max_L_bs - L_bs.sum()]}, or"
            f" {[B * max_L_bs - L_bs.sum(), *star]}"
        )

    return values_padded


def last_valid_value_padding_batched(
    values: npt.NDArray[NpGeneric],
    L_bs: npt.NDArray[np.integer],
    padding_value_empty_rows: Any = None,
    in_place: bool = False,
) -> npt.NDArray[NpGeneric]:
    """Pad the values with the last valid value for each sample.

    Args:
        values: The values to pad. Padding could be arbitrary.
            Shape: [B, max(L_bs), *]
        L_bs: The number of valid values in each sample.
            Shape: [B]
        padding_value_empty_rows: The value to pad empty rows with (i.e. when
            L_b == 0 for some b). If None, empty rows are padded with random
            values. This is faster than padding with a specific value.
        in_place: Whether to perform the operation in-place.

    Returns:
        The padded values. Padded with the last valid value.
            Shape: [B, max(L_bs), *]
    """
    B = len(L_bs)
    arange_B = np.arange(B)
    padding_value = values[arange_B, L_bs - 1]  # [B, *]
    if padding_value_empty_rows is not None:
        padding_value = padding_value.copy()
        padding_value[L_bs == 0] = padding_value_empty_rows
    padding_value_new = np.expand_dims(padding_value, 1)  # [B, 1, *]
    values = replace_padding_batched(
        values, L_bs, padding_value=padding_value_new, in_place=in_place
    )  # [B, max(L_bs), *]
    return values


def apply_mask_batched(
    values: npt.NDArray[NpGeneric],
    mask: npt.NDArray[np.bool_],
    L_bs: npt.NDArray[NpInteger],
    padding_value: Any = None,
) -> tuple[npt.NDArray[NpGeneric], npt.NDArray[NpInteger]]:
    """Apply an additional mask to a batch of values.

    All values that are not marked for removal by the mask will be kept, while
    the remaining values will be moved to the front of the array. Since the
    number of kept values in each sample can change, the function will also
    return the number of kept values in each sample.

    Args:
        values: The values to apply the mask to. Padding could be arbitrary.
            Shape: [B, max(L_bs), *]
        mask: The mask to apply. Padding could be arbitrary, since the function
            will apply the original mask internally in addition to the given
            mask.
            Shape: [B, max(L_bs)]
        L_bs: The number of valid values in each sample.
            Shape: [B]
        padding_value: The value to pad the values with. If None, the values
            are padded with random values. This is faster than padding with
            a specific value.

    Returns:
        Tuple containing:
        - The values with the given mask applied. Padded with padding_value.
            Shape: [B, max(L_bs_kept), *]
        - The number of kept values in each sample.
            Shape: [B]
    """
    # Create a mask that indicates which values should be kept in each sample.
    max_L_bs = values.shape[1]
    mask = mask & mask_padding_batched(L_bs, max_L_bs)  # [B, max(L_bs)]
    L_bs_kept = mask.sum(axis=1)  # [B]
    max_L_bs_kept = int(L_bs_kept.max())

    # Move the masked values to the front of the array.
    values = pad_packed_batched(
        values[mask], L_bs_kept, max_L_bs_kept, padding_value=padding_value
    )  # [B, max(L_bs_kept), *]

    return values, L_bs_kept
