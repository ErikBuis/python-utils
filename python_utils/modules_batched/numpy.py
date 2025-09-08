from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

# ######################### NUMPY & TORCH SHARED UTILS #########################


# Unlike with torch, in numpy we can't cache this function because numpy arrays
# are not hashable.
def mask_padding_batched(L_bs: npt.NDArray, max_L_bs: int) -> npt.NDArray:
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


def pack_padded_batched(values: npt.NDArray, L_bs: npt.NDArray) -> npt.NDArray:
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
    values: npt.NDArray,
    L_bs: npt.NDArray,
    max_L_bs: int,
    padding_value: Any = None,
) -> npt.NDArray:
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
    values: Sequence[npt.NDArray],
    L_bs: npt.NDArray,
    max_L_bs: int,
    padding_value: Any = None,
) -> npt.NDArray:
    """Pad a batch of sequences to create an array with a fixed size.

    This function is equivalent to torch.nn.utils.rnn.pad_sequence(), but
    surprisingly it is a bit faster, even if the padding value is not set to
    None! And if the padding value is set to None, the function will be even
    faster, since it will not need to overwrite the allocated memory. The former
    is because torch.nn.utils.rnn.pad_sequence() performs some extra checks that
    we skip here. It is also a bit more flexible, since it allows for the batch
    dimension to be at dim=1 instead of dim=0.

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
