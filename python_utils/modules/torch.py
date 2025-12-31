from __future__ import annotations

import random
from functools import lru_cache
from math import prod
from typing import Any, Callable, Literal, overload

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from torch.utils.data._utils.collate import (
    collate_str_fn,
    default_collate_fn_map,
)
from torch.utils.data.dataloader import default_collate

# ################################## COLLATE ###################################

# Allow the dataset to return None when a sample is corrupted. When it does,
# make torch's default collate function replace it with another sample.
default_collate_fn_map.update({type(None): collate_str_fn})


def collate_replace_corrupted(
    batch_list: list[Any],
    dataset: Dataset | Subset,
    collate_fn: Callable | None = None,
) -> Any:
    """Collate function that allows to replace corrupted samples in the batch.

    The given dataset should return None when a sample is corrupted. This
    function will then replace such a sample with another randomly selected
    sample from the dataset. A RuntimeError will be raised when all samples in
    the dataset are corrupted.

    Warning: It is recommended to keep track of corrupted samples in your
    Dataset class, so that you can immediately skip them when __getitem__() is
    called instead of having to perform potentially expensive loading/
    preprocessing on these samples again. To do this, you can maintain a set of
    corrupted sample indices in your Dataset class as follows:
    >>> class MyDataset(Dataset):
    ...     def __init__(self):
    ...         self.corrupted_samples = set()
    ...     def __getitem__(self, idx):
    ...         if idx in self.corrupted_samples:
    ...             return
    ...         sample = ...  # load and preprocess sample
    ...         if is_corrupted(sample):
    ...             self.corrupted_samples.add(idx)
    ...             return
    ...         return sample

    Furthermore, if you do decide to provide the `corrupted_samples` attribute,
    this function will use that information to avoid resampling corrupted
    samples, which can significantly speed up the replacement process.

    Warning: Since corrupted samples are replaced with random other samples from
    the dataset, a sample might be sampled multiple times in one pass through
    the dataloader (or even in one batch). This implies that a model might be
    trained on the same sample multiple times in one epoch or batch.

    Warning: Since a DataLoader only accepts collate functions with a single
    argument, you should use functools.partial() to pass your dataset object and
    your default collate function to this function. For example:
    >>> from functools import partial
    >>> from torch.utils.data import DataLoader
    >>> dataset = MyDataset()
    >>> collate_fn = partial(collate_replace_corrupted, dataset=dataset)
    >>> dataloader = DataLoader(dataset, collate_fn=collate_fn)

    This function was based on:
    https://stackoverflow.com/a/69578320/15636460

    Args:
        batch_list: List of samples from the DataLoader.
        dataset: Dataset or Subset that the DataLoader is passing through.
        collate_fn: The function to call once the batch has no corrupted samples
            any more. If None, torch.utils.data.dataloader.default_collate() is
            called.

    Returns:
        Batch with new samples instead of corrupted ones.
    """
    # Use torch.utils.data.dataloader.default_collate() if no other collate
    # function is specified.
    collate_fn = default_collate if collate_fn is None else collate_fn

    # Filter out all corrupted samples.
    B = len(batch_list)
    batch_list = [sample for sample in batch_list if sample is not None]

    # Replace the corrupted samples with other randomly selected samples.
    if len(batch_list) != B:
        D = len(dataset)  # type: ignore
        potential_samples = set(range(D))

        # If the dataset has a `corrupted_samples` attribute, use it to avoid
        # resampling corrupted samples.
        corrupted_samples: set[int] | None = getattr(
            dataset.dataset if isinstance(dataset, Subset) else dataset,
            "corrupted_samples",
            None,
        )
        if corrupted_samples is not None:
            potential_samples -= corrupted_samples

        # Keep sampling until we have a full batch.
        while len(batch_list) != B:
            if len(potential_samples) == 0:
                raise RuntimeError(
                    f"All samples in the {type(dataset)} are corrupted."
                )

            idx = random.choice(list(potential_samples))
            sample = dataset[idx]
            if sample is not None:
                batch_list.append(sample)
            else:
                potential_samples.remove(idx)

    # When the whole batch is fine, apply the default collate function.
    return collate_fn(batch_list)


# ########################### PACK/PAD/SEQUENTIALIZE ###########################

# Functions for packing, padding, and sequentializing torch tensors.
#
# Sometimes, a batch of samples contains variable-length data. For example, in a
# batch of sequences, each sequence could have a different length. To convert
# between the above-mentioned representations, we provide several functions.
#
# These functions use the following naming conventions:
# - L_bs has shape [B] (batch)
# - L_bs[b]    : L_b
# - L_bs.sum() : L
# - L_bs.max() : max_L_bs or max(L_bs)
#
# The *_multidim() functions in this module are a generalization of this concept
# to multiple dimensions. For example, in a batch of images, each image could
# have a different height and width. The multi-dimensional padding and packing
# functions allow us to handle such cases.
#
# These *_multidim() functions use the following naming conventions:
# - L_bsds has shape [B, D] (batch x dimension)
# - L_bsds[:, d]              : L_bsd
# - L_bsds[:, d].sum()        : L_d
# - L_bsds[b, :]              : L_bds
# - L_bsds[b, d]              : L_bd
# - L_bsds.prod(dim=1)       : L_bs
# - L_bsds.prod(dim=1)[b]    : L_b
# - L_bsds.prod(dim=1).sum() : L
# - L_bsds.prod(dim=1).max() : max_L_bs or max(L_bs)
# - L_bsds.max(dim=0)        : max_L_bsds or max(L_bsds)
# - L_bsds.max(dim=0)[d]     : max_L_bsd or max(L_bsd)
# - L_bsds.max(dim=0).prod() : theory_max_L_bs or prod(max(L_bsds))
#
# Note on max_L_bs vs theory_max_L_bs:
# max_L_bs gives the maximum number of values of an actual sample, while
# theory_max_L_bs gives the theoretical maximum number of values in any sample.
# The latter can be larger than max_L_bs in certain cases, such as when one
# sample has shape [10, 10] and another sample has shape [20, 5]. In this case,
# theory_max_L_bs = 200, but max_L_bs = 100.


# Warning: The lru_cache caches inputs by memory address, so if you call this
# function with different tensors that have the same values, it will not
# recognize them as the same input. On the other hand, if you call this function
# again with the same tensor after doing an in-place operation on it, it will
# recognize it as the same input, even if the values have changed. This can
# cause confusion, so be careful when using this function with tensors that
# might change in-place.
@lru_cache(maxsize=8)
def mask_padding(L_bs: torch.Tensor, max_L_bs: int) -> torch.Tensor:
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

    Examples:
    >>> L_bs = torch.tensor([3, 1, 0, 4])
    >>> max_L_bs = 4
    >>> mask = mask_padding(L_bs, max_L_bs)
    >>> mask
    tensor([[ True,  True,  True, False],
            [ True, False, False, False],
            [False, False, False, False],
            [ True,  True,  True,  True]])
    """
    device = L_bs.device
    dtype = L_bs.dtype

    return (
        torch.arange(max_L_bs, device=device, dtype=dtype)  # [max(L_bs)]
        < L_bs.unsqueeze(1)  # [B, 1]
    )  # [B, max(L_bs)]  # fmt: skip


def mask_padding_multidim(
    L_bsds: torch.Tensor, max_L_bsds: torch.Tensor
) -> torch.Tensor:
    """Create a multi-dimensional mask based on per-batch sizes.

    Args:
        L_bsds: The number of valid values in each sample for each dimension.
            Shape: [B, D]
        max_L_bsds: The maximum number of values of any element in the batch for
            each dimension. Must be equal to max(L_bsds).
            Shape: [D]

    Returns:
        A mask that indicates which values are valid in each sample.
        mask[b, i_0, ..., i_{D+1}] is True if i_d < L_bd for all d in [0, D)
        and False otherwise.
            Shape: [B, max(L_bs0), ..., max(L_bs{D-1})]

    Examples:
    >>> L_bsds = torch.tensor([
    ...     [2, 3],
    ...     [4, 1],
    ... ])
    >>> max_L_bsds = torch.tensor([4, 3])
    >>> mask = mask_padding_multidim(L_bsds, max_L_bsds)
    >>> mask
    tensor([[[ True,  True,  True],
             [ True,  True,  True],
             [False, False, False],
             [False, False, False]],
            [[ True, False, False],
             [ True, False, False],
             [ True, False, False],
             [ True, False, False]]])
    """
    B, D = L_bsds.shape
    device = L_bsds.device
    dtype = L_bsds.dtype

    # Create coordinate grids for each dimension.
    mask = torch.ones(
        (B, *max_L_bsds), device=device, dtype=torch.bool  # type: ignore
    )
    for d in range(D):
        # Create coordinates for dimension d.
        coords = torch.arange(
            max_L_bsds[d], device=device, dtype=dtype  # type: ignore
        )  # [max(L_bsd)]

        # Reshape coords to [1, 1, ..., max(L_bsd), ..., 1].
        unsqueeze_shape = (1, *[1] * d, max_L_bsds[d], *[1] * (D - d - 1))
        coords_unsqueezed = coords.reshape(unsqueeze_shape)

        # Reshape L_bsd to [B, 1, ..., 1] for broadcasting.
        unsqueeze_shape = (B, *[1] * D)
        L_bsd_unsqueezed = L_bsds[:, d].reshape(unsqueeze_shape)

        # Create the mask by comparing each coordinate with its limits.
        mask &= coords_unsqueezed < L_bsd_unsqueezed

    return mask


def pack_padded(values: torch.Tensor, L_bs: torch.Tensor) -> torch.Tensor:
    """Pack a batch of padded values into a single tensor.

    Args:
        values: The values to pack. Padding could be arbitrary.
            Shape: [B, max(L_bs), *]
        L_bs: The number of valid values in each sample.
            Shape: [B]

    Returns:
        The packed values.
            Shape: [L, *]

    Examples:
    >>> values = torch.tensor([
    ...     [1, 2, 3, 0],
    ...     [5, 0, 0, 0],
    ...     [0, 0, 0, 0],
    ...     [13, 14, 15, 16],
    ... ])
    >>> L_bs = torch.tensor([3, 1, 0, 4])
    >>> packed = pack_padded(values, L_bs)
    >>> packed
    tensor([ 1,  2,  3,  5, 13, 14, 15, 16])
    """
    max_L_bs = values.shape[1]
    mask = mask_padding(L_bs, max_L_bs)  # [B, max(L_bs)]
    return values[mask]


def pack_padded_multidim(
    values: torch.Tensor, L_bsds: torch.Tensor
) -> torch.Tensor:
    """Pack a batch of multi-dimensionally padded values into a single tensor.

    Args:
        values: The values to pack. Padding could be arbitrary.
            Shape: [B, max(L_bs0), ..., max(L_bs{D-1}), *]
        L_bsds: The number of valid values in each sample for each dimension.
            Shape: [B, D]

    Returns:
        The packed values.
            Shape: [L, *]

    Examples:
    >>> values = torch.tensor([
    ...     [
    ...         [1, 2, 3],
    ...         [4, 5, 6],
    ...         [0, 0, 0],
    ...         [0, 0, 0],
    ...     ],
    ...     [
    ...         [13, 0, 0],
    ...         [16, 0, 0],
    ...         [19, 0, 0],
    ...         [22, 0, 0],
    ...     ],
    ... ])
    >>> L_bsds = torch.tensor([
    ...     [2, 3],
    ...     [4, 1],
    ... ])
    >>> packed = pack_padded_multidim(values, L_bsds)
    >>> packed
    tensor([ 1,  2,  3,  4,  5,  6, 13, 16, 19, 22])
    """
    D = L_bsds.shape[1]
    device = L_bsds.device

    max_L_bsds = torch.tensor(values.shape[1 : D + 1], device=device)  # [D]
    mask = mask_padding_multidim(
        L_bsds, max_L_bsds
    )  # [B, max(L_bs0), ..., max(L_bs{D-1})]
    return values[mask]


def pack_sequence(
    values: tuple[torch.Tensor, ...] | list[torch.Tensor], max_L_bs: int
) -> torch.Tensor:
    """Pack a batch of sequences into a single tensor.

    Args:
        values: The sequence of values to pack.
            Length: B
            Shape of inner tensors: [L_b, *]
        max_L_bs: The maximum number of values of any element in the batch.
            Must be equal to max(L_bs).

    Returns:
        The packed values. If the input is empty, returns an empty tensor with
        dtype float32.
            Shape: [L, *]

    Examples:
    >>> values = [
    ...     torch.tensor([1, 2, 3]),
    ...     torch.tensor([5]),
    ...     torch.tensor([], dtype=torch.int64),
    ...     torch.tensor([13, 14, 15, 16]),
    ... ]
    >>> max_L_bs = 4
    >>> packed = pack_sequence(values, max_L_bs)
    >>> packed
    tensor([ 1,  2,  3,  5, 13, 14, 15, 16])
    """
    B = len(values)
    if B == 0:
        return torch.empty(0, dtype=torch.float32)

    return torch.concat(values)  # [L, *]


def pack_sequence_multidim(
    values: tuple[torch.Tensor, ...] | list[torch.Tensor],
    max_L_bsds: torch.Tensor,
) -> torch.Tensor:
    """Pack a batch of multi-dimensional sequences into a single tensor.

    Args:
        values: The sequence of values to pack.
            Length: B
            Shape of inner tensors: [L_b0, ..., L_b{D-1}, *]
        max_L_bsds: The maximum number of values of any element in the batch for
            each dimension. Must be equal to max(L_bsds).
            Shape: [D]

    Returns:
        The packed values. If the input is empty, returns an empty tensor with
        dtype float32.
            Shape: [L, *]

    Examples:
    >>> values = [
    ...     torch.tensor([
    ...         [1, 2, 3],
    ...         [4, 5, 6],
    ...     ]),
    ...     torch.tensor([
    ...         [13],
    ...         [16],
    ...         [19],
    ...         [22],
    ...     ]),
    ... ]
    >>> max_L_bsds = torch.tensor([4, 3])
    >>> packed = pack_sequence_multidim(values, max_L_bsds)
    >>> packed
    tensor([ 1,  2,  3,  4,  5,  6, 13, 16, 19, 22])
    """
    B = len(values)
    D = len(max_L_bsds)
    if B == 0:
        return torch.empty(0, dtype=torch.float32)

    return torch.concat([
        values[b].reshape((-1, *values[b].shape[D:]))  # [L_b, *]
        for b in range(B)
    ])  # [L, *]


def pad_packed(
    values: torch.Tensor,
    L_bs: torch.Tensor,
    max_L_bs: int,
    padding_value: Any = None,
) -> torch.Tensor:
    """Pad a batch of packed values to create a tensor with a fixed size.

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

    Examples:
    >>> values = torch.tensor([1, 2, 3, 5, 13, 14, 15, 16])
    >>> L_bs = torch.tensor([3, 1, 0, 4])
    >>> max_L_bs = 4
    >>> padded = pad_packed(values, L_bs, max_L_bs, padding_value=0)
    >>> padded
    tensor([[ 1,  2,  3,  0],
            [ 5,  0,  0,  0],
            [ 0,  0,  0,  0],
            [13, 14, 15, 16]])
    """
    B = len(L_bs)
    star = values.shape[1:]
    device = values.device
    dtype = values.dtype

    padded_shape = (B, max_L_bs, *star)
    if padding_value is None:
        values_padded = torch.empty(padded_shape, device=device, dtype=dtype)
    elif padding_value == 0:
        values_padded = torch.zeros(padded_shape, device=device, dtype=dtype)
    elif padding_value == 1:
        values_padded = torch.ones(padded_shape, device=device, dtype=dtype)
    else:
        values_padded = torch.full(
            padded_shape, padding_value, device=device, dtype=dtype
        )

    mask = mask_padding(L_bs, max_L_bs)  # [B, max(L_bs)]
    values_padded[mask] = values

    return values_padded


def pad_packed_multidim(
    values: torch.Tensor,
    L_bsds: torch.Tensor,
    max_L_bsds: torch.Tensor,
    padding_value: Any = None,
) -> torch.Tensor:
    """Pad a batch of multi-dimensionally packed values to create a tensor
    with a fixed size.

    Args:
        values: The values to pad.
            Shape: [L, *]
        L_bsds: The number of valid values in each sample for each dimension.
            Shape: [B, D]
        max_L_bsds: The maximum number of values of any element in the batch for
            each dimension. Must be equal to max(L_bsds).
            Shape: [D]
        padding_value: The value to pad the values with. If None, the values
            are padded with random values. This is faster than padding with
            a specific value.

    Returns:
        The padded values. Padded with padding_value.
            Shape: [B, max(L_bs0), ..., max(L_bs{D-1}), *]

    Examples:
    >>> values = torch.tensor([1, 2, 3, 4, 5, 6, 13, 16, 19, 22])
    >>> L_bsds = torch.tensor([
    ...     [2, 3],
    ...     [4, 1],
    ... ])
    >>> max_L_bsds = torch.tensor([4, 3])
    >>> padded = pad_packed_multidim(
    ...     values, L_bsds, max_L_bsds, padding_value=0
    ... )
    >>> padded
    tensor([[[ 1,  2,  3],
             [ 4,  5,  6],
             [ 0,  0,  0],
             [ 0,  0,  0]],
            [[13,  0,  0],
             [16,  0,  0],
             [19,  0,  0],
             [22,  0,  0]]])
    """
    B = len(L_bsds)
    star = values.shape[1:]
    device = values.device
    dtype = values.dtype

    padded_shape = (B, *max_L_bsds, *star)
    if padding_value is None:
        values_padded = torch.empty(
            padded_shape, device=device, dtype=dtype  # type: ignore
        )
    elif padding_value == 0:
        values_padded = torch.zeros(
            padded_shape, device=device, dtype=dtype  # type: ignore
        )
    elif padding_value == 1:
        values_padded = torch.ones(
            padded_shape, device=device, dtype=dtype  # type: ignore
        )
    else:
        values_padded = torch.full(
            padded_shape,  # type: ignore
            padding_value,
            device=device,
            dtype=dtype,
        )

    mask = mask_padding_multidim(
        L_bsds, max_L_bsds
    )  # [B, max(L_bs0), ..., max(L_bs{D-1})]
    values_padded[mask] = values

    return values_padded


def pad_sequence(
    values: tuple[torch.Tensor, ...] | list[torch.Tensor],
    L_bs: torch.Tensor,
    max_L_bs: int,
    padding_value: Any = None,
) -> torch.Tensor:
    """Pad a batch of sequences to create a tensor with a fixed size.

    This function is equivalent to torch.nn.utils.rnn.pad_sequence(), but
    surprisingly it is a bit faster, even if the padding value is not set to
    None! And if the padding value is set to None, the function will be even
    faster, since it will not need to overwrite the allocated memory. The former
    is because torch.nn.utils.rnn.pad_sequence() performs some extra checks that
    we skip here. It is also a bit more flexible, since it allows for the batch
    dimension to be at dim=1 instead of dim=0.

    In conclusion, you should almost always use this function instead.

    Args:
        values: The sequence of values to pad.
            Length: B
            Shape of inner tensors: [L_b, *]
        L_bs: The number of valid values in each sample.
            Shape: [B]
        max_L_bs: The maximum number of values of any element in the batch.
            Must be equal to max(L_bs).
        padding_value: The value to pad the values with. If None, the values are
            padded with random values. This is faster than padding with a
            specific value.

    Returns:
        The padded values. Padded with padding_value.
            Shape: [B, max(L_bs), *]

    Examples:
    >>> values = [
    ...     torch.tensor([1, 2, 3]),
    ...     torch.tensor([5]),
    ...     torch.tensor([], dtype=torch.int64),
    ...     torch.tensor([13, 14, 15, 16]),
    ... ]
    >>> L_bs = torch.tensor([3, 1, 0, 4])
    >>> max_L_bs = 4
    >>> padded = pad_sequence(values, L_bs, max_L_bs, padding_value=0)
    >>> padded
    tensor([[ 1,  2,  3,  0],
            [ 5,  0,  0,  0],
            [ 0,  0,  0,  0],
            [13, 14, 15, 16]])
    """
    return pad_packed(
        pack_sequence(values, max_L_bs),
        L_bs,
        max_L_bs,
        padding_value=padding_value,
    )  # [B, max(L_bs), *]


def pad_sequence_multidim(
    values: tuple[torch.Tensor, ...] | list[torch.Tensor],
    L_bsds: torch.Tensor,
    max_L_bsds: torch.Tensor,
    padding_value: Any = None,
) -> torch.Tensor:
    """Pad a batch of multi-dimensional sequences to create a tensor with a
    fixed size.

    This function is equivalent to torch.nn.utils.rnn.pad_sequence(), but
    surprisingly it is a bit faster, even if the padding value is not set to
    None! And if the padding value is set to None, the function will be even
    faster, since it will not need to overwrite the allocated memory. The former
    is because torch.nn.utils.rnn.pad_sequence() performs some extra checks that
    we skip here. It is also a bit more flexible, since it allows for the batch
    dimension to be at dim=1 instead of dim=0.

    In conclusion, you should almost always use this function instead.

    Args:
        values: The sequence of values to pad.
            Length: B
            Shape of inner tensors: [L_b0, ..., L_b{D-1}, *]
        L_bsds: The number of valid values in each sample for each dimension.
            Shape: [B, D]
        max_L_bsds: The maximum number of values of any element in the batch for
            each dimension. Must be equal to max(L_bsds).
            Shape: [D]
        padding_value: The value to pad the values with. If None, the values are
            padded with random values. This is faster than padding with a
            specific value.

    Returns:
        The padded values. Padded with padding_value.
            Shape: [B, max(L_bs0), ..., max(L_bs{D-1}), *]

    Examples:
    >>> values = [
    ...     torch.tensor([
    ...         [1, 2, 3],
    ...         [4, 5, 6],
    ...     ]),
    ...     torch.tensor([
    ...         [13],
    ...         [16],
    ...         [19],
    ...         [22],
    ...     ]),
    ... ]
    >>> L_bsds = torch.tensor([
    ...     [2, 3],
    ...     [4, 1],
    ... ])
    >>> max_L_bsds = torch.tensor([4, 3])
    >>> padded = pad_sequence_multidim(
    ...     values, L_bsds, max_L_bsds, padding_value=0
    ... )
    >>> padded
    tensor([[[ 1,  2,  3],
             [ 4,  5,  6],
             [ 0,  0,  0],
             [ 0,  0,  0]],
            [[13,  0,  0],
             [16,  0,  0],
             [19,  0,  0],
             [22,  0,  0]]])
    """
    return pad_packed_multidim(
        pack_sequence_multidim(values, max_L_bsds),
        L_bsds,
        max_L_bsds,
        padding_value=padding_value,
    )  # [B, max(L_bs0), ..., max(L_bs{D-1}), *]


def sequentialize_packed(
    values: torch.Tensor, L_bs: torch.Tensor
) -> list[torch.Tensor]:
    """Convert a batch of packed values to a sequence of values.

    Args:
        values: The packed values to convert.
            Shape: [L, *]
        L_bs: The number of valid values in each sample.
            Shape: [B]

    Returns:
        The values as a sequence.
            Length: B
            Shape of inner tensors: [L_b, *]

    Examples:
    >>> values = torch.tensor([1, 2, 3, 5, 13, 14, 15, 16])
    >>> L_bs = torch.tensor([3, 1, 0, 4])
    >>> sequentialized = sequentialize_packed(values, L_bs)
    >>> sequentialized
    [tensor([1, 2, 3]),
     tensor([5]),
     tensor([], dtype=torch.int64),
     tensor([13, 14, 15, 16])]
    """
    return list(torch.split(values, L_bs.tolist()))  # B x [L_b, *]


def sequentialize_packed_multidim(
    values: torch.Tensor, L_bsds: torch.Tensor
) -> list[torch.Tensor]:
    """Convert a batch of multi-dimensionally packed values to a sequence of
    values.

    Args:
        values: The packed values to convert.
            Shape: [L, *]
        L_bsds: The number of valid values in each sample for each dimension.
            Shape: [B, D]

    Returns:
        The values as a sequence.
            Length: B
            Shape of inner tensors: [L_b0, ..., L_b{D-1}, *]

    Examples:
    >>> values = torch.tensor([1, 2, 3, 4, 5, 6, 13, 16, 19, 22])
    >>> L_bsds = torch.tensor([
    ...     [2, 3],
    ...     [4, 1],
    ... ])
    >>> sequentialized = sequentialize_packed_multidim(values, L_bsds)
    >>> sequentialized
    [tensor([[1, 2, 3],
             [4, 5, 6]]),
     tensor([[13],
             [16],
             [19],
             [22]])]
    """
    star = values.shape[1:]
    L_bs = L_bsds.prod(dim=1)  # [B]
    values_split = torch.split(values, L_bs.tolist())  # B x [L_b, *]
    return [
        values_b.reshape(
            (*L_bsds[b], *star)  # type: ignore
        )  # [L_b0, ..., L_b{D-1}, *]
        for b, values_b in enumerate(values_split)
    ]


def sequentialize_padded(
    values: torch.Tensor, L_bs: torch.Tensor
) -> list[torch.Tensor]:
    """Convert a batch of padded values to a sequence of values.

    Args:
        values: The padded values to convert. Padding could be arbitrary.
            Shape: [B, max(L_bs), *]
        L_bs: The number of valid values in each sample.
            Shape: [B]

    Returns:
        The values as a sequence.
            Length: B
            Shape of inner tensors: [L_b, *]

    Examples:
    >>> values = torch.tensor([
    ...     [1, 2, 3, 0],
    ...     [5, 0, 0, 0],
    ...     [0, 0, 0, 0],
    ...     [13, 14, 15, 16],
    ... ])
    >>> L_bs = torch.tensor([3, 1, 0, 4])
    >>> sequentialized = sequentialize_padded(values, L_bs)
    >>> sequentialized
    [tensor([1, 2, 3]),
     tensor([5]),
     tensor([], dtype=torch.int64),
     tensor([13, 14, 15, 16])]
    """
    return sequentialize_packed(pack_padded(values, L_bs), L_bs)  # B x [L_b, *]


def sequentialize_padded_multidim(
    values: torch.Tensor, L_bsds: torch.Tensor
) -> list[torch.Tensor]:
    """Convert a batch of multi-dimensionally padded values to a sequence of
    values.

    Args:
        values: The padded values to convert. Padding could be arbitrary.
            Shape: [B, max(L_bs0), ..., max(L_bs{D-1}), *]
        L_bsds: The number of valid values in each sample for each dimension.
            Shape: [B, D]

    Returns:
        The values as a sequence.
            Length: B
            Shape of inner tensors: [L_b0, ..., L_b{D-1}, *]

    Examples:
    >>> values = torch.tensor([
    ...     [
    ...         [1, 2, 3],
    ...         [4, 5, 6],
    ...         [0, 0, 0],
    ...         [0, 0, 0],
    ...     ],
    ...     [
    ...         [13, 0, 0],
    ...         [16, 0, 0],
    ...         [19, 0, 0],
    ...         [22, 0, 0],
    ...     ],
    ... ])
    >>> L_bsds = torch.tensor([
    ...     [2, 3],
    ...     [4, 1],
    ... ])
    >>> sequentialized = sequentialize_padded_multidim(values, L_bsds)
    >>> sequentialized
    [tensor([[1, 2, 3],
             [4, 5, 6]]),
     tensor([[13],
             [16],
             [19],
             [22]])]
    """
    return sequentialize_packed_multidim(
        pack_padded_multidim(values, L_bsds), L_bsds
    )  # B x [L_b0, ..., L_b{D-1}, *]


def apply_mask(
    values: torch.Tensor,
    mask: torch.Tensor,
    L_bs: torch.Tensor,
    padding_value: Any = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply an additional mask to a batch of values.

    All values that are marked for keeping by the mask will be moved to the
    front of the tensor. Since the number of kept values in each sample can
    change, the function will also return the number of kept values in each
    sample.

    Args:
        values: The values to apply the mask to. Padding could be arbitrary.
            Shape: [B, max(L_bs), *]
        mask: The mask to apply. Padding could be arbitrary, since the function
            will apply the original mask in addition to the given mask.
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

    Examples:
    >>> values = torch.tensor([
    ...     [1, 2, 3, 0],
    ...     [5, 0, 0, 0],
    ...     [0, 0, 0, 0],
    ...     [13, 14, 15, 16],
    ... ])
    >>> mask = torch.tensor([
    ...     [ True, False,  True, False],
    ...     [False, False, False, False],
    ...     [False, False, False, False],
    ...     [ True,  True, False,  True],
    ... ])
    >>> L_bs = torch.tensor([3, 1, 0, 4])
    >>> values_masked, L_bs_kept = apply_mask(
    ...     values, mask, L_bs, padding_value=0
    ... )
    >>> values_masked
    tensor([[ 1,  3,  0],
            [ 0,  0,  0],
            [ 0,  0,  0],
            [13, 14, 16]])
    >>> L_bs_kept
    tensor([2, 0, 0, 3])
    """
    max_L_bs = values.shape[1]

    # Create a mask that indicates which values should be kept in each sample.
    mask = mask & mask_padding(L_bs, max_L_bs)  # [B, max(L_bs)]
    L_bs_kept = mask.sum(dim=1)  # [B]
    max_L_bs_kept = int(L_bs_kept.max()) if len(L_bs_kept) != 0 else 0

    # Move the masked values to the front of the tensor.
    values = pad_packed(
        values[mask], L_bs_kept, max_L_bs_kept, padding_value=padding_value
    )  # [B, max(L_bs_kept), *]
    return values, L_bs_kept


def apply_mask_multidim(
    values: torch.Tensor,
    mask: torch.Tensor,
    L_bsds: torch.Tensor,
    padding_value: Any = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply an additional mask to a batch of multi-dimensionally padded values.

    All values that are marked for keeping by the mask will be moved to the
    front of the tensor. Since the number of kept values in each sample can
    change, the function will also return the number of kept values in each
    sample.

    Warning: Since the number of kept values in each sample could be
    inconsistent across dimensions, it is not possible to return the kept values
    in a multi-dimensional format. Instead, the number of kept values is summed
    across all dimensions, and the kept values are returned in a padded format
    with a single padding dimension. Note that this differs from the behavior of
    other *_multidim() functions in our module!

    Args:
        values: The values to apply the mask to. Padding could be arbitrary.
            Shape: [B, max(L_bs0), ..., max(L_bs{D-1}), *]
        mask: The mask to apply. Padding could be arbitrary, since the function
            will apply the original mask in addition to the given mask.
            Shape: [B, max(L_bs0), ..., max(L_bs{D-1})]
        L_bsds: The number of valid values in each sample for each dimension.
            Shape: [B, D]
        padding_value: The value to pad the values with. If None, the values
            are padded with random values. This is faster than padding with
            a specific value.

    Returns:
        Tuple containing:
        - The values with the given mask applied. Padded with padding_value.
            Shape: [B, max(L_bs_kept), *]
        - The number of kept values in each sample.
            Shape: [B]

    Examples:
    >>> values = torch.tensor([
    ...     [
    ...         [1, 2, 3],
    ...         [4, 5, 6],
    ...         [0, 0, 0],
    ...         [0, 0, 0],
    ...     ],
    ...     [
    ...         [13, 0, 0],
    ...         [16, 0, 0],
    ...         [19, 0, 0],
    ...         [22, 0, 0],
    ...     ],
    ... ])
    >>> mask = torch.tensor([
    ...     [
    ...         [ True, False,  True],
    ...         [False,  True,  True],
    ...         [False, False, False],
    ...         [False, False, False],
    ...     ],
    ...     [
    ...         [ True, False, False],
    ...         [False, False, False],
    ...         [ True, False, False],
    ...         [ True, False, False],
    ...     ],
    ... ])
    >>> L_bsds = torch.tensor([
    ...     [2, 3],
    ...     [4, 1],
    ... ])
    >>> values_masked, L_bs_kept = apply_mask_multidim(
    ...     values, mask, L_bsds, padding_value=0
    ... )
    >>> values_masked
    tensor([[ 1,  3,  5,  6],
            [13, 19, 22,  0]])
    >>> L_bs_kept
    tensor([4, 3])
    """
    D = L_bsds.shape[1]
    device = L_bsds.device
    max_L_bsds = torch.tensor(values.shape[1 : D + 1], device=device)  # [D]

    # Create a mask that indicates which values should be kept in each sample.
    mask = mask & mask_padding_multidim(
        L_bsds, max_L_bsds
    )  # [B, max(L_bs0), ..., max(L_bs{D-1})]
    L_bs_kept = mask.sum(dim=tuple(range(1, D + 1)))  # [B]
    max_L_bs_kept = int(L_bs_kept.max()) if len(L_bs_kept) != 0 else 0

    # Move the masked values to the front of the tensor.
    values = pad_packed(
        values[mask], L_bs_kept, max_L_bs_kept, padding_value=padding_value
    )  # [B, max(L_bs_kept), *]
    return values, L_bs_kept


def replace_padding(
    values: torch.Tensor,
    L_bs: torch.Tensor,
    padding_value: Any = 0,
    in_place: bool = False,
) -> torch.Tensor:
    """Replace any existing values padding in one of several ways.

    Args:
        values: The values to pad. Padding could be arbitrary.
            Shape: [B, max(L_bs), *]
        L_bs: The number of valid values in each sample.
            Shape: [B]
        padding_value: The value to pad the values with. Can be one of:
            - A scalar value, or a tensor of shape [] or [*], containing the
              value to pad all elements with.
            - A tensor of shape [B, max(L_bs)] or [B, max(L_bs), *], containing
              the value to pad each element with.
            - A tensor of shape [B, 1] or [B, 1, *], containing the value to
              pad each row with.
            - A tensor of shape [1, max(L_bs)] or [1, max(L_bs), *], containing
              the value to pad each column with.
            - A tensor of shape [B * max(L_bs) - L] or [B * max(L_bs) - L, *],
              containing the value to pad each element in the padding mask with.
        in_place: Whether to perform the operation in-place.

    Returns:
        The padded values. Padded with padding_value.
            Shape: [B, max(L_bs), *]

    Examples:
    >>> values = torch.tensor([
    ...     [1, 2, 3, 0],
    ...     [5, 0, 0, 0],
    ...     [0, 0, 0, 0],
    ...     [13, 14, 15, 16],
    ... ])
    >>> L_bs = torch.tensor([3, 1, 0, 4])

    >>> padding_value = -1
    >>> replaced = replace_padding(values, L_bs, padding_value=padding_value)
    >>> replaced
    tensor([[ 1,  2,  3, -1],
            [ 5, -1, -1, -1],
            [-1, -1, -1, -1],
            [13, 14, 15, 16]])

    >>> padding_value = torch.tensor([
    ...     [-1, -2, -3, -4],
    ...     [-5, -6, -7, -8],
    ...     [-9, -10, -11, -12],
    ...     [-13, -14, -15, -16],
    ... ])
    >>> replaced = replace_padding(values, L_bs, padding_value=padding_value)
    >>> replaced
    tensor([[  1,   2,   3,  -4],
            [  5,  -6,  -7,  -8],
            [ -9, -10, -11, -12],
            [ 13,  14,  15,  16]])

    >>> padding_value = torch.tensor([
    ...     [-1],
    ...     [-2],
    ...     [-3],
    ...     [-4],
    ... ])
    >>> replaced = replace_padding(values, L_bs, padding_value=padding_value)
    >>> replaced
    tensor([[ 1,  2,  3, -1],
            [ 5, -2, -2, -2],
            [-3, -3, -3, -3],
            [13, 14, 15, 16]])

    >>> padding_value = torch.tensor([
    ...     [-1, -2, -3, -4],
    ... ])
    >>> replaced = replace_padding(values, L_bs, padding_value=padding_value)
    >>> replaced
    tensor([[ 1,  2,  3, -4],
            [ 5, -2, -3, -4],
            [-1, -2, -3, -4],
            [13, 14, 15, 16]])

    >>> padding_value = torch.tensor([-1, -2, -3, -4, -5, -6, -7, -8])
    >>> replaced = replace_padding(values, L_bs, padding_value=padding_value)
    >>> replaced
    tensor([[ 1,  2,  3, -1],
            [ 5, -2, -3, -4],
            [-5, -6, -7, -8],
            [13, 14, 15, 16]])
    """
    B, max_L_bs, *star = values.shape
    L = L_bs.sum()
    device = values.device
    dtype = values.dtype

    # Create a mask that indicates which values are valid in each sample.
    mask = mask_padding(L_bs, max_L_bs)  # [B, max(L_bs)]

    # Handle in-place operations.
    values_padded = values if in_place else values.clone()

    # Handle the case where padding_value is a scalar.
    if not isinstance(padding_value, torch.Tensor):
        values_padded[~mask] = padding_value
        return values_padded

    # Convert padding_value to the same device and dtype as values.
    padding_value = padding_value.to(device=device, dtype=dtype)

    # Handle shapes that are missing the star dimensions.
    if (
        padding_value.shape == ()
        or padding_value.shape == (B, max_L_bs)
        or padding_value.shape == (B, 1)
        or padding_value.shape == (1, max_L_bs)
        or padding_value.shape == (B * max_L_bs - L,)
    ):
        padding_value = padding_value.reshape(
            (*padding_value.shape, *[1] * len(star))
        )

    # Now handle all expected shapes.
    # Note that the shape of the tensor to be set, i.e. values_padded[~mask], is
    # always [B * max_L_bs - L, *].
    if padding_value.shape == tuple(star):
        values_padded[~mask] = padding_value
    elif padding_value.shape == (B, max_L_bs, *star):
        values_padded[~mask] = padding_value[~mask]
    elif padding_value.shape == (B, 1, *star):
        values_padded[~mask] = padding_value.squeeze(1).repeat_interleave(
            max_L_bs - L_bs, dim=0
        )
    elif padding_value.shape == (1, max_L_bs, *star):
        values_padded[~mask] = padding_value.expand(B, max_L_bs, *star)[~mask]
    elif padding_value.shape == (B * max_L_bs - L, *star):
        values_padded[~mask] = padding_value
    else:
        raise ValueError(
            "Shape of padding_value did not match any of the expected shapes."
            f" Got {list(padding_value.shape)}, but expected one of: [],"
            f" {star}, {[B, max_L_bs]}, {[B, max_L_bs, *star]}, {[B, 1]},"
            f" {[B, 1, *star]}, {[1, max_L_bs]},  {[1, max_L_bs, *star]},"
            f" {[B * max_L_bs - L.item()]}, or"
            f" {[B * max_L_bs - L.item(), *star]}"
        )

    return values_padded


def replace_padding_multidim(
    values: torch.Tensor,
    L_bsds: torch.Tensor,
    padding_value: Any = 0,
    in_place: bool = False,
) -> torch.Tensor:
    """Replace any existing multi-dimensional values padding in one of several
    ways.

    Args:
        values: The values to pad. Padding could be arbitrary.
            Shape: [B, max(L_bs0), ..., max(L_bs{D-1}), *]
        L_bsds: The number of valid values in each sample for each dimension.
            Shape: [B, D]
        padding_value: The value to pad the values with. Can be one of:
            - A scalar value, or a tensor of shape [] or [*], containing the
              value to pad all elements with.
            - A tensor of shape [B, max(L_bs0), ..., max(L_bs{D-1})] or
              [B, max(L_bs0), ..., max(L_bs{D-1}), *], containing the value to
              pad each element with.
            - A tensor of shape [B, 1, ..., 1] or [B, 1, ..., 1, *], containing
              the value to pad each row with.
            - A tensor of shape [1, max(L_bs0), ..., max(L_bs{D-1})] or
              [1, max(L_bs0), ..., max(L_bs{D-1}), *], containing the value to
              pad each column with.
            - A tensor of shape [B * prod(max(L_bsds)) - L] or
              [B * prod(max(L_bsds)) - L, *], containing the value to pad each
              element in the padding mask with.
        in_place: Whether to perform the operation in-place.

    Returns:
        The padded values. Padded with padding_value.
            Shape: [B, max(L_bs0), ..., max(L_bs{D-1}), *]

    Examples:
    >>> values = torch.tensor([
    ...     [
    ...         [1, 2, 3],
    ...         [4, 5, 6],
    ...         [0, 0, 0],
    ...         [0, 0, 0],
    ...     ],
    ...     [
    ...         [13, 0, 0],
    ...         [16, 0, 0],
    ...         [19, 0, 0],
    ...         [22, 0, 0],
    ...     ],
    ... ])
    >>> L_bsds = torch.tensor([
    ...     [2, 3],
    ...     [4, 1],
    ... ])

    >>> padding_value = -1
    >>> replaced = replace_padding_multidim(
    ...     values, L_bsds, padding_value=padding_value
    ... )
    >>> replaced
    tensor([[[ 1,  2,  3],
             [ 4,  5,  6],
             [-1, -1, -1],
             [-1, -1, -1]],
            [[13, -1, -1],
             [16, -1, -1],
             [19, -1, -1],
             [22, -1, -1]]])

    >>> padding_value = torch.tensor([
    ...     [
    ...         [-1, -2, -3],
    ...         [-4, -5, -6],
    ...         [-7, -8, -9],
    ...         [-10, -11, -12],
    ...     ],
    ...     [
    ...         [-13, -14, -15],
    ...         [-16, -17, -18],
    ...         [-19, -20, -21],
    ...         [-22, -23, -24],
    ...     ],
    ... ])
    >>> replaced = replace_padding_multidim(
    ...     values, L_bsds, padding_value=padding_value
    ... )
    >>> replaced
    tensor([[[  1,   2,   3],
             [  4,   5,   6],
             [ -7,  -8,  -9],
             [-10, -11, -12]],
            [[ 13, -14, -15],
             [ 16, -17, -18],
             [ 19, -20, -21],
             [ 22, -23, -24]]])

    >>> padding_value = torch.tensor([
    ...     [
    ...         [-1],
    ...     ],
    ...     [
    ...         [-2],
    ...     ],
    ... ])
    >>> replaced = replace_padding_multidim(
    ...     values, L_bsds, padding_value=padding_value
    ... )
    >>> replaced
    tensor([[[ 1,  2,  3],
             [ 4,  5,  6],
             [-1, -1, -1],
             [-1, -1, -1]],
            [[13, -2, -2],
             [16, -2, -2],
             [19, -2, -2],
             [22, -2, -2]]])

    >>> padding_value = torch.tensor([
    ...     [
    ...         [-1, -2, -3],
    ...         [-4, -5, -6],
    ...         [-7, -8, -9],
    ...         [-10, -11, -12],
    ...     ],
    ... ])
    >>> replaced = replace_padding_multidim(
    ...     values, L_bsds, padding_value=padding_value
    ... )
    >>> replaced
    tensor([[[  1,   2,   3],
             [  4,   5,   6],
             [ -7,  -8,  -9],
             [-10, -11, -12]],
            [[ 13,  -2,  -3],
             [ 16,  -5,  -6],
             [ 19,  -8,  -9],
             [ 22, -11, -12]]])

    >>> padding_value = torch.tensor([
    ...     -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14
    ... ])
    >>> replaced = replace_padding_multidim(
    ...     values, L_bsds, padding_value=padding_value
    ... )
    >>> replaced
    tensor([[[  1,   2,   3],
             [  4,   5,   6],
             [ -1,  -2,  -3],
             [ -4,  -5,  -6]],
            [[ 13,  -7,  -8],
             [ 16,  -9, -10],
             [ 19, -11, -12],
             [ 22, -13, -14]]])
    """
    B, D = L_bsds.shape
    device = L_bsds.device
    dtype = values.dtype
    max_L_bsds = torch.tensor(values.shape[1 : D + 1], device=device)  # [D]
    star = list(values.shape[D + 1 :])
    L_bs = L_bsds.prod(dim=1)  # [B]
    L = L_bs.sum()
    theory_max_L_bs = max_L_bsds.prod()

    # Create a mask that indicates which values are valid in each sample.
    mask = mask_padding_multidim(
        L_bsds, max_L_bsds
    )  # [B, max(L_bs0), ..., max(L_bs{D-1})]

    # Handle in-place operations.
    values_padded = values if in_place else values.clone()

    # Handle the case where padding_value is a scalar.
    if not isinstance(padding_value, torch.Tensor):
        values_padded[~mask] = padding_value
        return values_padded

    # Convert padding_value to the same device and dtype as values.
    padding_value = padding_value.to(device=device, dtype=dtype)

    # Handle shapes that are missing the star dimensions.
    if (
        padding_value.shape == ()
        or padding_value.shape == (B, *max_L_bsds)
        or padding_value.shape == (B, *[1] * D)
        or padding_value.shape == (1, *max_L_bsds)
        or padding_value.shape == (B * theory_max_L_bs - L,)
    ):
        padding_value = padding_value.reshape(
            (*padding_value.shape, *[1] * len(star))
        )

    # Now handle all expected shapes.
    # Note that the shape of the tensor to be set, i.e. values_padded[~mask], is
    # always [B * prod(max(L_bsds)) - L, *].
    if padding_value.shape == tuple(star):
        values_padded[~mask] = padding_value
    elif padding_value.shape == (B, *max_L_bsds, *star):
        values_padded[~mask] = padding_value[~mask]
    elif padding_value.shape == (B, *[1] * D, *star):
        values_padded[~mask] = padding_value.squeeze(
            tuple(range(1, D + 1))
        ).repeat_interleave(theory_max_L_bs - L_bs, dim=0)
    elif padding_value.shape == (1, *max_L_bsds, *star):
        values_padded[~mask] = padding_value.expand(B, *max_L_bsds, *star)[
            ~mask
        ]
    elif padding_value.shape == (B * theory_max_L_bs - L, *star):
        values_padded[~mask] = padding_value
    else:
        raise ValueError(
            "Shape of padding_value did not match any of the expected shapes."
            f" Got {list(padding_value.shape)}, but expected one of: [],"
            f" {star}, {[B, *max_L_bsds.tolist()]},"
            f" {[B, *max_L_bsds.tolist(), *star]}, {[B, *[1] * D]},"
            f" {[B, *[1] * D, *star]}, {[1, *max_L_bsds.tolist()]},"
            f" {[1, *max_L_bsds.tolist(), *star]},"
            f" {[B * theory_max_L_bs.item() - L.item()]}, or"
            f" {[B * theory_max_L_bs.item() - L.item(), *star]}"
        )

    return values_padded


def last_valid_value_padding(
    values: torch.Tensor,
    L_bs: torch.Tensor,
    padding_value_empty_rows: Any = None,
    in_place: bool = False,
) -> torch.Tensor:
    """Pad the values with the last valid value for each sample.

    Args:
        values: The values to pad. Padding could be arbitrary.
            Shape: [B, max(L_bs), *]
        L_bs: The number of valid values in each sample.
            Shape: [B]
        padding_value_empty_rows: The value to pad empty rows with (i.e. when
            L_b == 0 for some b). If None, empty rows are padded with random
            values. This is faster than padding with a specific value. Can be a
            scalar value, or a tensor of shape [] or [*], containing the value
            to pad all empty rows with.
        in_place: Whether to perform the operation in-place.

    Returns:
        The padded values. Padded with the last valid value.
            Shape: [B, max(L_bs), *]

    Examples:
    >>> values = torch.tensor([
    ...     [1, 2, 3, 0],
    ...     [5, 0, 0, 0],
    ...     [0, 0, 0, 0],
    ...     [13, 14, 15, 16],
    ... ])
    >>> L_bs = torch.tensor([3, 1, 0, 4])
    >>> padding_value_empty_rows = -1
    >>> padded = last_valid_value_padding(
    ...     values, L_bs, padding_value_empty_rows=padding_value_empty_rows
    ... )
    >>> padded
    tensor([[ 1,  2,  3,  3],
            [ 5,  5,  5,  5],
            [-1, -1, -1, -1],
            [13, 14, 15, 16]])
    """
    B, max_L_bs, *star = values.shape
    device = values.device
    dtype = values.dtype

    # Determine the padding value for each sample.
    arange_B = torch.arange(B, device=device)
    padding_value = (
        values[arange_B, L_bs - 1]
        if max_L_bs != 0
        else torch.empty((B, *star), device=device, dtype=dtype)
    )  # [B, *]
    if padding_value_empty_rows is not None:
        padding_value = padding_value.clone()
        padding_value[L_bs == 0] = padding_value_empty_rows

    # Replace the padding values using per row replacement.
    padding_value = padding_value.unsqueeze(1)  # [B, 1, *]
    values = replace_padding(
        values, L_bs, padding_value=padding_value, in_place=in_place
    )  # [B, max(L_bs), *]

    return values


def last_valid_value_padding_multidim(
    values: torch.Tensor,
    L_bsds: torch.Tensor,
    padding_value_empty_rows: Any = None,
    in_place: bool = False,
) -> torch.Tensor:
    """Pad the multi-dimensional values with the last valid value for each
    sample.

    The last valid value is defined as the last valid value if we were to
    flatten all padding dimensions into a single dimension.

    Args:
        values: The values to pad. Padding could be arbitrary.
            Shape: [B, max(L_bs0), ..., max(L_bs{D-1}), *]
        L_bsds: The number of valid values in each sample for each dimension.
            Shape: [B, D]
        padding_value_empty_rows: The value to pad empty rows with (i.e. when
            prod(L_bsds[b]) == 0 for some b). If None, empty rows are padded
            with random values. This is faster than padding with a specific
            value. Can be a scalar value, or a tensor of shape [] or [*],
            containing the value to pad all empty rows with.
        in_place: Whether to perform the operation in-place.

    Returns:
        The padded values. Padded with the last valid value.
            Shape: [B, max(L_bs0), ..., max(L_bs{D-1}), *]

    Examples:
    >>> values = torch.tensor([
    ...     [
    ...         [1, 2, 3],
    ...         [4, 5, 6],
    ...         [0, 0, 0],
    ...         [0, 0, 0],
    ...     ],
    ...     [
    ...         [13, 0, 0],
    ...         [16, 0, 0],
    ...         [19, 0, 0],
    ...         [22, 0, 0],
    ...     ],
    ... ])
    >>> L_bsds = torch.tensor([
    ...     [2, 3],
    ...     [4, 1],
    ... ])
    >>> padding_value_empty_rows = -1
    >>> padded = last_valid_value_padding_multidim(
    ...     values, L_bsds, padding_value_empty_rows=padding_value_empty_rows
    ... )
    >>> padded
    tensor([[[ 1,  2,  3],
             [ 4,  5,  6],
             [ 6,  6,  6],
             [ 6,  6,  6]],
            [[13, 22, 22],
             [16, 22, 22],
             [19, 22, 22],
             [22, 22, 22]]])
    """
    B, D = L_bsds.shape
    device = L_bsds.device
    dtype = values.dtype
    max_L_bsds = torch.tensor(values.shape[1 : D + 1], device=device)  # [D]
    star = list(values.shape[D + 1 :])
    L_bs = L_bsds.prod(dim=1)  # [B]
    theory_max_L_bs = max_L_bsds.prod()

    # Determine the padding value for each sample.
    arange_B = torch.arange(B, device=device)
    padding_value = (
        values[(arange_B, *torch.unbind(L_bsds - 1, dim=1))]
        if theory_max_L_bs != 0
        else torch.empty((B, *star), device=device, dtype=dtype)
    )  # [B, *]
    if padding_value_empty_rows is not None:
        padding_value = padding_value.clone()
        padding_value[L_bs == 0] = padding_value_empty_rows

    # Replace the padding values using per row replacement.
    padding_shape = (B, *[1] * D, *star)
    padding_value = padding_value.reshape(padding_shape)  # [B, 1, ..., 1, *]
    values = replace_padding_multidim(
        values, L_bsds, padding_value=padding_value, in_place=in_place
    )  # [B, max(L_bs0), ..., max(L_bs{D-1}), *]

    return values


# ################################### MATHS ####################################


def interp(
    x: torch.Tensor,
    xp: torch.Tensor,
    fp: torch.Tensor,
    left: float | None = None,
    right: float | None = None,
    period: float | None = None,
) -> torch.Tensor:
    """Like np.interp(), but for PyTorch tensors.

    Performs linear interpolation on a 1D tensor.

    Args:
        x: The x-coordinates at which to evaluate the interpolated values.
            Shape: [N]
        xp: The x-coordinates of the data points, must be increasing.
            Shape: [M]
        fp: The y-coordinates of the data points, same shape as xp.
            Shape: [M]
        left: Value to return for x < xp[0], default is fp[0].
        right: Value to return for x > xp[-1], default is fp[-1].
        period: A period for the x-coordinates. This parameter allows the
            proper interpolation of angular x-coordinates. Parameters left and
            right are ignored if period is specified.

    Returns:
        The interpolated values for each batch.
            Shape: [N]

    Examples:
    >>> x = torch.tensor([10.5, 200.0, 40.0, 56.0])
    >>> xp = torch.tensor([0.0, 1.0, 20.0, 100.0])
    >>> fp = torch.tensor([0.0, 100.0, 200.0, 300.0])
    >>> interp(x, xp, fp)
    tensor([150., 300., 225., 245.])
    """
    # Handle periodic interpolation.
    if period is not None:
        if period <= 0:
            raise ValueError("period must be positive.")

        # Normalize x and xp to [0, period).
        x %= period  # [N]
        xp %= period  # [M]

        # Re-sort xp and fp after the modulo operation.
        sorted_idcs = xp.argsort()  # [M]
        xp = xp[sorted_idcs]  # [M]
        fp = fp[sorted_idcs]  # [M]

        # Extend xp and fp tensors to handle wrap-around interpolation. Add the
        # last point before the first, and the first point after the last.
        xp = torch.concat([
            torch.tensor([xp[-1] - period]),
            xp,
            torch.tensor([xp[0] + period]),
        ])  # [M + 2]
        fp = torch.concat(
            [torch.tensor([fp[-1]]), fp, torch.tensor([fp[0]])]
        )  # [M + 2]

    M = len(xp)

    # Check if xp is strictly increasing.
    if not (xp.diff() > 0).all():
        raise ValueError(
            "xp must be strictly increasing along the last dimension."
        )

    # Find indices of neighbours in xp.
    right_idx = torch.searchsorted(xp, x)  # [N]
    left_idx = right_idx - 1  # [N]

    # Clamp indices to valid range (we will handle the edges later).
    left_idx = left_idx.clamp(min=0, max=M - 1)  # [N]
    right_idx = right_idx.clamp(min=0, max=M - 1)  # [N]

    # Gather neighbour values.
    x_left = xp[left_idx]  # [N]
    x_right = xp[right_idx]  # [N]
    y_left = fp[left_idx]  # [N]
    y_right = fp[right_idx]  # [N]

    # Avoid division by zero for x_left == x_right.
    denom = x_right - x_left  # [N]
    denom[denom == 0] = 1
    p = (x - x_left) / denom  # [N]

    # Perform interpolation.
    y = y_left + p * (y_right - y_left)  # [N]

    # Handle edges only if period is not specified.
    if period is None:
        # Handle left edge.
        if left is None:
            left = fp[0].item()
        is_left = x < xp[[0]]  # [N]
        y[is_left] = left

        # Handle right edge.
        if right is None:
            right = fp[-1].item()
        is_right = x > xp[[-1]]  # [N]
        y[is_right] = right

    return y


# ################################### RANDOM ###################################


def permuted(x: torch.Tensor, dim: int | None = None) -> torch.Tensor:
    """Like np.random.Generator.permuted(), but for PyTorch tensors.

    Randomly permute x along the given dimension.

    Args:
        x: Tensor to be shuffled.
        dim: Slices of x in this dimension are shuffled. Each slice is shuffled
            independently of the others. If None, the flattened tensor is
            shuffled.

    Returns:
        A shuffled copy of the input tensor.

    Examples:
    >>> x = torch.tensor([
    ...     [ 0,  1,  2,  3,  4,  5,  6,  7],
    ...     [ 8,  9, 10, 11, 12, 13, 14, 15],
    ...     [16, 17, 18, 19, 20, 21, 22, 23],
    ... ])

    >>> x_perm = permuted(x, dim=1)
    >>> x_perm  # doctest: +SKIP
    tensor([[ 4,  3,  6,  7,  1,  2,  5,  0],
            [15, 10, 14,  9, 12, 11,  8, 13],
            [17, 16, 20, 21, 18, 22, 23, 19]])

    >>> x_perm = permuted(x)
    >>> x_perm  # doctest: +SKIP
    tensor([[ 5, 11, 23,  0, 21,  2, 14, 22],
            [16, 18, 15,  6, 13,  3,  7, 10],
            [ 9,  8, 20, 12, 17,  1, 19,  4]])
    """
    device = x.device

    if dim is None:
        x_flat = x.flatten()
        perm = torch.randperm(len(x_flat), device=device)
        return x_flat[perm].reshape(x.shape)

    # Generate random permutation indices for the given dimension.
    # Use torch.rand() + argsort() to shuffle each slice independently.
    # This creates random values and sorts them, giving independent random
    # permutations for each slice without using a for-loop.
    random_vals = torch.rand(x.shape, device=device)
    indices = random_vals.argsort(dim)
    return x.gather(dim, indices)


# ######################### BASIC TENSOR MANIPULATION ##########################


def to_tensor(
    object: object,
    device: torch.device | str | int | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Convert an object to a tensor.

    Warning: Does not copy the data if possible. Thus, the returned tensor
    could share memory with the original object.

    Args:
        object: The object to convert to a tensor.
        device: The desired device of the returned tensor. If None, the
            device of the object will be used.
        dtype: The desired data type of the returned tensor. If None, the
            dtype of the object will be used.
    """
    # PyTorch supported types: bool, uint8, int8, int16, int32, int64, float16,
    # float32, float64, complex64, and complex128.
    if isinstance(object, torch.Tensor):
        return object.to(device=device, dtype=dtype)

    if isinstance(object, np.ndarray):
        return torch.from_numpy(object).to(device=device, dtype=dtype)

    # Last resort: because numpy recognizes more array-like types than torch,
    # we try to convert to numpy first.
    try:
        return to_tensor(np.array(object), device=device, dtype=dtype)
    except Exception as e:
        raise TypeError(
            f"Could not convert object of type {type(object)} to tensor."
        ) from e


def ravel_multi_index(
    multi_index: torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor],
    dims: torch.Tensor,
    mode: Literal["raise", "wrap", "clip"] = "raise",
    order: Literal["C", "F"] = "C",
) -> torch.Tensor:
    """Like np.ravel_multi_index(), but for PyTorch tensors.

    Converts a tuple of index tensors into a tensor of "flat" indices, applying
    boundary modes to the multi-index.

    Warning: Unlike np.ravel_multi_index(), this function does not support the
    mode being a tuple of modes.

    Args:
        multi_index: A tensor of shape [K, N_0, ..., N_{D-1}] or a tuple
            containing K [N_0, ..., N_{D-1}]-shaped sequences. K refers to the
            amount of elements in the tuples.
        dims: The shape of the tensor into which the indices point.
            Shape: [K]
        mode: Specifies how out-of-bounds indices are handled. Can specify
            either one mode or a tuple of modes, one mode per index.
            - "raise": Raise an error (default).
            - "wrap": Wrap around.
            - "clip": Clip to the range.
            In "clip" mode, a negative index which would normally wrap will
            clip to 0 instead.
        order: Determines whether the multi-index should be viewed as indexing
            in row-major (C-style) or column-major (Fortran-style) order.

    Returns:
        Tensor of indices into the flattened version of a tensor of shape dims.
            Shape: [N_0, ..., N_{D-1}]

    Examples:
    >>> multi_index = torch.tensor([
    ...     [3, 6, 6],
    ...     [4, 5, 1],
    ... ])

    >>> dims = torch.tensor([7, 6])
    >>> ravel_multi_index(multi_index, dims)
    tensor([22, 41, 37])

    >>> dims = torch.tensor([7, 6])
    >>> ravel_multi_index(multi_index, dims, order="F")
    tensor([31, 41, 13])

    >>> dims = torch.tensor([4, 6])
    >>> ravel_multi_index(multi_index, dims, mode="clip")
    tensor([22, 23, 19])

    >>> dims = torch.tensor([4, 4])
    >>> ravel_multi_index(multi_index, dims, mode="wrap")
    tensor([12, 9, 9])
    """
    device = dims.device

    # Check for integer overflow (an int64 contains one sign bit).
    if prod(dims.tolist()) > 2**63:
        raise ValueError(
            "invalid dims: tensor size defined by dims is larger than the"
            " maximum possible size."
        )

    # Convert the multi-index to a tensor if it is a tuple.
    if isinstance(multi_index, tuple) or isinstance(multi_index, list):
        multi_index = torch.stack(multi_index)  # [K, N_0, ..., N_{D-1}]

    # Calculate the factors with which the multi-index is multiplied to convert
    # it to a "flat" index.
    if order == "C":
        factors = torch.concat([
            dims[1:].flip(0).cumprod(0, dtype=torch.int64).flip(0),
            torch.ones(1, device=device, dtype=torch.int64),
        ])  # [K]
    elif order == "F":
        factors = torch.concat([
            torch.ones(1, device=device, dtype=torch.int64),
            dims[:-1].cumprod(0, dtype=torch.int64),
        ])  # [K]
    else:
        raise ValueError("only 'C' or 'F' order is permitted")

    # Reshape dims and factors to match the multi-index shape.
    dims = dims.reshape(
        -1, *(1 for _ in range(multi_index.ndim - 1))
    )  # [K, 1, ..., 1]
    factors = factors.reshape(
        -1, *([1] * (multi_index.ndim - 1))
    )  # [K, 1, ..., 1]

    if mode == "raise":
        # Raise an error if the multi-index is out of bounds.
        if not ((multi_index >= 0) & (multi_index < dims)).all():
            raise ValueError("invalid entry in coordinates tensor")
    elif mode == "wrap":
        # Wrap around the multi-index.
        multi_index = multi_index % dims
    elif mode == "clip":
        # Clip the multi-index to the range.
        multi_index = multi_index.clamp(
            torch.tensor(0, device=device), dims - 1
        )
    else:
        raise ValueError(
            f"clipmode must be one of 'clip', 'raise', or 'wrap' (got '{mode}')"
        )

    return (multi_index * factors).sum(dim=0)  # [N_0, ..., N_{D-1}]


def unravel_index(
    indices: torch.Tensor, shape: torch.Tensor, order: Literal["C", "F"] = "C"
) -> tuple[torch.Tensor, ...]:
    """Like np.unravel_index(), but for PyTorch tensors.

    Converts a tensor of "flat" indices into a tuple of coordinate tensors.

    Note: I am aware that torch.unravel_index() exists, but that function does
    not support the 'order' argument.

    Args:
        indices: A tensor of flat indices, each of which is an index into the
            flattened version of a tensor of dimensions shape.
            Shape: [N_0, ..., N_{D-1}]
        shape: The shape of the tensor into which the indices point.
            Shape: [K]
        order: Determines whether the multi-index should be viewed as indexing
            in row-major (C-style) or column-major (Fortran-style) order.

    Returns:
        Tuple of K coordinate tensors, each with the same shape as indices.
            Length: K
            Shape of inner tensors: [N_0, ..., N_{D-1}]

    Examples:
    >>> indices = torch.tensor([22, 41, 37])
    >>> shape = torch.tensor([7, 6])
    >>> unravel_index(indices, shape)
    (tensor([3, 6, 6]), tensor([4, 5, 1]))

    >>> indices = torch.tensor([31, 41, 13])
    >>> shape = torch.tensor([7, 6])
    >>> unravel_index(indices, shape, order="F")
    (tensor([3, 6, 6]), tensor([4, 5, 1]))
    """
    # Check for integer overflow (an int64 contains one sign bit).
    flat_size = prod(shape.tolist())
    if flat_size > 2**63:
        raise ValueError(
            "invalid shape: tensor size defined by shape is larger than the"
            " maximum possible size."
        )

    # Check that the indices are within bounds.
    if not ((0 <= indices) & (indices < flat_size)).all():
        raise ValueError(
            f"index is out of bounds for tensor with size {flat_size}"
        )

    # Convert flat indices to multi-indices.
    multi_index = []
    if order == "C":
        # For C-style ordering, process dimensions from last to first.
        for dim in reversed(shape):
            multi_index.append(indices % dim)
            indices = indices // dim
        multi_index.reverse()
    elif order == "F":
        # For Fortran-style ordering, process dimensions from first to last.
        for dim in shape:
            multi_index.append(indices % dim)
            indices = indices // dim
    else:
        raise ValueError("only 'C' or 'F' order is permitted")

    return tuple(multi_index)


# ######################## ADVANCED TENSOR MANIPULATION ########################


def swap_idcs_vals(x: torch.Tensor) -> torch.Tensor:
    """Swap the indices and values of a 1D tensor.

    The input tensor is assumed to contain exactly all integers from 0 to N - 1,
    in any order.

    Warning: This function does not explicitly check if the input tensor
    contains no duplicates. If x contains duplicates, the behavior is
    non-deterministic (one of the values from x will be picked arbitrarily).

    Args:
        x: The tensor to swap.
            Shape: [N]

    Returns:
        The swapped tensor.
            Shape: [N]

    Examples:
    >>> x = torch.tensor([2, 3, 0, 4, 1])
    >>> swap_idcs_vals(x)
    tensor([2, 4, 0, 1, 3])
    """
    if x.ndim != 1:
        raise ValueError("x must be 1D.")

    device = x.device
    dtype = x.dtype
    x_swapped = torch.empty_like(x)
    x_swapped[x] = torch.arange(len(x), device=device, dtype=dtype)
    return x_swapped


def swap_idcs_vals_duplicates(
    x: torch.Tensor, stable: bool = False
) -> torch.Tensor:
    """Swap the indices and values of a 1D tensor allowing duplicates.

    The input tensor is assumed to contain integers from 0 to M <= N, in any
    order, and may contain duplicates.

    The output tensor will contain exactly all integers from 0 to len(x) - 1,
    in any order.

    If the input doesn't contain duplicates, you should use swap_idcs_vals()
    instead since it is faster (especially for large tensors).

    Args:
        x: The tensor to swap.
            Shape: [N]
        stable: Whether to preserve the relative order of equal elements. If
            False (default), an unstable sort is used, which is faster.

    Returns:
        The swapped tensor.
            Shape: [N]

    Examples:
    >>> x = torch.tensor([1, 3, 0, 1, 3])
    >>> swap_idcs_vals_duplicates(x, stable=True)
    tensor([2, 0, 3, 1, 4])
    """
    if x.ndim != 1:
        raise ValueError("x must be 1D.")

    dtype = x.dtype

    # For some reason, this O(n log n) algorithm is actually faster than a
    # native implementation that uses a Python for loop with complexity O(n).
    return x.argsort(stable=stable).to(dtype)


# ############################ CONSECUTIVE SEGMENTS ############################


def starts_segments(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Find the start index of each consecutive segment.

    Args:
        x: The input tensor. Consecutive equal values will be grouped.
            Shape: [N_0, ..., N_dim, ..., N_{D-1}]
        dim: The dimension along which the segments are lined up.

    Returns:
        The start indices for each consecutive segment in x.
            Shape: [S]

    Examples:
    >>> x = torch.tensor([4, 4, 4, 2, 2, 8, 3, 3, 3, 3])

    >>> starts = starts_segments(x)
    >>> starts
    tensor([0, 3, 5, 6])
    """
    N_dim = x.shape[dim]
    device = x.device

    # Find the indices where the values change.
    is_change = (
        torch.concat(
            [
                torch.ones(1, device=device, dtype=torch.bool),
                (
                    x.index_select(
                        dim, torch.arange(0, N_dim - 1, device=device)
                    )  # [N_0, ..., N_dim - 1, ..., N_{D-1}]
                    != x.index_select(
                        dim, torch.arange(1, N_dim, device=device)
                    )  # [N_0, ..., N_dim - 1, ..., N_{D-1}]
                ).any(
                    dim=tuple(i for i in range(x.ndim) if i != dim)
                ),  # [N_dim - 1]
            ],
            dim=0,
        )  # [N_dim]
        if N_dim > 0
        else torch.empty(0, device=device, dtype=torch.bool)
    )  # [N_dim]

    # Find the start of each consecutive segment.
    return is_change.nonzero(as_tuple=True)[0]  # [S]


@overload
def counts_segments(  # type: ignore
    x: torch.Tensor, dim: int = 0, return_starts: Literal[False] = ...
) -> torch.Tensor:
    pass


@overload
def counts_segments(
    x: torch.Tensor, dim: int = 0, return_starts: Literal[True] = ...
) -> tuple[torch.Tensor, torch.Tensor]:
    pass


def counts_segments(
    x: torch.Tensor, dim: int = 0, return_starts: bool = False
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Count the length of each consecutive segment.

    Args:
        x: The input tensor. Consecutive equal values will be grouped.
            Shape: [N_0, ..., N_dim, ..., N_{D-1}]
        dim: The dimension along which the segments are lined up.
        return_starts: Whether to also return the start indices of each
            consecutive segment.

    Returns:
        Tuple containing:
        - The counts for each consecutive segment in x.
            Shape: [S]
        - (Optional) If return_starts is True, the start indices for each
            consecutive segment in x.
            Shape: [S]

    Examples:
    >>> x = torch.tensor([4, 4, 4, 2, 2, 8, 3, 3, 3, 3])
    >>> counts = counts_segments(x)
    >>> counts
    tensor([3, 2, 1, 4])
    """
    N_dim = x.shape[dim]
    device = x.device

    # Find the start of each consecutive segment.
    starts = starts_segments(x, dim=dim)  # [S]

    # Prepare starts for count calculation.
    starts_with_N_dim = torch.concat(
        [starts, torch.full((1,), N_dim, device=device)], dim=0
    )  # [S + 1]

    # Find the count of each consecutive segment.
    counts = (
        starts_with_N_dim.diff(dim=0)  # [S]
        if N_dim > 0
        else torch.empty(0, device=device, dtype=torch.int64)
    )  # [S]

    if return_starts:
        return counts, starts
    return counts


@overload
def outer_indices_segments(  # type: ignore
    x: torch.Tensor,
    dim: int = 0,
    return_counts: Literal[False] = ...,
    return_starts: Literal[False] = ...,
) -> torch.Tensor:
    pass


@overload
def outer_indices_segments(
    x: torch.Tensor,
    dim: int = 0,
    return_counts: Literal[True] = ...,
    return_starts: Literal[False] = ...,
) -> tuple[torch.Tensor, torch.Tensor]:
    pass


@overload
def outer_indices_segments(
    x: torch.Tensor,
    dim: int = 0,
    return_counts: Literal[False] = ...,
    return_starts: Literal[True] = ...,
) -> tuple[torch.Tensor, torch.Tensor]:
    pass


@overload
def outer_indices_segments(
    x: torch.Tensor,
    dim: int = 0,
    return_counts: Literal[True] = ...,
    return_starts: Literal[True] = ...,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pass


def outer_indices_segments(
    x: torch.Tensor,
    dim: int = 0,
    return_counts: bool = False,
    return_starts: bool = False,
) -> (
    torch.Tensor
    | tuple[torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
):
    """Get the outer indices for each consecutive segment.

    Args:
        x: The input tensor. Consecutive equal values will be grouped.
            Shape: [N_0, ..., N_dim, ..., N_{D-1}]
        dim: The dimension along which the segments are lined up.
        return_counts: Whether to also return the counts of each consecutive
            segment.
        return_starts: Whether to also return the start indices of each
            consecutive segment.

    Returns:
        Tuple containing:
        - The outer indices for each consecutive segment in x.
            Shape: [N_dim]
        - (Optional) If return_counts is True, the counts for each consecutive
            segment in x.
            Shape: [S]
        - (Optional) If return_starts is True, the start indices for each
            consecutive segment in x.
            Shape: [S]

    Examples:
    >>> x = torch.tensor([4, 4, 4, 2, 2, 8, 3, 3, 3, 3])
    >>> outer_idcs = outer_indices_segments(x)
    >>> outer_idcs
    tensor([0, 0, 0, 1, 1, 2, 3, 3, 3, 3])
    """
    device = x.device

    # Find the start (optional) and count of each consecutive segment.
    if return_starts:
        counts, starts = counts_segments(
            x, dim=dim, return_starts=True
        )  # [S], [S]
    else:
        counts = counts_segments(x, dim=dim)  # [S]

    # Calculate the outer indices.
    outer_idcs = torch.arange(counts.shape[0], device=device).repeat_interleave(
        counts, dim=0
    )  # [N_dim]

    if return_counts and return_starts:
        return outer_idcs, counts, starts  # type: ignore
    if return_counts:
        return outer_idcs, counts
    if return_starts:
        return outer_idcs, starts  # type: ignore
    return outer_idcs


@overload
def inner_indices_segments(  # type: ignore
    x: torch.Tensor,
    dim: int = 0,
    return_counts: Literal[False] = ...,
    return_starts: Literal[False] = ...,
) -> torch.Tensor:
    pass


@overload
def inner_indices_segments(
    x: torch.Tensor,
    dim: int = 0,
    return_counts: Literal[True] = ...,
    return_starts: Literal[False] = ...,
) -> tuple[torch.Tensor, torch.Tensor]:
    pass


@overload
def inner_indices_segments(
    x: torch.Tensor,
    dim: int = 0,
    return_counts: Literal[False] = ...,
    return_starts: Literal[True] = ...,
) -> tuple[torch.Tensor, torch.Tensor]:
    pass


@overload
def inner_indices_segments(
    x: torch.Tensor,
    dim: int = 0,
    return_counts: Literal[True] = ...,
    return_starts: Literal[True] = ...,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pass


def inner_indices_segments(
    x: torch.Tensor,
    dim: int = 0,
    return_counts: bool = False,
    return_starts: bool = False,
) -> (
    torch.Tensor
    | tuple[torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
):
    """Get the inner indices for each consecutive segment.

    Args:
        x: The input tensor. Consecutive equal values will be grouped.
            Shape: [N_0, ..., N_dim, ..., N_{D-1}]
        dim: The dimension along which the segments are lined up.
        return_counts: Whether to also return the counts of each consecutive
            segment.
        return_starts: Whether to also return the start indices of each
            consecutive segment.

    Returns:
        Tuple containing:
        - The inner indices for each consecutive segment in x.
            Shape: [N_dim]
        - (Optional) If return_counts is True, the counts for each consecutive
            segment in x.
            Shape: [S]
        - (Optional) If return_starts is True, the start indices for each
            consecutive segment in x.
            Shape: [S]

    Examples:
    >>> x = torch.tensor([4, 4, 4, 2, 2, 8, 3, 3, 3, 3])
    >>> inner_idcs = inner_indices_segments(x)
    >>> inner_idcs
    tensor([0, 1, 2, 0, 1, 0, 0, 1, 2, 3])
    """
    N_dim = x.shape[dim]

    # Find the start and count of each consecutive segment.
    counts, starts = counts_segments(x, dim=dim, return_starts=True)  # [S], [S]

    # Calculate the inner indices.
    inner_idcs = (
        torch.arange(N_dim)  # [N_dim]
        - starts.repeat_interleave(counts, dim=0)  # [N_dim]
    )  # [N_dim]  # fmt: skip

    if return_counts and return_starts:
        return inner_idcs, counts, starts
    if return_counts:
        return inner_idcs, counts
    if return_starts:
        return inner_idcs, starts
    return inner_idcs


# ################################## LEXSORT ###################################


def lexsort(
    keys: torch.Tensor | tuple[torch.Tensor, ...],
    dim: int = -1,
    stable: bool = False,
) -> torch.Tensor:
    """Like np.lexsort(), but for PyTorch tensors.

    Perform an indirect sort using a sequence of keys.

    Given multiple sorting keys, which can be interpreted as elements of a
    tuple, lexsort returns a tensor of integer indices that describes the sort
    order of the given tuples. The last key in the tuple is used for the
    primary sort order, the second-to-last key for the secondary sort order,
    and so on. The first dimension is always interpreted as the dimension
    along which the tuples lie.

    Args:
        keys: Tensor of shape [K, N_0, ..., N_dim, ..., N_{D-1}] or a tuple
            containing K [N_0, ..., N_dim, ..., N_{D-1}]-shaped sequences. K
            refers to the amount of elements in the tuples. The last element
            is the primary sort key.
        dim: Dimension to be indirectly sorted.
        stable: Whether to preserve the relative order of equal elements. If
            False (default), an unstable sort is used, which is faster.

    Returns:
        Tensor of indices that sort the keys along the specified dimension.
            Shape: [N_dim]

    Examples:
    >>> lexsort((
    ...     torch.tensor([ 1, 17, 18]),
    ...     torch.tensor([23, 10,  9]),
    ...     torch.tensor([14, 12,  0]),
    ...     torch.tensor([19,  5,  6]),
    ...     torch.tensor([21, 20, 22]),
    ...     torch.tensor([ 7,  3,  8]),
    ...     torch.tensor([13,  4,  2]),
    ...     torch.tensor([15, 11, 16]),
    ... ))
    tensor([1, 0, 2])

    >>> lexsort(
    ...     torch.tensor([
    ...         [4, 8, 2, 8, 3, 7, 3],
    ...         [9, 4, 0, 4, 0, 4, 1],
    ...         [1, 5, 1, 4, 3, 4, 4],
    ...     ])
    ... )
    tensor([2, 0, 4, 6, 5, 3, 1])
    """
    if isinstance(keys, tuple):
        keys = torch.stack(keys)  # [K, N_0, ..., N_dim, ..., N_{D-1}]

    device = keys.device
    dtype = keys.dtype

    # If the tensor is an integer tensor, first try sorting by representing
    # each of the "tuples" as a single integer. This is much faster than
    # lexsorting along the given dimension.
    if (
        dtype
        in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
            torch.uint16,
            torch.uint32,
            torch.uint64,
        )
        and keys.numel() != 0
    ):
        # Compute the minimum and maximum values for each key.
        dims_flat = tuple(range(1, keys.ndim))
        maxs = keys.amax(dim=dims_flat, keepdim=True)  # [K, 1, ..., 1]
        mins = keys.amin(dim=dims_flat, keepdim=True)  # [K, 1, ..., 1]
        extents = (maxs - mins + 1).squeeze(dim=dims_flat)  # [K]
        keys_dense = keys - mins  # [K, N_0, ..., N_dim, ..., N_{D-1}]

        try:
            # Convert the tuples to single integers.
            idcs = ravel_multi_index(
                keys_dense.long(), extents, mode="raise", order="F"
            )  # [N_0, ..., N_dim, ..., N_{D-1}]

            # Sort the integers.
            return idcs.argsort(dim=dim, stable=stable)
        except ValueError:
            # Overflow would occur when converting to integers.
            pass

    # If the tensor is not an integer tensor or if overflow would occur when
    # converting to integers, we have to use np.lexsort(). Unfortunately, torch
    # doesn't have a np.lexsort() equivalent, so we have to use numpy here.
    return torch.from_numpy(np.lexsort(keys.numpy(force=True), axis=dim)).to(
        device=device
    )


def lexsort_along(
    x: torch.Tensor, dim: int = -1, stable: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sort a tensor along dim, taking all others as tuples.

    This is like torch.sort(), but the other dimensions are treated as tuples.
    This function is roughly equivalent to the following Python code, but it is
    much faster.
    >>> torch.stack(  # doctest: +SKIP
    ...     sorted(
    ...         x.unbind(dim),
    ...         key=tuple,
    ...     ),
    ...     dim=dim,
    ... )

    Args:
        x: The tensor to sort.
            Shape: [N_0, ..., N_dim, ..., N_{D-1}]
        dim: The dimension to sort along.
        stable: Whether to preserve the relative order of equal elements. If
            False (default), an unstable sort is used, which is faster.

    Returns:
        Tuple containing:
        - Sorted version of x.
            Shape: [N_0, ..., N_dim, ..., N_{D-1}]
        - The backmap tensor, which contains the indices of the sorted values
            in the original input.
            The sorted version of x can be retrieved as follows:
            x_sorted = x.index_select(dim, backmap)
            Shape: [N_dim]

    Examples:
    >>> x = torch.tensor([
    ...     [2, 1],
    ...     [3, 0],
    ...     [1, 2],
    ...     [1, 3],
    ... ])
    >>> dim = 0

    >>> x_sorted, backmap = lexsort_along(x, dim=dim)
    >>> x_sorted
    tensor([[1, 2],
            [1, 3],
            [2, 1],
            [3, 0]])
    >>> backmap
    tensor([2, 3, 0, 1])

    >>> # Get the lexicographically sorted version of x:
    >>> x.index_select(dim, backmap)
    tensor([[1, 2],
            [1, 3],
            [2, 1],
            [3, 0]])
    """
    # We can use lexsort() to sort only the requested dimension.
    # First, we prepare the tensor for lexsort(). The input to this function
    # must be a tuple of tensor-like objects, that are evaluated from last to
    # first. This is quite confusing, so I'll put an example here. If we have:
    # >>> x = tensor([
    # ...     [
    # ...         [15, 13],
    # ...         [11,  4],
    # ...         [16,  2],
    # ...     ],
    # ...     [
    # ...         [ 7, 21],
    # ...         [ 3, 20],
    # ...         [ 8, 22],
    # ...     ],
    # ...     [
    # ...         [19, 14],
    # ...         [ 5, 12],
    # ...         [ 6,  0],
    # ...     ],
    # ...     [
    # ...         [23,  1],
    # ...         [10, 17],
    # ...         [ 9, 18],
    # ...     ],
    # ... ])
    # And dim=1, then the input to lexsort() must be:
    # >>> lexsort(tensor([
    # ...     [ 1, 17, 18],
    # ...     [23, 10,  9],
    # ...     [14, 12,  0],
    # ...     [19,  5,  6],
    # ...     [21, 20, 22],
    # ...     [ 7,  3,  8],
    # ...     [13,  4,  2],
    # ...     [15, 11, 16],
    # ... ]))
    # Note that the first row is evaluated last and the last row is evaluated
    # first. We can now see that the sorting order will be 11 < 15 < 16, so
    # lexsort() will return tensor([1, 0, 2]). I thouroughly tested what the
    # absolute fastest way is to perform this operation, and it turns out that
    # the following is the best way to do it.
    N_dim = x.shape[dim]

    if x.ndim == 1:
        y = x.unsqueeze(0)  # [1, N_dim]
    else:
        y = x.movedim(
            dim, -1
        )  # [N_0, ..., N_{dim-1}, N_{dim+1}, ..., N_{D-1}, N_dim]
        y = y.reshape(
            -1, N_dim
        )  # [N_0 * ... * N_{dim-1} * N_{dim+1} * ... * N_{D-1}, N_dim]
    y = y.flip(
        dims=(0,)
    )  # [N_0 * ... * N_{dim-1} * N_{dim+1} * ... * N_{D-1}, N_dim]
    backmap = lexsort(y, dim=-1, stable=stable)  # [N_dim]

    # Sort the tensor along the given dimension.
    x_sorted = x.index_select(dim, backmap)  # [N_0, ..., N_dim, ..., N_{D-1}]

    # Finally, we return the sorted tensor and the backmap.
    return x_sorted, backmap


# ################################### UNIQUE ###################################


@overload
def unique_consecutive(  # type: ignore
    x: torch.Tensor,
    return_inverse: Literal[False] = ...,
    return_counts: Literal[False] = ...,
    dim: int | None = None,
) -> torch.Tensor:
    pass


@overload
def unique_consecutive(
    x: torch.Tensor,
    return_inverse: Literal[True] = ...,
    return_counts: Literal[False] = ...,
    dim: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    pass


@overload
def unique_consecutive(
    x: torch.Tensor,
    return_inverse: Literal[False] = ...,
    return_counts: Literal[True] = ...,
    dim: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    pass


@overload
def unique_consecutive(
    x: torch.Tensor,
    return_inverse: Literal[True] = ...,
    return_counts: Literal[True] = ...,
    dim: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pass


def unique_consecutive(
    x: torch.Tensor,
    return_inverse: bool = False,
    return_counts: bool = False,
    dim: int | None = None,
) -> (
    torch.Tensor
    | tuple[torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
):
    """Like torch.unique_consecutive(), but WAY more efficient.

    The returned unique elements are retrieved along the requested dimension,
    taking all the other dimensions as constant tuples.

    Args:
        x: The input tensor. If it contains equal values, they must be
            consecutive along the given dimension.
            Shape: [N_0, ..., N_dim, ..., N_{D-1}]
        return_inverse: Whether to also return the inverse mapping tensor.
            This can be used to reconstruct the original tensor from the unique
            tensor.
        return_counts: Whether to also return the number of times each unique
            element occurred in the original tensor.
        dim: The dimension to operate on. If None, the unique of the flattened
            input is returned. Otherwise, each of the tensors indexed by the
            given dimension is treated as one of the elements to apply the
            unique operation on. See examples for more details.

    Returns:
        Tuple containing:
        - The unique elements.
            Shape: [N_0, ..., N_{dim-1}, U, N_{dim+1}, ..., N_{D-1}]
        - (Optional) If return_inverse is True, the indices where elements
            in the original input ended up in the returned unique values.
            The original tensor can be reconstructed as follows:
            x_reconstructed = uniques.index_select(dim, inverse)
            Shape: [N_dim]
        - (Optional) If return_counts is True, the counts for each unique
            element.
            Shape: [U]

    Examples:
    >>> # 1D example: -----------------------------------------------------
    >>> x = torch.tensor([9, 9, 9, 9, 10, 10])
    >>> dim = 0

    >>> uniques, inverse, counts = unique_consecutive(
    ...     x, return_inverse=True, return_counts=True, dim=dim
    ... )
    >>> uniques
    tensor([ 9, 10])
    >>> inverse
    tensor([0, 0, 0, 0, 1, 1])
    >>> counts
    tensor([4, 2])

    >>> # Reconstruct the original tensor:
    >>> uniques.index_select(dim, inverse)
    tensor([ 9,  9,  9,  9, 10, 10])

    >>> # 2D example: -----------------------------------------------------
    >>> x = torch.tensor([
    ...     [7, 9, 9, 10],
    ...     [8, 10, 10, 9],
    ...     [9, 8, 8, 7],
    ...     [9, 7, 7, 7],
    ... ])
    >>> dim = 1

    >>> uniques, inverse, counts = unique_consecutive(
    ...     x, return_inverse=True, return_counts=True, dim=dim
    ... )
    >>> uniques
    tensor([[ 7,  9, 10],
            [ 8, 10,  9],
            [ 9,  8,  7],
            [ 9,  7,  7]])
    >>> inverse
    tensor([0, 1, 1, 2])
    >>> counts
    tensor([1, 2, 1])

    >>> # Reconstruct the original tensor:
    >>> uniques.index_select(dim, inverse)
    tensor([[ 7,  9,  9, 10],
            [ 8, 10, 10,  9],
            [ 9,  8,  8,  7],
            [ 9,  7,  7,  7]])

    >>> # 3D example: -----------------------------------------------------
    >>> x = torch.tensor([
    ...     [
    ...         [0, 1, 2, 2],
    ...         [4, 6, 5, 5],
    ...         [9, 8, 7, 7],
    ...     ],
    ...     [
    ...         [4, 2, 8, 8],
    ...         [3, 3, 7, 7],
    ...         [0, 2, 1, 1],
    ...     ],
    ... ])
    >>> dim = 2

    >>> uniques, inverse, counts = unique_consecutive(
    ...     x, return_inverse=True, return_counts=True, dim=dim
    ... )
    >>> uniques
    tensor([[[0, 1, 2],
             [4, 6, 5],
             [9, 8, 7]],
            [[4, 2, 8],
             [3, 3, 7],
             [0, 2, 1]]])
    >>> inverse
    tensor([0, 1, 2, 2])
    >>> counts
    tensor([1, 1, 2])

    >>> # Reconstruct the original tensor:
    >>> uniques.index_select(dim, inverse)
    tensor([[[0, 1, 2, 2],
             [4, 6, 5, 5],
             [9, 8, 7, 7]],
            [[4, 2, 8, 8],
             [3, 3, 7, 7],
             [0, 2, 1, 1]]])
    """
    if dim is None:
        raise NotImplementedError(
            "dim=None is not implemented yet. Please specify a dimension"
            " explicitly."
        )

    # Find each consecutive segment.
    if return_inverse and return_counts:
        outer_idcs, counts, starts = outer_indices_segments(
            x, dim=dim, return_counts=True, return_starts=True
        )  # [N_dim], [U], [U]
    elif return_inverse:
        outer_idcs, starts = outer_indices_segments(
            x, dim=dim, return_starts=True
        )  # [N_dim], [U]
    elif return_counts:
        counts, starts = counts_segments(
            x, dim=dim, return_starts=True
        )  # [U], [U]
    else:
        starts = starts_segments(x, dim=dim)  # [U]

    # Find the unique values.
    uniques = x.index_select(
        dim, starts
    )  # [N_0, ..., N_{dim-1}, U, N_{dim+1}, ..., N_{D-1}]

    # Return the requested values.
    if return_inverse and return_counts:
        return uniques, outer_idcs, counts  # type: ignore
    if return_inverse:
        return uniques, outer_idcs  # type: ignore
    if return_counts:
        return uniques, counts  # type: ignore
    return uniques


@overload
def unique(  # type: ignore
    x: torch.Tensor,
    return_backmap: Literal[False] = ...,
    return_inverse: Literal[False] = ...,
    return_counts: Literal[False] = ...,
    dim: int | None = None,
    stable: bool = False,
) -> torch.Tensor:
    pass


@overload
def unique(
    x: torch.Tensor,
    return_backmap: Literal[True] = ...,
    return_inverse: Literal[False] = ...,
    return_counts: Literal[False] = ...,
    dim: int | None = None,
    stable: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    pass


@overload
def unique(
    x: torch.Tensor,
    return_backmap: Literal[False] = ...,
    return_inverse: Literal[True] = ...,
    return_counts: Literal[False] = ...,
    dim: int | None = None,
    stable: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    pass


@overload
def unique(
    x: torch.Tensor,
    return_backmap: Literal[False] = ...,
    return_inverse: Literal[False] = ...,
    return_counts: Literal[True] = ...,
    dim: int | None = None,
    stable: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    pass


@overload
def unique(
    x: torch.Tensor,
    return_backmap: Literal[True] = ...,
    return_inverse: Literal[True] = ...,
    return_counts: Literal[False] = ...,
    dim: int | None = None,
    stable: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pass


@overload
def unique(
    x: torch.Tensor,
    return_backmap: Literal[True] = ...,
    return_inverse: Literal[False] = ...,
    return_counts: Literal[True] = ...,
    dim: int | None = None,
    stable: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pass


@overload
def unique(
    x: torch.Tensor,
    return_backmap: Literal[False] = ...,
    return_inverse: Literal[True] = ...,
    return_counts: Literal[True] = ...,
    dim: int | None = None,
    stable: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pass


@overload
def unique(
    x: torch.Tensor,
    return_backmap: Literal[True] = ...,
    return_inverse: Literal[True] = ...,
    return_counts: Literal[True] = ...,
    dim: int | None = None,
    stable: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pass


def unique(
    x: torch.Tensor,
    return_backmap: bool = False,
    return_inverse: bool = False,
    return_counts: bool = False,
    dim: int | None = None,
    stable: bool = False,
) -> (
    torch.Tensor
    | tuple[torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
):
    """Like torch.unique(), but WAY more efficient.

    The returned unique elements are retrieved along the requested dimension,
    taking all the other dimensions as constant tuples.

    Args:
        x: The input tensor.
            Shape: [N_0, ..., N_dim, ..., N_{D-1}]
        return_backmap: Whether to also return the backmap tensor.
            This can be used to sort the original tensor.
        return_inverse: Whether to also return the inverse mapping tensor.
            This can be used to reconstruct the original tensor from the unique
            tensor.
        return_counts: Whether to also return the counts of each unique
            element.
        dim: The dimension to operate on. If None, the unique of the flattened
            input is returned. Otherwise, each of the tensors indexed by the
            given dimension is treated as one of the elements to apply the
            unique operation on. See examples for more details.
        stable: Whether to preserve the relative order of equal elements. If
            False (default), an unstable sort is used, which is faster. Note
            that this only has an effect on the backmap tensor, so setting
            stable=True while return_backmap=False will have no effect. We will
            throw an error in this case to avoid degrading performance
            unnecessarily.

    Returns:
        Tuple containing:
        - The unique elements, guaranteed to be sorted along the given
            dimension.
            Shape: [N_0, ..., N_{dim-1}, U, N_{dim+1}, ..., N_{D-1}]
        - (Optional) If return_backmap is True, the backmap tensor, which
            contains the indices of the unique values in the original input.
            The sorted version of x can be retrieved as follows:
            x_sorted = x.index_select(dim, backmap)
            Shape: [N_dim]
        - (Optional) If return_inverse is True, the indices where elements
            in the original input ended up in the returned unique values.
            The original tensor can be reconstructed as follows:
            x_reconstructed = uniques.index_select(dim, inverse)
            Shape: [N_dim]
        - (Optional) If return_counts is True, the counts for each unique
            element.
            Shape: [U]

    Examples:
    >>> # 1D example: -----------------------------------------------------
    >>> x = torch.tensor([9, 10, 9, 9, 10, 9])
    >>> dim = 0

    >>> uniques, backmap, inverse, counts = unique(
    ...     x,
    ...     return_backmap=True,
    ...     return_inverse=True,
    ...     return_counts=True,
    ...     dim=dim,
    ...     stable=True,
    ... )
    >>> uniques
    tensor([ 9, 10])
    >>> backmap
    tensor([0, 2, 3, 5, 1, 4])
    >>> inverse
    tensor([0, 1, 0, 0, 1, 0])
    >>> counts
    tensor([4, 2])

    >>> # Get the lexicographically sorted version of x:
    >>> x.index_select(dim, backmap)
    tensor([ 9,  9,  9,  9, 10, 10])

    >>> # Reconstruct the original tensor:
    >>> uniques.index_select(dim, inverse)
    tensor([ 9, 10,  9,  9, 10,  9])

    >>> # 2D example: -----------------------------------------------------
    >>> x = torch.tensor([
    ...     [9, 10, 7, 9],
    ...     [10, 9, 8, 10],
    ...     [8, 7, 9, 8],
    ...     [7, 7, 9, 7],
    ... ])
    >>> dim = 1

    >>> uniques, backmap, inverse, counts = unique(
    ...     x,
    ...     return_backmap=True,
    ...     return_inverse=True,
    ...     return_counts=True,
    ...     dim=dim,
    ...     stable=True,
    ... )
    >>> uniques
    tensor([[ 7,  9, 10],
            [ 8, 10,  9],
            [ 9,  8,  7],
            [ 9,  7,  7]])
    >>> backmap
    tensor([2, 0, 3, 1])
    >>> inverse
    tensor([1, 2, 0, 1])
    >>> counts
    tensor([1, 2, 1])

    >>> # Get the lexicographically sorted version of x:
    >>> x.index_select(dim, backmap)
    tensor([[ 7,  9,  9, 10],
            [ 8, 10, 10,  9],
            [ 9,  8,  8,  7],
            [ 9,  7,  7,  7]])

    >>> # Reconstruct the original tensor:
    >>> uniques.index_select(dim, inverse)
    tensor([[ 9, 10,  7,  9],
            [10,  9,  8, 10],
            [ 8,  7,  9,  8],
            [ 7,  7,  9,  7]])

    >>> # 3D example: -----------------------------------------------------
    >>> x = torch.tensor([
    ...     [
    ...         [0, 2, 1, 2],
    ...         [4, 5, 6, 5],
    ...         [9, 7, 8, 7],
    ...     ],
    ...     [
    ...         [4, 8, 2, 8],
    ...         [3, 7, 3, 7],
    ...         [0, 1, 2, 1],
    ...     ],
    ... ])
    >>> dim = 2

    >>> uniques, backmap, inverse, counts = unique(
    ...     x,
    ...     return_backmap=True,
    ...     return_inverse=True,
    ...     return_counts=True,
    ...     dim=dim,
    ...     stable=True,
    ... )
    >>> uniques
    tensor([[[0, 1, 2],
             [4, 6, 5],
             [9, 8, 7]],
            [[4, 2, 8],
             [3, 3, 7],
             [0, 2, 1]]])
    >>> backmap
    tensor([0, 2, 1, 3])
    >>> inverse
    tensor([0, 2, 1, 2])
    >>> counts
    tensor([1, 1, 2])

    >>> # Get the lexicographically sorted version of x:
    >>> x.index_select(dim, backmap)
    tensor([[[0, 1, 2, 2],
             [4, 6, 5, 5],
             [9, 8, 7, 7]],
            [[4, 2, 8, 8],
             [3, 3, 7, 7],
             [0, 2, 1, 1]]])

    >>> # Reconstruct the original tensor:
    >>> uniques.index_select(dim, inverse)
    tensor([[[0, 2, 1, 2],
             [4, 5, 6, 5],
             [9, 7, 8, 7]],
            [[4, 8, 2, 8],
             [3, 7, 3, 7],
             [0, 1, 2, 1]]])
    """
    if dim is None:
        raise NotImplementedError(
            "dim=None is not implemented yet. Please specify a dimension"
            " explicitly."
        )

    if stable and not return_backmap:
        raise ValueError(
            "stable=True has no effect when return_backmap=False, but it"
            " degrades performance. Please use either stable=False or"
            " return_backmap=True."
        )

    # Sort along the given dimension, taking all the other dimensions as
    # constant tuples. PyTorch's sort() doesn't work here since it will sort the
    # other dimensions independently.
    x_sorted, backmap = lexsort_along(
        x, dim=dim, stable=stable
    )  # [N_0, ..., N_dim, ..., N_{D-1}], [N_dim]

    out = unique_consecutive(
        x_sorted,
        return_inverse=return_inverse,
        return_counts=return_counts,
        dim=dim,
    )

    aux = []
    if return_backmap:
        aux.append(backmap)
    if return_inverse:
        # The backmap wasn't taken into account by unique_consecutive(), so we
        # have to apply it to the inverse mapping here.
        backmap_inv = swap_idcs_vals(backmap)  # [N_dim]
        aux.append(out[1][backmap_inv])
    if return_counts:
        aux.append(out[-1])

    if aux:
        return out[0], *aux
    return out


# ############################ CONSECUTIVE SEGMENTS ############################


def counts_segments_ints(x: torch.Tensor, high: int) -> torch.Tensor:
    """Count the frequency of each consecutive value, with values in [0, high).

    Args:
        x: The tensor for which to count the frequency of each integer value.
            Consecutive values in x are grouped together. It is assumed that
            every segment has a unique integer value that is not present in any
            other segment. The values in x must be in the range [0, high).
            Shape: [N]
        high: The highest value to include in the count (exclusive). May be
            higher than the maximum value in x, in which case the remaining
            values will be set to 0.

    Returns:
        The frequency of each element in x in range [0, high).
            Shape: [high]

    Examples:
    >>> x = torch.tensor([4, 4, 4, 2, 2, 8, 3, 3, 3, 3])

    >>> freqs = counts_segments_ints(x, 10)
    >>> freqs
    tensor([0, 0, 2, 4, 3, 0, 0, 0, 1, 0])
    """
    if x.ndim != 1:
        raise ValueError("x must be a 1D tensor.")

    device = x.device
    freqs = torch.zeros(high, device=device, dtype=torch.int64)
    uniques, counts = unique_consecutive(
        x, return_counts=True, dim=0
    )  # [U], [U]
    freqs[uniques] = counts
    return freqs


# ################################## GROUPBY ###################################


@overload
def groupby(  # type: ignore
    keys: torch.Tensor,
    vals: torch.Tensor | None = None,
    stable: bool = False,
    as_sequence: Literal[True] = ...,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    pass


@overload
def groupby(
    keys: torch.Tensor,
    vals: torch.Tensor | None = None,
    stable: bool = False,
    as_sequence: Literal[False] = ...,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pass


def groupby(
    keys: torch.Tensor,
    vals: torch.Tensor | None = None,
    stable: bool = False,
    as_sequence: bool = True,
) -> (
    list[tuple[torch.Tensor, torch.Tensor]]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
):
    """Group values by keys.

    Args:
        keys: The keys to group by.
            Shape: [N, *]
        vals: The values to group. If None, the values are set to the indices of
            the keys (i.e. vals = torch.arange(N)).
            Shape: [N, **]
        stable: Whether to preserve the order of vals that have the same key. If
            False (default), an unstable sort is used, which is faster.
        as_sequence: Whether to return the result as a sequence of (key, vals)
            tuples (True) or as packed tensors (False).

    Returns:
        - If as_sequence is True (default), a list of tuples containing:
            - A unique key. Will be yielded in sorted order.
                Shape: [*]
            - The values that correspond to the key. Sorted if stable is True.
                Shape: [N_key, **]
        - If as_sequence is False, a tuple containing:
            - Tensor of unique keys, sorted.
                Shape: [U, *]
            - Tensor of values stored a packed manner, grouped by key. The first
                N_key0 values correspond to the first key, the next N_key1
                values correspond to the second key, etc. Each group of values
                is sorted if stable is True.
                Shape: [N, **]
            - Tensor containing the number of values for each unique key.
                Shape: [U]

    Examples:
    >>> keys = torch.tensor([4, 2, 4, 3, 2, 8, 4])
    >>> vals = torch.tensor([
    ...     [0, 1],
    ...     [2, 3],
    ...     [4, 5],
    ...     [6, 7],
    ...     [8, 9],
    ...     [10, 11],
    ...     [12, 13],
    ... ])

    >>> # Return as sequence of (key, vals) tuples:
    >>> grouped = groupby(keys, vals, stable=True, as_sequence=True)
    >>> for key, vals_group in grouped:
    ...     print("Key:")
    ...     print(key)
    ...     print("Grouped vals:")
    ...     print(vals_group)
    ...     print()
    Key:
    tensor(2)
    Grouped vals:
    tensor([[2, 3],
            [8, 9]])
    <BLANKLINE>
    Key:
    tensor(3)
    Grouped vals:
    tensor([[6, 7]])
    <BLANKLINE>
    Key:
    tensor(4)
    Grouped vals:
    tensor([[ 0,  1],
            [ 4,  5],
            [12, 13]])
    <BLANKLINE>
    Key:
    tensor(8)
    Grouped vals:
    tensor([[10, 11]])

    >>> # Return as packed tensors:
    >>> keys_unique, vals_grouped, counts = groupby(
    ...     keys, vals, stable=True, as_sequence=False
    ... )
    >>> keys_unique
    tensor([2, 3, 4, 8])
    >>> vals_grouped
    tensor([[ 2,  3],
            [ 8,  9],
            [ 6,  7],
            [ 0,  1],
            [ 4,  5],
            [12, 13],
            [10, 11]])
    >>> counts
    tensor([2, 1, 3, 1])
    """
    # Create a mapping from keys to values.
    keys_unique, backmap, counts = unique(
        keys, return_backmap=True, return_counts=True, dim=0, stable=stable
    )  # [U, *], [N], [U]

    if vals is None:
        # Use the backmap directly as values.
        vals_grouped = backmap  # [N]
    else:
        # Rearrange values to match keys_unique.
        vals_grouped = vals.index_select(0, backmap)  # [N, **]

    # Return the results.
    if not as_sequence:
        return keys_unique, vals_grouped, counts

    # Create the sequences of (key, vals_group) tuples.
    return list(zip(keys_unique, sequentialize_packed(vals_grouped, counts)))
