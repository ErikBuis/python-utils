from __future__ import annotations

import unittest

import numpy as np
import pytest
import torch

from python_utils.modules.torch import (
    apply_mask,
    interp,
    last_valid_value_padding,
    lexsort_along,
    mask_padding,
    pack_padded,
    pack_sequence,
    pad_packed,
    pad_sequence,
    ravel_multi_index,
    replace_padding,
    sequentialize_packed,
    sequentialize_padded,
    swap_idcs_vals,
    swap_idcs_vals_duplicates,
    unique,
)


@pytest.fixture
def simple_packed() -> torch.Tensor:
    return torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])


@pytest.fixture
def simple_padded_zeros() -> torch.Tensor:
    return torch.tensor(
        [[0.1, 0.2, 0.3, 0], [0.4, 0, 0, 0], [0.5, 0.6, 0.7, 0.8]]
    )


@pytest.fixture
def simple_padded_ones() -> torch.Tensor:
    return torch.tensor(
        [[0.1, 0.2, 0.3, 1], [0.4, 1, 1, 1], [0.5, 0.6, 0.7, 0.8]]
    )


@pytest.fixture
def simple_padded_neg_ones() -> torch.Tensor:
    return torch.tensor(
        [[0.1, 0.2, 0.3, -1], [0.4, -1, -1, -1], [0.5, 0.6, 0.7, 0.8]]
    )


@pytest.fixture
def simple_sequence() -> list[torch.Tensor]:
    return [
        torch.tensor([0.1, 0.2, 0.3]),
        torch.tensor([0.4]),
        torch.tensor([0.5, 0.6, 0.7, 0.8]),
    ]


@pytest.fixture
def simple_L_bs() -> torch.Tensor:
    return torch.tensor([3, 1, 4])


@pytest.fixture
def simple_mask() -> torch.Tensor:
    return torch.tensor([
        [True, True, True, False],
        [True, False, False, False],
        [True, True, True, True],
    ])


@pytest.fixture
def multidim_packed() -> torch.Tensor:
    return torch.tensor([
        [0.1, 0.2],
        [0.3, 0.4],
        [0.5, 0.6],
        [0.7, 0.8],
        [0.9, 1.0],
        [1.1, 1.2],
        [1.3, 1.4],
        [1.5, 1.6],
    ])


@pytest.fixture
def multidim_padded() -> torch.Tensor:
    return torch.tensor([
        [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0, 0]],
        [[0.7, 0.8], [0, 0], [0, 0], [0, 0]],
        [[0.9, 1.0], [1.1, 1.2], [1.3, 1.4], [1.5, 1.6]],
    ])


@pytest.fixture
def multidim_sequence() -> list[torch.Tensor]:
    return [
        torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),
        torch.tensor([[0.7, 0.8]]),
        torch.tensor([[0.9, 1.0], [1.1, 1.2], [1.3, 1.4], [1.5, 1.6]]),
    ]


@pytest.fixture
def multidim_L_bs() -> torch.Tensor:
    return torch.tensor([3, 1, 4])


class TestMaskPadding:
    def test_mask_padding_simple(
        self, simple_L_bs: torch.Tensor, simple_mask: torch.Tensor
    ) -> None:
        max_L_bs = int(simple_L_bs.max())
        mask = mask_padding(simple_L_bs, max_L_bs)
        assert torch.equal(mask, simple_mask)

    def test_mask_padding_all_zeros(self) -> None:
        L_bs = torch.tensor([0, 0, 0])
        max_L_bs = int(L_bs.max())
        mask = mask_padding(L_bs, max_L_bs)
        expected_mask = torch.empty((3, 0), dtype=torch.bool)
        assert torch.equal(mask, expected_mask)

    def test_mask_padding_empty(self) -> None:
        L_bs = torch.empty(0, dtype=torch.int32)
        max_L_bs = 0
        mask = mask_padding(L_bs, max_L_bs)
        expected_mask = torch.empty((0, 0), dtype=torch.bool)
        assert torch.equal(mask, expected_mask)

    def test_mask_padding_max_L_bs_too_high(self) -> None:
        L_bs = torch.tensor([3, 1, 4])
        max_L_bs = 5
        mask = mask_padding(L_bs, max_L_bs)
        expected_mask = torch.tensor([
            [True, True, True, False, False],
            [True, False, False, False, False],
            [True, True, True, True, False],
        ])
        assert torch.equal(mask, expected_mask)


class TestPackPadded:
    def test_pack_padded_simple(
        self,
        simple_padded_zeros: torch.Tensor,
        simple_L_bs: torch.Tensor,
        simple_packed: torch.Tensor,
    ) -> None:
        packed = pack_padded(simple_padded_zeros, simple_L_bs)
        assert torch.equal(packed, simple_packed)

    def test_pack_padded_multi_dimensional(
        self,
        multidim_padded: torch.Tensor,
        multidim_L_bs: torch.Tensor,
        multidim_packed: torch.Tensor,
    ) -> None:
        packed = pack_padded(multidim_padded, multidim_L_bs)
        assert torch.equal(packed, multidim_packed)

    def test_pack_padded_all_zeros(self) -> None:
        values = torch.empty((3, 0))
        L_bs = torch.tensor([0, 0, 0])
        packed = pack_padded(values, L_bs)
        expected_packed = torch.empty(0)
        assert torch.equal(packed, expected_packed)

    def test_pack_padded_empty(self) -> None:
        values = torch.empty((0, 0))
        L_bs = torch.empty(0, dtype=torch.int32)
        packed = pack_padded(values, L_bs)
        expected_packed = torch.empty(0)
        assert torch.equal(packed, expected_packed)


class TestPackSequence:
    def test_pack_sequence_simple(
        self, simple_sequence: list[torch.Tensor], simple_packed: torch.Tensor
    ) -> None:
        packed = pack_sequence(simple_sequence)
        assert torch.equal(packed, simple_packed)

    def test_pack_sequence_multi_dimensional(
        self,
        multidim_sequence: list[torch.Tensor],
        multidim_packed: torch.Tensor,
    ) -> None:
        packed = pack_sequence(multidim_sequence)
        assert torch.equal(packed, multidim_packed)

    def test_pack_sequence_all_zeros(self) -> None:
        values = [torch.empty(0), torch.empty(0), torch.empty(0)]
        packed = pack_sequence(values)
        expected_packed = torch.empty(0)
        assert torch.equal(packed, expected_packed)

    def test_pack_sequence_empty(self) -> None:
        values = []
        packed = pack_sequence(values)
        expected_packed = torch.empty(0)
        assert torch.equal(packed, expected_packed)


class TestPadPacked:
    def test_pad_packed_simple(
        self,
        simple_packed: torch.Tensor,
        simple_L_bs: torch.Tensor,
        simple_padded_zeros: torch.Tensor,
    ) -> None:
        max_L_bs = int(simple_L_bs.max())
        padded = pad_packed(simple_packed, simple_L_bs, max_L_bs)
        assert all(
            torch.equal(
                padded[b, : simple_L_bs[b]],
                simple_padded_zeros[b, : simple_L_bs[b]],
            )
            for b in range(len(simple_L_bs))
        )

    def test_pad_packed_padding_value_zero(
        self,
        simple_packed: torch.Tensor,
        simple_L_bs: torch.Tensor,
        simple_padded_zeros: torch.Tensor,
    ) -> None:
        max_L_bs = int(simple_L_bs.max())
        padded = pad_packed(
            simple_packed, simple_L_bs, max_L_bs, padding_value=0
        )
        assert torch.equal(padded, simple_padded_zeros)

    def test_pad_packed_padding_value_one(
        self,
        simple_packed: torch.Tensor,
        simple_L_bs: torch.Tensor,
        simple_padded_ones: torch.Tensor,
    ) -> None:
        max_L_bs = int(simple_L_bs.max())
        padded = pad_packed(
            simple_packed, simple_L_bs, max_L_bs, padding_value=1
        )
        assert torch.equal(padded, simple_padded_ones)

    def test_pad_packed_padding_value_other(
        self,
        simple_packed: torch.Tensor,
        simple_L_bs: torch.Tensor,
        simple_padded_neg_ones: torch.Tensor,
    ) -> None:
        max_L_bs = int(simple_L_bs.max())
        padded = pad_packed(
            simple_packed, simple_L_bs, max_L_bs, padding_value=-1
        )
        assert torch.equal(padded, simple_padded_neg_ones)

    def test_pad_packed_multi_dimensional(
        self,
        multidim_packed: torch.Tensor,
        multidim_L_bs: torch.Tensor,
        multidim_padded: torch.Tensor,
    ) -> None:
        max_L_bs = int(multidim_L_bs.max())
        padded = pad_packed(multidim_packed, multidim_L_bs, max_L_bs)
        assert all(
            torch.equal(
                padded[b, : multidim_L_bs[b]],
                multidim_padded[b, : multidim_L_bs[b]],
            )
            for b in range(len(multidim_L_bs))
        )

    def test_pad_packed_all_zeros(self) -> None:
        values = torch.empty(0)
        L_bs = torch.tensor([0, 0, 0])
        max_L_bs = int(L_bs.max())
        padded = pad_packed(values, L_bs, max_L_bs)
        expected_padded = torch.empty((3, 0))
        assert torch.equal(padded, expected_padded)

    def test_pad_packed_empty(self) -> None:
        values = torch.empty(0)
        L_bs = torch.empty(0, dtype=torch.int32)
        max_L_bs = 0
        padded = pad_packed(values, L_bs, max_L_bs)
        expected_padded = torch.empty((0, 0))
        assert torch.equal(padded, expected_padded)


class TestPadSequence:
    def test_pad_sequence_simple(
        self,
        simple_sequence: list[torch.Tensor],
        simple_L_bs: torch.Tensor,
        simple_padded_zeros: torch.Tensor,
    ) -> None:
        max_L_bs = int(simple_L_bs.max())
        padded = pad_sequence(simple_sequence, simple_L_bs, max_L_bs)
        assert all(
            torch.equal(
                padded[b, : simple_L_bs[b]],
                simple_padded_zeros[b, : simple_L_bs[b]],
            )
            for b in range(len(simple_L_bs))
        )

    def test_pad_sequence_padding_value_zero(
        self,
        simple_sequence: list[torch.Tensor],
        simple_L_bs: torch.Tensor,
        simple_padded_zeros: torch.Tensor,
    ) -> None:
        max_L_bs = int(simple_L_bs.max())
        padded = pad_sequence(
            simple_sequence, simple_L_bs, max_L_bs, padding_value=0
        )
        assert torch.equal(padded, simple_padded_zeros)

    def test_pad_sequence_padding_value_one(
        self,
        simple_sequence: list[torch.Tensor],
        simple_L_bs: torch.Tensor,
        simple_padded_ones: torch.Tensor,
    ) -> None:
        max_L_bs = int(simple_L_bs.max())
        padded = pad_sequence(
            simple_sequence, simple_L_bs, max_L_bs, padding_value=1
        )
        assert torch.equal(padded, simple_padded_ones)

    def test_pad_sequence_padding_value_other(
        self,
        simple_sequence: list[torch.Tensor],
        simple_L_bs: torch.Tensor,
        simple_padded_neg_ones: torch.Tensor,
    ) -> None:
        max_L_bs = int(simple_L_bs.max())
        padded = pad_sequence(
            simple_sequence, simple_L_bs, max_L_bs, padding_value=-1
        )
        assert torch.equal(padded, simple_padded_neg_ones)

    def test_pad_sequence_multi_dimensional(
        self,
        multidim_sequence: list[torch.Tensor],
        multidim_L_bs: torch.Tensor,
        multidim_padded: torch.Tensor,
    ) -> None:
        max_L_bs = int(multidim_L_bs.max())
        padded = pad_sequence(multidim_sequence, multidim_L_bs, max_L_bs)
        assert all(
            torch.equal(
                padded[b, : multidim_L_bs[b]],
                multidim_padded[b, : multidim_L_bs[b]],
            )
            for b in range(len(multidim_L_bs))
        )

    def test_pad_sequence_all_zeros(self) -> None:
        values = [torch.empty(0), torch.empty(0), torch.empty(0)]
        L_bs = torch.tensor([0, 0, 0])
        max_L_bs = int(L_bs.max())
        padded = pad_sequence(values, L_bs, max_L_bs)
        expected_padded = torch.empty((3, 0))
        assert torch.equal(padded, expected_padded)

    def test_pad_sequence_empty(self) -> None:
        values = []
        L_bs = torch.empty(0, dtype=torch.int32)
        max_L_bs = 0
        padded = pad_sequence(values, L_bs, max_L_bs)
        expected_padded = torch.empty((0, 0))
        assert torch.equal(padded, expected_padded)


class TestSequentializePacked:
    def test_sequentialize_packed_simple(
        self,
        simple_packed: torch.Tensor,
        simple_L_bs: torch.Tensor,
        simple_sequence: list[torch.Tensor],
    ) -> None:
        sequentialized = sequentialize_packed(simple_packed, simple_L_bs)
        assert all(
            torch.equal(sequentialized[b], simple_sequence[b])
            for b in range(len(simple_L_bs))
        )

    def test_sequentialize_packed_multi_dimensional(
        self,
        multidim_packed: torch.Tensor,
        multidim_L_bs: torch.Tensor,
        multidim_sequence: list[torch.Tensor],
    ) -> None:
        sequentialized = sequentialize_packed(multidim_packed, multidim_L_bs)
        assert all(
            torch.equal(sequentialized[b], multidim_sequence[b])
            for b in range(len(multidim_L_bs))
        )

    def test_sequentialize_packed_all_zeros(self) -> None:
        values = torch.empty(0)
        L_bs = torch.tensor([0, 0, 0])
        sequentialized = sequentialize_packed(values, L_bs)
        expected_sequentialized = [
            torch.empty(0),
            torch.empty(0),
            torch.empty(0),
        ]
        assert all(
            torch.equal(sequentialized[b], expected_sequentialized[b])
            for b in range(len(L_bs))
        )

    def test_sequentialize_packed_empty(self) -> None:
        values = torch.empty(0)
        L_bs = torch.empty(0, dtype=torch.int32)
        sequentialized = sequentialize_packed(values, L_bs)
        expected_sequentialized = []
        assert sequentialized == expected_sequentialized


class TestSequentializePadded:
    def test_sequentialize_padded_simple(
        self,
        simple_padded_zeros: torch.Tensor,
        simple_L_bs: torch.Tensor,
        simple_sequence: list[torch.Tensor],
    ) -> None:
        sequentialized = sequentialize_padded(simple_padded_zeros, simple_L_bs)
        assert all(
            torch.equal(sequentialized[b], simple_sequence[b])
            for b in range(len(simple_L_bs))
        )

    def test_sequentialize_padded_multi_dimensional(
        self,
        multidim_padded: torch.Tensor,
        multidim_L_bs: torch.Tensor,
        multidim_sequence: list[torch.Tensor],
    ) -> None:
        sequentialized = sequentialize_padded(multidim_padded, multidim_L_bs)
        assert all(
            torch.equal(sequentialized[b], multidim_sequence[b])
            for b in range(len(multidim_L_bs))
        )

    def test_sequentialize_padded_all_zeros(self) -> None:
        values = torch.empty((3, 0))
        L_bs = torch.tensor([0, 0, 0])
        sequentialized = sequentialize_padded(values, L_bs)
        expected_sequentialized = [
            torch.empty(0),
            torch.empty(0),
            torch.empty(0),
        ]
        assert all(
            torch.equal(sequentialized[b], expected_sequentialized[b])
            for b in range(len(L_bs))
        )

    def test_sequentialize_padded_empty(self) -> None:
        values = torch.empty((0, 0))
        L_bs = torch.empty(0, dtype=torch.int32)
        sequentialized = sequentialize_padded(values, L_bs)
        expected_sequentialized = []
        assert sequentialized == expected_sequentialized


class TestApplyMask:
    def test_apply_mask_simple(
        self, simple_padded_zeros: torch.Tensor, simple_L_bs: torch.Tensor
    ) -> None:
        # Mask that keeps indices [0, 2] from first, [] from second, [0, 2, 3]
        # from third.
        mask = torch.tensor([
            [True, False, True, False],
            [False, False, False, False],
            [True, False, True, True],
        ])
        values_filtered, L_bs_kept = apply_mask(
            simple_padded_zeros, mask, simple_L_bs, padding_value=0
        )
        expected_values = torch.tensor(
            [[0.1, 0.3, 0], [0, 0, 0], [0.5, 0.7, 0.8]]
        )
        expected_L_bs_kept = torch.tensor([2, 0, 3])
        assert torch.equal(values_filtered, expected_values)
        assert torch.equal(L_bs_kept, expected_L_bs_kept)

    def test_apply_mask_multi_dimensional(
        self, multidim_padded: torch.Tensor, multidim_L_bs: torch.Tensor
    ) -> None:
        # Mask that keeps indices [0, 2] from first, [] from second, [0, 2, 3]
        # from third.
        mask = torch.tensor([
            [True, False, True, False],
            [False, False, False, False],
            [True, False, True, True],
        ])
        values_filtered, L_bs_kept = apply_mask(
            multidim_padded, mask, multidim_L_bs, padding_value=0
        )
        expected_values = torch.tensor([
            [[0.1, 0.2], [0.5, 0.6], [0, 0]],
            [[0, 0], [0, 0], [0, 0]],
            [[0.9, 1.0], [1.3, 1.4], [1.5, 1.6]],
        ])
        expected_L_bs_kept = torch.tensor([2, 0, 3])
        assert torch.equal(values_filtered, expected_values)
        assert torch.equal(L_bs_kept, expected_L_bs_kept)

    def test_apply_mask_all_true(
        self,
        simple_padded_zeros: torch.Tensor,
        simple_padded_neg_ones: torch.Tensor,
        simple_L_bs: torch.Tensor,
    ) -> None:
        # Mask keeps all valid elements, changing padding from 0 to -1 to check
        # that the old padding is properly replaced.
        mask = torch.tensor([
            [True, True, True, True],
            [True, True, True, True],
            [True, True, True, True],
        ])
        values_filtered, L_bs_kept = apply_mask(
            simple_padded_zeros, mask, simple_L_bs, padding_value=-1
        )
        assert torch.equal(values_filtered, simple_padded_neg_ones)
        assert torch.equal(L_bs_kept, simple_L_bs)

    def test_apply_mask_all_false(
        self, simple_padded_zeros: torch.Tensor, simple_L_bs: torch.Tensor
    ) -> None:
        # Mask removes all elements.
        mask = torch.tensor([
            [False, False, False, False],
            [False, False, False, False],
            [False, False, False, False],
        ])
        values_filtered, L_bs_kept = apply_mask(
            simple_padded_zeros, mask, simple_L_bs, padding_value=0
        )
        expected_values = torch.empty((3, 0))
        expected_L_bs_kept = torch.tensor([0, 0, 0])
        assert torch.equal(values_filtered, expected_values)
        assert torch.equal(L_bs_kept, expected_L_bs_kept)

    def test_apply_mask_all_zeros(self) -> None:
        values = torch.empty((3, 0))
        L_bs = torch.tensor([0, 0, 0])
        mask = torch.empty((3, 0), dtype=torch.bool)
        values_filtered, L_bs_kept = apply_mask(
            values, mask, L_bs, padding_value=0
        )
        assert torch.equal(values_filtered, values)
        assert torch.equal(L_bs_kept, L_bs)

    def test_apply_mask_empty(self) -> None:
        values = torch.empty((0, 0))
        L_bs = torch.empty(0, dtype=torch.int32)
        mask = torch.empty((0, 0), dtype=torch.bool)
        values_filtered, L_bs_kept = apply_mask(
            values, mask, L_bs, padding_value=0
        )
        assert torch.equal(values_filtered, values)
        assert torch.equal(L_bs_kept, L_bs)


class TestReplacePadding:
    def test_replace_padding_scalar(
        self,
        simple_padded_zeros: torch.Tensor,
        simple_padded_ones: torch.Tensor,
        simple_L_bs: torch.Tensor,
    ) -> None:
        # Padding value as scalar.
        values_repadded = replace_padding(
            simple_padded_zeros, simple_L_bs, padding_value=1
        )
        assert torch.equal(values_repadded, simple_padded_ones)

    def test_replace_padding_all(
        self,
        simple_padded_zeros: torch.Tensor,
        simple_padded_ones: torch.Tensor,
        simple_L_bs: torch.Tensor,
    ) -> None:
        # Padding value with shape [].
        padding_value = torch.tensor(1)
        values_repadded = replace_padding(
            simple_padded_zeros, simple_L_bs, padding_value=padding_value
        )
        assert torch.equal(values_repadded, simple_padded_ones)

    def test_replace_padding_all_multi_dimensional(
        self, multidim_padded: torch.Tensor, multidim_L_bs: torch.Tensor
    ) -> None:
        # Padding value with shape [*] = [2].
        padding_value = torch.tensor([1, 2])
        values_repadded = replace_padding(
            multidim_padded, multidim_L_bs, padding_value=padding_value
        )
        expected_values = torch.tensor([
            [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [1, 2]],
            [[0.7, 0.8], [1, 2], [1, 2], [1, 2]],
            [[0.9, 1.0], [1.1, 1.2], [1.3, 1.4], [1.5, 1.6]],
        ])
        assert torch.equal(values_repadded, expected_values)

    def test_replace_padding_per_element(
        self, simple_padded_zeros: torch.Tensor, simple_L_bs: torch.Tensor
    ) -> None:
        # Padding value with shape [B, max(L_bs)] = [3, 4].
        padding_value = torch.tensor(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
        )
        values_repadded = replace_padding(
            simple_padded_zeros, simple_L_bs, padding_value=padding_value
        )
        expected_values = torch.tensor(
            [[0.1, 0.2, 0.3, 4], [0.4, 6, 7, 8], [0.5, 0.6, 0.7, 0.8]]
        )
        assert torch.equal(values_repadded, expected_values)

    def test_replace_padding_per_element_multi_dimensional(
        self, multidim_padded: torch.Tensor, multidim_L_bs: torch.Tensor
    ) -> None:
        # Padding value with shape [B, max(L_bs), *] = [3, 4, 2].
        padding_value = torch.tensor([
            [[1, 2], [3, 4], [5, 6], [7, 8]],
            [[9, 10], [11, 12], [13, 14], [15, 16]],
            [[17, 18], [19, 20], [21, 22], [23, 24]],
        ])
        values_repadded = replace_padding(
            multidim_padded, multidim_L_bs, padding_value=padding_value
        )
        expected_values = torch.tensor([
            [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [7, 8]],
            [[0.7, 0.8], [11, 12], [13, 14], [15, 16]],
            [[0.9, 1.0], [1.1, 1.2], [1.3, 1.4], [1.5, 1.6]],
        ])
        assert torch.equal(values_repadded, expected_values)

    def test_replace_padding_per_row(
        self, simple_padded_zeros: torch.Tensor, simple_L_bs: torch.Tensor
    ) -> None:
        # Padding value with shape [B, 1] = [3, 1].
        padding_value = torch.tensor([[1], [2], [3]])
        values_repadded = replace_padding(
            simple_padded_zeros, simple_L_bs, padding_value=padding_value
        )
        expected_values = torch.tensor(
            [[0.1, 0.2, 0.3, 1], [0.4, 2, 2, 2], [0.5, 0.6, 0.7, 0.8]]
        )
        assert torch.equal(values_repadded, expected_values)

    def test_replace_padding_per_row_multi_dimensional(
        self, multidim_padded: torch.Tensor, multidim_L_bs: torch.Tensor
    ) -> None:
        # Padding value with shape [B, 1, *] = [3, 1, 2].
        padding_value = torch.tensor([[[1, 2]], [[3, 4]], [[5, 6]]])
        values_repadded = replace_padding(
            multidim_padded, multidim_L_bs, padding_value=padding_value
        )
        expected_values = torch.tensor([
            [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [1, 2]],
            [[0.7, 0.8], [3, 4], [3, 4], [3, 4]],
            [[0.9, 1.0], [1.1, 1.2], [1.3, 1.4], [1.5, 1.6]],
        ])
        assert torch.equal(values_repadded, expected_values)

    def test_replace_padding_per_col(
        self, simple_padded_zeros: torch.Tensor, simple_L_bs: torch.Tensor
    ) -> None:
        # Padding value with shape [1, max(L_bs)] = [1, 4].
        padding_value = torch.tensor([[1, 2, 3, 4]])
        values_repadded = replace_padding(
            simple_padded_zeros, simple_L_bs, padding_value=padding_value
        )
        expected_values = torch.tensor(
            [[0.1, 0.2, 0.3, 4], [0.4, 2, 3, 4], [0.5, 0.6, 0.7, 0.8]]
        )
        assert torch.equal(values_repadded, expected_values)

    def test_replace_padding_per_col_multi_dimensional(
        self, multidim_padded: torch.Tensor, multidim_L_bs: torch.Tensor
    ) -> None:
        # Padding value with shape [1, max(L_bs), *] = [1, 4, 2].
        padding_value = torch.tensor([[[1, 2], [3, 4], [5, 6], [7, 8]]])
        values_repadded = replace_padding(
            multidim_padded, multidim_L_bs, padding_value=padding_value
        )
        expected_values = torch.tensor([
            [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [7, 8]],
            [[0.7, 0.8], [3, 4], [5, 6], [7, 8]],
            [[0.9, 1.0], [1.1, 1.2], [1.3, 1.4], [1.5, 1.6]],
        ])
        assert torch.equal(values_repadded, expected_values)

    def test_replace_padding_in_mask(
        self, simple_padded_zeros: torch.Tensor, simple_L_bs: torch.Tensor
    ) -> None:
        # Padding value with shape [B * max(L_bs) - L] = [3*4 - 8] = [4].
        padding_value = torch.tensor([1, 2, 3, 4])
        values_repadded = replace_padding(
            simple_padded_zeros, simple_L_bs, padding_value=padding_value
        )
        expected_values = torch.tensor(
            [[0.1, 0.2, 0.3, 1], [0.4, 2, 3, 4], [0.5, 0.6, 0.7, 0.8]]
        )
        assert torch.equal(values_repadded, expected_values)

    def test_replace_padding_in_mask_multi_dimensional(
        self, multidim_padded: torch.Tensor, multidim_L_bs: torch.Tensor
    ) -> None:
        # Padding value with shape [B * max(L_bs) - L, *] = [3*4 - 8, 2]
        # = [4, 2].
        padding_value = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]])
        values_repadded = replace_padding(
            multidim_padded, multidim_L_bs, padding_value=padding_value
        )
        expected_values = torch.tensor([
            [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [1, 2]],
            [[0.7, 0.8], [3, 4], [5, 6], [7, 8]],
            [[0.9, 1.0], [1.1, 1.2], [1.3, 1.4], [1.5, 1.6]],
        ])
        assert torch.equal(values_repadded, expected_values)

    def test_replace_padding_in_place(
        self, simple_padded_zeros: torch.Tensor, simple_L_bs: torch.Tensor
    ) -> None:
        # Verify in_place=True modifies the original tensor.
        values_original = simple_padded_zeros.clone()
        values_repadded = replace_padding(
            values_original, simple_L_bs, padding_value=1, in_place=True
        )
        expected_values = torch.tensor(
            [[0.1, 0.2, 0.3, 1], [0.4, 1, 1, 1], [0.5, 0.6, 0.7, 0.8]]
        )
        assert values_repadded is values_original
        assert torch.equal(values_original, expected_values)

    def test_replace_padding_all_zeros(self) -> None:
        values = torch.empty((3, 0))
        L_bs = torch.tensor([0, 0, 0])
        values_repadded = replace_padding(values, L_bs, padding_value=-1)
        assert torch.equal(values_repadded, values)

    def test_replace_padding_empty(self) -> None:
        values = torch.empty((0, 0))
        L_bs = torch.empty(0, dtype=torch.int32)
        values_repadded = replace_padding(values, L_bs, padding_value=-1)
        assert torch.equal(values_repadded, values)


class TestLastValidValuePadding:
    def test_last_valid_value_padding_simple(
        self, simple_padded_zeros: torch.Tensor, simple_L_bs: torch.Tensor
    ) -> None:
        # Pad with last valid value from each row.
        values_repadded = last_valid_value_padding(
            simple_padded_zeros, simple_L_bs
        )
        expected_values = torch.tensor(
            [[0.1, 0.2, 0.3, 0.3], [0.4, 0.4, 0.4, 0.4], [0.5, 0.6, 0.7, 0.8]]
        )
        assert torch.equal(values_repadded, expected_values)

    def test_last_valid_value_padding_empty_rows_with_value(
        self, simple_padded_zeros: torch.Tensor
    ) -> None:
        # Test empty rows (L_b == 0) with padding_value_empty_rows specified.
        L_bs = torch.tensor([2, 0, 3])
        values_repadded = last_valid_value_padding(
            simple_padded_zeros, L_bs, padding_value_empty_rows=-1
        )
        expected_values = torch.tensor(
            [[0.1, 0.2, 0.2, 0.2], [-1, -1, -1, -1], [0.5, 0.6, 0.7, 0.7]]
        )
        assert torch.equal(values_repadded, expected_values)

    def test_last_valid_value_padding_empty_rows_without_value(
        self, simple_padded_zeros: torch.Tensor
    ) -> None:
        # Test empty rows (L_b == 0) with padding_value_empty_rows=None.
        # No assertion on padding content for empty row.
        L_bs = torch.tensor([2, 0, 3])
        values_repadded = last_valid_value_padding(
            simple_padded_zeros, L_bs, padding_value_empty_rows=None
        )
        expected_values_row_1 = torch.tensor([0.1, 0.2, 0.2, 0.2])
        expected_values_row_3 = torch.tensor([0.5, 0.6, 0.7, 0.7])
        assert torch.equal(values_repadded[0], expected_values_row_1)
        assert torch.equal(values_repadded[2], expected_values_row_3)

    def test_last_valid_value_padding_multi_dimensional(
        self, multidim_padded: torch.Tensor, multidim_L_bs: torch.Tensor
    ) -> None:
        # Pad with last valid value for multidimensional values.
        values_repadded = last_valid_value_padding(
            multidim_padded, multidim_L_bs
        )
        expected_values = torch.tensor([
            [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.5, 0.6]],
            [[0.7, 0.8], [0.7, 0.8], [0.7, 0.8], [0.7, 0.8]],
            [[0.9, 1.0], [1.1, 1.2], [1.3, 1.4], [1.5, 1.6]],
        ])
        assert torch.equal(values_repadded, expected_values)

    def test_last_valid_value_padding_in_place(
        self, simple_padded_zeros: torch.Tensor, simple_L_bs: torch.Tensor
    ) -> None:
        # Verify in_place=True modifies the original tensor.
        values_original = simple_padded_zeros.clone()
        values_repadded = last_valid_value_padding(
            values_original, simple_L_bs, in_place=True
        )
        expected_values = torch.tensor(
            [[0.1, 0.2, 0.3, 0.3], [0.4, 0.4, 0.4, 0.4], [0.5, 0.6, 0.7, 0.8]]
        )
        assert values_repadded is values_original
        assert torch.equal(values_original, expected_values)

    def test_last_valid_value_padding_all_empty_rows(
        self, simple_padded_zeros: torch.Tensor
    ) -> None:
        # Verify all rows empty (L_b == 0) with padding_value_empty_rows
        # specified.
        L_bs = torch.tensor([0, 0, 0])
        values_repadded = last_valid_value_padding(
            simple_padded_zeros, L_bs, padding_value_empty_rows=1
        )
        expected_values = torch.tensor(
            [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
        )
        assert torch.equal(values_repadded, expected_values)

    def test_last_valid_value_padding_all_zeros(self) -> None:
        values = torch.empty((3, 0))
        L_bs = torch.tensor([0, 0, 0])
        values_repadded = last_valid_value_padding(
            values, L_bs, padding_value_empty_rows=1
        )
        assert torch.equal(values_repadded, values)

    def test_last_valid_value_padding_empty(self) -> None:
        values = torch.empty((0, 0))
        L_bs = torch.empty(0, dtype=torch.int32)
        values_repadded = last_valid_value_padding(
            values, L_bs, padding_value_empty_rows=1
        )
        assert torch.equal(values_repadded, values)


class TestInterp(unittest.TestCase):
    def test_interp_equivalent_np(self) -> None:
        x = torch.rand(100) * 102 - 1  # in [-1, 101)
        xp = (torch.rand(100)).sort().values * 100  # in [0, 100)
        fp = torch.rand(100)  # in [0, 1)
        left = -1
        right = 101
        self.assertTrue(
            torch.allclose(
                interp(x, xp, fp, left, right),
                torch.from_numpy(
                    np.interp(x, xp, fp, left, right).astype(np.float32)
                ),
            )
        )


class TestRavelMultiIndex(unittest.TestCase):
    def test_ravel_multi_index_equivalent_np(self) -> None:
        dims = torch.arange(10, 20)  # [10]
        multi_index = torch.stack(
            [torch.randint(0, int(dim), (10, 10)) for dim in dims]
        )  # [10, 10, 10]
        self.assertTrue(
            torch.equal(
                ravel_multi_index(multi_index, dims),
                torch.from_numpy(
                    np.ravel_multi_index(
                        tuple(multi_index.numpy(force=True)),
                        tuple(dims.numpy(force=True)),
                    )
                ),
            )
        )
        self.assertTrue(
            torch.equal(
                ravel_multi_index(multi_index, dims, order="F"),
                torch.from_numpy(
                    np.ravel_multi_index(
                        tuple(multi_index.numpy(force=True)),
                        tuple(dims.numpy(force=True)),
                        order="F",
                    )
                ),
            )
        )
        multi_index = torch.stack([
            torch.concat([
                torch.randint(-2 * int(dim), -int(dim), (5, 10)),
                torch.randint(int(dim), 2 * int(dim), (5, 10)),
            ])
            for dim in dims
        ])  # [10, 10, 10]
        self.assertRaises(ValueError, ravel_multi_index, multi_index, dims)
        self.assertTrue(
            torch.equal(
                ravel_multi_index(multi_index, dims, mode="wrap"),
                torch.from_numpy(
                    np.ravel_multi_index(
                        tuple(multi_index.numpy(force=True)),
                        tuple(dims.numpy(force=True)),
                        mode="wrap",
                    )
                ),
            )
        )
        self.assertTrue(
            torch.equal(
                ravel_multi_index(multi_index, dims, mode="clip"),
                torch.from_numpy(
                    np.ravel_multi_index(
                        tuple(multi_index.numpy(force=True)),
                        tuple(dims.numpy(force=True)),
                        mode="clip",
                    )
                ),
            )
        )


class TestSwapIdcsVals(unittest.TestCase):
    def test_swap_idcs_vals_len5(self) -> None:
        x = torch.tensor([2, 3, 0, 4, 1])
        self.assertTrue(
            torch.equal(swap_idcs_vals(x), torch.tensor([2, 4, 0, 1, 3]))
        )

    def test_swap_idcs_vals_len10(self) -> None:
        x = torch.tensor([6, 3, 0, 1, 4, 7, 2, 8, 9, 5])
        self.assertTrue(
            torch.equal(
                swap_idcs_vals(x), torch.tensor([2, 3, 6, 1, 4, 9, 0, 5, 7, 8])
            )
        )

    def test_swap_idcs_vals_2D(self) -> None:
        x = torch.tensor([[2, 3], [0, 4], [1, 5]])
        with self.assertRaises(ValueError):
            swap_idcs_vals(x)


class TestSwapIdcsValsDuplicates(unittest.TestCase):
    def test_swap_idcs_vals_duplicates_len5(self) -> None:
        x = torch.tensor([1, 2, 0, 1, 2])
        self.assertTrue(
            torch.equal(
                swap_idcs_vals_duplicates(x, stable=True),
                torch.tensor([2, 0, 3, 1, 4]),
            )
        )

    def test_swap_idcs_vals_duplicates_len10(self) -> None:
        x = torch.tensor([3, 3, 0, 3, 4, 2, 1, 1, 2, 0])
        self.assertTrue(
            torch.equal(
                swap_idcs_vals_duplicates(x, stable=True),
                torch.tensor([2, 9, 6, 7, 5, 8, 0, 1, 3, 4]),
            )
        )

    def test_swap_idcs_vals_duplicates_2D(self) -> None:
        x = torch.tensor([[2, 3], [0, 4], [1, 5]])
        with self.assertRaises(ValueError):
            swap_idcs_vals_duplicates(x)

    def test_swap_idcs_vals_duplicates_no_duplicates(self) -> None:
        x = torch.tensor([2, 3, 0, 4, 1])
        self.assertTrue(
            torch.equal(
                swap_idcs_vals_duplicates(x), torch.tensor([2, 4, 0, 1, 3])
            )
        )


class TestLexsortAlong(unittest.TestCase):
    def test_lexsort_along_1D_dim0(self) -> None:
        x = torch.tensor([4, 6, 2, 7, 0, 5, 1, 3])
        values, backmap = lexsort_along(x, dim=0)
        self.assertTrue(
            torch.equal(values, torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]))
        )
        self.assertTrue(
            torch.equal(backmap, torch.tensor([4, 6, 2, 7, 0, 5, 1, 3]))
        )

    def test_lexsort_along_2D_dim0(self) -> None:
        x = torch.tensor([[2, 1], [3, 0], [1, 2], [1, 3]])
        values, backmap = lexsort_along(x, dim=0)
        self.assertTrue(
            torch.equal(values, torch.tensor([[1, 2], [1, 3], [2, 1], [3, 0]]))
        )
        self.assertTrue(torch.equal(backmap, torch.tensor([2, 3, 0, 1])))

    def test_lexsort_along_3D_dim1(self) -> None:
        x = torch.tensor([
            [[15, 13], [11, 4], [16, 2]],
            [[7, 21], [3, 20], [8, 22]],
            [[19, 14], [5, 12], [6, 0]],
            [[23, 1], [10, 17], [9, 18]],
        ])
        values, backmap = lexsort_along(x, dim=1)
        self.assertTrue(
            torch.equal(
                values,
                torch.tensor([
                    [[11, 4], [15, 13], [16, 2]],
                    [[3, 20], [7, 21], [8, 22]],
                    [[5, 12], [19, 14], [6, 0]],
                    [[10, 17], [23, 1], [9, 18]],
                ]),
            )
        )
        self.assertTrue(torch.equal(backmap, torch.tensor([1, 0, 2])))

    def test_lexsort_along_3D_dimminus1(self) -> None:
        x = torch.tensor([
            [[15, 13], [11, 4], [16, 2]],
            [[7, 21], [3, 20], [8, 22]],
            [[19, 14], [5, 12], [6, 0]],
            [[23, 1], [10, 17], [9, 18]],
        ])
        values, backmap = lexsort_along(x, dim=-1)
        self.assertTrue(
            torch.equal(
                values,
                torch.tensor([
                    [[13, 15], [4, 11], [2, 16]],
                    [[21, 7], [20, 3], [22, 8]],
                    [[14, 19], [12, 5], [0, 6]],
                    [[1, 23], [17, 10], [18, 9]],
                ]),
            )
        )
        self.assertTrue(torch.equal(backmap, torch.tensor([1, 0])))


class TestUnique(unittest.TestCase):
    def test_unique_1D_dim0(self) -> None:
        # Should be the same as dim=None in the 1D case.
        x = torch.tensor([9, 10, 9, 9, 10, 9])
        dim = 0
        uniques, backmap, inverse, counts = unique(
            x,
            return_backmap=True,
            return_inverse=True,
            return_counts=True,
            dim=dim,
            stable=True,
        )
        self.assertTrue(torch.equal(uniques, torch.tensor([9, 10])))
        self.assertTrue(torch.equal(backmap, torch.tensor([0, 2, 3, 5, 1, 4])))
        self.assertTrue(torch.equal(inverse, torch.tensor([0, 1, 0, 0, 1, 0])))
        self.assertTrue(torch.equal(counts, torch.tensor([4, 2])))

        self.assertTrue(
            torch.equal(x[backmap], torch.tensor([9, 9, 9, 9, 10, 10]))
        )

        self.assertTrue(
            torch.equal(backmap[: counts[0]], torch.tensor([0, 2, 3, 5]))
        )

        cumcounts = counts.cumsum(dim=0)
        get_idcs = lambda i: backmap[
            cumcounts[i - 1] : cumcounts[i]
        ]  # noqa: E731
        self.assertTrue(torch.equal(get_idcs(1), torch.tensor([1, 4])))

    def test_unique_1D_dimNone(self) -> None:
        # Not implemented, skip this test for now.
        return

    def test_unique_2D_dim1(self) -> None:
        x = torch.tensor(
            [[9, 10, 7, 9], [10, 9, 8, 10], [8, 7, 9, 8], [7, 7, 9, 7]]
        )
        dim = 1
        uniques, backmap, inverse, counts = unique(
            x,
            return_backmap=True,
            return_inverse=True,
            return_counts=True,
            dim=dim,
            stable=True,
        )
        self.assertTrue(
            torch.equal(
                uniques,
                torch.tensor([[7, 9, 10], [8, 10, 9], [9, 8, 7], [9, 7, 7]]),
            )
        )
        self.assertTrue(torch.equal(backmap, torch.tensor([2, 0, 3, 1])))
        self.assertTrue(torch.equal(inverse, torch.tensor([1, 2, 0, 1])))
        self.assertTrue(torch.equal(counts, torch.tensor([1, 2, 1])))

        self.assertTrue(
            torch.equal(
                x[:, backmap],
                torch.tensor(
                    [[7, 9, 9, 10], [8, 10, 10, 9], [9, 8, 8, 7], [9, 7, 7, 7]]
                ),
            )
        )

        self.assertTrue(torch.equal(backmap[: counts[0]], torch.tensor([2])))

        cumcounts = counts.cumsum(dim=0)
        get_idcs = lambda i: backmap[
            cumcounts[i - 1] : cumcounts[i]
        ]  # noqa: E731
        self.assertTrue(torch.equal(get_idcs(1), torch.tensor([0, 3])))

    def test_unique_2D_dimNone(self) -> None:
        # Not implemented, skip this test for now.
        return

    def test_unique_3D_dim2(self) -> None:
        x = torch.tensor([
            [[0, 2, 1, 2], [4, 5, 6, 5], [9, 7, 8, 7]],
            [[4, 8, 2, 8], [3, 7, 3, 7], [0, 1, 2, 1]],
        ])
        dim = 2
        uniques, backmap, inverse, counts = unique(
            x,
            return_backmap=True,
            return_inverse=True,
            return_counts=True,
            dim=dim,
            stable=True,
        )
        self.assertTrue(
            torch.equal(
                uniques,
                torch.tensor([
                    [[0, 1, 2], [4, 6, 5], [9, 8, 7]],
                    [[4, 2, 8], [3, 3, 7], [0, 2, 1]],
                ]),
            )
        )
        self.assertTrue(torch.equal(backmap, torch.tensor([0, 2, 1, 3])))
        self.assertTrue(torch.equal(inverse, torch.tensor([0, 2, 1, 2])))
        self.assertTrue(torch.equal(counts, torch.tensor([1, 1, 2])))

        self.assertTrue(
            torch.equal(
                x[:, :, backmap],
                torch.tensor([
                    [[0, 1, 2, 2], [4, 6, 5, 5], [9, 8, 7, 7]],
                    [[4, 2, 8, 8], [3, 3, 7, 7], [0, 2, 1, 1]],
                ]),
            )
        )

        self.assertTrue(torch.equal(backmap[: counts[0]], torch.tensor([0])))

        cumcounts = counts.cumsum(dim=0)
        get_idcs = lambda i: backmap[
            cumcounts[i - 1] : cumcounts[i]
        ]  # noqa: E731
        self.assertTrue(torch.equal(get_idcs(1), torch.tensor([2])))
