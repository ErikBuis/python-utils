from __future__ import annotations

import unittest

import numpy as np
import numpy.typing as npt
import pytest

from python_utils.modules.numpy import (
    apply_mask,
    last_valid_value_padding,
    lexsort_along,
    mask_padding,
    pack_padded,
    pack_sequence,
    pad_packed,
    pad_sequence,
    replace_padding,
    sequentialize_packed,
    sequentialize_padded,
    unique,
)


@pytest.fixture
def simple_packed() -> npt.NDArray[np.float64]:
    return np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])


@pytest.fixture
def simple_padded_zeros() -> npt.NDArray[np.float64]:
    return np.array([[0.1, 0.2, 0.3, 0], [0.4, 0, 0, 0], [0.5, 0.6, 0.7, 0.8]])


@pytest.fixture
def simple_padded_ones() -> npt.NDArray[np.float64]:
    return np.array([[0.1, 0.2, 0.3, 1], [0.4, 1, 1, 1], [0.5, 0.6, 0.7, 0.8]])


@pytest.fixture
def simple_padded_neg_ones() -> npt.NDArray[np.float64]:
    return np.array(
        [[0.1, 0.2, 0.3, -1], [0.4, -1, -1, -1], [0.5, 0.6, 0.7, 0.8]]
    )


@pytest.fixture
def simple_sequence() -> list[npt.NDArray[np.float64]]:
    return [
        np.array([0.1, 0.2, 0.3]),
        np.array([0.4]),
        np.array([0.5, 0.6, 0.7, 0.8]),
    ]


@pytest.fixture
def simple_L_bs() -> npt.NDArray[np.int64]:
    return np.array([3, 1, 4])


@pytest.fixture
def simple_mask() -> npt.NDArray[np.bool_]:
    return np.array([
        [True, True, True, False],
        [True, False, False, False],
        [True, True, True, True],
    ])


@pytest.fixture
def multidim_packed() -> npt.NDArray[np.float64]:
    return np.array([
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
def multidim_padded() -> npt.NDArray[np.float64]:
    return np.array([
        [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0, 0]],
        [[0.7, 0.8], [0, 0], [0, 0], [0, 0]],
        [[0.9, 1.0], [1.1, 1.2], [1.3, 1.4], [1.5, 1.6]],
    ])


@pytest.fixture
def multidim_sequence() -> list[npt.NDArray[np.float64]]:
    return [
        np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),
        np.array([[0.7, 0.8]]),
        np.array([[0.9, 1.0], [1.1, 1.2], [1.3, 1.4], [1.5, 1.6]]),
    ]


@pytest.fixture
def multidim_L_bs() -> npt.NDArray[np.int64]:
    return np.array([3, 1, 4])


class TestMaskPadding:
    def test_mask_padding_simple(
        self,
        simple_L_bs: npt.NDArray[np.int64],
        simple_mask: npt.NDArray[np.bool_],
    ) -> None:
        max_L_bs = int(simple_L_bs.max())
        mask = mask_padding(simple_L_bs, max_L_bs)
        assert np.array_equal(mask, simple_mask)

    def test_mask_padding_all_zeros(self) -> None:
        L_bs = np.array([0, 0, 0])
        max_L_bs = int(L_bs.max())
        mask = mask_padding(L_bs, max_L_bs)
        expected_mask = np.empty((3, 0), dtype=np.bool_)
        assert np.array_equal(mask, expected_mask)

    def test_mask_padding_empty(self) -> None:
        L_bs = np.empty(0, dtype=np.int64)
        max_L_bs = 0
        mask = mask_padding(L_bs, max_L_bs)
        expected_mask = np.empty((0, 0), dtype=np.bool_)
        assert np.array_equal(mask, expected_mask)

    def test_mask_padding_max_L_bs_too_high(self) -> None:
        L_bs = np.array([3, 1, 4])
        max_L_bs = 5
        mask = mask_padding(L_bs, max_L_bs)
        expected_mask = np.array([
            [True, True, True, False, False],
            [True, False, False, False, False],
            [True, True, True, True, False],
        ])
        assert np.array_equal(mask, expected_mask)


class TestPackPadded:
    def test_pack_padded_simple(
        self,
        simple_padded_zeros: npt.NDArray[np.float64],
        simple_L_bs: npt.NDArray[np.int64],
        simple_packed: npt.NDArray[np.float64],
    ) -> None:
        packed = pack_padded(simple_padded_zeros, simple_L_bs)
        assert np.array_equal(packed, simple_packed)

    def test_pack_padded_multi_dimensional(
        self,
        multidim_padded: npt.NDArray[np.float64],
        multidim_L_bs: npt.NDArray[np.int64],
        multidim_packed: npt.NDArray[np.float64],
    ) -> None:
        packed = pack_padded(multidim_padded, multidim_L_bs)
        assert np.array_equal(packed, multidim_packed)

    def test_pack_padded_all_zeros(self) -> None:
        values = np.empty((3, 0))
        L_bs = np.array([0, 0, 0])
        packed = pack_padded(values, L_bs)
        expected_packed = np.empty(0)
        assert np.array_equal(packed, expected_packed)

    def test_pack_padded_empty(self) -> None:
        values = np.empty((0, 0))
        L_bs = np.empty(0, dtype=np.int64)
        packed = pack_padded(values, L_bs)
        expected_packed = np.empty(0)
        assert np.array_equal(packed, expected_packed)


class TestPackSequence:
    def test_pack_sequence_simple(
        self,
        simple_sequence: list[npt.NDArray[np.float64]],
        simple_packed: npt.NDArray[np.float64],
        simple_L_bs: npt.NDArray[np.int64],
    ) -> None:
        max_L_bs = int(simple_L_bs.max())
        packed = pack_sequence(simple_sequence, max_L_bs)
        assert np.array_equal(packed, simple_packed)

    def test_pack_sequence_multi_dimensional(
        self,
        multidim_sequence: list[npt.NDArray[np.float64]],
        multidim_packed: npt.NDArray[np.float64],
        multidim_L_bs: npt.NDArray[np.int64],
    ) -> None:
        max_L_bs = int(multidim_L_bs.max())
        packed = pack_sequence(multidim_sequence, max_L_bs)
        assert np.array_equal(packed, multidim_packed)

    def test_pack_sequence_all_zeros(self) -> None:
        values = [np.empty(0), np.empty(0), np.empty(0)]
        packed = pack_sequence(values, 0)
        expected_packed = np.empty(0)
        assert np.array_equal(packed, expected_packed)

    def test_pack_sequence_empty(self) -> None:
        values = []
        packed = pack_sequence(values, 0)
        expected_packed = np.empty(0)
        assert np.array_equal(packed, expected_packed)


class TestPadPacked:
    def test_pad_packed_simple(
        self,
        simple_packed: npt.NDArray[np.float64],
        simple_L_bs: npt.NDArray[np.int64],
        simple_padded_zeros: npt.NDArray[np.float64],
    ) -> None:
        max_L_bs = int(simple_L_bs.max())
        padded = pad_packed(simple_packed, simple_L_bs, max_L_bs)
        assert all(
            np.array_equal(
                padded[b, : simple_L_bs[b]],
                simple_padded_zeros[b, : simple_L_bs[b]],
            )
            for b in range(len(simple_L_bs))
        )

    def test_pad_packed_padding_value_zero(
        self,
        simple_packed: npt.NDArray[np.float64],
        simple_L_bs: npt.NDArray[np.int64],
        simple_padded_zeros: npt.NDArray[np.float64],
    ) -> None:
        max_L_bs = int(simple_L_bs.max())
        padded = pad_packed(
            simple_packed, simple_L_bs, max_L_bs, padding_value=0
        )
        assert np.array_equal(padded, simple_padded_zeros)

    def test_pad_packed_padding_value_one(
        self,
        simple_packed: npt.NDArray[np.float64],
        simple_L_bs: npt.NDArray[np.int64],
        simple_padded_ones: npt.NDArray[np.float64],
    ) -> None:
        max_L_bs = int(simple_L_bs.max())
        padded = pad_packed(
            simple_packed, simple_L_bs, max_L_bs, padding_value=1
        )
        assert np.array_equal(padded, simple_padded_ones)

    def test_pad_packed_padding_value_other(
        self,
        simple_packed: npt.NDArray[np.float64],
        simple_L_bs: npt.NDArray[np.int64],
        simple_padded_neg_ones: npt.NDArray[np.float64],
    ) -> None:
        max_L_bs = int(simple_L_bs.max())
        padded = pad_packed(
            simple_packed, simple_L_bs, max_L_bs, padding_value=-1
        )
        assert np.array_equal(padded, simple_padded_neg_ones)

    def test_pad_packed_multi_dimensional(
        self,
        multidim_packed: npt.NDArray[np.float64],
        multidim_L_bs: npt.NDArray[np.int64],
        multidim_padded: npt.NDArray[np.float64],
    ) -> None:
        max_L_bs = int(multidim_L_bs.max())
        padded = pad_packed(multidim_packed, multidim_L_bs, max_L_bs)
        assert all(
            np.array_equal(
                padded[b, : multidim_L_bs[b]],
                multidim_padded[b, : multidim_L_bs[b]],
            )
            for b in range(len(multidim_L_bs))
        )

    def test_pad_packed_all_zeros(self) -> None:
        values = np.empty(0)
        L_bs = np.array([0, 0, 0])
        max_L_bs = int(L_bs.max())
        padded = pad_packed(values, L_bs, max_L_bs)
        expected_padded = np.empty((3, 0))
        assert np.array_equal(padded, expected_padded)

    def test_pad_packed_empty(self) -> None:
        values = np.empty(0)
        L_bs = np.empty(0, dtype=np.int64)
        max_L_bs = 0
        padded = pad_packed(values, L_bs, max_L_bs)
        expected_padded = np.empty((0, 0))
        assert np.array_equal(padded, expected_padded)


class TestPadSequence:
    def test_pad_sequence_simple(
        self,
        simple_sequence: list[npt.NDArray[np.float64]],
        simple_L_bs: npt.NDArray[np.int64],
        simple_padded_zeros: npt.NDArray[np.float64],
    ) -> None:
        max_L_bs = int(simple_L_bs.max())
        padded = pad_sequence(simple_sequence, simple_L_bs, max_L_bs)
        assert all(
            np.array_equal(
                padded[b, : simple_L_bs[b]],
                simple_padded_zeros[b, : simple_L_bs[b]],
            )
            for b in range(len(simple_L_bs))
        )

    def test_pad_sequence_padding_value_zero(
        self,
        simple_sequence: list[npt.NDArray[np.float64]],
        simple_L_bs: npt.NDArray[np.int64],
        simple_padded_zeros: npt.NDArray[np.float64],
    ) -> None:
        max_L_bs = int(simple_L_bs.max())
        padded = pad_sequence(
            simple_sequence, simple_L_bs, max_L_bs, padding_value=0
        )
        assert np.array_equal(padded, simple_padded_zeros)

    def test_pad_sequence_padding_value_one(
        self,
        simple_sequence: list[npt.NDArray[np.float64]],
        simple_L_bs: npt.NDArray[np.int64],
        simple_padded_ones: npt.NDArray[np.float64],
    ) -> None:
        max_L_bs = int(simple_L_bs.max())
        padded = pad_sequence(
            simple_sequence, simple_L_bs, max_L_bs, padding_value=1
        )
        assert np.array_equal(padded, simple_padded_ones)

    def test_pad_sequence_padding_value_other(
        self,
        simple_sequence: list[npt.NDArray[np.float64]],
        simple_L_bs: npt.NDArray[np.int64],
        simple_padded_neg_ones: npt.NDArray[np.float64],
    ) -> None:
        max_L_bs = int(simple_L_bs.max())
        padded = pad_sequence(
            simple_sequence, simple_L_bs, max_L_bs, padding_value=-1
        )
        assert np.array_equal(padded, simple_padded_neg_ones)

    def test_pad_sequence_multi_dimensional(
        self,
        multidim_sequence: list[npt.NDArray[np.float64]],
        multidim_L_bs: npt.NDArray[np.int64],
        multidim_padded: npt.NDArray[np.float64],
    ) -> None:
        max_L_bs = int(multidim_L_bs.max())
        padded = pad_sequence(multidim_sequence, multidim_L_bs, max_L_bs)
        assert all(
            np.array_equal(
                padded[b, : multidim_L_bs[b]],
                multidim_padded[b, : multidim_L_bs[b]],
            )
            for b in range(len(multidim_L_bs))
        )

    def test_pad_sequence_all_zeros(self) -> None:
        values = [np.empty(0), np.empty(0), np.empty(0)]
        L_bs = np.array([0, 0, 0])
        max_L_bs = int(L_bs.max())
        padded = pad_sequence(values, L_bs, max_L_bs)
        expected_padded = np.empty((3, 0))
        assert np.array_equal(padded, expected_padded)

    def test_pad_sequence_empty(self) -> None:
        values = []
        L_bs = np.empty(0, dtype=np.int64)
        max_L_bs = 0
        padded = pad_sequence(values, L_bs, max_L_bs)
        expected_padded = np.empty((0, 0))
        assert np.array_equal(padded, expected_padded)


class TestSequentializePacked:
    def test_sequentialize_packed_simple(
        self,
        simple_packed: npt.NDArray[np.float64],
        simple_L_bs: npt.NDArray[np.int64],
        simple_sequence: list[npt.NDArray[np.float64]],
    ) -> None:
        sequentialized = sequentialize_packed(simple_packed, simple_L_bs)
        assert all(
            np.array_equal(sequentialized[b], simple_sequence[b])
            for b in range(len(simple_L_bs))
        )

    def test_sequentialize_packed_multi_dimensional(
        self,
        multidim_packed: npt.NDArray[np.float64],
        multidim_L_bs: npt.NDArray[np.int64],
        multidim_sequence: list[npt.NDArray[np.float64]],
    ) -> None:
        sequentialized = sequentialize_packed(multidim_packed, multidim_L_bs)
        assert all(
            np.array_equal(sequentialized[b], multidim_sequence[b])
            for b in range(len(multidim_L_bs))
        )

    def test_sequentialize_packed_all_zeros(self) -> None:
        values = np.empty(0)
        L_bs = np.array([0, 0, 0])
        sequentialized = sequentialize_packed(values, L_bs)
        expected_sequentialized = [np.empty(0), np.empty(0), np.empty(0)]
        assert all(
            np.array_equal(sequentialized[b], expected_sequentialized[b])
            for b in range(len(L_bs))
        )

    def test_sequentialize_packed_empty(self) -> None:
        values = np.empty(0)
        L_bs = np.empty(0, dtype=np.int64)
        sequentialized = sequentialize_packed(values, L_bs)
        expected_sequentialized = []
        assert sequentialized == expected_sequentialized


class TestSequentializePadded:
    def test_sequentialize_padded_simple(
        self,
        simple_padded_zeros: npt.NDArray[np.float64],
        simple_L_bs: npt.NDArray[np.int64],
        simple_sequence: list[npt.NDArray[np.float64]],
    ) -> None:
        sequentialized = sequentialize_padded(simple_padded_zeros, simple_L_bs)
        assert all(
            np.array_equal(sequentialized[b], simple_sequence[b])
            for b in range(len(simple_L_bs))
        )

    def test_sequentialize_padded_multi_dimensional(
        self,
        multidim_padded: npt.NDArray[np.float64],
        multidim_L_bs: npt.NDArray[np.int64],
        multidim_sequence: list[npt.NDArray[np.float64]],
    ) -> None:
        sequentialized = sequentialize_padded(multidim_padded, multidim_L_bs)
        assert all(
            np.array_equal(sequentialized[b], multidim_sequence[b])
            for b in range(len(multidim_L_bs))
        )

    def test_sequentialize_padded_all_zeros(self) -> None:
        values = np.empty((3, 0))
        L_bs = np.array([0, 0, 0])
        sequentialized = sequentialize_padded(values, L_bs)
        expected_sequentialized = [np.empty(0), np.empty(0), np.empty(0)]
        assert all(
            np.array_equal(sequentialized[b], expected_sequentialized[b])
            for b in range(len(L_bs))
        )

    def test_sequentialize_padded_empty(self) -> None:
        values = np.empty((0, 0))
        L_bs = np.empty(0, dtype=np.int64)
        sequentialized = sequentialize_padded(values, L_bs)
        expected_sequentialized = []
        assert sequentialized == expected_sequentialized


class TestApplyMask:
    def test_apply_mask_simple(
        self,
        simple_padded_zeros: npt.NDArray[np.float64],
        simple_L_bs: npt.NDArray[np.int64],
    ) -> None:
        # Mask that keeps indices [0, 2] from first, [] from second, [0, 2, 3]
        # from third.
        mask = np.array([
            [True, False, True, False],
            [False, False, False, False],
            [True, False, True, True],
        ])
        values_filtered, L_bs_kept = apply_mask(
            simple_padded_zeros, mask, simple_L_bs, padding_value=0
        )
        expected_values = np.array([[0.1, 0.3, 0], [0, 0, 0], [0.5, 0.7, 0.8]])
        expected_L_bs_kept = np.array([2, 0, 3])
        assert np.array_equal(values_filtered, expected_values)
        assert np.array_equal(L_bs_kept, expected_L_bs_kept)

    def test_apply_mask_multi_dimensional(
        self,
        multidim_padded: npt.NDArray[np.float64],
        multidim_L_bs: npt.NDArray[np.int64],
    ) -> None:
        # Mask that keeps indices [0, 2] from first, [] from second, [0, 2, 3]
        # from third.
        mask = np.array([
            [True, False, True, False],
            [False, False, False, False],
            [True, False, True, True],
        ])
        values_filtered, L_bs_kept = apply_mask(
            multidim_padded, mask, multidim_L_bs, padding_value=0
        )
        expected_values = np.array([
            [[0.1, 0.2], [0.5, 0.6], [0, 0]],
            [[0, 0], [0, 0], [0, 0]],
            [[0.9, 1.0], [1.3, 1.4], [1.5, 1.6]],
        ])
        expected_L_bs_kept = np.array([2, 0, 3])
        assert np.array_equal(values_filtered, expected_values)
        assert np.array_equal(L_bs_kept, expected_L_bs_kept)

    def test_apply_mask_all_true(
        self,
        simple_padded_zeros: npt.NDArray[np.float64],
        simple_padded_neg_ones: npt.NDArray[np.float64],
        simple_L_bs: npt.NDArray[np.int64],
    ) -> None:
        # Mask keeps all valid elements, changing padding from 0 to -1 to check
        # that the old padding is properly replaced.
        mask = np.array([
            [True, True, True, True],
            [True, True, True, True],
            [True, True, True, True],
        ])
        values_filtered, L_bs_kept = apply_mask(
            simple_padded_zeros, mask, simple_L_bs, padding_value=-1
        )
        assert np.array_equal(values_filtered, simple_padded_neg_ones)
        assert np.array_equal(L_bs_kept, simple_L_bs)

    def test_apply_mask_all_false(
        self,
        simple_padded_zeros: npt.NDArray[np.float64],
        simple_L_bs: npt.NDArray[np.int64],
    ) -> None:
        # Mask removes all elements.
        mask = np.array([
            [False, False, False, False],
            [False, False, False, False],
            [False, False, False, False],
        ])
        values_filtered, L_bs_kept = apply_mask(
            simple_padded_zeros, mask, simple_L_bs, padding_value=0
        )
        expected_values = np.empty((3, 0))
        expected_L_bs_kept = np.array([0, 0, 0])
        assert np.array_equal(values_filtered, expected_values)
        assert np.array_equal(L_bs_kept, expected_L_bs_kept)

    def test_apply_mask_all_zeros(self) -> None:
        values = np.empty((3, 0))
        L_bs = np.array([0, 0, 0])
        mask = np.empty((3, 0), dtype=np.bool_)
        values_filtered, L_bs_kept = apply_mask(
            values, mask, L_bs, padding_value=0
        )
        assert np.array_equal(values_filtered, values)
        assert np.array_equal(L_bs_kept, L_bs)

    def test_apply_mask_empty(self) -> None:
        values = np.empty((0, 0))
        L_bs = np.empty(0, dtype=np.int64)
        mask = np.empty((0, 0), dtype=np.bool_)
        values_filtered, L_bs_kept = apply_mask(
            values, mask, L_bs, padding_value=0
        )
        assert np.array_equal(values_filtered, values)
        assert np.array_equal(L_bs_kept, L_bs)


class TestReplacePadding:
    def test_replace_padding_scalar(
        self,
        simple_padded_zeros: npt.NDArray[np.float64],
        simple_padded_ones: npt.NDArray[np.float64],
        simple_L_bs: npt.NDArray[np.int64],
    ) -> None:
        # Padding value as scalar.
        values_repadded = replace_padding(
            simple_padded_zeros, simple_L_bs, padding_value=1
        )
        assert np.array_equal(values_repadded, simple_padded_ones)

    def test_replace_padding_all(
        self,
        simple_padded_zeros: npt.NDArray[np.float64],
        simple_padded_ones: npt.NDArray[np.float64],
        simple_L_bs: npt.NDArray[np.int64],
    ) -> None:
        # Padding value with shape [].
        padding_value = np.array(1)
        values_repadded = replace_padding(
            simple_padded_zeros, simple_L_bs, padding_value=padding_value
        )
        assert np.array_equal(values_repadded, simple_padded_ones)

    def test_replace_padding_all_multi_dimensional(
        self,
        multidim_padded: npt.NDArray[np.float64],
        multidim_L_bs: npt.NDArray[np.int64],
    ) -> None:
        # Padding value with shape [*] = [2].
        padding_value = np.array([1, 2])
        values_repadded = replace_padding(
            multidim_padded, multidim_L_bs, padding_value=padding_value
        )
        expected_values = np.array([
            [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [1, 2]],
            [[0.7, 0.8], [1, 2], [1, 2], [1, 2]],
            [[0.9, 1.0], [1.1, 1.2], [1.3, 1.4], [1.5, 1.6]],
        ])
        assert np.array_equal(values_repadded, expected_values)

    def test_replace_padding_per_element(
        self,
        simple_padded_zeros: npt.NDArray[np.float64],
        simple_L_bs: npt.NDArray[np.int64],
    ) -> None:
        # Padding value with shape [B, max(L_bs)] = [3, 4].
        padding_value = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
        values_repadded = replace_padding(
            simple_padded_zeros, simple_L_bs, padding_value=padding_value
        )
        expected_values = np.array(
            [[0.1, 0.2, 0.3, 4], [0.4, 6, 7, 8], [0.5, 0.6, 0.7, 0.8]]
        )
        assert np.array_equal(values_repadded, expected_values)

    def test_replace_padding_per_element_multi_dimensional(
        self,
        multidim_padded: npt.NDArray[np.float64],
        multidim_L_bs: npt.NDArray[np.int64],
    ) -> None:
        # Padding value with shape [B, max(L_bs), *] = [3, 4, 2].
        padding_value = np.array([
            [[1, 2], [3, 4], [5, 6], [7, 8]],
            [[9, 10], [11, 12], [13, 14], [15, 16]],
            [[17, 18], [19, 20], [21, 22], [23, 24]],
        ])
        values_repadded = replace_padding(
            multidim_padded, multidim_L_bs, padding_value=padding_value
        )
        expected_values = np.array([
            [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [7, 8]],
            [[0.7, 0.8], [11, 12], [13, 14], [15, 16]],
            [[0.9, 1.0], [1.1, 1.2], [1.3, 1.4], [1.5, 1.6]],
        ])
        assert np.array_equal(values_repadded, expected_values)

    def test_replace_padding_per_row(
        self,
        simple_padded_zeros: npt.NDArray[np.float64],
        simple_L_bs: npt.NDArray[np.int64],
    ) -> None:
        # Padding value with shape [B, 1] = [3, 1].
        padding_value = np.array([[1], [2], [3]])
        values_repadded = replace_padding(
            simple_padded_zeros, simple_L_bs, padding_value=padding_value
        )
        expected_values = np.array(
            [[0.1, 0.2, 0.3, 1], [0.4, 2, 2, 2], [0.5, 0.6, 0.7, 0.8]]
        )
        assert np.array_equal(values_repadded, expected_values)

    def test_replace_padding_per_row_multi_dimensional(
        self,
        multidim_padded: npt.NDArray[np.float64],
        multidim_L_bs: npt.NDArray[np.int64],
    ) -> None:
        # Padding value with shape [B, 1, *] = [3, 1, 2].
        padding_value = np.array([[[1, 2]], [[3, 4]], [[5, 6]]])
        values_repadded = replace_padding(
            multidim_padded, multidim_L_bs, padding_value=padding_value
        )
        expected_values = np.array([
            [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [1, 2]],
            [[0.7, 0.8], [3, 4], [3, 4], [3, 4]],
            [[0.9, 1.0], [1.1, 1.2], [1.3, 1.4], [1.5, 1.6]],
        ])
        assert np.array_equal(values_repadded, expected_values)

    def test_replace_padding_per_col(
        self,
        simple_padded_zeros: npt.NDArray[np.float64],
        simple_L_bs: npt.NDArray[np.int64],
    ) -> None:
        # Padding value with shape [1, max(L_bs)] = [1, 4].
        padding_value = np.array([[1, 2, 3, 4]])
        values_repadded = replace_padding(
            simple_padded_zeros, simple_L_bs, padding_value=padding_value
        )
        expected_values = np.array(
            [[0.1, 0.2, 0.3, 4], [0.4, 2, 3, 4], [0.5, 0.6, 0.7, 0.8]]
        )
        assert np.array_equal(values_repadded, expected_values)

    def test_replace_padding_per_col_multi_dimensional(
        self,
        multidim_padded: npt.NDArray[np.float64],
        multidim_L_bs: npt.NDArray[np.int64],
    ) -> None:
        # Padding value with shape [1, max(L_bs), *] = [1, 4, 2].
        padding_value = np.array([[[1, 2], [3, 4], [5, 6], [7, 8]]])
        values_repadded = replace_padding(
            multidim_padded, multidim_L_bs, padding_value=padding_value
        )
        expected_values = np.array([
            [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [7, 8]],
            [[0.7, 0.8], [3, 4], [5, 6], [7, 8]],
            [[0.9, 1.0], [1.1, 1.2], [1.3, 1.4], [1.5, 1.6]],
        ])
        assert np.array_equal(values_repadded, expected_values)

    def test_replace_padding_in_mask(
        self,
        simple_padded_zeros: npt.NDArray[np.float64],
        simple_L_bs: npt.NDArray[np.int64],
    ) -> None:
        # Padding value with shape [B * max(L_bs) - L] = [3*4 - 8] = [4].
        padding_value = np.array([1, 2, 3, 4])
        values_repadded = replace_padding(
            simple_padded_zeros, simple_L_bs, padding_value=padding_value
        )
        expected_values = np.array(
            [[0.1, 0.2, 0.3, 1], [0.4, 2, 3, 4], [0.5, 0.6, 0.7, 0.8]]
        )
        assert np.array_equal(values_repadded, expected_values)

    def test_replace_padding_in_mask_multi_dimensional(
        self,
        multidim_padded: npt.NDArray[np.float64],
        multidim_L_bs: npt.NDArray[np.int64],
    ) -> None:
        # Padding value with shape [B * max(L_bs) - L, *] = [3*4 - 8, 2]
        # = [4, 2].
        padding_value = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        values_repadded = replace_padding(
            multidim_padded, multidim_L_bs, padding_value=padding_value
        )
        expected_values = np.array([
            [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [1, 2]],
            [[0.7, 0.8], [3, 4], [5, 6], [7, 8]],
            [[0.9, 1.0], [1.1, 1.2], [1.3, 1.4], [1.5, 1.6]],
        ])
        assert np.array_equal(values_repadded, expected_values)

    def test_replace_padding_in_place(
        self,
        simple_padded_zeros: npt.NDArray[np.float64],
        simple_L_bs: npt.NDArray[np.int64],
    ) -> None:
        # Verify in_place=True modifies the original array.
        values_original = simple_padded_zeros.copy()
        values_repadded = replace_padding(
            values_original, simple_L_bs, padding_value=1, in_place=True
        )
        expected_values = np.array(
            [[0.1, 0.2, 0.3, 1], [0.4, 1, 1, 1], [0.5, 0.6, 0.7, 0.8]]
        )
        assert values_repadded is values_original
        assert np.array_equal(values_original, expected_values)

    def test_replace_padding_all_zeros(self) -> None:
        values = np.empty((3, 0))
        L_bs = np.array([0, 0, 0])
        values_repadded = replace_padding(values, L_bs, padding_value=-1)
        assert np.array_equal(values_repadded, values)

    def test_replace_padding_empty(self) -> None:
        values = np.empty((0, 0))
        L_bs = np.empty(0, dtype=np.int64)
        values_repadded = replace_padding(values, L_bs, padding_value=-1)
        assert np.array_equal(values_repadded, values)


class TestLastValidValuePadding:
    def test_last_valid_value_padding_simple(
        self,
        simple_padded_zeros: npt.NDArray[np.float64],
        simple_L_bs: npt.NDArray[np.int64],
    ) -> None:
        # Pad with last valid value from each row.
        values_repadded = last_valid_value_padding(
            simple_padded_zeros, simple_L_bs
        )
        expected_values = np.array(
            [[0.1, 0.2, 0.3, 0.3], [0.4, 0.4, 0.4, 0.4], [0.5, 0.6, 0.7, 0.8]]
        )
        assert np.array_equal(values_repadded, expected_values)

    def test_last_valid_value_padding_empty_rows_with_value(
        self, simple_padded_zeros: npt.NDArray[np.float64]
    ) -> None:
        # Test empty rows (L_b == 0) with padding_value_empty_rows specified.
        L_bs = np.array([2, 0, 3])
        values_repadded = last_valid_value_padding(
            simple_padded_zeros, L_bs, padding_value_empty_rows=-1
        )
        expected_values = np.array(
            [[0.1, 0.2, 0.2, 0.2], [-1, -1, -1, -1], [0.5, 0.6, 0.7, 0.7]]
        )
        assert np.array_equal(values_repadded, expected_values)

    def test_last_valid_value_padding_empty_rows_without_value(
        self, simple_padded_zeros: npt.NDArray[np.float64]
    ) -> None:
        # Test empty rows (L_b == 0) with padding_value_empty_rows=None.
        # No assertion on padding content for empty row.
        L_bs = np.array([2, 0, 3])
        values_repadded = last_valid_value_padding(
            simple_padded_zeros, L_bs, padding_value_empty_rows=None
        )
        expected_values_row_1 = np.array([0.1, 0.2, 0.2, 0.2])
        expected_values_row_3 = np.array([0.5, 0.6, 0.7, 0.7])
        assert np.array_equal(values_repadded[0], expected_values_row_1)
        assert np.array_equal(values_repadded[2], expected_values_row_3)

    def test_last_valid_value_padding_multi_dimensional(
        self,
        multidim_padded: npt.NDArray[np.float64],
        multidim_L_bs: npt.NDArray[np.int64],
    ) -> None:
        # Pad with last valid value for multidimensional values.
        values_repadded = last_valid_value_padding(
            multidim_padded, multidim_L_bs
        )
        expected_values = np.array([
            [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.5, 0.6]],
            [[0.7, 0.8], [0.7, 0.8], [0.7, 0.8], [0.7, 0.8]],
            [[0.9, 1.0], [1.1, 1.2], [1.3, 1.4], [1.5, 1.6]],
        ])
        assert np.array_equal(values_repadded, expected_values)

    def test_last_valid_value_padding_in_place(
        self,
        simple_padded_zeros: npt.NDArray[np.float64],
        simple_L_bs: npt.NDArray[np.int64],
    ) -> None:
        # Verify in_place=True modifies the original array.
        values_original = simple_padded_zeros.copy()
        values_repadded = last_valid_value_padding(
            values_original, simple_L_bs, in_place=True
        )
        expected_values = np.array(
            [[0.1, 0.2, 0.3, 0.3], [0.4, 0.4, 0.4, 0.4], [0.5, 0.6, 0.7, 0.8]]
        )
        assert values_repadded is values_original
        assert np.array_equal(values_original, expected_values)

    def test_last_valid_value_padding_all_empty_rows(
        self, simple_padded_zeros: npt.NDArray[np.float64]
    ) -> None:
        # Verify all rows empty (L_b == 0) with padding_value_empty_rows
        # specified.
        L_bs = np.array([0, 0, 0])
        values_repadded = last_valid_value_padding(
            simple_padded_zeros, L_bs, padding_value_empty_rows=1
        )
        expected_values = np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]])
        assert np.array_equal(values_repadded, expected_values)

    def test_last_valid_value_padding_all_zeros(self) -> None:
        values = np.empty((3, 0))
        L_bs = np.array([0, 0, 0])
        values_repadded = last_valid_value_padding(
            values, L_bs, padding_value_empty_rows=1
        )
        assert np.array_equal(values_repadded, values)

    def test_last_valid_value_padding_empty(self) -> None:
        values = np.empty((0, 0))
        L_bs = np.empty(0, dtype=np.int64)
        values_repadded = last_valid_value_padding(
            values, L_bs, padding_value_empty_rows=1
        )
        assert np.array_equal(values_repadded, values)


class TestLexsortAlong(unittest.TestCase):
    def test_lexsort_along_1D_axis0(self) -> None:
        x = np.array([4, 6, 2, 7, 0, 5, 1, 3])
        values, backmap = lexsort_along(x, axis=0)
        self.assertTrue(
            np.array_equal(values, np.array([0, 1, 2, 3, 4, 5, 6, 7]))
        )
        self.assertTrue(
            np.array_equal(backmap, np.array([4, 6, 2, 7, 0, 5, 1, 3]))
        )

    def test_lexsort_along_2D_axis0(self) -> None:
        x = np.array([[2, 1], [3, 0], [1, 2], [1, 3]])
        values, backmap = lexsort_along(x, axis=0)
        self.assertTrue(
            np.array_equal(values, np.array([[1, 2], [1, 3], [2, 1], [3, 0]]))
        )
        self.assertTrue(np.array_equal(backmap, np.array([2, 3, 0, 1])))

    def test_lexsort_along_3D_axis1(self) -> None:
        x = np.array([
            [[15, 13], [11, 4], [16, 2]],
            [[7, 21], [3, 20], [8, 22]],
            [[19, 14], [5, 12], [6, 0]],
            [[23, 1], [10, 17], [9, 18]],
        ])
        values, backmap = lexsort_along(x, axis=1)
        self.assertTrue(
            np.array_equal(
                values,
                np.array([
                    [[11, 4], [15, 13], [16, 2]],
                    [[3, 20], [7, 21], [8, 22]],
                    [[5, 12], [19, 14], [6, 0]],
                    [[10, 17], [23, 1], [9, 18]],
                ]),
            )
        )
        self.assertTrue(np.array_equal(backmap, np.array([1, 0, 2])))

    def test_lexsort_along_3D_axisminus1(self) -> None:
        x = np.array([
            [[15, 13], [11, 4], [16, 2]],
            [[7, 21], [3, 20], [8, 22]],
            [[19, 14], [5, 12], [6, 0]],
            [[23, 1], [10, 17], [9, 18]],
        ])
        values, backmap = lexsort_along(x, axis=-1)
        self.assertTrue(
            np.array_equal(
                values,
                np.array([
                    [[13, 15], [4, 11], [2, 16]],
                    [[21, 7], [20, 3], [22, 8]],
                    [[14, 19], [12, 5], [0, 6]],
                    [[1, 23], [17, 10], [18, 9]],
                ]),
            )
        )
        self.assertTrue(np.array_equal(backmap, np.array([1, 0])))


class TestUnique(unittest.TestCase):
    def test_unique_1D_axis0(self) -> None:
        # Should be the same as axis=None in the 1D case.
        x = np.array([9, 10, 9, 9, 10, 9])
        axis = 0
        uniques, backmap, inverse, counts = unique(
            x,
            return_backmap=True,
            return_inverse=True,
            return_counts=True,
            axis=axis,
            stable=True,
        )
        self.assertTrue(np.array_equal(uniques, np.array([9, 10])))
        self.assertTrue(np.array_equal(backmap, np.array([0, 2, 3, 5, 1, 4])))
        self.assertTrue(np.array_equal(inverse, np.array([0, 1, 0, 0, 1, 0])))
        self.assertTrue(np.array_equal(counts, np.array([4, 2])))

        self.assertTrue(
            np.array_equal(x[backmap], np.array([9, 9, 9, 9, 10, 10]))
        )

        self.assertTrue(
            np.array_equal(backmap[: counts[0]], np.array([0, 2, 3, 5]))
        )

        cumcounts = counts.cumsum(axis=0)
        get_idcs = lambda i: backmap[
            cumcounts[i - 1] : cumcounts[i]
        ]  # noqa: E731
        self.assertTrue(np.array_equal(get_idcs(1), np.array([1, 4])))

    def test_unique_1D_axisNone(self) -> None:
        # Not implemented, skip this test for now.
        return

    def test_unique_2D_axis1(self) -> None:
        x = np.array(
            [[9, 10, 7, 9], [10, 9, 8, 10], [8, 7, 9, 8], [7, 7, 9, 7]]
        )
        axis = 1
        uniques, backmap, inverse, counts = unique(
            x,
            return_backmap=True,
            return_inverse=True,
            return_counts=True,
            axis=axis,
            stable=True,
        )
        self.assertTrue(
            np.array_equal(
                uniques,
                np.array([[7, 9, 10], [8, 10, 9], [9, 8, 7], [9, 7, 7]]),
            )
        )
        self.assertTrue(np.array_equal(backmap, np.array([2, 0, 3, 1])))
        self.assertTrue(np.array_equal(inverse, np.array([1, 2, 0, 1])))
        self.assertTrue(np.array_equal(counts, np.array([1, 2, 1])))

        self.assertTrue(
            np.array_equal(
                x[:, backmap],
                np.array(
                    [[7, 9, 9, 10], [8, 10, 10, 9], [9, 8, 8, 7], [9, 7, 7, 7]]
                ),
            )
        )

        self.assertTrue(np.array_equal(backmap[: counts[0]], np.array([2])))

        cumcounts = counts.cumsum(axis=0)
        get_idcs = lambda i: backmap[
            cumcounts[i - 1] : cumcounts[i]
        ]  # noqa: E731
        self.assertTrue(np.array_equal(get_idcs(1), np.array([0, 3])))

    def test_unique_2D_axisNone(self) -> None:
        # Not implemented, skip this test for now.
        return

    def test_unique_3D_axis2(self) -> None:
        x = np.array([
            [[0, 2, 1, 2], [4, 5, 6, 5], [9, 7, 8, 7]],
            [[4, 8, 2, 8], [3, 7, 3, 7], [0, 1, 2, 1]],
        ])
        axis = 2
        uniques, backmap, inverse, counts = unique(
            x,
            return_backmap=True,
            return_inverse=True,
            return_counts=True,
            axis=axis,
            stable=True,
        )
        self.assertTrue(
            np.array_equal(
                uniques,
                np.array([
                    [[0, 1, 2], [4, 6, 5], [9, 8, 7]],
                    [[4, 2, 8], [3, 3, 7], [0, 2, 1]],
                ]),
            )
        )
        self.assertTrue(np.array_equal(backmap, np.array([0, 2, 1, 3])))
        self.assertTrue(np.array_equal(inverse, np.array([0, 2, 1, 2])))
        self.assertTrue(np.array_equal(counts, np.array([1, 1, 2])))

        self.assertTrue(
            np.array_equal(
                x[:, :, backmap],
                np.array([
                    [[0, 1, 2, 2], [4, 6, 5, 5], [9, 8, 7, 7]],
                    [[4, 2, 8, 8], [3, 3, 7, 7], [0, 2, 1, 1]],
                ]),
            )
        )

        self.assertTrue(np.array_equal(backmap[: counts[0]], np.array([0])))

        cumcounts = counts.cumsum(axis=0)
        get_idcs = lambda i: backmap[
            cumcounts[i - 1] : cumcounts[i]
        ]  # noqa: E731
        self.assertTrue(np.array_equal(get_idcs(1), np.array([2])))
