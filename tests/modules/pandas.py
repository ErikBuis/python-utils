from __future__ import annotations

import numpy as np
import pandas as pd

from python_utils.modules.pandas import (
    create_func_values2idcs,
    remap_series_to_idcs,
)


class TestCreateFuncValues2Idcs:
    def test_basic_functionality_no_missing(self) -> None:
        """Test basic mapping functionality without handling missing values."""
        values_unique = np.array([10, 20, 30], dtype=np.int64)
        func = create_func_values2idcs(
            values_unique, handle_missing_values=False
        )
        x = np.array([10, 20, 10, 30], dtype=np.int64)
        expected = np.array([0, 1, 0, 2], dtype=np.int64)
        np.testing.assert_array_equal(func(x), expected)

    def test_handle_missing_values_enabled(self) -> None:
        """Test mapping functionality with missing value handling."""
        values_unique = np.array([10, 20, 30], dtype=np.int64)
        func = create_func_values2idcs(
            values_unique, handle_missing_values=True
        )
        x = np.array([10, 40, 20, 50], dtype=np.int64)
        expected = np.array([0, np.nan, 1, np.nan], dtype=np.float64)
        np.testing.assert_array_equal(func(x), expected)

    def test_large_dense_integer_range(self) -> None:
        """Test mapping for a large, densely packed integer range."""
        values_unique = np.arange(1, 1_000_001, dtype=np.int64)
        func = create_func_values2idcs(
            values_unique, handle_missing_values=False
        )
        x = np.array([1, 500_000, 1_000_000], dtype=np.int64)
        expected = np.array([0, 499_999, 999_999], dtype=np.int64)
        np.testing.assert_array_equal(func(x), expected)

    def test_large_sparse_integer_range(self) -> None:
        """Test mapping for a sparse range, falling back to dict mapping."""
        values_unique = np.array([1, 10, 100, 1_000, 10_000], dtype=np.int64)
        func = create_func_values2idcs(
            values_unique, handle_missing_values=False
        )
        x = np.array([10, 100, 1_000], dtype=np.int64)
        expected = np.array([1, 2, 3], dtype=np.int64)
        np.testing.assert_array_equal(func(x), expected)

    def test_non_integer_values(self) -> None:
        """Test mapping for non-integer values, falling back to dictionary."""
        values_unique = np.array([1.1, 2.2, 3.3], dtype=np.float64)
        func = create_func_values2idcs(
            values_unique, handle_missing_values=False
        )
        x = np.array([3.3, 2.2], dtype=np.float64)
        expected = np.array([2, 1], dtype=np.int64)
        np.testing.assert_array_equal(func(x), expected)

    def test_empty_values_unique(self) -> None:
        """Test behavior when the input unique values array is empty."""
        values_unique = np.array([], dtype=np.int64)
        func = create_func_values2idcs(
            values_unique, handle_missing_values=False
        )
        x = np.array([], dtype=np.int64)
        expected = np.array([], dtype=np.int64)
        np.testing.assert_array_equal(func(x), expected)


class TestRemapSeriesToIdcs:
    def test_basic_functionality_no_values_unique(self) -> None:
        """Test remapping without providing values_unique."""
        series = pd.Series([10, 20, 10, 30, 20])
        expected_unique = np.array([10, 20, 30], dtype=np.int64)
        expected_remapped = np.array([0, 1, 0, 2, 1], dtype=np.int64)

        unique, value2idx, remapped = remap_series_to_idcs(series)
        np.testing.assert_array_equal(unique, expected_unique)
        np.testing.assert_array_equal(remapped, expected_remapped)
        np.testing.assert_array_equal(
            value2idx(expected_unique), np.array([0, 1, 2])
        )

    def test_provided_values_unique_no_missing(self) -> None:
        """Test remapping with provided values_unique and no missing values."""
        series = pd.Series([10, 20, 30, 10])
        values_unique = np.array([10, 20, 30], dtype=np.int64)
        expected_remapped = np.array([0, 1, 2, 0], dtype=np.int64)

        unique, value2idx, remapped = remap_series_to_idcs(
            series, values_unique
        )
        np.testing.assert_array_equal(unique, values_unique)
        np.testing.assert_array_equal(remapped, expected_remapped)
        np.testing.assert_array_equal(
            value2idx(values_unique), np.array([0, 1, 2])
        )

    def test_provided_values_unique_with_missing(self) -> None:
        """Test remapping with provided values_unique and missing values."""
        series = pd.Series([10, 40, 20, 50])
        values_unique = np.array([10, 20, 30], dtype=np.int64)
        expected_remapped = np.array([0, np.nan, 1, np.nan], dtype=np.float64)

        unique, value2idx, remapped = remap_series_to_idcs(
            series, values_unique
        )
        np.testing.assert_array_equal(unique, values_unique)
        np.testing.assert_array_equal(remapped, expected_remapped)
        np.testing.assert_array_equal(
            value2idx(values_unique), np.array([0, 1, 2])
        )

    def test_no_values_unique_with_non_integer_values(self) -> None:
        """Test remapping without values_unique for non-integer values."""
        series = pd.Series(["a", "b", "a", "c", "b"])
        expected_unique = np.array(["a", "b", "c"], dtype=object)
        expected_remapped = np.array([0, 1, 0, 2, 1], dtype=np.int64)

        unique, value2idx, remapped = remap_series_to_idcs(series)
        np.testing.assert_array_equal(unique, expected_unique)
        np.testing.assert_array_equal(remapped, expected_remapped)
        np.testing.assert_array_equal(
            value2idx(expected_unique), np.array([0, 1, 2])
        )

    def test_empty_series(self) -> None:
        """Test behavior with an empty series."""
        series = pd.Series([], dtype=np.int64)
        expected_unique = np.array([], dtype=np.int64)
        expected_remapped = np.array([], dtype=np.int64)

        unique, value2idx, remapped = remap_series_to_idcs(series)
        np.testing.assert_array_equal(unique, expected_unique)
        np.testing.assert_array_equal(remapped, expected_remapped)
        np.testing.assert_array_equal(
            value2idx(expected_unique), np.array([], dtype=np.int64)
        )

    def test_provided_values_unique_empty(self) -> None:
        """Test behavior when provided values_unique is empty."""
        series = pd.Series([10, 20, 30])
        values_unique = np.array([], dtype=np.int64)
        expected_remapped = np.array(
            [np.nan, np.nan, np.nan], dtype=np.float64
        )

        unique, value2idx, remapped = remap_series_to_idcs(
            series, values_unique
        )
        np.testing.assert_array_equal(unique, values_unique)
        np.testing.assert_array_equal(remapped, expected_remapped)
        np.testing.assert_array_equal(
            value2idx(values_unique), np.array([], dtype=np.int64)
        )
