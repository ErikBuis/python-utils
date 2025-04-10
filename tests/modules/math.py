import unittest

from python_utils.modules.math import optimal_grid_layout


class TestOptimalGridLayout(unittest.TestCase):
    def test_optimal_grid_layout_1(self) -> None:
        n = 16
        cols, rows = optimal_grid_layout(n)
        self.assertEqual(cols, 4)
        self.assertEqual(rows, 4)

    def test_optimal_grid_layout_2(self) -> None:
        n = 24
        cols, rows = optimal_grid_layout(n)
        self.assertEqual(cols, 4)
        self.assertEqual(rows, 6)

    def test_optimal_grid_layout_constrained_rows_1(self) -> None:
        n = 16
        cols, rows = optimal_grid_layout(n, max_rows=3)
        self.assertEqual(cols, 8)
        self.assertEqual(rows, 2)

    def test_optimal_grid_layout_constrained_rows_2(self) -> None:
        n = 24
        cols, rows = optimal_grid_layout(n, max_rows=5)
        self.assertEqual(cols, 6)
        self.assertEqual(rows, 4)

    def test_optimal_grid_layout_constrained_cols_1(self) -> None:
        n = 16
        cols, rows = optimal_grid_layout(n, max_cols=1)
        self.assertEqual(cols, 1)
        self.assertEqual(rows, 16)

    def test_optimal_grid_layout_constrained_cols_2(self) -> None:
        n = 24
        cols, rows = optimal_grid_layout(n, max_cols=2)
        self.assertEqual(cols, 2)
        self.assertEqual(rows, 12)

    def test_optimal_grid_layout_constrained_nrows_x_ncols_eq_n_1(
        self,
    ) -> None:
        n = 16
        cols, rows = optimal_grid_layout(n, max_cols=10, max_rows=3)
        self.assertEqual(cols, 8)
        self.assertEqual(rows, 2)

    def test_optimal_grid_layout_constrained_nrows_x_ncols_eq_n_2(
        self,
    ) -> None:
        n = 24
        cols, rows = optimal_grid_layout(n, max_cols=10, max_rows=10)
        self.assertEqual(cols, 4)
        self.assertEqual(rows, 6)

    def test_optimal_grid_layout_constrained_nrows_x_ncols_gt_n_1(
        self,
    ) -> None:
        n = 24
        cols, rows = optimal_grid_layout(n, max_cols=5, max_rows=5)
        self.assertEqual(cols, 5)
        self.assertEqual(rows, 5)

    def test_optimal_grid_layout_constrained_nrows_x_ncols_gt_n_2(
        self,
    ) -> None:
        n = 21
        cols, rows = optimal_grid_layout(n, max_cols=6, max_rows=6)
        self.assertEqual(cols, 4)
        self.assertEqual(rows, 6)
