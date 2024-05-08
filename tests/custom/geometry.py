# pyright: reportPrivateUsage=false
# pyright: reportUninitializedInstanceVariable=false


import random
import unittest
from math import inf

import custom.geometry as geometry


class TestMatrix2D(unittest.TestCase):
    # __init__ should initialize the matrix with the given components.
    def test_init(self):
        matrix = geometry.Matrix2D(1, 2, 3, 4)
        self.assertEqual(matrix.a, 1)
        self.assertEqual(matrix.b, 2)
        self.assertEqual(matrix.c, 3)
        self.assertEqual(matrix.d, 4)

    # __iter__ should return an iterator over the matrix's components.
    def test_iter(self):
        matrix = geometry.Matrix2D(1, 2, 3, 4)
        components = list(matrix)
        self.assertEqual(components, [1, 2, 3, 4])

    # __getitem__ should return the component at the given index.
    def test_getitem(self):
        matrix = geometry.Matrix2D(1, 2, 3, 4)
        self.assertEqual(matrix[0], 1)
        self.assertEqual(matrix[1], 2)
        self.assertEqual(matrix[2], 3)
        self.assertEqual(matrix[3], 4)

    # __getitem__ should raise an IndexError if the index is out of bounds.
    def test_getitem_index_error(self):
        matrix = geometry.Matrix2D(1, 2, 3, 4)
        with self.assertRaises(IndexError):
            matrix[4]

    # __repr__ should return a string representation of the matrix.
    def test_repr(self):
        matrix = geometry.Matrix2D(1, 2, 3, 4)
        expected_output = "Matrix2D(1, 2, 3, 4)"
        self.assertEqual(repr(matrix), expected_output)

    # __str__ should return a formatted string representation of the matrix.
    def test_str(self):
        matrix = geometry.Matrix2D(1, 2, 3, 4)
        expected_output = "| 1  2 |\n| 3  4 |"
        self.assertEqual(str(matrix), expected_output)

    # __hash__ should return a unique hash id for the matrix.
    def test_hash(self):
        matrix1 = geometry.Matrix2D(1, 2, 3, 4)
        matrix2 = geometry.Matrix2D(1, 2, 3, 4)
        matrix3 = geometry.Matrix2D(5, 6, 7, 8)

        self.assertEqual(hash(matrix1), hash(matrix2))
        self.assertNotEqual(hash(matrix1), hash(matrix3))

    # __eq__ should return True iff two matrices are equal.
    def test_eq(self):
        matrix1 = geometry.Matrix2D(1, 2, 3, 4)
        matrix2 = geometry.Matrix2D(1, 2, 3, 4)
        self.assertTrue(matrix1 == matrix2)

        matrix1 = geometry.Matrix2D(1, 2, 3, 4)
        matrix2 = geometry.Matrix2D(5, 6, 7, 8)
        self.assertFalse(matrix1 == matrix2)

    # __eq__ should return False if the other object is not a matrix.
    def test_eq_non_matrix(self):
        matrix = geometry.Matrix2D(1, 2, 3, 4)
        non_matrix = "not a matrix"
        self.assertFalse(matrix == non_matrix)

    # __matmul__ should return a new matrix with the correct components when
    # multiplying a matrix with another matrix.
    def test_matmul_matrix(self):
        matrix1 = geometry.Matrix2D(1, 2, 3, 4)
        matrix2 = geometry.Matrix2D(5, 6, 7, 8)
        result = matrix1 @ matrix2
        self.assertEqual(result.a, 19)
        self.assertEqual(result.b, 22)
        self.assertEqual(result.c, 43)
        self.assertEqual(result.d, 50)

    # __matmul__ should return a new transformed geometric object when
    # multiplying a matrix with a geometric object.
    def test_matmul_geometric_object(self):
        matrix = geometry.Matrix2D(2, 0, 0, 2)
        point = geometry.Vector2D(1, 1)
        transformed_point = matrix @ point
        self.assertEqual(transformed_point.x, 2)
        self.assertEqual(transformed_point.y, 2)


class TestInterval(unittest.TestCase):
    # __init__ should initialize the interval with the given components.
    def test_init(self):
        interval = geometry.Interval("[", 1, 5, "]")
        self.assertEqual(interval.left_bracket, "[")
        self.assertEqual(interval.start, 1)
        self.assertEqual(interval.end, 5)
        self.assertEqual(interval.right_bracket, "]")

        interval = geometry.Interval("(", -3, 7, ")")
        self.assertEqual(interval.left_bracket, "(")
        self.assertEqual(interval.start, -3)
        self.assertEqual(interval.end, 7)
        self.assertEqual(interval.right_bracket, ")")

        interval = geometry.Interval("(", 0, 10, "]")
        self.assertEqual(interval.left_bracket, "(")
        self.assertEqual(interval.start, 0)
        self.assertEqual(interval.end, 10)
        self.assertEqual(interval.right_bracket, "]")

    # __init__ should raise a ValueError if the interval starts or ends with
    # infinity and the bracket is not closed.
    def test_init_inf_bracket_not_square(self):
        with self.assertRaises(ValueError):
            geometry.Interval("(", -inf, 5, "]")
        with self.assertRaises(ValueError):
            geometry.Interval("[", 1, inf, ")")

    # __init__ should raise a ValueError if the interval is empty.
    def test_init_interval_not_empty(self):
        with self.assertRaises(ValueError):
            geometry.Interval("[", 5, 1, "]")
        with self.assertRaises(ValueError):
            geometry.Interval("(", 5, 5, "]")
        with self.assertRaises(ValueError):
            geometry.Interval("[", 5, 5, ")")
        with self.assertRaises(ValueError):
            geometry.Interval("(", 5, 5, ")")

    # __init__ should raise a ValueError if the interval is a point.
    def test_init_interval_not_point(self):
        with self.assertRaises(ValueError):
            geometry.Interval("[", 5, 5, "]")

    # __init__ should raise a ValueError if a bracket is the incorrect way
    # around.
    def test_init_invalid_bracket(self):
        with self.assertRaises(ValueError):
            geometry.Interval("]", 1, 5, "]")  # type: ignore
        with self.assertRaises(ValueError):
            geometry.Interval("[", 1, 5, "(")  # type: ignore

    # __iter__ should return an iterator over the interval's components.
    def test_iter(self):
        interval = geometry.Interval("[", 1, 5, "]")
        components = list(interval)
        self.assertEqual(components, [True, 1, 5, True])

    # __getitem__ should return the component at the given index.
    def test_getitem(self):
        interval = geometry.Interval("[", 1, 5, "]")
        self.assertEqual(interval[0], True)
        self.assertEqual(interval[1], 1)
        self.assertEqual(interval[2], 5)
        self.assertEqual(interval[3], True)

    # __getitem__ should raise an IndexError if the index is out of bounds.
    def test_getitem_invalid_index(self):
        interval = geometry.Interval("[", 1, 5, "]")
        with self.assertRaises(IndexError):
            interval[4]

    # __repr__ should return a string representation of the interval.
    def test_interval_repr(self):
        interval = geometry.Interval("[", 1, 5, "]")
        self.assertEqual(repr(interval), "Interval('[', 1, 5, ']')")
        interval = geometry.Interval("[", -inf, 10, "]")
        self.assertEqual(repr(interval), "Interval('[', -inf, 10, ']')")
        interval = geometry.Interval("[", -5, inf, "]")
        self.assertEqual(repr(interval), "Interval('[', -5, inf, ']')")
        interval = geometry.Interval("(", -3.5, 7.8, ")")
        self.assertEqual(repr(interval), "Interval('(', -3.5, 7.8, ')')")

    # __str__ should return a formatted string representation of the interval.
    def test_interval_str(self):
        interval = geometry.Interval("[", 1, 5, "]")
        self.assertEqual(str(interval), "[1, 5]")
        interval = geometry.Interval("[", -inf, 10, "]")
        self.assertEqual(str(interval), "[-inf, 10]")
        interval = geometry.Interval("[", -5, inf, "]")
        self.assertEqual(str(interval), "[-5, inf]")
        interval = geometry.Interval("(", -3.5, 7.8, ")")
        self.assertEqual(str(interval), "(-3.5, 7.8)")

    # __hash__ should return a unique hash id for the interval.
    def test_interval_hash(self):
        interval1 = geometry.Interval("[", 1, 5, "]")
        interval2 = geometry.Interval("[", 1, 5, "]")
        interval3 = geometry.Interval("[", 2, 5, "]")

        self.assertEqual(hash(interval1), hash(interval2))
        self.assertNotEqual(hash(interval1), hash(interval3))

    # __eq__ should return True iff two intervals are equal.
    def test_eq(self):
        interval1 = geometry.Interval("[", 1, 5, "]")
        interval2 = geometry.Interval("[", 1, 5, "]")
        self.assertEqual(interval1, interval2)

        interval1 = geometry.Interval("[", -inf, 10, "]")
        interval2 = geometry.Interval("[", -inf, 10, ")")
        self.assertNotEqual(interval1, interval2)

    # __eq__ should return False if the other object is not an interval.
    def test_eq_non_interval(self):
        interval = geometry.Interval("[", 1, 5, "]")
        non_interval_object = "not an interval"
        self.assertFalse(interval == non_interval_object)


class TestNumberSet(unittest.TestCase):
    def setUp(self) -> None:
        self.numberset = geometry.NumberSet()._direct_init(
            [0, 0, 1, 2, 3, 5, 5, 6, 8, inf],
            [True, True, True, True, False, False, False, False, False, True],
        )

    # amount_components should return the correct amount of components in the
    # set.
    def test_amount_components(self):
        self.assertEqual(self.numberset.amount_components, 5)

    # components should return the correct components.
    def test_components(self):
        self.assertEqual(
            list(self.numberset.components),
            [
                0,
                geometry.Interval("[", 1, 2, "]"),
                geometry.Interval("(", 3, 5, ")"),
                geometry.Interval("(", 5, 6, ")"),
                geometry.Interval("(", 8, inf, "]"),
            ],
        )

    # __iter__ should return an iterator over the set's components.
    def test_iter(self):
        components = list(self.numberset)
        self.assertEqual(
            components,
            [
                0,
                geometry.Interval("[", 1, 2, "]"),
                geometry.Interval("(", 3, 5, ")"),
                geometry.Interval("(", 5, 6, ")"),
                geometry.Interval("(", 8, inf, "]"),
            ],
        )

    # __getitem__ should return the component at the given index.
    def test_getitem(self):
        self.assertEqual(self.numberset[0], 0)
        self.assertEqual(self.numberset[1], geometry.Interval("[", 1, 2, "]"))
        self.assertEqual(self.numberset[2], geometry.Interval("(", 3, 5, ")"))
        self.assertEqual(self.numberset[3], geometry.Interval("(", 5, 6, ")"))
        self.assertEqual(
            self.numberset[4], geometry.Interval("(", 8, inf, "]")
        )

    # __getitem__ should raise an IndexError if the index is out of bounds.
    def test_getitem_index_error(self):
        with self.assertRaises(IndexError):
            self.numberset[5]

    # __repr__ should return a string representation of the set.
    def test_repr(self):
        self.assertEqual(
            repr(self.numberset),
            "NumberSet(0, Interval('[', 1, 2, ']'), "
            "Interval('(', 3, 5, ')'), Interval('(', 5, 6, ')'), "
            "Interval('(', 8, inf, ']'))",
        )

    # __str__ should return a formatted string representation of the set.
    def test_str(self):
        self.assertEqual(
            str(self.numberset),
            "NumberSet{0, [1, 2], (3, 5), (5, 6), (8, inf]}",
        )

    # __bool__ should return True iff the set is not empty.
    def test_bool(self):
        self.assertTrue(self.numberset)
        self.assertFalse(geometry.NumberSet())

    # __eq__ should return True iff two sets are equal.
    def test_eq(self):
        numberset1 = geometry.NumberSet._direct_init(
            [0, 0, 1, 2, 3, 5, 5, 6, 8, inf],
            [True, True, True, True, False, False, False, False, False, True],
        )
        numberset2 = geometry.NumberSet._direct_init(
            [0, 0, 1, 2, 3, 5, 5, 6, 8, inf],
            [True, True, True, True, False, False, False, False, True, True],
        )
        self.assertEqual(self.numberset, numberset1)
        self.assertNotEqual(self.numberset, numberset2)

    # __eq__ should return False if the other object is not a set.
    def test_eq_non_set(self):
        self.assertNotEqual(self.numberset, "not a set")

    # __lt__ should return True iff all numbers in the first set are left of
    # all numbers in the second set.
    def test_lt(self):
        numberset1 = geometry.NumberSet._direct_init(
            [0, 0, 1, 2, 3, 5, 5, 6, 8, inf],
            [True, True, True, True, False, False, False, False, False, True],
        )
        numberset2 = geometry.NumberSet._direct_init(
            [-inf, -8, -6, -5, -5, -3, -2, -1, 0, 0],
            [True, False, False, False, False, False, True, True, True, True],
        )
        numberset3 = geometry.NumberSet._direct_init(
            [-inf, -8, -6, -5, -5, -3, -2, -1],
            [True, False, False, False, False, False, True, True],
        )
        self.assertFalse(self.numberset < numberset1)
        self.assertFalse(numberset1 < self.numberset)
        self.assertFalse(self.numberset < numberset2)
        self.assertFalse(numberset2 < self.numberset)
        self.assertFalse(self.numberset < numberset3)
        self.assertTrue(numberset3 < self.numberset)

    # __gt__ should return True iff all numbers in the first set are right of
    # all numbers in the second set.
    def test_gt(self):
        numberset1 = geometry.NumberSet._direct_init(
            [0, 0, 1, 2, 3, 5, 5, 6, 8, inf],
            [True, True, True, True, False, False, False, False, False, True],
        )
        numberset2 = geometry.NumberSet._direct_init(
            [-inf, -8, -6, -5, -5, -3, -2, -1, 0, 0],
            [True, False, False, False, False, False, True, True, True, True],
        )
        numberset3 = geometry.NumberSet._direct_init(
            [-inf, -8, -6, -5, -5, -3, -2, -1],
            [True, False, False, False, False, False, True, True],
        )
        self.assertFalse(self.numberset > numberset1)
        self.assertFalse(numberset1 > self.numberset)
        self.assertFalse(self.numberset > numberset2)
        self.assertFalse(numberset2 > self.numberset)
        self.assertTrue(self.numberset > numberset3)
        self.assertFalse(numberset3 > self.numberset)

    # __contains__ should return True iff the number is in the set.
    def test_contains(self):
        self.assertFalse(-inf in self.numberset)
        self.assertFalse(-1 in self.numberset)
        self.assertTrue(0 in self.numberset)
        self.assertTrue(1 in self.numberset)
        self.assertTrue(2 in self.numberset)
        self.assertFalse(3 in self.numberset)
        self.assertTrue(4 in self.numberset)
        self.assertFalse(5 in self.numberset)
        self.assertFalse(6 in self.numberset)
        self.assertFalse(7 in self.numberset)
        self.assertFalse(8 in self.numberset)
        self.assertTrue(9 in self.numberset)
        self.assertTrue(inf in self.numberset)

    # __invert__ should return the complement of the set.
    def test_invert(self):
        inv_numberset = ~self.numberset
        self.assertEqual(
            inv_numberset._boundaries, [-inf, 0, 0, 1, 2, 3, 5, 5, 6, 8]
        )
        self.assertEqual(
            inv_numberset._boundaries_included,
            [True, False, False, False, False, True, True, True, True, True],
        )

    # __lshift__ should return the set shifted left by the given amount.
    def test_lshift(self):
        numberset = self.numberset << 2
        self.assertEqual(
            numberset._boundaries, [-2, -2, -1, 0, 1, 3, 3, 4, 6, inf]
        )
        self.assertEqual(
            numberset._boundaries_included,
            [True, True, True, True, False, False, False, False, False, True],
        )

    # __rshift__ should return the set shifted right by the given amount.
    def test_rshift(self):
        numberset = self.numberset >> 2
        self.assertEqual(
            numberset._boundaries, [2, 2, 3, 4, 5, 7, 7, 8, 10, inf]
        )
        self.assertEqual(
            numberset._boundaries_included,
            [True, True, True, True, False, False, False, False, False, True],
        )

    # copy should return a copy of the set that doesn't share any references
    # with the original set.
    def test_copy(self):
        numberset_copy = self.numberset.copy()
        self.assertIsNot(numberset_copy, self.numberset)
        self.assertIsNot(
            numberset_copy._boundaries, self.numberset._boundaries
        )
        self.assertIsNot(
            numberset_copy._boundaries_included,
            self.numberset._boundaries_included,
        )

    # contains_parallel should return which sets the numbers are contained in.
    def test_contains_parallel(self):
        # Set random seed for reproducibility.
        random.seed(69)

        # Decide on a set of possible bounds.
        possible_bounds = list(range(-10, 11))
        possible_inclusions = [True, False]

        # Generate a list of random intervals. Do this by randomly choosing a
        # start and end bound for each interval and whether they are included
        # from the possible bounds and inclusions.
        intervals = []
        for _ in range(100):
            start, end = 0, 0
            start_included, end_included = True, True
            while start == end:
                start = random.choice(possible_bounds)
                end = random.choice(possible_bounds)
                start_included = random.choice(possible_inclusions)
                end_included = random.choice(possible_inclusions)
            if start > end:
                start, end = end, start
                start_included, end_included = end_included, start_included
            intervals.append(
                geometry.Interval(start_included, start, end, end_included)
            )
        numbersets = [geometry.NumberSet(interval) for interval in intervals]

        # For each number, check that the result of contains_parallel is the
        # same as the result of contains for each set.
        results_parallel = geometry.NumberSet.contains_parallel(
            numbersets, possible_bounds
        )
        for number, result_parallel in zip(possible_bounds, results_parallel):
            result_parallel = sorted(result_parallel)
            expected_result = [
                i
                for i, numberset in enumerate(numbersets)
                if number in numberset
            ]
            self.assertEqual(expected_result, result_parallel)

    # lookup returns 4 elements of the correct type.
    def test_lookup_type(self):
        numberset = geometry.NumberSet()
        in_set, on_start, on_end, idx = numberset.lookup(0)
        self.assertIsInstance(in_set, bool)
        self.assertIsInstance(on_start, bool)
        self.assertIsInstance(on_end, bool)
        self.assertIsInstance(idx, int)

    # lookup returns the correct values if number is on a point.
    def test_lookup_point(self):
        in_set, on_start, on_end, idx = self.numberset.lookup(0)
        self.assertTrue(in_set)
        self.assertTrue(on_start)
        self.assertTrue(on_end)
        self.assertEqual(idx, 0)

    # lookup returns the correct values if number is on an included start
    # bound.
    def test_lookup_included_start_bound(self):
        in_set, on_start, on_end, idx = self.numberset.lookup(1)
        self.assertTrue(in_set)
        self.assertTrue(on_start)
        self.assertFalse(on_end)
        self.assertEqual(idx, 2)

    # lookup returns the correct values if number is on an included end bound.
    def test_lookup_included_end_bound(self):
        in_set, on_start, on_end, idx = self.numberset.lookup(2)
        self.assertTrue(in_set)
        self.assertFalse(on_start)
        self.assertTrue(on_end)
        self.assertEqual(idx, 3)

    # lookup returns the correct values if number is in an interval.
    def test_lookup_in_interval(self):
        in_set, on_start, on_end, idx = self.numberset.lookup(4)
        self.assertTrue(in_set)
        self.assertFalse(on_start)
        self.assertFalse(on_end)
        self.assertEqual(idx, 5)

    # lookup returns the correct values if number is in a hole between
    # intervals.
    def test_lookup_hole(self):
        in_set, on_start, on_end, idx = self.numberset.lookup(5)
        self.assertFalse(in_set)
        self.assertTrue(on_start)
        self.assertTrue(on_end)
        self.assertEqual(idx, 5)

    # lookup returns the correct values if number is on an excluded start
    # bound.
    def test_lookup_excluded_start_bound(self):
        in_set, on_start, on_end, idx = self.numberset.lookup(3)
        self.assertFalse(in_set)
        self.assertTrue(on_start)
        self.assertFalse(on_end)
        self.assertEqual(idx, 4)

    # lookup returns the correct values if number is on an excluded end bound.
    def test_lookup_excluded_end_bound(self):
        in_set, on_start, on_end, idx = self.numberset.lookup(6)
        self.assertFalse(in_set)
        self.assertFalse(on_start)
        self.assertTrue(on_end)
        self.assertEqual(idx, 7)

    # lookup returns the correct values if number is outside all components.
    def test_lookup_outside_components(self):
        in_set, on_start, on_end, idx = self.numberset.lookup(7)
        self.assertFalse(in_set)
        self.assertFalse(on_start)
        self.assertFalse(on_end)
        self.assertEqual(idx, 8)

    # lookup returns the correct values if number is -inf or inf.
    def test_lookup_inf(self):
        in_set, on_start, on_end, idx = self.numberset.lookup(-inf)
        self.assertFalse(in_set)
        self.assertFalse(on_start)
        self.assertFalse(on_end)
        self.assertEqual(idx, 0)
        in_set, on_start, on_end, idx = self.numberset.lookup(inf)
        self.assertTrue(in_set)
        self.assertFalse(on_start)
        self.assertTrue(on_end)
        self.assertEqual(idx, 9)

    # add_number should add a number to the set if number is on a point.
    def test_add_number_point(self):
        numberset = self.numberset.copy()
        numberset.add_number(0)
        self.assertEqual(
            numberset._boundaries, [0, 0, 1, 2, 3, 5, 5, 6, 8, inf]
        )
        self.assertEqual(
            numberset._boundaries_included,
            [True, True, True, True, False, False, False, False, False, True],
        )

    # add_number should add a number to the set if number is on an included
    # start bound.
    def test_add_number_included_start_bound(self):
        numberset = self.numberset.copy()
        numberset.add_number(1)
        self.assertEqual(
            numberset._boundaries, [0, 0, 1, 2, 3, 5, 5, 6, 8, inf]
        )
        self.assertEqual(
            numberset._boundaries_included,
            [True, True, True, True, False, False, False, False, False, True],
        )

    # add_number should add a number to the set if number is on an included
    # end bound.
    def test_add_number_included_end_bound(self):
        numberset = self.numberset.copy()
        numberset.add_number(2)
        self.assertEqual(
            numberset._boundaries, [0, 0, 1, 2, 3, 5, 5, 6, 8, inf]
        )
        self.assertEqual(
            numberset._boundaries_included,
            [True, True, True, True, False, False, False, False, False, True],
        )

    # add_number should add a number to the set if number is in an interval.
    def test_add_number_in_interval(self):
        numberset = self.numberset.copy()
        numberset.add_number(4)
        self.assertEqual(
            numberset._boundaries, [0, 0, 1, 2, 3, 5, 5, 6, 8, inf]
        )
        self.assertEqual(
            numberset._boundaries_included,
            [True, True, True, True, False, False, False, False, False, True],
        )

    # add_number should add a number to the set if number is in a hole between
    # intervals.
    def test_add_number_hole(self):
        numberset = self.numberset.copy()
        numberset.add_number(5)
        self.assertEqual(numberset._boundaries, [0, 0, 1, 2, 3, 6, 8, inf])
        self.assertEqual(
            numberset._boundaries_included,
            [True, True, True, True, False, False, False, True],
        )

    # add_number should add a number to the set if number is on an excluded
    # start bound.
    def test_add_number_excluded_start_bound(self):
        numberset = self.numberset.copy()
        numberset.add_number(3)
        self.assertEqual(
            numberset._boundaries, [0, 0, 1, 2, 3, 5, 5, 6, 8, inf]
        )
        self.assertEqual(
            numberset._boundaries_included,
            [True, True, True, True, True, False, False, False, False, True],
        )

    # add_number should add a number to the set if number is on an excluded
    # end bound.
    def test_add_number_excluded_end_bound(self):
        numberset = self.numberset.copy()
        numberset.add_number(6)
        self.assertEqual(
            numberset._boundaries, [0, 0, 1, 2, 3, 5, 5, 6, 8, inf]
        )
        self.assertEqual(
            numberset._boundaries_included,
            [True, True, True, True, False, False, False, True, False, True],
        )

    # add_number should add a number to the set if number is outside all
    # components.
    def test_add_number_outside_components(self):
        numberset = self.numberset.copy()
        numberset.add_number(7)
        self.assertEqual(
            numberset._boundaries, [0, 0, 1, 2, 3, 5, 5, 6, 7, 7, 8, inf]
        )
        self.assertEqual(
            numberset._boundaries_included,
            [
                True,
                True,
                True,
                True,
                False,
                False,
                False,
                False,
                True,
                True,
                False,
                True,
            ],
        )

    # add_number should raise a ValueError if number is -inf or inf.
    def test_add_number_inf(self):
        numberset = self.numberset.copy()
        with self.assertRaises(ValueError):
            numberset.add_number(-inf)
        with self.assertRaises(ValueError):
            numberset.add_number(inf)

    # remove_number should remove a number from the set if number is on a
    # point.
    def test_remove_number_point(self):
        numberset = self.numberset.copy()
        numberset.remove_number(0)
        self.assertEqual(numberset._boundaries, [1, 2, 3, 5, 5, 6, 8, inf])
        self.assertEqual(
            numberset._boundaries_included,
            [True, True, False, False, False, False, False, True],
        )

    # remove_number should remove a number from the set if number is on an
    # included start bound.
    def test_remove_number_included_start_bound(self):
        numberset = self.numberset.copy()
        numberset.remove_number(1)
        self.assertEqual(
            numberset._boundaries, [0, 0, 1, 2, 3, 5, 5, 6, 8, inf]
        )
        self.assertEqual(
            numberset._boundaries_included,
            [True, True, False, True, False, False, False, False, False, True],
        )

    # remove_number should remove a number from the set if number is on an
    # included end bound.
    def test_remove_number_included_end_bound(self):
        numberset = self.numberset.copy()
        numberset.remove_number(2)
        self.assertEqual(
            numberset._boundaries, [0, 0, 1, 2, 3, 5, 5, 6, 8, inf]
        )
        self.assertEqual(
            numberset._boundaries_included,
            [True, True, True, False, False, False, False, False, False, True],
        )

    # remove_number should remove a number from the set if number is in an
    # interval.
    def test_remove_number_in_interval(self):
        numberset = self.numberset.copy()
        numberset.remove_number(4)
        self.assertEqual(
            numberset._boundaries, [0, 0, 1, 2, 3, 4, 4, 5, 5, 6, 8, inf]
        )
        self.assertEqual(
            numberset._boundaries_included,
            [
                True,
                True,
                True,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                True,
            ],
        )

    # remove_number should remove a number from the set if number is in a hole
    # between intervals.
    def test_remove_number_hole(self):
        numberset = self.numberset.copy()
        numberset.remove_number(5)
        self.assertEqual(
            numberset._boundaries, [0, 0, 1, 2, 3, 5, 5, 6, 8, inf]
        )
        self.assertEqual(
            numberset._boundaries_included,
            [True, True, True, True, False, False, False, False, False, True],
        )

    # remove_number should remove a number from the set if number is on an
    # excluded start bound.
    def test_remove_number_excluded_start_bound(self):
        numberset = self.numberset.copy()
        numberset.remove_number(3)
        self.assertEqual(
            numberset._boundaries, [0, 0, 1, 2, 3, 5, 5, 6, 8, inf]
        )
        self.assertEqual(
            numberset._boundaries_included,
            [True, True, True, True, False, False, False, False, False, True],
        )

    # remove_number should remove a number from the set if number is on an
    # excluded end bound.
    def test_remove_number_excluded_end_bound(self):
        numberset = self.numberset.copy()
        numberset.remove_number(6)
        self.assertEqual(
            numberset._boundaries, [0, 0, 1, 2, 3, 5, 5, 6, 8, inf]
        )
        self.assertEqual(
            numberset._boundaries_included,
            [True, True, True, True, False, False, False, False, False, True],
        )

    # remove_number should remove a number from the set if number is outside
    # all components.
    def test_remove_number_outside_components(self):
        numberset = self.numberset.copy()
        numberset.remove_number(7)
        self.assertEqual(
            numberset._boundaries, [0, 0, 1, 2, 3, 5, 5, 6, 8, inf]
        )
        self.assertEqual(
            numberset._boundaries_included,
            [True, True, True, True, False, False, False, False, False, True],
        )

    # remove_number should raise a ValueError if number is -inf or inf.
    def test_remove_number_inf(self):
        numberset = self.numberset.copy()
        with self.assertRaises(ValueError):
            numberset.remove_number(-inf)
        with self.assertRaises(ValueError):
            numberset.remove_number(inf)

    # _extract_subset should return the correct subset if the subset is
    # discarded.
    def test_extract_subset_discarded(self):
        subset = self.numberset._extract_subset(-inf, inf, True, True, "00")
        self.assertEqual(subset._boundaries, [])
        self.assertEqual(subset._boundaries_included, [])
        subset = self.numberset._extract_subset(0, 5, True, True, "00")
        self.assertEqual(subset._boundaries, [])
        self.assertEqual(subset._boundaries_included, [])
        subset = self.numberset._extract_subset(0, 5, False, False, "00")
        self.assertEqual(subset._boundaries, [])
        self.assertEqual(subset._boundaries_included, [])

    # _extract_subset should return the correct subset if the subset is
    # returned in full.
    def test_extract_subset_full(self):
        subset = self.numberset._extract_subset(-inf, inf, True, True, "11")
        self.assertEqual(subset._boundaries, [-inf, inf])
        self.assertEqual(subset._boundaries_included, [True, True])
        subset = self.numberset._extract_subset(0, 5, True, True, "11")
        self.assertEqual(subset._boundaries, [0, 5])
        self.assertEqual(subset._boundaries_included, [True, True])
        subset = self.numberset._extract_subset(0, 5, False, False, "11")
        self.assertEqual(subset._boundaries, [0, 5])
        self.assertEqual(subset._boundaries_included, [False, False])

    # _extract_subset should raise a ValueError if the start is greater than
    # the end.
    def test_extract_subset_start_greater_end(self):
        with self.assertRaises(ValueError):
            self.numberset._extract_subset(1, 0, True, True, "11")

    # _extract_subset should raise a ValueError if the start is -inf and it is
    # not included, or if the end is inf and it is not included.
    def test_extract_subset_inf_not_included(self):
        with self.assertRaises(ValueError):
            self.numberset._extract_subset(-inf, 0, False, True, "11")
        with self.assertRaises(ValueError):
            self.numberset._extract_subset(0, inf, True, False, "11")

    # _extract_subset should return the correct subset if a boundary is on a
    # point.
    def test_extract_subset_ab_point(self):
        subset = self.numberset._extract_subset(-inf, 0, True, True, "01")
        self.assertEqual(subset._boundaries, [0, 0])
        self.assertEqual(subset._boundaries_included, [True, True])
        subset = self.numberset._extract_subset(-inf, 0, True, False, "01")
        self.assertEqual(subset._boundaries, [])
        self.assertEqual(subset._boundaries_included, [])
        subset = self.numberset._extract_subset(0, inf, True, True, "01")
        self.assertEqual(subset._boundaries, [0, 0, 1, 2, 3, 5, 5, 6, 8, inf])
        self.assertEqual(
            subset._boundaries_included,
            [True, True, True, True, False, False, False, False, False, True],
        )
        subset = self.numberset._extract_subset(0, inf, False, True, "01")
        self.assertEqual(subset._boundaries, [1, 2, 3, 5, 5, 6, 8, inf])
        self.assertEqual(
            subset._boundaries_included,
            [True, True, False, False, False, False, False, True],
        )

        subset = self.numberset._extract_subset(-inf, 0, True, True, "10")
        self.assertEqual(subset._boundaries, [-inf, 0])
        self.assertEqual(subset._boundaries_included, [True, False])
        subset = self.numberset._extract_subset(-inf, 0, True, False, "10")
        self.assertEqual(subset._boundaries, [-inf, 0])
        self.assertEqual(subset._boundaries_included, [True, False])
        subset = self.numberset._extract_subset(0, inf, True, True, "10")
        self.assertEqual(subset._boundaries, [0, 1, 2, 3, 5, 5, 6, 8])
        self.assertEqual(
            subset._boundaries_included,
            [False, False, False, True, True, True, True, True],
        )
        subset = self.numberset._extract_subset(0, inf, False, True, "10")
        self.assertEqual(subset._boundaries, [0, 1, 2, 3, 5, 5, 6, 8])
        self.assertEqual(
            subset._boundaries_included,
            [False, False, False, True, True, True, True, True],
        )

    # _extract_subset should return the correct subset if a boundary is on an
    # included start bound.
    def test_extract_subset_ab_included_start_bound(self):
        subset = self.numberset._extract_subset(-inf, 1, True, True, "01")
        self.assertEqual(subset._boundaries, [0, 0, 1, 1])
        self.assertEqual(subset._boundaries_included, [True, True, True, True])
        subset = self.numberset._extract_subset(-inf, 1, True, False, "01")
        self.assertEqual(subset._boundaries, [0, 0])
        self.assertEqual(subset._boundaries_included, [True, True])
        subset = self.numberset._extract_subset(1, inf, True, True, "01")
        self.assertEqual(subset._boundaries, [1, 2, 3, 5, 5, 6, 8, inf])
        self.assertEqual(
            subset._boundaries_included,
            [True, True, False, False, False, False, False, True],
        )
        subset = self.numberset._extract_subset(1, inf, False, True, "01")
        self.assertEqual(subset._boundaries, [1, 2, 3, 5, 5, 6, 8, inf])
        self.assertEqual(
            subset._boundaries_included,
            [False, True, False, False, False, False, False, True],
        )

        subset = self.numberset._extract_subset(-inf, 1, True, True, "10")
        self.assertEqual(subset._boundaries, [-inf, 0, 0, 1])
        self.assertEqual(
            subset._boundaries_included, [True, False, False, False]
        )
        subset = self.numberset._extract_subset(-inf, 1, True, False, "10")
        self.assertEqual(subset._boundaries, [-inf, 0, 0, 1])
        self.assertEqual(
            subset._boundaries_included, [True, False, False, False]
        )
        subset = self.numberset._extract_subset(1, inf, True, True, "10")
        self.assertEqual(subset._boundaries, [2, 3, 5, 5, 6, 8])
        self.assertEqual(
            subset._boundaries_included, [False, True, True, True, True, True]
        )
        subset = self.numberset._extract_subset(1, inf, False, True, "10")
        self.assertEqual(subset._boundaries, [2, 3, 5, 5, 6, 8])
        self.assertEqual(
            subset._boundaries_included, [False, True, True, True, True, True]
        )

    # _extract_subset should return the correct subset if a boundary is on an
    # included end bound.
    def test_extract_subset_ab_included_end_bound(self):
        subset = self.numberset._extract_subset(-inf, 2, True, True, "01")
        self.assertEqual(subset._boundaries, [0, 0, 1, 2])
        self.assertEqual(subset._boundaries_included, [True, True, True, True])
        subset = self.numberset._extract_subset(-inf, 2, True, False, "01")
        self.assertEqual(subset._boundaries, [0, 0, 1, 2])
        self.assertEqual(
            subset._boundaries_included, [True, True, True, False]
        )
        subset = self.numberset._extract_subset(2, inf, True, True, "01")
        self.assertEqual(subset._boundaries, [2, 2, 3, 5, 5, 6, 8, inf])
        self.assertEqual(
            subset._boundaries_included,
            [True, True, False, False, False, False, False, True],
        )
        subset = self.numberset._extract_subset(2, inf, False, True, "01")
        self.assertEqual(subset._boundaries, [3, 5, 5, 6, 8, inf])
        self.assertEqual(
            subset._boundaries_included,
            [False, False, False, False, False, True],
        )

        subset = self.numberset._extract_subset(-inf, 2, True, True, "10")
        self.assertEqual(subset._boundaries, [-inf, 0, 0, 1])
        self.assertEqual(
            subset._boundaries_included, [True, False, False, False]
        )
        subset = self.numberset._extract_subset(-inf, 2, True, False, "10")
        self.assertEqual(subset._boundaries, [-inf, 0, 0, 1])
        self.assertEqual(
            subset._boundaries_included, [True, False, False, False]
        )
        subset = self.numberset._extract_subset(2, inf, True, True, "10")
        self.assertEqual(subset._boundaries, [2, 3, 5, 5, 6, 8])
        self.assertEqual(
            subset._boundaries_included, [False, True, True, True, True, True]
        )
        subset = self.numberset._extract_subset(2, inf, False, True, "10")
        self.assertEqual(subset._boundaries, [2, 3, 5, 5, 6, 8])
        self.assertEqual(
            subset._boundaries_included, [False, True, True, True, True, True]
        )

    # _extract_subset should return the correct subset if a boundary is in an
    # interval.
    def test_extract_subset_ab_in_interval(self):
        subset = self.numberset._extract_subset(-inf, 4, True, True, "01")
        self.assertEqual(subset._boundaries, [0, 0, 1, 2, 3, 4])
        self.assertEqual(
            subset._boundaries_included, [True, True, True, True, False, True]
        )
        subset = self.numberset._extract_subset(-inf, 4, True, False, "01")
        self.assertEqual(subset._boundaries, [0, 0, 1, 2, 3, 4])
        self.assertEqual(
            subset._boundaries_included, [True, True, True, True, False, False]
        )
        subset = self.numberset._extract_subset(4, inf, True, True, "01")
        self.assertEqual(subset._boundaries, [4, 5, 5, 6, 8, inf])
        self.assertEqual(
            subset._boundaries_included,
            [True, False, False, False, False, True],
        )
        subset = self.numberset._extract_subset(4, inf, False, True, "01")
        self.assertEqual(subset._boundaries, [4, 5, 5, 6, 8, inf])
        self.assertEqual(
            subset._boundaries_included,
            [False, False, False, False, False, True],
        )

        subset = self.numberset._extract_subset(-inf, 4, True, True, "10")
        self.assertEqual(subset._boundaries, [-inf, 0, 0, 1, 2, 3])
        self.assertEqual(
            subset._boundaries_included,
            [True, False, False, False, False, True],
        )
        subset = self.numberset._extract_subset(-inf, 4, True, False, "10")
        self.assertEqual(subset._boundaries, [-inf, 0, 0, 1, 2, 3])
        self.assertEqual(
            subset._boundaries_included,
            [True, False, False, False, False, True],
        )
        subset = self.numberset._extract_subset(4, inf, True, True, "10")
        self.assertEqual(subset._boundaries, [5, 5, 6, 8])
        self.assertEqual(subset._boundaries_included, [True, True, True, True])
        subset = self.numberset._extract_subset(4, inf, False, True, "10")
        self.assertEqual(subset._boundaries, [5, 5, 6, 8])
        self.assertEqual(subset._boundaries_included, [True, True, True, True])

    # _extract_subset should return the correct subset if a boundary is in a
    # hole between intervals.
    def test_extract_subset_ab_hole(self):
        subset = self.numberset._extract_subset(-inf, 5, True, True, "01")
        self.assertEqual(subset._boundaries, [0, 0, 1, 2, 3, 5])
        self.assertEqual(
            subset._boundaries_included, [True, True, True, True, False, False]
        )
        subset = self.numberset._extract_subset(-inf, 5, True, False, "01")
        self.assertEqual(subset._boundaries, [0, 0, 1, 2, 3, 5])
        self.assertEqual(
            subset._boundaries_included, [True, True, True, True, False, False]
        )
        subset = self.numberset._extract_subset(5, inf, True, True, "01")
        self.assertEqual(subset._boundaries, [5, 6, 8, inf])
        self.assertEqual(
            subset._boundaries_included, [False, False, False, True]
        )
        subset = self.numberset._extract_subset(5, inf, False, True, "01")
        self.assertEqual(subset._boundaries, [5, 6, 8, inf])
        self.assertEqual(
            subset._boundaries_included, [False, False, False, True]
        )

        subset = self.numberset._extract_subset(-inf, 5, True, True, "10")
        self.assertEqual(subset._boundaries, [-inf, 0, 0, 1, 2, 3, 5, 5])
        self.assertEqual(
            subset._boundaries_included,
            [True, False, False, False, False, True, True, True],
        )
        subset = self.numberset._extract_subset(-inf, 5, True, False, "10")
        self.assertEqual(subset._boundaries, [-inf, 0, 0, 1, 2, 3])
        self.assertEqual(
            subset._boundaries_included,
            [True, False, False, False, False, True],
        )
        subset = self.numberset._extract_subset(5, inf, True, True, "10")
        self.assertEqual(subset._boundaries, [5, 5, 6, 8])
        self.assertEqual(subset._boundaries_included, [True, True, True, True])
        subset = self.numberset._extract_subset(5, inf, False, True, "10")
        self.assertEqual(subset._boundaries, [6, 8])
        self.assertEqual(subset._boundaries_included, [True, True])

    # _extract_subset should return the correct subset if a boundary is on an
    # excluded start bound.
    def test_extract_subset_ab_excluded_start_bound(self):
        subset = self.numberset._extract_subset(-inf, 3, True, True, "01")
        self.assertEqual(subset._boundaries, [0, 0, 1, 2])
        self.assertEqual(subset._boundaries_included, [True, True, True, True])
        subset = self.numberset._extract_subset(-inf, 3, True, False, "01")
        self.assertEqual(subset._boundaries, [0, 0, 1, 2])
        self.assertEqual(subset._boundaries_included, [True, True, True, True])
        subset = self.numberset._extract_subset(3, inf, True, True, "01")
        self.assertEqual(subset._boundaries, [3, 5, 5, 6, 8, inf])
        self.assertEqual(
            subset._boundaries_included,
            [False, False, False, False, False, True],
        )
        subset = self.numberset._extract_subset(3, inf, False, True, "01")
        self.assertEqual(subset._boundaries, [3, 5, 5, 6, 8, inf])
        self.assertEqual(
            subset._boundaries_included,
            [False, False, False, False, False, True],
        )

        subset = self.numberset._extract_subset(-inf, 3, True, True, "10")
        self.assertEqual(subset._boundaries, [-inf, 0, 0, 1, 2, 3])
        self.assertEqual(
            subset._boundaries_included,
            [True, False, False, False, False, True],
        )
        subset = self.numberset._extract_subset(-inf, 3, True, False, "10")
        self.assertEqual(subset._boundaries, [-inf, 0, 0, 1, 2, 3])
        self.assertEqual(
            subset._boundaries_included,
            [True, False, False, False, False, False],
        )
        subset = self.numberset._extract_subset(3, inf, True, True, "10")
        self.assertEqual(subset._boundaries, [3, 3, 5, 5, 6, 8])
        self.assertEqual(
            subset._boundaries_included, [True, True, True, True, True, True]
        )
        subset = self.numberset._extract_subset(3, inf, False, True, "10")
        self.assertEqual(subset._boundaries, [5, 5, 6, 8])
        self.assertEqual(subset._boundaries_included, [True, True, True, True])

    # _extract_subset should return the correct subset if a boundary is on an
    # excluded end bound.
    def test_extract_subset_ab_excluded_end_bound(self):
        subset = self.numberset._extract_subset(-inf, 6, True, True, "01")
        self.assertEqual(subset._boundaries, [0, 0, 1, 2, 3, 5, 5, 6])
        self.assertEqual(
            subset._boundaries_included,
            [True, True, True, True, False, False, False, False],
        )
        subset = self.numberset._extract_subset(-inf, 6, True, False, "01")
        self.assertEqual(subset._boundaries, [0, 0, 1, 2, 3, 5, 5, 6])
        self.assertEqual(
            subset._boundaries_included,
            [True, True, True, True, False, False, False, False],
        )
        subset = self.numberset._extract_subset(6, inf, True, True, "01")
        self.assertEqual(subset._boundaries, [8, inf])
        self.assertEqual(subset._boundaries_included, [False, True])
        subset = self.numberset._extract_subset(6, inf, False, True, "01")
        self.assertEqual(subset._boundaries, [8, inf])
        self.assertEqual(subset._boundaries_included, [False, True])

        subset = self.numberset._extract_subset(-inf, 6, True, True, "10")
        self.assertEqual(subset._boundaries, [-inf, 0, 0, 1, 2, 3, 5, 5, 6, 6])
        self.assertEqual(
            subset._boundaries_included,
            [True, False, False, False, False, True, True, True, True, True],
        )
        subset = self.numberset._extract_subset(-inf, 6, True, False, "10")
        self.assertEqual(subset._boundaries, [-inf, 0, 0, 1, 2, 3, 5, 5])
        self.assertEqual(
            subset._boundaries_included,
            [True, False, False, False, False, True, True, True],
        )
        subset = self.numberset._extract_subset(6, inf, True, True, "10")
        self.assertEqual(subset._boundaries, [6, 8])
        self.assertEqual(subset._boundaries_included, [True, True])
        subset = self.numberset._extract_subset(6, inf, False, True, "10")
        self.assertEqual(subset._boundaries, [6, 8])
        self.assertEqual(subset._boundaries_included, [False, True])

    # _extract_subset should return the correct subset if a boundary is outside
    # all components.
    def test_extract_subset_ab_outside_components(self):
        subset = self.numberset._extract_subset(-inf, 7, True, True, "01")
        self.assertEqual(subset._boundaries, [0, 0, 1, 2, 3, 5, 5, 6])
        self.assertEqual(
            subset._boundaries_included,
            [True, True, True, True, False, False, False, False],
        )
        subset = self.numberset._extract_subset(-inf, 7, True, False, "01")
        self.assertEqual(subset._boundaries, [0, 0, 1, 2, 3, 5, 5, 6])
        self.assertEqual(
            subset._boundaries_included,
            [True, True, True, True, False, False, False, False],
        )
        subset = self.numberset._extract_subset(7, inf, True, True, "01")
        self.assertEqual(subset._boundaries, [8, inf])
        self.assertEqual(subset._boundaries_included, [False, True])
        subset = self.numberset._extract_subset(7, inf, False, True, "01")
        self.assertEqual(subset._boundaries, [8, inf])
        self.assertEqual(subset._boundaries_included, [False, True])

        subset = self.numberset._extract_subset(-inf, 7, True, True, "10")
        self.assertEqual(subset._boundaries, [-inf, 0, 0, 1, 2, 3, 5, 5, 6, 7])
        self.assertEqual(
            subset._boundaries_included,
            [True, False, False, False, False, True, True, True, True, True],
        )
        subset = self.numberset._extract_subset(-inf, 7, True, False, "10")
        self.assertEqual(subset._boundaries, [-inf, 0, 0, 1, 2, 3, 5, 5, 6, 7])
        self.assertEqual(
            subset._boundaries_included,
            [True, False, False, False, False, True, True, True, True, False],
        )
        subset = self.numberset._extract_subset(7, inf, True, True, "10")
        self.assertEqual(subset._boundaries, [7, 8])
        self.assertEqual(subset._boundaries_included, [True, True])
        subset = self.numberset._extract_subset(7, inf, False, True, "10")
        self.assertEqual(subset._boundaries, [7, 8])
        self.assertEqual(subset._boundaries_included, [False, True])

    # _concat_subsets should return the correct subset if the subsets are
    # bounded on a point.
    def test_concat_subsets_bounded_point(self):
        subset1 = self.numberset._extract_subset(-inf, 0, True, True, "01")
        subset2 = self.numberset._extract_subset(0, inf, True, True, "01")
        subset = self.numberset._concat_subsets([subset1, subset2], [0])
        self.assertEqual(subset._boundaries, self.numberset._boundaries)
        self.assertEqual(
            subset._boundaries_included, self.numberset._boundaries_included
        )

        subset1 = self.numberset._extract_subset(-inf, 0, True, True, "10")
        subset2 = self.numberset._extract_subset(0, inf, True, True, "10")
        subset = self.numberset._concat_subsets([subset1, subset2], [0])
        inv_numberset = ~self.numberset
        self.assertEqual(subset._boundaries, inv_numberset._boundaries)
        self.assertEqual(
            subset._boundaries_included, inv_numberset._boundaries_included
        )

    # _concat_subsets should return the correct subset if the subsets are
    # bounded on an included start bound.
    def test_concat_subsets_bounded_included_start_bound(self):
        subset1 = self.numberset._extract_subset(-inf, 1, True, True, "01")
        subset2 = self.numberset._extract_subset(1, inf, True, True, "01")
        subset = self.numberset._concat_subsets([subset1, subset2], [1])
        self.assertEqual(subset._boundaries, self.numberset._boundaries)
        self.assertEqual(
            subset._boundaries_included, self.numberset._boundaries_included
        )

        subset1 = self.numberset._extract_subset(-inf, 1, True, True, "10")
        subset2 = self.numberset._extract_subset(1, inf, True, True, "10")
        subset = self.numberset._concat_subsets([subset1, subset2], [1])
        inv_numberset = ~self.numberset
        self.assertEqual(subset._boundaries, inv_numberset._boundaries)
        self.assertEqual(
            subset._boundaries_included, inv_numberset._boundaries_included
        )

    # _concat_subsets should return the correct subset if the subsets are
    # bounded on an included end bound.
    def test_concat_subsets_bounded_included_end_bound(self):
        subset1 = self.numberset._extract_subset(-inf, 2, True, True, "01")
        subset2 = self.numberset._extract_subset(2, inf, True, True, "01")
        subset = self.numberset._concat_subsets([subset1, subset2], [2])
        self.assertEqual(subset._boundaries, self.numberset._boundaries)
        self.assertEqual(
            subset._boundaries_included, self.numberset._boundaries_included
        )

        subset1 = self.numberset._extract_subset(-inf, 2, True, True, "10")
        subset2 = self.numberset._extract_subset(2, inf, True, True, "10")
        subset = self.numberset._concat_subsets([subset1, subset2], [2])
        inv_numberset = ~self.numberset
        self.assertEqual(subset._boundaries, inv_numberset._boundaries)
        self.assertEqual(
            subset._boundaries_included, inv_numberset._boundaries_included
        )

    # _concat_subsets should return the correct subset if the subsets are
    # bounded in an interval.
    def test_concat_subsets_bounded_in_interval(self):
        subset1 = self.numberset._extract_subset(-inf, 4, True, True, "01")
        subset2 = self.numberset._extract_subset(4, inf, True, True, "01")
        subset = self.numberset._concat_subsets([subset1, subset2], [4])
        self.assertEqual(subset._boundaries, self.numberset._boundaries)
        self.assertEqual(
            subset._boundaries_included, self.numberset._boundaries_included
        )

        subset1 = self.numberset._extract_subset(-inf, 4, True, True, "10")
        subset2 = self.numberset._extract_subset(4, inf, True, True, "10")
        subset = self.numberset._concat_subsets([subset1, subset2], [4])
        inv_numberset = ~self.numberset
        self.assertEqual(subset._boundaries, inv_numberset._boundaries)
        self.assertEqual(
            subset._boundaries_included, inv_numberset._boundaries_included
        )

    # _concat_subsets should return the correct subset if the subsets are
    # bounded in a hole between intervals.
    def test_concat_subsets_bounded_in_hole(self):
        subset1 = self.numberset._extract_subset(-inf, 5, True, True, "01")
        subset2 = self.numberset._extract_subset(5, inf, True, True, "01")
        subset = self.numberset._concat_subsets([subset1, subset2], [5])
        self.assertEqual(subset._boundaries, self.numberset._boundaries)
        self.assertEqual(
            subset._boundaries_included, self.numberset._boundaries_included
        )

        subset1 = self.numberset._extract_subset(-inf, 5, True, True, "10")
        subset2 = self.numberset._extract_subset(5, inf, True, True, "10")
        subset = self.numberset._concat_subsets([subset1, subset2], [5])
        inv_numberset = ~self.numberset
        self.assertEqual(subset._boundaries, inv_numberset._boundaries)
        self.assertEqual(
            subset._boundaries_included, inv_numberset._boundaries_included
        )

    # _concat_subsets should return the correct subset if the subsets are
    # bounded on an excluded start bound.
    def test_concat_subsets_bounded_excluded_start_bound(self):
        subset1 = self.numberset._extract_subset(-inf, 3, True, True, "01")
        subset2 = self.numberset._extract_subset(3, inf, True, True, "01")
        subset = self.numberset._concat_subsets([subset1, subset2], [3])
        self.assertEqual(subset._boundaries, self.numberset._boundaries)
        self.assertEqual(
            subset._boundaries_included, self.numberset._boundaries_included
        )

        subset1 = self.numberset._extract_subset(-inf, 3, True, True, "10")
        subset2 = self.numberset._extract_subset(3, inf, True, True, "10")
        subset = self.numberset._concat_subsets([subset1, subset2], [3])
        inv_numberset = ~self.numberset
        self.assertEqual(subset._boundaries, inv_numberset._boundaries)
        self.assertEqual(
            subset._boundaries_included, inv_numberset._boundaries_included
        )

    # _concat_subsets should return the correct subset if the subsets are
    # bounded on an excluded end bound.
    def test_concat_subsets_bounded_excluded_end_bound(self):
        subset1 = self.numberset._extract_subset(-inf, 6, True, True, "01")
        subset2 = self.numberset._extract_subset(6, inf, True, True, "01")
        subset = self.numberset._concat_subsets([subset1, subset2], [6])
        self.assertEqual(subset._boundaries, self.numberset._boundaries)
        self.assertEqual(
            subset._boundaries_included, self.numberset._boundaries_included
        )

        subset1 = self.numberset._extract_subset(-inf, 6, True, True, "10")
        subset2 = self.numberset._extract_subset(6, inf, True, True, "10")
        subset = self.numberset._concat_subsets([subset1, subset2], [6])
        inv_numberset = ~self.numberset
        self.assertEqual(subset._boundaries, inv_numberset._boundaries)
        self.assertEqual(
            subset._boundaries_included, inv_numberset._boundaries_included
        )

    # _concat_subsets should return the correct subset if the subsets are
    # bounded outside all components.
    def test_concat_subsets_bounded_outside_components(self):
        subset1 = self.numberset._extract_subset(-inf, 7, True, True, "01")
        subset2 = self.numberset._extract_subset(7, inf, True, True, "01")
        subset = self.numberset._concat_subsets([subset1, subset2], [7])
        self.assertEqual(subset._boundaries, self.numberset._boundaries)
        self.assertEqual(
            subset._boundaries_included, self.numberset._boundaries_included
        )

        subset1 = self.numberset._extract_subset(-inf, 7, True, True, "10")
        subset2 = self.numberset._extract_subset(7, inf, True, True, "10")
        subset = self.numberset._concat_subsets([subset1, subset2], [7])
        inv_numberset = ~self.numberset
        self.assertEqual(subset._boundaries, inv_numberset._boundaries)
        self.assertEqual(
            subset._boundaries_included, inv_numberset._boundaries_included
        )

    # __and__ should return the correct set if the other set is empty.
    def test_and_other_empty(self):
        numberset = self.numberset & geometry.NumberSet()
        self.assertEqual(numberset._boundaries, [])
        self.assertEqual(numberset._boundaries_included, [])

    # __and__ should return the correct set if the other set is full.
    def test_and_other_full(self):
        interval = geometry.Interval("[", -inf, inf, "]")
        numberset = self.numberset & geometry.NumberSet(interval)
        self.assertEqual(numberset._boundaries, self.numberset._boundaries)
        self.assertEqual(
            numberset._boundaries_included, self.numberset._boundaries_included
        )

    # __and__ should return the correct set if the other set is a single
    # number.
    def test_and_other_number(self):
        for number in range(9):
            numberset = self.numberset & geometry.NumberSet(number)
            self.assertEqual(
                numberset._boundaries,
                [number, number] if number in self.numberset else [],
            )
            self.assertEqual(
                numberset._boundaries_included,
                [True, True] if number in self.numberset else [],
            )

    # __and__ should return the correct set if the other set's borders are
    # included.
    def test_and_other_borders_included(self):
        for start in range(9):
            interval = geometry.Interval("[", start, start + 1, "]")
            numberset_and = self.numberset & geometry.NumberSet(interval)
            numberset_subset = self.numberset._extract_subset(
                start, start + 1, True, True, "01"
            )
            self.assertEqual(
                numberset_and._boundaries, numberset_subset._boundaries
            )
            self.assertEqual(
                numberset_and._boundaries_included,
                numberset_subset._boundaries_included,
            )

    # __and__ should return the correct set if the other set's borders are
    # excluded.
    def test_and_other_borders_excluded(self):
        for start in range(9):
            interval = geometry.Interval("(", start, start + 1, ")")
            numberset_and = self.numberset & geometry.NumberSet(interval)
            numberset_subset = self.numberset._extract_subset(
                start, start + 1, False, False, "01"
            )
            self.assertEqual(
                numberset_and._boundaries, numberset_subset._boundaries
            )

    # __or__ should return the correct set if the other set is empty.
    def test_or_other_empty(self):
        numberset = self.numberset | geometry.NumberSet()
        self.assertEqual(numberset._boundaries, self.numberset._boundaries)
        self.assertEqual(
            numberset._boundaries_included, self.numberset._boundaries_included
        )

    # __or__ should return the correct set if the other set is full.
    def test_or_other_full(self):
        interval = geometry.Interval("[", -inf, inf, "]")
        numberset = self.numberset | geometry.NumberSet(interval)
        self.assertEqual(numberset._boundaries, [-inf, inf])
        self.assertEqual(numberset._boundaries_included, [True, True])

    # __or__ should return the correct set if the other set is a single
    # number.
    def test_or_other_number(self):
        for number in range(9):
            numberset_or = self.numberset | geometry.NumberSet(number)
            numberset_add = self.numberset.copy()
            numberset_add.add_number(number)
            self.assertEqual(
                numberset_or._boundaries, numberset_add._boundaries
            )
            self.assertEqual(
                numberset_or._boundaries_included,
                numberset_add._boundaries_included,
            )

    # __or__ should return the correct set if the other set's borders are
    # included.
    def test_or_other_borders_included(self):
        interval = geometry.Interval("[", 0, 1, "]")
        numberset_or = self.numberset | geometry.NumberSet(interval)
        self.assertEqual(numberset_or._boundaries, [0, 2, 3, 5, 5, 6, 8, inf])
        self.assertEqual(
            numberset_or._boundaries_included,
            [True, True, False, False, False, False, False, True],
        )
        interval = geometry.Interval("[", 1, 2, "]")
        numberset_or = self.numberset | geometry.NumberSet(interval)
        self.assertEqual(
            numberset_or._boundaries, [0, 0, 1, 2, 3, 5, 5, 6, 8, inf]
        )
        self.assertEqual(
            numberset_or._boundaries_included,
            [True, True, True, True, False, False, False, False, False, True],
        )
        interval = geometry.Interval("[", 2, 3, "]")
        numberset_or = self.numberset | geometry.NumberSet(interval)
        self.assertEqual(numberset_or._boundaries, [0, 0, 1, 5, 5, 6, 8, inf])
        self.assertEqual(
            numberset_or._boundaries_included,
            [True, True, True, False, False, False, False, True],
        )
        interval = geometry.Interval("[", 3, 4, "]")
        numberset_or = self.numberset | geometry.NumberSet(interval)
        self.assertEqual(
            numberset_or._boundaries, [0, 0, 1, 2, 3, 5, 5, 6, 8, inf]
        )
        self.assertEqual(
            numberset_or._boundaries_included,
            [True, True, True, True, True, False, False, False, False, True],
        )
        interval = geometry.Interval("[", 4, 5, "]")
        numberset_or = self.numberset | geometry.NumberSet(interval)
        self.assertEqual(numberset_or._boundaries, [0, 0, 1, 2, 3, 6, 8, inf])
        self.assertEqual(
            numberset_or._boundaries_included,
            [True, True, True, True, False, False, False, True],
        )
        interval = geometry.Interval("[", 5, 6, "]")
        numberset_or = self.numberset | geometry.NumberSet(interval)
        self.assertEqual(numberset_or._boundaries, [0, 0, 1, 2, 3, 6, 8, inf])
        self.assertEqual(
            numberset_or._boundaries_included,
            [True, True, True, True, False, True, False, True],
        )
        interval = geometry.Interval("[", 6, 7, "]")
        numberset_or = self.numberset | geometry.NumberSet(interval)
        self.assertEqual(
            numberset_or._boundaries, [0, 0, 1, 2, 3, 5, 5, 7, 8, inf]
        )
        self.assertEqual(
            numberset_or._boundaries_included,
            [True, True, True, True, False, False, False, True, False, True],
        )
        interval = geometry.Interval("[", 7, 8, "]")
        numberset_or = self.numberset | geometry.NumberSet(interval)
        self.assertEqual(
            numberset_or._boundaries, [0, 0, 1, 2, 3, 5, 5, 6, 7, inf]
        )
        self.assertEqual(
            numberset_or._boundaries_included,
            [True, True, True, True, False, False, False, False, True, True],
        )
        interval = geometry.Interval("[", 8, 9, "]")
        numberset_or = self.numberset | geometry.NumberSet(interval)
        self.assertEqual(
            numberset_or._boundaries, [0, 0, 1, 2, 3, 5, 5, 6, 8, inf]
        )
        self.assertEqual(
            numberset_or._boundaries_included,
            [True, True, True, True, False, False, False, False, True, True],
        )

    # __or__ should return the correct set if the other set's borders are
    # excluded.
    def test_or_other_borders_excluded(self):
        interval = geometry.Interval("(", 0, 1, ")")
        numberset_or = self.numberset | geometry.NumberSet(interval)
        self.assertEqual(numberset_or._boundaries, [0, 2, 3, 5, 5, 6, 8, inf])
        self.assertEqual(
            numberset_or._boundaries_included,
            [True, True, False, False, False, False, False, True],
        )
        interval = geometry.Interval("(", 1, 2, ")")
        numberset_or = self.numberset | geometry.NumberSet(interval)
        self.assertEqual(
            numberset_or._boundaries, [0, 0, 1, 2, 3, 5, 5, 6, 8, inf]
        )
        self.assertEqual(
            numberset_or._boundaries_included,
            [True, True, True, True, False, False, False, False, False, True],
        )
        interval = geometry.Interval("(", 2, 3, ")")
        numberset_or = self.numberset | geometry.NumberSet(interval)
        self.assertEqual(
            numberset_or._boundaries, [0, 0, 1, 3, 3, 5, 5, 6, 8, inf]
        )
        self.assertEqual(
            numberset_or._boundaries_included,
            [True, True, True, False, False, False, False, False, False, True],
        )
        interval = geometry.Interval("(", 3, 4, ")")
        numberset_or = self.numberset | geometry.NumberSet(interval)
        self.assertEqual(
            numberset_or._boundaries, [0, 0, 1, 2, 3, 5, 5, 6, 8, inf]
        )
        self.assertEqual(
            numberset_or._boundaries_included,
            [True, True, True, True, False, False, False, False, False, True],
        )
        interval = geometry.Interval("(", 4, 5, ")")
        numberset_or = self.numberset | geometry.NumberSet(interval)
        self.assertEqual(
            numberset_or._boundaries, [0, 0, 1, 2, 3, 5, 5, 6, 8, inf]
        )
        self.assertEqual(
            numberset_or._boundaries_included,
            [True, True, True, True, False, False, False, False, False, True],
        )
        interval = geometry.Interval("(", 5, 6, ")")
        numberset_or = self.numberset | geometry.NumberSet(interval)
        self.assertEqual(
            numberset_or._boundaries, [0, 0, 1, 2, 3, 5, 5, 6, 8, inf]
        )
        self.assertEqual(
            numberset_or._boundaries_included,
            [True, True, True, True, False, False, False, False, False, True],
        )
        interval = geometry.Interval("(", 6, 7, ")")
        numberset_or = self.numberset | geometry.NumberSet(interval)
        self.assertEqual(
            numberset_or._boundaries, [0, 0, 1, 2, 3, 5, 5, 6, 6, 7, 8, inf]
        )
        self.assertEqual(
            numberset_or._boundaries_included,
            [
                True,
                True,
                True,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                True,
            ],
        )
        interval = geometry.Interval("(", 7, 8, ")")
        numberset_or = self.numberset | geometry.NumberSet(interval)
        self.assertEqual(
            numberset_or._boundaries, [0, 0, 1, 2, 3, 5, 5, 6, 7, 8, 8, inf]
        )
        self.assertEqual(
            numberset_or._boundaries_included,
            [
                True,
                True,
                True,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                True,
            ],
        )
        interval = geometry.Interval("(", 8, 9, ")")
        numberset_or = self.numberset | geometry.NumberSet(interval)
        self.assertEqual(
            numberset_or._boundaries, [0, 0, 1, 2, 3, 5, 5, 6, 8, inf]
        )
        self.assertEqual(
            numberset_or._boundaries_included,
            [True, True, True, True, False, False, False, False, False, True],
        )

    # __xor__ should return the correct set if the other set is empty.
    def test_xor_other_empty(self):
        numberset = self.numberset ^ geometry.NumberSet()
        self.assertEqual(numberset._boundaries, self.numberset._boundaries)
        self.assertEqual(
            numberset._boundaries_included, self.numberset._boundaries_included
        )

    # __xor__ should return the correct set if the other set is full.
    def test_xor_other_full(self):
        interval = geometry.Interval("[", -inf, inf, "]")
        numberset = self.numberset ^ geometry.NumberSet(interval)
        inv_numberset = ~self.numberset
        self.assertEqual(numberset._boundaries, inv_numberset._boundaries)
        self.assertEqual(
            numberset._boundaries_included, inv_numberset._boundaries_included
        )

    # __xor__ should return the correct set if the other set is a single
    # number.
    def test_xor_other_number(self):
        for number in range(9):
            numberset_number = geometry.NumberSet(number)
            numberset_xor = self.numberset ^ numberset_number
            if number in self.numberset:
                numberset_verify = self.numberset.copy()
                numberset_verify.remove_number(number)
            else:
                numberset_verify = self.numberset.copy()
                numberset_verify.add_number(number)
            self.assertEqual(
                numberset_xor._boundaries, numberset_verify._boundaries
            )
            self.assertEqual(
                numberset_xor._boundaries_included,
                numberset_verify._boundaries_included,
            )

    # __xor__ should return the correct set if the other set's borders are
    # included.
    def test_xor_other_borders_included(self):
        interval = geometry.Interval("[", 0, 1, "]")
        numberset_or = self.numberset ^ geometry.NumberSet(interval)
        self.assertEqual(
            numberset_or._boundaries, [0, 1, 1, 2, 3, 5, 5, 6, 8, inf]
        )
        self.assertEqual(
            numberset_or._boundaries_included,
            [
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                True,
            ],
        )
        interval = geometry.Interval("[", 1, 2, "]")
        numberset_or = self.numberset ^ geometry.NumberSet(interval)
        self.assertEqual(numberset_or._boundaries, [0, 0, 3, 5, 5, 6, 8, inf])
        self.assertEqual(
            numberset_or._boundaries_included,
            [True, True, False, False, False, False, False, True],
        )
        interval = geometry.Interval("[", 2, 3, "]")
        numberset_or = self.numberset ^ geometry.NumberSet(interval)
        self.assertEqual(
            numberset_or._boundaries, [0, 0, 1, 2, 2, 5, 5, 6, 8, inf]
        )
        self.assertEqual(
            numberset_or._boundaries_included,
            [True, True, True, False, False, False, False, False, False, True],
        )
        interval = geometry.Interval("[", 3, 4, "]")
        numberset_or = self.numberset ^ geometry.NumberSet(interval)
        self.assertEqual(
            numberset_or._boundaries, [0, 0, 1, 2, 3, 3, 4, 5, 5, 6, 8, inf]
        )
        self.assertEqual(
            numberset_or._boundaries_included,
            [
                True,
                True,
                True,
                True,
                True,
                True,
                False,
                False,
                False,
                False,
                False,
                True,
            ],
        )
        interval = geometry.Interval("[", 4, 5, "]")
        numberset_or = self.numberset ^ geometry.NumberSet(interval)
        self.assertEqual(
            numberset_or._boundaries, [0, 0, 1, 2, 3, 4, 5, 6, 8, inf]
        )
        self.assertEqual(
            numberset_or._boundaries_included,
            [True, True, True, True, False, False, True, False, False, True],
        )
        interval = geometry.Interval("[", 5, 6, "]")
        numberset_or = self.numberset ^ geometry.NumberSet(interval)
        self.assertEqual(
            numberset_or._boundaries, [0, 0, 1, 2, 3, 5, 6, 6, 8, inf]
        )
        self.assertEqual(
            numberset_or._boundaries_included,
            [True, True, True, True, False, True, True, True, False, True],
        )
        interval = geometry.Interval("[", 6, 7, "]")
        numberset_or = self.numberset ^ geometry.NumberSet(interval)
        self.assertEqual(
            numberset_or._boundaries, [0, 0, 1, 2, 3, 5, 5, 7, 8, inf]
        )
        self.assertEqual(
            numberset_or._boundaries_included,
            [True, True, True, True, False, False, False, True, False, True],
        )
        interval = geometry.Interval("[", 7, 8, "]")
        numberset_or = self.numberset ^ geometry.NumberSet(interval)
        self.assertEqual(
            numberset_or._boundaries, [0, 0, 1, 2, 3, 5, 5, 6, 7, inf]
        )
        self.assertEqual(
            numberset_or._boundaries_included,
            [True, True, True, True, False, False, False, False, True, True],
        )
        interval = geometry.Interval("[", 8, 9, "]")
        numberset_or = self.numberset ^ geometry.NumberSet(interval)
        self.assertEqual(
            numberset_or._boundaries, [0, 0, 1, 2, 3, 5, 5, 6, 8, 8, 9, inf]
        )
        self.assertEqual(
            numberset_or._boundaries_included,
            [
                True,
                True,
                True,
                True,
                False,
                False,
                False,
                False,
                True,
                True,
                False,
                True,
            ],
        )

    # __xor__ should return the correct set if the other set's borders are
    # excluded.
    def test_xor_other_borders_excluded(self):
        interval = geometry.Interval("(", 0, 1, ")")
        numberset_or = self.numberset ^ geometry.NumberSet(interval)
        self.assertEqual(numberset_or._boundaries, [0, 2, 3, 5, 5, 6, 8, inf])
        self.assertEqual(
            numberset_or._boundaries_included,
            [True, True, False, False, False, False, False, True],
        )
        interval = geometry.Interval("(", 1, 2, ")")
        numberset_or = self.numberset ^ geometry.NumberSet(interval)
        self.assertEqual(
            numberset_or._boundaries, [0, 0, 1, 1, 2, 2, 3, 5, 5, 6, 8, inf]
        )
        self.assertEqual(
            numberset_or._boundaries_included,
            [
                True,
                True,
                True,
                True,
                True,
                True,
                False,
                False,
                False,
                False,
                False,
                True,
            ],
        )
        interval = geometry.Interval("(", 2, 3, ")")
        numberset_or = self.numberset ^ geometry.NumberSet(interval)
        self.assertEqual(
            numberset_or._boundaries, [0, 0, 1, 3, 3, 5, 5, 6, 8, inf]
        )
        self.assertEqual(
            numberset_or._boundaries_included,
            [True, True, True, False, False, False, False, False, False, True],
        )
        interval = geometry.Interval("(", 3, 4, ")")
        numberset_or = self.numberset ^ geometry.NumberSet(interval)
        self.assertEqual(
            numberset_or._boundaries, [0, 0, 1, 2, 4, 5, 5, 6, 8, inf]
        )
        self.assertEqual(
            numberset_or._boundaries_included,
            [True, True, True, True, True, False, False, False, False, True],
        )
        interval = geometry.Interval("(", 4, 5, ")")
        numberset_or = self.numberset ^ geometry.NumberSet(interval)
        self.assertEqual(
            numberset_or._boundaries, [0, 0, 1, 2, 3, 4, 5, 6, 8, inf]
        )
        self.assertEqual(
            numberset_or._boundaries_included,
            [True, True, True, True, False, True, False, False, False, True],
        )
        interval = geometry.Interval("(", 5, 6, ")")
        numberset_or = self.numberset ^ geometry.NumberSet(interval)
        self.assertEqual(numberset_or._boundaries, [0, 0, 1, 2, 3, 5, 8, inf])
        self.assertEqual(
            numberset_or._boundaries_included,
            [True, True, True, True, False, False, False, True],
        )
        interval = geometry.Interval("(", 6, 7, ")")
        numberset_or = self.numberset ^ geometry.NumberSet(interval)
        self.assertEqual(
            numberset_or._boundaries, [0, 0, 1, 2, 3, 5, 5, 6, 6, 7, 8, inf]
        )
        self.assertEqual(
            numberset_or._boundaries_included,
            [
                True,
                True,
                True,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                True,
            ],
        )
        interval = geometry.Interval("(", 7, 8, ")")
        numberset_or = self.numberset ^ geometry.NumberSet(interval)
        self.assertEqual(
            numberset_or._boundaries, [0, 0, 1, 2, 3, 5, 5, 6, 7, 8, 8, inf]
        )
        self.assertEqual(
            numberset_or._boundaries_included,
            [
                True,
                True,
                True,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                True,
            ],
        )
        interval = geometry.Interval("(", 8, 9, ")")
        numberset_or = self.numberset ^ geometry.NumberSet(interval)
        self.assertEqual(
            numberset_or._boundaries, [0, 0, 1, 2, 3, 5, 5, 6, 9, inf]
        )
        self.assertEqual(
            numberset_or._boundaries_included,
            [True, True, True, True, False, False, False, False, True, True],
        )

    # __init__ with no arguments.
    def test_init_empty(self):
        numberset = geometry.NumberSet()
        self.assertEqual(numberset._boundaries, [])
        self.assertEqual(numberset._boundaries_included, [])

    # __init__ with a single number.
    def test_init_number(self):
        number_set = geometry.NumberSet(5)
        self.assertEqual(number_set._boundaries, [5, 5])
        self.assertEqual(number_set._boundaries_included, [True, True])

    # __init__ should raise a ValueError if it is initialized with a single
    # number that is -inf or inf.
    def test_init_number_inf(self):
        with self.assertRaises(ValueError):
            geometry.NumberSet(-inf)
        with self.assertRaises(ValueError):
            geometry.NumberSet(inf)

    # __init__ with a single interval.
    def test_init_interval(self):
        interval = geometry.Interval("[", 1, 5, "]")
        numberset = geometry.NumberSet(interval)
        self.assertEqual(numberset._boundaries, [1, 5])
        self.assertEqual(numberset._boundaries_included, [True, True])

    # __init__ with a single NumberSet instance.
    def test_init_numberset(self):
        interval = geometry.Interval("[", -3, 1, ")")
        numberset1 = geometry.NumberSet(interval)
        numberset2 = geometry.NumberSet(numberset1)
        self.assertEqual(numberset2._boundaries, [-3, 1])
        self.assertEqual(numberset2._boundaries_included, [True, False])

    # __init__ with multiple non-overlapping numbers.
    def test_init_numbers(self):
        numberset = geometry.NumberSet(1, 2, 3, 4, 5)
        self.assertEqual(numberset._boundaries, [1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
        self.assertEqual(
            numberset._boundaries_included,
            [True, True, True, True, True, True, True, True, True, True],
        )

    # __init__ with multiple non-overlapping intervals.
    def test_init_intervals(self):
        interval1 = geometry.Interval("[", 1, 5, ")")
        interval2 = geometry.Interval("(", 10, 15, "]")
        interval3 = geometry.Interval("[", 20, 25, "]")
        numberset = geometry.NumberSet(interval1, interval2, interval3)
        self.assertEqual(numberset._boundaries, [1, 5, 10, 15, 20, 25])
        self.assertEqual(
            numberset._boundaries_included,
            [True, False, False, True, True, True],
        )

    # __init__ with multiple non-overlapping numbersets.
    def test_init_numbersets(self):
        interval1 = geometry.Interval("[", 1, 5, ")")
        numberset1 = geometry.NumberSet(0, interval1)
        interval2 = geometry.Interval("(", 10, 15, "]")
        numberset2 = geometry.NumberSet(6, 7, interval2)
        numberset = geometry.NumberSet(numberset1, numberset2)
        self.assertEqual(
            numberset._boundaries, [0, 0, 1, 5, 6, 6, 7, 7, 10, 15]
        )
        self.assertEqual(
            numberset._boundaries_included,
            [True, True, True, False, True, True, True, True, False, True],
        )

    # __init__ with multiple non-overlapping numbers, intervals, and
    # numbersets.
    def test_init_numbers_intervals_numbersets(self):
        interval = geometry.Interval("(", -3, 1, ")")
        interval1 = geometry.Interval("[", 6, 8, ")")
        numberset1 = geometry.NumberSet(5, interval1)
        numberset = geometry.NumberSet(-4, interval, numberset1)
        self.assertEqual(numberset._boundaries, [-4, -4, -3, 1, 5, 5, 6, 8])
        self.assertEqual(
            numberset._boundaries_included,
            [True, True, False, False, True, True, True, False],
        )

    # __init__ with multiple overlapping numbers.
    def test_init_numbers_overlapping(self):
        numberset = geometry.NumberSet(1, 2, 3, 4, 5, 1, 3, 4, 4)
        self.assertEqual(numberset._boundaries, [1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
        self.assertEqual(
            numberset._boundaries_included,
            [True, True, True, True, True, True, True, True, True, True],
        )

    # __init__ with multiple overlapping intervals.
    def test_init_intervals_overlapping(self):
        interval1 = geometry.Interval("[", 1, 5, "]")
        interval2 = geometry.Interval("(", 4, 10, ")")
        interval3 = geometry.Interval("[", 10, 12, ")")
        interval4 = geometry.Interval("(", 6, 8, "]")
        numberset = geometry.NumberSet(
            interval1, interval2, interval3, interval4
        )
        self.assertEqual(numberset._boundaries, [1, 12])
        self.assertEqual(numberset._boundaries_included, [True, False])

    # __init__ with multiple overlapping numbersets.
    def test_init_numbersets_overlapping(self):
        interval1 = geometry.Interval("[", 1, 5, "]")
        numberset1 = geometry.NumberSet(0, interval1)
        interval2 = geometry.Interval("(", 4, 10, ")")
        numberset2 = geometry.NumberSet(6, 7, interval2)
        numberset = geometry.NumberSet(numberset1, numberset2)
        self.assertEqual(numberset._boundaries, [0, 0, 1, 10])
        self.assertEqual(
            numberset._boundaries_included, [True, True, True, False]
        )

    # is_empty should return True iff the NumberSet is empty.
    def test_is_empty(self):
        numberset = geometry.NumberSet()
        self.assertTrue(numberset.is_empty())

        numberset = geometry.NumberSet(3)
        self.assertFalse(numberset.is_empty())

        numberset = geometry.NumberSet(geometry.Interval("[", 3, 4, "]"))
        self.assertFalse(numberset.is_empty())

    # is_number should return True iff the NumberSet contains a single number.
    def test_is_number(self):
        numberset = geometry.NumberSet()
        self.assertFalse(numberset.is_number())

        numberset = geometry.NumberSet(3)
        self.assertTrue(numberset.is_number())

        numberset = geometry.NumberSet(geometry.Interval("[", 3, 4, "]"))
        self.assertFalse(numberset.is_number())

    # is_interval should return True iff the NumberSet contains a single
    # interval.
    def test_is_interval(self):
        numberset = geometry.NumberSet()
        self.assertFalse(numberset.is_interval())

        numberset = geometry.NumberSet(3)
        self.assertFalse(numberset.is_interval())

        numberset = geometry.NumberSet(geometry.Interval("[", 3, 4, "]"))
        self.assertTrue(numberset.is_interval())

        numberset = geometry.NumberSet(
            geometry.Interval("[", 3, 4, "]"),
            geometry.Interval("[", 5, 6, "]"),
        )
        self.assertFalse(numberset.is_interval())

    # is_reducible should return True iff the NumberSet is reducible.
    def test_is_reducible(self):
        numberset = geometry.NumberSet()
        self.assertTrue(numberset.is_reducible())

        numberset = geometry.NumberSet(3)
        self.assertTrue(numberset.is_reducible())

        numberset = geometry.NumberSet(geometry.Interval("[", 3, 4, "]"))
        self.assertTrue(numberset.is_reducible())

        numberset = geometry.NumberSet(
            geometry.Interval("[", 3, 4, "]"),
            geometry.Interval("[", 5, 6, "]"),
        )
        self.assertFalse(numberset.is_reducible())

        numberset = geometry.NumberSet(
            geometry.Interval("[", 3, 4, "]"),
            geometry.Interval("[", 4, 5, "]"),
        )
        self.assertTrue(numberset.is_reducible())

    # reduce should return the correct object if the NumberSet is reducible.
    def test_reduce(self):
        numberset = geometry.NumberSet()
        self.assertEqual(numberset.reduce(), None)

        numberset = geometry.NumberSet(3)
        self.assertEqual(numberset.reduce(), 3)

        numberset = geometry.NumberSet(geometry.Interval("[", 3, 4, "]"))
        self.assertEqual(numberset.reduce(), geometry.Interval("[", 3, 4, "]"))

        numberset = geometry.NumberSet(
            geometry.Interval("[", 3, 4, "]"),
            geometry.Interval("[", 5, 6, "]"),
        )
        self.assertEqual(numberset.reduce(), numberset)

        numberset = geometry.NumberSet(
            geometry.Interval("[", 3, 4, "]"),
            geometry.Interval("[", 4, 5, "]"),
        )
        self.assertEqual(numberset.reduce(), geometry.Interval("[", 3, 5, "]"))

    # is_overlapping should return True iff the NumberSet overlaps with another
    # NumberSet.
    def test_is_overlapping(self):
        numberset1 = geometry.NumberSet()
        numberset2 = geometry.NumberSet()
        self.assertFalse(numberset1.is_overlapping(numberset2))

        numberset1 = geometry.NumberSet(3)
        numberset2 = geometry.NumberSet()
        self.assertFalse(numberset1.is_overlapping(numberset2))

        numberset1 = geometry.NumberSet(3)
        numberset2 = geometry.NumberSet(4)
        self.assertFalse(numberset1.is_overlapping(numberset2))

        numberset1 = geometry.NumberSet(3)
        numberset2 = geometry.NumberSet(3)
        self.assertTrue(numberset1.is_overlapping(numberset2))

        numberset1 = geometry.NumberSet(3)
        numberset2 = geometry.NumberSet(geometry.Interval("[", 3, 4, "]"))
        self.assertTrue(numberset1.is_overlapping(numberset2))

        numberset1 = geometry.NumberSet(3)
        numberset2 = geometry.NumberSet(geometry.Interval("[", 4, 5, "]"))
        self.assertFalse(numberset1.is_overlapping(numberset2))

        numberset1 = geometry.NumberSet(geometry.Interval("[", 3, 4, "]"))
        numberset2 = geometry.NumberSet(geometry.Interval("[", 4, 5, "]"))
        self.assertTrue(numberset1.is_overlapping(numberset2))

        numberset1 = geometry.NumberSet(geometry.Interval("[", 3, 4, "]"))
        numberset2 = geometry.NumberSet(geometry.Interval("(", 4, 5, "]"))
        self.assertFalse(numberset1.is_overlapping(numberset2))

    # is_disjoint should return True iff the NumberSet is disjoint with another
    # NumberSet.
    def test_is_disjoint(self):
        numberset1 = geometry.NumberSet()
        numberset2 = geometry.NumberSet()
        self.assertTrue(numberset1.is_disjoint(numberset2))

        numberset1 = geometry.NumberSet(3)
        numberset2 = geometry.NumberSet()
        self.assertTrue(numberset1.is_disjoint(numberset2))

        numberset1 = geometry.NumberSet(3)
        numberset2 = geometry.NumberSet(4)
        self.assertTrue(numberset1.is_disjoint(numberset2))

        numberset1 = geometry.NumberSet(3)
        numberset2 = geometry.NumberSet(3)
        self.assertFalse(numberset1.is_disjoint(numberset2))

        numberset1 = geometry.NumberSet(3)
        numberset2 = geometry.NumberSet(geometry.Interval("[", 3, 4, "]"))
        self.assertFalse(numberset1.is_disjoint(numberset2))

        numberset1 = geometry.NumberSet(3)
        numberset2 = geometry.NumberSet(geometry.Interval("[", 4, 5, "]"))
        self.assertTrue(numberset1.is_disjoint(numberset2))

        numberset1 = geometry.NumberSet(geometry.Interval("[", 3, 4, "]"))
        numberset2 = geometry.NumberSet(geometry.Interval("[", 4, 5, "]"))
        self.assertFalse(numberset1.is_disjoint(numberset2))

        numberset1 = geometry.NumberSet(geometry.Interval("[", 3, 4, "]"))
        numberset2 = geometry.NumberSet(geometry.Interval("(", 4, 5, "]"))
        self.assertTrue(numberset1.is_disjoint(numberset2))

    # is_subset should return True iff the NumberSet is a subset of another
    # NumberSet.
    def test_is_subset(self):
        numberset1 = geometry.NumberSet()
        numberset2 = geometry.NumberSet()
        self.assertTrue(numberset1.is_subset(numberset2))

        numberset1 = geometry.NumberSet(3)
        numberset2 = geometry.NumberSet()
        self.assertFalse(numberset1.is_subset(numberset2))

        numberset1 = geometry.NumberSet(3)
        numberset2 = geometry.NumberSet(4)
        self.assertFalse(numberset1.is_subset(numberset2))

        numberset1 = geometry.NumberSet(3)
        numberset2 = geometry.NumberSet(3)
        self.assertTrue(numberset1.is_subset(numberset2))

        numberset1 = geometry.NumberSet(3)
        numberset2 = geometry.NumberSet(geometry.Interval("[", 3, 4, "]"))
        self.assertTrue(numberset1.is_subset(numberset2))

        numberset1 = geometry.NumberSet(3)
        numberset2 = geometry.NumberSet(geometry.Interval("[", 4, 5, "]"))
        self.assertFalse(numberset1.is_subset(numberset2))

        numberset1 = geometry.NumberSet(geometry.Interval("[", 3, 4, "]"))
        numberset2 = geometry.NumberSet(geometry.Interval("[", 4, 5, "]"))
        self.assertFalse(numberset1.is_subset(numberset2))

        numberset1 = geometry.NumberSet(geometry.Interval("[", 3, 4, "]"))
        numberset2 = geometry.NumberSet(geometry.Interval("[", 3, 5, "]"))
        self.assertTrue(numberset1.is_subset(numberset2))

        numberset1 = geometry.NumberSet(geometry.Interval("[", 3, 4, "]"))
        numberset2 = geometry.NumberSet(geometry.Interval("(", 3, 5, "]"))
        self.assertFalse(numberset1.is_subset(numberset2))

    # is_superset should return True iff the NumberSet is a superset of another
    # NumberSet.
    def test_is_superset(self):
        numberset1 = geometry.NumberSet()
        numberset2 = geometry.NumberSet()
        self.assertTrue(numberset1.is_superset(numberset2))

        numberset1 = geometry.NumberSet(3)
        numberset2 = geometry.NumberSet()
        self.assertTrue(numberset1.is_superset(numberset2))

        numberset1 = geometry.NumberSet(3)
        numberset2 = geometry.NumberSet(4)
        self.assertFalse(numberset1.is_superset(numberset2))

        numberset1 = geometry.NumberSet(3)
        numberset2 = geometry.NumberSet(3)
        self.assertTrue(numberset1.is_superset(numberset2))

        numberset1 = geometry.NumberSet(3)
        numberset2 = geometry.NumberSet(geometry.Interval("[", 3, 4, "]"))
        self.assertFalse(numberset1.is_superset(numberset2))

        numberset1 = geometry.NumberSet(3)
        numberset2 = geometry.NumberSet(geometry.Interval("[", 4, 5, "]"))
        self.assertFalse(numberset1.is_superset(numberset2))

        numberset1 = geometry.NumberSet(geometry.Interval("[", 3, 4, "]"))
        numberset2 = geometry.NumberSet(geometry.Interval("[", 4, 5, "]"))
        self.assertFalse(numberset1.is_superset(numberset2))

        numberset1 = geometry.NumberSet(geometry.Interval("[", 3, 5, "]"))
        numberset2 = geometry.NumberSet(geometry.Interval("[", 4, 5, "]"))
        self.assertTrue(numberset1.is_superset(numberset2))

        numberset1 = geometry.NumberSet(geometry.Interval("[", 3, 5, ")"))
        numberset2 = geometry.NumberSet(geometry.Interval("(", 4, 5, "]"))
        self.assertFalse(numberset1.is_superset(numberset2))

    # is_adjacent should return True iff the NumberSet is adjacent to another
    # NumberSet.
    def test_is_adjacent(self):
        numberset1 = geometry.NumberSet()
        numberset2 = geometry.NumberSet()
        self.assertFalse(numberset1.is_adjacent(numberset2))

        numberset1 = geometry.NumberSet(3)
        numberset2 = geometry.NumberSet()
        self.assertFalse(numberset1.is_adjacent(numberset2))

        numberset1 = geometry.NumberSet(3)
        numberset2 = geometry.NumberSet(4)
        self.assertFalse(numberset1.is_adjacent(numberset2))

        numberset1 = geometry.NumberSet(3)
        numberset2 = geometry.NumberSet(3)
        self.assertFalse(numberset1.is_adjacent(numberset2))

        numberset1 = geometry.NumberSet(3)
        numberset2 = geometry.NumberSet(geometry.Interval("(", 3, 4, "]"))
        self.assertTrue(numberset1.is_adjacent(numberset2))

        numberset1 = geometry.NumberSet(3)
        numberset2 = geometry.NumberSet(geometry.Interval("[", 3, 5, "]"))
        self.assertFalse(numberset1.is_adjacent(numberset2))

        numberset1 = geometry.NumberSet(geometry.Interval("[", 3, 4, "]"))
        numberset2 = geometry.NumberSet(geometry.Interval("[", 4, 5, "]"))
        self.assertFalse(numberset1.is_adjacent(numberset2))

        numberset1 = geometry.NumberSet(geometry.Interval("[", 3, 4, "]"))
        numberset2 = geometry.NumberSet(geometry.Interval("(", 4, 5, "]"))
        self.assertTrue(numberset1.is_adjacent(numberset2))

        numberset1 = geometry.NumberSet(geometry.Interval("[", 3, 4, ")"))
        numberset2 = geometry.NumberSet(geometry.Interval("(", 4, 5, "]"))
        self.assertFalse(numberset1.is_adjacent(numberset2))

    # starts_equal should return True iff the NumberSet starts with the same
    # number as another NumberSet.
    def test_starts_equal(self):
        numberset1 = geometry.NumberSet()
        numberset2 = geometry.NumberSet()
        self.assertFalse(numberset1.starts_equal(numberset2))

        numberset1 = geometry.NumberSet(3)
        numberset2 = geometry.NumberSet()
        self.assertFalse(numberset1.starts_equal(numberset2))

        numberset1 = geometry.NumberSet(3)
        numberset2 = geometry.NumberSet(4)
        self.assertFalse(numberset1.starts_equal(numberset2))

        numberset1 = geometry.NumberSet(3)
        numberset2 = geometry.NumberSet(3)
        self.assertTrue(numberset1.starts_equal(numberset2))

        numberset1 = geometry.NumberSet(3)
        numberset2 = geometry.NumberSet(geometry.Interval("[", 3, 4, "]"))
        self.assertTrue(numberset1.starts_equal(numberset2))

        numberset1 = geometry.NumberSet(3)
        numberset2 = geometry.NumberSet(geometry.Interval("[", 4, 5, "]"))
        self.assertFalse(numberset1.starts_equal(numberset2))

        numberset1 = geometry.NumberSet(geometry.Interval("[", 3, 4, "]"))
        numberset2 = geometry.NumberSet(geometry.Interval("[", 4, 5, "]"))
        self.assertFalse(numberset1.starts_equal(numberset2))

        numberset1 = geometry.NumberSet(geometry.Interval("[", 3, 5, "]"))
        numberset2 = geometry.NumberSet(geometry.Interval("[", 3, 5, "]"))
        self.assertTrue(numberset1.starts_equal(numberset2))

        numberset1 = geometry.NumberSet(geometry.Interval("[", 3, 5, ")"))
        numberset2 = geometry.NumberSet(geometry.Interval("(", 3, 5, "]"))
        self.assertFalse(numberset1.starts_equal(numberset2))

    # ends_equal should return True iff the NumberSet ends with the same
    # number as another NumberSet.
    def test_ends_equal(self):
        numberset1 = geometry.NumberSet()
        numberset2 = geometry.NumberSet()
        self.assertFalse(numberset1.ends_equal(numberset2))

        numberset1 = geometry.NumberSet(3)
        numberset2 = geometry.NumberSet()
        self.assertFalse(numberset1.ends_equal(numberset2))

        numberset1 = geometry.NumberSet(3)
        numberset2 = geometry.NumberSet(4)
        self.assertFalse(numberset1.ends_equal(numberset2))

        numberset1 = geometry.NumberSet(3)
        numberset2 = geometry.NumberSet(3)
        self.assertTrue(numberset1.ends_equal(numberset2))

        numberset1 = geometry.NumberSet(3)
        numberset2 = geometry.NumberSet(geometry.Interval("[", 3, 4, "]"))
        self.assertFalse(numberset1.ends_equal(numberset2))

        numberset1 = geometry.NumberSet(3)
        numberset2 = geometry.NumberSet(geometry.Interval("[", 4, 5, "]"))
        self.assertFalse(numberset1.ends_equal(numberset2))

        numberset1 = geometry.NumberSet(geometry.Interval("[", 3, 4, "]"))
        numberset2 = geometry.NumberSet(geometry.Interval("[", 4, 5, "]"))
        self.assertFalse(numberset1.ends_equal(numberset2))

        numberset1 = geometry.NumberSet(geometry.Interval("[", 3, 5, "]"))
        numberset2 = geometry.NumberSet(geometry.Interval("[", 4, 5, "]"))
        self.assertTrue(numberset1.ends_equal(numberset2))

        numberset1 = geometry.NumberSet(geometry.Interval("[", 3, 5, ")"))
        numberset2 = geometry.NumberSet(geometry.Interval("(", 4, 5, "]"))
        self.assertFalse(numberset1.ends_equal(numberset2))

    # starts_left should return True iff the NumberSet starts to the left of
    # another NumberSet.
    def test_starts_left(self):
        numberset1 = geometry.NumberSet()
        numberset2 = geometry.NumberSet()
        self.assertFalse(numberset1.starts_left(numberset2))

        numberset1 = geometry.NumberSet(3)
        numberset2 = geometry.NumberSet()
        self.assertFalse(numberset1.starts_left(numberset2))

        numberset1 = geometry.NumberSet(3)
        numberset2 = geometry.NumberSet(4)
        self.assertTrue(numberset1.starts_left(numberset2))

        numberset1 = geometry.NumberSet(3)
        numberset2 = geometry.NumberSet(3)
        self.assertFalse(numberset1.starts_left(numberset2))

        numberset1 = geometry.NumberSet(3)
        numberset2 = geometry.NumberSet(geometry.Interval("(", 3, 4, "]"))
        self.assertTrue(numberset1.starts_left(numberset2))

        numberset1 = geometry.NumberSet(3)
        numberset2 = geometry.NumberSet(geometry.Interval("[", 4, 5, "]"))
        self.assertTrue(numberset1.starts_left(numberset2))

        numberset1 = geometry.NumberSet(geometry.Interval("[", 3, 4, "]"))
        numberset2 = geometry.NumberSet(geometry.Interval("[", 4, 5, "]"))
        self.assertTrue(numberset1.starts_left(numberset2))

        numberset1 = geometry.NumberSet(geometry.Interval("[", 3, 5, "]"))
        numberset2 = geometry.NumberSet(geometry.Interval("(", 3, 5, "]"))
        self.assertTrue(numberset1.starts_left(numberset2))

        numberset1 = geometry.NumberSet(geometry.Interval("[", 3, 5, ")"))
        numberset2 = geometry.NumberSet(geometry.Interval("[", 3, 5, "]"))
        self.assertFalse(numberset1.starts_left(numberset2))

    # starts_right should return True iff the NumberSet starts to the right of
    # another NumberSet.
    def test_starts_right(self):
        numberset1 = geometry.NumberSet()
        numberset2 = geometry.NumberSet()
        self.assertFalse(numberset1.starts_right(numberset2))

        numberset1 = geometry.NumberSet(3)
        numberset2 = geometry.NumberSet()
        self.assertFalse(numberset1.starts_right(numberset2))

        numberset1 = geometry.NumberSet(3)
        numberset2 = geometry.NumberSet(4)
        self.assertFalse(numberset1.starts_right(numberset2))

        numberset1 = geometry.NumberSet(3)
        numberset2 = geometry.NumberSet(3)
        self.assertFalse(numberset1.starts_right(numberset2))

        numberset1 = geometry.NumberSet(4)
        numberset2 = geometry.NumberSet(geometry.Interval("(", 3, 4, "]"))
        self.assertTrue(numberset1.starts_right(numberset2))

        numberset1 = geometry.NumberSet(4)
        numberset2 = geometry.NumberSet(geometry.Interval("[", 4, 5, "]"))
        self.assertFalse(numberset1.starts_right(numberset2))

        numberset1 = geometry.NumberSet(geometry.Interval("[", 3, 4, "]"))
        numberset2 = geometry.NumberSet(geometry.Interval("[", 4, 5, "]"))
        self.assertFalse(numberset1.starts_right(numberset2))

        numberset1 = geometry.NumberSet(geometry.Interval("(", 3, 5, "]"))
        numberset2 = geometry.NumberSet(geometry.Interval("[", 3, 5, "]"))
        self.assertTrue(numberset1.starts_right(numberset2))

        numberset1 = geometry.NumberSet(geometry.Interval("[", 3, 5, ")"))
        numberset2 = geometry.NumberSet(geometry.Interval("[", 3, 5, "]"))
        self.assertFalse(numberset1.starts_right(numberset2))

    # ends_left should return True iff the NumberSet ends to the left of
    # another NumberSet.
    def test_ends_left(self):
        numberset1 = geometry.NumberSet()
        numberset2 = geometry.NumberSet()
        self.assertFalse(numberset1.ends_left(numberset2))

        numberset1 = geometry.NumberSet(3)
        numberset2 = geometry.NumberSet()
        self.assertFalse(numberset1.ends_left(numberset2))

        numberset1 = geometry.NumberSet(3)
        numberset2 = geometry.NumberSet(4)
        self.assertTrue(numberset1.ends_left(numberset2))

        numberset1 = geometry.NumberSet(3)
        numberset2 = geometry.NumberSet(3)
        self.assertFalse(numberset1.ends_left(numberset2))

        numberset1 = geometry.NumberSet(4)
        numberset2 = geometry.NumberSet(geometry.Interval("(", 3, 4, "]"))
        self.assertFalse(numberset1.ends_left(numberset2))

        numberset1 = geometry.NumberSet(4)
        numberset2 = geometry.NumberSet(geometry.Interval("[", 4, 5, "]"))
        self.assertTrue(numberset1.ends_left(numberset2))

        numberset1 = geometry.NumberSet(geometry.Interval("[", 3, 4, ")"))
        numberset2 = geometry.NumberSet(geometry.Interval("[", 4, 5, "]"))
        self.assertTrue(numberset1.ends_left(numberset2))

        numberset1 = geometry.NumberSet(geometry.Interval("[", 3, 5, "]"))
        numberset2 = geometry.NumberSet(geometry.Interval("(", 3, 5, "]"))
        self.assertFalse(numberset1.ends_left(numberset2))

        numberset1 = geometry.NumberSet(geometry.Interval("[", 3, 5, ")"))
        numberset2 = geometry.NumberSet(geometry.Interval("[", 3, 5, "]"))
        self.assertTrue(numberset1.ends_left(numberset2))

    # ends_right should return True iff the NumberSet ends to the right of
    # another NumberSet.
    def test_ends_right(self):
        numberset1 = geometry.NumberSet()
        numberset2 = geometry.NumberSet()
        self.assertFalse(numberset1.ends_right(numberset2))

        numberset1 = geometry.NumberSet(3)
        numberset2 = geometry.NumberSet()
        self.assertFalse(numberset1.ends_right(numberset2))

        numberset1 = geometry.NumberSet(3)
        numberset2 = geometry.NumberSet(4)
        self.assertFalse(numberset1.ends_right(numberset2))

        numberset1 = geometry.NumberSet(3)
        numberset2 = geometry.NumberSet(3)
        self.assertFalse(numberset1.ends_right(numberset2))

        numberset1 = geometry.NumberSet(4)
        numberset2 = geometry.NumberSet(geometry.Interval("(", 3, 4, ")"))
        self.assertTrue(numberset1.ends_right(numberset2))

        numberset1 = geometry.NumberSet(5)
        numberset2 = geometry.NumberSet(geometry.Interval("[", 4, 5, "]"))
        self.assertFalse(numberset1.ends_right(numberset2))

        numberset1 = geometry.NumberSet(geometry.Interval("[", 3, 4, ")"))
        numberset2 = geometry.NumberSet(geometry.Interval("[", 4, 5, "]"))
        self.assertFalse(numberset1.ends_right(numberset2))

        numberset1 = geometry.NumberSet(geometry.Interval("[", 3, 5, "]"))
        numberset2 = geometry.NumberSet(geometry.Interval("(", 3, 5, "]"))
        self.assertFalse(numberset1.ends_right(numberset2))

        numberset1 = geometry.NumberSet(geometry.Interval("[", 3, 5, "]"))
        numberset2 = geometry.NumberSet(geometry.Interval("[", 3, 5, ")"))
        self.assertTrue(numberset1.ends_right(numberset2))


if __name__ == "__main__":
    unittest.main()
