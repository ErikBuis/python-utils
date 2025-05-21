"""A cheatsheet for calculating relationships between geometric objects.

Geometric objects implemented in this module:
- Vector2D
- Line2D
- Circle2D
- Rectangle2D
- Polygon2D
- Triangle2D

Other classes implemented in this module:
- Matrix2D
- Interval
- NumberSet

This cheatsheet does not use any external libraries.
"""

# ! Style guide for this file:
# !
# ! In multiple places in this file, I use Ellipsis (...) to indicate that
# ! I've omitted some code. Here, I use the following convention:
# ! ... = Details to be defined elsewhere, e.g. in a method of an ABC.
# ! pass = Not implemented yet / do nothing.
# !
# ! Docstrings:
# ! - Use triple double quotes (""") for docstrings. Directly after the
# !   opening quotes, there should be one line explaining what the function
# !   does.
# ! - This one line should not start with "Return". It should start with a
# !   verb's dictionary form in the present tense, e.g. "Create", "Perform",
# !   "Calculate", etc. Do not use the singular form of the verb, e.g.
# !   "Creates", "Performs", "Calculates", etc.
# ! - After that, there should be a blank line.
# ! - Then, the rest of the docstring further explains the function. If the
# !   function has parameters that are non-trivial, they should be explained
# !   in the docstring as well.
# !
# ! Method order within a class:
# ! - First, the class's properties should be defined.
# ! - Then, the class's magic methods should be defined in the following order:
# !     - Construction and iteration:
# !         - __init__ and other constructors with @classmethod
# !         - __iter__, __getitem__
# !         - __len__
# !     - Conversion and representation:
# !         - __repr__, __str__
# !         - __bool__
# !         - __hash__
# !         - __int__, __float__, __complex__
# !         - __round__, __floor__, __ceil__, __trunc__
# !     - Comparisons and containment:
# !         - __eq__, __lt__, __le__, __gt__, __ge__
# !         - __contains__
# !     - Arithmetic:
# !         - __pos__, __neg__, __abs__
# !         - __add__, __radd__, __iadd__
# !         - __sub__, __rsub__, __isub__
# !         - __mul__, __rmul__, __imul__
# !         - __truediv__, __rtruediv__, __itruediv__
# !         - __floordiv__, __rfloordiv__, __ifloordiv__
# !         - __mod__, __rmod__, __imod__
# !         - __divmod__, __rdivmod__
# !         - __pow__, __rpow__, __ipow__
# !         - __matmul__, __rmatmul__, __imatmul__
# !     - Bitwise:
# !         - __invert__
# !         - __and__, __rand__, __iand__
# !         - __or__, __ror__, __ior__
# !         - __xor__, __rxor__, __ixor__
# !         - __lshift__, __rlshift__, __ilshift__
# !         - __rshift__, __rrshift__, __irshift__
# !     - Miscellaneous:
# !         - __index__
# !         - __copy__, __deepcopy__
# !         - __enter__, __exit__
# !         - __call__
# !         - __dir__
# !         - __format__
# !         - __getattr__, __setattr__, __delattr__
# !         - __reduce__, __reduce_ex__, __getnewargs_ex__
# !         - __setstate__, __getstate__
# !         - __sizeof__
# !         - __slots__
# !         - __subclasshook__
# ! - Then, the class's other methods should be defined in the following order:
# !     - Private methods
# !     - Overridden methods from ABCs
# !     - Other methods

from __future__ import annotations

from abc import ABC, abstractmethod
from bisect import bisect_left
from collections.abc import Iterable, Iterator, Sequence
from heapq import heapify, heappop
from math import acos, atan2, cos, dist, hypot, inf, isinf, pi, sin, sqrt, tan
from typing import Any, Literal, TypeVar, cast, overload

from typing_extensions import override

from ..modules.bisect import bisect_right_using_left

number_type = (int, float)  # used for isinstance() checks

# This alias is mainly meant for functions where multiple arguments or an
# argument and the return value must be of the same subclass of
# GeometricObject2D.
geometricobject2d = TypeVar("geometricobject2d", bound="GeometricObject2D")


# GENERAL CLASSES #############################################################


class Matrix2D:
    """A 2x2 matrix.

    The matrix looks like:
    | a b |
    | c d |

    Once created, a Matrix2D is immutable.
    """

    @property
    def a(self) -> float:
        """The first component of the matrix."""
        return self._a

    @property
    def b(self) -> float:
        """The second component of the matrix."""
        return self._b

    @property
    def c(self) -> float:
        """The third component of the matrix."""
        return self._c

    @property
    def d(self) -> float:
        """The fourth component of the matrix."""
        return self._d

    def __init__(self, a: float, b: float, c: float, d: float) -> None:
        self._a = a
        self._b = b
        self._c = c
        self._d = d

    def __iter__(self) -> Iterator[float]:
        """Iterate over the matrix's components."""
        return iter((self._a, self._b, self._c, self._d))

    def __getitem__(self, idx: int) -> float:
        """Get the component at the given index."""
        if idx == 0:
            return self._a
        elif idx == 1:
            return self._b
        elif idx == 2:
            return self._c
        elif idx == 3:
            return self._d
        else:
            raise IndexError("Matrix index out of range.")

    @override
    def __repr__(self) -> str:
        """Represent the matrix as a string."""
        return f"{self.__class__.__name__}({', '.join(map(repr, self))})"

    @override
    def __str__(self) -> str:
        """Represent the matrix as a string."""
        return f"| {self._a}  {self._b} |\n| {self._c}  {self._d} |"

    @override
    def __hash__(self) -> int:
        """Create a hash id for the matrix."""
        return hash((self._a, self._b, self._c, self._d))

    @override
    def __eq__(self, other: object) -> bool:
        """Compare two matrices.

        Warning: this function internally uses a threshold of self.epsilon to
        determine if the matrices are equal.
        """
        if not isinstance(other, Matrix2D):
            return False
        return all(
            component_self == component_other
            for component_self, component_other in zip(self, other)
        )

    @overload
    def __matmul__(self, other: "Matrix2D") -> "Matrix2D":
        pass

    @overload
    def __matmul__(self, other: geometricobject2d) -> geometricobject2d:
        pass

    def __matmul__(
        self, other: "Matrix2D" | geometricobject2d
    ) -> "Matrix2D | geometricobject2d":
        """Multiply a matrix with another matrix or a geometric object."""
        if isinstance(other, Matrix2D):
            return Matrix2D(
                self._a * other._a + self._b * other._c,
                self._a * other._b + self._b * other._d,
                self._c * other._a + self._d * other._c,
                self._c * other._b + self._d * other._d,
            )
        return other.transform(self)


class Interval:
    """An interval on the real number line.

    The interval could be open or closed on either side. Possible formats are:
    - [a, b] = closed interval
    - (a, b) = open interval
    - (a, b] = half-open interval on the left
    - [a, b) = half-open interval on the right

    a and b can be any real number, including -inf and +inf. However, due to
    implementation details, if a is -inf, the left bracket must be '[', and if
    b is +inf, the right bracket must be ']'. We do this for algorithmic
    efficiency reasons.

    The interval's starting point (a) must be less than its ending point (b).
    Therefore, by extension, the interval cannot represent a point. We chose
    this intentionally to avoid complications with the implementation.

    Once created, an Interval is immutable.
    """

    @property
    def start_included(self) -> bool:
        """Whether the start of the interval is included.

        Returns:
            True if the starting point of the interval is included, False
            otherwise.
        """
        return self._start_included

    @property
    def end_included(self) -> bool:
        """Whether the end of the interval is included.

        Returns:
            True if the ending point of the interval is included, False
            otherwise.
        """
        return self._end_included

    @property
    def left_bracket(self) -> Literal["[", "("]:
        """The left bracket of the interval.

        Returns:
            '[' if the start of the interval is included, '(' otherwise.
        """
        return "[" if self._start_included else "("

    @property
    def right_bracket(self) -> Literal["]", ")"]:
        """The right bracket of the interval.

        Returns:
            ']' if the end of the interval is included, ')' otherwise.
        """
        return "]" if self._end_included else ")"

    @property
    def start(self) -> float:
        """The start of the interval.

        Returns:
            The starting point of the interval.
        """
        return self._start

    @property
    def end(self) -> float:
        """The end of the interval.

        Returns:
            The ending point of the interval.
        """
        return self._end

    def __init__(
        self,
        left_bracket: bool | Literal["[", "("],
        start: float,
        end: float,
        right_bracket: bool | Literal["]", ")"],
    ) -> None:
        """Create an Interval instance.

        The interval is defined by its start and end points, and whether the
        start and end points are included in the interval. The start and end
        points can be any real number, including -inf and +inf. However, due
        to implementation details, if the start is -inf, the left bracket must
        be '[', and if the end is +inf, the right bracket must be ']'. We do
        this for algorithmic efficiency reasons.

        The start point must be less than the end point. Thus, the interval
        cannot be empty, i.e. the start and end points cannot be equal.

        Args:
            left_bracket: Whether the start of the interval is included. If '['
                or True, the start is included. If '(', or False, the start is
                not excluded.
            start: The start of the interval.
            end: The end of the interval.
            right_bracket: Whether the end of the interval is included. If ']'
                or True, the end is included. If ')', or False, the end is not
                excluded.
        """
        if left_bracket not in ("[", "(", True, False):
            raise ValueError("Left bracket must be '[', '(', or bool.")
        if right_bracket not in ("]", ")", True, False):
            raise ValueError("Right bracket must be ']', ')', or bool.")
        if start >= end:
            raise ValueError("Interval start must be less than interval end.")
        if start == -inf and left_bracket not in ("[", True):
            raise ValueError("If start is -inf, left bracket must be '['.")
        if end == inf and right_bracket not in ("]", True):
            raise ValueError("If end is +inf, right bracket must be ']'.")

        # We internally included infinity in the interval because it makes the
        # operations performed on intervals consistent.
        self._start_included = (
            left_bracket
            if isinstance(left_bracket, bool)
            else left_bracket == "["
        )
        self._start = start
        self._end = end
        self._end_included = (
            right_bracket
            if isinstance(right_bracket, bool)
            else right_bracket == "]"
        )

    @override
    def __repr__(self) -> str:
        """Represent the interval as a string.

        Returns:
            A python representation of the Interval. The format is:
            "Interval(left_bracket, start, end, right_bracket)"

        Examples:
        >>> repr(Interval("[", 1, 2, ")"))
        "Interval('[', 1, 2, ')')"
        """
        return (
            f"{self.__class__.__name__}("
            f"{repr(self.left_bracket)}, "
            f"{repr(self._start)}, "
            f"{repr(self._end)}, "
            f"{repr(self.right_bracket)})"
        )

    @override
    def __str__(self) -> str:
        """Represent the interval as a string.

        Returns:
            A human-readable representation of the Interval. The format is:
            - "[start, end]" if the interval is closed.
            - "(start, end)" if the interval is open.
            - "(start, end]" if the interval is half-open on the left.
            - "[start, end)" if the interval is half-open on the right.

        Examples:
        >>> str(Interval("[", 1, 2, ")"))
        "[1, 2)"
        """
        return (
            f'{"[" if self._start_included else "("}{self._start}, '
            f'{self._end}{"]" if self._end_included else ")"}'
        )

    @override
    def __hash__(self) -> int:
        """Create a hash id for the interval.

        Returns:
            A hash id for the interval. The hash id is created by hashing the
            components of the interval.
        """
        return hash(
            (self._start_included, self._start, self._end, self._end_included)
        )

    @override
    def __eq__(self, other: object) -> bool:
        """Check if two objects represent the same set of numbers.

        Args:
            other: The other object to compare to.

        Returns:
            True if the two objects represent the same set of numbers, False
            otherwise.
        """
        if not isinstance(other, Interval):
            return False
        return (
            self._start_included == other._start_included
            and self._start == other._start
            and self._end == other._end
            and self._end_included == other._end_included
        )


class NumberSet:
    """A set of disjoint numbers and intervals.

    The set efficiently stores a subset of the real number line. It internally
    maintains a finite set of numbers and Interval objects, which are kept
    disjoint (non-overlapping and non-adjacent) at all times by merging
    overlapping and adjacent numbers and intervals. Note that while the set can
    contain an infinite amount of *real numbers* (since an interval on the real
    number line contains an infinite amount of real numbers), the amount of
    *objects* in the set is finite.

    A NumberSet is mutable after creation.

    TODO Write something general about which operations are supported and
    TODO what this class is useful for. The main aim should be to provide
    TODO an introduction for users who are not familiar with the class.
    """

    @property
    def components(self) -> Iterator[float | Interval]:
        """All numbers and interval objects stored in the set.

        Returns:
            An iterator of numbers and interval objects stored in the set.
            The components are returned in a sorted manner (ascending).

        Examples:
        >>> ns = NumberSet(0, 10, Interval("[", 1, 5, ")"))
        >>> list(ns.components)
        [0, Interval('[', 1, 5, ')'), 10]
        """
        for idx in range(0, len(self._boundaries), 2):
            if self._boundaries[idx] == self._boundaries[idx + 1]:
                yield self._boundaries[idx]
            else:
                yield Interval(
                    self._boundaries_included[idx],
                    self._boundaries[idx],
                    self._boundaries[idx + 1],
                    self._boundaries_included[idx + 1],
                )

    @property
    def amount_components(self) -> int:
        """The amount of numbers and intervals contained in the set.

        Returns:
            The amount of numbers and intervals contained in the set. Each
            standalone number counts as one component, and each interval counts
            as one component.

        Examples:
        >>> ns = NumberSet(0, 10, Interval("[", 1, 5, ")"))
        >>> ns.amount_components
        3
        """
        return len(self._boundaries) // 2

    def __init__(self, *components: "float | Interval | NumberSet") -> None:
        """Create a NumberSet instance.

        Note that -inf or inf cannot be added to the set directly, as they are
        not valid numbers. However, they can be added to the set indirectly by
        adding an interval that contains them.

        Overlapping and adjacent numbers and intervals will automatically be
        merged into one interval. By doing this, we internally maintain an
        efficient data structure that allows for fast lookup and modification
        of the set.

        Args:
            *components: A list of numbers, intervals, or other NumberSets to
                add to the set. If a component is a NumberSet, its components
                are added to the set by reference. If a component is an
                Interval, it is added to the set as a separate component.

        Examples:
        >>> NumberSet(0, 10, Interval("[", 1, 5, ")"))
        NumberSet(0, Interval('[', 1, 5, ')'), 10)

        >>> NumberSet(
        >>>     Interval("(", 3, 4, "]"),
        >>>     8.7,
        >>>     Interval("(", 3.1, 10, ")"),
        >>>     2.4,
        >>>     3,
        >>> )
        NumberSet(2.4, Interval('[', 3, 10, ')'))
        """
        # We maintain a sorted list of boundary values for fast lookup.
        # It is accompanied by a list of booleans that indicate whether each
        # boundary is included in the set or not.
        # Since the NumberSet always contains disjoint numbers and intervals,
        # the boundaries represent points at which the number line switches
        # between in the set and out of the set. For example, if:
        # > self._boundaries = [-inf, -2, 1, 1, 3, 4]
        # > self._boundaries_included = [True, False, True, True, False, True]
        # Then the set contains all numbers in the intervals:
        # [-inf, -2), [1, 1], and (3, 4].
        # We can note multiple things from this example:
        # - A point is represented by two adjancent equal boundaries, both of
        #   which are included in the set.
        # - The amount of boundaries is always a multiple of two, since both
        #   Interval objects and numbers have a start and an end boundary.
        # - The boundary for -inf and inf is always included in the set.
        # - A number x can be in the _boundaries list at most twice, since the
        #   only possible ways to have two equal numbers in the list is if:
        #   (1) They are from a point at x "[ ]".
        #   (2) They are from two separate non-adjacent intervals ") (".
        #       One ends with x, the other starts with x, but both exclude it.
        #   In all other cases, the two numbers/intervals would be adjacent,
        #   which is guaranteed to never happen since we always merge
        #   overlapping and adjacent intervals and numbers.
        self._boundaries: list[float] = []
        self._boundaries_included: list[bool] = []

        # Add the components to the set.
        for component in components:
            self |= component

    @classmethod
    def __direct_init(
        cls, boundaries: list[float], boundaries_included: list[bool]
    ) -> "NumberSet":
        """Create a NumberSet instance directly.

        Warning: Do not call this method as an end user! It is meant for
        internal use only.

        Args:
            boundaries: A list of boundaries for the set.
            boundaries_included: A list of booleans that indicate whether each
                boundary is included in the set or not.

        Returns:
            A NumberSet instance with the given attribute values.
        """
        self = cls.__new__(cls)
        self._boundaries = boundaries
        self._boundaries_included = boundaries_included
        return self

    @classmethod
    def __from_interval(cls, interval: Interval) -> "NumberSet":
        """Create a NumberSet instance from an interval.

        Args:
            interval: The Interval object to create the NumberSet from.

        Returns:
            A NumberSet instance with the given interval as its only component.
        """
        return cls.__direct_init(
            [interval.start, interval.end],
            [interval.start_included, interval.end_included],
        )

    def __iter__(self) -> Iterator[float | Interval]:
        """Iterate over the set's components.

        See the `components` property for more information.
        """
        return self.components

    def __getitem__(self, idx: int) -> float | Interval:
        """Get the component at the given index.

        Args:
            idx: The index of the component to get. Indexing from the end is
                supported by using negative indices. The components are:

        Returns:
            The component at the given index, assuming all components are
            sorted in ascending order.
        """
        idx = self.amount_components + idx if idx < 0 else idx
        if not 0 <= idx < self.amount_components:
            raise IndexError("NumberSet index out of range.")

        comp_idx = idx * 2
        if self._boundaries[comp_idx] == self._boundaries[comp_idx + 1]:
            return self._boundaries[comp_idx]
        return Interval(
            self._boundaries_included[comp_idx],
            self._boundaries[comp_idx],
            self._boundaries[comp_idx + 1],
            self._boundaries_included[comp_idx + 1],
        )

    @override
    def __repr__(self) -> str:
        """Represent the number set as a string.

        Returns:
            A python representation of the NumberSet. The format is:
            "NumberSet(component1, component2, ...)"
            The components are returned in a sorted manner (ascending).

        Examples:
        >>> ns = NumberSet(0, 10, Interval("[", 1, 5, ")"))
        >>> repr(ns)
        "NumberSet(0, Interval('[', 1, 5, ')'), 10)"
        """
        return (
            f"{self.__class__.__name__}("
            f"{', '.join(map(repr, self.components))})"
        )

    @override
    def __str__(self) -> str:
        """Represent the number set as a string.

        Returns:
            A human-readable representation of the NumberSet. The format is:
            "NumberSet(component1, component2, ...)"
            The components are returned in a sorted manner (ascending).

        Examples:
        >>> ns = NumberSet(0, 10, Interval("[", 1, 5, ")"))
        >>> str(ns)
        "{0, [1, 5), 10}"
        """
        return f"{{{', '.join(map(str, self.components))}}}"

    def __bool__(self) -> bool:
        """Check if the set contains any numbers or intervals.

        Returns:
            True if the set contains any numbers or intervals, False otherwise.
            This is equivalent to checking if the set is empty.
        """
        return bool(self._boundaries)

    @override
    def __eq__(self, other: object) -> bool:
        """Check if two sets contain the same components.

        Args:
            other: The object to compare to.

        Returns:
            True if the two sets contain the same components, False otherwise.
        """
        if not isinstance(other, NumberSet):
            return False
        return (
            self._boundaries == other._boundaries
            and self._boundaries_included == other._boundaries_included
        )

    def __lt__(self, other: "NumberSet") -> bool:
        """Check if all components in this set are smaller than all in other.

        Args:
            other: The other NumberSet to compare to.

        Returns:
            True if all components in the set are smaller than all in other,
            False otherwise.
        """
        return (
            self._boundaries[-1] < other._boundaries[0]
            or self._boundaries[-1] == other._boundaries[0]
            and (
                not self._boundaries_included[-1]
                or not other._boundaries_included[0]
            )
        )

    def __gt__(self, other: "NumberSet") -> bool:
        """Check if all components in this set are larger than all in other.

        Args:
            other: The other NumberSet to compare to.

        Returns:
            True if all components in the set are larger than all in other,
            False otherwise.
        """
        return (
            other._boundaries[-1] < self._boundaries[0]
            or other._boundaries[-1] == self._boundaries[0]
            and (
                not other._boundaries_included[-1]
                or not self._boundaries_included[0]
            )
        )

    def __contains__(self, other: "float | Interval | NumberSet") -> bool:
        """Check if other is completely contained in this set.

        This is equivalent to checking if other is a subset of this set.
        Note that -inf and inf are considered to be included in this set if
        they are the start or end of an interval, respectively.

        Args:
            other: The object to check for containment. Can be a number, an
                interval, or another NumberSet.

        Returns:
            True if other is completely contained in this set, False otherwise.
        """
        if isinstance(other, number_type):
            return self.__contains_number(other)
        if isinstance(other, NumberSet) and other.is_number():
            return self.__contains_number(other._boundaries[0])
        if isinstance(other, Interval):
            other = NumberSet.__from_interval(other)
        return self & other == other

    def __sub__(self, other: "float | Interval | NumberSet") -> "NumberSet":
        """Subtract another object from this set.

        This is equivalent to taking the difference between this set and the
        other object. The result is a new NumberSet instance that contains all
        components in this set that are not in the other object.

        Args:
            other: The object to subtract from this set. Can be a number, an
                interval, or another NumberSet.

        Returns:
            A new NumberSet instance with the result of the subtraction.
        """
        if isinstance(other, number_type):
            return self.copy().__isub_number(other)
        if isinstance(other, NumberSet) and other.is_number():
            return self.copy().__isub_number(other._boundaries[0])
        if isinstance(other, Interval):
            other = NumberSet.__from_interval(other)
        return self & ~other

    def __isub__(self, other: "float | Interval | NumberSet") -> "NumberSet":
        """Subtract another object from this set in place.

        This is equivalent to taking the difference between this set and the
        other object. The result is a new NumberSet instance that contains all
        components in this set that are not in the other object.

        Args:
            other: The object to subtract from this set. Can be a number, an
                interval, or another NumberSet.

        Returns:
            The NumberSet instance with the result of the subtraction.
        """
        if isinstance(other, number_type):
            return self.__isub_number(other)
        if isinstance(other, NumberSet) and other.is_number():
            return self.__isub_number(other._boundaries[0])
        if isinstance(other, Interval):
            other = NumberSet.__from_interval(other)
        self &= ~other
        return self

    def __invert__(self) -> "NumberSet":
        """Take the complement of the set.

        Returns:
            A new NumberSet instance with the complement of the set.
        """
        contains_minus_inf = self._boundaries and self._boundaries[0] == -inf
        contains_plus_inf = self._boundaries and self._boundaries[-1] == inf
        start_idx = 1 if contains_minus_inf else 0
        end_idx = -1 if contains_plus_inf else len(self._boundaries)
        boundaries = self._boundaries[start_idx:end_idx]
        boundaries_included = [
            not x for x in self._boundaries_included[start_idx:end_idx]
        ]
        if not contains_minus_inf:
            boundaries.insert(0, -inf)
            boundaries_included.insert(0, True)
        if not contains_plus_inf:
            boundaries.append(inf)
            boundaries_included.append(True)
        return NumberSet.__direct_init(boundaries, boundaries_included)

    def __and__(self, other: "float | Interval | NumberSet") -> "NumberSet":
        """Take the intersection between this set and another object.

        Args:
            other: The object to intersect with. Can be a number, an interval,
                or another NumberSet.

        Returns:
            A new NumberSet instance with the intersection of the two objects.
        """
        if isinstance(other, number_type):
            return self.copy().__iand_number(other)
        if isinstance(other, NumberSet) and other.is_number():
            return self.copy().__iand_number(other._boundaries[0])
        if isinstance(other, Interval):
            other = NumberSet.__from_interval(other)
        return self._perform_boolean_operation(other, True, "01", "00")

    def __iand__(self, other: "float | Interval | NumberSet") -> "NumberSet":
        """Take the intersection between this set and another object in place.

        Args:
            other: The object to intersect with. Can be a number, an interval,
                or another NumberSet.

        Returns:
            The NumberSet instance with the intersection of the two objects.
        """
        if isinstance(other, number_type):
            return self.__iand_number(other)
        if isinstance(other, NumberSet) and other.is_number():
            return self.__iand_number(other._boundaries[0])
        if isinstance(other, Interval):
            other = NumberSet.__from_interval(other)
        self._perform_boolean_operation_ip(other, True, "01", "00")
        return self

    def __or__(self, other: "float | Interval | NumberSet") -> "NumberSet":
        """Take the union between this set and another object.

        Args:
            other: The object to union with. Can be a number, an interval, or
                another NumberSet.

        Returns:
            A new NumberSet instance with the union of the two objects.
        """
        if isinstance(other, number_type):
            return self.copy().__ior_number(other)
        if isinstance(other, NumberSet) and other.is_number():
            return self.copy().__ior_number(other._boundaries[0])
        if isinstance(other, Interval):
            other = NumberSet.__from_interval(other)
        return self._perform_boolean_operation(other, True, "11", "01")

    def __ior__(self, other: "float | Interval | NumberSet") -> "NumberSet":
        """Take the union between this set and another object in place.

        Args:
            other: The object to union with. Can be a number, an interval, or
                another NumberSet.

        Returns:
            The NumberSet instance with the union of the two objects.
        """
        if isinstance(other, number_type):
            return self.__ior_number(other)
        if isinstance(other, NumberSet) and other.is_number():
            return self.__ior_number(other._boundaries[0])
        if isinstance(other, Interval):
            other = NumberSet.__from_interval(other)
        self._perform_boolean_operation_ip(other, True, "11", "01")
        return self

    def __xor__(self, other: "float | Interval | NumberSet") -> "NumberSet":
        """Take the symmetric difference between this set and another object.

        Args:
            other: The object to symmetric difference with. Can be a number,
                an interval, or another NumberSet.

        Returns:
            A new NumberSet instance with the symmetric difference of the two
            objects.
        """
        if isinstance(other, number_type):
            return self.copy().__ixor_number(other)
        if isinstance(other, NumberSet) and other.is_number():
            return self.copy().__ixor_number(other._boundaries[0])
        if isinstance(other, Interval):
            other = NumberSet.__from_interval(other)
        return self._perform_boolean_operation(other, True, "10", "01")

    def __ixor__(self, other: "float | Interval | NumberSet") -> "NumberSet":
        """Take the symmetric difference between this set and another object
        in place.

        Args:
            other: The object to symmetric difference with. Can be a number,
                an interval, or another NumberSet.

        Returns:
            The NumberSet instance with the symmetric difference of the two
            objects.
        """
        if isinstance(other, number_type):
            return self.__ixor_number(other)
        if isinstance(other, NumberSet) and other.is_number():
            return self.__ixor_number(other._boundaries[0])
        if isinstance(other, Interval):
            other = NumberSet.__from_interval(other)
        self._perform_boolean_operation_ip(other, True, "10", "01")
        return self

    def __lshift__(self, shift: float) -> "NumberSet":
        """Shift the set to the left by the given amount.

        Args:
            shift: The amount to shift the set to the left.

        Returns:
            A new NumberSet instance with the shifted components.
        """
        return NumberSet.__direct_init(
            [x - shift for x in self._boundaries], self._boundaries_included
        )

    def __ilshift__(self, shift: float) -> "NumberSet":
        """Shift the set to the left by the given amount in place.

        Args:
            shift: The amount to shift the set to the left.

        Returns:
            The NumberSet instance with the shifted components.
        """
        for idx in range(0, len(self._boundaries)):
            self._boundaries[idx] -= shift
        return self

    def __rshift__(self, shift: float) -> "NumberSet":
        """Shift the set to the right by the given amount.

        Args:
            shift: The amount to shift the set to the right.

        Returns:
            A new NumberSet instance with the shifted components.
        """
        return NumberSet.__direct_init(
            [x + shift for x in self._boundaries], self._boundaries_included
        )

    def __irshift__(self, shift: float) -> "NumberSet":
        """Shift the set to the right by the given amount in place.

        Args:
            shift: The amount to shift the set to the right.

        Returns:
            The NumberSet instance with the shifted components.
        """
        for idx in range(0, len(self._boundaries)):
            self._boundaries[idx] += shift
        return self

    def __contains_number(self, number: float) -> bool:
        """Check if the given number is in the set.

        Args:
            number: The number to check for.

        Returns:
            True if the number is in the set, False otherwise.
        """
        in_set, _, _, _ = self.lookup(number)
        return in_set

    def __isub_number(
        self, number: float, idx: int | None = None
    ) -> "NumberSet":
        """Remove the given number from the set in place.

        Args:
            number: The number to remove from the set.
            idx: The index of the number in the set as returned by
                bisect_left(). If not given, it will be calculated by the
                function.

        Returns:
            The NumberSet instance with the removed number.
        """
        if isinf(number):
            raise ValueError(
                "Cannot remove inf from a NumberSet directly. If you want to"
                " remove inf, use an Interval instead."
            )

        in_set, on_start, on_end, idx = self.lookup(number, idx=idx)
        if not in_set:
            # The number is not in the set.
            return self
        # The number is in the set.
        if not on_start and not on_end:
            # The number is in an interval.
            self._boundaries.insert(idx, number)
            self._boundaries.insert(idx + 1, number)
            self._boundaries_included.insert(idx, False)
            self._boundaries_included.insert(idx + 1, False)
            return self
        if on_start and on_end:
            # The number is on a point.
            self._boundaries.pop(idx + 1)
            self._boundaries.pop(idx)
            self._boundaries_included.pop(idx + 1)
            self._boundaries_included.pop(idx)
            return self
        # The number is on an included boundary.
        self._boundaries_included[idx] = False
        return self

    def __iand_number(self, number: float) -> "NumberSet":
        """Take the intersection of the set and the given number in place.

        Args:
            number: The number to intersect with.

        Returns:
            The NumberSet instance with the intersection of the two objects.
        """
        if isinf(number):
            raise ValueError(
                "Cannot intersect with inf directly. If you want to take the"
                " intersection with inf, use an Interval instead."
            )

        in_set, _, _, _ = self.lookup(number)
        if in_set:
            # The number is in the set.
            self._boundaries = [number, number]
            self._boundaries_included = [True, True]
        else:
            # The number is not in the set.
            self._boundaries = []
            self._boundaries_included = []
        return self

    def __ior_number(
        self, number: float, idx: int | None = None
    ) -> "NumberSet":
        """Add the given number to the set in place.

        Args:
            number: The number to add to the set.
            idx: The index of the number in the set as returned by
                bisect_left(). If not given, it will be calculated by the
                function.

        Returns:
            The NumberSet instance with the added number.
        """
        if isinf(number):
            raise ValueError(
                "Cannot add inf to a NumberSet directly. If you want to add"
                " inf, use an Interval instead."
            )

        in_set, on_start, on_end, idx = self.lookup(number, idx=idx)
        if in_set:
            # The number is already in the set.
            return self
        # The number is not in the set yet.
        if not on_start and not on_end:
            # The number is outside all components.
            self._boundaries.insert(idx, number)
            self._boundaries.insert(idx + 1, number)
            self._boundaries_included.insert(idx, True)
            self._boundaries_included.insert(idx + 1, True)
            return self
        if on_start and on_end:
            # The number fills a hole between two intervals.
            self._boundaries.pop(idx + 1)
            self._boundaries.pop(idx)
            self._boundaries_included.pop(idx + 1)
            self._boundaries_included.pop(idx)
            return self
        # The number is on an excluded boundary.
        self._boundaries_included[idx] = True
        return self

    def __ixor_number(self, number: float) -> "NumberSet":
        """Take the symmetric difference of the set and the given number in
        place.

        Args:
            number: The number to symmetric difference with.

        Returns:
            The NumberSet instance with the symmetric difference of the two
            objects.
        """
        if isinf(number):
            raise ValueError(
                "Cannot symmetric difference with inf directly. If you want to"
                " take the symmetric difference with inf, use an Interval"
                " instead."
            )

        in_set, _, _, idx = self.lookup(number)
        if in_set:
            # The number is in the set.
            return self.__isub_number(number, idx=idx)
        # The number is not in the set yet.
        return self.__ior_number(number, idx=idx)

    def copy(self) -> "NumberSet":
        """Return a copy of the set."""
        return NumberSet.__direct_init(
            self._boundaries.copy(), self._boundaries_included.copy()
        )

    def lookup(
        self, number: float, idx: int | None = None
    ) -> tuple[bool, bool, bool, int]:
        """Look up where the number is located within the set.

        idx is the index of the number in the set as returned by bisect_left().
        If idx is not given, it will be calculated by the function.

        The table below provides an overview of the possible situations.
        In    | On     | On     | Bracket | Bracket | Situation
        set?  | start? | end?   | @ idx   | @ idx+1 |
        ------+--------+--------+---------+---------+--------------------------
        True  | True   | True   | [       | ]       | @ Point
        True  | True   | False  | [       |         | @ Included start bound
        True  | False  | True   | ]       |         | @ Included end bound
        True  | False  | False  |         |         | @ In interval
        False | True   | True   | )       | (       | @ Hole between intervals
        False | True   | False  | (       |         | @ Excluded start bound
        False | False  | True   | )       |         | @ Excluded end bound
        False | False  | False  |         |         | @ Outside all components

        Returns:
            Tuple containing:
            - Boolean indicating if the number is already in the set.
            - Boolean indicating if the number is on a start bound.
            - Boolean indicating if the number is on an end bound.
            - The index returned by bisect_left().
        """
        if idx is None:
            idx = bisect_left(self._boundaries, number)

        if idx == len(self._boundaries) or self._boundaries[idx] != number:
            # The number is between two boundaries or outside all components.
            if idx % 2 == 1:
                # The number lies in an existing interval.
                return True, False, False, idx

            # The number is between two intervals or outside all components.
            return False, False, False, idx

        # The number is equal to a boundary.
        if self._boundaries_included[idx]:
            # The number is on an included boundary.
            if idx % 2 == 1:
                # The number lies on an end bound.
                return True, False, True, idx

            # The number lies on a start bound or on a point.
            if self._boundaries[idx + 1] != number:
                # The number lies on a start bound.
                return True, True, False, idx

            # The number lies on a point.
            return True, True, True, idx

        # The number is on an excluded boundary.
        if idx % 2 == 0:
            # The number lies on a start bound.
            return False, True, False, idx

        # The number lies on an end bound or in a hole between 2 intervals.
        if (
            idx + 1 == len(self._boundaries)
            or self._boundaries[idx + 1] != number
        ):
            # The number lies on an end bound.
            return False, False, True, idx

        # The number lies in a hole between 2 intervals.
        return False, True, True, idx

    @staticmethod
    def contains_parallel(
        numbersets: Sequence["NumberSet"], numbers: Sequence[float]
    ) -> list[list[int]]:
        """Check in which sets the numbers are contained.

        This function uses a specialized algorithm to check for containment
        in multiple NumberSet objects at once. Thus, it is faster when
        checking for containment in multiple sets at once. However, it still
        scales linearly with the amount of numbers, so it is not recommended to
        use this function if you only need to check for containment in one set.

        Formally, when:
        N = amount of numbers
        S = amount of numbersets
        C = amount of components over all numbersets
        A naive approach of looping over all numbersets and numbers would have
        a runtime of O(N * S * log(C)), while this function has a runtime of
        O((C + N) * (log(C) + S)). In most cases, log(C) is much smaller than
        S, so the runtime is approximately O((C + N) * S).

        Hence, this function is faster under the assumption that N > C.
        (In this case, the complexity can be reduced to O(N * S).)
        Note, however, that this is still pessimistic, since this estimate
        which will only be reached if all numbersets overlap. In practice, the
        less overlap there is between the nunmbersets, the faster this function
        will be.

        Args:
            numbersets: A list of NumberSet objects to check for containment.
                Length: S
            numbers: A list of numbers to check for containment.
                Length: N

        Returns:
            A list of lists of indices representing the indices of the sets
            that contain the corresponding number. The n-th element of the
            returned outer list contains a list of indices s such that
            numbers[n] is in numbersets[s]. The inner lists of indices s are in
            no particular order. If a number is not in any of the sets, the
            corresponding list will be empty.
                Length: N
                Length of inner lists: <= S
        """
        # First, we build a lookup table that stores all boundaries of the
        # sets. To clear up how the data structure is built, consider the
        # following example:
        # numbersets = [
        #     NumberSet(Interval("(", -1, 2, ")"), Interval("[", 4, 5, "]")),
        #     NumberSet(Interval("[", -inf, -1, ")"), 4),
        #     NumberSet(Interval("[", -2, 1, "]")),
        # ]
        # Then the numbersets can be visualized as follows:
        # (0)             (-----------------)           [-----]
        # (1) <-----------)                             .
        # (2)       [-----------------]
        #
        # In this case, the lookup table will be:
        #   -inf   -2    -1     0     1     2     3     4     5   inf
        #     <-----|-----|-----|-----|-----|-----|-----|-----|----->
        #                d0          b0    _0          c0    a0
        #    c1    b1    _1                            a1
        #          c2    b2          a2
        # Here, I have used a {letter}{digit} notation to indicate:
        # - For the {letter}:
        #   _ = interval ends here and bound is excluded, NOT stored in lookup!
        #   a = interval ends here and bound is included
        #   b = interval continues here, thus bound is always included
        #   c = interval starts here and bound is included
        #   d = interval starts here and bound is excluded
        # - The {digit} corresponds to the numberset index s.
        #
        # Now, we can easily find the sets that contain a number by using the
        # following pseudocode:
        # ```
        # idx = bisect_left(lookup, number)
        # if number == lookup[idx].number:  # number is on a boundary
        #     return [s for s, letter in lookup[idx].sets if letter in "abc"]
        # else:  # number is between two boundaries
        #     return [s for s, letter in lookup[idx-1].sets if letter in "bcd"]
        # ```
        #
        # In the actual code, the letters are not stored in the lookup table.
        # Instead, for every number in the lookup table, its corresponding
        # {letter}{digit} entries will be added one-by-one in alphabetical
        # order. We can use this fact to our advantage, since we only need to
        # know which sets belong to group "abc" and which to group "bcd". For
        # example, if our lookup table for a certain number looks like this:
        # lookup[idx].sets   = [0, 9, 7, 4, 2, 1, 8]
        # lookup[idx].letter = [a, a, b, c, c, c, d]  # not actually stored!
        # Then we can see that the first 6 entries belong to group "abc" and
        # the last 5 entries belong to group "bcd". This means that if we just
        # store the list [0, 9, 7, 4, 2, 1, 8] and the index at which group
        # "abc" ends and "bcd" starts, we can quickly select the sets from one
        # of the two groups whenever we need to using Python slicing.
        #
        # The updated pseudocode would look like this:
        # ```
        # idx = bisect_left(lookup, number)
        # if number == lookup[idx].number:  # number is on a boundary
        #     return lookup[idx].sets[: lookup[idx].abc_ends]
        # else:  # number is between two boundaries
        #     return lookup[idx-1].sets[lookup[idx-1].bcd_starts :]
        # ```

        # Go through all the sets and add their boundaries to a priority queue
        # so we can easily extract them in sorted order.
        # Runtime: O(C)
        prioq: list[tuple[float, str, int]] = []
        for s, numberset in enumerate(numbersets):
            for component in numberset.components:
                if isinstance(component, number_type):
                    prioq.append((component, "c", s))
                else:
                    letter = "c" if component.start_included else "d"
                    prioq.append((component.start, letter, s))
                    # "_" won't be included in the lookup table, but we need
                    # it here so we don't miss a boundary and can delete the
                    # interval from the continued_entries when it ends.
                    letter = "a" if component.end_included else "_"
                    prioq.append((component.end, letter, s))
        heapify(prioq)  # O(C), see docs

        # Go through all the boundaries and build the lookup table. In addition
        # to the lookup table being sorted by number, the tuples (str, int)
        # are also sorted. This way, we can easily extract which sets the
        # number is in without actually having to filter through all the sets
        # in lookup[idx] as in the naive pseudocode above.
        # Runtime: O(C * (S + log C))
        lookup: list[tuple[float, list[int], int, int]] = []
        curr_number: float | None = None
        curr_sets: list[int] = []
        curr_abc_ends_idx: int | None = None  # exclusive
        curr_bcd_starts_idx: int | None = None  # inclusive
        ongoing_b_sets: set[int] = set()
        while prioq:
            number, letter, s = heappop(prioq)  # O(log(C))

            # If we are at a new number, add the previous number to the lookup
            # table and reset the current lookup variables.
            if curr_number is None:
                curr_number = number
            elif number != curr_number:
                # If we did not get to the point where the "bcd" group starts,
                # we will start it now.
                if curr_bcd_starts_idx is None:
                    curr_bcd_starts_idx = len(curr_sets)
                    curr_sets.extend(ongoing_b_sets)  # O(S)

                # If we did not get to the point where the "abc" group ends,
                # we will end it now.
                if curr_abc_ends_idx is None:
                    curr_abc_ends_idx = len(curr_sets)

                # Add the current number to the lookup table.
                lookup.append((
                    curr_number,
                    curr_sets,
                    curr_abc_ends_idx,
                    curr_bcd_starts_idx,
                ))

                # Reset the current lookup variables.
                curr_number = number
                curr_sets = []
                curr_abc_ends_idx = None
                curr_bcd_starts_idx = None

            # Process entries that end here.
            if letter == "_":
                ongoing_b_sets.remove(s)
                continue

            if letter == "a":
                curr_sets.append(s)
                ongoing_b_sets.remove(s)
                continue

            # If we get here, we are at a "b", "c", or "d" entry.
            # Thus, the "bcd" group starts here.
            if curr_bcd_starts_idx is None:
                curr_bcd_starts_idx = len(curr_sets)
                curr_sets.extend(ongoing_b_sets)  # O(S)

            # Process entries that start here.
            if letter == "c":
                curr_sets.append(s)
                ongoing_b_sets.add(s)
                continue

            # If we get here, we are at a "d" entry.
            # Thus, the "abc" group ends here.
            if curr_abc_ends_idx is None:
                curr_abc_ends_idx = len(curr_sets)

            if letter == "d":
                curr_sets.append(s)
                ongoing_b_sets.add(s)

        # Add the last number to the lookup table.
        if curr_number is not None:
            # If we did not get to the point where the "bcd" group starts,
            # we will start it now.
            if curr_bcd_starts_idx is None:
                curr_bcd_starts_idx = len(curr_sets)
                curr_sets.extend(ongoing_b_sets)  # O(S)

            # If we did not get to the point where the "abc" group ends,
            # we will end it now.
            if curr_abc_ends_idx is None:
                curr_abc_ends_idx = len(curr_sets)

            # Add the current number to the lookup table.
            lookup.append((
                curr_number,
                curr_sets,
                curr_abc_ends_idx,
                curr_bcd_starts_idx,
            ))

        # Now, we can easily find the sets that contain a number.
        # Runtime: O(N * (log(C) + S))
        results: list[list[int]] = []
        for number in numbers:
            idx = bisect_left(lookup, (number,))  # tuple to allow comparison
            if idx == len(lookup) or idx == 0 and lookup[idx][0] != number:
                results.append([])
            elif lookup[idx][0] == number:
                results.append(lookup[idx][1][: lookup[idx][2]])
            else:
                results.append(lookup[idx - 1][1][lookup[idx - 1][3] :])

        return results

    # TODO Refactor this class. I have refactored all the above code already
    # TODO by improving the docstrings, making it more streamlined and
    # TODO readable, and improving the corresponding docstrings. However, the
    # TODO code below has not been refactored yet.

    @staticmethod
    def __closest_bound_right_if_kept(
        in_set: bool, on_start: bool, on_end: bool, bisect_right_idx: int
    ) -> int:
        """Get the closest boundary to the right if keeping the set.

        in_set, on_start, and on_end must be taken from the lookup() method.
        The given idx must be the index as returned by bisect_right().

        If there is no boundary right of the given number, then
        len(self._boundaries) will be returned.

        The possible situations are illustrated below.
        In      | On     | On     | Illustration
        set?    | start? | end?   | ^ = bisect_right_idx, $ = return idx,
                |        |        | * = number, written e.g. "*$" if same pos.
                |        |        | [ or ] = included boundary
                |        |        | ( or ) = excluded boundary
                |        |        | { or } = unknown boundary
        --------+--------+--------+--------------------------------------------
        True    | True   | True   | {-----}     [     ]     {-----}
                |        |        |             $  *        ^       (exception)
        True    | True   | False  | {-----}     [-----}
                |        |        |             *$    ^
        True    | False  | True   |       {-----]     {-----}
                |        |        |             *$    ^
        True    | False  | False  |          {-----}
                |        |        |             *  $^
        False   | True   | True   |    {-----)     (-----}
                |        |        |             *  $     ^
        False   | True   | False  | {-----}     (-----}
                |        |        |             *$    ^
        False   | False  | True   |       {-----)     {-----}
                |        |        |             *     $^
        False   | False  | False  |    {-----}     {-----}
                |        |        |             *  $^
        """
        # The above illustration but compressed (how many indices from ^ to $):
        # -2
        # -1
        # -1
        #  0
        # -1
        # -1
        #  0
        #  0
        return bisect_right_idx - (
            in_set and (on_start + on_end) or not in_set and on_start
        )

    @staticmethod
    def __closest_bound_left_if_kept(
        in_set: bool, on_start: bool, on_end: bool, bisect_left_idx: int
    ) -> int:
        """Get the closest boundary to the left if keeping the set.

        in_set, on_start, and on_end must be taken from the lookup() method.
        The given idx must be the index as returned by bisect_left().

        If there is no boundary left of the given number, then -1 will be
        returned.

        The possible situations are illustrated below.
        In      | On     | On     | Illustration
        set?    | start? | end?   | ^ = bisect_left_idx, $ = return idx,
                |        |        | * = number, written e.g. "*$" if same pos.
                |        |        | [ or ] = included boundary
                |        |        | ( or ) = excluded boundary
                |        |        | { or } = unknown boundary
        --------+--------+--------+--------------------------------------------
        True    | True   | True   | {-----}     [     ]     {-----}
                |        |        |             ^  *  $             (exception)
        True    | True   | False  | {-----}     [-----}
                |        |        |             *$^
        True    | False  | True   |       {-----]     {-----}
                |        |        |             *$^
        True    | False  | False  |          {-----}
                |        |        |          $  *  ^
        False   | True   | True   |    {-----)     (-----}
                |        |        |          $^ *
        False   | True   | False  | {-----}     (-----}
                |        |        |       $     *^
        False   | False  | True   |       {-----)     {-----}
                |        |        |             *$^
        False   | False  | False  |    {-----}     {-----}
                |        |        |          $  *  ^
        """
        # The above illustration but compressed (how many indices from ^ to $):
        # +1 --- +1 ---> +2
        #  0 --- +1 ---> +1
        #  0 --- +1 ---> +1
        # -1 --- +1 --->  0
        #  0 --- +1 ---> +1
        # -1 --- +1 --->  0
        #  0 --- +1 ---> +1
        # -1 --- +1 --->  0
        return (
            bisect_left_idx
            - 1
            + (in_set and (on_start + on_end) or not in_set and on_end)
        )

    @staticmethod
    def __closest_bound_right_if_swapped(
        in_set: bool, on_start: bool, on_end: bool, bisect_right_idx: int
    ) -> int:
        """Get the closest boundary to the right if swapping the set.

        in_set, on_start, and on_end must be taken from the lookup() method.
        The given idx must be the index as returned by bisect_right().

        If there is no boundary right of the given number, then
        len(self._boundaries) will be returned.

        The possible situations are illustrated below.
        In      | On     | On     | Illustration
        set?    | start? | end?   | ^ = bisect_right_idx, $ = return idx,
                |        |        | * = number, written e.g. "*$" if same pos.
                |        |        | [ or ] = included boundary
                |        |        | ( or ) = excluded boundary
                |        |        | { or } = unknown boundary
        --------+--------+--------+--------------------------------------------
        True    | True   | True   | {-----}     [     ]     {-----}
                |        |        |                *  $     ^
        True    | True   | False  | {-----}     [-----}
                |        |        |             *     $^
        True    | False  | True   |       {-----]     {-----}
                |        |        |             *$    ^
        True    | False  | False  |          {-----}
                |        |        |             *  $^
        False   | True   | True   |    {-----)     (-----}
                |        |        |          $  *        ^          (exception)
        False   | True   | False  | {-----}     (-----}
                |        |        |             *$    ^
        False   | False  | True   |       {-----)     {-----}
                |        |        |             *$    ^
        False   | False  | False  |    {-----}     {-----}
                |        |        |             *  $^
        """
        # The above illustration but compressed (how many indices from ^ to $):
        # -1
        #  0
        # -1
        #  0
        # -2
        # -1
        # -1
        #  0
        return bisect_right_idx - (
            in_set and on_end or not in_set and (on_start + on_end)
        )

    @staticmethod
    def __closest_bound_left_if_swapped(
        in_set: bool, on_start: bool, on_end: bool, bisect_left_idx: int
    ) -> int:
        """Get the closest boundary to the left if swapping the set.

        in_set, on_start, and on_end must be taken from the lookup() method.
        The given idx must be the index as returned by bisect_left().

        If there is no boundary left of the given number, then -1 will be
        returned.

        The possible situations are illustrated below.
        In      | On     | On     | Illustration
        set?    | start? | end?   | ^ = bisect_left_idx, $ = return idx,
                |        |        | * = number, written e.g. "*$" if same pos.
                |        |        | [ or ] = included boundary
                |        |        | ( or ) = excluded boundary
                |        |        | { or } = unknown boundary
        --------+--------+--------+--------------------------------------------
        True    | True   | True   | {-----}     [     ]     {-----}
                |        |        |             $^ *
        True    | True   | False  | {-----}     [-----}
                |        |        |             *$^
        True    | False  | True   |       {-----]     {-----}
                |        |        |       $     *^
        True    | False  | False  |          {-----}
                |        |        |          $  *  ^
        False   | True   | True   |    {-----)     (-----}
                |        |        |          ^  *  $                (exception)
        False   | True   | False  | {-----}     (-----}
                |        |        |             *$^
        False   | False  | True   |       {-----)     {-----}
                |        |        |             *$^
        False   | False  | False  |    {-----}     {-----}
                |        |        |          $  *  ^
        """
        # The above illustration but compressed (how many indices from ^ to $):
        #  0 --- +1 ---> +1
        #  0 --- +1 ---> +1
        # -1 --- +1 --->  0
        # -1 --- +1 --->  0
        # +1 --- +1 ---> +2
        #  0 --- +1 ---> +1
        #  0 --- +1 ---> +1
        # -1 --- +1 --->  0
        return (
            bisect_left_idx
            - 1
            + (in_set and on_start or not in_set and (on_start + on_end))
        )

    def __extract_subset_helper(
        self,
        a: float,
        b: float,
        a_included: bool,
        b_included: bool,
        convert_to: Literal["00", "01", "10", "11"],
        a_idx: int | None = None,
        b_idx: int | None = None,
    ) -> tuple[list[float], list[bool]]:
        """Extract the given subset of the number line from the set.

        a and b are the boundaries of the subset. They can be any number,
        including inf and -inf, but a must be smaller than or equal to b.

        a_included and b_included indicate whether the boundaries should be
        extracted from the set too. Note that this does not say anything about
        whether the boundaries will actually be included in the subset or not.

        If a or b is -inf or inf, then the corresponding a_included or
        b_included must be True.

        convert_to signifies what to do with the selected subset:
        - If "00", the subset is discarded.
        - If "01", the subset is kept unchanged.
        - If "10", the subset is swapped (i.e. a 'not' operation is applied).
        - If "11", the subset is returned in full.

        a_idx and b_idx are the indices returned by bisect_left().
        If they are not given, they will be calculated by the function.
        """
        if a > b:
            raise ValueError(
                "The start of the subset must be smaller than or equal to the"
                " end."
            )
        if a == -inf and not a_included:
            raise ValueError(
                "The start of the subset must be included if it is -inf."
            )
        if b == inf and not b_included:
            raise ValueError(
                "The end of the subset must be included if it is inf."
            )

        if convert_to == "00":
            return [], []
        if convert_to == "11":
            return [a, b], [a_included, b_included]

        a_in_set, a_on_start, a_on_end, a_idx = self.lookup(a, idx=a_idx)
        b_in_set, b_on_start, b_on_end, b_idx = self.lookup(b, idx=b_idx)
        a_idx = bisect_right_using_left(self._boundaries, a, a_idx)

        # The select_from and select_to bounds are inclusive.
        if convert_to == "01":
            select_from = NumberSet.__closest_bound_right_if_kept(
                a_in_set, a_on_start, a_on_end, a_idx
            )
            select_to = NumberSet.__closest_bound_left_if_kept(
                b_in_set, b_on_start, b_on_end, b_idx
            )
            add_a_before = a_in_set and not a_on_start
            add_b_after = b_in_set and not b_on_end
        else:  # convert_to == "10"
            select_from = NumberSet.__closest_bound_right_if_swapped(
                a_in_set, a_on_start, a_on_end, a_idx
            )
            select_to = NumberSet.__closest_bound_left_if_swapped(
                b_in_set, b_on_start, b_on_end, b_idx
            )
            add_a_before = not a_in_set and not a_on_end
            add_b_after = not b_in_set and not b_on_start

        # Extract the subset from the set.
        boundaries = self._boundaries[select_from : select_to + 1]
        if convert_to == "01":
            boundaries_included = self._boundaries_included[
                select_from : select_to + 1
            ]
        else:  # convert_to == "10"
            boundaries_included = [
                not x
                for x in self._boundaries_included[select_from : select_to + 1]
            ]

        # Add the start and end boundaries if necessary.
        if add_a_before:
            boundaries.insert(0, a)
            boundaries_included.insert(0, True)
        if add_b_after:
            boundaries.append(b)
            boundaries_included.append(True)

        # Exclude the boundaries if requested.
        if (
            boundaries
            and boundaries[0] == a
            and boundaries_included[0]
            and not a_included
        ):
            if boundaries[1] == a:
                # The number is on a point.
                boundaries = boundaries[2:]
                boundaries_included = boundaries_included[2:]
            else:
                # The number is on an included boundary.
                boundaries_included[0] = False
        if (
            boundaries
            and boundaries[-1] == b
            and boundaries_included[-1]
            and not b_included
        ):
            if boundaries[-2] == b:
                # The number is on a point.
                boundaries = boundaries[:-2]
                boundaries_included = boundaries_included[:-2]
            else:
                # The number is on an included boundary.
                boundaries_included[-1] = False

        return boundaries, boundaries_included

    def _extract_subset(
        self,
        a: float,
        b: float,
        a_included: bool,
        b_included: bool,
        convert_to: Literal["00", "01", "10", "11"],
        a_idx: int | None = None,
        b_idx: int | None = None,
    ) -> "NumberSet":
        """Extract the given subset of the number line from the set.

        See __extract_subset_helper() for more information.
        """
        return NumberSet.__direct_init(
            *self.__extract_subset_helper(
                a,
                b,
                a_included,
                b_included,
                convert_to,
                a_idx=a_idx,
                b_idx=b_idx,
            )
        )

    @staticmethod
    def __concat_subsets_helper(
        subsets: list["NumberSet"], transitions: list[float]
    ) -> tuple[list[float], list[bool]]:
        """Concatenate extracted subsets.

        The transitions indicate locations of boundaries between the subsets.

        The list of subsets and transitions contain n sets and n-1 transitions
        respectively. It is assumed that subset i does not contain any numbers
        outside of transitions[i-1] and transitions[i]. This is automatically
        ensured by _extract_subset(), so please use that function.

        The following drawing illustates how this function will deal with
        transitions between subsets.
          left     right  | concatenated      | case
        ------------------+-------------------+--------------
            []     []     |     []            |    (2)
            []     [----- |            [----- |    (2)
            []     (----- |            [----- |    (2)
            []            |     []            | (1)
        -----]     []     | -----]            |    (2)
        -----]     [----- | ----------------- |       (3)
        -----]     (----- | ----------------- |       (3)
        -----]            | -----]            | (1)
        -----)     []     | -----]            |    (2)
        -----)     [----- | ----------------- |       (3)
        -----)     (----- | -----)     (----- |          (4)
        -----)            | -----)            | (1)
                   []     |            []     | (1)
                   [----- |            [----- | (1)
                   (----- |            (----- | (1)
                          |                   | (1)
        """
        # idx_start and idx_end are inclusive indices.
        boundaries = []
        boundaries_included = []
        idx_start = 0
        for subset_left, subset_right, transition in zip(
            subsets[:-1], subsets[1:], transitions
        ):
            # Determine the type of the left boundary.
            if (
                subset_left.is_empty()
                or subset_left._boundaries[-1] != transition
            ):
                left_type = " "
            elif subset_left._boundaries_included[-1] is False:
                left_type = ")"
            elif subset_left._boundaries[-2] != transition:
                left_type = "]"
            else:
                left_type = "[]"

            # Determine the type of the right boundary.
            if (
                subset_right.is_empty()
                or subset_right._boundaries[0] != transition
            ):
                right_type = " "
            elif subset_right._boundaries_included[0] is False:
                right_type = "("
            elif subset_right._boundaries[1] != transition:
                right_type = "["
            else:
                right_type = "[]"

            # Determine what part of the subset to keep.
            # fmt: off
            if (
                left_type == " " or right_type == " "  # case (1)
                or left_type == ")" and right_type == "("  # case (4)
            ):
                idx_end = len(subset_left._boundaries) - 1
                new_idx_start = 0
            elif (
                left_type == "[]" or right_type == "[]"  # case (2)
                or left_type == "]" or right_type == "["  # case (3)
            ):
                idx_end = len(subset_left._boundaries) - 2
                new_idx_start = 1
            else:
                raise RuntimeError("This should never happen.")
            # fmt: on

            boundaries.extend(subset_left._boundaries[idx_start : idx_end + 1])
            boundaries_included.extend(
                subset_left._boundaries_included[idx_start : idx_end + 1]
            )

            idx_start = new_idx_start

        boundaries.extend(subsets[-1]._boundaries[idx_start:])
        boundaries_included.extend(
            subsets[-1]._boundaries_included[idx_start:]
        )

        return boundaries, boundaries_included

    @staticmethod
    def _concat_subsets(
        subsets: list["NumberSet"], transitions: list[float]
    ) -> "NumberSet":
        """Concatenate extracted subsets.

        See __concat_subsets_helper() for more information.
        """
        return NumberSet.__direct_init(
            *NumberSet.__concat_subsets_helper(subsets, transitions)
        )

    def _concat_subsets_ip(
        self, subsets: list["NumberSet"], transitions: list[float]
    ) -> None:
        """Concatenate extracted subsets.

        See __concat_subsets_helper() for more information.
        """
        self._boundaries, self._boundaries_included = (
            NumberSet.__concat_subsets_helper(subsets, transitions)
        )

    def __perform_boolean_operation_helper(
        self,
        other: "NumberSet",
        is_symmetric: bool,
        convert_to_if_other_exists: Literal["00", "01", "10", "11"],
        convert_to_if_other_empty: Literal["00", "01", "10", "11"],
    ) -> tuple[list[float], list[bool]]:
        """Perform a boolean operation between two NumberSets.

        is_symmetric indicates whether the operation is symmetric, like "and",
        "or", or "xor". Operations like "subtract" are not symmetric.

        convert_to_if_other_exists indicates what to convert the set to if
        other contains any numbers or intervals.
        convert_to_if_other_empty indicates what to convert the set to if other
        does not contain any numbers or intervals.
        """
        # We are going to extract subsets between the boundaries of other, and
        # then concatenate them to form the new set.
        # Because extracting a subset is expensive, we want to cut as few
        # times as possible. If the operation is symmetric, we can swap the
        # sets if other contains more components (thus saving cuts).
        if is_symmetric and len(other._boundaries) > len(self._boundaries):
            return other.__perform_boolean_operation_helper(
                self,
                is_symmetric,
                convert_to_if_other_exists,
                convert_to_if_other_empty,
            )

        # If other is empty, we can return self directly by performing the
        # requested conversion.
        if other.is_empty():
            return self.__extract_subset_helper(
                -inf, inf, True, True, convert_to_if_other_empty
            )

        # Look up where the boundaries of the other set are in the current set.
        boundaries = other._boundaries.copy()
        boundaries_included = other._boundaries_included.copy()
        idcs = [
            bisect_left(self._boundaries, boundary) for boundary in boundaries
        ]
        exists_on_even_idcs = True
        if boundaries[0] != -inf:
            exists_on_even_idcs = False
            boundaries.insert(0, -inf)
            boundaries_included.insert(0, False)  # will be inverted later
            idcs.insert(0, 0)
        if boundaries[-1] != inf:
            boundaries.append(inf)
            boundaries_included.append(False)  # will be inverted later
            idcs.append(len(self._boundaries))

        # Cut out the subsets between the boundaries of the other set.
        subsets = [
            self._extract_subset(
                boundaries[i],
                boundaries[i + 1],
                (
                    boundaries_included[i]
                    if i % 2 != exists_on_even_idcs
                    else not boundaries_included[i]
                ),
                (
                    boundaries_included[i + 1]
                    if i % 2 != exists_on_even_idcs
                    else not boundaries_included[i + 1]
                ),
                (
                    convert_to_if_other_exists
                    if i % 2 != exists_on_even_idcs
                    else convert_to_if_other_empty
                ),
                idcs[i],
                idcs[i + 1],
            )
            for i in range(len(boundaries) - 1)
        ]

        # Concatenate the subsets.
        return NumberSet.__concat_subsets_helper(subsets, boundaries[1:-1])

    def _perform_boolean_operation(
        self,
        other: "NumberSet",
        is_symmetric: bool,
        convert_to_if_other_exists: Literal["00", "01", "10", "11"],
        convert_to_if_other_empty: Literal["00", "01", "10", "11"],
    ) -> "NumberSet":
        """Perform a boolean operation between two NumberSets.

        See __perform_boolean_operation_helper() for more information.
        """
        return NumberSet.__direct_init(
            *self.__perform_boolean_operation_helper(
                other,
                is_symmetric,
                convert_to_if_other_exists,
                convert_to_if_other_empty,
            )
        )

    def _perform_boolean_operation_ip(
        self,
        other: "NumberSet",
        is_symmetric: bool,
        convert_to_if_other_exists: Literal["00", "01", "10", "11"],
        convert_to_if_other_empty: Literal["00", "01", "10", "11"],
    ) -> None:
        """Perform a boolean operation between two NumberSets.

        See __perform_boolean_operation_helper() for more information.
        """
        self._boundaries, self._boundaries_included = (
            self.__perform_boolean_operation_helper(
                other,
                is_symmetric,
                convert_to_if_other_exists,
                convert_to_if_other_empty,
            )
        )

    def is_empty(self) -> bool:
        """Check if the set is empty."""
        return not self._boundaries

    def is_number(self) -> bool:
        """Check if the set represents a single number."""
        return (
            self.amount_components == 1
            and self._boundaries[0] == self._boundaries[1]
        )

    def is_interval(self) -> bool:
        """Check if the set represents a single interval."""
        return (
            self.amount_components == 1
            and self._boundaries[0] < self._boundaries[1]
        )

    def is_reducible(self) -> bool:
        """Check if the set can be reduced to one or no componets."""
        return self.amount_components <= 1

    def reduce(self) -> "NumberSet | Interval | float | None":
        """Reduce the set to an Interval, number or None (if empty)."""
        if not self.is_reducible():
            return self
        if self.is_empty():
            return None
        if self.is_number():
            return self._boundaries[0]
        return Interval(
            self._boundaries_included[0],
            self._boundaries[0],
            self._boundaries[1],
            self._boundaries_included[1],
        )

    def is_overlapping(self, other: "NumberSet") -> bool:
        """Check if the set contains any of the numbers in other."""
        return bool(self & other)

    def is_disjoint(self, other: "NumberSet") -> bool:
        """Check if the set contains none of the numbers in other."""
        return not (self & other)

    def is_subset(self, other: "NumberSet") -> bool:
        """Check if the set is a subset of the other set."""
        return self | other == other

    def is_superset(self, other: "NumberSet") -> bool:
        """Check if the set is a superset of the other set."""
        return self | other == self

    def is_adjacent(self, other: "NumberSet") -> bool:
        """Check if the set is adjacent to the other set."""
        return (
            len(self._boundaries) > 0
            and len(other._boundaries) > 0
            and (
                self._boundaries[-1] == other._boundaries[0]
                and self._boundaries_included[-1]
                != other._boundaries_included[0]
                or other._boundaries[-1] == self._boundaries[0]
                and other._boundaries_included[-1]
                != self._boundaries_included[0]
            )
        )

    def starts_equal(self, other: "NumberSet") -> bool:
        """Check if the sets start with the same number."""
        return (
            len(self._boundaries) > 0
            and len(other._boundaries) > 0
            and self._boundaries[0] == other._boundaries[0]
            and self._boundaries_included[0] == other._boundaries_included[0]
        )

    def ends_equal(self, other: "NumberSet") -> bool:
        """Check if the sets end with the same number."""
        return (
            len(self._boundaries) > 0
            and len(other._boundaries) > 0
            and self._boundaries[-1] == other._boundaries[-1]
            and self._boundaries_included[-1] == other._boundaries_included[-1]
        )

    def starts_left(self, other: "NumberSet") -> bool:
        """Check if the set starts left of the other set."""
        return (
            len(self._boundaries) > 0
            and len(other._boundaries) > 0
            and (
                self._boundaries[0] < other._boundaries[0]
                or self._boundaries[0] == other._boundaries[0]
                and self._boundaries_included[0]
                and not other._boundaries_included[0]
            )
        )

    def starts_right(self, other: "NumberSet") -> bool:
        """Check if the set starts right of the other set."""
        return (
            len(self._boundaries) > 0
            and len(other._boundaries) > 0
            and (
                self._boundaries[0] > other._boundaries[0]
                or self._boundaries[0] == other._boundaries[0]
                and not self._boundaries_included[0]
                and other._boundaries_included[0]
            )
        )

    def ends_left(self, other: "NumberSet") -> bool:
        """Check if the set ends left of the other set."""
        return (
            len(self._boundaries) > 0
            and len(other._boundaries) > 0
            and (
                self._boundaries[-1] < other._boundaries[-1]
                or self._boundaries[-1] == other._boundaries[-1]
                and not self._boundaries_included[-1]
                and other._boundaries_included[-1]
            )
        )

    def ends_right(self, other: "NumberSet") -> bool:
        """Check if the set ends right of the other set."""
        return (
            len(self._boundaries) > 0
            and len(other._boundaries) > 0
            and (
                self._boundaries[-1] > other._boundaries[-1]
                or self._boundaries[-1] == other._boundaries[-1]
                and self._boundaries_included[-1]
                and not other._boundaries_included[-1]
            )
        )


# GEOMETRIC CLASSES ###########################################################


class GeometricObject2D(ABC):
    """A generic two-dimensional geometric object.

    This class is mainly meant to be subclassed by other 2D geometric objects.
    It provides some abstract methods that should be implemented by the
    subclasses.

    Once created, a GeometricObject2D is immutable.
    """

    @property
    def epsilon(self) -> float:
        """The tolerance of floating point calculations.

        Defaults to 1e-6.

        This value acts as a numerical margin in various methods to account
        for floating point arithmetic errors.

        While it's possible to change epsilon for a specific instance of this
        class, all the other instances will retain the default value. Changing
        epsilon on a specific instance however could lead to some asymmetric
        behavior where symmetry would be expected, such as:

        >>> u = GeometricObject(a, b)
        >>> v = GeometricObject(a, b + 0.2)
        >>> u.epsilon = 0.5  # don't set it nearly this large
        >>>
        >>> print(u == v)  # True
        >>> print(v == u)  # False

        You'll probably never have to change epsilon from the default value,
        but in rare situations you might find that either the margin is too
        large or too small, in which case changing epsilon slightly might help
        you out.
        """
        return self._epsilon

    @epsilon.setter
    def epsilon(self, value: float) -> None:
        """Set the tolerance of floating point calculations."""
        self._epsilon = value

    @abstractmethod
    def __init__(self) -> None:
        """Create a GeometricObject2D instance."""
        self._epsilon = 1e-6
        super().__init__()

    @abstractmethod
    def __iter__(self) -> Iterator[Any]:
        """Iterate over the object's components."""
        ...

    def __getitem__(self, idx: int) -> Any:
        """Get the component at the given index."""
        for i, component in enumerate(self):
            if i == idx:
                return component
        raise IndexError(f"{self.__class__.__name__} index out of range.")

    @override
    def __repr__(self) -> str:
        """Represent the object as a string."""
        return f"{self.__class__.__name__}({', '.join(map(repr, self))})"

    @override
    def __hash__(self) -> int:
        """Create a hash id for the geometric object."""
        return hash(tuple(self))

    @override
    def __eq__(self, other: object) -> bool:
        """Compare two geometric objects.

        Warning: this function internally uses a threshold of self.epsilon to
        determine if the objects are equal.
        """
        if not isinstance(other, self.__class__):
            return False
        return all(
            abs(component_self - component_other) < self._epsilon
            for component_self, component_other in zip(self, other)
        )

    def __matmul__(self, other: Matrix2D) -> "Matrix2D":
        """Transform the object by the given matrix.

        Returns a new object that is transformed by the given matrix.
        """
        return self.transform(other)

    @abstractmethod
    def copy(self) -> "Matrix2D":
        """Create a copy of this object.

        Returns a new object that is a copy of itself.
        """
        ...

    @abstractmethod
    def translate(self, v: "Vector2D") -> "Matrix2D":
        """Translate the object by the given amount.

        Returns a new object that is translated by the given amount.
        """
        ...

    @abstractmethod
    def scale(self, x: float, y: float) -> "Matrix2D":
        """Scale the object by the given amount.

        Returns a new object that is scaled by the given amount.
        """
        ...

    @abstractmethod
    def rotate(self, angle: float) -> "Matrix2D":
        """Rotate the object by the given angle in radians.

        Returns a new object that is rotated by the given angle.
        """
        ...

    @abstractmethod
    def reflect(self, v: "Vector2D") -> "Matrix2D":
        """Reflect the object over the given vector.

        Returns a new object that is reflected over the given vector.
        """
        ...

    @abstractmethod
    def transform(self, matrix: Matrix2D) -> "Matrix2D":
        """Transform the object by the given matrix.

        Returns a new object that is transformed by the given matrix.
        """
        ...

    @abstractmethod
    def closest_to(self, v: "Vector2D") -> "Vector2D":
        """Get the point on this object closest to the given point.

        Returns a Vector2D that is closest to the given point.
        """
        ...

    @abstractmethod
    def distance_to(self, v: "Vector2D") -> float:
        """Calculate the shortest distance to the given point.

        Returns the shortest distance to the given point.
        """
        ...

    @abstractmethod
    def intersections_x(
        self, x: float
    ) -> NumberSet | Iterable[float | Interval]:
        """Calculate the y-coordinates of the intersections at the given x.

        If the amount of numbers/intervals that represent the y-coordinates of
        the intersections is finite, a NumberSet is returned. If there are
        infinitely many numbers/intervals, an infinite iterable of them is
        returned.
        """
        ...

    @abstractmethod
    def intersections_y(
        self, y: float
    ) -> NumberSet | Iterable[float | Interval]:
        """Calculate the x-coordinates of the intersections at the given y.

        If the amount of numbers/intervals that represent the x-coordinates of
        the intersections is finite, a NumberSet is returned. If there are
        infinitely many numbers/intervals, an infinite iterable of them is
        returned.
        """
        ...


class Vector2D(GeometricObject2D):
    """A two-dimensional vector.

    Methods and documentation are based on the Vector2 class of Pygame.
    See https://www.pygame.org/docs/ref/math.html#pygame.math.Vector2 for
    more information.

    Once created, a Vector2D is immutable.

    Warning: this class is not a drop-in replacement for Pygame's Vector2
    class. It contains a lot of methods that are not in Pygame's Vector2 class
    and vice versa. In addition, some methods have different names or different
    behavior.
    """

    @property
    def x(self) -> float:
        """The x-coordinate of the vector."""
        return self._x

    @property
    def y(self) -> float:
        """The y-coordinate of the vector."""
        return self._y

    def __init__(self, x: float, y: float) -> None:
        """Create a Vector2D instance."""
        self._x = x
        self._y = y
        super().__init__()

    @override
    def __iter__(self) -> Iterator[float]:
        """Iterate over the vector's components."""
        return iter((self._x, self._y))

    def __bool__(self) -> bool:
        """Check if the vector is not the zero vector.

        Returns True if the vector is not the zero vector.

        Warning: this function internally uses a threshold of self.epsilon to
        determine if the vector is not the zero vector.
        """
        return abs(self._x) > self._epsilon or abs(self._y) > self._epsilon

    def __neg__(self) -> "Vector2D":
        """Negate the vector."""
        return Vector2D(-self._x, -self._y)

    def __add__(self, other: "Vector2D") -> "Vector2D":
        """Add the given vector to this one."""
        return Vector2D(self._x + other._x, self._y + other._y)

    def __sub__(self, other: "Vector2D") -> "Vector2D":
        """Subtract the given vector from this one."""
        return Vector2D(self._x - other._x, self._y - other._y)

    def __mul__(self, other: float) -> "Vector2D":
        """Multiply the vector by the given scalar."""
        return Vector2D(self._x * other, self._y * other)

    def __rmul__(self, other: float) -> "Vector2D":
        """Multiply the given scalar by the vector."""
        return Vector2D(other * self._x, other * self._y)

    def __truediv__(self, other: float) -> "Vector2D":
        """Perform floating point division by the given scalar."""
        return Vector2D(self._x / other, self._y / other)

    def __floordiv__(self, other: float) -> "Vector2D":
        """Perform integer division by the given scalar."""
        return Vector2D(self._x // other, self._y // other)

    @override
    def copy(self) -> "Vector2D":
        """Create a copy of this vector.

        Returns a new vector that is a copy of itself.
        """
        return Vector2D(self._x, self._y)

    @override
    def translate(self, v: "Vector2D") -> "Vector2D":
        """Translate the vector by the given amount.

        Returns a new vector that is translated by the given amount.
        """
        return Vector2D(self._x + v._x, self._y + v._y)

    @override
    def scale(self, x: float, y: float) -> "Vector2D":
        """Scale the vector by the given amount.

        Returns a new vector that is scaled by the given amount.
        """
        return Vector2D(self._x * x, self._y * y)

    @override
    def rotate(self, angle: float) -> "Vector2D":
        """Rotate the vector by the given angle in radians.

        Returns a vector which has the same length as self but is rotated
        counterclockwise by the given angle in radians.

        Warning: this function internally uses a threshold of self.epsilon to
        determine if the angle is a multiple of pi/2.
        """
        angle %= 2 * pi
        if angle < self._epsilon or 2 * pi - angle < self._epsilon:
            return Vector2D(self._x, self._y)
        if abs(angle - pi / 2) < self._epsilon:
            return Vector2D(-self._y, self._x)
        if abs(angle - pi) < self._epsilon:
            return Vector2D(-self._x, -self._y)
        if abs(angle - 3 * pi / 2) < self._epsilon:
            return Vector2D(self._y, -self._x)
        return Vector2D(
            self._x * cos(angle) - self._y * sin(angle),
            self._x * sin(angle) + self._y * cos(angle),
        )

    @override
    def reflect(self, v: "Vector2D") -> "Vector2D":
        """Reflect the vector over the given vector.

        Returns a new vector that is reflected over the given vector.
        """
        return 2 * v - self

    @override
    def transform(self, matrix: Matrix2D) -> "Vector2D":
        """Transform the vector by the given matrix.

        Returns a new vector that is transformed by the given matrix.
        """
        return Vector2D(
            self._x * matrix[0] + self._y * matrix[1],
            self._x * matrix[2] + self._y * matrix[3],
        )

    @override
    def closest_to(self, v: "Vector2D") -> "Vector2D":
        """Return the closest point on the vector to the given vector.

        Returns a copy of the vector itself. This method is a mandatory
        override for GeometricObject2D.
        """
        return self.copy()

    @override
    def distance_to(self, v: "Vector2D") -> float:
        """Calculate the Euclidean distance to the given vector."""
        return dist((self._x, self._y), (v._x, v._y))

    @override
    def intersections_x(
        self, x: float
    ) -> NumberSet | Iterable[float | Interval]:
        """Calculate the y-coordinates of the intersections at the given x.

        See GeometricObject2D.intersections_x() for more information.
        """
        return NumberSet(self._y) if self._x == x else NumberSet()

    @override
    def intersections_y(
        self, y: float
    ) -> NumberSet | Iterable[float | Interval]:
        """Calculate the x-coordinates of the intersections at the given y.

        See GeometricObject2D.intersections_y() for more information.
        """
        return NumberSet(self._x) if self._y == y else NumberSet()

    def distance_squared_to(self, other: "Vector2D") -> float:
        """Calculate the squared Euclidean distance to the given vector."""
        return (self._x - other._x) ** 2 + (self._y - other._y) ** 2

    def dot(self, other: "Vector2D") -> float:
        """Calculate the dot product with the given vector."""
        return self._x * other._x + self._y * other._y

    def cross(self, other: "Vector2D") -> float:
        """Calculate the cross product with the given vector.

        Returns the third component of the cross product.
        """
        return self._x * other._y - self._y * other._x

    def length(self) -> float:
        """Return the Euclidean length of the vector.

        Calculates the length of the vector as:
        sqrt(x^2 + y^2)
        """
        return hypot(self._x, self._y)

    def length_squared(self) -> float:
        """Return the squared Euclidean length of the vector.

        Calculates the length of the vector as:
        x^2 + y^2
        This is faster than .length() because it avoids the square root.
        """
        return self._x**2 + self._y**2

    def normalize(self) -> "Vector2D":
        """Normalize the vector.

        Returns a vector with the same direction as self but with length equal
        to 1. If the vector is the zero vector (i.e. has length 0 thus no
        direction) a ValueError is raised.

        Warning: this function internally uses a threshold of self.epsilon to
        determine if the vector is not the zero vector.
        """
        if not self:
            raise ValueError("Cannot normalize the zero vector.")
        return self / self.length()

    def is_normalized(self) -> bool:
        """Check if the vector is normalized.

        Returns True if the vector has length equal to 1. Otherwise it returns
        False.

        Warning: this function internally uses a threshold of self.epsilon to
        determine if the vector is normalized.
        """
        return abs(self.length_squared()) < self._epsilon

    def scale_to_length(self, length: float) -> "Vector2D":
        """Scale the vector to the given length.

        Returns a new vector that points in the same direction as self but has
        length equal to the given length. You can also scale to length 0. If
        the vector is the zero vector (i.e. has length 0 thus no direction) a
        ValueError is raised.

        Warning: this function internally uses a threshold of self.epsilon to
        determine if the vector is not the zero vector.
        """
        if not self:
            raise ValueError("Cannot scale the zero vector.")
        return length / self.length() * self

    def clamp_magnitude(
        self, min_length: float, max_length: float | None = None
    ) -> "Vector2D":
        """Clamp the vector's magnitude between min_length and max_length.

        Returns a new copy of a vector with the magnitude clamped between
        min_length and max_length. If only one argument is passed, it is taken
        to be the max_length.
        """
        if max_length is None:
            max_length = min_length
            min_length = -inf
        elif min_length > max_length:
            raise ValueError(
                "min_length must be smaller than or equal to max_length."
            )
        length = self.length()
        if length > max_length:
            return self.scale_to_length(max_length)
        if length < min_length:
            return self.scale_to_length(min_length)
        return Vector2D(self._x, self._y)

    def move_towards(self, target: "Vector2D", distance: float) -> "Vector2D":
        """Move towards the given vector by the given distance.

        Returns a Vector which is moved towards the given Vector by the given
        distance and does not overshoot past its target Vector. The first
        parameter determines the target Vector, while the second parameter
        determines the delta distance. If the distance is in the negatives,
        then it will move away from the target Vector.

        Warning: this function internally uses a threshold of self.epsilon to
        determine if the vectors are equal.
        """
        if self == target:
            return Vector2D(self._x, self._y)
        direction = target - self
        length = direction.length()
        if length <= distance:
            return Vector2D(target._x, target._y)
        return self + direction * (distance / length)

    def project(self, other: "Vector2D") -> "Vector2D":
        """Project the given vector onto this one.

        Returns the projected vector. This is useful for collision detection in
        finding the components in a certain direction (e.g. in direction of the
        wall). For a more detailed explanation see Wikipedia.

        Warning: this function internally uses a threshold of self.epsilon to
        determine if the vector onto which you project is not the zero vector.

        Warning: this function is the opposite of what is used in Pygame. In
        Pygame, you project this vector onto another.
        """
        if not self:
            raise ValueError("Cannot project onto the zero vector.")
        return self.dot(other) / self.length_squared() * self

    def angle_to(self, other: "Vector2D") -> float:
        """Calculate the angle to the given vector in radians.

        Returns the angle from self to the passed Vector that would rotate self
        to be aligned with the passed Vector without crossing over the postive
        x-axis.
        """
        return atan2(other._y, other._x) - atan2(self._y, self._x)

    def lerp(self, other: "Vector2D", t: float) -> "Vector2D":
        """Perform linear interpolation to the given vector.

        Returns a Vector which is a linear interpolation between self and the
        given Vector. The second parameter determines how far between self and
        other the result is going to be. It must be a value between 0 and 1
        where 0 means self and 1 means other will be returned.
        """
        if t < 0 or t > 1:
            raise ValueError("t must be in the range [0, 1].")
        return self + t * (other - self)

    def slerp(self, other: "Vector2D", t: float) -> "Vector2D":
        """Perform spherical interpolation to the given vector.

        Calculates the spherical interpolation from self to the given Vector.
        The second argument - often called t - must be in the range [-1, 1]. It
        parametrizes where - in between the two vectors - the result should be.
        If a negative value is given the interpolation will not take the
        complement of the shortest path.

        Warning: this function internally uses a threshold of self.epsilon to
        determine if the vectors are not the zero vector.
        """
        if t < -1 or t > 1:
            raise ValueError("t must be in the range [-1, 1].")
        if not self:
            raise ValueError("Self is the zero vector, can't slurp with it.")
        if not other:
            raise ValueError("Other is the zero vector, can't slurp with it.")
        if t == 0:
            return Vector2D(self._x, self._y)
        if t == 1:
            return Vector2D(other._x, other._y)
        angle = self.angle_to(other)
        if t < 0:
            angle = -angle
        return self.rotate(angle * t)


class Line2D(GeometricObject2D):
    """A two-dimensional line.

    The line is assumed to be infinite in both directions.

    Once created, a Line2D is immutable.
    """

    # ! Note to self: all line representations can be cheaply converted to
    # ! standard form. Thus, it is preferred to use the standard form for
    # ! further calculations inside this class.

    @property
    def a(self) -> float:
        """First coefficient of the standard form."""
        if hasattr(self, "_a"):
            pass
        elif self._native_repr == "slope_intercept":
            self._a = -self._slope
        elif self._native_repr == "vectors":
            self._a = self._v2.y
        elif self._native_repr == "polar":
            self._a = cos(self._theta)
        else:
            raise ValueError("Invalid native representation.")
        return self._a

    @property
    def b(self) -> float:
        """Second coefficient of the standard form."""
        if hasattr(self, "_b"):
            pass
        elif self._native_repr == "slope_intercept":
            self._b = 1
        elif self._native_repr == "vectors":
            self._b = -self._v2.x
        elif self._native_repr == "polar":
            self._b = sin(self._theta)
        else:
            raise ValueError("Invalid native representation.")
        return self._b

    @property
    def c(self) -> float:
        """Third coefficient of the standard form."""
        if hasattr(self, "_c"):
            pass
        elif self._native_repr == "slope_intercept":
            self._c = self._intercept
        elif self._native_repr == "vectors":
            self._c = self._v1.cross(self._v2)
        elif self._native_repr == "polar":
            self._c = self._r
        else:
            raise ValueError("Invalid native representation.")
        return self._c

    @property
    def slope(self) -> float:
        """Slope of the line."""
        if hasattr(self, "_slope"):
            pass
        elif self._native_repr == "standard":
            if self._b == 0:
                self._slope = inf  # actually also -inf but we ignore that here
            else:
                self._slope = -self._a / self._b
        elif self._native_repr == "vectors":
            if self._v2.x == 0:
                self._slope = inf  # actually also -inf but we ignore that here
            else:
                self._slope = self._v2.y / self._v2.x
        elif self._native_repr == "polar":
            if self._theta == 0:
                self._slope = inf  # actually also -inf but we ignore that here
            else:
                self._slope = -1 / tan(self._theta)
        else:
            raise ValueError("Invalid native representation.")
        return self._slope

    @property
    def intercept(self) -> float:
        """Y-intercept of the line."""
        if hasattr(self, "_intercept"):
            pass
        elif self._native_repr == "standard":
            if self._b == 0:
                self._intercept = inf  # mathematically incorrect but useful
            else:
                self._intercept = self._c / self._b
        elif self._native_repr == "vectors":
            if self._v2.x == 0:
                self._intercept = inf  # mathematically incorrect but useful
            else:
                self._intercept = (
                    self._v1.y - self._v2.y / self._v2.x * self._v1.x
                )
        elif self._native_repr == "polar":
            if self._theta == 0:
                self._intercept = inf  # mathematically incorrect but useful
            else:
                self._intercept = self._r / sin(self._theta)
        else:
            raise ValueError("Invalid native representation.")
        return self._intercept

    @property
    def v1(self) -> Vector2D:
        """Support vector (Dutch: steunvector) of the line."""
        if hasattr(self, "_v1"):
            pass
        elif self._native_repr == "standard":
            if self._b == 0:
                self._v1 = Vector2D(self._c / self._a, 0)
            elif self._a == 0:
                self._v1 = Vector2D(0, self._c / self._b)
            else:
                self._v1 = Vector2D(self._c / self._a, self._c / self._b)
        elif self._native_repr == "slope_intercept":
            self._v1 = Vector2D(0, self._intercept)
        elif self._native_repr == "polar":
            self._v1 = Vector2D(
                self._r * cos(self._theta), self._r * sin(self._theta)
            )
        else:
            raise ValueError("Invalid native representation.")
        return self._v1

    @property
    def v2(self) -> Vector2D:
        """Direction vector (Dutch: richtingsvector) of the line."""
        if hasattr(self, "_v2"):
            pass
        elif self._native_repr == "standard":
            self._v2 = Vector2D(-self._b, self._a)
        elif self._native_repr == "slope_intercept":
            self._v2 = Vector2D(1, self._slope)
        elif self._native_repr == "polar":
            self._v2 = Vector2D(-sin(self._theta), cos(self._theta))
        else:
            raise ValueError("Invalid native representation.")
        return self._v2

    @property
    def normal(self) -> Vector2D:
        """Normal vector of the line (always normalized)."""
        if hasattr(self, "_normal"):
            pass
        elif self._native_repr == "standard":
            self._normal = Vector2D(self._a, self._b)
        elif self._native_repr == "slope_intercept":
            self._normal = Vector2D(-self._slope, 1)
        elif self._native_repr == "vectors":
            self._normal = Vector2D(-self._v2.y, self._v2.x)
        elif self._native_repr == "polar":
            self._normal = Vector2D(cos(self._theta), sin(self._theta))
        else:
            raise ValueError("Invalid native representation.")
        self._normal = self._normal.normalize()
        return self._normal

    @property
    def r(self) -> float:
        """Smallest distance from the origin to the line."""
        if hasattr(self, "_r"):
            pass
        elif self._native_repr == "standard":
            self._r = abs(self._c) / hypot(self._a, self._b)
        elif self._native_repr == "slope_intercept":
            self._r = abs(self._intercept) / sqrt(1 + self._slope**2)
        elif self._native_repr == "vectors":
            self._r = abs(self._v1.cross(self._v2)) / self._v2.length()
        else:
            raise ValueError("Invalid native representation.")
        return self._r

    @property
    def theta(self) -> float:
        """Angle between the x-axis and the closest vector to the line."""
        if hasattr(self, "_theta"):
            pass
        elif self._native_repr == "standard":
            if self._c >= 0 and self._b >= 0 or self._c < 0 and self._b < 0:
                # intercept >= 0 and 0 <= theta < pi
                if self._b >= 0:
                    self._theta = atan2(self._b, self._a)
                else:
                    self._theta = atan2(-self._b, -self._a)
            # intercept < 0 and pi <= theta < 2 * pi
            elif self._b >= 0:
                self._theta = atan2(-self._b, -self._a) + 2 * pi
            else:
                self._theta = atan2(self._b, self._a) + 2 * pi
        elif self._native_repr == "slope_intercept":
            if self._intercept >= 0:
                # intercept >= 0 and 0 <= theta < pi
                self._theta = atan2(1, -self._slope)
            # intercept < 0 and pi <= theta < 2 * pi
            else:
                self._theta = atan2(-1, self._slope) + 2 * pi
        elif self._native_repr == "vectors":
            if self._v1.cross(self._v2) <= 0:
                # intercept >= 0 and 0 <= theta < pi
                if self._v2.x >= 0:
                    self._theta = atan2(self._v2.x, -self._v2.y)
                else:
                    self._theta = atan2(-self._v2.x, self._v2.y)
            # intercept < 0 and pi <= theta < 2 * pi
            elif self._v2.x >= 0:
                self._theta = atan2(-self._v2.x, self._v2.y) + 2 * pi
            else:
                self._theta = atan2(self._v2.x, -self._v2.y) + 2 * pi
        else:
            raise ValueError("Invalid native representation.")
        return self._theta

    def __init__(self, _native_repr: str, *args: float | Vector2D) -> None:
        """Create a Line2D instance.

        Only meant for internal use, do not call this function directly.
        Please use one of the from_* methods instead.
        """
        self._native_repr = _native_repr
        if _native_repr == "standard" and len(args) == 3:
            args = cast(tuple[float, float, float], args)
            self._a, self._b, self._c = args
        elif _native_repr == "slope_intercept" and len(args) == 2:
            args = cast(tuple[float, float], args)
            self._slope, self._intercept = args
        elif _native_repr == "vectors" and len(args) == 2:
            args = cast(tuple[Vector2D, Vector2D], args)
            self._v1, self._v2 = args
        elif _native_repr == "polar" and len(args) == 2:
            args = cast(tuple[float, float], args)
            self._r, self._theta = args
        else:
            raise ValueError("Invalid native representation.")
        super().__init__()

    @classmethod
    def from_standard(cls, a: float, b: float, c: float) -> "Line2D":
        """Initialize the line from its standard representation.

        Represents the line as the equation:
        a * x + b * y = c

        Args:
            a: First coefficient of the line. When you change this parameter,
                the line will rotate around its y-intercept.
            b: Second coefficient of the line. When you change this parameter,
                the line will rotate around its x-intercept.
            c: Third coefficient of the line. When you change this parameter,
                the line will shift left or right.
        """
        if a == 0 and b == 0:
            raise ValueError("a and b cannot both be 0.")
        return cls("standard", a, b, c)

    @classmethod
    def from_slope_intercept(cls, slope: float, intercept: float) -> "Line2D":
        """Initialize the line from its slope and intercept.

        Represents the line as the equation:
        y = slope * x + intercept

        Warning: be aware that this representation is not suitable for vertical
        lines, because they have an infinite slope. If you need to represent a
        vertical line, use another from_* method instead.

        Args:
            slope: Slope of the line.
            intercept: Y-intercept of the line.
        """
        if isinf(slope):
            raise ValueError("slope must not be infinite.")
        return cls("slope_intercept", slope, intercept)

    @classmethod
    def from_vectors(cls, v1: Vector2D, v2: Vector2D) -> "Line2D":
        """Initialize the line from two vectors.

        Represents the line as the equation:
        (x, y) = v1 + t * v2

        Args:
            v1: Support vector (Dutch: steunvector).
            v2: Direction vector (Dutch: richtingsvector).
        """
        if not v2:
            raise ValueError("v2 must not be the zero vector.")
        return cls("vectors", v1, v2)

    @classmethod
    def from_points(cls, p1: Vector2D, p2: Vector2D) -> "Line2D":
        """Initialize the line from two points.

        Represents the line as the equation:
        (x, y) = p1 + t * (p2 - p1)

        Args:
            p1: A point on the line.
            p2: Another point on the line.
        """
        if p1 == p2:
            raise ValueError("p1 and p2 must not be the same point.")
        return cls.from_vectors(p1, p2 - p1)

    @classmethod
    def from_polar(cls, r: float, theta: float) -> "Line2D":
        """Initialize the line from its polar representation.

        Represents the line as the equation:
        x * cos(theta) + y * sin(theta) = r

        Args:
            r: Smallest distance from the origin to the line.
            theta: Angle between the x-axis and the closest vector to the line.
        """
        if r < 0:
            raise ValueError("r must be non-negative.")
        return cls("polar", r, theta % (2 * pi))

    @override
    def __iter__(self) -> Iterator[float | Vector2D]:
        """Iterate over the parameters of the line."""
        if self._native_repr == "standard":
            yield self._a
            yield self._b
            yield self._c
        if self._native_repr == "slope_intercept":
            yield self._slope
            yield self._intercept
        if self._native_repr == "vectors":
            yield self._v1
            yield self._v2
        if self._native_repr == "polar":
            yield self._r
            yield self._theta

    @override
    def __repr__(self) -> str:
        """Represent the line as a string."""
        return (
            f"{self.__class__.__name__}.from_{self._native_repr}("
            + ", ".join(map(repr, self))
        )

    @override
    def __hash__(self) -> int:
        """Returns a hash of the line."""
        return hash(tuple(self))

    @override
    def __eq__(self, other: object) -> bool:
        """Check if two lines are equal.

        Warning: if the lines being compared are in different native
        representations, this function will convert both lines to standard form
        and then compare the parameters. To avoid floating point errors, the
        lines will in this scenario be considered equal if the parameters are
        equal within a threshold of self.epsilon.
        """
        if not isinstance(other, Line2D):
            return False
        # If the native representations are the same, we can just compare the
        # parameters. Otherwise, we convert both lines to standard form and
        # compare the parameters.
        return (
            self._native_repr == other._native_repr
            and tuple(self) == tuple(other)
            or abs(self.a - other.a) < self.epsilon
            and abs(self.b - other.b) < self.epsilon
            and abs(self.c - other.c) < self.epsilon
        )

    def __contains__(self, point: Vector2D) -> bool:
        """Check if a point is on the line.

        Warning: this function internally uses a threshold of self.epsilon to
        determine if the point is on the line.
        """
        return abs(self.a * point.x + self.b * point.y - self.c) < self.epsilon

    @override
    def copy(self) -> "Line2D":
        """Create a copy of this line.

        Returns a new line that is a copy of itself.
        """
        return Line2D(self._native_repr, *self)

    @override
    def translate(self, v: Vector2D) -> "Line2D":
        """Translate the line by the given vector.

        Returns a new line that is translated by the given vector.
        """
        return self.from_vectors(self.v1 + v, self.v2)

    @override
    def scale(self, x: float, y: float) -> "Line2D":
        """Scale the line by the given amount.

        Returns a new line that is scaled by the given factor in the x and y.
        """
        return self.from_standard(self.a / x, self.b / y, self.c)

    @override
    def rotate(self, angle: float) -> "Line2D":
        """Rotate the line by the given angle.

        Returns a new line that is rotated by the given angle.
        """
        return self.from_polar(self.r, self.theta + angle)

    @override
    def reflect(self, v: Vector2D) -> "Line2D":
        """Reflect the line over the given point.

        Returns a new line that is the reflection of the current line over the
        given point.
        """
        return self.from_vectors(
            self.v1 + 2 * self.normal.project(v - self.v1), self.v2
        )

    @override
    def transform(self, matrix: Matrix2D) -> "Line2D":
        """Transform the line by the given matrix.

        Returns a new line that is transformed by the given matrix.
        """
        return self.from_points(matrix @ self.v1, matrix @ self.v2)

    @override
    def closest_to(self, v: Vector2D) -> Vector2D:
        """Get the point on this line that is closest to the given point.

        Returns a Vector2D that is closest to the given point.
        """
        return self.v1 + self.v2.project(v - self.v1)

    @override
    def distance_to(self, v: Vector2D) -> float:
        """Calculate the distance from the line to the given point.

        Returns the distance from the line to the given point. The distance is
        positive if the point is on the left side of the line (when looking
        from v1 to v1 + v2) and negative if the point is on the right side of
        the line.
        """
        return self.normal.dot(v - self.v1)

    @override
    def intersections_x(
        self, x: float
    ) -> NumberSet | Iterable[float | Interval]:
        """Calculate the y-coordinates of the intersections at the given x.

        See GeometricObject2D.intersections_x() for more information.
        """
        if self.b == 0:  # the line is vertical
            if self.a * x == self.c:
                return NumberSet(Interval(True, -inf, inf, True))
            return NumberSet()
        return NumberSet((self.c - self.a * x) / self.b)

    @override
    def intersections_y(
        self, y: float
    ) -> NumberSet | Iterable[float | Interval]:
        """Calculate the x-coordinates of the intersections at the given y.

        See GeometricObject2D.intersections_y() for more information.
        """
        if self.a == 0:  # the line is horizontal
            if self.b * y == self.c:
                return NumberSet(Interval(True, -inf, inf, True))
            return NumberSet()
        return NumberSet((self.c - self.b * y) / self.a)

    def reflect_point(self, point: Vector2D) -> Vector2D:
        """Reflect a point over the line.

        Returns the point that is the reflection of the given point over the
        current line.
        """
        return self.closest_to(point) * 2 - point

    def angle_to(self, other: "Line2D") -> float:
        """Calculate the smallest absolute angle to the given line in radians.

        Returns the smallest angle from self to the given line that would
        rotate self counterclockwise to be aligned with the passed line. The
        angle is per definition always between 0 and pi: 0 <= angle < pi.
        """
        return (other.theta - self.theta) % pi

    def is_parallel(self, other: "Line2D") -> bool:
        """Check if two lines are parallel.

        Checks if the two lines are parallel. If the lines are parallel, they
        will never intersect.
        """
        return abs(self.normal.dot(other.normal)) == 1

    def is_perpendicular(self, other: "Line2D") -> bool:
        """Check if two lines are perpendicular.

        Checks if the two lines are perpendicular. If the lines are
        perpendicular, they will always intersect.
        """
        return abs(self.normal.dot(other.normal)) == 0

    def is_intersecting(self, other: "Line2D") -> bool:
        """Check if two lines are intersecting.

        Checks if the two lines are intersecting. If the lines intersect, they
        will intersect in exactly one point.
        """
        return not self.is_parallel(other)

    def parallel_line(self, point: Vector2D) -> "Line2D":
        """Get the parallel line that goes through the given point.

        Returns a new line that is parallel to the current line and goes
        through the given point.
        """
        return self.from_vectors(point, self.v2)

    def perpendicular_line(self, point: Vector2D) -> "Line2D":
        """Get the perpendicular line that goes through the given point.

        Returns a new line that is perpendicular to the current line and goes
        through the given point.
        """
        return self.from_vectors(point, self.normal)

    def intersection_point(self, other: "Line2D") -> Vector2D | None:
        """Get the intersection point of the line with the given line.

        Returns the intersection point of the two lines. If the lines are
        parallel, this function will raise a ValueError.
        """
        if not self.is_intersecting(other):
            raise ValueError("The lines are parallel, they do not intersect.")
        # https://www.cuemath.com/geometry/intersection-of-two-lines/
        divider = self.a * other.b - other.a * self.b
        return Vector2D(
            (self.b * other.c - other.b * self.c) / divider,
            (self.c * other.a - other.c * self.a) / divider,
        )


# TODO Move the below functions to a separate class some day.


def get_arc_length(radius: float, angle_rad: float) -> float:
    """Returns the length of an arc of a circle.

    Args:
        radius: radius of the circle.
        angle_rad: angle of the arc in radians.

    Returns:
        The length of the arc.
    """
    return radius * angle_rad


def get_line_length(x1: float, y1: float, x2: float, y2: float) -> float:
    return dist((x1, x2), (y1, y2))


def get_circle_interceptions(
    x1: float, y1: float, r1: float, x2: float, y2: float, r2: float
) -> list[tuple[float, float]]:
    # x1, y1: coordinates of the center of the first circle.
    # r1: radius of the first circle.
    # x2, y2: coordinates of the center of the second circle.
    # r2: radius of the second circle.
    # Returns the coordinates of the interceptions of the two circles.

    if x1 == x2 and y1 == y2 and r1 == r2:
        # The circles are the same.
        return []
    if get_line_length(x1, y1, x2, y2) > r1 + r2:
        # The circles do not intercept.
        return []
    if get_line_length(x1, y1, x2, y2) < abs(r1 - r2):
        # One circle is inside the other, and they do not intercept.
        return []
    if get_line_length(x1, y1, x2, y2) == r1 + r2:
        # The circles are tangent to each other.
        interception_factor = r1 / (r1 + r2)
        return [(
            x1 + (x2 - x1) * interception_factor,
            y1 + (y2 - y1) * interception_factor,
        )]
    if get_line_length(x1, y1, x2, y2) == abs(r1 - r2):
        # One circle is inside the other, and they are tangent to each other.
        interception_factor = r1 / (r1 + r2)
        return [(
            x1 + (x2 - x1) * interception_factor,
            y1 + (y2 - y1) * interception_factor,
        )]
    # The circles intercept in two points.
    # https://math.stackexchange.com/questions/256100/
    # how-can-i-find-the-points-at-which-two-circles-intercept says:
    d = get_line_length(x1, y1, x2, y2)
    l = (r1**2 - r2**2 + d**2) / (2 * d)
    h = sqrt(r1**2 - l**2)

    x_base = x1 + l / d * (x2 - x1)
    y_base = y1 + l / d * (y2 - y1)
    x_extra = h / d * (y2 - y1)
    y_extra = h / d * (x2 - x1)
    return [
        (x_base + x_extra, y_base - y_extra),
        (x_base - x_extra, y_base + y_extra),
    ]


def get_tangent_point(
    xl: float, yl: float, xc: float, yc: float, r: float
) -> list[tuple[float, float]]:
    # xl, yl: coordinates of a point on the line.
    # xc, yc: coordinates of the center of the circle.
    # r: radius of the circle.
    # Returns the coordinates of the tangent point on the circle.

    # https://www.hhofstede.nl/modules/raaklijncirkel.htm says:
    # Raaklijnen vanaf een punt buiten de cirkel.
    # Kijk eens naar het plaatje hiernaast. Daarin zie je dat er vanaf een
    # punt P buiten een cirkel twee raaklijnen aan die cirkel te tekenen zijn.
    # Hoe zijn die te vinden?
    # Omdat de raaklijnen loodrecht staan op de lijnen MR, zijn de driehoeken
    # PMR1 en PMR2 hiernaast rechthoekig.
    # Maar als je dan de lengte van PM weet en ook MR1 (de straal van de
    # cirkel), dan kun je met Pythagoras PR1 berekenen.
    # Omdat PR1 = PR2 liggen beide punten R op een cirkel met middelpunt P en
    # straal PR1.
    # Nou, simpel: stel een vergelijking van die tweede cirkel op, en bereken
    # de snijpunten daarvan met de eerste cirkel.

    # 1.  Bereken PR met Pythagoras.
    mr = r
    pm = get_line_length(xl, yl, xc, yc)
    pr = sqrt(pm**2 - mr**2)

    # 2.  Snij de cirkel met middelpunt P en straal PR met de gegeven cirkel.
    return get_circle_interceptions(xc, yc, r, xl, yl, pr)


def get_angle_general_triangle(a: float, b: float, c: float) -> float:
    # a: length of the side opposite to the angle we want to calculate.
    # b: length of one side adjacent to the angle we want to calculate.
    # c: length of the remaining side.
    # Returns the angle in radians.

    # https://www.math4all.nl/venster/bekijk/driehoeksmeting/41/1 says:
    # Teken de hoogtelijn uit C, de lengte ervan is b*sin(alpha), maar ook
    # a*sin(beta).
    # Zo krijg je het begin van de sinusregel:
    # a*sin(alpha) = b*sin(beta) = c*sin(gamma).
    # Met behulp van de stelling van Pythagoras vind je de cosinusregel:
    # a^2 = b^2 + c^2 - 2*bc*cos(alpha). Daar zijn drie varianten van.
    # Met deze twee regels kun je vanuit drie gegevens alle zijden en hoeken
    # van een willekeurige driehoek berekenen. Dat heet "triangulatie"
    # (driehoeksmeting).
    return acos((b**2 + c**2 - a**2) / (2 * b * c))
