from collections.abc import Sequence


def bisect_left_using_right(
    a: Sequence[float], x: float, bisect_right_idx: int
) -> int:
    """Return bisect_left(a, x) using only bisect_right(a, x).

    This function is only faster than bisect_left(a, x) if a contains no more
    than log(|a|) x's.
    """
    for bisect_left_idx in range(bisect_right_idx - 1, -1, -1):
        if a[bisect_left_idx] < x:
            return bisect_left_idx + 1
    return 0


def bisect_right_using_left(
    a: Sequence[float], x: float, bisect_left_idx: int
) -> int:
    """Return bisect_right(a, x) using only bisect_left(a, x).

    This function is only faster than bisect_right(a, x) if a contains no more
    than log(|a|) x's.
    """
    for bisect_right_idx in range(bisect_left_idx, len(a)):
        if a[bisect_right_idx] > x:
            return bisect_right_idx
    return len(a)
