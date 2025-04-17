import itertools
import operator
from collections.abc import Iterable, Iterator
from math import ceil, exp, floor, gcd, isqrt, pi, sqrt
from typing import TypeVar

NumberT = TypeVar("NumberT", int, float)  # used for typing dependent vars


def floor_to_multiple(x: float, base: NumberT) -> NumberT:
    """Floor a number to the nearest multiple of a base number.

    Args:
        x: The number to floor.
        base: The base to floor to.

    Returns:
        The floored number.
    """
    # This is faster than floor(x // base) and int(x / base) and int(x // base)
    # for some reason, I tested it with timeit.
    # Unfortunately, float(x) // base returns a float, even is base is an int.
    return floor(x / base) * base


def ceil_to_multiple(x: float, base: NumberT) -> NumberT:
    """Ceil a number to the nearest multiple of a base number.

    Args:
        x: The number to ceil.
        base: The base to ceil to.

    Returns:
        The ceiled number.
    """
    return ceil(x / base) * base


def round_to_multiple(x: float, base: NumberT) -> NumberT:
    """Round a number to the nearest multiple of a base number.

    Args:
        x: The number to round.
        base: The base to round to.

    Returns:
        The rounded number.
    """
    return round(x / base) * base


def cumprod(iterable: Iterable[NumberT]) -> Iterator[NumberT]:
    """Calculate the cumulative product of an iterable.

    Args:
        iterable: The iterable to calculate the cumulative product of.

    Yields:
        The cumulative product of the iterable.
    """
    return itertools.accumulate(iterable, operator.mul)


def extended_gcd(a: int, b: int) -> tuple[int, int, int]:
    """Calculate (gcd, s, t) in the equation a*s + b*t = gcd(a, b).

    Note: You should use gcd() from the math library if you're only interested
    in the gcd! It is much faster than an iterative implementation in Python.

    Args:
        a: Variable a in the equation.
        b: Variable b in the equation.

    Returns:
        Tuple containing:
        - The greatest common divisor of a and b.
        - The coefficient s in the equation.
        - The coefficient t in the equation.
    """
    old_r, r = a, b
    old_s, s = 1, 0
    while r:
        q = old_r // r
        old_r, r = r, old_r - r * q
        old_s, s = s, old_s - s * q
    t = (old_r - old_s * a) // b if b else 0
    return old_r, old_s, t


def prime_factors(n: int) -> Iterator[int]:
    """Generate all prime factors of the given number.

    Complexity: O(sqrt(n))

    Args:
        n: The number to factorize.

    Yields:
        The prime factors of n.
    """
    while n % 2 == 0:
        yield 2
        n //= 2
    for i in range(3, isqrt(n) + 1, 2):
        while n % i == 0:
            yield i
            n //= i
        if n == 1:
            return
    if n > 2:
        yield n


def factors(n: int) -> set[int]:
    """Generate all factors of the given number.

    Complexity: O(sqrt(n))

    Args:
        n: The number to find all factors of.

    Returns:
        A set containing all factors of n.
    """
    # Explanation for the runtime of the following factorization function:
    # This function's runtime is bounded by the runtime of computing the prime
    # factors. The runtime of actually finding the factors is much smaller,
    # namely O[n^(1/k)], where k is an arbitrary constant.
    # -------------------------------------------------------------------------
    # Proof:
    # 1. In the worst case, our number n is the product of the first p primes.
    # Now you might think that this would lead to a very inefficient
    # runtime, since we would have to try all combinations of these primes,
    # which would lead to O[2^p] runtime. If you thought this, you are
    # absolutely right! However, the key insight is that p will be much,
    # much smaller than n. The following bullet points will explain why. I
    # have ordered them in the same order as my thought process, as opposed
    # to the order in which they would be used in a formal proof.
    # 2. The first thing you have to realize is that by definition, we have:
    # prod_{i=1}^p p_i = n
    # However, we don't have the values of the primes p_i. If we manage to
    # express these in terms of p and n, we would be able to express p in
    # terms of n!
    # 3. Luckily, there exists something called the prime number theorem, which
    # states that the number of primes pi(m) <= m is approximately
    # m / log(m). Source:
    # https://en.wikipedia.org/wiki/Prime_number_theorem
    # This allows us to locate the i-th prime p_i, since according to our
    # approximation, when pi(m) = i, we will encounter p_i. Thus,
    # p_i / log(p_i) = i.
    # 4. We now have an equation with only p's and n's. We could stop here,
    # but our current expression is not very helpful. After all, our goal
    # was to approximate the runtime using big-O notation, which we
    # currently cannot do. Therefore, we will make a few assumptions,
    # after which we will still end up with a runtime of O[n^(1/k)].  :D
    # 5. Since we can't solve this equation analytically, we will overestimate
    # the amount of primes, since this will still result in a valid big-O
    # upper bound. Therefore, we will assume that the i-th prime is located
    # at p_i = i. This results in prod_{i=1}^p i = n, or p! = n.
    # 6. We can approximate p! using Stirling's approximation, which states
    # that p! ~ sqrt(2*pi*p) * (p/e)^p. Source:
    # https://en.wikipedia.org/wiki/Stirling%27s_approximation
    # 7. While this expression is nice, we still can't use it to approximate
    # the runtime. Therefore, we will make one final approximation, namely
    # that p! ~ (p/e)^p. The difference between the two is negligible, since
    # the first term grows far less quickly than the second for large p.
    # 8. Now, we know that the runtime is O[2^p], but this is still not very
    # helpful. Therefore, we will simplify this expression by substituting
    # p with log2(q), resulting in O[2^p] = O[2^log2(q)] = O[q]. Let's
    # also calculate a new approximation for p!:
    # p! ~ (p/e)^p
    #     = (log2(q)/e)^log2(q)
    #     = log2(q)^log2(q) / e^log2(q)
    #     in O[log2(q)^log2(q) * something smaller than a polynomial]
    #     = O[log2(q)^log2(q)]
    #     > O[any polynomial in q]   (since it is an exponential function)
    # 9. Finally, we can express the runtime in terms of q:
    # O[n] = O[p!]
    #         = O[log2(q)^log2(q)]
    #         > O[q^k] for any k
    # O[q] < O[n^(1/k)] for any k
    pfactors = prime_factors(n)
    factors = {1}
    for p in pfactors:
        factors.update({p * f for f in factors})
    return factors


def check_primality(n: int) -> list[bool]:
    """Check the primality of every number up to n-1.

    Complexity: O(n)

    Args:
        n: The number up to which to check primality.

    Returns:
        A list of booleans, where the value at index i is True if i is prime.
            Length: n
    """
    is_prime = [True] * n
    is_prime[0] = is_prime[1] = False
    for i in range(4, isqrt(n) + 1, 2):
        is_prime[i] = False
    for i in range(3, isqrt(n) + 1, 2):
        if is_prime[i]:  # use the Sieve of Eratosthenes
            for j in range(i**2, n, i):
                is_prime[j] = False
    return is_prime


def interp1d(x: float, x1: float, y1: float, x2: float, y2: float) -> float:
    """Return an interpolated value y given two points and a value x.

    Args:
        x: The x-value to interpolate.
        x1: The x-value of the first point.
        y1: The y-value of the first point.
        x2: The x-value of the second point.
        y2: The y-value of the second point.

    Returns:
        The interpolated value y.
    """
    return y1 + (x - x1) * (y2 - y1) / (x2 - x1)


def lcm(a: int, b: int) -> int:
    """Return the least common multiple of a and b.

    Note: math.lcm in Python 3.9 and later is around 10% faster than this
    implementation.

    Args:
        a: The first number.
        b: The second number.

    Returns:
        The least common multiple of a and b.
    """
    return a // gcd(a, b) * b


def pascal(n: int) -> list[int]:
    """Returns the nth row of Pascal's triangle.

    Complexity: O(n)

    Args:
        n: The row to return.

    Returns:
        The nth row of Pascal's triangle.
            Length: n + 1
    """
    row = [1] * (n + 1)
    for k in range(n // 2):
        x = row[k] * (n - k) // (k + 1)
        row[k + 1] = x
        row[n - k - 1] = x
    return row


def crt(*pairs: tuple[int, int]) -> tuple[int, int] | None:
    """The Chinese remainder theorem finds an integer x that satisfies:
        x = a1 (mod n1)
        ...
        x = am (mod nm)

    Args:
        *pairs: Pairs of (ai, ni) where ai is the remainder and ni is the
            modulus.

    Returns:
        None if there is no solution, otherwise a tuple containing:
        - The solution x.
        - The least common multiple of the moduli, lcm(n1, ..., nm).
    """
    a1, n1 = pairs[0]
    for a2, n2 in pairs[1:]:
        d, x_prime, _ = extended_gcd(n1, n2)
        if a1 % d != a2 % d:
            return
        a1 += x_prime * (a2 - a1) // d * n1
        n1 = lcm(n1, n2)
        a1 %= n1
    return a1, n1


def pos_mod(n: int, m: int) -> int:
    """Perform n % m, but substitute 0 by m.

    Args:
        n: The dividend.
        m: The divisor.

    Returns:
        The positive modulo of n and m.
    """
    return (n - 1) % m + 1


def gaussian(x: float, mu: float, sigma: float) -> float:
    """Calculate the value of a Gaussian distribution at x.

    Args:
        x: The number at which to evaluate the Gaussian distribution.
        mu: The mean of the Gaussian distribution.
        sigma: The standard deviation of the Gaussian distribution.

    Returns:
        The value of the Gaussian distribution at x.
    """
    return exp(-(((x - mu) / sigma) ** 2) / 2) / (sigma * sqrt(2 * pi))


def monotonic_hyperbolic_rescaling(x: float, r: float) -> float:
    """Monotonically rescale a number using a hyperbolic function.

    The function is made to be useful for rescaling numbers between 0 and 1,
    and it will always return a number between 0 and 1.

    The inverse of the function is:
    >>> y = monotonic_hyperbolic_rescaling(x, r)
    >>> y_inv = monotonic_hyperbolic_rescaling(y, -r)
    >>> assert x - y_inv < 1e-6
    True

    Args:
        x: The number to rescale. Must be between 0 and 1.
        r: The rescaling factor. Can be any number from -inf to inf.
            If r is positive, the function will be above the line y=x and its
            slope will decrease as x increases.
            If r is negative, the function will be below the line y=x and its
            slope will increase as x increases.

    Returns:
        The rescaled number. Will be between 0 and 1.
    """
    f = (r + 2 - sqrt(r**2 + 4)) / 2
    return x / (1 - f + f * x)


def optimal_grid_layout(
    n: int, max_cols: int | None = None, max_rows: int | None = None
) -> tuple[int, int]:
    """Find a grid layout so that the grid will be filled as much as possible.

    Formally, this function finds a pair (ncols, nrows) such that:
    - ncols <= max_cols
    - nrows <= max_rows
    - ncols * nrows >= n
    - ncols * nrows is minimal

    If there are multiple solutions, the one with ncols < nrows is preferred if
    max_cols <= max_rows, otherwise the one with ncols > nrows is preferred.
    If max_cols or max_rows is None, they are set to n internally.

    Args:
        n: The minimum number of cells in the grid.
        max_cols: The maximum number of columns. If None, there is no limit.
        max_rows: The maximum number of rows. If None, there is no limit.

    Returns:
        Tuple containing:
        - The number of columns ncols.
        - The number of rows nrows.
    """
    if max_cols is None:
        max_cols = n
    if max_rows is None:
        max_rows = n
    if max_cols * max_rows < n:
        raise ValueError("max_cols * max_rows must be at least n.")

    # Calculate the optimal amount of rows and columns so that the grid will
    # be filled as much as possible with plots.
    small_side = min(max_cols, max_rows)
    large_side = max(max_cols, max_rows)
    for m in range(n, max_cols * max_rows + 1):
        factors_m = sorted(factors(m))
        for i in range((len(factors_m) - 1) // 2, -1, -1):
            f1 = factors_m[i]  # middle left factor
            f2 = m // f1  # middle right factor

            if f1 <= small_side and f2 <= large_side:  # optimal grid found
                if max_cols <= max_rows:
                    return f1, f2
                else:
                    return f2, f1

    raise RuntimeError(
        "Optimal grid layout not found. This error should never occur."
    )


def optimal_size(
    ratio: float, max_size_x: float, max_size_y: float
) -> tuple[float, float]:
    """Get the largest possible size for the given aspect ratio.

    Formally, this function finds a pair (size_x, size_y) such that:
    - size_x <= max_size_x
    - size_y <= max_size_y
    - size_x / size_y = ratio
    - size_x * size_y is maximal

    Args:
        ratio: The aspect ratio, defined as x / y.
        max_size_x: The maximum horizontal length.
        max_size_y: The maximum vertical length.

    Returns:
        Tuple containing:
        - The horizontal size size_x.
        - The vertical size size_y.
    """
    if ratio > max_size_x / max_size_y:
        # There is space left in the y direction.
        return max_size_x, max_size_x / ratio
    else:
        # There is space left in the x direction.
        return max_size_y * ratio, max_size_y
