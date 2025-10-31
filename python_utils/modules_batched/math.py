from __future__ import annotations

from math import pi, sqrt
from typing import overload

import numpy as np
import numpy.typing as npt
import torch


@overload
def floor_to_multiple_batched(
    x: npt.NDArray, base: float | npt.NDArray
) -> npt.NDArray:
    pass


@overload
def floor_to_multiple_batched(
    x: float | npt.NDArray, base: npt.NDArray
) -> npt.NDArray:
    pass


@overload
def floor_to_multiple_batched(
    x: torch.Tensor, base: float | torch.Tensor
) -> torch.Tensor:
    pass


@overload
def floor_to_multiple_batched(
    x: float | torch.Tensor, base: torch.Tensor
) -> torch.Tensor:
    pass


def floor_to_multiple_batched(
    x: float | npt.NDArray | torch.Tensor,
    base: float | npt.NDArray | torch.Tensor,
) -> npt.NDArray | torch.Tensor:
    """Round numbers to the nearest lower multiple of the base numbers.

    Args:
        x: The number(s) to floor.
        base: The base(s) to floor to.

    Returns:
        The rounded number(s).
            Shape: Broadcasted shape of x and base.
    """
    return x // base * base  # type: ignore


@overload
def ceil_to_multiple_batched(
    x: npt.NDArray, base: float | npt.NDArray
) -> npt.NDArray:
    pass


@overload
def ceil_to_multiple_batched(
    x: float | npt.NDArray, base: npt.NDArray
) -> npt.NDArray:
    pass


@overload
def ceil_to_multiple_batched(
    x: torch.Tensor, base: float | torch.Tensor
) -> torch.Tensor:
    pass


@overload
def ceil_to_multiple_batched(
    x: float | torch.Tensor, base: torch.Tensor
) -> torch.Tensor:
    pass


def ceil_to_multiple_batched(
    x: float | npt.NDArray | torch.Tensor,
    base: float | npt.NDArray | torch.Tensor,
) -> npt.NDArray | torch.Tensor:
    """Round numbers to the nearest higher multiple of the base numbers.

    Args:
        x: The number(s) to ceil.
        base: The base(s) to ceil to.

    Returns:
        The rounded number(s).
            Shape: Broadcasted shape of x and base.
    """
    if isinstance(x, np.ndarray) or isinstance(base, np.ndarray):
        ceil_func = np.ceil
    else:
        ceil_func = torch.ceil

    return ceil_func(x / base) * base  # type: ignore


@overload
def round_to_multiple_batched(
    x: npt.NDArray, base: float | npt.NDArray
) -> npt.NDArray:
    pass


@overload
def round_to_multiple_batched(
    x: float | npt.NDArray, base: npt.NDArray
) -> npt.NDArray:
    pass


@overload
def round_to_multiple_batched(
    x: torch.Tensor, base: float | torch.Tensor
) -> torch.Tensor:
    pass


@overload
def round_to_multiple_batched(
    x: float | torch.Tensor, base: torch.Tensor
) -> torch.Tensor:
    pass


def round_to_multiple_batched(
    x: float | npt.NDArray | torch.Tensor,
    base: float | npt.NDArray | torch.Tensor,
) -> npt.NDArray | torch.Tensor:
    """Round numbers to the nearest multiple of the base numbers.

    Args:
        x: The number(s) to round.
        base: The base(s) to round to.

    Returns:
        The rounded number(s).
            Shape: Broadcasted shape of x and base.
    """
    if isinstance(x, np.ndarray) or isinstance(base, np.ndarray):
        round_func = np.round
    else:
        round_func = torch.round

    return round_func(x / base) * base  # type: ignore


@overload
def gaussian_batched(
    x: npt.NDArray, mu: float | npt.NDArray, sigma: float | npt.NDArray
) -> npt.NDArray:
    pass


@overload
def gaussian_batched(
    x: float | npt.NDArray, mu: npt.NDArray, sigma: float | npt.NDArray
) -> npt.NDArray:
    pass


@overload
def gaussian_batched(
    x: float | npt.NDArray, mu: float | npt.NDArray, sigma: npt.NDArray
) -> npt.NDArray:
    pass


@overload
def gaussian_batched(
    x: torch.Tensor, mu: float | torch.Tensor, sigma: float | torch.Tensor
) -> torch.Tensor:
    pass


@overload
def gaussian_batched(
    x: float | torch.Tensor, mu: torch.Tensor, sigma: float | torch.Tensor
) -> torch.Tensor:
    pass


@overload
def gaussian_batched(
    x: float | torch.Tensor, mu: float | torch.Tensor, sigma: torch.Tensor
) -> torch.Tensor:
    pass


def gaussian_batched(
    x: float | npt.NDArray | torch.Tensor,
    mu: float | npt.NDArray | torch.Tensor,
    sigma: float | npt.NDArray | torch.Tensor,
) -> npt.NDArray | torch.Tensor:
    """Calculate the values of Gaussian distributions at specific points.

    Args:
        x: The number(s) at which to evaluate the Gaussian distribution(s).
        mu: The mean(s) of the Gaussian distribution(s).
        sigma: The standard deviation(s) of the Gaussian distribution(s).

    Returns:
        The value(s) of the Gaussian distribution(s) at x.
            Shape: Broadcasted shape of x, mu, and sigma.
    """
    if (
        isinstance(x, np.ndarray)
        or isinstance(mu, np.ndarray)
        or isinstance(sigma, np.ndarray)
    ):
        exp_func = np.exp
    else:
        exp_func = torch.exp

    return exp_func(-(((x - mu) / sigma) ** 2) / 2) / (  # type: ignore
        sigma * sqrt(2 * pi)
    )


@overload
def monotonic_linear_fractional_rescaling_batched(
    x: npt.NDArray, r: float | npt.NDArray
) -> npt.NDArray:
    pass


@overload
def monotonic_linear_fractional_rescaling_batched(
    x: float | npt.NDArray, r: npt.NDArray
) -> npt.NDArray:
    pass


@overload
def monotonic_linear_fractional_rescaling_batched(
    x: torch.Tensor, r: float | torch.Tensor
) -> torch.Tensor:
    pass


@overload
def monotonic_linear_fractional_rescaling_batched(
    x: float | torch.Tensor, r: torch.Tensor
) -> torch.Tensor:
    pass


def monotonic_linear_fractional_rescaling_batched(
    x: float | npt.NDArray | torch.Tensor, r: float | npt.NDArray | torch.Tensor
) -> npt.NDArray | torch.Tensor:
    """Monotonically rescale numbers using a linear functional function.

    The function is made to be useful for rescaling numbers between 0 and 1, and
    it will always return a number between 0 and 1.

    The inverse of the function is:
    >>> y = monotonic_linear_fractional_rescaling_batched(x, r)
    >>> y_inv = monotonic_linear_fractional_rescaling_batched(y, -r)
    >>> assert torch.allclose(x, y_inv)
    True

    Args:
        x: The number(s) to rescale. Must be between 0 and 1.
        r: The rescaling factor(s). Can be any number from -inf to inf.
            If r is positive, the function will be above the line y=x and its
            slope will decrease as x increases.
            If r is negative, the function will be below the line y=x and its
            slope will increase as x increases.

    Returns:
        The rescaled number(s). Will be between 0 and 1.
            Shape: Broadcasted shape of x and r.
    """
    if isinstance(r, (int, float)):
        sqrt_func = sqrt
    elif isinstance(r, np.ndarray):
        sqrt_func = np.sqrt
    else:
        sqrt_func = torch.sqrt

    f = (r + 2 - sqrt_func(r**2 + 4)) / 2  # type: ignore
    return x / (1 - f + f * x)  # type: ignore
