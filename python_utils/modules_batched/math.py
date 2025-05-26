from __future__ import annotations

from math import pi, sqrt

import numpy.typing as npt
import torch


def floor_to_multiple_batched(
    x: float | npt.NDArray | torch.Tensor,
    base: float | npt.NDArray | torch.Tensor,
) -> torch.Tensor:
    """Round numbers to the nearest lower multiple of the base numbers.

    Args:
        x: The number(s) to floor.
        base: The base(s) to floor to.

    Returns:
        The rounded number(s).
            Shape: Broadcasted shape of x and base.
    """
    x = torch.as_tensor(x)
    base = torch.as_tensor(base)
    return torch.floor_divide(x, base).long() * base


def ceil_to_multiple_batched(
    x: float | npt.NDArray | torch.Tensor,
    base: float | npt.NDArray | torch.Tensor,
) -> torch.Tensor:
    """Round numbers to the nearest higher multiple of the base numbers.

    Args:
        x: The number(s) to ceil.
        base: The base(s) to ceil to.

    Returns:
        The rounded number(s).
            Shape: Broadcasted shape of x and base.
    """
    x = torch.as_tensor(x)
    base = torch.as_tensor(base)
    return torch.ceil(x / base).long() * base


def round_to_multiple_batched(
    x: float | npt.NDArray | torch.Tensor,
    base: float | npt.NDArray | torch.Tensor,
) -> torch.Tensor:
    """Round numbers to the nearest multiple of the base numbers.

    Args:
        x: The number(s) to round.
        base: The base(s) to round to.

    Returns:
        The rounded number(s).
            Shape: Broadcasted shape of x and base.
    """
    x = torch.as_tensor(x)
    base = torch.as_tensor(base)
    return torch.round(x / base).long() * base


def interp1d_batched(
    x: float | npt.NDArray | torch.Tensor,
    x1: float | npt.NDArray | torch.Tensor,
    y1: float | npt.NDArray | torch.Tensor,
    x2: float | npt.NDArray | torch.Tensor,
    y2: float | npt.NDArray | torch.Tensor,
) -> torch.Tensor:
    """Return interpolated values y given two points and values x.

    Args:
        x: The x-value(s) to interpolate.
        x1: The x-value(s) of the first point(s).
        y1: The y-value(s) of the first point(s).
        x2: The x-value(s) of the second point(s).
        y2: The y-value(s) of the second point(s).

    Returns:
        The interpolated value(s) y.
            Shape: Broadcasted shape of x, x1, y1, x2, and y2.
    """
    x = torch.as_tensor(x)
    x1 = torch.as_tensor(x1)
    y1 = torch.as_tensor(y1)
    x2 = torch.as_tensor(x2)
    y2 = torch.as_tensor(y2)
    return y1 + (x - x1) * (y2 - y1) / (x2 - x1)


def gaussian_batched(
    x: float | npt.NDArray | torch.Tensor,
    mu: float | npt.NDArray | torch.Tensor,
    sigma: float | npt.NDArray | torch.Tensor,
) -> torch.Tensor:
    """Calculate the values of Gaussian distributions at specific points.

    Args:
        x: The number(s) at which to evaluate the Gaussian distribution(s).
        mu: The mean(s) of the Gaussian distribution(s).
        sigma: The standard deviation(s) of the Gaussian distribution(s).

    Returns:
        The value(s) of the Gaussian distribution(s) at x.
            Shape: Broadcasted shape of x, mu, and sigma.
    """
    x = torch.as_tensor(x)
    mu = torch.as_tensor(mu)
    sigma = torch.as_tensor(sigma)
    return (-((x - mu) / sigma).square() / 2).exp() / (sigma * sqrt(2 * pi))


def monotonic_hyperbolic_rescaling_batched(
    x: float | npt.NDArray | torch.Tensor, r: float | npt.NDArray | torch.Tensor
) -> torch.Tensor:
    """Monotonically rescale numbers using hyperbolic functions.

    The function is made to be useful for rescaling numbers between 0 and 1,
    and it will always return a number between 0 and 1.

    The inverse of the function is:
    >>> y = monotonic_hyperbolic_rescaling(x, r)
    >>> y_inv = monotonic_hyperbolic_rescaling(y, -r)
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
    x = torch.as_tensor(x)
    r = torch.as_tensor(r)
    f = (r + 2 - (r.square() + 4).sqrt()) / 2
    return x / (1 - f + f * x)
