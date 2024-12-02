from math import pi, sqrt
from typing import TypeVar, overload

import numpy.typing as npt
import torch


number_type = (int, float)  # used for isinstance() checks
NumberT = TypeVar("NumberT", int, float)  # used for typing dependent vars


def floor_to_multiple_batched(x: torch.Tensor, base: NumberT) -> torch.Tensor:
    """Floor a tensor of numbers to the nearest multiple of a base number.

    Args:
        x: The tensor of numbers to floor.
            Shape: [*]
        base: The base to floor to.

    Returns:
        The floored tensor.
            Shape: [*]
    """
    return torch.floor_divide(x, base).long() * base


def ceil_to_multiple_batched(x: torch.Tensor, base: NumberT) -> torch.Tensor:
    """Ceil a tensor of numbers to the nearest multiple of a base number.

    Args:
        x: The tensor of numbers to ceil.
            Shape: [*]
        base: The base to ceil to.

    Returns:
        The ceiled tensor.
            Shape: [*]
    """
    return torch.ceil(x / base).long() * base


def round_to_multiple_batched(x: torch.Tensor, base: NumberT) -> torch.Tensor:
    """Round a tensor of numbers to the nearest multiple of a base number.

    Args:
        x: The tensor of numbers to round.
            Shape: [*]
        base: The base to round to.

    Returns:
        The rounded tensor.
            Shape: [*]
    """
    return torch.round(x / base).long() * base


@overload
def interp1d_batched(
    x: npt.NDArray,
    x1: float | npt.NDArray,
    y1: float | npt.NDArray,
    x2: float | npt.NDArray,
    y2: float | npt.NDArray,
) -> npt.NDArray:
    """Return interpolated value(s) y given two points and value(s) x."""
    ...


@overload
def interp1d_batched(
    x: torch.Tensor,
    x1: float | torch.Tensor,
    y1: float | torch.Tensor,
    x2: float | torch.Tensor,
    y2: float | torch.Tensor,
) -> torch.Tensor:
    """Return interpolated value(s) y given two points and value(s) x."""
    ...


def interp1d_batched(
    x: npt.NDArray | torch.Tensor,
    x1: float | npt.NDArray | torch.Tensor,
    y1: float | npt.NDArray | torch.Tensor,
    x2: float | npt.NDArray | torch.Tensor,
    y2: float | npt.NDArray | torch.Tensor,
) -> npt.NDArray | torch.Tensor:
    """Return interpolated value(s) y given two points and value(s) x.

    Args:
        x: The x-value(s) to interpolate.
        x1: The x-value of the first point.
        y1: The y-value of the first point.
        x2: The x-value of the second point.
        y2: The y-value of the second point.

    Returns:
        The interpolated value(s) y.
    """
    return y1 + (x - x1) * (y2 - y1) / (x2 - x1)


def gaussian_batched(
    x: float | torch.Tensor,
    mu: float | torch.Tensor,
    sigma: float | torch.Tensor,
) -> torch.Tensor:
    """Calculate the value of a batch of Gaussian distributions at x.

    Args:
        x: The numbers at which to evaluate the Gaussian distribution.
            Shape: [] or [B]
        mu: The mean of the Gaussian distribution.
            Shape: [] or [B]
        sigma: The standard deviation of the Gaussian distribution.
            Shape: [] or [B]

    Returns:
        The value of the Gaussian distribution at x.
            Shape: [1] or [B]
    """
    if isinstance(x, torch.Tensor):
        device = x.device
    elif isinstance(mu, torch.Tensor):
        device = mu.device
    elif isinstance(sigma, torch.Tensor):
        device = sigma.device
    else:
        device = "cpu"

    if isinstance(x, number_type):
        x = torch.tensor(x, device=device)
    if isinstance(mu, number_type):
        mu = torch.tensor(mu, device=device)
    if isinstance(sigma, number_type):
        sigma = torch.tensor(sigma, device=device)

    if x.ndim == 0:
        x = x.unsqueeze(0)
    if mu.ndim == 0:
        mu = mu.unsqueeze(0)
    if sigma.ndim == 0:
        sigma = sigma.unsqueeze(0)

    return (-((x - mu) / sigma).square() / 2).exp() / (sigma * sqrt(2 * pi))


def monotonic_hyperbolic_rescaling_batched(
    x: float | torch.Tensor, r: float | torch.Tensor
) -> torch.Tensor:
    """Monotonically rescale a batch of numbers using hyperbolic functions.

    The function is made to be useful for rescaling numbers between 0 and 1,
    and it will always return a number between 0 and 1.

    The inverse of the function is:
    >>> y = monotonic_hyperbolic_rescaling(x, r)
    >>> y_inv = monotonic_hyperbolic_rescaling(y, -r)
    >>> assert torch.allclose(x, y_inv)
    True

    Args:
        x: The numbers to rescale. Must be between 0 and 1.
            Shape: [] or [B]
        r: The rescaling factors. Can be any number from -inf to inf.
            If r is positive, the function will be above the line y=x and its
            slope will decrease as x increases.
            If r is negative, the function will be below the line y=x and its
            slope will increase as x increases.
            Shape: [] or [B]

    Returns:
        The rescaled numbers. Will be between 0 and 1.
            Shape: [1] or [B]
    """
    if isinstance(x, torch.Tensor):
        device = x.device
    elif isinstance(r, torch.Tensor):
        device = r.device
    else:
        device = "cpu"

    if isinstance(x, number_type):
        x = torch.tensor(x, device=device)
    if isinstance(r, number_type):
        r = torch.tensor(r, device=device)

    if x.ndim == 0:
        x = x.unsqueeze(0)
    if r.ndim == 0:
        r = r.unsqueeze(0)

    f = (r + 2 - (r.square() + 4).sqrt()) / 2
    return x / (1 - f + f * x)
