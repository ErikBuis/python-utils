from math import pi, sqrt
from typing import TypeVar

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
    return torch.floor_divide(x, base).to(torch.int64) * base


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
    return torch.ceil(x / base).to(torch.int64) * base


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
    return torch.round(x / base).to(torch.int64) * base


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
    if isinstance(x, number_type):
        x = torch.tensor(x)
    if isinstance(mu, number_type):
        mu = torch.tensor(mu)
    if isinstance(sigma, number_type):
        sigma = torch.tensor(sigma)

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
    if isinstance(x, number_type):
        x = torch.tensor(x)
    if isinstance(r, number_type):
        r = torch.tensor(r)

    if x.ndim == 0:
        x = x.unsqueeze(0)
    if r.ndim == 0:
        r = r.unsqueeze(0)

    f = (r + 2 - (r.square() + 4).sqrt()) / 2
    return x / (1 - f + f * x)
