from math import sqrt
from typing import TypeVar

import torch


NumberT = TypeVar("NumberT", int, float)


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


def monotonic_hyperbolic_rescaling_batched(
    x: torch.Tensor, r: float
) -> torch.Tensor:
    """Monotonically rescale a tensor of numbers using a hyperbolic function.

    The function is made to be useful for rescaling numbers between 0 and 1,
    and it will always return a number between 0 and 1.

    The inverse of the function is:
    >>> y = monotonic_hyperbolic_rescaling(x, r)
    >>> y_inv = monotonic_hyperbolic_rescaling(y, -r)
    >>> assert torch.allclose(x, y_inv)
    True

    Args:
        x: The tensor of numbers to rescale. Each must be between 0 and 1.
            Shape: [*]
        r: The rescaling factor. Can be any number from -inf to inf.
            If r is positive, the function will be above the line y=x and its
            slope will decrease as x increases.
            If r is negative, the function will be below the line y=x and its
            slope will increase as x increases.

    Returns:
        The rescaled numbers. Each will be between 0 and 1.
            Shape: [*]
    """
    f = (r + 2 - sqrt(r**2 + 4)) / 2
    return x / (1 - f + f * x)
