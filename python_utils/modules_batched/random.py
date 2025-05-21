from __future__ import annotations

from typing import Any

import torch


def rand_float_decreasingly_likely(*args: Any, **kwargs: Any) -> torch.Tensor:
    """Generate random floats having a decreasing probability of increasing.

    The expected value of the generated random floats is 1.

    Args:
        size: A sequence of integers defining the shape of the output tensor.
            Can be a variable number of arguments or a collection like a list
            or tuple.

    Keyword args:
        generator: A pseudorandom number generator for sampling.
        out: The output tensor.
        dtype: The desired data type of returned tensor. If None, uses a global
            default (see torch.set_default_dtype). Defaults to None.
        layout: The desired layout of returned Tensor. Defaults to
            torch.strided.
        device: The desired device of the returned tensor. If None, uses the
            current device for the default tensor type (see
            torch.set_default_device). The device will be the CPU for CPU
            tensor types and the current CUDA device for CUDA tensor types.
            Defaults to None.
        requires_grad: If autograd should record operations on the returned
            tensor. Defaults to False.
        pin_memory: If set, the returned tensor would be allocated in the
            pinned memory. Works only for CPU tensors. Defaults to False.

    Returns:
        A tensor of random floats.
    """
    # Note: R = -torch.log2(1 - torch.rand(1)) will generate a random float R
    # with an exponentially decreasing probability of increasing. This is
    # because the probability of R is half that of R + 1, which is half that of
    # R + 2, and so on. The above formula is equivalent to:
    # 0-1: 50%
    # 1-2: 25%
    # 2-3: 12.5%
    # 3-4: 6.25%
    # ...
    # Thus, the expected value is:
    # E(R) = 0 * 0.5 + 1 * 0.25 + 2 * 0.125 + 3 * 0.0625 + ...
    #      = sum_{i=0}^inf i * 0.5^(i+1)
    #      = 1
    # This implies that if you multiply R by a constant c, the expected value
    # will be c. This is useful for generating random numbers with a specific
    # expected value.
    return -torch.log2(1 - torch.rand(*args, **kwargs))


def rand_int_decreasingly_likely(*args: Any, **kwargs: Any) -> torch.Tensor:
    """Generate random integers having a decreasing probability of increasing.

    The expected value of the generated random integers is 1.

    Args:
        size: A sequence of integers defining the shape of the output tensor.
            Can be a variable number of arguments or a collection like a list
            or tuple.

    Keyword args:
        generator: A pseudorandom number generator for sampling.
        out: The output tensor.
        dtype: The desired data type of returned tensor. If None, uses a global
            default (see torch.set_default_dtype). Defaults to None.
        layout: The desired layout of returned Tensor. Defaults to
            torch.strided.
        device: The desired device of the returned tensor. If None, uses the
            current device for the default tensor type (see
            torch.set_default_device). The device will be the CPU for CPU
            tensor types and the current CUDA device for CUDA tensor types.
            Defaults to None.
        requires_grad: If autograd should record operations on the returned
            tensor. Defaults to False.
        pin_memory: If set, the returned tensor would be allocated in the
            pinned memory. Works only for CPU tensors. Defaults to False.

    Returns:
        A tensor of random integers.
    """
    return rand_float_decreasingly_likely(*args, **kwargs).long()
