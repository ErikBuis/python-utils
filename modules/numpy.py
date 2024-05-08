from typing import Any, Generic, TypeVar

import numpy as np
import numpy.typing as npt


T = TypeVar("T")


class NDArrayGeneric(np.ndarray, Generic[T]):
    """np.ndarray that allows for static type hinting of generics."""

    def __getitem__(self, key: Any) -> T:
        return super().__getitem__(key)  # type: ignore


def pad_sequence(
    sequences: list[npt.ArrayLike],
    batch_first: bool = False,
    padding_value: Any = 0,
) -> npt.NDArray:
    """Pad a list of variable length arrays with padding_value.

    Note: This function is a numpy equivalent of torch.nn.utils.rnn.
    pad_sequence! It is faster than this implementation, so please use the
    torch version if you are working with PyTorch tensors.

    Args:
        sequences: A sequence of variable length arrays.
            Length: B
            Inner shape: [L_b, *]
        batch_first: Whether to return the batch dimension as the first
            dimension. If False, the output will have shape [max(L_b), B, *].
            If True, the output will have shape [B, max(L_b), *].
        padding_value: The value to use for padding the inner sequences.

    Returns:
        Array of shape [max(L_b), B, *] if batch_first is False, otherwise
        array of shape [B, max(L_b), *].
    """
    sequences_arr = [np.array(seq) for seq in sequences]
    star_shape = sequences_arr[0].shape[1:]
    assert all(
        (arr.shape[1:] == star_shape) for arr in sequences_arr[1:]
    ), "All arrays must have the same shape after the first dimension."
    B = len(sequences_arr)
    max_L_b = max(len(arr) for arr in sequences_arr)
    shape = (
        (B, max_L_b, *star_shape) if batch_first else (max_L_b, B, *star_shape)
    )
    dtype = np.result_type(*sequences_arr)
    padded = np.full(shape, padding_value, dtype=dtype)
    for b, arr in enumerate(sequences_arr):
        if batch_first:
            padded[b, : len(arr)] = arr
        else:
            padded[: len(arr), b] = arr
    return padded


def unequal_seqs_add(a: npt.NDArray, b: npt.NDArray) -> npt.NDArray:
    """Add two arrays, and adjust the size of the result if necessary.

    Args:
        a: The first array.
            Shape: [N]
        b: The second array.
            Shape: [M]

    Returns:
        The sum of the two arrays.
            Shape: [max(N, M)]
    """
    if len(a) < len(b):
        a = np.pad(a, (0, len(b) - len(a)))
    elif len(a) > len(b):
        b = np.pad(b, (0, len(a) - len(b)))
    return a + b


def init_normalized_histogram(
    bin_edges: npt.ArrayLike,
) -> tuple[float, npt.NDArray[np.float64]]:
    return 0, np.zeros(len(bin_edges) - 1)  # type: ignore


def update_normalized_histogram(
    old_value_avg: float,
    old_histogram: npt.NDArray[np.float64],
    amount_old_values: int,
    new_values: npt.ArrayLike,
    bin_edges: npt.ArrayLike,
) -> tuple[float, npt.NDArray[np.float64]]:
    """Update a normalized histogram with new values.

    Args:
        old_value_avg: The average of the old values in the histogram.
        old_histogram: The old normalized histogram.
            Shape: [bins]
        amount_old_values: The amount of old values in the histogram.
        new_values: The new values to add to the histogram.
            Shape: [amount_new_values]
        bin_edges: The bin edges of the histogram.
            Length: [bins + 1]

    Returns:
        Tuple containing:
            The updated average of the histogram.
            The updated normalized histogram.
                Shape: [bins]
    """
    new_values = np.array(new_values)

    # Update the average of the histogram.
    updated_value_avg = (
        old_value_avg * amount_old_values + new_values.mean() * len(new_values)
    ) / (amount_old_values + len(new_values))

    # Create a normalized histogram of the new values.
    new_histogram = np.histogram(new_values, bin_edges)[0] / len(new_values)

    # Weigh both histograms by the amount of values they contain.
    old_histogram *= amount_old_values
    new_histogram *= len(new_values)
    updated_histogram = (old_histogram + new_histogram) / (
        amount_old_values + len(new_values)
    )

    # Add the histograms and renormalize them.
    return updated_value_avg, updated_histogram
