from __future__ import annotations

import unittest
from typing import Any

import torch
from typing_extensions import override

from python_utils.modules.lightning import OnlySaveDirectHyperparameters


class A_not(OnlySaveDirectHyperparameters):
    """A that does NOT call save_hyperparameters."""

    def __init__(self, arg1: str, arg2: int, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.arg1 = arg1
        self.arg2 = arg2

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class A_save(OnlySaveDirectHyperparameters):
    """A that DOES call save_hyperparameters."""

    def __init__(self, arg1: str, arg2: int, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.save_hyperparameters("arg2")
        self.arg1 = arg1
        self.arg2 = arg2

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class B_not_A_not(A_not):
    """B that does NOT call save_hyperparameters, A that does NOT call it."""

    def __init__(
        self, arg3: float, arg4: bool, *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.arg3 = arg3
        self.arg4 = arg4


class B_not_A_save(A_save):
    """B that does NOT call save_hyperparameters, A that DOES call it."""

    def __init__(
        self, arg3: float, arg4: bool, *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.arg3 = arg3
        self.arg4 = arg4


class B_save_A_not(A_not):
    """B that DOES call save_hyperparameters, A that does NOT call it."""

    def __init__(
        self, arg3: float, arg4: bool, *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.arg3 = arg3
        self.arg4 = arg4


class B_save_A_save(A_save):
    """B that DOES call save_hyperparameters, A that DOES call it."""

    def __init__(
        self, arg3: float, arg4: bool, *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(ignore="arg4")
        self.arg3 = arg3
        self.arg4 = arg4


class TestOnlySaveDirectHyperparameters(unittest.TestCase):
    @override
    def setUp(self) -> None:
        """Set up the test case."""
        arg1 = "test_string"
        arg2 = 42
        arg3 = 3.14
        arg4 = True
        some_other_arg = "ignored"
        some_other_kwarg = "ignored"
        self.model_a_not = A_not(
            arg1, arg2, some_other_arg, extra_kwarg=some_other_kwarg
        )
        self.model_a_save = A_save(
            arg1, arg2, some_other_arg, extra_kwarg=some_other_kwarg
        )
        self.model_b_not_a_not = B_not_A_not(
            arg3, arg4, arg1, arg2, some_other_arg, extra_kwarg=some_other_kwarg
        )
        self.model_b_not_a_save = B_not_A_save(
            arg3, arg4, arg1, arg2, some_other_arg, extra_kwarg=some_other_kwarg
        )
        self.model_b_save_a_not = B_save_A_not(
            arg3, arg4, arg1, arg2, some_other_arg, extra_kwarg=some_other_kwarg
        )
        self.model_b_save_a_save = B_save_A_save(
            arg3, arg4, arg1, arg2, some_other_arg, extra_kwarg=some_other_kwarg
        )

    def test_a_not(self) -> None:
        self.assertDictEqual(self.model_a_not.hparams, {})

    def test_a_save(self) -> None:
        self.assertDictEqual(self.model_a_save.hparams, {"arg2": 42})

    def test_b_not_a_not(self) -> None:
        self.assertDictEqual(self.model_b_not_a_not.hparams, {})

    def test_b_not_a_save(self) -> None:
        self.assertDictEqual(self.model_b_not_a_save.hparams, {"arg2": 42})

    def test_b_save_a_not(self) -> None:
        self.assertDictEqual(
            self.model_b_save_a_not.hparams, {"arg3": 3.14, "arg4": True}
        )

    def test_b_save_a_save(self) -> None:
        self.assertDictEqual(
            self.model_b_save_a_save.hparams, {"arg2": 42, "arg3": 3.14}
        )
