from __future__ import annotations

import unittest
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import lightning.pytorch.callbacks as pl_callbacks
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing_extensions import override

from python_utils.modules.lightning import (
    OnlySaveDirectHyperparameters,
    prepare_best_model_loading,
)


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


class ScoreDecreasesModel(pl.LightningModule):
    """Model whose validation score is highest after the first epoch."""

    def __init__(self) -> None:
        super().__init__()
        # ModelCheckpoint only saves a checkpoint if an optimizer step was
        # taken since the last one, so the model needs something to optimize.
        self.weight = nn.Parameter(torch.zeros(1))

    @override
    def training_step(
        self, batch: list[torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        return self.weight.sum()

    @override
    def validation_step(
        self, batch: list[torch.Tensor], batch_idx: int
    ) -> None:
        self.log("score/val", -float(self.current_epoch))

    @override
    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.parameters(), lr=0.1)


def create_trainer(dirpath: Path, **trainer_kwargs: Any) -> pl.Trainer:
    """Create a small CPU trainer that keeps only the best checkpoint.

    Args:
        dirpath: Directory in which the ModelCheckpoint callback saves the
            checkpoints.
        **trainer_kwargs: Additional keyword arguments to pass to the Trainer.

    Returns:
        The trainer.
    """
    return pl.Trainer(
        accelerator="cpu",
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[
            pl_callbacks.ModelCheckpoint(
                dirpath=dirpath,
                monitor="score/val",
                mode="max",
                save_last=True,
                save_top_k=1,
            )
        ],
        **trainer_kwargs,
    )


def fit_score_decreases_model(dirpath: Path) -> Path:
    """Train a ScoreDecreasesModel for 3 epochs.

    Args:
        dirpath: Directory in which the ModelCheckpoint callback saves the
            checkpoints.

    Returns:
        Path to the best checkpoint, which is the one from the first epoch.
        last.ckpt is saved next to it and contains the weights of the third
        epoch.
    """
    trainer = create_trainer(dirpath, max_epochs=3, num_sanity_val_steps=0)
    dataloader = DataLoader(TensorDataset(torch.randn(4, 2)), batch_size=2)
    trainer.fit(ScoreDecreasesModel(), dataloader, dataloader)
    assert isinstance(trainer.checkpoint_callback, pl_callbacks.ModelCheckpoint)
    return Path(trainer.checkpoint_callback.best_model_path)


def prepare_and_get_best_model_path(dirpath: Path) -> Path:
    """Call prepare_best_model_loading() for the checkpoints in dirpath.

    Args:
        dirpath: Directory in which the ModelCheckpoint callback saves the
            checkpoints.

    Returns:
        Path to the checkpoint that prepare_best_model_loading() selected.
    """
    trainer = create_trainer(dirpath)
    prepare_best_model_loading(trainer, ScoreDecreasesModel())
    assert isinstance(trainer.checkpoint_callback, pl_callbacks.ModelCheckpoint)
    return Path(trainer.checkpoint_callback.best_model_path)


class TestPrepareBestModelLoading:
    def test_prepare_best_model_loading_ignores_last_ckpt(
        self, tmp_path: Path
    ) -> None:
        # last.ckpt stores the same best_model_score as the best checkpoint and
        # is saved after it, so it must not win because of that.
        path_best_ckpt = fit_score_decreases_model(tmp_path)
        assert (tmp_path / "last.ckpt").is_file()
        assert prepare_and_get_best_model_path(tmp_path) == path_best_ckpt

    def test_prepare_best_model_loading_moved_dir(self, tmp_path: Path) -> None:
        # The checkpoints store absolute paths, which no longer exist after
        # moving the directory. The function should still be able to find the
        # best checkpoint.
        path_best_ckpt = fit_score_decreases_model(tmp_path / "original")
        (tmp_path / "original").rename(tmp_path / "moved")
        assert (
            prepare_and_get_best_model_path(tmp_path / "moved")
            == tmp_path / "moved" / path_best_ckpt.name
        )
