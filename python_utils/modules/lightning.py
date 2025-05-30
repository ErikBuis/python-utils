from __future__ import annotations

import logging
import os
import time
from collections import defaultdict
from glob import glob
from typing import Any, TypeVar

import lightning.pytorch as pl
import lightning.pytorch.callbacks as pl_callbacks
import lightning.pytorch.loggers as pl_loggers
import torch

logger = logging.getLogger(__name__)

LightningModuleT = TypeVar("LightningModuleT", bound=pl.LightningModule)


def load_best_model(
    trainer: pl.Trainer,
    model: LightningModuleT,
    reuse_version: bool = False,
    **kwargs: Any,
) -> LightningModuleT:
    """Set up the best previously trained model for testing.

    Args:
        trainer: The PyTorch Lightning trainer.
        model: An 'empty' instance of the LightningModule that has only been
            initialized.
        reuse_version: Whether to reuse the version of the logger that was
            used during training. This is useful to prevent TensorBoard from
            creating a new directory for the test results.
            Warning: This argument is not yet implemented.
        **kwargs: Additional keyword arguments needed to initialize the model,
            such as when the save_hyperparameters() method was never called
            or when save_hyperparameters(ignore=[...]) was used to ignore
            certain hyperparameters. Can also be used to override saved
            hyperparameter values.

    Returns:
        The trained model, loaded from the checkpoint.
    """
    if reuse_version:
        raise NotImplementedError(
            "The reuse_version argument is not yet implemented."
        )

    # Unfortunately, we cannot use the trainer.test(ckpt_path="best") method
    # without first having called trainer.fit(), since the path to the best
    # model is saved in the trainer.checkpoint_callback.best_model_path
    # attribute, which is only set during/after training. See this GitHub
    # issue for further details on why this is the default behaviour:
    # https://github.com/Lightning-AI/pytorch-lightning/issues/17312

    # Therefore, we have to load the best model ourselves. We could ask the
    # user to provide the path to the best model, but I think it is more
    # user-friendly to just try to infer the path to the best model from the
    # path to the model directory. To do this, we could load every checkpoint
    # and check them all for the best_model_score. Afterwards, we still need
    # to verify that the state_dict is compatible with the model. This is a
    # bit of a hack, but it is the best solution currently proposed in the
    # aforementioned GitHub issue.

    # Determine where the checkpoints are saved.
    if not isinstance(
        trainer.checkpoint_callback, pl_callbacks.ModelCheckpoint
    ):
        raise ValueError("The trainer has no checkpoint callback")
    dirpath = trainer.checkpoint_callback.dirpath
    list_of_checkpoints = glob(f"{dirpath}/*.ckpt")
    if len(list_of_checkpoints) == 0:
        raise ValueError(
            'ckpt_path="best" is set but ModelCheckpoint has not saved any'
            f" checkpoints to '{dirpath}'"
        )

    # Find information about each model checkpoint.
    # We only keep one model in memory at a time to prevent running out of RAM.
    # models_dict will be a dict with the following structure:
    # {
    #     (monitor, mode): [
    #         (best_model_score, creation_time, checkpoint, internal_state),
    #         ...
    #     ],
    #     ...
    # }
    # where:
    # - monitor: Metric that was monitored during training, e.g. 'loss/val'.
    # - mode: Mode that was used to monitor the metric, e.g. 'min' or 'max'.
    # - best_model_score: The best score of the model for the monitored metric.
    # - creation_time: Time when the checkpoint was created.
    # - checkpoint: Path to the checkpoint file as a string.
    # - internal_state: A dictionary containing the internal state of the
    #   ModelCheckpoint callback, which includes the best_model_score.
    map_location = trainer.strategy.root_device
    models_dict: dict[
        tuple[str, str], list[tuple[float, float, str, dict[str, Any]]]
    ] = defaultdict(list)
    for checkpoint in list_of_checkpoints:
        curr_model = torch.load(
            checkpoint, map_location=map_location, weights_only=True
        )
        for callback_name, internal_state in curr_model["callbacks"].items():
            if callback_name.startswith("ModelCheckpoint"):
                kwargs = eval(callback_name[len("ModelCheckpoint") :])
                models_dict[(kwargs["monitor"], kwargs["mode"])].append((
                    internal_state["best_model_score"],
                    os.path.getctime(checkpoint),
                    checkpoint,
                    internal_state,
                ))
                break
    if not models_dict:
        raise FileNotFoundError(
            "No checkpoints found with a ModelCheckpoint callback."
        )

    # Determine the monitored value and mode that were used to save the models,
    # e.g. ('loss/val', 'min') or ('acc/val', 'max'). There should only
    # be one unique combination of (monitor, mode) in the models_dict, and it
    # should be equal to the monitor and mode of the given Trainer's
    # ModelCheckpoint. If this is not the case, we will issue a warning or
    # error and use the most appropriate model.
    monitor_trainer = trainer.checkpoint_callback.monitor
    if monitor_trainer is None:
        raise ValueError(
            "The Trainer's ModelCheckpoint has no monitor set. Please set"
            " the monitor argument to a valid value."
        )
    monitor_mode_trainer = (monitor_trainer, trainer.checkpoint_callback.mode)
    if monitor_mode_trainer in models_dict:
        candidate_models = models_dict[monitor_mode_trainer]
        if len(models_dict) > 1:
            logger.warning(
                "Multiple saved models found, but their (monitor, mode)"
                " combinations are not compatible with each other. The saved"
                " (monitor, mode) combinations found are"
                f" {list(models_dict.keys())!r}. Using the Trainer's"
                f" ModelCheckpoint, which is {monitor_mode_trainer!r}."
            )
        monitor_mode = monitor_mode_trainer
    else:
        if len(models_dict) == 1:
            monitor_mode, candidate_models = models_dict.popitem()
            logger.error(
                "The saved model's (monitor, mode) combination does not match"
                " the Trainer's ModelCheckpoint. The saved (monitor, mode)"
                f" combination found is {monitor_mode!r}, but the Trainer's"
                f" ModelCheckpoint is {monitor_mode_trainer!r}. Using the"
                f" saved model, which is {monitor_mode!r}."
            )
        else:
            # Use the most recent model if none of the models are compatible.
            monitor_mode, candidate_models = max(
                models_dict.items(),
                key=lambda kv: max(
                    candidate_model[1] for candidate_model in kv[1]
                ),
            )
            logger.error(
                "Multiple saved models found, but none of their"
                " (monitor, mode) combinations match the Trainer's"
                " ModelCheckpoint. The saved (monitor, mode) combinations"
                f" found are {list(models_dict.keys())!r}, but the Trainer's"
                f" ModelCheckpoint is {monitor_mode_trainer!r}. Using the most"
                f" recenty saved model, which is {monitor_mode!r}."
            )

    # Sort the models by their best score, then by creation time.
    sorted_models = sorted(
        candidate_models,
        key=lambda candidate_model: (
            (
                candidate_model[0]
                if monitor_mode[1] == "max"
                else -candidate_model[0]
            ),
            candidate_model[1],
        ),
        reverse=True,
    )

    # Check if the best model's state_dict is compatible with the given model.
    expected_state_dict_keys = set(model.state_dict().keys())
    for best_model_score, ctime, checkpoint, internal_state in sorted_models:
        try:
            best_model = model.__class__.load_from_checkpoint(
                checkpoint, map_location=map_location, **kwargs
            )
        except (TypeError, RuntimeError) as e:
            logger.warning(
                f"Could not load model from '{checkpoint}'. This is likely due"
                " to a change in the model architecture or hyperparameters."
                " Skipping this checkpoint and trying the next one.\nOriginal"
                f" error message: {e}"
            )
            continue

        found_state_dict_keys = set(best_model.state_dict().keys())
        if expected_state_dict_keys != found_state_dict_keys:
            logger.warning(
                f"The state_dict of the model loaded from '{checkpoint}' is not"
                " compatible with the provided LightningModule. The keys in"
                " the state_dict are different. This is likely due to a change"
                " in the model architecture or hyperparameters. Skipping this"
                " checkpoint and trying the next one.\nExpected keys:"
                f" {expected_state_dict_keys}\nFound keys:"
                f" {found_state_dict_keys}"
            )
            continue

        logger.info(
            f"Found best model at '{checkpoint}' with"
            f" {monitor_mode[0]}={best_model_score}. File was created on"
            f" {time.ctime(ctime)}."
        )
        break
    else:
        raise FileNotFoundError(
            "No checkpoint found with a state dictionary compatible with the"
            " provided LightningModule. Note that we only searched through"
            f" models which had their (monitor, mode) set to {monitor_mode!r}."
        )

    # Hack the trainer to think that we have just trained a model.
    for key, value in internal_state.items():
        setattr(trainer.checkpoint_callback, key, value)

    # Set the logger to the version that was used during training.
    if reuse_version:
        if isinstance(trainer.logger, pl_loggers.TensorBoardLogger):
            # TODO Loop through all logger versions and determine which one was
            # TODO used during training. This is a bit tricky, since the logger
            # TODO version is not saved in the checkpoint.
            # trainer.logger = pl_loggers.TensorBoardLogger(
            #     save_dir=TODO,
            #     name=TODO,
            #     version=TODO,
            #     default_hp_metric=(
            #         trainer.logger._default_hp_metric,
            #     ),
            # )
            pass
        else:
            logger.warning(
                "The reuse_version argument is set, but the logger used"
                " during training is not supported yet. Thus, the logger "
                " version will not be reused."
            )

    # Return the best model.
    return best_model
