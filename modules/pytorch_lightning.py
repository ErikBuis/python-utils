import logging
import os
import time
from collections import defaultdict
from glob import glob
from typing import Any

import pytorch_lightning as pl
import pytorch_lightning.callbacks as pl_callbacks
import pytorch_lightning.loggers as pl_loggers
import torch


logger = logging.getLogger(__name__)


def load_best_model(
    trainer: pl.Trainer, model: pl.LightningModule, **kwargs: Any
) -> pl.LightningModule:
    """Set up the best previously trained model for testing.

    Args:
        trainer: The PyTorch Lightning trainer.
        model: An 'empty' instance of the LightningModule that has only been
            initialized.
        **kwargs: Additional keyword arguments needed to initialize the model,
            such as when the `save_hyperparameters()` method was never called
            or when `save_hyperparameters(ignore=[...])` was used to ignore
            certain hyperparameters. Can also be used to override saved
            hyperparameter values.

    Returns:
        The trained model, loaded from the checkpoint.
    """
    # Unfortunately, we cannot use the `trainer.test(ckpt_path="best")` method
    # without first having called `trainer.fit()`, since the path to the best
    # model is saved in the `trainer.checkpoint_callback.best_model_path`
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
            '`.test(ckpt_path="best")` is set but `ModelCheckpoint` has not'
            f" saved any checkpoints to '{dirpath}'"
        )

    # Find information about each model checkpoint.
    # We only keep one model in memory at a time to prevent running out of RAM.
    map_location = trainer.strategy.root_device
    models_dict: dict[tuple[str, str], dict[str, tuple[float, float]]] = (
        defaultdict(dict)
    )
    for checkpoint in list_of_checkpoints:
        curr_model = torch.load(checkpoint, map_location=map_location)
        for callback_name, keys in curr_model["callbacks"].items():
            if callback_name.startswith("ModelCheckpoint"):
                ckpt_kwargs = eval(callback_name[len("ModelCheckpoint") :])
                models_dict[(ckpt_kwargs["monitor"], ckpt_kwargs["mode"])][
                    keys["best_model_path"]
                ] = (
                    keys["best_model_score"],
                    os.path.getctime(keys["best_model_path"]),
                )
                break
    if not models_dict:
        raise ValueError(
            "No checkpoints found with a ModelCheckpoint callback."
        )

    # Determine the monitored value and mode that were used to save the models.
    # e.g. ('loss/val', 'min') or ('accuracy/val', 'max'). There should only
    # be one unique combination of (monitor, mode) in the models_dict, and it
    # should be equal to the monitor and mode of the current ModelCheckpoint.
    # If this is not the case, we will issue a warning or error and use the
    # most appropriate model.
    monitor_mode = (
        trainer.checkpoint_callback.monitor,
        trainer.checkpoint_callback.mode,
    )
    if monitor_mode in models_dict:
        candidate_models = models_dict[monitor_mode]
        if len(models_dict) > 1:
            logger.warning(
                "Multiple saved models found, but their monitored"
                " monitors/modes are not compatible with each other. Using"
                f" the Trainer's ModelCheckpoint, which is {monitor_mode!r}."
                " The (monitor, mode) combinations found were:"
                f" {list(models_dict.keys())!r}."
            )
    else:
        if len(models_dict) == 1:
            monitor_mode, candidate_models = models_dict.popitem()
            logger.error(
                "The saved model's (monitor, mode) combination does not match"
                f" the Trainer's ModelCheckpoint, which is {monitor_mode!r}."
                " The saved (monitor, mode) combination found was:"
                f" {list(models_dict.keys())!r}."
            )
        else:
            monitor_mode, candidate_models = max(
                models_dict.items(),
                key=lambda kv: max(kv[1].values(), key=lambda kv2: kv2[1])[1],
            )
            logger.error(
                "None of the saved models' (monitor, mode) combinations match"
                f" the Trainer's ModelCheckpoint, which is {monitor_mode!r}."
                " The saved (monitor, mode) combination(s) found were:"
                f" {list(models_dict.keys())!r}. Using the most recenty saved"
                f" model, which is {monitor_mode!r}."
            )

    # Sort the models by their best score and creation time.
    reverse = monitor_mode[1] == "min"
    if reverse:
        candidate_models = {
            path: (score, -ctime)
            for path, (score, ctime) in candidate_models.items()
        }
    sorted_models = sorted(
        candidate_models.items(), key=lambda kv: kv[1], reverse=reverse
    )

    # Check if the best model's state_dict is compatible with the given model.
    for path, (score, ctime) in sorted_models:
        best_model = model.__class__.load_from_checkpoint(
            path, map_location=map_location, **kwargs
        )
        if set(model.state_dict().keys()) == set(
            best_model.state_dict().keys()
        ):
            logger.info(
                f"Found best model at '{path}' with {monitor_mode[0]}={score}."
                " File was created on"
                f" {time.ctime(-ctime if reverse else ctime)}."
            )
            break
    else:
        raise ValueError(
            "No checkpoint found with a state dictionary compatible with the"
            " provided LightningModule. Note that we only searched through"
            f" models which had their (monitor, mode) set to {monitor_mode!r}."
        )

    # Hack the Trainer to think that we have just trained a model.
    trainer.checkpoint_callback.best_model_path = path

    # Set the logger to the version that was used during training.
    if isinstance(trainer.logger, pl_loggers.TensorBoardLogger):
        # TODO Loop through all logger versions and determine which one was
        # TODO used during training. This is a bit tricky, since the logger
        # TODO version is not saved in the checkpoint.
        # trainer.logger = pl_loggers.TensorBoardLogger(
        #     save_dir=TODO,
        #     name=TODO,
        #     version=TODO,
        #     default_hp_metric=(
        #         trainer.logger._default_hp_metric,  # type: ignore
        #     ),
        # )
        pass

    # Return the best model.
    return best_model
