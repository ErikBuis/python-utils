# pyright: reportMissingImports=false

from __future__ import annotations

import argparse
import inspect
import logging
import os
import time
import types
from collections import defaultdict
from collections.abc import Sequence
from glob import glob
from typing import Any, cast

import lightning.pytorch as pl
import lightning.pytorch.callbacks as pl_callbacks
import torch
from typing_extensions import override

try:
    from omegaconf import OmegaConf
except ImportError:
    OmegaConf = None

logger = logging.getLogger(__name__)
logging_logger = logger


class OnlySaveDirectHyperparameters(pl.LightningModule):
    """Class that only saves hparams directly passed to the class' constructor.

    This class is a workaround for the issue that PyTorch Lightning's
    save_hyperparameters() method saves all arguments passed to the constructor,
    including those from super- and subclasses. This can lead to bugs when
    some (keyword)arguments are passed on to the constructor of a superclass via
    *args or **kwargs, since the superclass might want to ignore some of those
    arguments when saving hyperparameters. This class allows you to only save
    the hyperparameters that are directly passed to the constructor of the
    class that calls the save_hyperparameters() method, while still allowing
    you to pass *args and **kwargs to the superclass constructor.

    Examples:
    >>> class MyParentModel(OnlySaveDirectHyperparameters):
    ...     def __init__(self, arg1: str, arg2: int, *args, **kwargs):
    ...         super().__init__(*args, **kwargs)
    ...         self.save_hyperparameters("arg2")
    ...
    >>> class MyModel(MyParentModel):
    ...     def __init__(self, arg3: float, arg4: bool, *args, **kwargs):
    ...         super().__init__(*args, **kwargs)
    ...         self.save_hyperparameters(ignore="arg4")
    ...
    >>> my_model = MyModel(
    ...     arg1="test_string",
    ...     arg2=42,
    ...     arg3=3.14,
    ...     arg4=True,
    ...     some_other_kwarg="ignored",
    ... )
    >>> my_model.hparams
    {'arg2': '42', 'arg3': 3.14}
    """

    @override
    def save_hyperparameters(
        self,
        *args: Any,
        ignore: Sequence[str] | str | None = None,
        frame: types.FrameType | None = None,
        logger: bool = True,
    ) -> None:
        """Only save hparams directly passed to the calling class' constructor.

        Args:
            *args: Arguments to save as hyperparameters. Can be a single object
                of type dict, NameSpace or OmegaConf specifying a mapping from
                argument names to values, or strings representing argument names
                from your class' __init__().
            ignore: An argument name or list of argument names from class
                `__init__` to be ignored.
            frame: The frame from which to extract the calling class'
                constructor arguments. If not provided, the frame corresponding
                to the function calling this method will be used.
            logger: Whether to log the hyperparameters being saved to the
                trainer's logger.
        """
        # Get the frame that called this function.
        if not frame:
            current_frame = inspect.currentframe()
            if current_frame:
                frame = current_frame.f_back
        if not isinstance(frame, types.FrameType):
            raise AttributeError(
                "There is no `frame` available while being required."
            )

        # Get the arguments of the calling class' constructor.
        _, _, _, local_vars = inspect.getargvalues(frame)
        cls = local_vars["__class__"]
        parameters = dict(inspect.signature(cls.__init__).parameters)

        # Ignore the self, *args, and **kwargs arguments.
        ignore_args = {"self", "args", "kwargs"}
        for arg in ignore_args:
            parameters.pop(arg, None)

        # Convert the provided args into a dictionary or list of strings.
        if not args:
            extracted_args = {}
        elif len(args) == 1:
            arg_first = args[0]
            if OmegaConf is not None and isinstance(arg_first, OmegaConf):
                container = OmegaConf.to_container(arg_first, resolve=True)
                if container is None:
                    extracted_args = {}
                elif isinstance(container, list):
                    extracted_args = container
                elif isinstance(container, dict):
                    extracted_args = cast(dict[str, Any], container)
                elif isinstance(container, tuple):
                    extracted_args = list(container)
                elif isinstance(container, str):
                    extracted_args = [container]
                else:
                    raise TypeError(
                        f"Unsupported type for OmegaConf: {type(container)}."
                        " Expected dict, list, tuple, or str."
                    )
            elif isinstance(arg_first, dict):
                extracted_args = arg_first
            elif isinstance(arg_first, argparse.Namespace):
                extracted_args = arg_first.__dict__
            else:
                extracted_args = [arg_first]
        elif all(isinstance(arg, str) for arg in args):
            extracted_args = cast(list[str], list(args))
        else:
            raise TypeError(
                f"Unsupported type for args: {type(args)}. Expected dict,"
                " argparse.Namespace, OmegaConf, str, or sequence of strings."
            )

        # Only include arguments the user specified.
        if not extracted_args:
            direct_args = parameters
        elif isinstance(extracted_args, dict):
            direct_args = {}
            for arg, value in extracted_args.items():
                if arg in parameters:
                    direct_args[arg] = value
                else:
                    logging_logger.error(
                        f"Argument '{arg}' specified in `*args` but not found"
                        f" in the constructor of `{cls.__name__}`, which only"
                        f" has {list(parameters.keys())}."
                    )
        else:
            direct_args = []
            for arg in extracted_args:
                if arg in parameters:
                    direct_args.append(arg)
                else:
                    logging_logger.error(
                        f"Argument '{arg}' specified in `*args` but not found"
                        f" in the constructor of `{cls.__name__}`, which only"
                        f" has {list(parameters.keys())}."
                    )

        # Filter out arguments the user wants to ignore.
        if ignore is not None:
            if isinstance(ignore, str):
                ignore = [ignore]
            for ignored_arg in ignore:
                try:
                    if isinstance(direct_args, dict):
                        del direct_args[ignored_arg]
                    else:
                        direct_args.remove(ignored_arg)
                except KeyError:
                    logging_logger.error(
                        f"Argument '{ignored_arg}' specified in `ignore` but"
                        f" not found in the constructor of `{cls.__name__}`,"
                        f" which only has {list(parameters.keys())}."
                    )

        return super().save_hyperparameters(
            *direct_args, frame=frame, logger=logger
        )


def prepare_best_model_loading(
    trainer: pl.Trainer, model: pl.LightningModule, **kwargs: Any
) -> None:
    """Prepare the PyTorch Lightning trainer to load the best model.

    This function is used to load the best model from the checkpoints saved
    during training. It is useful when you want to test/validate/predict using
    the best model after training, without having to specfy the path to the
    best model manually. It will automatically find the best model based on
    the monitor and mode used in the ModelCheckpoint callback of the trainer.

    After calling this function, you can call the `trainer.test()`,
    `trainer.validate()`, or `trainer.predict()` methods with the
    `ckpt_path="best"` argument to let PyTorch Lightning load the best
    model automatically.

    Args:
        trainer: The PyTorch Lightning trainer.
        model: An 'empty' instance of the LightningModule that has only been
            initialized.
        **kwargs: Additional keyword arguments needed to initialize the model,
            such as when the save_hyperparameters() method was never called
            or when save_hyperparameters(ignore=[...]) was used to ignore
            certain hyperparameters. Can also be used to override saved
            hyperparameter values.
    """
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
    ckpt_path_candidates = glob(f"{dirpath}/*.ckpt")
    if len(ckpt_path_candidates) == 0:
        raise ValueError(
            "ckpt_path='best' is set but ModelCheckpoint has not saved any"
            f" checkpoints to '{dirpath}'"
        )

    # Find information about each model checkpoint.
    # We only keep one model in memory at a time to prevent running out of RAM.
    # models_dict will be a dict with the following structure:
    # {
    #     (monitor, mode): [
    #         (best_model_score, creation_time, ckpt_path),
    #         ...
    #     ],
    #     ...
    # }
    # where:
    # - monitor: Metric that was monitored during training, e.g. 'loss/val'.
    # - mode: Mode that was used to monitor the metric, e.g. 'min' or 'max'.
    # - best_model_score: The best score of the model for the monitored metric.
    # - creation_time: Time when the checkpoint was created.
    # - ckpt_path: Path to the checkpoint file as a string.
    models_dict: dict[
        tuple[str, str], list[tuple[float | None, float, str]]
    ] = defaultdict(list)
    for ckpt_path in ckpt_path_candidates:
        curr_model = torch.load(
            ckpt_path, map_location="cpu", weights_only=True
        )
        for callback_name, internal_state in curr_model["callbacks"].items():
            if callback_name.startswith("ModelCheckpoint"):
                curr_kwargs = eval(callback_name[len("ModelCheckpoint") :])
                models_dict[
                    (curr_kwargs["monitor"], curr_kwargs["mode"])
                ].append((
                    internal_state["best_model_score"],
                    os.path.getctime(ckpt_path),
                    ckpt_path,
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
                "Multiple saved models found, but none of their (monitor,"
                " mode) combinations match the Trainer's ModelCheckpoint. The"
                " saved (monitor, mode) combinations found are"
                f" {list(models_dict.keys())!r}, but the Trainer's"
                f" ModelCheckpoint is {monitor_mode_trainer!r}. Using the most"
                f" recenty saved model, which is {monitor_mode!r}."
            )

    # At this point, we have a list of candidate models that match the
    # (monitor, mode) combination of the Trainer's ModelCheckpoint. We
    # will now filter out models that have no best_model_score set and warn
    # the user about it.
    for i, (best_model_score, _, ckpt_path) in reversed(
        list(enumerate(candidate_models))
    ):
        if best_model_score is None:
            logger.warning(
                f"Skipping checkpoint at '{ckpt_path}' because it has no"
                " `best_model_score`. If you want to use this checkpoint"
                " anyway, please specify the path to the checkpoint manually"
                " using `ckpt_path='path/to/your/model.ckpt'`."
            )
            del candidate_models[i]
    if not candidate_models:
        raise FileNotFoundError(
            "No checkpoints found with a valid `best_model_score` for the"
            f" (monitor, mode) combination {monitor_mode!r}. Note that we only"
            " searched through models which had their (monitor, mode) set to"
            f" {monitor_mode!r}."
        )
    candidate_models = cast(list[tuple[float, float, str]], candidate_models)

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
    for best_model_score, ctime, ckpt_path in sorted_models:
        try:
            best_model = model.__class__.load_from_checkpoint(
                ckpt_path, map_location=trainer.strategy.root_device, **kwargs
            )
        except (TypeError, RuntimeError) as e:
            logger.warning(
                f"Could not load model from '{ckpt_path}'. This is likely due"
                " to a change in the model architecture or hyperparameters."
                " Skipping this checkpoint and trying the next one.\nOriginal"
                f" error message: {e}"
            )
            continue

        found_state_dict_keys = set(best_model.state_dict().keys())
        if expected_state_dict_keys != found_state_dict_keys:
            logger.warning(
                f"The state_dict of the model loaded from '{ckpt_path}' is not"
                " compatible with the provided LightningModule. The keys in"
                " the state_dict are different. This is likely due to a change"
                " in the model architecture or hyperparameters. Skipping this"
                " checkpoint and trying the next one.\nExpected keys:"
                f" {expected_state_dict_keys}\nFound keys:"
                f" {found_state_dict_keys}"
            )
            continue

        logger.info(
            f"Found best model at '{ckpt_path}' with"
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
    trainer.checkpoint_callback.best_model_path = ckpt_path
