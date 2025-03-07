"""
File for calculating various metrics for binary classifiers.

The main question this file answers is: If a random model that predicts
positive with a fixed probability is used, what would its metrics be on
average?
"""

import argparse
import math
import sys
from typing import Literal, cast, overload

from .cheatsheet import (
    ternary_search_continuous_func_left,
    ternary_search_continuous_func_right,
)


def calculate_metrics_binary(
    TP: float, FN: float, FP: float, TN: float
) -> dict[str, float]:
    """Calculate various metrics based on the given confusion matrix.

    Args:
        TP: The number/proportion of true positives.
        FN: The number/proportion of false negatives.
        FP: The number/proportion of false positives.
        TN: The number/proportion of true negatives.

    Returns:
        Dictionary containing several metrics for a binary classifier. Each
        metric is a float or NaN if it is undefined.
    """
    # Calculate accuracy.
    acc = (
        (TP + TN) / (TP + FN + FP + TN)
        if not math.isclose(TP + FN + FP + TN, 0)
        else math.nan
    )

    # Calculate precision, recall, and F1-score for both classes.
    precision_pos = (
        TP / (TP + FP) if not math.isclose(TP + FP, 0) else math.nan
    )
    recall_pos = TP / (TP + FN) if not math.isclose(TP + FN, 0) else math.nan
    F1_pos = (
        2 * TP / (2 * TP + FN + FP)
        if not math.isclose(TP + FN + FP, 0)
        else math.nan
    )
    precision_neg = (
        TN / (TN + FN) if not math.isclose(TN + FN, 0) else math.nan
    )
    recall_neg = TN / (TN + FP) if not math.isclose(TN + FP, 0) else math.nan
    F1_neg = (
        2 * TN / (2 * TN + FN + FP)
        if not math.isclose(TN + FN + FP, 0)
        else math.nan
    )

    # Calculate macro-averaged metrics (=each class is weighted equally).
    precision_macro = (precision_pos + precision_neg) / 2
    recall_macro = (recall_pos + recall_neg) / 2
    F1_macro = (F1_pos + F1_neg) / 2

    # Calculate mirco-averaged metrics (=each sample is weighted equally).
    precision_micro = acc
    recall_micro = acc
    F1_micro = acc

    # Calculate balanced accuracy.
    balanced_acc = recall_macro

    # Calculate Matthews correlation coefficient.
    MCC = (
        (TP * TN - FN * FP)
        / math.sqrt((TP + FN) * (TP + FP) * (FN + TN) * (FP + TN))
        if not math.isclose((TP + FN) * (TP + FP) * (FN + TN) * (FP + TN), 0)
        else math.nan
    )

    # Calculate Cohen's Kappa.
    Kappa = (
        (2 * (TP * TN - FN * FP))
        / ((TP + FN) * (FN + TN) + (TP + FP) * (FP + TN))
        if not math.isclose((TP + FN) * (FN + TN) + (TP + FP) * (FP + TN), 0)
        else math.nan
    )

    return {
        "acc": acc,
        "precision_pos": precision_pos,
        "recall_pos": recall_pos,
        "F1_pos": F1_pos,
        "precision_neg": precision_neg,
        "recall_neg": recall_neg,
        "F1_neg": F1_neg,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "F1_macro": F1_macro,
        "precision_micro": precision_micro,
        "recall_micro": recall_micro,
        "F1_micro": F1_micro,
        "balanced_acc": balanced_acc,
        "MCC": MCC,
        "Kappa": Kappa,
    }


def confusion_matrix_random_model(
    frac_labels_pos: float, frac_preds_pos: float
) -> tuple[float, float, float, float]:
    """Calculate the confusion matrix for a random model.

    Args:
        frac_labels_pos: The fraction of positive labels.
        frac_preds_pos: The fraction of positive predictions.

    Returns:
        The confusion matrix as a tuple of floats (TP, FN, FP, TN).
        Satisfies the following conditions:
        - TP + FN + FP + TN = 1
        - (TP + FN) / (TP + FN + FP + TN) = frac_labels_pos
        - (TP + FP) / (TP + FN + FP + TN) = frac_preds_pos
    """
    TP = frac_labels_pos * frac_preds_pos
    FN = frac_labels_pos * (1 - frac_preds_pos)
    FP = (1 - frac_labels_pos) * frac_preds_pos
    TN = (1 - frac_labels_pos) * (1 - frac_preds_pos)

    return TP, FN, FP, TN


def confusion_matrix_best_model(
    frac_labels_pos: float, frac_preds_pos: float
) -> tuple[float, float, float, float]:
    """Calculate the confusion matrix for the best possible model.

    Args:
        frac_labels_pos: The fraction of positive labels.
        frac_preds_pos: The fraction of positive predictions.

    Returns:
        The confusion matrix as a tuple of floats (TP, FN, FP, TN).
        Satisfies the following conditions:
        - TP + FN + FP + TN = 1
        - (TP + FN) / (TP + FN + FP + TN) = frac_labels_pos
        - (TP + FP) / (TP + FN + FP + TN) = frac_preds_pos
    """
    TP = min(frac_labels_pos, frac_preds_pos)
    TN = min((1 - frac_labels_pos), (1 - frac_preds_pos))
    FN = (1 - frac_preds_pos) - TN
    FP = frac_preds_pos - TP

    return TP, FN, FP, TN


def confusion_matrix_worst_model(
    frac_labels_pos: float, frac_preds_pos: float
) -> tuple[float, float, float, float]:
    """Calculate the confusion matrix for the worst possible model.

    Args:
        frac_labels_pos: The fraction of positive labels.
        frac_preds_pos: The fraction of positive predictions.

    Returns:
        The confusion matrix as a tuple of floats (TP, FN, FP, TN).
        Satisfies the following conditions:
        - TP + FN + FP + TN = 1
        - (TP + FN) / (TP + FN + FP + TN) = frac_labels_pos
        - (TP + FP) / (TP + FN + FP + TN) = frac_preds_pos
    """
    FN = min(frac_labels_pos, (1 - frac_preds_pos))
    FP = min((1 - frac_labels_pos), frac_preds_pos)
    TP = frac_preds_pos - FP
    TN = (1 - frac_preds_pos) - FN

    return TP, FN, FP, TN


@overload
def determine_extreme_model_metric(
    frac_labels_pos: float,
    frac_preds_pos: float,
    metric_to_analyze: str,
    model: Literal["random", "best", "worst"],
    extreme: Literal["best", "worst"],
) -> tuple[float, float, float]:
    pass


@overload
def determine_extreme_model_metric(
    frac_labels_pos: float,
    frac_preds_pos: None,
    metric_to_analyze: str,
    model: Literal["random", "best", "worst"],
    extreme: Literal["best", "worst"],
) -> tuple[float, float, float]:
    pass


@overload
def determine_extreme_model_metric(
    frac_labels_pos: None,
    frac_preds_pos: float,
    metric_to_analyze: str,
    model: Literal["random", "best", "worst"],
    extreme: Literal["best", "worst"],
) -> tuple[float, float, float]:
    pass


def determine_extreme_model_metric(
    frac_labels_pos: float | None,
    frac_preds_pos: float | None,
    metric_to_analyze: str,
    model: Literal["random", "best", "worst"],
    extreme: Literal["best", "worst"],
) -> tuple[float, float, float]:
    """Calculate the metrics for the best/worst possible model.

    Args:
        frac_labels_pos: The fraction of positive labels.
        frac_preds_pos: The fraction of positive predictions.
        metric_to_analyze: The metric to analyze.
        model: The model to calculate the metrics for (random, best, worst).
        extreme: For which extreme to calculate the metrics (best, worst).

    Returns:
        Tuple containing:
        - The metric score.
        - The left boundary of the range of fractions of positive labels/
            predictions that would yield the best score. If frac_labels_pos and
            frac_preds_pos are given, this value is NaN.
        - The right boundary of the range of fractions of positive labels/
            predictions that would yield the best score. If frac_labels_pos and
            frac_preds_pos are given, this value is NaN.
    """
    confusion_matrix_func = {
        "random": confusion_matrix_random_model,
        "best": confusion_matrix_best_model,
        "worst": confusion_matrix_worst_model,
    }[model]

    def calc_metrics_func(
        frac_labels_pos: float, frac_preds_pos: float
    ) -> float:
        return round(
            calculate_metrics_binary(
                *confusion_matrix_func(frac_labels_pos, frac_preds_pos)
            )[metric_to_analyze],
            12,
        )  # round to prevent floating point errors

    if frac_labels_pos is None or frac_preds_pos is None:
        extreme_factor = -1 if extreme == "best" else 1

        def calc_metrics_func_partial(
            frac_pos: float,
        ) -> float:  # type: ignore
            if frac_labels_pos is None:
                return (
                    calc_metrics_func(frac_pos, frac_preds_pos)  # type: ignore
                    * extreme_factor
                )
            elif frac_preds_pos is None:
                return (
                    calc_metrics_func(frac_labels_pos, frac_pos)
                    * extreme_factor
                )

        frac_pos_left = ternary_search_continuous_func_left(
            calc_metrics_func_partial, 0, 1
        )
        frac_pos_right = ternary_search_continuous_func_right(
            calc_metrics_func_partial, frac_pos_left, 1
        )

        if frac_labels_pos is None:
            frac_labels_pos = (frac_pos_left + frac_pos_right) / 2
        elif frac_preds_pos is None:
            frac_preds_pos = (frac_pos_left + frac_pos_right) / 2

        return (
            calc_metrics_func(frac_labels_pos, frac_preds_pos),  # type: ignore
            frac_pos_left,
            frac_pos_right,
        )
    else:
        return (
            calc_metrics_func(frac_labels_pos, frac_preds_pos),
            math.nan,
            math.nan,
        )


@overload
def log_extreme_model_metric(
    frac_labels_pos: float, frac_preds_pos: float, metric_to_analyze: str
) -> None:
    pass


@overload
def log_extreme_model_metric(
    frac_labels_pos: float, frac_preds_pos: None, metric_to_analyze: str
) -> None:
    pass


@overload
def log_extreme_model_metric(
    frac_labels_pos: None, frac_preds_pos: float, metric_to_analyze: str
) -> None:
    pass


def log_extreme_model_metric(
    frac_labels_pos: float | None,
    frac_preds_pos: float | None,
    metric_to_analyze: str,
) -> None:
    """Log the metrics for a random/best/worst model.

    Args:
        frac_labels_pos: The fraction of positive labels.
        frac_preds_pos: The fraction of positive predictions.
        metric_to_analyze: The metric to analyze.
    """
    frac_type = (
        "frac_labels_pos" if frac_labels_pos is not None else "frac_preds_pos"
    )

    logger.info("")
    logger.info(
        f"The {metric_to_analyze} for the best random, best overall, and worst"
        " overall models are:"
    )

    # Calculate the score of the best random model.
    metric, frac_pos_left, frac_pos_right = determine_extreme_model_metric(
        frac_labels_pos,  # type: ignore
        frac_preds_pos,  # type: ignore
        metric_to_analyze,
        "random",
        "best",
    )
    if frac_labels_pos is not None and frac_preds_pos is not None:
        logger.info("\t1. A random model would get:")
    else:
        if math.isclose(frac_pos_left, frac_pos_right, abs_tol=1e-5):
            logger.info(
                "\t1. The best random model would be achieved when"
                f" {frac_type} = {frac_pos_left:.5f}. It would"
                " get the best score out of all possible random models:"
            )
        else:
            logger.info(
                "\t1. The best random model would be achieved when"
                f" {frac_type} is anywhere in the range"
                f" [{frac_pos_left:.5f}, {frac_pos_right:.5f}]. It would get"
                " the best score out of all possible random models:"
            )
    logger.info(f"\t\t{metric_to_analyze} = {metric:.5f}")

    # Calculate the score of the best possible model.
    metric, frac_pos_left, frac_pos_right = determine_extreme_model_metric(
        frac_labels_pos,  # type: ignore
        frac_preds_pos,  # type: ignore
        metric_to_analyze,
        "best",
        "best",
    )
    if frac_labels_pos is not None and frac_preds_pos is not None:
        logger.info("\t2. The best model would get:")
    else:
        if math.isclose(frac_pos_left, frac_pos_right, abs_tol=1e-5):
            logger.info(
                "\t2. The best model would be achieved when"
                f" {frac_type} = {frac_pos_left:.5f}. It would"
                " get the best score out of all possible models:"
            )
        else:
            logger.info(
                "\t2. The best model would be achieved when"
                f" {frac_type} is anywhere in the range"
                f" [{frac_pos_left:.5f}, {frac_pos_right:.5f}]. It would get"
                "the best score out of all possible models:"
            )
    logger.info(f"\t\t{metric_to_analyze} = {metric:.5f}")

    # Calculate the score of the worst possible model.
    metric, frac_pos_left, frac_pos_right = determine_extreme_model_metric(
        frac_labels_pos,  # type: ignore
        frac_preds_pos,  # type: ignore
        metric_to_analyze,
        "worst",
        "worst",
    )
    if frac_labels_pos is not None and frac_preds_pos is not None:
        logger.info("\t3. The worst model would get:")
    else:
        if math.isclose(frac_pos_left, frac_pos_right, abs_tol=1e-5):
            logger.info(
                "\t3. The worst model would be achieved when"
                f" {frac_type} = {frac_pos_left:.5f}. It would"
                " get the worst score out of all possible models:"
            )
        else:
            logger.info(
                "\t3. The worst model would be achieved when"
                f" {frac_type} is anywhere in the range"
                f" [{frac_pos_left:.5f}, {frac_pos_right:.5f}]. It would get"
                " the worst score out of all possible models:"
            )
    logger.info(f"\t\t{metric_to_analyze} = {metric:.5f}")


def main(args: argparse.Namespace) -> None:
    """The entry point of the program.

    Args:
        args: The command line arguments.
    """
    import matplotlib.colors
    import matplotlib.patches
    import matplotlib.pyplot as plt
    import matplotlib.tri
    import mpl_toolkits.mplot3d
    import numpy as np

    from ..modules.math import optimal_grid_layout, optimal_size
    from ..modules.numpy import NDArrayGeneric

    # If the user gave a specific fraction of positive labels and/or
    # predictions, calculate the metrics.
    if not (args.frac_labels_pos is None and args.frac_preds_pos is None):
        logger.info("Using the following assumptions:")
        if args.frac_labels_pos is not None:
            logger.info(f"\tfrac_labels_pos = {args.frac_labels_pos}")
        if args.frac_preds_pos is not None:
            logger.info(f"\tfrac_preds_pos = {args.frac_preds_pos}")

        for metric_to_analyze in args.metrics_to_analyze:
            log_extreme_model_metric(
                args.frac_labels_pos, args.frac_preds_pos, metric_to_analyze
            )

        return

    # Define whether to calculate/plot the worst, random, and best models.
    calculate_metrics_worst = False
    calculate_metrics_random = False
    calculate_metrics_best = False
    show_worst = False
    show_random = False
    show_best = False
    show_diff = False
    show_average = False
    if args.show_diff:
        calculate_metrics_worst = True
        calculate_metrics_best = True
        show_diff = True
    elif args.show_average:
        calculate_metrics_worst = True
        calculate_metrics_random = True
        calculate_metrics_best = True
        show_random = True
        show_average = True
    else:
        calculate_metrics_worst = args.show_best_worst
        calculate_metrics_random = True
        calculate_metrics_best = args.show_best_worst
        show_worst = args.show_best_worst
        show_random = True
        show_best = args.show_best_worst

    # Calculate the metric scores for a range of fractions of positive labels
    # and predictions. Use a cross-like pattern to prevent artifacts due to
    # triangulation.
    resolution = 5
    frac_labels_poss_even = np.linspace(0, 1, resolution)  # [R]
    frac_preds_poss_even = np.linspace(0, 1, resolution)  # [R]
    frac_labels_poss_even[0] = 1e-5  # Prevent division by zero.
    frac_labels_poss_even[-1] = 1 - 1e-5  # Prevent division by zero.
    frac_preds_poss_even[0] = 1e-5  # Prevent division by zero.
    frac_preds_poss_even[-1] = 1 - 1e-5  # Prevent division by zero.
    frac_labels_poss_even, frac_preds_poss_even = np.meshgrid(
        frac_labels_poss_even, frac_preds_poss_even
    )  # [R, R], [R, R]
    frac_labels_poss_even = frac_labels_poss_even.flatten()  # [R^2]
    frac_preds_poss_even = frac_preds_poss_even.flatten()  # [R^2]
    frac_labels_poss_odd = np.linspace(0, 1, resolution * 2 - 1)[1::2]  # [R-1]
    frac_preds_poss_odd = np.linspace(0, 1, resolution * 2 - 1)[1::2]  # [R-1]
    frac_labels_poss_odd, frac_preds_poss_odd = np.meshgrid(
        frac_labels_poss_odd, frac_preds_poss_odd
    )  # [R-1, R-1], [R-1, R-1]
    frac_labels_poss_odd = frac_labels_poss_odd.flatten()  # [(R-1)^2]
    frac_preds_poss_odd = frac_preds_poss_odd.flatten()  # [(R-1)^2]
    frac_labels_poss = np.concat(
        [frac_labels_poss_even, frac_labels_poss_odd]
    )  # [R^2 + (R-1)^2]
    frac_preds_poss = np.concat(
        [frac_preds_poss_even, frac_preds_poss_odd]
    )  # [R^2 + (R-1)^2]

    metrics_shape = (
        resolution**2 + (resolution - 1) ** 2,
        len(args.metrics_to_analyze),
    )
    metrics_worst = np.empty(metrics_shape)  # [R^2 + (R-1)^2, M]
    metrics_random = np.empty(metrics_shape)  # [R^2 + (R-1)^2, M]
    metrics_best = np.empty(metrics_shape)  # [R^2 + (R-1)^2, M]

    if calculate_metrics_worst:
        for i, (frac_labels_pos, frac_preds_pos) in enumerate(
            zip(frac_labels_poss, frac_preds_poss)
        ):
            TP, FN, FP, TN = confusion_matrix_worst_model(
                frac_labels_pos, frac_preds_pos
            )
            metrics = calculate_metrics_binary(TP, FN, FP, TN)
            metrics_worst[i] = [
                metrics[metric_to_analyze]
                for metric_to_analyze in args.metrics_to_analyze
            ]  # [M]
    if calculate_metrics_random:
        for i, (frac_labels_pos, frac_preds_pos) in enumerate(
            zip(frac_labels_poss, frac_preds_poss)
        ):
            TP, FN, FP, TN = confusion_matrix_random_model(
                frac_labels_pos, frac_preds_pos
            )
            metrics = calculate_metrics_binary(TP, FN, FP, TN)
            metrics_random[i] = [
                metrics[metric_to_analyze]
                for metric_to_analyze in args.metrics_to_analyze
            ]  # [M]
    if calculate_metrics_best:
        for i, (frac_labels_pos, frac_preds_pos) in enumerate(
            zip(frac_labels_poss, frac_preds_poss)
        ):
            TP, FN, FP, TN = confusion_matrix_best_model(
                frac_labels_pos, frac_preds_pos
            )
            metrics = calculate_metrics_binary(TP, FN, FP, TN)
            metrics_best[i] = [
                metrics[metric_to_analyze]
                for metric_to_analyze in args.metrics_to_analyze
            ]  # [M]

    # Plot the metrics in 3D space.
    ncols, nrows = optimal_grid_layout(
        len(args.metrics_to_analyze), max_rows=4
    )
    size_x, size_y = optimal_size(ncols / nrows, max_size_x=15, max_size_y=10)
    _, axs = plt.subplots(
        nrows,
        ncols,
        figsize=(size_x, size_y),
        subplot_kw={"projection": "3d"},
        squeeze=False,
    )
    axs = cast(
        NDArrayGeneric[mpl_toolkits.mplot3d.Axes3D], axs.reshape(-1)
    )  # [M]

    for i, metric_to_analyze in enumerate(args.metrics_to_analyze):
        ax = axs[i]
        metric_best = metrics_best[:, i]  # [R^2 + (R-1)^2]
        metric_random = metrics_random[:, i]  # [R^2 + (R-1)^2]
        metric_worst = metrics_worst[:, i]  # [R^2 + (R-1)^2]

        # Plot metric scores of the worst model as a red wireframe.
        if show_worst:
            ax.plot_trisurf(
                frac_labels_poss,  # [R^2 + (R-1)^2]
                frac_preds_poss,  # [R^2 + (R-1)^2]
                metric_worst,  # [R^2 + (R-1)^2]
                cmap=matplotlib.colormaps.get_cmap("Reds"),
                edgecolors="darkred",
                alpha=0.3,
                label="Worst model",
            )

        # Plot the metric scores of the random model as a colored wireframe.
        if show_random:
            ax.plot_trisurf(
                frac_labels_poss,  # [R^2 + (R-1)^2]
                frac_preds_poss,  # [R^2 + (R-1)^2]
                metric_random,  # [R^2 + (R-1)^2]
                cmap=matplotlib.colormaps.get_cmap("Blues"),
                edgecolors="darkblue",
                alpha=0.3,
                label="Random model",
            )

        # Plot the metric scores of the best model as a green wireframe.
        if show_best:
            ax.plot_trisurf(
                frac_labels_poss,  # [R^2 + (R-1)^2]
                frac_preds_poss,  # [R^2 + (R-1)^2]
                metric_best,  # [R^2 + (R-1)^2]
                cmap=matplotlib.colormaps.get_cmap("Greens"),
                edgecolors="darkgreen",
                alpha=0.3,
                label="Best model",
            )

        # Plot the diff between the best and worst models as a black wireframe.
        if show_diff:
            metric_diff = metric_best - metric_worst  # [R^2 + (R-1)^2]
            ax.plot_trisurf(
                frac_labels_poss,  # [R^2 + (R-1)^2]
                frac_preds_poss,  # [R^2 + (R-1)^2]
                metric_diff,  # [R^2 + (R-1)^2]
                cmap=matplotlib.colormaps.get_cmap("Greys"),
                edgecolors="black",
                alpha=0.3,
                label="Difference (Best - Worst)",
            )

        # Plot the avg of the best and worst models as a purple wireframe.
        if show_average:
            metric_average = (
                metric_best + metric_worst
            ) / 2  # [R^2 + (R-1)^2]
            ax.plot_trisurf(
                frac_labels_poss,  # [R^2 + (R-1)^2]
                frac_preds_poss,  # [R^2 + (R-1)^2]
                metric_average,  # [R^2 + (R-1)^2]
                cmap=matplotlib.colormaps.get_cmap("Purples"),
                edgecolors="purple",
                alpha=0.3,
                label="Average model (Best + Worst) / 2",
            )

        # Customize labels and title.
        ax.set_title(metric_to_analyze)
        ax.set_xlabel("frac_labels_pos")
        ax.set_ylabel("frac_preds_pos")
        ax.set_zlabel(metric_to_analyze)

        # Customize limits. Prevent artifacts due to floating point errors.
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        curr_zlim_min, curr_zlim_max = ax.get_zlim()
        ax.set_zlim(min(curr_zlim_min, 0), max(curr_zlim_max, 1e-5))

        # Customize the view angle.
        ax.view_init(elev=15, azim=-115)

        # Add a legend.
        legend = ax.legend(loc="upper right")
        for i, handle in enumerate(legend.legend_handles):
            handle = cast(matplotlib.patches.Rectangle, handle)
            edgecolor = handle.get_edgecolor()
            handle.set_edgecolor(edgecolor)
            handle.set_facecolor(
                matplotlib.colors.to_rgba(edgecolor, alpha=0.2)
            )
            handle.set_linewidth(1)

    # Show the plot.
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    from loguru import logger

    # Create the argument parser.
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Define command line arguments.
    parser.add_argument(
        "--logging_level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="The logging level to use.",
    )
    parser.add_argument(
        "--frac_labels_pos",
        type=float,
        default=None,
        help="The fraction of positive labels.",
    )
    parser.add_argument(
        "--frac_preds_pos",
        type=float,
        default=None,
        help="The fraction of positive predictions.",
    )
    parser.add_argument(
        "--metrics_to_analyze",
        type=str,
        nargs="+",
        default=[
            "acc",
            "precision_macro",
            "recall_macro",
            "F1_macro",
            "MCC",
            "Kappa",
        ],
        choices=calculate_metrics_binary(0, 0, 0, 0).keys(),
        help="The metric to analyze.",
    )
    parser.add_argument(
        "--hide_best_worst",
        action="store_false",
        dest="show_best_worst",
        help="Whether to hide the best and worst models in the plot.",
    )
    parser.add_argument(
        "--show_diff",
        action="store_true",
        help=(
            "Whether to show the difference between the best and worst models"
            " in the plot. If this option is enabled, all other plots will be"
            " hidden."
        ),
    )
    parser.add_argument(
        "--show_average",
        action="store_true",
        help=(
            "Whether to show the average model in the plot. If this option is"
            " enabled, all other plots except the random model will be hidden."
        ),
    )

    # Parse the command line arguments.
    args = parser.parse_args()

    # Configure the root logger.
    logger.remove()
    logger.add(
        sys.stderr,
        level=args.logging_level,
        format=(
            "<green>{time:HH:mm:ss}</green>"
            " | <level>{level}</level>"
            " | <cyan>{name}:{line}</cyan>"
            " | <level>{message}</level>"
        ),
        filter={
            "": "INFO",  # Default level for external libraries.
            "__main__": "TRACE",  # All levels for the main file.
            __package__: "TRACE",  # All levels for internal modules.
        },
    )

    # Log the command line arguments for reproducibility.
    logger.debug(f"{args=}")

    # Run the program.
    with logger.catch():
        main(args)
