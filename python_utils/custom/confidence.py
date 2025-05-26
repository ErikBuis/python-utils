from __future__ import annotations

import argparse
import inspect
import sys
from collections.abc import Sequence
from typing import Any, Callable, cast

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import RANSACRegressor
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


def _mammen_rvs(size: int) -> npt.NDArray[np.float64]:
    """Generate Mammen random variables.

    Mammen random variables are used in the wild bootstrap method for robust
    bootstrapping. They are defined as a mixture of two point masses, which
    allows it to deal with heteroscedasticity and distributions that are not
    normally distributed. This is particularly useful in regression contexts
    where the residuals may not follow a normal distribution.

    See https://en.wikipedia.org/wiki/Bootstrapping_(statistics)#Wild_bootstrap
    for more details.

    Args:
        size: The number of Mammen random variables to generate.

    Returns:
        An array of Mammen random variables of the specified size.
    """
    sqrt5 = np.sqrt(5)
    p = (sqrt5 + 1) / (2 * sqrt5)
    w1 = (1 - sqrt5) / 2
    w2 = (1 + sqrt5) / 2
    return np.where(np.random.rand(size) < p, w1, w2)


class _ModelWrapper(BaseEstimator):
    """Wrapper class to make a model function compatible with RANSACRegressor.

    This class takes a callable model function and wraps it in a scikit-learn
    BaseEstimator structure. This allows parameter estimation with the RANSAC
    algorithm.

    Attributes:
        model_func: The callable function representing the model to be fitted.
            This function should take x-values as its first argument, followed
            by model parameters.
        curve_fit_kwargs: Keyword arguments to pass to
            `scipy.optimize.curve_fit`.
        params: The fitted parameters of the model after `fit` is called.
            None if the model has not been fitted.
    """

    def __init__(
        self,
        model_func: Callable[..., npt.NDArray[np.floating]],
        curve_fit_kwargs: dict[str, Any] = {},
    ) -> None:
        """Initialize the ModelWrapper.

        Args:
            model_func: The callable function representing the model to be
                fitted. This function should take x-values as its first
                argument, followed by model parameters.
            curve_fit_kwargs: Keyword arguments to pass to
                `scipy.optimize.curve_fit`.
        """
        self.model_func = model_func
        self.curve_fit_kwargs = curve_fit_kwargs
        self.params = None

    def fit(
        self, x: npt.NDArray[np.floating], y: npt.NDArray[np.floating]
    ) -> _ModelWrapper:
        """Fit the model to the provided data.

        Uses `scipy.optimize.curve_fit` to find the optimal parameters for the
        `model_func` given the input data `x` and `y`.

        Args:
            x: Observed data on the x-axis (independent variable).
            y: Observed data on the y-axis (dependent variable).

        Returns:
            The fitted ModelWrapper instance.
        """
        self.params, _ = curve_fit(
            self.model_func, x.ravel(), y, **self.curve_fit_kwargs
        )
        return self

    def predict(self, x: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Predict target values for the given data using the fitted model.

        Args:
            x: The independent variable data for which to make predictions.

        Returns:
            The predicted target values.
        """
        if self.params is None:
            raise NotFittedError(
                "Model is not fitted yet. Call 'fit' before 'predict'."
            )
        return self.model_func(x.ravel(), *self.params)

    def score(
        self, x: npt.NDArray[np.floating], y: npt.NDArray[np.floating]
    ) -> np.floating:
        """Calculate the R-squared score for the model.

        Args:
            x: Observed data on the x-axis (independent variable).
            y: Observed data on the y-axis (dependent variable).

        Returns:
            The R-squared score.
        """
        y_preds = self.predict(x)
        ss_res = np.sum((y - y_preds) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot


def bootstrap_confidence_intervals(
    x_data: npt.NDArray[np.floating],
    y_data: npt.NDArray[np.floating],
    model_func: Callable[..., npt.NDArray[np.floating]],
    x_query: npt.NDArray[np.floating],
    conf_levels: Sequence[float],
    n_bootstraps: int = 500,
    curve_fit_kwargs: dict[str, Any] = {},
) -> tuple[
    npt.NDArray[np.float64],
    list[tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]],
]:
    """Calculate confidence intervals using wild bootstrapping.

    This function fits a model to the provided data using RANSAC and calculates
    confidence intervals for the predictions at specified `x_query` points.

    A confidence interval is a range of values that is likely to contain the
    true value of a parameter with a specified level of confidence. More
    specifically, for each given x_query point, the function returns the
    following:
    - The most likely y-value (the model prediction).
    - Two bounds between which the true y-value is expected to lie with a
      certain confidence level (e.g. 68% and 95%).

    The function is able to deal with the following issues well:
    - Heteroscedasticity: The residuals (differences between observed and
      predicted values) do not have constant variance.
    - Non-normal distribution of residuals: The residuals do not follow a normal
      distribution but follow an unknown distribution.
    - Asymmetric residuals: The negative and positive residuals may not be
      symmetrically distributed around the "true" value.
    - Outliers: The data may contain outliers that can affect the model fitting.

    Args:
        x_data: Observed data on the x-axis (independent variable).
            Shape: [N]
        y_data: Observed data on the y-axis (dependent variable).
            Shape: [N]
        model_func: A function that is assumed to be able to represent the
            data's underlying trend. This function should take an array of
            x-values as its first argument, followed by the model parameters.
            For example, if you assume a linear model, the function body should
            be `a * x + b`, where `a` and `b` are the parameters to be fitted.
            Amount of parameters after first input: P
        x_query: The x-values for which to calculate confidence intervals.
            Shape: [M]
        conf_levels: A sequence of confidence levels (as percentages) for which
            to calculate the confidence intervals. For example, if you want 68%
            and 95% confidence intervals, you would pass `(68, 95)`.
            Length: C
        n_bootstraps: The number of bootstrap samples to generate. A higher
            number of bootstraps will result in more smooth and accurate
            confidence intervals, but will also increase the computation time.
        **curve_fit_kwargs: Keyword arguments to pass to
            `scipy.optimize.curve_fit`.

    Returns:
        Tuple containing:
        - The optimal parameter values found by the model fitted with RANSAC.
          Shape: [P]
        - List of tuples of lower and upper bounds, one tuple for each
          confidence interval. Each tuple contains:
            - Smoothed lower bound of the interval.
              Shape: [M]
            - Smoothed upper bound of the interval.
              Shape: [M]
          Length: C
    """
    # Here's how this function works on a high level:
    # 1. Model fitting: The function first fits a model to the provided `x_data`
    #    and `y_data` using RANSAC.
    # 2. Residual calculation: The residuals (differences between the observed
    #    `y_data` and model predictions) are calculated.
    # 3. Bootstrapping: Artificial samples (bootstraps) are generated by
    #    perturbing each residual using the local neighborhood around it. This
    #    is called "wild bootstrapping", see the following for more details:
    #    https://en.wikipedia.org/wiki/Bootstrapping_(statistics)#Wild_bootstrap
    #    For each bootstrap sample, a new model is fit around the artificial
    #    data (without using RANSAC for efficiency reasons). The `y` values
    #    are again predicted, now using the bootstrapped model.
    # 4. Confidence interval calculation: Lastly, it calculates the specified
    #    intervals (e.g. 68% and 95%) of the predicted `y` values across all
    #    bootstrap samples to form the lower and upper bounds of the confidence
    #    intervals. These bounds are then smoothed using a Gaussian filter to
    #    reduce the disparity between neighboring query points.

    # Reshape the data to ensure it is 2D, as required by sklearn.
    N = len(x_data)  # Number of data points
    x_data_reshaped = x_data.reshape(-1, 1)  # [N, 1]
    x_query_reshaped = x_query.reshape(-1, 1)  # [M, 1]

    # Step 1: Fit the model using RANSAC.
    P = len(inspect.signature(model_func).parameters)
    ransac = RANSACRegressor(
        estimator=_ModelWrapper(model_func, curve_fit_kwargs=curve_fit_kwargs),
        min_samples=P,
    )
    ransac.fit(x_data_reshaped, y_data)
    popt = ransac.estimator_.params  # [P]
    curve_fit_kwargs["p0"] = popt

    # Step 2: Calculate residuals.
    y_fit = model_func(x_data, *popt)  # [N]
    residuals = y_data - y_fit  # [N]

    # Fit KNN once to avoid repeated work.
    K = N // 3
    knn = NearestNeighbors(n_neighbors=K)
    knn.fit(x_data_reshaped)
    dists, nbr_idcs = knn.kneighbors(x_query_reshaped)  # [M, K], [M, K]
    dists /= np.ptp(x_data)  # normalize for better weight calculation

    # Calculate weights for each neighbor based on distance.
    weights = np.exp(-20 * dists**2)  # [M, K]
    weights = np.where(weights >= 1e-10, weights, 1e-10)  # avoid zero weights
    weights /= np.sum(weights, axis=1, keepdims=True)  # normalize weights

    # Perform local residual resampling for each x_query point.
    # This is done in advance to avoid repeated work in the bootstrapping step.
    residuals_rearranged = np.take_along_axis(
        np.expand_dims(residuals, 0), nbr_idcs, axis=1
    )  # [M, K]
    local_residuals = np.stack([
        np.random.choice(
            residuals_rearranged_i, size=n_bootstraps, p=weights_i
        )  # [n_bootstraps]
        for residuals_rearranged_i, weights_i in zip(
            residuals_rearranged, weights
        )
    ])  # [M, n_bootstraps]

    # Step 3: Wild bootstrapping.
    y_pred_boots = []
    for b in tqdm(range(n_bootstraps)):
        # Generate wild bootstrap noise using Mammen random variables.
        wild_noise_boot = residuals * _mammen_rvs(N)  # [N]
        y_boot = y_fit + wild_noise_boot  # [N]

        # Fit model to bootstrapped data.
        popt_boot, _ = curve_fit(
            model_func, x_data, y_boot, **curve_fit_kwargs
        )  # [P], _

        # Add local noise to simulate new observations.
        y_pred_boot = (
            model_func(x_query, *popt_boot) + local_residuals[:, b]
        )  # [M]
        y_pred_boots.append(y_pred_boot)

    y_pred_boots = np.stack(y_pred_boots)  # [n_bootstraps, M]

    # Step 4: Confidence interval calculation.
    y_intervals = []
    for conf_level in conf_levels:
        lower_bound = (100 - conf_level) / 2
        upper_bound = 100 - lower_bound

        # Calculate percentiles for the confidence intervals.
        y_low, y_high = np.percentile(
            y_pred_boots, [lower_bound, upper_bound], axis=0
        )  # [M], [M]

        # Smooth the bounds using a Gaussian filter.
        y_low = gaussian_filter1d(y_low, sigma=2)
        y_high = gaussian_filter1d(y_high, sigma=2)
        y_intervals.append((y_low, y_high))

    return popt, y_intervals


def remap_unseen_x_to_y_intervals(
    x_unseen: npt.NDArray[np.floating],
    model_func: Callable[..., npt.NDArray[np.floating]],
    popt: npt.NDArray[np.floating],
    x_query: npt.NDArray[np.floating],
    y_intervals: list[
        tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]
    ],
) -> tuple[
    npt.NDArray[np.floating],
    list[tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]],
]:
    """Remap unseen x-values to predicted y-values and confidence intervals.

    This function takes the x-values and the fitted model parameters, and remaps
    the y-intervals to the corresponding x-values. This is useful when you
    want to get a predicted y-value together with its confidence for an unseen
    x value.

    Args:
        x_unseen: The unseen x-values to remap to y-values and confidence
            scores.
            Shape: [U]
        model_func: The model function used to fit the data. This function
            should take an array of x-values as its first argument, followed by
            the model parameters.
        popt: The optimal parameters calculated by the
            `bootstrap_confidence_intervals` function.
            Shape: [P]
        x_query: The x-values for which the confidence intervals were
            calculated. This is the same as the `x_query` parameter passed to
            the `bootstrap_confidence_intervals` function.
            Shape: [M]
        y_intervals: The confidence intervals calculated by the
            `bootstrap_confidence_intervals` function. This is a list of tuples
            of lower and upper bounds, one tuple for each confidence interval.
            Each tuple contains:
                - Smoothed lower bound of the interval.
                Shape: [M]
                - Smoothed upper bound of the interval.
                Shape: [M]
            Length: C

    Returns:
        Tuple containing:
        - The predicted y-values for the given x_unseen.
          Shape: [U]
        - List of tuples of lower and upper bounds, one tuple for each
          confidence interval. Each tuple contains:
            - Smoothed lower bound of the interval.
              Shape: [U]
            - Smoothed upper bound of the interval.
              Shape: [U]
          Length: C
    """
    # Predict the y-values for the unseen x-values.
    y_pred_unseen = model_func(x_unseen, *popt)  # [U]

    # Map each x_unseen to the corresponding confidence intervals.
    y_intervals_unseen = []
    for y_low, y_high in y_intervals:
        # Interpolate the y-intervals to the unseen x-values.
        y_low_unseen = np.interp(x_unseen, x_query, y_low)  # [U]
        y_high_unseen = np.interp(x_unseen, x_query, y_high)  # [U]
        y_intervals_unseen.append((y_low_unseen, y_high_unseen))

    return y_pred_unseen, y_intervals_unseen


def main(args: argparse.Namespace) -> None:
    """The entry point of the program.

    Args:
        args: The command line arguments.
    """

    def linear(
        x: npt.NDArray[np.floating], a: float, b: float
    ) -> npt.NDArray[np.floating]:
        return a * x + b

    def exponential(
        x: npt.NDArray[np.floating], a: float, b: float
    ) -> npt.NDArray[np.floating]:
        return a * np.exp(b * x)

    # Set the random seed for reproducibility.
    np.random.seed(69)

    # Generate artificial datasets.
    N = 1000
    M = 50
    U = 100
    x = np.linspace(0, 10, N)

    # Dataset 1: Asymmetric residuals.
    true_y1 = linear(x, 1.0, 10.0)
    noise1 = np.where(
        np.random.randn(N) > 0,
        np.abs(np.random.normal(0, 0.5, N)),
        -np.abs(np.random.normal(0, 3.0, N)),
    )
    y1 = true_y1 + noise1

    # Dataset 2: Exponential model.
    true_y2 = exponential(x, 1.0, 0.3)
    noise2 = np.random.normal(0, 5.0, size=N)
    y2 = true_y2 + noise2

    # Dataset 3: Heteroscedastic noise.
    true_y3 = linear(x, -1.0, 20.0)
    noise3 = np.random.normal(0, 0.2 + 0.4 * x, size=N)
    y3 = true_y3 + noise3

    # Dataset 4: Assymetric and heteroscedastic noise, exponential model.
    true_y4 = exponential(x, 1.0, 0.3)
    noise4 = np.where(
        np.random.randn(N) > 0,
        np.abs(np.random.normal(0, 3.0 + 1.0 * x, N)),
        -np.abs(np.random.normal(0, 0.5 + 0.4 * x, N)),
    )
    y4 = true_y4 + noise4

    # Gather the datasets for plotting.
    datasets = [
        (
            x,
            y1,
            linear,
            "Asymmetric Noise",
            r"{:.2f} \cdot x + {:.2f}",
            ([-10, -10], [10, 30]),
        ),
        (
            x,
            y2,
            exponential,
            "Exponential Model",
            r"{:.2f} \cdot \exp({:.2f} * x)",
            ([-10, -1], [10, 1]),
        ),
        (
            x,
            y3,
            linear,
            "Heteroscedastic Noise",
            r"{:.2f} \cdot x + {:.2f}",
            ([-10, -10], [10, 30]),
        ),
        (
            x,
            y4,
            exponential,
            "Everything Together",
            r"{:.2f} \cdot \exp({:.2f} * x)",
            ([-10, -1], [10, 1]),
        ),
    ]

    # Plot the confidence intervals.
    _, axs = plt.subplots(2, 2, figsize=(10, 8))
    x_query = np.linspace(x.min(), x.max(), M)
    x_unseen = np.linspace(x.min(), x.max(), U)

    for ax, (x_data, y_data, model_func, title, formula, bounds) in zip(
        axs.flatten(), datasets
    ):
        ax = cast(plt.Axes, ax)
        confs = (95, 68)
        colors = ["red", "orange"]
        alphas = [0.2, 0.4]

        # Fit the model and calculate confidence intervals.
        popt, y_intervals = bootstrap_confidence_intervals(
            x_data,
            y_data,
            model_func,
            x_query,
            confs,
            curve_fit_kwargs={"maxfev": 10000, "bounds": bounds},
        )
        y_pred_unseen, y_intervals_unseen = remap_unseen_x_to_y_intervals(
            x_unseen, model_func, popt, x_query, y_intervals
        )

        # Plot the observed data.
        ax.scatter(x_data, y_data, alpha=0.5, label="Data")

        # Plot the confidence intervals.
        for (y_low_unseen, y_high_unseen), conf, color, alpha in zip(
            y_intervals_unseen, confs, colors, alphas
        ):
            ax.fill_between(
                x_unseen,
                y_low_unseen,
                y_high_unseen,
                color=color,
                alpha=alpha,
                label=f"{conf}%",
            )

        # Plot the predictions for the unseen x-values.
        ax.plot(
            x_unseen,
            y_pred_unseen,
            color="black",
            label=f"Fitted Curve: ${formula.format(*popt)}$",
        )

        # Set figure-wide properties.
        ax.set_title(title)
        ax.legend()

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

    # Parse the command line arguments.
    args = parser.parse_args()

    # Configure the root logger.
    logger.remove()
    logger.add(
        sys.stderr,
        level=args.logging_level,
        format=(
            "<green>{time:HH:mm:ss}</green>"
            + " | <level>{level:<8}</level>"
            + " | <cyan>{name}:{line}</cyan>"
            + " | <level>{message}</level>"
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
