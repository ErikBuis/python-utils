# TODO Actually make this code a usable util function

from __future__ import annotations

import inspect
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from sklearn.base import BaseEstimator
from sklearn.linear_model import RANSACRegressor
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

np.random.seed(42)

# === Step 1: Define model functions ===


def linear(x: np.ndarray, a: float, b: float) -> np.ndarray:
    return a * x + b


def exponential(x: np.ndarray, a: float, b: float) -> np.ndarray:
    return a * np.exp(b * x)


# === Step 2: Define Mammen random variable ===


def mammen_rvs(size: int) -> np.ndarray:
    sqrt5 = np.sqrt(5)
    p = (sqrt5 + 1) / (2 * sqrt5)
    w1 = (1 - sqrt5) / 2
    w2 = (1 + sqrt5) / 2
    return np.where(np.random.rand(size) < p, w1, w2)


# === Step 3: Define bootstrap prediction interval using wild bootstrap ===
# Ensure model_func is compatible with RANSAC
class ModelWrapper(BaseEstimator):
    def __init__(self, model_func: Callable):
        self.model_func = model_func
        self.params = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "ModelWrapper":
        # Perform curve fitting to determine parameters.
        self.params, _ = curve_fit(self.model_func, x.ravel(), y, maxfev=10000)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        # Predict using the fitted parameters.
        if self.params is None:
            raise ValueError("Model is not fitted yet.")
        return self.model_func(x.ravel(), *self.params)

    def score(self, x: np.ndarray, y: np.ndarray) -> float:
        # Optional: Implement a scoring method (e.g., R^2).
        y_pred = self.predict(x)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot


def wild_bootstrap_prediction_intervals_fixed(
    x_data: np.ndarray,
    y_data: np.ndarray,
    model_func: Callable,
    x_query: np.ndarray,
    n_bootstraps: int = 500,
    percentiles: tuple[float, float] = (68, 95),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    # ghet the number of parameters from the model function
    n_params = len(inspect.signature(model_func).parameters)

    ransac = RANSACRegressor(
        estimator=ModelWrapper(model_func), min_samples=n_params
    )
    ransac.fit(x_data.reshape(-1, 1), y_data)
    popt = ransac.estimator_.params

    y_fit = model_func(x_data, *popt)
    residuals = y_data - y_fit

    y_preds = []

    # Fit KNN once to avoid repeated work
    knn = NearestNeighbors(n_neighbors=100)
    knn.fit(x_data.reshape(-1, 1))  # Make sure x_data is 2D
    distances, neighbor_idcs = knn.kneighbors(x_query.reshape(-1, 1))

    # Calculate weights based on distances
    weights = np.exp(-0.5 * (distances / 1.0) ** 2)
    weights = np.where(weights >= 1e-10, weights, 1e-10)  # Avoid zero weights
    weights /= np.sum(weights, axis=1, keepdims=True)  # Normalize weights

    for _ in tqdm(range(n_bootstraps)):
        # Wild bootstrapping: perturb residuals to get new y
        wild_noise = residuals * mammen_rvs(len(x_data))
        y_boot = y_fit + wild_noise

        # Fit model to bootstrapped data
        popt_boot, _ = curve_fit(model_func, x_data, y_boot, p0=popt)
        y_query_boot = model_func(x_query, *popt_boot)

        # Local residual resampling for each x_query point
        local_noises = np.array([
            np.random.choice(residuals[neighbors], p=weights[i])
            for i, neighbors in enumerate(neighbor_idcs)
        ])

        # Add local noise to simulate new observations
        y_query_boot += local_noises

        y_preds.append(y_query_boot)

    y_preds = np.stack(y_preds)
    y_low_68, y_high_68 = np.percentile(y_preds, [16, 84], axis=0)
    y_low_68_smooth = gaussian_filter1d(y_low_68, sigma=2)
    y_high_68_smooth = gaussian_filter1d(y_high_68, sigma=2)
    y_low_95, y_high_95 = np.percentile(y_preds, [2.5, 97.5], axis=0)
    y_low_95_smooth = gaussian_filter1d(y_low_95, sigma=2)
    y_high_95_smooth = gaussian_filter1d(y_high_95, sigma=2)
    y_mean = model_func(x_query, *popt)

    return (
        y_mean,
        y_low_68_smooth,
        y_high_68_smooth,
        y_low_95_smooth,
        y_high_95_smooth,
    )


# === Step 4: Generate artificial datasets ===

n = 100
x = np.linspace(0, 10, n)

# 1. Asymmetric residuals
true_y1 = linear(x, 2.0, 1.0)
noise1 = np.where(
    np.random.randn(n) > 0,
    np.abs(np.random.normal(0, 0.5, n)),
    -np.abs(np.random.normal(0, 3.0, n)),
)
y1 = true_y1 + noise1

# 2. Exponential model
true_y2 = exponential(x, 1.0, 0.3)
noise2 = np.random.normal(0, 5.0, size=n)
y2 = true_y2 + noise2

# 3. Heteroscedastic noise
true_y3 = linear(x, -1.5, 20)
noise3 = np.random.normal(0, 0.2 + 0.4 * x, size=n)
y3 = true_y3 + noise3

# 4. Assymetric and heteroscedastic noise, exponential model
true_y4 = exponential(x, 1.0, 0.3)
noise4 = np.where(
    np.random.randn(n) > 0,
    np.abs(np.random.normal(0, 3.0 + 0.5 * x, n)),
    -np.abs(np.random.normal(0, 0.5 + 0.1 * x, n)),
)
y4 = true_y4 + noise4

# === Step 5: Plot with prediction intervals ===

x_query = np.linspace(x.min(), x.max(), 200)

datasets = [
    (x, y1, linear, "Asymmetric Noise"),
    (x, y2, exponential, "Exponential Model"),
    (x, y3, linear, "Heteroscedastic Noise"),
    (x, y4, exponential, "Everything Together"),
]

fig, axs = plt.subplots(2, 2, figsize=(10, 8))

for ax, (x_data, y_data, model_func, title) in zip(axs.flatten(), datasets):
    y_mean, l68, h68, l95, h95 = wild_bootstrap_prediction_intervals_fixed(
        x_data, y_data, model_func, x_query
    )

    ax.scatter(x_data, y_data, alpha=0.5, label="Data")
    ax.plot(x_query, y_mean, color="black", label="Fitted Curve")
    ax.fill_between(x_query, l95, h95, color="red", alpha=0.2, label="95% PI")
    ax.fill_between(
        x_query, l68, h68, color="orange", alpha=0.4, label="68% PI"
    )
    ax.set_title(title)
    ax.legend()

plt.tight_layout()
plt.show()
