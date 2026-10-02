"""Evaluation, uncertainty, and error-analysis utilities."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true: Iterable[float], y_pred: Iterable[float]) -> dict[str, float]:
    y_true_array = np.asarray(y_true, dtype=float)
    y_pred_array = np.asarray(y_pred, dtype=float)
    percentage_errors = np.abs(y_true_array - y_pred_array) / y_true_array * 100
    return {
        "rmse_cycles": float(np.sqrt(mean_squared_error(y_true_array, y_pred_array))),
        "mae_cycles": float(mean_absolute_error(y_true_array, y_pred_array)),
        "mape_pct": float(percentage_errors.mean()),
        "median_ape_pct": float(np.median(percentage_errors)),
        "r2": float(r2_score(y_true_array, y_pred_array)),
    }


def bootstrap_metric_intervals(
    y_true: Iterable[float],
    y_pred: Iterable[float],
    *,
    n_bootstrap: int = 5000,
    random_state: int = 42,
) -> dict[str, dict[str, float]]:
    """Cell-level percentile bootstrap intervals for the locked test set."""
    y_true_array = np.asarray(y_true, dtype=float)
    y_pred_array = np.asarray(y_pred, dtype=float)
    rng = np.random.default_rng(random_state)
    collected = {"rmse_cycles": [], "mae_cycles": [], "mape_pct": []}
    for _ in range(n_bootstrap):
        index = rng.integers(0, len(y_true_array), size=len(y_true_array))
        metrics = regression_metrics(y_true_array[index], y_pred_array[index])
        for key in collected:
            collected[key].append(metrics[key])
    return {
        key: {
            "lower_95": float(np.quantile(values, 0.025)),
            "upper_95": float(np.quantile(values, 0.975)),
        }
        for key, values in collected.items()
    }


def conformal_interval(
    predictions: Iterable[float],
    calibration_true: Iterable[float],
    calibration_pred: Iterable[float],
    *,
    coverage: float = 0.90,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Build a simple split-conformal interval from out-of-fold absolute residuals."""
    prediction_array = np.asarray(predictions, dtype=float)
    residuals = np.abs(
        np.asarray(calibration_true, dtype=float) - np.asarray(calibration_pred, dtype=float)
    )
    n = len(residuals)
    quantile_level = min(1.0, np.ceil((n + 1) * coverage) / n)
    radius = float(np.quantile(residuals, quantile_level, method="higher"))
    lower = np.maximum(0.0, prediction_array - radius)
    upper = prediction_array + radius
    return lower, upper, radius


def prediction_frame(
    metadata: pd.DataFrame,
    y_true: Iterable[float],
    y_pred: Iterable[float],
    lower: Iterable[float] | None = None,
    upper: Iterable[float] | None = None,
) -> pd.DataFrame:
    result = metadata[["cell_id", "batch", "charge_policy"]].reset_index(drop=True).copy()
    result["actual_cycle_life"] = np.asarray(y_true, dtype=float)
    result["predicted_cycle_life"] = np.asarray(y_pred, dtype=float)
    result["residual_cycles"] = result["actual_cycle_life"] - result["predicted_cycle_life"]
    result["absolute_percentage_error"] = (
        result["residual_cycles"].abs() / result["actual_cycle_life"] * 100
    )
    if lower is not None and upper is not None:
        result["prediction_lower_90"] = np.asarray(lower, dtype=float)
        result["prediction_upper_90"] = np.asarray(upper, dtype=float)
        result["covered_90"] = (
            (result["actual_cycle_life"] >= result["prediction_lower_90"])
            & (result["actual_cycle_life"] <= result["prediction_upper_90"])
        )
    return result.sort_values("absolute_percentage_error", ascending=False)

