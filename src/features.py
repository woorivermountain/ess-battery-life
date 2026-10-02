"""Feature diagnostics used to connect EDA findings to the modeling strategy."""

from __future__ import annotations

import numpy as np
import pandas as pd


def target_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Return one-row-per-batch target statistics."""
    summary = (
        frame.groupby("batch")["cycle_life"]
        .agg(["count", "mean", "std", "median", "min", "max"])
        .reset_index()
    )
    summary["iqr"] = frame.groupby("batch")["cycle_life"].quantile(0.75).values - frame.groupby(
        "batch"
    )["cycle_life"].quantile(0.25).values
    return summary


def feature_shift_table(
    train: pd.DataFrame, target_test: pd.DataFrame, features: list[str]
) -> pd.DataFrame:
    """Quantify cross-batch covariate shift with training-standardized differences."""
    rows: list[dict[str, float | str]] = []
    for feature in features:
        train_values = train[feature].astype(float)
        test_values = target_test[feature].astype(float)
        train_std = float(train_values.std(ddof=1))
        smd = (
            float((test_values.mean() - train_values.mean()) / train_std)
            if train_std > 0
            else np.nan
        )
        train_low, train_high = np.quantile(train_values, [0.01, 0.99])
        outside = ((test_values < train_low) | (test_values > train_high)).mean()
        rows.append(
            {
                "feature": feature,
                "train_mean": float(train_values.mean()),
                "target_test_mean": float(test_values.mean()),
                "standardized_mean_diff": smd,
                "target_test_outside_train_1_99_pct": float(outside * 100),
            }
        )
    return pd.DataFrame(rows).sort_values(
        "standardized_mean_diff", key=lambda series: series.abs(), ascending=False
    )


def correlation_table(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Compute Pearson and Spearman correlations using Batch 1 only."""
    rows = []
    for feature in features:
        rows.append(
            {
                "feature": feature,
                "pearson": frame[feature].corr(frame["cycle_life"], method="pearson"),
                "spearman": frame[feature].corr(frame["cycle_life"], method="spearman"),
            }
        )
    return pd.DataFrame(rows).sort_values(
        "spearman", key=lambda series: series.abs(), ascending=False
    )

