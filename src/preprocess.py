"""Load the compact cell-level feature table and enforce leakage guards."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    B1_CONTINUATION_TARGETS,
    DATA_PATH,
    FORBIDDEN_MODEL_FEATURES,
    SENSOR_FEATURES,
    TARGET,
    TARGET_TEST_BATCH,
    TRAIN_BATCH,
)


class DataContractError(ValueError):
    """Raised when the modeling data violates the declared experiment contract."""


@dataclass(frozen=True)
class CourseSplit:
    train: pd.DataFrame
    target_test: pd.DataFrame


REQUIRED_COLUMNS = {
    "cell_id",
    "batch",
    "partition",
    "charge_policy",
    TARGET,
    *SENSOR_FEATURES,
}


def load_feature_table(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Load and validate one-row-per-cell features.

    The function fails loudly instead of silently imputing identifiers, targets, or
    missing required features. Imputation of predictor values, if ever needed, belongs
    inside a fitted sklearn pipeline and therefore only sees training folds.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Feature table not found: {path}. See data/README.md for provenance."
        )

    frame = pd.read_csv(path)
    # Apply the official continuation merge deterministically. This is idempotent
    # and prevents the five long-lived Batch-1 cells from being mislabeled.
    frame[TARGET] = frame.apply(
        lambda row: B1_CONTINUATION_TARGETS.get(row["cell_id"], row[TARGET]), axis=1
    )
    missing_columns = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing_columns:
        raise DataContractError(f"Missing required columns: {missing_columns}")
    if frame.empty:
        raise DataContractError("Feature table is empty")
    if frame["cell_id"].duplicated().any():
        duplicates = frame.loc[frame["cell_id"].duplicated(), "cell_id"].tolist()
        raise DataContractError(f"Duplicate cell IDs: {duplicates}")

    expected_batch_sizes = {"b1": 41, "b2": 43, "b3": 40}
    actual_batch_sizes = frame["batch"].value_counts().to_dict()
    if actual_batch_sizes != expected_batch_sizes:
        raise DataContractError(
            f"Unexpected batch sizes: {actual_batch_sizes}; "
            f"expected {expected_batch_sizes}"
        )

    expected_partition_sizes = {"train": 41, "primary": 43, "secondary": 40}
    actual_partition_sizes = frame["partition"].value_counts().to_dict()
    if actual_partition_sizes != expected_partition_sizes:
        raise DataContractError(
            f"Unexpected paper partition sizes: {actual_partition_sizes}; "
            f"expected {expected_partition_sizes}"
        )

    numeric_columns = [TARGET, *SENSOR_FEATURES]
    numeric = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        bad = numeric.columns[numeric.isna().any()].tolist()
        raise DataContractError(f"Missing/non-numeric values in required fields: {bad}")
    if not np.isfinite(numeric.to_numpy()).all():
        raise DataContractError("Non-finite values found in required numeric fields")
    if (frame[TARGET] <= 100).any():
        raise DataContractError("All cells must survive past the 100-cycle feature window")

    for cell_id, expected in B1_CONTINUATION_TARGETS.items():
        actual = frame.loc[frame["cell_id"] == cell_id, TARGET]
        if len(actual) != 1 or float(actual.iloc[0]) != expected:
            raise DataContractError(f"Continuation merge failed for {cell_id}")

    return frame.sort_values("cell_id").reset_index(drop=True)


def validate_feature_set(features: list[str], *, allow_policy: bool = False) -> None:
    """Reject identifiers, targets, batch labels, and treatment-encoding features."""
    forbidden_set = FORBIDDEN_MODEL_FEATURES.copy()
    if allow_policy:
        forbidden_set -= {"cc1", "cc2", "q1_pct", "avg_charge_time"}
    forbidden = sorted(set(features) & forbidden_set)
    if forbidden:
        raise DataContractError(f"Forbidden/leaky model features: {forbidden}")
    if len(features) != len(set(features)):
        raise DataContractError("Feature list contains duplicates")


def make_course_split(frame: pd.DataFrame) -> CourseSplit:
    """Create the grading split: Batch 1 development, Batch 2 locked target-test."""
    train = frame.loc[frame["batch"] == TRAIN_BATCH].copy()
    target_test = frame.loc[frame["batch"] == TARGET_TEST_BATCH].copy()
    if len(train) != 41 or len(target_test) != 43:
        raise DataContractError(
            f"Course split mismatch: train={len(train)}, target_test={len(target_test)}"
        )
    if set(train["cell_id"]) & set(target_test["cell_id"]):
        raise DataContractError("Cell leakage between Batch 1 and Batch 2")
    return CourseSplit(train=train, target_test=target_test)
