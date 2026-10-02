"""Project-wide constants and feature contracts."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "severson_features.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
RANDOM_STATE = 42
TARGET = "cycle_life"

# Course contract: model development on Batch 1, exactly one locked evaluation on Batch 2.
TRAIN_BATCH = "b1"
TARGET_TEST_BATCH = "b2"

# Five Batch-1 cells continued in Batch 2.  The compact source table originally
# contained the pre-continuation value; these are the official merged lifetimes.
B1_CONTINUATION_TARGETS = {
    "b1c0": 1852,
    "b1c1": 2160,
    "b1c2": 2237,
    "b1c3": 1434,
    "b1c4": 1709,
}

# Features derived from the common discharge diagnostic. These do not directly encode
# which fast-charge policy was assigned to a cell.
VARIANCE_FEATURES = ["log_var_dq"]

DISCHARGE_FEATURES = [
    "log_var_dq",
    "log_min_dq",
    "dq_mean",
    "dq_skew",
    "dq_kurt",
    "cap_c2",
    "cap_c100",
    "cap_diff_100_2",
    "cap_slope_2_100",
    "cap_intercept_2_100",
]

# Additional early sensor streams. Charge-policy fields and nominal charge time are
# intentionally excluded because they can encode the assigned treatment.
SENSOR_FEATURES = DISCHARGE_FEATURES + [
    "temp_mean_2_100",
    "temp_max_2_100",
    "temp_slope_2_100",
    "ir_c2",
    "ir_c100",
    "ir_diff_100_2",
    "ir_slope_2_100",
]

POLICY_FEATURES = ["cc1", "cc2", "q1_pct", "avg_charge_time"]

FEATURE_SETS = {
    "variance": VARIANCE_FEATURES,
    "discharge": DISCHARGE_FEATURES,
    "sensor": SENSOR_FEATURES,
    # Treatment-encoding variables are used only as an explicitly labelled
    # sensitivity analysis, never for selecting the policy-blind final model.
    "policy_aware": SENSOR_FEATURES + POLICY_FEATURES,
    "policy_only": POLICY_FEATURES,
}

FORBIDDEN_MODEL_FEATURES = {
    TARGET,
    "batch",
    "partition",
    "cell_id",
    "charge_policy",
    "cc1",
    "cc2",
    "q1_pct",
    "avg_charge_time",
}

# Published Table 1, all-cell values. Parenthetical values excluded one anomalous
# primary-test cell, so both are reported and never silently interchanged.
PAPER_BENCHMARKS = {
    "variance": {
        "primary_mape_pct": 14.7,
        "secondary_mape_pct": 11.4,
        "primary_rmse_cycles": 138.0,
        "secondary_rmse_cycles": 196.0,
    },
    "discharge": {
        "primary_mape_pct": 13.0,
        "secondary_mape_pct": 8.6,
        "primary_rmse_cycles": 91.0,
        "secondary_rmse_cycles": 173.0,
    },
    "full": {
        "primary_mape_pct": 14.1,
        "primary_mape_excluding_anomaly_pct": 7.5,
        "secondary_mape_pct": 10.7,
        "primary_rmse_cycles": 118.0,
        "secondary_rmse_cycles": 214.0,
    },
    "abstract_headline_test_error_pct": 9.1,
}
