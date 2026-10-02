"""Leakage-safe model definitions for the small-n battery experiment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from .config import FEATURE_SETS, RANDOM_STATE


def log10_target(y):
    return np.log10(np.asarray(y, dtype=float))


def inverse_log10_target(y):
    return np.power(10.0, np.asarray(y, dtype=float))


@dataclass(frozen=True)
class Candidate:
    name: str
    feature_set: str
    estimator: object
    param_grid: dict[str, list]
    eligible_for_selection: bool = True
    note: str = ""

    @property
    def features(self) -> list[str]:
        return FEATURE_SETS[self.feature_set]


def _linear(model):
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", RobustScaler()),
            ("model", model),
        ]
    )
    return TransformedTargetRegressor(
        regressor=pipeline, func=log10_target, inverse_func=inverse_log10_target
    )


def candidates() -> list[Candidate]:
    alphas = np.logspace(-3, 2, 14).tolist()
    return [
        Candidate(
            "median_baseline", "variance", DummyRegressor(strategy="median"), {}
        ),
        Candidate(
            "variance_ridge",
            "variance",
            _linear(Ridge()),
            {"regressor__model__alpha": alphas},
        ),
        Candidate(
            "discharge_ridge",
            "discharge",
            _linear(Ridge()),
            {"regressor__model__alpha": alphas},
        ),
        Candidate(
            "discharge_elasticnet",
            "discharge",
            _linear(ElasticNet(max_iter=100_000, random_state=RANDOM_STATE)),
            {
                "regressor__model__alpha": np.logspace(-3, 0, 10).tolist(),
                "regressor__model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
            },
        ),
        Candidate(
            "discharge_random_forest",
            "discharge",
            RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1),
            {
                "n_estimators": [300],
                "max_depth": [3, None],
                "min_samples_leaf": [1, 2, 4],
                "max_features": [0.5, 1.0],
            },
        ),
        Candidate(
            "sensor_ridge",
            "sensor",
            _linear(Ridge()),
            {"regressor__model__alpha": alphas},
        ),
        Candidate(
            "policy_aware_elasticnet",
            "policy_aware",
            _linear(ElasticNet(max_iter=100_000, random_state=RANDOM_STATE)),
            {
                "regressor__model__alpha": np.logspace(-3, 0, 10).tolist(),
                "regressor__model__l1_ratio": [0.1, 0.5, 0.9, 1.0],
            },
            eligible_for_selection=False,
            note="Protocol leakage sensitivity analysis only",
        ),
    ]
