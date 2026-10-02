"""End-to-end Batch-1 development and locked Batch-2 evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import (
    GridSearchCV,
    GroupKFold,
    KFold,
    RepeatedKFold,
    cross_val_predict,
    cross_validate,
)

from .config import PAPER_BENCHMARKS, RANDOM_STATE, RESULTS_DIR, TARGET
from .evaluate import (
    bootstrap_metric_intervals,
    conformal_interval,
    prediction_frame,
    regression_metrics,
)
from .features import correlation_table, feature_shift_table, target_summary
from .models import Candidate, candidates
from .preprocess import load_feature_table, make_course_split, validate_feature_set


def _mape_scorer(estimator, x, y):
    pred = estimator.predict(x)
    return -float(np.mean(np.abs(np.asarray(y) - pred) / np.asarray(y)) * 100)


def _fit_candidate(candidate: Candidate, train: pd.DataFrame, repeats: int):
    validate_feature_set(
        candidate.features, allow_policy=candidate.feature_set.startswith("policy")
    )
    x, y = train[candidate.features], train[TARGET]
    inner = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    if candidate.param_grid:
        search = GridSearchCV(
            candidate.estimator,
            candidate.param_grid,
            scoring=_mape_scorer,
            cv=inner,
            n_jobs=1,
            refit=True,
        ).fit(x, y)
        fitted = search.best_estimator_
        params = search.best_params_
    else:
        fitted = clone(candidate.estimator).fit(x, y)
        params = {}

    repeated = RepeatedKFold(
        n_splits=5, n_repeats=repeats, random_state=RANDOM_STATE + 7
    )
    scores = -cross_validate(
        clone(fitted), x, y, cv=repeated, scoring=_mape_scorer, n_jobs=1
    )["test_score"]

    groups = train["charge_policy"].astype(str)
    grouped = -cross_validate(
        clone(fitted),
        x,
        y,
        cv=GroupKFold(n_splits=5),
        groups=groups,
        scoring=_mape_scorer,
        n_jobs=1,
    )["test_score"]
    return fitted, params, scores, grouped


def run(*, repeats: int = 10, bootstrap: int = 5000) -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "figures").mkdir(exist_ok=True)
    frame = load_feature_table()
    split = make_course_split(frame)
    train, test = split.train, split.target_test

    selection_rows, fitted_models = [], {}
    for candidate in candidates():
        fitted, params, scores, group_scores = _fit_candidate(candidate, train, repeats)
        fitted_models[candidate.name] = (candidate, fitted)
        selection_rows.append(
            {
                "model": candidate.name,
                "feature_set": candidate.feature_set,
                "n_features": len(candidate.features),
                "eligible_for_selection": candidate.eligible_for_selection,
                "cv_mape_mean_pct": scores.mean(),
                "cv_mape_std_pct": scores.std(ddof=1),
                "cv_mape_p90_pct": np.quantile(scores, 0.9),
                "policy_group_cv_mape_pct": group_scores.mean(),
                "best_params": json.dumps(params, sort_keys=True),
                "note": candidate.note,
            }
        )

    selection = pd.DataFrame(selection_rows)
    eligible = selection.query("eligible_for_selection").copy()
    # Pre-registered stability rule: first find the best mean; among models within
    # one percentage point choose the lowest policy-group CV error. Batch 2 is not
    # consulted at any point in this selection.
    threshold = eligible["cv_mape_mean_pct"].min() + 1.0
    finalists = eligible.loc[eligible["cv_mape_mean_pct"] <= threshold]
    selected_name = finalists.sort_values(
        ["policy_group_cv_mape_pct", "cv_mape_mean_pct", "n_features"]
    ).iloc[0]["model"]
    selection["selected_before_target_test"] = selection["model"].eq(selected_name)
    selection.to_csv(RESULTS_DIR / "model_selection.csv", index=False)

    performance_rows = []
    test_predictions = {}
    for name, (candidate, model) in fitted_models.items():
        for split_name, data in [("train_apparent", train), ("target_test_b2", test)]:
            pred = model.predict(data[candidate.features])
            metrics = regression_metrics(data[TARGET], pred)
            benchmark_key = (
                "variance" if candidate.feature_set == "variance" else "discharge"
            )
            paper_primary = PAPER_BENCHMARKS[benchmark_key]["primary_mape_pct"]
            performance_rows.append(
                {
                    "model": name,
                    "feature_set": candidate.feature_set,
                    "split": split_name,
                    "n": len(data),
                    **metrics,
                    "gap_vs_headline_9_1_pp": metrics["mape_pct"] - 9.1,
                    "gap_vs_comparable_paper_primary_pp": metrics["mape_pct"]
                    - paper_primary,
                    "selected_before_target_test": name == selected_name,
                }
            )
            if split_name == "target_test_b2":
                test_predictions[name] = pred

    performance = pd.DataFrame(performance_rows)
    performance.to_csv(RESULTS_DIR / "model_performance.csv", index=False)

    selected_candidate, selected_model = fitted_models[selected_name]
    x_train = train[selected_candidate.features]
    oof = cross_val_predict(
        clone(selected_model),
        x_train,
        train[TARGET],
        cv=KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE + 99),
        n_jobs=1,
    )
    selected_pred = test_predictions[selected_name]
    lower, upper, radius = conformal_interval(
        selected_pred, train[TARGET], oof, coverage=0.90
    )
    predictions = prediction_frame(test, test[TARGET], selected_pred, lower, upper)
    predictions.to_csv(RESULTS_DIR / "batch2_predictions.csv", index=False)
    predictions.head(10).to_csv(RESULTS_DIR / "error_analysis_top10.csv", index=False)

    intervals = bootstrap_metric_intervals(
        test[TARGET], selected_pred, n_bootstrap=bootstrap, random_state=RANDOM_STATE
    )
    intervals["conformal_90"] = {
        "radius_cycles": radius,
        "empirical_coverage_pct": float(predictions["covered_90"].mean() * 100),
    }
    (RESULTS_DIR / "uncertainty.json").write_text(
        json.dumps(intervals, indent=2), encoding="utf-8"
    )

    all_shift_features = selected_candidate.features
    feature_shift_table(train, test, all_shift_features).to_csv(
        RESULTS_DIR / "feature_shift.csv", index=False
    )
    correlation_table(train, all_shift_features).to_csv(
        RESULTS_DIR / "batch1_correlations.csv", index=False
    )
    target_summary(frame).to_csv(RESULTS_DIR / "target_summary.csv", index=False)
    joblib.dump(selected_model, RESULTS_DIR / "final_model.joblib")

    manifest = {
        "selected_model": selected_name,
        "selection_rule": "within 1 pp of best repeated-CV mean, then lowest charge-policy-group CV MAPE",
        "target_test_opened_after_selection": True,
        "train_cells": len(train),
        "target_test_cells": len(test),
        "batch3_status": "not scored: optional set requires raw re-extraction audit",
    }
    (RESULTS_DIR / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--bootstrap", type=int, default=5000)
    args = parser.parse_args()
    print(json.dumps(run(repeats=args.repeats, bootstrap=args.bootstrap), indent=2))


if __name__ == "__main__":
    main()
