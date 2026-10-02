"""Generate publication-style figures from saved, reproducible results."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .config import RESULTS_DIR
from .preprocess import load_feature_table, make_course_split


def _style():
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update({"figure.dpi": 140, "axes.titleweight": "bold"})


def generate_figures() -> None:
    _style()
    out = RESULTS_DIR / "figures"
    out.mkdir(parents=True, exist_ok=True)
    frame = load_feature_table()
    split = make_course_split(frame)

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(
        data=frame.query("batch in ['b1','b2']"), x="cycle_life", hue="batch",
        bins=16, element="step", common_norm=False, alpha=.30, ax=ax
    )
    ax.set(title="Cycle-life distribution: development vs locked test", xlabel="Cycle life", ylabel="Cells")
    fig.tight_layout(); fig.savefig(out / "01_target_distribution.png"); plt.close(fig)

    selection = pd.read_csv(RESULTS_DIR / "model_selection.csv")
    perf = pd.read_csv(RESULTS_DIR / "model_performance.csv")
    test_perf = perf.query("split == 'target_test_b2'")[["model", "mape_pct"]]
    comparison = selection.merge(test_perf, on="model").sort_values("cv_mape_mean_pct")
    fig, ax = plt.subplots(figsize=(11, 6))
    x = range(len(comparison))
    ax.bar([i-.2 for i in x], comparison["cv_mape_mean_pct"], width=.4, label="Batch 1 repeated CV")
    ax.bar([i+.2 for i in x], comparison["mape_pct"], width=.4, label="Batch 2 locked test")
    ax.axhline(9.1, color="#E31B23", linestyle="--", label="paper headline 9.1%")
    ax.set_xticks(list(x), comparison["model"], rotation=35, ha="right")
    ax.set(ylabel="MAPE (%)", title="Internal validation does not predict cross-batch performance")
    ax.legend(); fig.tight_layout(); fig.savefig(out / "02_cv_vs_test.png"); plt.close(fig)

    pred = pd.read_csv(RESULTS_DIR / "batch2_predictions.csv")
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.errorbar(pred["actual_cycle_life"], pred["predicted_cycle_life"],
                yerr=[pred["predicted_cycle_life"]-pred["prediction_lower_90"],
                      pred["prediction_upper_90"]-pred["predicted_cycle_life"]],
                fmt="o", alpha=.65, ecolor="#9AA0A6", color="#0078D4")
    low = min(pred["actual_cycle_life"].min(), pred["prediction_lower_90"].min())
    high = max(pred["actual_cycle_life"].max(), pred["prediction_upper_90"].max())
    ax.plot([low, high], [low, high], "--", color="#E31B23")
    ax.set(xlabel="Actual cycle life", ylabel="Predicted cycle life", title="Locked Batch 2 predictions (90% conformal intervals)")
    fig.tight_layout(); fig.savefig(out / "03_prediction_parity.png"); plt.close(fig)

    errors = pred.head(10).sort_values("absolute_percentage_error")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(errors["cell_id"], errors["absolute_percentage_error"], color="#F59E0B")
    ax.set(xlabel="Absolute percentage error (%)", ylabel="Cell", title="Largest Batch 2 errors")
    fig.tight_layout(); fig.savefig(out / "04_top_errors.png"); plt.close(fig)


if __name__ == "__main__":
    generate_figures()
