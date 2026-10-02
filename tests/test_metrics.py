import numpy as np
from src.evaluate import regression_metrics, conformal_interval


def test_metrics_are_exact_for_perfect_prediction():
    metrics = regression_metrics([100, 200], [100, 200])
    assert metrics["mape_pct"] == 0
    assert metrics["rmse_cycles"] == 0
    assert metrics["r2"] == 1


def test_conformal_interval_is_ordered():
    lower, upper, radius = conformal_interval([120], [100, 200], [90, 190])
    assert radius == 10
    assert np.all(lower <= upper)
