import numpy as np
from sklearn.base import clone
from src.models import candidates
from src.preprocess import load_feature_table, make_course_split


def test_discharge_model_predicts_positive_values():
    train = make_course_split(load_feature_table()).train
    candidate = next(c for c in candidates() if c.name == "discharge_ridge")
    model = clone(candidate.estimator).fit(train[candidate.features], train.cycle_life)
    pred = model.predict(train[candidate.features].head(3))
    assert np.all(np.isfinite(pred)) and np.all(pred > 0)
