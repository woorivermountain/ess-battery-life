import pytest

from src.config import DISCHARGE_FEATURES
from src.preprocess import DataContractError, load_feature_table, make_course_split, validate_feature_set


def test_official_split_and_continuation_merge():
    frame = load_feature_table()
    split = make_course_split(frame)
    assert (len(split.train), len(split.target_test)) == (41, 43)
    assert frame.set_index("cell_id").loc["b1c0", "cycle_life"] == 1852


def test_leakage_guard():
    validate_feature_set(DISCHARGE_FEATURES)
    with pytest.raises(DataContractError):
        validate_feature_set(["cycle_life"])
    with pytest.raises(DataContractError):
        validate_feature_set(["cc1"])
    validate_feature_set(["cc1"], allow_policy=True)
