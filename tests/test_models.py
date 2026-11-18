import numpy as np
from src.models.blend import blend_scores, choose_blend_weight


def test_blend_rejects_shape_mismatch():
    try:
        blend_scores([0.1, 0.2], [0.3])
    except ValueError:
        return
    raise AssertionError("shape mismatch was not rejected")


def test_blend_weight_is_valid_and_grid_search_runs():
    y = np.array([0, 1, 0, 1])
    xgb = np.array([0.1, 0.8, 0.2, 0.9])
    gnn = np.array([0.2, 0.7, 0.3, 0.6])
    weight, score = choose_blend_weight(y, xgb, gnn)
    assert 0 <= weight <= 1
    assert 0 <= score <= 1
