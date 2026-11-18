import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score


def blend_scores(xgb_score, gnn_score, weight=0.7):
    xgb_score = np.asarray(xgb_score, dtype=float)
    gnn_score = np.asarray(gnn_score, dtype=float)
    if xgb_score.shape != gnn_score.shape:
        raise ValueError("xgb_score and gnn_score must have the same shape")
    if not 0 <= weight <= 1:
        raise ValueError("weight must be between 0 and 1")
    return weight * xgb_score + (1 - weight) * gnn_score


def choose_blend_weight(y_true, xgb_score, gnn_score):
    best_weight, best_score = 0.5, -1.0
    for weight in np.linspace(0, 1, 21):
        score = blend_scores(xgb_score, gnn_score, float(weight))
        candidate = average_precision_score(y_true, score)
        if candidate > best_score:
            best_weight, best_score = float(weight), float(candidate)
    return best_weight, best_score


def stack_scores(y_true, xgb_score, gnn_score):
    model = LogisticRegression(class_weight="balanced", max_iter=500)
    model.fit(np.column_stack([xgb_score, gnn_score]), y_true)
    return model, model.predict_proba(np.column_stack([xgb_score, gnn_score]))[:, 1]
