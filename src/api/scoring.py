from time import perf_counter

BLEND_WEIGHT = 0.7


def blend_for_transaction(xgb_score, gnn_score, weight=BLEND_WEIGHT):
    return float(weight * xgb_score + (1 - weight) * gnn_score)


def score_payload(payload, xgb_model=None, gnn_scores=None, threshold=0.5):
    started = perf_counter()
    gnn_scores = gnn_scores or {}
    if xgb_model is not None:
        features = payload.get("features")
        if features:
            xgb_score = float(xgb_model.predict_proba([features])[0][1])
        else:
            xgb_score = float(payload.get("xgb_score", 0.0))
    else:
        xgb_score = float(payload.get("xgb_score", 0.0))
    account = str(payload.get("card1", payload.get("account_id", "")))
    gnn_score = float(gnn_scores.get(account, payload.get("gnn_score", 0.0)))
    fraud_score = blend_for_transaction(xgb_score, gnn_score)
    return {
        "transaction_id": str(payload.get("transaction_id", payload.get("TransactionID", ""))),
        "fraud_score": fraud_score,
        "xgb_score": xgb_score,
        "gnn_score": gnn_score,
        "decision": "alert" if fraud_score >= threshold else "review",
        "latency_ms": (perf_counter() - started) * 1000,
        "ring_id": payload.get("ring_id"),
    }
