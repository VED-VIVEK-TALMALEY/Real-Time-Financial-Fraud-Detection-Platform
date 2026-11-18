import pandas as pd
from src.features.build_features import build_features


def test_features_do_not_use_future_transactions():
    frame = pd.DataFrame({
        "TransactionDT": [1, 2, 3],
        "TransactionID": [10, 11, 12],
        "TransactionAmt": [10.0, 20.0, 30.0],
        "card1": ["a", "a", "a"],
        "DeviceID": ["d1", "d1", "d2"],
        "P_emaildomain": ["x", "x", "y"],
    })
    features = build_features(frame)
    assert features["velocity_count_1h"].tolist() == [0.0, 1.0, 2.0]
    assert features["spend_1h"].tolist() == [0.0, 10.0, 30.0]
    assert features["new_device_flag"].tolist() == [1, 0, 1]
