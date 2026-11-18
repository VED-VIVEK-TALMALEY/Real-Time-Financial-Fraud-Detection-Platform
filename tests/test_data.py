import pandas as pd
from src.data.split import time_split


def test_time_split_is_chronological_and_complete():
    frame = pd.DataFrame({"TransactionDT": [400, 100, 300, 200], "isFraud": [0, 1, 0, 1]})
    train, validation, test = time_split(frame)
    assert train["TransactionDT"].tolist() == [100, 200]
    assert validation["TransactionDT"].tolist() == [300]
    assert test["TransactionDT"].tolist() == [400]
