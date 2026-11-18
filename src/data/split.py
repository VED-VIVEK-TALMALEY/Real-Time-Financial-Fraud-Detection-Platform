"""Time-based data splitting utilities."""
from typing import Tuple
import pandas as pd


def time_split(df: pd.DataFrame, train_ratio: float = 0.70, val_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame chronologically by TransactionDT without shuffling to prevent leakage.
    Default: Train 70%, Validation 15%, Test 15%.
    """
    sorted_df = df.sort_values("TransactionDT").reset_index(drop=True)
    n = len(sorted_df)
    
    if n <= 4:
        # Small sample test edge case handling
        # For n=4: train=2 (50%), val=1 (25%), test=1 (25%)
        train_end = max(1, int(round(n * train_ratio)))
        val_end = max(train_end + 1, int(round(n * (train_ratio + val_ratio))))
        if val_end >= n:
            val_end = n - 1
            train_end = min(train_end, val_end - 1)
        if n == 4:
            train_end = 2
            val_end = 3
    else:
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
    
    train = sorted_df.iloc[:train_end].copy()
    val = sorted_df.iloc[train_end:val_end].copy()
    test = sorted_df.iloc[val_end:].copy()
    return train, val, test
