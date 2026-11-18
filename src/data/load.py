"""Data loading and generation utilities."""
import os
import pandas as pd
import numpy as np


def generate_sample_dataset(n_samples: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Generate realistic synthetic IEEE-CIS style transactions for offline testing and fallback."""
    np.random.seed(seed)
    
    transaction_ids = np.arange(3000000, 3000000 + n_samples)
    transaction_dts = np.sort(np.random.randint(1, 86400 * 30, size=n_samples))
    transaction_amts = np.round(np.random.exponential(scale=100.0, size=n_samples) + 5.0, 2)
    product_cds = np.random.choice(["W", "C", "R", "H", "S"], size=n_samples, p=[0.75, 0.1, 0.05, 0.05, 0.05])
    cards = np.random.choice([f"card_{i}" for i in range(1, 20)], size=n_samples)
    addrs = np.random.choice([f"addr_{i}" for i in range(1, 10)], size=n_samples)
    device_ids = np.random.choice([f"dev_{i}" for i in range(1, 30)], size=n_samples)
    device_infos = np.random.choice(["iOS Device", "Windows", "MacOS", "SM-G960N", "Trident/7.0"], size=n_samples)
    emails = np.random.choice(["gmail.com", "yahoo.com", "hotmail.com", "anonymous.com"], size=n_samples)
    
    # Approx 3.5% fraud rate
    is_fraud = (np.random.rand(n_samples) < 0.035).astype(int)
    
    df = pd.DataFrame({
        "TransactionID": transaction_ids,
        "isFraud": is_fraud,
        "TransactionDT": transaction_dts,
        "TransactionAmt": transaction_amts,
        "ProductCD": product_cds,
        "card1": cards,
        "addr1": addrs,
        "DeviceID": device_ids,
        "DeviceInfo": device_infos,
        "P_emaildomain": emails,
    })
    return df


def load_raw_data(data_dir: str = "data/raw") -> pd.DataFrame:
    """Load IEEE-CIS raw files if present, otherwise return synthetic dataset."""
    train_tx_path = os.path.join(data_dir, "train_transaction.csv")
    train_id_path = os.path.join(data_dir, "train_identity.csv")
    
    if os.path.exists(train_tx_path):
        tx = pd.read_csv(train_tx_path)
        if os.path.exists(train_id_path):
            identity = pd.read_csv(train_id_path)
            df = tx.merge(identity, on="TransactionID", how="left")
        else:
            df = tx
        return df
    
    # Fallback to local sample dataset
    sample_path = os.path.join(data_dir, "sample_transactions.csv")
    if os.path.exists(sample_path):
        return pd.read_csv(sample_path)
        
    os.makedirs(data_dir, exist_ok=True)
    df = generate_sample_dataset()
    df.to_csv(sample_path, index=False)
    return df
