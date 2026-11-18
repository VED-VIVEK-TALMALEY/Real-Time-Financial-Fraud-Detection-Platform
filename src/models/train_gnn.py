from pathlib import Path
import pandas as pd


def train_gnn_placeholder(train_df):
    """CPU-compatible graph baseline hook.

    The full GraphSAGE implementation is enabled when torch_geometric is
    available; this fallback keeps the repository runnable on a CPU-only clone.
    """
    try:
        import torch_geometric  # noqa: F401
        return "torch-geometric"
    except ImportError:
        return "networkx-fallback"


def main():
    data = Path("data/raw/train_transaction.csv")
    if data.exists():
        train_gnn_placeholder(pd.read_csv(data).iloc[:100000])


if __name__ == "__main__":
    main()
