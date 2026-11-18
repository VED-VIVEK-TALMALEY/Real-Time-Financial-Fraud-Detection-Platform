import argparse
import json
import time
from pathlib import Path
import pandas as pd


def replay_transactions(input_path, speed=100, sender=None):
    frame = pd.read_csv(input_path).sort_values("TransactionDT")
    sent = 0
    for _, row in frame.iterrows():
        payload = json.loads(row.to_json())
        if sender:
            sender(payload)
        sent += 1
        if speed > 0:
            time.sleep(1 / speed)
    return sent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/train_transaction.csv")
    parser.add_argument("--speed", type=float, default=100)
    args = parser.parse_args()
    sent = replay_transactions(args.input, args.speed)
    print(f"sent={sent}")


if __name__ == "__main__":
    main()
