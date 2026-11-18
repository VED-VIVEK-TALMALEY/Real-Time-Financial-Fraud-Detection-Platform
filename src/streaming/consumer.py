import json
import sqlite3
from pathlib import Path


def ensure_database(db_path="data/fraud.db"):
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE IF NOT EXISTS alerts (transaction_id TEXT PRIMARY KEY, fraud_score REAL, ring_id TEXT, created_at TEXT)"
    )
    connection.execute(
        "CREATE TABLE IF NOT EXISTS feedback (transaction_id TEXT, label TEXT, analyst_id TEXT, note TEXT, created_at TEXT)"
    )
    connection.commit()
    return connection


def process_scored_event(event, db_path="data/fraud.db"):
    score = float(event.get("fraud_score", 0))
    threshold = float(event.get("threshold", 0.5))
    if score >= threshold:
        connection = ensure_database(db_path)
        connection.execute(
            "INSERT OR REPLACE INTO alerts VALUES (?, ?, ?, datetime('now'))",
            (str(event.get("transaction_id", "")), score, event.get("ring_id")),
        )
        connection.commit()
        connection.close()
        return True
    return False


def main():
    ensure_database()
    print("fraud consumer initialized; database ready")


if __name__ == "__main__":
    main()
