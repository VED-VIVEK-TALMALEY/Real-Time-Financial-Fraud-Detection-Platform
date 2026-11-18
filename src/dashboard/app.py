import sqlite3
import pandas as pd
import streamlit as st


def load_alerts(db_path="data/fraud.db"):
    with sqlite3.connect(db_path) as connection:
        return pd.read_sql_query("SELECT * FROM alerts", connection)


def submit_feedback(transaction_id, label, analyst_id, note, db_path="data/fraud.db"):
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS feedback (transaction_id TEXT, label TEXT, analyst_id TEXT, note TEXT, created_at TEXT)"
        )
        connection.execute(
            "INSERT INTO feedback VALUES (?, ?, ?, ?, datetime('now'))",
            (transaction_id, label, analyst_id, note),
        )
        connection.commit()


def main():
    st.title("Fraud Analyst Dashboard")
    st.metric("Alert count", "0")
    st.metric("Drift status", "Unknown")
    try:
        alerts = load_alerts()
        st.dataframe(alerts)
        if not alerts.empty:
            transaction_id = st.selectbox("Case", alerts["transaction_id"].tolist())
            status = st.selectbox("Case status", ["New", "In review", "Confirmed fraud", "False positive", "Escalated"])
            note = st.text_area("Analyst notes")
            analyst = st.text_input("Analyst ID")
            if st.button("Submit feedback"):
                submit_feedback(transaction_id, status, analyst, note)
                st.success("Feedback saved")
    except Exception as exc:
        st.info(f"No live alerts yet: {exc}")


if __name__ == "__main__":
    main()
