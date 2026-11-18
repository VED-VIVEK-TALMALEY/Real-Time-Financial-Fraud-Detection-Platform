import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier

SEED = 42


def choose_threshold(y_true, probabilities, min_precision=0.5):
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    valid = np.flatnonzero(precision[:-1] >= min_precision)
    if valid.size:
        index = valid[np.argmax(recall[:-1][valid])]
        return float(thresholds[index])
    return float(thresholds[np.argmax(recall[:-1])])


def evaluate_model(y_true, probabilities, threshold=None):
    threshold = 0.5 if threshold is None else threshold
    predictions = (probabilities >= threshold).astype(int)
    precision, recall, _ = precision_recall_curve(y_true, probabilities)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    precision_at_threshold = tp / max(tp + fp, 1)
    recall_at_threshold = tp / max(tp + fn, 1)
    return {
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "precision_at_threshold": float(precision_at_threshold),
        "recall_at_threshold": float(recall_at_threshold),
        "threshold": float(threshold),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def train_baseline(train_df, valid_df, label="isFraud", excluded=None):
    excluded = set(excluded or []) | {label}
    feature_columns = [c for c in train_df.select_dtypes(exclude=["object"]).columns if c not in excluded]
    x_train = train_df[feature_columns].replace([np.inf, -np.inf], np.nan)
    x_valid = valid_df[feature_columns].replace([np.inf, -np.inf], np.nan)
    y_train = train_df[label].astype(int)
    y_valid = valid_df[label].astype(int)
    positives = max(int(y_train.sum()), 1)
    negatives = int(len(y_train) - y_train.sum())

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=negatives / positives,
        eval_metric="aucpr",
        random_state=SEED,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_valid)[:, 1]
    threshold = choose_threshold(y_valid, probabilities)
    metrics = evaluate_model(y_valid, probabilities, threshold)
    metrics["feature_columns"] = feature_columns
    return model, threshold, metrics


def main():
    transaction = Path("data/raw/train_transaction.csv")
    if not transaction.exists():
        return
    df = pd.read_csv(transaction)
    df = df.sort_values("TransactionDT").reset_index(drop=True)
    train_end = int(len(df) * 0.70)
    valid_end = int(len(df) * 0.85)
    model, threshold, metrics = train_baseline(
        df.iloc[:train_end], df.iloc[train_end:valid_end], excluded={"TransactionID", "TransactionDT"}
    )
    out = Path("reports")
    out.mkdir(parents=True, exist_ok=True)
    (out / "xgb_metrics.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
