import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve, confusion_matrix
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.data.load import load_raw_data
from src.data.split import time_split
from src.features.build_features import build_features

SEED = 42

def choose_threshold(y_true, probabilities, min_precision=0.5):
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    valid = np.flatnonzero(precision[:-1] >= min_precision)
    if valid.size:
        index = valid[np.argmax(recall[:-1][valid])]
        return float(thresholds[index])
    if len(thresholds) > 0:
        return float(thresholds[np.argmax(recall[:-1])])
    return 0.5

def evaluate_model(y_true, probabilities, threshold=None):
    threshold = 0.5 if threshold is None else threshold
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    precision_at_threshold = tp / max(tp + fp, 1)
    recall_at_threshold = tp / max(tp + fn, 1)
    
    return {
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)) if len(np.unique(y_true)) > 1 else 0.5,
        "precision_at_threshold": float(precision_at_threshold),
        "recall_at_threshold": float(recall_at_threshold),
        "threshold": float(threshold),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }

def train_and_evaluate(df=None):
    if df is None:
        df = load_raw_data()
        
    feat_df = build_features(df)
    full_df = df[["TransactionID", "TransactionDT", "isFraud"]].copy()
    for col in feat_df.columns:
        if col not in ["entity_id", "event_timestamp"]:
            full_df[col] = feat_df[col].values
            
    train_df, valid_df, test_df = time_split(full_df)
    
    feature_cols = [c for c in full_df.columns if c not in ["TransactionID", "TransactionDT", "isFraud", "entity_id", "event_timestamp"]]
    
    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df["isFraud"].astype(int)
    X_val = valid_df[feature_cols].fillna(0)
    y_val = valid_df["isFraud"].astype(int)
    X_test = test_df[feature_cols].fillna(0)
    y_test = test_df["isFraud"].astype(int)
    
    positives = max(int(y_train.sum()), 1)
    negatives = int(len(y_train) - y_train.sum())
    scale_pos_weight = negatives / positives
    
    # 1. Baseline: Logistic Regression
    lr = LogisticRegression(class_weight="balanced", random_state=SEED, max_iter=500)
    lr.fit(X_train, y_train)
    lr_probs = lr.predict_proba(X_val)[:, 1]
    lr_metrics = evaluate_model(y_val, lr_probs)
    
    # 2. Main Model: XGBoost with scale_pos_weight
    xgb_model = XGBClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=SEED,
        n_jobs=-1,
    )
    xgb_model.fit(X_train, y_train)
    val_probs = xgb_model.predict_proba(X_val)[:, 1]
    threshold = choose_threshold(y_val, val_probs, min_precision=0.4)
    val_metrics = evaluate_model(y_val, val_probs, threshold)
    
    # Evaluate on holdout test set
    test_probs = xgb_model.predict_proba(X_test)[:, 1]
    test_metrics = evaluate_model(y_test, test_probs, threshold)
    test_metrics["feature_columns"] = feature_cols
    test_metrics["scale_pos_weight"] = float(scale_pos_weight)
    test_metrics["val_pr_auc"] = val_metrics["pr_auc"]
    test_metrics["baseline_lr_val_pr_auc"] = lr_metrics["pr_auc"]
    
    # Generate PR curve plot
    precision, recall, _ = precision_recall_curve(y_test, test_probs)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, marker=".", label=f"XGBoost Test PR-AUC={test_metrics['pr_auc']:.3f}", color="crimson")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (Holdout Test Set)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    Path("reports").mkdir(exist_ok=True)
    plt.savefig("reports/pr_curve.png", dpi=150)
    plt.close()
    
    # Save metrics
    Path("reports/xgb_metrics.json").write_text(json.dumps(test_metrics, indent=2))
    print(f"XGBoost training complete. Test PR-AUC: {test_metrics['pr_auc']:.4f}, ROC-AUC: {test_metrics['roc_auc']:.4f}")
    return xgb_model, threshold, test_metrics, (X_test, y_test, test_probs)

def main():
    train_and_evaluate()

if __name__ == "__main__":
    main()
