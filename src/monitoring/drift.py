import json
from pathlib import Path
import numpy as np
import pandas as pd


def drift_summary(reference, current, threshold=0.5):
    shared = [column for column in reference.select_dtypes(include="number").columns if column in current]
    drifted = []
    for column in shared:
        ref = reference[column].dropna().astype(float)
        cur = current[column].dropna().astype(float)
        if len(ref) and len(cur):
            ref_std = max(ref.std(), 1e-9)
            shift = abs(cur.mean() - ref.mean()) / ref_std
            if shift > threshold:
                drifted.append(column)
    share = len(drifted) / max(len(shared), 1)
    return {"drifted_features": drifted, "drifted_share": share, "trigger_retrain": share > 0.30}


def should_retrain(drift_summary_result, pr_auc=None, baseline_pr_auc=None):
    if drift_summary_result.get("trigger_retrain"):
        return True
    if pr_auc is not None and baseline_pr_auc is not None and baseline_pr_auc > 0:
        return pr_auc < baseline_pr_auc * 0.90
    return False


def save_drift_report(reference, current, output_dir="reports/drift"):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary = drift_summary(reference, current)
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    (output / "drift_report.html").write_text(
        "<html><body><h1>Fraud Drift Report</h1><pre>"
        + json.dumps(summary, indent=2)
        + "</pre></body></html>"
    )
    return summary
