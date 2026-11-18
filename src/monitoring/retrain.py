from pathlib import Path
import pandas as pd


def retrain_if_needed(reference, current, latest_validation_pr_auc, baseline_pr_auc):
    if latest_validation_pr_auc >= baseline_pr_auc:
        return {"status": "skip", "reason": "new model did not beat current model"}
    return {"status": "triggered", "reason": "performance or drift threshold exceeded"}
