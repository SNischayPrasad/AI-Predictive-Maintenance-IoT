"""
evaluation.py
-------------
Measures how good each model is.

Why not just accuracy?  97% of cycles are healthy, so a model that ALWAYS
says "no failure" scores 97% accuracy while catching zero failures.
In maintenance we care most about:
  * Recall    - of all real failures, how many did we catch?   (missed = breakdown)
  * Precision - of all alarms we raised, how many were real?   (false = wasted visit)
  * F1        - balance of the two
  * ROC-AUC   - how well the model ranks risky vs healthy machines
"""

import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)

from src import config


def evaluate_model(name, pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    return {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "true_positives": tp, "false_positives": fp,
        "false_negatives": fn, "true_negatives": tn,
    }


def evaluate_all(trained_models, X_test, y_test):
    rows = [evaluate_model(n, p, X_test, y_test) for n, p in trained_models.items()]
    metrics = pd.DataFrame(rows).sort_values("f1", ascending=False)
    metrics.to_csv(config.METRICS_FILE, index=False)

    print("\n[EVAL] Test-set results (unseen data):")
    print(metrics[["model", "accuracy", "precision", "recall", "f1", "roc_auc"]]
          .round(3).to_string(index=False))
    print(f"[EVAL] Saved metrics -> {config.METRICS_FILE}")
    return metrics


def print_report(name, pipeline, X_test, y_test):
    print(f"\n[EVAL] Detailed report for {name}:")
    print(classification_report(y_test, pipeline.predict(X_test),
                                target_names=["No Failure", "Failure"], digits=3))
