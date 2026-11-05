"""Evaluation helpers: point metrics, operating-point thresholds, and
per-family / held-out-family detection rates."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)


def classification_metrics(y_true, y_score, threshold: float = 0.5) -> dict:
    """Accuracy, precision, recall, F1, ROC-AUC, FPR and confusion counts."""
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score),
        "fpr": fp / (fp + tn),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def threshold_at_fpr(y_true, y_score, target_fpr: float) -> float:
    """Smallest score threshold whose false-positive rate is <= ``target_fpr``.

    SOC deployments are budgeted in false alarms, so we also report recall at
    a fixed FPR (e.g. 1 %) rather than only at the default 0.5 threshold.
    """
    fpr, _, thr = roc_curve(y_true, y_score)
    ok = np.where(fpr <= target_fpr)[0]
    return float(thr[ok[-1]])


def tpr_at_fpr(y_true, y_score, target_fpr: float) -> float:
    thr = threshold_at_fpr(y_true, y_score, target_fpr)
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    return float((y_score[y_true == 1] >= thr).mean())


def detection_by_family(families, y_score, thresholds: dict[str, float]) -> pd.DataFrame:
    """Share of each family flagged as DGA at each named threshold.

    For the benign family this is the false-positive rate; for DGA families it
    is the detection rate (recall).
    """
    df = pd.DataFrame({"family": np.asarray(families), "score": np.asarray(y_score)})
    out = df.groupby("family").size().rename("n").to_frame()
    for name, t in thresholds.items():
        out[f"flagged@{name}"] = df.assign(f=df.score >= t).groupby("family")["f"].mean()
    return out.reset_index()
