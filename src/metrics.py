import numpy as np
import pandas as pd
from typing import Dict

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    classification_report,
    confusion_matrix,
)


def _topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """
    Return indices of the top-k scores.
    Ties are broken deterministically by smaller index.
    """
    n = int(scores.shape[0])
    if k < 1 or k > n:
        raise ValueError(f"k must be in [1, {n}], got {k}")

    idx = np.arange(n)
    order = np.lexsort((idx, -scores))  # sort by (-score, index)
    return order[:k]


def precision_recall_at_k(y_true: pd.Series, scores: np.ndarray, k: int) -> tuple[float, float]:
    idx = _topk_indices(scores, k)
    precision_k = float(y_true.iloc[idx].mean())
    recall_k = float(y_true.iloc[idx].sum() / max(float(y_true.sum()), 1.0))
    return precision_k, recall_k


def threshold_f1(y_true: pd.Series, scores: np.ndarray) -> float:
    prec, rec, thr = precision_recall_curve(y_true, scores)
    f1 = 2 * (prec * rec) / (prec + rec + 1e-12)

    # thr length = len(prec)-1, align safely
    best_i = int(np.argmax(f1[1:]) + 1)
    return float(thr[best_i - 1])


def threshold_topk(scores: np.ndarray, k: int) -> float:
    """
    Return the k-th largest score (cutoff at rank k).

    Note: Using (scores >= cutoff) can select MORE than k items when ties exist at the cutoff.
    For an exact top-k decision rule, use report_at_topk.
    """
    s = np.sort(scores)[::-1]
    n = len(s)
    if k < 1 or k > n:
        raise ValueError(f"k must be in [1, {n}], got {k}")
    return float(s[k - 1])


def report_at_threshold(y_true: pd.Series, scores: np.ndarray, thr: float) -> Dict:
    pred = (scores >= thr).astype(int)
    return {
        "threshold": float(thr),
        "selected": int(pred.sum()),
        "confusion_matrix": confusion_matrix(y_true, pred).tolist(),
        "classification_report": classification_report(y_true, pred, output_dict=True),
    }


def report_at_topk(y_true: pd.Series, scores: np.ndarray, k: int) -> Dict:
    """
    Report metrics when selecting exactly k samples with highest scores.
    Includes diagnostics showing how many scores are >= cutoff (tie effect).
    """
    idx = _topk_indices(scores, k)
    pred = np.zeros(scores.shape[0], dtype=int)
    pred[idx] = 1

    cutoff = float(scores[idx].min())
    num_ge_cutoff = int((scores >= cutoff).sum())  # may be > k if ties at cutoff

    return {
        "k": int(k),
        "cutoff_score": cutoff,
        "selected": int(pred.sum()),          # exactly k
        "num_ge_cutoff": num_ge_cutoff,       # tie diagnostic
        "confusion_matrix": confusion_matrix(y_true, pred).tolist(),
        "classification_report": classification_report(y_true, pred, output_dict=True),
    }


def scalar_metrics(y_true: pd.Series, scores: np.ndarray) -> Dict:
    return {
        "pr_auc": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
    }
