"""
metrics.py
~~~~~~~~~~
Pure metric computation functions for threshold tuning.

All functions are stateless and independently testable.
No model loading, no file I/O — inputs are plain Python lists/ints.
"""
from __future__ import annotations

from typing import Dict, List, Tuple


def confusion_counts(
    labels: List[bool],
    predictions: List[bool],
) -> Tuple[int, int, int, int]:
    """
    Compute (TP, FP, FN, TN) from parallel label and prediction lists.

    Parameters
    ----------
    labels      : ground-truth boolean list  (True = covered)
    predictions : model prediction list      (True = predicted covered)

    Returns
    -------
    (tp, fp, fn, tn)
    """
    if len(labels) != len(predictions):
        raise ValueError(
            f"labels and predictions must be the same length: "
            f"{len(labels)} vs {len(predictions)}"
        )
    tp = fp = fn = tn = 0
    for true, pred in zip(labels, predictions):
        if true and pred:
            tp += 1
        elif not true and pred:
            fp += 1
        elif true and not pred:
            fn += 1
        else:
            tn += 1
    return tp, fp, fn, tn


def precision(tp: int, fp: int) -> float:
    """Precision = TP / (TP + FP).  Returns 0.0 when denominator is zero."""
    denom = tp + fp
    return tp / denom if denom > 0 else 0.0


def recall(tp: int, fn: int) -> float:
    """Recall = TP / (TP + FN).  Returns 0.0 when denominator is zero."""
    denom = tp + fn
    return tp / denom if denom > 0 else 0.0


def f1_score(p: float, r: float) -> float:
    """F1 = 2 * P * R / (P + R).  Returns 0.0 when denominator is zero."""
    denom = p + r
    return 2 * p * r / denom if denom > 0 else 0.0


def compute_all_metrics(
    labels: List[bool],
    predictions: List[bool],
) -> Dict[str, float]:
    """
    Convenience wrapper — returns a dict with all four metrics.

    Returns
    -------
    {
      "precision": float,
      "recall":    float,
      "f1":        float,
      "tp": int, "fp": int, "fn": int, "tn": int
    }
    """
    tp, fp, fn, tn = confusion_counts(labels, predictions)
    p = precision(tp, fp)
    r = recall(tp, fn)
    f = f1_score(p, r)
    return {
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1": round(f, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }
