"""
run_sweep.py
~~~~~~~~~~~~
Threshold sweep over the hand-labelled similarity dataset.

Usage
-----
    python -m optimization.threshold_tuning.run_sweep

What it does
------------
1. Loads optimization/data/similarity_samples.json
2. Encodes each (sub_query, content_chunk) pair using the production
   gap_analyzer embedding model — same model the API uses, no drift.
3. Computes per-pair cosine similarity (diagonal, not full matrix).
4. Sweeps thresholds from 0.40 to 0.95 in steps of 0.02.
5. At each threshold: apply_threshold → precision / recall / F1.
6. Writes a timestamped CSV to optimization/threshold_tuning/reports/.
7. Prints a human-readable table with the best-F1 row highlighted.

Output CSV columns
------------------
threshold | precision | recall | f1 | tp | fp | fn | tn
"""
from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np

# ── sys.path setup (must come before app imports) ──────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Reuse the three production helpers directly — no duplicate logic
from app.services.gap_analyzer import _get_model, apply_threshold, l2_normalise  # noqa: E402

# ---------------------------------------------------------------------------
# Paths & sweep parameters — edit here to change the search space
# ---------------------------------------------------------------------------

_DATA_FILE   = _REPO_ROOT / "optimization" / "data" / "similarity_samples.json"
_REPORTS_DIR = _REPO_ROOT / "optimization" / "threshold_tuning" / "reports"

_THRESHOLD_START = 0.40
_THRESHOLD_END   = 0.95
_THRESHOLD_STEP  = 0.02

_CSV_FIELDS = ["threshold", "precision", "recall", "f1", "tp", "fp", "fn", "tn"]

# ---------------------------------------------------------------------------
# Pure metric helpers (no model, no I/O — independently testable)
# ---------------------------------------------------------------------------


def _confusion(labels: List[bool], predictions: List[bool]):
    """Return (tp, fp, fn, tn)."""
    tp = fp = fn = tn = 0
    for true, pred in zip(labels, predictions):
        if true and pred:       tp += 1
        elif not true and pred: fp += 1
        elif true and not pred: fn += 1
        else:                   tn += 1
    return tp, fp, fn, tn


def _score(labels: List[bool], predictions: List[bool]) -> Dict[str, float]:
    """Compute precision, recall, F1 and confusion counts as a flat dict."""
    tp, fp, fn, tn = _confusion(labels, predictions)
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {
        "precision": round(p, 4),
        "recall":    round(r, 4),
        "f1":        round(f, 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------


def _threshold_range() -> List[float]:
    values, t = [], _THRESHOLD_START
    while t <= _THRESHOLD_END + 1e-9:
        values.append(round(t, 4))
        t += _THRESHOLD_STEP
    return values


def run_sweep() -> List[dict]:
    """Load data, encode, sweep thresholds, write CSV, print table."""
    # 1. Load hand-labelled pairs
    samples = json.loads(_DATA_FILE.read_text())
    print(f"Loaded {len(samples)} labelled samples from {_DATA_FILE.name}")

    sub_queries    = [s["sub_query"]     for s in samples]
    content_chunks = [s["content_chunk"] for s in samples]
    labels         = [bool(s["label"])   for s in samples]

    # 2. Encode — reuses the same production model (_get_model is the lazy
    #    singleton from gap_analyzer; identical to what analyse_gaps() uses)
    model = _get_model()
    print("Encoding sub-queries and content chunks…")
    query_vecs = l2_normalise(
        model.encode(sub_queries,     convert_to_numpy=True, show_progress_bar=False)
    )
    chunk_vecs = l2_normalise(
        model.encode(content_chunks,  convert_to_numpy=True, show_progress_bar=False)
    )

    # 3. Per-pair cosine similarity — diagonal of the full matrix
    #    Each sample i pairs sub_query_i with content_chunk_i only
    similarities = np.sum(query_vecs * chunk_vecs, axis=1).astype(np.float32)

    # 4. Sweep — reuses apply_threshold from gap_analyzer (same function the
    #    API calls; ensures the sweep reflects production behaviour exactly)
    results = []
    for threshold in _threshold_range():
        predictions = apply_threshold(similarities, threshold)
        row = _score(labels, predictions)
        row["threshold"] = threshold
        results.append(row)

    # 5. Write CSV
    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path  = _REPORTS_DIR / f"sweep_{timestamp}.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row[k] for k in _CSV_FIELDS})
    print(f"\nReport written → {csv_path}")

    # 6. Print table
    _print_table(results)
    return results


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------


def _print_table(results: List[dict]) -> None:
    best   = max(results, key=lambda r: r["f1"])
    header = (f"{'threshold':>10} {'precision':>10} {'recall':>8} "
              f"{'f1':>8} {'tp':>4} {'fp':>4} {'fn':>4} {'tn':>4}")
    print("\n" + header)
    print("-" * len(header))
    for row in results:
        marker = " ← best F1" if row["threshold"] == best["threshold"] else ""
        print(
            f"{row['threshold']:>10.2f} {row['precision']:>10.4f} "
            f"{row['recall']:>8.4f} {row['f1']:>8.4f} "
            f"{row['tp']:>4} {row['fp']:>4} {row['fn']:>4} {row['tn']:>4}"
            f"{marker}"
        )
    print(f"\nOptimal threshold: {best['threshold']:.2f}  (F1 = {best['f1']:.4f})")


if __name__ == "__main__":
    run_sweep()
