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
   gap_analyzer embedding model (same model the API uses — no drift).
3. Computes cosine similarity for every pair.
4. Sweeps thresholds from 0.40 to 0.95 in steps of 0.02.
5. At each threshold:  apply_threshold → predictions → precision/recall/F1.
6. Writes a timestamped CSV to optimization/threshold_tuning/reports/.
7. Prints a human-readable table to stdout with the optimal F1 row highlighted.

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

import numpy as np

# Make sure the repo root is on sys.path when run directly
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.services.gap_analyzer import (  # noqa: E402
    apply_threshold,
    l2_normalise,
    _get_model,
)
from optimization.threshold_tuning.metrics import compute_all_metrics  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_DATA_FILE = _REPO_ROOT / "optimization" / "data" / "similarity_samples.json"
_REPORTS_DIR = _REPO_ROOT / "optimization" / "threshold_tuning" / "reports"

# ---------------------------------------------------------------------------
# Sweep parameters — edit here to change the search space
# ---------------------------------------------------------------------------

_THRESHOLD_START = 0.40
_THRESHOLD_END = 0.95
_THRESHOLD_STEP = 0.02


def _threshold_range() -> list[float]:
    """Generate threshold values from start to end (inclusive) in step increments."""
    values = []
    t = _THRESHOLD_START
    while t <= _THRESHOLD_END + 1e-9:
        values.append(round(t, 4))
        t += _THRESHOLD_STEP
    return values


# ---------------------------------------------------------------------------
# Core sweep logic
# ---------------------------------------------------------------------------


def run_sweep() -> list[dict]:
    """
    Load data, encode pairs, sweep thresholds, return list of result dicts.
    Also writes CSV and prints table.
    """
    # 1. Load labelled data
    samples = json.loads(_DATA_FILE.read_text())
    print(f"Loaded {len(samples)} labelled samples from {_DATA_FILE.name}")

    sub_queries = [s["sub_query"] for s in samples]
    content_chunks = [s["content_chunk"] for s in samples]
    labels = [bool(s["label"]) for s in samples]

    # 2. Encode — uses the same production model
    model = _get_model()
    print("Encoding sub-queries and content chunks…")
    query_vecs = l2_normalise(
        model.encode(sub_queries, convert_to_numpy=True, show_progress_bar=False)
    )
    chunk_vecs = l2_normalise(
        model.encode(content_chunks, convert_to_numpy=True, show_progress_bar=False)
    )

    # 3. Compute per-pair cosine similarities (diagonal of the full matrix)
    #    Each sample i: similarity(sub_query_i, content_chunk_i)
    similarities = np.sum(query_vecs * chunk_vecs, axis=1).astype(np.float32)

    # 4. Sweep
    results = []
    for threshold in _threshold_range():
        predictions = apply_threshold(similarities, threshold)
        metrics = compute_all_metrics(labels, predictions)
        metrics["threshold"] = threshold
        results.append(metrics)

    # 5. Write CSV
    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = _REPORTS_DIR / f"sweep_{timestamp}.csv"
    _write_csv(results, csv_path)
    print(f"\nReport written → {csv_path}")

    # 6. Print table
    _print_table(results)

    return results


def _write_csv(results: list[dict], path: Path) -> None:
    fieldnames = ["threshold", "precision", "recall", "f1", "tp", "fp", "fn", "tn"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row[k] for k in fieldnames})


def _print_table(results: list[dict]) -> None:
    best = max(results, key=lambda r: r["f1"])
    header = f"{'threshold':>10} {'precision':>10} {'recall':>8} {'f1':>8} {'tp':>4} {'fp':>4} {'fn':>4} {'tn':>4}"
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
