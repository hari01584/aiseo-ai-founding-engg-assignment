"""
run_prompt_eval.py
~~~~~~~~~~~~~~~~~~
Fan-out output logger — generates a CSV of the actual sub-queries produced
by ``gpt-4o-mini`` for each target query in the evaluation set, plus a
per-query score row based on the README's explicit requirements.

Usage
-----
    python -m optimization.prompt_tuning.run_prompt_eval

What it does
------------
For each target query in ``optimization/data/prompt_eval_queries.json``:
  1. Calls ``generate_sub_queries`` (production engine — same retry logic,
     same validation, same prompt as the live API).
  2. All 5 queries are executed in parallel (batch size = 5).
  3. Writes two CSV files to ``optimization/prompt_tuning/logs/``:

     run_<id>_subqueries.csv  — one row per sub-query (what did the model say?)
     run_<id>_scores.csv      — one row per query    (did it meet the spec?)

Score dimensions (from README requirements)
-------------------------------------------
  json_valid          — engine parsed + validated the response without error
  count_in_range      — 10 ≤ total ≤ 15
  all_6_types_present — all 6 type identifiers appear at least once
  min_2_per_type      — every type has ≥ 2 sub-queries
  no_missing_types    — alias: same as all_6_types_present (explicit README check)
  composite_score     — (passed dimensions) / 4  → 0.0–1.0

Environment variables required
-------------------------------
OPENAI_API_KEY  (loaded automatically from .env if present)
"""
from __future__ import annotations

import csv
import json
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── sys.path & env setup (must come before app imports) ────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(_REPO_ROOT / ".env")

from app.models.schemas import LLMUnavailableError, SubQuery  # noqa: E402
from app.services.fanout_engine import (  # noqa: E402
    FanOutConfig,
    _VALID_TYPES,
    generate_sub_queries,
)

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

_DATA_FILE  = _REPO_ROOT / "optimization" / "data" / "prompt_eval_queries.json"
_LOGS_DIR   = _REPO_ROOT / "optimization" / "prompt_tuning" / "logs"
_MODEL      = "gpt-4o-mini"
_BATCH_SIZE = 5   # parallel workers — one per query in the default eval set

_SQ_FIELDS = [
    "run_id",
    "query_id",
    "target_query",
    "sub_query_index",
    "sub_query_type",
    "sub_query_text",
]

_SCORE_FIELDS = [
    "run_id",
    "query_id",
    "target_query",
    "total_sub_queries",   # raw count
    "count_in_range",      # 10–15 → True/False
    "all_6_types_present", # all 6 type identifiers found → True/False
    "min_2_per_type",      # every type has ≥ 2 → True/False
    "types_breakdown",     # JSON string: {"comparative": 3, ...}
    "missing_types",       # comma-separated missing types, empty if none
    "composite_score",     # 0.0–1.0  (4 binary dimensions)
    "error",               # non-empty if LLMUnavailableError was raised
]

# ---------------------------------------------------------------------------
# Post-analysis scorer
# (operates on validated SubQuery objects — no raw text parsing needed here
#  because the engine already guaranteed structural validity before returning)
# ---------------------------------------------------------------------------

_MIN_TOTAL = 10
_MAX_TOTAL = 15
_MIN_PER_TYPE = 2
# 4 binary dimensions the README explicitly checks:
_N_DIMENSIONS = 4


def score_sub_queries(
    sub_queries: List[SubQuery],
) -> Dict[str, Any]:
    """
    Score a list of validated SubQuery objects against the README spec.

    Dimensions (each binary, equally weighted):
      1. count_in_range      — 10 ≤ len ≤ 15
      2. all_6_types_present — all 6 valid types appear ≥ 1 time
      3. min_2_per_type      — every type appears ≥ 2 times

    composite_score = passed / _N_DIMENSIONS  (range 0.0–1.0)

    Note: ``json_valid`` is implicitly True here — the engine raises
    ``LLMUnavailableError`` before returning if the JSON was bad.
    That failure path is recorded as an error row in the score CSV.
    """
    counts: Dict[str, int] = {t: 0 for t in _VALID_TYPES}
    for sq in sub_queries:
        if sq.type in counts:
            counts[sq.type] += 1

    total            = len(sub_queries)
    count_ok         = _MIN_TOTAL <= total <= _MAX_TOTAL
    all_types_ok     = all(counts[t] >= 1 for t in _VALID_TYPES)
    min_per_type_ok  = all(counts[t] >= _MIN_PER_TYPE for t in _VALID_TYPES)
    missing          = sorted(t for t in _VALID_TYPES if counts[t] == 0)

    passed           = sum([count_ok, all_types_ok, min_per_type_ok])
    composite        = round(passed / _N_DIMENSIONS, 4)

    return {
        "total_sub_queries":   total,
        "count_in_range":      count_ok,
        "all_6_types_present": all_types_ok,
        "min_2_per_type":      min_per_type_ok,
        "types_breakdown":     json.dumps(counts),
        "missing_types":       ", ".join(missing),
        "composite_score":     composite,
        "error":               "",
    }


def _error_score(exc: Exception) -> Dict[str, Any]:
    """Return a zero-score dict for a query that failed entirely."""
    return {
        "total_sub_queries":   0,
        "count_in_range":      False,
        "all_6_types_present": False,
        "min_2_per_type":      False,
        "types_breakdown":     json.dumps({t: 0 for t in _VALID_TYPES}),
        "missing_types":       ", ".join(sorted(_VALID_TYPES)),
        "composite_score":     0.0,
        "error":               str(exc),
    }


# ---------------------------------------------------------------------------
# Per-query worker  (runs inside a thread-pool worker)
# ---------------------------------------------------------------------------


def _run_query(
    run_id: str,
    config: FanOutConfig,
    query_id: str,
    target_query: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Generate sub-queries for one target query and return:
      - list of sub-query row dicts  (for the subqueries CSV)
      - one score row dict           (for the scores CSV)

    Designed to be called from a ThreadPoolExecutor worker.
    """
    try:
        sub_queries = generate_sub_queries(target_query, config)
    except LLMUnavailableError as exc:
        score_row = {"run_id": run_id, "query_id": query_id,
                     "target_query": target_query, **_error_score(exc)}
        return [], score_row

    sq_rows = [
        {
            "run_id":          run_id,
            "query_id":        query_id,
            "target_query":    target_query,
            "sub_query_index": i,
            "sub_query_type":  sq.type,
            "sub_query_text":  sq.query,
        }
        for i, sq in enumerate(sub_queries, start=1)
    ]

    score_row = {
        "run_id":       run_id,
        "query_id":     query_id,
        "target_query": target_query,
        **score_sub_queries(sub_queries),
    }

    return sq_rows, score_row


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_eval() -> None:
    """Run all queries in parallel, write both CSVs, print a summary."""
    queries = json.loads(_DATA_FILE.read_text())
    run_id  = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    config  = FanOutConfig(model_name=_MODEL)

    print(f"\nFan-out eval  run_id={run_id}  model={_MODEL}")
    print(f"Queries: {len(queries)}  workers: {min(_BATCH_SIZE, len(queries))}\n")

    all_sq_rows:    List[Dict[str, Any]] = []
    all_score_rows: List[Dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=_BATCH_SIZE) as pool:
        futures = {
            pool.submit(_run_query, run_id, config, q["id"], q["query"]): q
            for q in queries
        }
        for future in as_completed(futures):
            q = futures[future]
            sq_rows, score_row = future.result()
            all_sq_rows.extend(sq_rows)
            all_score_rows.append(score_row)

            status = (
                f"{score_row['total_sub_queries']} sub-queries  "
                f"score={score_row['composite_score']:.2f}"
                if not score_row["error"]
                else f"FAILED — {score_row['error'][:60]}"
            )
            print(f"  [{q['id']}] {q['query'][:55]} … {status}")

    # Sort by query_id so the CSV is deterministic regardless of thread order
    all_sq_rows.sort(key=lambda r: (r["query_id"], r["sub_query_index"]))
    all_score_rows.sort(key=lambda r: r["query_id"])

    _LOGS_DIR.mkdir(parents=True, exist_ok=True)
    sq_path    = _LOGS_DIR / f"run_{run_id}_subqueries.csv"
    score_path = _LOGS_DIR / f"run_{run_id}_scores.csv"

    _write_csv(all_sq_rows, _SQ_FIELDS, sq_path)
    _write_csv(all_score_rows, _SCORE_FIELDS, score_path)

    print(f"\nSubqueries CSV → {sq_path}")
    print(f"Scores CSV     → {score_path}")
    _print_summary(all_score_rows)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_csv(rows: List[Dict[str, Any]], fields: List[str], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _print_summary(score_rows: List[Dict[str, Any]]) -> None:
    print("\n── Per-query score summary ────────────────────────────────────")
    header = f"{'query_id':>8}  {'total':>5}  {'range':>5}  {'6types':>6}  {'min2':>4}  {'score':>6}"
    print(header)
    print("-" * len(header))
    for r in score_rows:
        if r["error"]:
            print(f"  {r['query_id']:>6}  ERROR: {r['error'][:50]}")
            continue
        print(
            f"  {r['query_id']:>6}  "
            f"{r['total_sub_queries']:>5}  "
            f"{'✓' if r['count_in_range']      else '✗':>5}  "
            f"{'✓' if r['all_6_types_present'] else '✗':>6}  "
            f"{'✓' if r['min_2_per_type']      else '✗':>4}  "
            f"{r['composite_score']:>6.2f}"
        )
    avg = sum(r["composite_score"] for r in score_rows) / len(score_rows)
    print(f"\n  avg composite: {avg:.3f}")


if __name__ == "__main__":
    run_eval()
