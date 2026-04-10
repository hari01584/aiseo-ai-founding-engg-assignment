"""
run_prompt_eval.py
~~~~~~~~~~~~~~~~~~
Fan-out output logger — generates a single combined CSV with one row per
query: all sub-queries grouped by type in markdown, plus per-query scores.

Usage
-----
    python -m optimization.prompt_tuning.run_prompt_eval

What it does
------------
For each target query in ``optimization/data/prompt_eval_queries.json``:
  1. Calls ``generate_sub_queries`` (production engine — same retry logic,
     same validation, same prompt as the live API).
  2. All 5 queries are executed in parallel (batch size = 5).
  3. Writes one combined CSV to ``optimization/prompt_tuning/logs/``.

CSV layout — one row per query
-------------------------------
  run_id | query_id | target_query
  [per type column]  comparative | feature_specific | use_case |
                     trust_signals | how_to | definitional
      Each cell contains a markdown block:
        ## comparative
        - Jasper vs Copy.ai for SEO writing
        - Surfer SEO vs Frase for AI content
  total_sub_queries | count_in_range | all_6_types_present | min_2_per_type
  composite_score   | missing_types  | error

Score dimensions (from README requirements)
-------------------------------------------
  count_in_range      — 10 ≤ total ≤ 15
  all_6_types_present — all 6 type identifiers appear at least once
  min_2_per_type      — every type has ≥ 2 sub-queries
  composite_score     — passed / 3  → 0.0–1.0
  missing_types       — comma-separated list of types with 0 sub-queries

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
from typing import Any, Dict, List, Tuple

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

# Types in a stable display order
_TYPE_ORDER: List[str] = [
    "comparative",
    "feature_specific",
    "use_case",
    "trust_signals",
    "how_to",
    "definitional",
]

_CSV_FIELDS: List[str] = [
    "run_id",
    "query_id",
    "target_query",
    *_TYPE_ORDER,             # one markdown column per type
    "total_sub_queries",
    "count_in_range",
    "all_6_types_present",
    "min_2_per_type",
    "composite_score",
    "missing_types",
    "error",
]

# ---------------------------------------------------------------------------
# Markdown formatter
# ---------------------------------------------------------------------------


def _type_markdown(type_name: str, queries: List[str]) -> str:
    """
    Render one type's sub-queries as a compact markdown block.

      ## comparative
      - Jasper vs Copy.ai for SEO
      - Surfer SEO vs Frase for AI content

    Empty if no queries were generated for this type.
    """
    if not queries:
        return ""
    lines = [f"## {type_name}"] + [f"- {q}" for q in queries]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Post-analysis scorer
# ---------------------------------------------------------------------------

_MIN_TOTAL    = 10
_MAX_TOTAL    = 15
_MIN_PER_TYPE = 2
_N_DIMENSIONS = 3   # count_in_range, all_6_types_present, min_2_per_type


def _score(sub_queries: List[SubQuery]) -> Dict[str, Any]:
    """
    Score validated SubQuery objects against the README spec.

    Returns a flat dict ready to merge into a CSV row.
    ``json_valid`` is implicitly True — the engine raises before returning
    on bad JSON; that failure is recorded in the ``error`` column instead.
    """
    counts: Dict[str, int] = {t: 0 for t in _VALID_TYPES}
    grouped: Dict[str, List[str]] = {t: [] for t in _VALID_TYPES}
    for sq in sub_queries:
        if sq.type in counts:
            counts[sq.type] += 1
            grouped[sq.type].append(sq.query)

    total           = len(sub_queries)
    count_ok        = _MIN_TOTAL <= total <= _MAX_TOTAL
    all_types_ok    = all(counts[t] >= 1 for t in _VALID_TYPES)
    min_per_type_ok = all(counts[t] >= _MIN_PER_TYPE for t in _VALID_TYPES)
    missing         = sorted(t for t in _VALID_TYPES if counts[t] == 0)
    under           = sorted(
        f"{t}({counts[t]})" for t in _VALID_TYPES
        if 0 < counts[t] < _MIN_PER_TYPE
    )

    # missing_types shows zero-count types; also flag under-represented ones
    missing_label = ", ".join(missing)
    if under:
        missing_label += (" | under-represented: " + ", ".join(under)) if missing_label else "under-represented: " + ", ".join(under)

    passed    = sum([count_ok, all_types_ok, min_per_type_ok])
    composite = round(passed / _N_DIMENSIONS, 4)

    return {
        **{t: _type_markdown(t, grouped[t]) for t in _TYPE_ORDER},
        "total_sub_queries":   total,
        "count_in_range":      count_ok,
        "all_6_types_present": all_types_ok,
        "min_2_per_type":      min_per_type_ok,
        "composite_score":     composite,
        "missing_types":       missing_label,
        "error":               "",
    }


def _error_row() -> Dict[str, Any]:
    """Zero-score row used when the engine raises LLMUnavailableError."""
    return {
        **{t: "" for t in _TYPE_ORDER},
        "total_sub_queries":   0,
        "count_in_range":      False,
        "all_6_types_present": False,
        "min_2_per_type":      False,
        "composite_score":     0.0,
        "missing_types":       ", ".join(_TYPE_ORDER),
        "error":               "",   # filled in by caller
    }


# ---------------------------------------------------------------------------
# Per-query worker
# ---------------------------------------------------------------------------


def _run_query(
    run_id: str,
    config: FanOutConfig,
    query_id: str,
    target_query: str,
) -> Dict[str, Any]:
    """Generate sub-queries and return one combined CSV row dict."""
    base = {"run_id": run_id, "query_id": query_id, "target_query": target_query}
    try:
        sub_queries = generate_sub_queries(target_query, config)
        return {**base, **_score(sub_queries)}
    except LLMUnavailableError as exc:
        row = _error_row()
        row["error"] = str(exc)
        return {**base, **row}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_eval() -> None:
    """Run all queries in parallel, write combined CSV, print a summary."""
    queries = json.loads(_DATA_FILE.read_text())
    run_id  = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    config  = FanOutConfig(model_name=_MODEL)

    print(f"\nFan-out eval  run_id={run_id}  model={_MODEL}")
    print(f"Queries: {len(queries)}  workers: {min(_BATCH_SIZE, len(queries))}\n")

    rows: List[Dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=_BATCH_SIZE) as pool:
        futures = {
            pool.submit(_run_query, run_id, config, q["id"], q["query"]): q
            for q in queries
        }
        for future in as_completed(futures):
            q   = futures[future]
            row = future.result()
            rows.append(row)

            status = (
                f"{row['total_sub_queries']} sub-queries  score={row['composite_score']:.2f}"
                if not row["error"]
                else f"FAILED — {row['error'][:60]}"
            )
            print(f"  [{q['id']}] {q['query'][:55]} … {status}")

    rows.sort(key=lambda r: r["query_id"])

    _LOGS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = _LOGS_DIR / f"run_{run_id}.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nCSV written → {csv_path}")
    _print_summary(rows)


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------


def _print_summary(rows: List[Dict[str, Any]]) -> None:
    print("\n── Per-query score summary ────────────────────────────────────")
    print(f"{'query_id':>8}  {'total':>5}  {'range':>5}  {'6types':>6}  {'min2':>4}  {'score':>6}  missing")
    print("-" * 72)
    for r in rows:
        if r["error"]:
            print(f"  {r['query_id']:>6}  ERROR: {r['error'][:55]}")
            continue
        missing = r["missing_types"] or "—"
        print(
            f"  {r['query_id']:>6}  "
            f"{r['total_sub_queries']:>5}  "
            f"{'✓' if r['count_in_range']      else '✗':>5}  "
            f"{'✓' if r['all_6_types_present'] else '✗':>6}  "
            f"{'✓' if r['min_2_per_type']      else '✗':>4}  "
            f"{r['composite_score']:>6.2f}  {missing}"
        )
    ok   = [r for r in rows if not r["error"]]
    avg  = sum(r["composite_score"] for r in ok) / len(ok) if ok else 0.0
    print(f"\n  avg composite: {avg:.3f}  ({len(ok)}/{len(rows)} queries succeeded)")


if __name__ == "__main__":
    run_eval()
