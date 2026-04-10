"""
run_prompt_eval.py
~~~~~~~~~~~~~~~~~~
Model / temperature evaluation runner.

Usage
-----
    python -m optimization.prompt_tuning.run_prompt_eval

What it does
------------
Runs the single production prompt across a fixed set of target queries,
optionally varying model name and/or temperature.  For each (variant × query):
  1. Calls the real OpenAI API using the production _call_llm helper.
  2. Captures the raw response string.
  3. Scores it with scorer.score_response().
  4. Appends one row to a timestamped CSV in optimization/prompt_tuning/logs/.

Each run produces a new file so you have a full history.

CSV columns
-----------
run_id | variant_id | model_name | temperature | query_id | target_query |
json_valid | total_sub_queries | total_in_range | all_6_types_present |
min_2_per_type | no_extra_top_level_keys | composite_score |
scorer_notes | raw_llm_response

Environment variables required
-------------------------------
OPENAI_API_KEY
"""
from __future__ import annotations

import csv
import json
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Make sure the repo root is on sys.path when run directly
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.services.fanout_engine import _SYSTEM_PROMPT, _USER_TEMPLATE, _call_llm  # noqa: E402
from optimization.prompt_tuning.scorer import score_response  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_DATA_FILE = _REPO_ROOT / "optimization" / "data" / "prompt_eval_queries.json"
_LOGS_DIR = _REPO_ROOT / "optimization" / "prompt_tuning" / "logs"

# ---------------------------------------------------------------------------
# Variant definition — one row per (model, temperature) combination
# ---------------------------------------------------------------------------


@dataclass
class EvalVariant:
    """A single evaluation configuration to benchmark."""

    variant_id: str          # e.g. "gpt-4o-mini-t02"
    model_name: str          # OpenAI model identifier
    temperature: float = 0.2  # default matches production


# Default variants grid — edit to add/remove combinations
DEFAULT_VARIANTS: List[EvalVariant] = [
    EvalVariant("gpt-4o-mini-t02", "gpt-4o-mini", temperature=0.2),
    EvalVariant("gpt-4o-mini-t05", "gpt-4o-mini", temperature=0.5),
    EvalVariant("gpt-4o-t02",      "gpt-4o",      temperature=0.2),
]

# ---------------------------------------------------------------------------
# CSV columns
# ---------------------------------------------------------------------------

_CSV_FIELDS = [
    "run_id",
    "variant_id",
    "model_name",
    "temperature",
    "query_id",
    "target_query",
    "json_valid",
    "total_sub_queries",
    "total_in_range",
    "all_6_types_present",
    "min_2_per_type",
    "no_extra_top_level_keys",
    "composite_score",
    "scorer_notes",
    "raw_llm_response",
]


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def _run_one(
    run_id: str,
    variant: EvalVariant,
    query_id: str,
    target_query: str,
) -> Dict[str, Any]:
    """
    Call OpenAI with the production prompt for one variant × query pair.
    Score the response and return a row dict.
    """
    from openai import OpenAI
    import os

    user_message = _USER_TEMPLATE.format(target_query=target_query)

    try:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY is not set")
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=variant.model_name,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=variant.temperature,
        )
        raw_response = response.choices[0].message.content or ""
    except Exception as exc:  # noqa: BLE001
        raw_response = ""
        score: Dict[str, Any] = {
            "json_valid": False,
            "total_sub_queries": None,
            "total_in_range": False,
            "all_6_types_present": False,
            "min_2_per_type": False,
            "no_extra_top_level_keys": False,
            "composite_score": 0.0,
            "scorer_notes": f"LLM call failed: {exc}",
        }
    else:
        score = score_response(raw_response)

    return {
        "run_id": run_id,
        "variant_id": variant.variant_id,
        "model_name": variant.model_name,
        "temperature": variant.temperature,
        "query_id": query_id,
        "target_query": target_query,
        **score,
        "raw_llm_response": raw_response,
    }


def run_eval(
    variants: Optional[List[EvalVariant]] = None,
) -> List[Dict[str, Any]]:
    """
    Evaluate all variants × all queries. Write CSV. Return rows.

    Parameters
    ----------
    variants : list of EvalVariant to benchmark. Defaults to DEFAULT_VARIANTS.
    """
    if variants is None:
        variants = DEFAULT_VARIANTS

    queries = json.loads(_DATA_FILE.read_text())
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]

    print(
        f"\nStarting eval run {run_id}\n"
        f"  Variants : {[v.variant_id for v in variants]}\n"
        f"  Queries  : {len(queries)}\n"
        f"  Total    : {len(variants) * len(queries)} LLM calls\n"
    )

    rows: List[Dict[str, Any]] = []
    for variant in variants:
        for q in queries:
            print(f"  [{variant.variant_id}] {q['query']} …", end=" ", flush=True)
            row = _run_one(run_id, variant, q["id"], q["query"])
            rows.append(row)
            print(f"score={row['composite_score']:.2f}")

    # Write CSV
    _LOGS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = _LOGS_DIR / f"run_{run_id}.csv"
    _write_csv(rows, csv_path)
    print(f"\nLog written → {csv_path}")

    _print_summary(rows, variants)
    return rows


def _write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _print_summary(rows: List[Dict[str, Any]], variants: List[EvalVariant]) -> None:
    print("\n── Summary by variant ─────────────────────────────────────────")
    print(f"{'variant_id':>24}  {'avg_composite':>14}  {'json_valid_rate':>16}")
    print("-" * 60)
    for variant in variants:
        v_rows = [r for r in rows if r["variant_id"] == variant.variant_id]
        if not v_rows:
            continue
        avg_score = sum(r["composite_score"] for r in v_rows) / len(v_rows)
        json_rate = sum(1 for r in v_rows if r["json_valid"]) / len(v_rows)
        print(f"{variant.variant_id:>24}  {avg_score:>14.3f}  {json_rate:>15.1%}")


if __name__ == "__main__":
    run_eval()
