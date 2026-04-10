"""
run_prompt_eval.py
~~~~~~~~~~~~~~~~~~
Iterative prompt evaluation runner.

Usage
-----
    python -m optimization.prompt_tuning.run_prompt_eval

What it does
------------
For every (prompt_version × target_query) combination:
  1. Calls the real LLM using the production fanout_engine.
  2. Captures the raw response string.
  3. Scores it with scorer.score_response().
  4. Appends one row to a timestamped CSV in optimization/prompt_tuning/logs/.

You open the CSV, read the raw_llm_response and scorer_notes columns,
tweak the prompt in app/services/fanout_prompts.py, and run again.
Each run produces a new file so you have a full history.

CSV columns
-----------
run_id | prompt_version | query_id | target_query | json_valid |
total_sub_queries | total_in_range | all_6_types_present |
min_2_per_type | no_extra_top_level_keys | composite_score |
scorer_notes | raw_llm_response

Environment variables required
-------------------------------
GEMINI_API_KEY  or  OPENAI_API_KEY   (depending on model)
"""
from __future__ import annotations

import csv
import json
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

# Make sure the repo root is on sys.path when run directly
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.services.fanout_engine import FanOutConfig, _call_llm  # noqa: E402
from app.services.fanout_prompts import ALL_PROMPTS, PromptConfig  # noqa: E402
from optimization.prompt_tuning.scorer import score_response  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_DATA_FILE = _REPO_ROOT / "optimization" / "data" / "prompt_eval_queries.json"
_LOGS_DIR = _REPO_ROOT / "optimization" / "prompt_tuning" / "logs"

# ---------------------------------------------------------------------------
# CSV columns — order determines column order in the output file
# ---------------------------------------------------------------------------

_CSV_FIELDS = [
    "run_id",
    "prompt_version",
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
    prompt: PromptConfig,
    query_id: str,
    target_query: str,
    model_name: str = "gemini-1.5-flash",
) -> Dict[str, Any]:
    """
    Call the LLM with one prompt × query pair, score the response, return a row dict.

    The raw LLM call is isolated here so the caller can catch errors per-pair
    without aborting the entire run.
    """
    user_message = prompt.user_template.format(target_query=target_query)

    try:
        raw_response = _call_llm(model_name, prompt.system_message, user_message)
    except Exception as exc:  # noqa: BLE001
        # LLM call itself failed — record as invalid JSON with error note
        raw_response = ""
        score = {
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
        "prompt_version": prompt.version_id,
        "query_id": query_id,
        "target_query": target_query,
        **score,
        "raw_llm_response": raw_response,
    }


def run_eval(
    prompts: List[PromptConfig] | None = None,
    model_name: str = "gemini-1.5-flash",
) -> List[Dict[str, Any]]:
    """
    Evaluate all prompts × all queries. Write CSV. Return rows.

    Parameters
    ----------
    prompts    : list of PromptConfig to evaluate. Defaults to ALL_PROMPTS.
    model_name : LLM model to use for all calls.
    """
    if prompts is None:
        prompts = ALL_PROMPTS

    queries = json.loads(_DATA_FILE.read_text())
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]

    print(
        f"\nStarting prompt eval run {run_id}\n"
        f"  Prompts : {[p.version_id for p in prompts]}\n"
        f"  Queries : {len(queries)}\n"
        f"  Total   : {len(prompts) * len(queries)} LLM calls\n"
    )

    rows: List[Dict[str, Any]] = []
    for prompt in prompts:
        for q in queries:
            print(f"  [{prompt.version_id}] {q['query']} …", end=" ", flush=True)
            row = _run_one(run_id, prompt, q["id"], q["query"], model_name)
            rows.append(row)
            print(f"score={row['composite_score']:.2f}")

    # Write CSV
    _LOGS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = _LOGS_DIR / f"run_{run_id}.csv"
    _write_csv(rows, csv_path)
    print(f"\nLog written → {csv_path}")

    # Print summary table per prompt version
    _print_summary(rows, prompts)

    return rows


def _write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _print_summary(rows: List[Dict[str, Any]], prompts: List[PromptConfig]) -> None:
    print("\n── Summary by prompt version ──────────────────────────────────")
    print(f"{'version':>20}  {'avg_composite':>14}  {'json_valid_rate':>16}")
    print("-" * 56)
    for prompt in prompts:
        prompt_rows = [r for r in rows if r["prompt_version"] == prompt.version_id]
        if not prompt_rows:
            continue
        avg_score = sum(r["composite_score"] for r in prompt_rows) / len(prompt_rows)
        json_rate = sum(1 for r in prompt_rows if r["json_valid"]) / len(prompt_rows)
        print(f"{prompt.version_id:>20}  {avg_score:>14.3f}  {json_rate:>15.1%}")


if __name__ == "__main__":
    run_eval()
