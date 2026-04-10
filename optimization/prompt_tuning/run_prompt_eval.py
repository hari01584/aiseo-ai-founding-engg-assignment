"""
run_prompt_eval.py
~~~~~~~~~~~~~~~~~~
Fan-out output logger — generates a CSV of the actual sub-queries produced
by ``gpt-4o-mini`` for each target query in the evaluation set.

Usage
-----
    python -m optimization.prompt_tuning.run_prompt_eval

What it does
------------
For each target query in ``optimization/data/prompt_eval_queries.json``:
  1. Calls ``generate_sub_queries`` (production engine — same retry logic,
     same validation, same prompt as the live API).
  2. Writes one CSV row per sub-query to a timestamped file in
     ``optimization/prompt_tuning/logs/``.

Open the CSV in any spreadsheet tool to inspect what the model produced.

CSV columns
-----------
run_id | query_id | target_query | sub_query_index |
sub_query_type | sub_query_text

Environment variables required
-------------------------------
OPENAI_API_KEY  (loaded automatically from .env if present)
"""
from __future__ import annotations

import csv
import json
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# ── sys.path & env setup (must come before app imports) ────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(_REPO_ROOT / ".env")

from app.models.schemas import LLMUnavailableError  # noqa: E402
from app.services.fanout_engine import FanOutConfig, generate_sub_queries  # noqa: E402

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

_DATA_FILE = _REPO_ROOT / "optimization" / "data" / "prompt_eval_queries.json"
_LOGS_DIR  = _REPO_ROOT / "optimization" / "prompt_tuning" / "logs"
_MODEL     = "gpt-4o-mini"

_CSV_FIELDS = [
    "run_id",
    "query_id",
    "target_query",
    "sub_query_index",
    "sub_query_type",
    "sub_query_text",
]

# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------


def run_eval() -> List[Dict[str, Any]]:
    """Run the evaluation and write a CSV. Returns all rows."""
    queries = json.loads(_DATA_FILE.read_text())
    run_id  = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    config  = FanOutConfig(model_name=_MODEL)

    print(f"\nFan-out eval  run_id={run_id}  model={_MODEL}")
    print(f"Queries: {len(queries)}\n")

    rows: List[Dict[str, Any]] = []

    for q in queries:
        print(f"  [{q['id']}] {q['query']} …", end=" ", flush=True)
        try:
            sub_queries = generate_sub_queries(q["query"], config)
            for i, sq in enumerate(sub_queries, start=1):
                rows.append({
                    "run_id":          run_id,
                    "query_id":        q["id"],
                    "target_query":    q["query"],
                    "sub_query_index": i,
                    "sub_query_type":  sq.type,
                    "sub_query_text":  sq.query,
                })
            print(f"{len(sub_queries)} sub-queries ✓")
        except LLMUnavailableError as exc:
            print(f"FAILED — {exc}")

    _LOGS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = _LOGS_DIR / f"run_{run_id}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nCSV written → {csv_path}")
    print(f"Total sub-queries logged: {len(rows)}")
    return rows


if __name__ == "__main__":
    run_eval()
