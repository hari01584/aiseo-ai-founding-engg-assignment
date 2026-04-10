"""
scorer.py
~~~~~~~~~
Structural scorer for raw LLM fan-out responses.

No human labels needed — scores a response purely on structural compliance:
  1. JSON validity
  2. Total sub-query count (between 10 and 15)
  3. All 6 required types present
  4. Each type has at least 2 sub-queries
  5. No unexpected top-level fields beyond "sub_queries"

Each dimension is binary (pass/fail) and contributes equally to a composite
score between 0.0 and 1.0.

These are pure functions — no LLM calls, no file I/O, no model loading.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

from app.services.fanout_engine import (
    _VALID_TYPES,
    parse_llm_response,
    validate_sub_queries,
)

# Expected top-level keys in the LLM JSON response
_ALLOWED_TOP_LEVEL_KEYS = frozenset(["sub_queries", "queries", "results"])

# Score weights — all equal for now; easy to adjust
_WEIGHTS = {
    "json_valid": 1,
    "total_in_range": 1,
    "all_6_types_present": 1,
    "min_2_per_type": 1,
    "no_extra_top_level_keys": 1,
}
_MAX_SCORE = sum(_WEIGHTS.values())


def score_response(
    raw_llm_response: str,
    min_total: int = 10,
    max_total: int = 15,
    min_per_type: int = 2,
) -> Dict[str, Any]:
    """
    Score a raw LLM response string on structural compliance.

    Parameters
    ----------
    raw_llm_response : the raw string returned by the LLM
    min_total        : minimum required sub-queries
    max_total        : maximum allowed sub-queries
    min_per_type     : minimum sub-queries per type

    Returns
    -------
    {
      "json_valid":               bool,
      "total_sub_queries":        int | None,
      "total_in_range":           bool,
      "all_6_types_present":      bool,
      "min_2_per_type":           bool,
      "no_extra_top_level_keys":  bool,
      "composite_score":          float,   # 0.0–1.0
      "scorer_notes":             str,     # human-readable summary
    }
    """
    result: Dict[str, Any] = {
        "json_valid": False,
        "total_sub_queries": None,
        "total_in_range": False,
        "all_6_types_present": False,
        "min_2_per_type": False,
        "no_extra_top_level_keys": False,
        "composite_score": 0.0,
        "scorer_notes": "",
    }
    notes: List[str] = []

    # ── 1. JSON validity ────────────────────────────────────────────────────
    try:
        raw_list = parse_llm_response(raw_llm_response)
        result["json_valid"] = True
    except (json.JSONDecodeError, ValueError) as exc:
        notes.append(f"JSON invalid: {exc}")
        result["scorer_notes"] = " | ".join(notes)
        return result  # nothing else can be scored

    # ── 2. Validate sub-queries through Pydantic ────────────────────────────
    sub_queries = validate_sub_queries(raw_list)
    result["total_sub_queries"] = len(sub_queries)

    # ── 3. Total count in range ─────────────────────────────────────────────
    if min_total <= len(sub_queries) <= max_total:
        result["total_in_range"] = True
    else:
        notes.append(
            f"Count out of range: got {len(sub_queries)}, "
            f"expected {min_total}–{max_total}"
        )

    # ── 4. All 6 types present ──────────────────────────────────────────────
    found_types = {sq.type for sq in sub_queries}
    missing = _VALID_TYPES - found_types
    if not missing:
        result["all_6_types_present"] = True
    else:
        notes.append(f"Missing types: {sorted(missing)}")

    # ── 5. Min per type ─────────────────────────────────────────────────────
    type_counts: Dict[str, int] = {t: 0 for t in _VALID_TYPES}
    for sq in sub_queries:
        if sq.type in type_counts:
            type_counts[sq.type] += 1
    under = {t: c for t, c in type_counts.items() if c < min_per_type}
    if not under:
        result["min_2_per_type"] = True
    else:
        notes.append(f"Under-represented types: {under}")

    # ── 6. No extra top-level keys ──────────────────────────────────────────
    try:
        from app.services.fanout_engine import strip_markdown_fences
        parsed_json = json.loads(strip_markdown_fences(raw_llm_response))
        if isinstance(parsed_json, dict):
            extra_keys = set(parsed_json.keys()) - _ALLOWED_TOP_LEVEL_KEYS
            if not extra_keys:
                result["no_extra_top_level_keys"] = True
            else:
                notes.append(f"Extra top-level keys: {sorted(extra_keys)}")
        else:
            # Top-level list — no envelope keys to check
            result["no_extra_top_level_keys"] = True
    except Exception:  # noqa: BLE001
        result["no_extra_top_level_keys"] = False

    # ── Composite score ─────────────────────────────────────────────────────
    earned = sum(
        _WEIGHTS[k] for k in _WEIGHTS if result.get(k) is True
    )
    result["composite_score"] = round(earned / _MAX_SCORE, 4)
    result["scorer_notes"] = " | ".join(notes) if notes else "all checks passed"

    return result
