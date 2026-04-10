"""
fanout_engine.py
~~~~~~~~~~~~~~~~
LLM-powered query fan-out engine.

Responsibility
--------------
1. Accept a target query + a prompt config object (injected — no hardcoded text).
2. Call the configured LLM with retry + exponential back-off.
3. Parse and validate the raw LLM response string into a List[SubQuery].
4. Enforce structural constraints (min total, min per type).
5. Return the validated sub-query list or raise LLMUnavailableError.

What lives HERE
---------------
  - LLM client wrappers (Gemini / OpenAI — selected by env var)
  - JSON extraction from raw text (fence stripping, json.loads, Pydantic)
  - Retry / back-off logic
  - Structural validation helpers (independently testable pure functions)

What does NOT live here
-----------------------
  - Prompt text strings  →  app/services/fanout_prompts.py
  - Embedding / cosine logic  →  app/services/gap_analyzer.py
  - HTTP routing  →  app/api/fanout.py
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.models.schemas import LLMUnavailableError, SubQuery, SubQueryType

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration dataclass — injected at call site, never hardcoded here
# ---------------------------------------------------------------------------

#: All valid sub-query type literals in one place for validation
_VALID_TYPES: frozenset[str] = frozenset(
    [
        "comparative",
        "feature_specific",
        "use_case",
        "trust_signals",
        "how_to",
        "definitional",
    ]
)


@dataclass
class FanOutConfig:
    """
    Injected configuration for one fan-out request.

    Parameters
    ----------
    system_prompt:
        The system-role message sent to the LLM.  Must instruct the model to
        return a JSON object with a ``sub_queries`` array.  Lives in
        fanout_prompts.py — never inlined here.
    user_template:
        A format-string with a single ``{target_query}`` placeholder.
    model_name:
        LLM model identifier, e.g. ``"gemini-1.5-flash"`` or
        ``"gpt-4o-mini"``.
    min_total:
        Minimum number of sub-queries required.  Raise if the LLM returns
        fewer even after retries.
    min_per_type:
        Minimum sub-queries per query type.
    max_retries:
        How many LLM call attempts before giving up.
    retry_base_delay:
        Seconds to wait before the first retry (doubles each attempt).
    """

    system_prompt: str
    user_template: str
    model_name: str = "gemini-1.5-flash"
    min_total: int = 10
    min_per_type: int = 2
    max_retries: int = 3
    retry_base_delay: float = 1.0


# ---------------------------------------------------------------------------
# Pure helper functions  (independently testable — no LLM, no I/O)
# ---------------------------------------------------------------------------


def strip_markdown_fences(text: str) -> str:
    """
    Remove surrounding markdown code fences that some LLMs add.

    Handles:
      ```json\\n{...}\\n```
      ```\\n{...}\\n```
      plain JSON without fences
    """
    stripped = text.strip()
    # Match an optional language tag after the opening fence
    fence_pattern = re.compile(r"^```[a-zA-Z]*\s*([\s\S]*?)\s*```$", re.DOTALL)
    match = fence_pattern.match(stripped)
    if match:
        return match.group(1).strip()
    return stripped


def parse_llm_response(raw: str) -> List[Dict[str, Any]]:
    """
    Extract a list of sub-query dicts from the LLM's raw output string.

    Steps:
      1. Strip markdown fences.
      2. json.loads — raises json.JSONDecodeError on failure.
      3. Accept either a top-level list  OR  ``{"sub_queries": [...]}``
         envelope (LLMs sometimes wrap the array).
      4. Return the raw list of dicts for downstream Pydantic validation.

    Raises
    ------
    json.JSONDecodeError
        If the text is not valid JSON after fence stripping.
    ValueError
        If the JSON is valid but does not contain a list of objects.
    """
    clean = strip_markdown_fences(raw)
    data = json.loads(clean)  # may raise json.JSONDecodeError

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        # Accept {"sub_queries": [...]} or {"queries": [...]}
        for key in ("sub_queries", "queries", "results"):
            if key in data and isinstance(data[key], list):
                return data[key]

    raise ValueError(
        f"LLM response JSON is valid but has unexpected structure: {type(data)}"
    )


def validate_sub_queries(
    raw_list: List[Dict[str, Any]],
) -> List[SubQuery]:
    """
    Validate each raw dict through the SubQuery Pydantic model.

    - Unknown ``type`` values are skipped (logged at WARNING).
    - Dicts missing required fields are skipped (logged at WARNING).
    - Extra fields are silently ignored by Pydantic (model_config).

    Returns a list containing only the successfully validated SubQuery objects.
    """
    valid: List[SubQuery] = []
    for i, item in enumerate(raw_list):
        try:
            sq = SubQuery.model_validate(item)
            valid.append(sq)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Sub-query #%d failed validation (skipped): %s", i, exc)
    return valid


def check_structural_constraints(
    sub_queries: List[SubQuery],
    min_total: int,
    min_per_type: int,
) -> List[str]:
    """
    Return a list of human-readable violation strings.
    An empty list means all constraints are satisfied.

    Checks:
      - Total count ≥ min_total
      - Each of the 6 valid types appears ≥ min_per_type times
    """
    violations: List[str] = []

    if len(sub_queries) < min_total:
        violations.append(
            f"Too few sub-queries: got {len(sub_queries)}, need ≥ {min_total}"
        )

    type_counts: Dict[str, int] = {t: 0 for t in _VALID_TYPES}
    for sq in sub_queries:
        if sq.type in type_counts:
            type_counts[sq.type] += 1

    for qtype, count in type_counts.items():
        if count < min_per_type:
            violations.append(
                f"Type '{qtype}' has {count} sub-queries, need ≥ {min_per_type}"
            )

    return violations


# ---------------------------------------------------------------------------
# LLM client helpers  (thin wrappers — swap provider via env var)
# ---------------------------------------------------------------------------


def _call_gemini(model_name: str, system_prompt: str, user_message: str) -> str:
    """Call Google Gemini and return the raw text response."""
    import google.generativeai as genai  # lazy import — optional dep

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY environment variable is not set")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name=model_name,
        system_instruction=system_prompt,
    )
    response = model.generate_content(user_message)
    return response.text


def _call_openai(model_name: str, system_prompt: str, user_message: str) -> str:
    """Call OpenAI and return the raw text response."""
    from openai import OpenAI  # lazy import — optional dep

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY environment variable is not set")

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content or ""


def _call_llm(model_name: str, system_prompt: str, user_message: str) -> str:
    """
    Dispatch to the correct provider based on model name prefix.

    ``gpt-*``    → OpenAI
    ``gemini-*`` → Google Gemini
    """
    if model_name.startswith("gpt"):
        return _call_openai(model_name, system_prompt, user_message)
    return _call_gemini(model_name, system_prompt, user_message)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_sub_queries(
    target_query: str,
    config: FanOutConfig,
) -> List[SubQuery]:
    """
    Generate fan-out sub-queries for *target_query* using the given config.

    Retry loop
    ----------
    Each attempt:
      1. Build the user message from config.user_template.
      2. Call the LLM.
      3. parse_llm_response  →  validate_sub_queries  →  check_structural_constraints.
      4. If constraints pass → return.
      5. If constraints fail or parsing fails → log and retry.

    After max_retries exhausted → raise LLMUnavailableError.

    Parameters
    ----------
    target_query:
        The user's original query string.
    config:
        Injected FanOutConfig — prompt text and model settings live here.

    Returns
    -------
    List[SubQuery]
        Validated, structurally-sound sub-queries.  ``covered`` and
        ``similarity_score`` fields are always None here; gap_analyzer
        fills them in.

    Raises
    ------
    LLMUnavailableError
        If all retries are exhausted without a valid response.
    """
    user_message = config.user_template.format(target_query=target_query)
    last_error: Optional[str] = None

    for attempt in range(1, config.max_retries + 1):
        try:
            logger.info(
                "Fan-out attempt %d/%d for query: %r",
                attempt,
                config.max_retries,
                target_query,
            )
            raw = _call_llm(config.model_name, config.system_prompt, user_message)

            raw_list = parse_llm_response(raw)
            sub_queries = validate_sub_queries(raw_list)
            violations = check_structural_constraints(
                sub_queries, config.min_total, config.min_per_type
            )

            if violations:
                last_error = "; ".join(violations)
                logger.warning(
                    "Attempt %d structural violations: %s", attempt, last_error
                )
            else:
                logger.info(
                    "Fan-out succeeded on attempt %d: %d sub-queries",
                    attempt,
                    len(sub_queries),
                )
                return sub_queries

        except json.JSONDecodeError as exc:
            last_error = f"JSONDecodeError on attempt {attempt}: {exc}"
            logger.warning(last_error)
        except ValueError as exc:
            last_error = f"ValueError on attempt {attempt}: {exc}"
            logger.warning(last_error)
        except Exception as exc:  # noqa: BLE001
            last_error = f"Unexpected error on attempt {attempt}: {exc}"
            logger.error(last_error)

        # Exponential back-off before retry (skip sleep on last attempt)
        if attempt < config.max_retries:
            delay = config.retry_base_delay * (2 ** (attempt - 1))
            logger.info("Retrying in %.1fs…", delay)
            time.sleep(delay)

    raise LLMUnavailableError(
        f"Fan-out generation failed after {config.max_retries} retries. "
        f"Last error: {last_error}"
    )
