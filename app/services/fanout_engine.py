"""
fanout_engine.py
~~~~~~~~~~~~~~~~
LLM-powered query fan-out engine (OpenAI only).

Responsibility
--------------
1. Accept a target query and call OpenAI with retry + exponential back-off.
2. Parse and validate the raw LLM response string into a List[SubQuery].
3. Enforce structural constraints (min total, min per type).
4. Return the validated sub-query list or raise LLMUnavailableError.

What lives HERE
---------------
  - The prompt (single source of truth — _SYSTEM_PROMPT / _USER_TEMPLATE)
  - OpenAI client call
  - JSON extraction from raw text (fence stripping, json.loads, Pydantic)
  - Retry / back-off logic
  - Structural validation helpers (independently testable pure functions)

What does NOT live here
-----------------------
  - Embedding / cosine logic  →  app/services/gap_analyzer.py
  - HTTP routing              →  app/api/fanout.py
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from app.models.schemas import LLMUnavailableError, SubQuery

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Valid sub-query types
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Prompt — single source of truth
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a query decomposition engine for an AI content optimisation platform.

YOUR ROLE
You simulate how AI search engines (Perplexity, ChatGPT Search, Google AI Mode) break
a user query into sub-queries before generating a comprehensive answer. Your output is
used to identify gaps in a content article.

TASK
Given a target query, generate between 10 and 15 sub-queries that span all six types
listed below. Cover the query topic thoroughly and generically — the same prompt must
work for any domain (SEO tools, CRM software, project management, etc.).

THE SIX SUB-QUERY TYPES
You must use EXACTLY these identifiers as the "type" field value:

  comparative      — how the subject compares against competitors or alternatives
  feature_specific — a specific feature, capability, or attribute
  use_case         — a concrete real-world application or audience segment
  trust_signals    — social proof: used by teams/companies, most popular among professionals, trusted by industry
  how_to           — a procedural, instructional, or "how do I" angle
  definitional     — a conceptual, explanatory, or "what is" angle

HARD CONSTRAINTS
1. Generate at least 2 sub-queries for EVERY type (12 minimum across all six types).
2. Total sub-queries must be between 10 and 15 inclusive.
3. Each sub-query must be a realistic search query string — not a sentence fragment.
4. Keep each query concise: 4–9 words maximum. No redundant prepositions or filler
   words (avoid patterns like "X for Y for Z" — collapse to "X for Y Z" instead).
5. Write queries the way a real user would type them into a search engine — natural,
   direct, no unnecessary repetition.
6. Return ONLY a valid JSON object. No markdown fences. No prose. No extra fields.
7. The JSON must match this schema exactly:

{
  "sub_queries": [
    {"type": "<one of the six types above>", "query": "<search query string>"},
    ...
  ]
}

EXAMPLE — for target query "best project management software for remote teams":
{
  "sub_queries": [
    {"type": "comparative", "query": "Asana vs Monday.com for remote teams"},
    {"type": "comparative", "query": "Jira vs ClickUp distributed engineering"},
    {"type": "feature_specific", "query": "PM tool with async video updates"},
    {"type": "feature_specific", "query": "project management time zone features"},
    {"type": "use_case", "query": "project management for remote agencies"},
    {"type": "use_case", "query": "best PM tool distributed startup teams"},
    {"type": "trust_signals", "query": "project management tools used by large teams"},
    {"type": "trust_signals", "query": "most popular PM software among engineering teams"},
    {"type": "how_to", "query": "how to manage remote projects without meetings"},
    {"type": "how_to", "query": "track distributed team progress in real time"},
    {"type": "definitional", "query": "what is asynchronous project management"},
    {"type": "definitional", "query": "remote-first project workflow definition"}
  ]
}

Now generate sub-queries for the target query provided by the user.\
"""

_USER_TEMPLATE = "Target query: {target_query}"

# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------


@dataclass
class FanOutConfig:
    """
    Runtime configuration for one fan-out request.

    Parameters
    ----------
    model_name:
        OpenAI model identifier, e.g. ``"gpt-4o-mini"``.
    temperature:
        Sampling temperature passed to the OpenAI API (0.0–2.0).
        Lower values produce more deterministic output; default 0.2 matches
        production.  Vary this in the prompt optimizer to benchmark stability.
    min_total:
        Minimum number of sub-queries required.
    min_per_type:
        Minimum sub-queries per query type.
    max_retries:
        How many LLM call attempts before giving up.
    retry_base_delay:
        Seconds to wait before the first retry (doubles each attempt).
    """

    model_name: str = "gpt-4o-mini"
    temperature: float = 0.2
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
# LLM client helper  (OpenAI only)
# ---------------------------------------------------------------------------


def _call_llm(
    model_name: str,
    system_prompt: str,
    user_message: str,
    temperature: float = 0.2,
) -> str:
    """Call OpenAI and return the raw text response."""
    from openai import OpenAI  # lazy import — keeps startup fast

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
        temperature=temperature,
    )
    return response.choices[0].message.content or ""


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
    user_message = _USER_TEMPLATE.format(target_query=target_query)
    last_error: Optional[str] = None

    for attempt in range(1, config.max_retries + 1):
        try:
            logger.info(
                "Fan-out attempt %d/%d for query: %r",
                attempt,
                config.max_retries,
                target_query,
            )
            raw = _call_llm(
                config.model_name,
                _SYSTEM_PROMPT,
                user_message,
                config.temperature,
            )

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
