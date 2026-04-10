"""
fanout_prompts.py
~~~~~~~~~~~~~~~~~
All prompt text for the fan-out engine lives here — nowhere else.

Structure
---------
Each prompt version is a PromptConfig dataclass that pairs a system message
with a user-message template.  The API layer imports DEFAULT_PROMPT.
Optimization scripts import any version they want to evaluate.

Versioning convention
---------------------
  V1  — first draft (baseline, kept for history)
  V2  — added explicit JSON schema + type list
  V3  — current production prompt (structured, defensive, example-driven)

Why prompts live in a separate module
--------------------------------------
  fanout_engine.py contains only logic (parsing, retry, validation).
  Keeping prompt text here means:
    - The engine is testable without caring about prompt content.
    - PROMPT_LOG.md can reference version IDs directly.
    - Optimization scripts can import V1/V2/V3 without touching engine code.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptConfig:
    """A versioned prompt pair injected into FanOutConfig."""

    version_id: str
    description: str
    system_message: str
    user_template: str  # must contain exactly one {target_query} placeholder


# ---------------------------------------------------------------------------
# V1 — Naive single-paragraph prompt (baseline, kept for comparison)
# ---------------------------------------------------------------------------

PROMPT_V1 = PromptConfig(
    version_id="v1_naive",
    description=(
        "Single-paragraph instruction. No schema, no example. "
        "Baseline — known to hallucinate extra fields and return markdown."
    ),
    system_message=(
        "You are a search query expert. "
        "Generate 10 to 15 sub-queries that explore different angles of the user's query. "
        "Return your answer as a JSON array of objects with 'type' and 'query' fields."
    ),
    user_template="Query: {target_query}",
)


# ---------------------------------------------------------------------------
# V2 — Explicit type list + JSON schema, no few-shot example
# ---------------------------------------------------------------------------

PROMPT_V2 = PromptConfig(
    version_id="v2_schema",
    description=(
        "Added explicit JSON schema and required type list. "
        "Reduced hallucinated fields. Still sometimes returns markdown fences."
    ),
    system_message=(
        "You are a search query decomposition engine used inside an AI content platform.\n\n"
        "TASK\n"
        "Generate between 10 and 15 sub-queries that cover all the ways an AI search engine\n"
        "might decompose the user's query when building a comprehensive answer.\n\n"
        "SUB-QUERY TYPES — you must use EXACTLY these six type identifiers:\n"
        "  comparative      — compare the subject against alternatives\n"
        "  feature_specific — focus on a particular capability or attribute\n"
        "  use_case         — real-world application or scenario\n"
        "  trust_signals    — reviews, case studies, credibility, proof points\n"
        "  how_to           — procedural or instructional angle\n"
        "  definitional     — conceptual or 'what is' angle\n\n"
        "CONSTRAINTS\n"
        "- Produce at least 2 sub-queries per type (12 minimum from types alone).\n"
        "- Return ONLY a JSON object. No markdown, no prose, no code fences.\n\n"
        "SCHEMA\n"
        '{"sub_queries": [{"type": "<one of the six types>", "query": "<sub-query string>"}]}'
    ),
    user_template="Target query: {target_query}",
)


# ---------------------------------------------------------------------------
# V3 — Structured system role + inline few-shot example + strict constraints
#       This is the DEFAULT production prompt.
# ---------------------------------------------------------------------------

_V3_SYSTEM = """\
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
  trust_signals    — reviews, testimonials, case studies, awards, proof points
  how_to           — a procedural, instructional, or "how do I" angle
  definitional     — a conceptual, explanatory, or "what is" angle

HARD CONSTRAINTS
1. Generate at least 2 sub-queries for EVERY type (12 minimum across all six types).
2. Total sub-queries must be between 10 and 15 inclusive.
3. Each sub-query must be a realistic search query string — not a sentence fragment.
4. Return ONLY a valid JSON object. No markdown fences. No prose. No extra fields.
5. The JSON must match this schema exactly:

{
  "sub_queries": [
    {"type": "<one of the six types above>", "query": "<search query string>"},
    ...
  ]
}

EXAMPLE — for target query "best project management software for remote teams":
{
  "sub_queries": [
    {"type": "comparative", "query": "Asana vs Monday.com vs Notion for remote teams"},
    {"type": "comparative", "query": "Jira vs ClickUp for distributed engineering teams"},
    {"type": "feature_specific", "query": "project management tool with async video updates"},
    {"type": "feature_specific", "query": "PM software with time zone management features"},
    {"type": "use_case", "query": "project management software for remote marketing agencies"},
    {"type": "use_case", "query": "best PM tool for fully distributed startup teams"},
    {"type": "trust_signals", "query": "project management software reviews from remote-first companies"},
    {"type": "trust_signals", "query": "Monday.com vs Asana case studies remote work 2025"},
    {"type": "how_to", "query": "how to manage remote team projects with no meetings"},
    {"type": "how_to", "query": "how to track distributed team progress in real time"},
    {"type": "definitional", "query": "what is asynchronous project management"},
    {"type": "definitional", "query": "definition of remote-first project workflow"}
  ]
}

Now generate sub-queries for the target query provided by the user.\
"""

PROMPT_V3 = PromptConfig(
    version_id="v3_structured",
    description=(
        "Structured system role with inline few-shot example. "
        "Explicit schema, hard constraints numbered, domain-agnostic wording. "
        "Current production default."
    ),
    system_message=_V3_SYSTEM,
    user_template="Target query: {target_query}",
)


# ---------------------------------------------------------------------------
# Default — what the API uses
# ---------------------------------------------------------------------------

DEFAULT_PROMPT: PromptConfig = PROMPT_V3

#: All versions in order — used by optimization/prompt_tuning/run_prompt_eval.py
ALL_PROMPTS: list[PromptConfig] = [PROMPT_V1, PROMPT_V2, PROMPT_V3]
