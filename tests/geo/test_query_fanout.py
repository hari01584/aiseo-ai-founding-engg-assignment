"""
tests/geo/test_query_fanout.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Tests for generate_sub_queries() in fanout_engine.py.

All LLM calls are mocked via _call_llm so no API key is needed.

  TestGenerateSubQueries  — one happy path + one test per failure mode
  TestFanoutLive          — real LLM call, skipped unless RUN_INTEGRATION=1
"""
from __future__ import annotations

import json
import os
from unittest.mock import patch

import pytest

from app.models.schemas import LLMUnavailableError, SubQuery
from app.services.fanout_engine import FanOutConfig, _VALID_TYPES, generate_sub_queries

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MOCK_TARGET = "best AI writing tool for SEO"

# Zero sleep so retry tests run instantly
_FAST_CONFIG = FanOutConfig(max_retries=3, retry_base_delay=0.0)


def _valid_llm_response(n_per_type: int = 2) -> str:
    """Build a JSON string that satisfies all structural constraints."""
    items = [
        {"type": t, "query": f"{t} query {i + 1}"}
        for t in sorted(_VALID_TYPES)
        for i in range(n_per_type)
    ]
    return json.dumps({"sub_queries": items})


def _mock_llm(return_value: str):
    """Patch _call_llm to always return the given string."""
    return patch("app.services.fanout_engine._call_llm", return_value=return_value)


def _mock_llm_seq(*responses: str):
    """Patch _call_llm to return responses in sequence (for retry tests)."""
    return patch("app.services.fanout_engine._call_llm", side_effect=list(responses))


# ===========================================================================
# Main test class
# ===========================================================================

class TestGenerateSubQueries:

    # ── Happy path ──────────────────────────────────────────────────────────

    def test_valid_response_returns_all_six_types(self):
        """Golden path: LLM returns clean JSON → all 6 types present."""
        with _mock_llm(_valid_llm_response()):
            result = generate_sub_queries(_MOCK_TARGET, _FAST_CONFIG)

        assert isinstance(result, list)
        assert all(isinstance(sq, SubQuery) for sq in result)
        assert {sq.type for sq in result} == _VALID_TYPES
        # gap analysis hasn't run yet — these fields must be unset
        assert all(sq.covered is None for sq in result)
        assert all(sq.similarity_score is None for sq in result)

    # ── Resilience: weird-but-parseable LLM outputs ─────────────────────────

    def test_markdown_fence_in_response_is_stripped(self):
        """LLM wraps the JSON in ```json ... ``` — should be handled gracefully."""
        wrapped = f"```json\n{_valid_llm_response()}\n```"
        with _mock_llm(wrapped):
            result = generate_sub_queries(_MOCK_TARGET, _FAST_CONFIG)
        assert {sq.type for sq in result} == _VALID_TYPES

    def test_extra_fields_in_response_are_ignored(self):
        """LLM adds hallucinated top-level keys — should not break parsing."""
        data = json.loads(_valid_llm_response())
        data["model_notes"] = "I added this myself"
        with _mock_llm(json.dumps(data)):
            result = generate_sub_queries(_MOCK_TARGET, _FAST_CONFIG)
        assert {sq.type for sq in result} == _VALID_TYPES

    def test_unknown_sub_query_types_are_dropped(self):
        """LLM invents a type like 'opinion' — those items are silently skipped."""
        data = json.loads(_valid_llm_response(n_per_type=3))  # 18 valid items
        data["sub_queries"].append({"type": "opinion", "query": "this is wrong"})
        with _mock_llm(json.dumps(data)):
            result = generate_sub_queries(_MOCK_TARGET, _FAST_CONFIG)
        assert all(sq.type in _VALID_TYPES for sq in result)

    # ── Failure modes: each exhausts all retries and raises ─────────────────

    def test_invalid_json_raises_llm_unavailable(self):
        """LLM returns plain prose instead of JSON every attempt."""
        with _mock_llm("Sorry, I cannot help with that."):
            with pytest.raises(LLMUnavailableError):
                generate_sub_queries(_MOCK_TARGET, _FAST_CONFIG)

    def test_truncated_json_raises_llm_unavailable(self):
        """LLM response gets cut off mid-JSON every attempt."""
        with _mock_llm('{"sub_queries": [{"type": "how_to"'):
            with pytest.raises(LLMUnavailableError):
                generate_sub_queries(_MOCK_TARGET, _FAST_CONFIG)

    def test_too_few_sub_queries_raises_llm_unavailable(self):
        """LLM returns only 1 per type (6 total) — below the 10-query minimum."""
        too_few = json.dumps({"sub_queries": [{"type": t, "query": "q"} for t in _VALID_TYPES]})
        with _mock_llm(too_few):
            with pytest.raises(LLMUnavailableError):
                generate_sub_queries(_MOCK_TARGET, _FAST_CONFIG)

    def test_missing_type_raises_llm_unavailable(self):
        """LLM omits 'comparative' entirely every attempt."""
        items = [
            {"type": t, "query": f"q{i}"}
            for t in _VALID_TYPES if t != "comparative"
            for i in range(2)
        ]
        with _mock_llm(json.dumps({"sub_queries": items})):
            with pytest.raises(LLMUnavailableError):
                generate_sub_queries(_MOCK_TARGET, _FAST_CONFIG)

    # ── Retry behaviour ─────────────────────────────────────────────────────

    def test_recovers_after_one_bad_attempt(self):
        """First call returns garbage; second call returns valid JSON."""
        with _mock_llm_seq("not json", _valid_llm_response()):
            result = generate_sub_queries(_MOCK_TARGET, _FAST_CONFIG)
        assert {sq.type for sq in result} == _VALID_TYPES

    def test_recovers_after_two_bad_attempts(self):
        """Two bad calls then one good call — still succeeds within max_retries=3."""
        with _mock_llm_seq("bad", "also bad", _valid_llm_response()):
            result = generate_sub_queries(_MOCK_TARGET, _FAST_CONFIG)
        assert {sq.type for sq in result} == _VALID_TYPES

    def test_error_message_includes_retry_count(self):
        """LLMUnavailableError message should tell the caller how many retries ran."""
        with _mock_llm("bad json"):
            with pytest.raises(LLMUnavailableError, match="3 retries"):
                generate_sub_queries(_MOCK_TARGET, _FAST_CONFIG)


# ===========================================================================
# Live integration tests — skipped unless RUN_INTEGRATION=1
# ===========================================================================

@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="Set OPENAI_API_KEY key to run live LLM tests",
)
class TestFanoutLive:
    """
    Makes a real OpenAI call. Validates structure only — not specific content.
    Requires OPENAI_API_KEY in the environment.
    """

    def test_seo_query_returns_all_six_types(self):
        result = generate_sub_queries(_MOCK_TARGET, FanOutConfig())
        assert len(result) >= 10
        assert {sq.type for sq in result} == _VALID_TYPES

    def test_different_domain_returns_same_structure(self):
        """Prompt must be domain-agnostic — CRM query should produce identical structure."""
        result = generate_sub_queries("best CRM software for startups", FanOutConfig())
        assert len(result) >= 10
        assert {sq.type for sq in result} == _VALID_TYPES
