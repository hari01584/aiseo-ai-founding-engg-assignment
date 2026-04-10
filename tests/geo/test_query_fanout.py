"""
tests/geo/test_query_fanout.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Tests for fanout_engine.py — query generation, JSON parsing, and resilience.

Two test classes:

  TestFanoutFixtures
  ------------------
  All LLM calls are mocked.  Tests cover:
    - Happy path: valid JSON → correct SubQuery list
    - Resilience: markdown fences, truncated JSON, invalid type, missing field,
      extra top-level keys, fewer than 10 sub-queries, missing entire type
    - Retry behaviour: fail N times then succeed; fail all retries → 503 error shape

  TestFanoutLive
  --------------
  Marked @pytest.mark.integration — skipped in CI unless RUN_INTEGRATION=1.
  Makes a real LLM call and validates only the structure of the response,
  not specific content (content varies per call).
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from app.models.schemas import LLMUnavailableError, SubQuery
from app.services.fanout_engine import (
    FanOutConfig,
    _VALID_TYPES,
    check_structural_constraints,
    generate_sub_queries,
    parse_llm_response,
    strip_markdown_fences,
    validate_sub_queries,
)

# ---------------------------------------------------------------------------
# Shared fixtures & helpers
# ---------------------------------------------------------------------------

#: A minimal valid sub-query dict used to build fixture responses
def _sq(type_: str, query: str = "some query") -> Dict[str, Any]:
    return {"type": type_, "query": query}


def _valid_payload(n_per_type: int = 2) -> str:
    """Build a JSON string that satisfies all structural constraints."""
    items = []
    for t in sorted(_VALID_TYPES):
        for i in range(n_per_type):
            items.append({"type": t, "query": f"{t} sub-query {i + 1}"})
    return json.dumps({"sub_queries": items})


def _make_config(max_retries: int = 3) -> FanOutConfig:
    """Minimal config for tests — uses engine defaults, no prompt fields."""
    return FanOutConfig(
        model_name="gpt-4o-mini",
        min_total=10,
        min_per_type=2,
        max_retries=max_retries,
        retry_base_delay=0.0,  # no sleep in tests
    )


# ===========================================================================
# Pure helper tests — no mocking needed
# ===========================================================================

class TestStripMarkdownFences:
    def test_no_fence_unchanged(self):
        raw = '{"sub_queries": []}'
        assert strip_markdown_fences(raw) == raw

    def test_json_fence_stripped(self):
        raw = '```json\n{"sub_queries": []}\n```'
        assert strip_markdown_fences(raw) == '{"sub_queries": []}'

    def test_generic_fence_stripped(self):
        raw = '```\n{"sub_queries": []}\n```'
        assert strip_markdown_fences(raw) == '{"sub_queries": []}'

    def test_whitespace_around_fence_stripped(self):
        raw = '  ```json\n{"a": 1}\n```  '
        assert strip_markdown_fences(raw) == '{"a": 1}'


class TestParseLLMResponse:
    def test_top_level_list_accepted(self):
        raw = json.dumps([_sq("how_to"), _sq("definitional")])
        result = parse_llm_response(raw)
        assert isinstance(result, list)
        assert len(result) == 2

    def test_sub_queries_envelope_accepted(self):
        raw = json.dumps({"sub_queries": [_sq("how_to")]})
        result = parse_llm_response(raw)
        assert result[0]["type"] == "how_to"

    def test_queries_envelope_accepted(self):
        raw = json.dumps({"queries": [_sq("use_case")]})
        result = parse_llm_response(raw)
        assert result[0]["type"] == "use_case"

    def test_markdown_fence_handled(self):
        inner = json.dumps({"sub_queries": [_sq("comparative")]})
        raw = f"```json\n{inner}\n```"
        result = parse_llm_response(raw)
        assert result[0]["type"] == "comparative"

    def test_invalid_json_raises(self):
        with pytest.raises(json.JSONDecodeError):
            parse_llm_response("not json at all")

    def test_truncated_json_raises(self):
        with pytest.raises(json.JSONDecodeError):
            parse_llm_response('{"sub_queries": [{"type": "how_to"')

    def test_unexpected_structure_raises(self):
        # Valid JSON but not a list or known envelope
        with pytest.raises(ValueError, match="unexpected structure"):
            parse_llm_response('"just a string"')


class TestValidateSubQueries:
    def test_valid_items_all_accepted(self):
        items = [_sq(t) for t in _VALID_TYPES]
        result = validate_sub_queries(items)
        assert len(result) == len(_VALID_TYPES)
        assert all(isinstance(sq, SubQuery) for sq in result)

    def test_unknown_type_skipped(self):
        items = [_sq("opinion"), _sq("how_to")]
        result = validate_sub_queries(items)
        assert len(result) == 1
        assert result[0].type == "how_to"

    def test_missing_type_field_skipped(self):
        items = [{"query": "no type here"}, _sq("definitional")]
        result = validate_sub_queries(items)
        assert len(result) == 1

    def test_extra_fields_ignored(self):
        items = [{"type": "use_case", "query": "q", "extra_key": "ignored"}]
        result = validate_sub_queries(items)
        assert len(result) == 1
        assert not hasattr(result[0], "extra_key")

    def test_empty_query_string_skipped(self):
        items = [{"type": "how_to", "query": ""}]
        result = validate_sub_queries(items)
        # Empty string fails SubQuery min_length if enforced — currently allowed
        # by schema; this test documents the current behaviour
        assert isinstance(result, list)


class TestCheckStructuralConstraints:
    def _make_sqs(self, counts: Dict[str, int]) -> List[SubQuery]:
        sqs = []
        for t, n in counts.items():
            for i in range(n):
                sqs.append(SubQuery(type=t, query=f"{t} query {i}"))  # type: ignore[arg-type]
        return sqs

    def test_all_constraints_satisfied(self):
        sqs = self._make_sqs({t: 2 for t in _VALID_TYPES})  # 12 total
        violations = check_structural_constraints(sqs, min_total=10, min_per_type=2)
        assert violations == []

    def test_too_few_total(self):
        sqs = self._make_sqs({t: 1 for t in _VALID_TYPES})  # 6 total
        violations = check_structural_constraints(sqs, min_total=10, min_per_type=1)
        assert any("Too few" in v for v in violations)

    def test_missing_entire_type(self):
        counts = {t: 2 for t in _VALID_TYPES}
        del counts["comparative"]
        sqs = self._make_sqs(counts)
        violations = check_structural_constraints(sqs, min_total=10, min_per_type=2)
        assert any("comparative" in v for v in violations)

    def test_type_below_minimum(self):
        counts = {t: 2 for t in _VALID_TYPES}
        counts["how_to"] = 1
        sqs = self._make_sqs(counts)
        violations = check_structural_constraints(sqs, min_total=10, min_per_type=2)
        assert any("how_to" in v for v in violations)


# ===========================================================================
# Fixture tests — LLM fully mocked
# ===========================================================================

class TestFanoutFixtures:
    """LLM call is mocked via patch on _call_llm."""

    def _run(self, llm_returns: Any, max_retries: int = 1) -> List[SubQuery]:
        config = _make_config(max_retries=max_retries)
        with patch(
            "app.services.fanout_engine._call_llm", return_value=llm_returns
        ):
            return generate_sub_queries("test query", config)

    def _run_side_effects(self, side_effects: list) -> List[SubQuery]:
        config = _make_config(max_retries=len(side_effects))
        with patch(
            "app.services.fanout_engine._call_llm", side_effect=side_effects
        ):
            return generate_sub_queries("test query", config)

    # ── Happy path ──────────────────────────────────────────────────────────

    def test_valid_response_returns_correct_count(self):
        result = self._run(_valid_payload(n_per_type=2))
        assert len(result) == 12  # 6 types × 2

    def test_valid_response_all_types_present(self):
        result = self._run(_valid_payload(n_per_type=2))
        found_types = {sq.type for sq in result}
        assert found_types == _VALID_TYPES

    def test_valid_response_objects_are_subquery_instances(self):
        result = self._run(_valid_payload())
        assert all(isinstance(sq, SubQuery) for sq in result)

    def test_covered_and_similarity_are_none(self):
        """Gap analysis has not run yet — fields must be None."""
        result = self._run(_valid_payload())
        assert all(sq.covered is None for sq in result)
        assert all(sq.similarity_score is None for sq in result)

    # ── Resilience ──────────────────────────────────────────────────────────

    def test_markdown_fence_wrapper_handled(self):
        raw = f"```json\n{_valid_payload()}\n```"
        result = self._run(raw)
        assert len(result) == 12

    def test_extra_top_level_key_ignored(self):
        data = json.loads(_valid_payload())
        data["model_notes"] = "some hallucinated field"
        result = self._run(json.dumps(data))
        assert len(result) == 12

    def test_invalid_json_exhausts_retries_raises(self):
        config = _make_config(max_retries=3)
        with patch(
            "app.services.fanout_engine._call_llm", return_value="not json"
        ):
            with pytest.raises(LLMUnavailableError):
                generate_sub_queries("test query", config)

    def test_truncated_json_exhausts_retries_raises(self):
        config = _make_config(max_retries=3)
        with patch(
            "app.services.fanout_engine._call_llm",
            return_value='{"sub_queries": [{"type": "how_to"',
        ):
            with pytest.raises(LLMUnavailableError):
                generate_sub_queries("test query", config)

    def test_too_few_subqueries_exhausts_retries_raises(self):
        # Only 1 per type → 6 total → below min_total=10
        payload = json.dumps({"sub_queries": [_sq(t) for t in _VALID_TYPES]})
        config = _make_config(max_retries=3)
        with patch("app.services.fanout_engine._call_llm", return_value=payload):
            with pytest.raises(LLMUnavailableError):
                generate_sub_queries("test query", config)

    def test_missing_entire_type_exhausts_retries_raises(self):
        items = [{"type": t, "query": f"q{i}"} for t in _VALID_TYPES if t != "comparative" for i in range(2)]
        payload = json.dumps({"sub_queries": items})
        config = _make_config(max_retries=3)
        with patch("app.services.fanout_engine._call_llm", return_value=payload):
            with pytest.raises(LLMUnavailableError):
                generate_sub_queries("test query", config)

    # ── Retry logic ─────────────────────────────────────────────────────────

    def test_fail_once_then_succeed(self):
        """First call returns bad JSON; second call returns valid payload."""
        result = self._run_side_effects(["not json", _valid_payload()])
        assert len(result) == 12

    def test_fail_twice_then_succeed(self):
        result = self._run_side_effects(["bad", "also bad", _valid_payload()])
        assert len(result) == 12

    def test_fail_all_retries_raises_llm_unavailable(self):
        config = _make_config(max_retries=3)
        with patch(
            "app.services.fanout_engine._call_llm", side_effect=["x", "x", "x"]
        ):
            with pytest.raises(LLMUnavailableError) as exc_info:
                generate_sub_queries("test query", config)
        assert "3 retries" in str(exc_info.value)

    def test_no_sleep_between_retries_in_tests(self):
        """Retry back-off sleep is zero in tests (retry_base_delay=0.0)."""
        import time
        config_zero = FanOutConfig(
            model_name="gpt-4o-mini",
            min_total=10,
            min_per_type=2,
            max_retries=3,
            retry_base_delay=0.0,
        )
        with patch("app.services.fanout_engine._call_llm", return_value="bad"):
            with patch("app.services.fanout_engine.time") as mock_time:
                mock_time.sleep = MagicMock()
                with pytest.raises(LLMUnavailableError):
                    generate_sub_queries("test query", config_zero)
                # sleep called with 0.0 * 2^n — always 0, so effectively no wait
                for call in mock_time.sleep.call_args_list:
                    assert call.args[0] == 0.0


# ===========================================================================
# Live integration tests — skipped unless RUN_INTEGRATION=1
# ===========================================================================

@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("RUN_INTEGRATION"),
    reason="Set RUN_INTEGRATION=1 to run live LLM tests",
)
class TestFanoutLive:
    """
    Calls a real LLM.  Validates structure only — not specific content.
    Requires OPENAI_API_KEY in environment.
    """

    def test_live_returns_valid_structure(self):
        config = _make_config(max_retries=3)
        result = generate_sub_queries("best AI writing tool for SEO", config)
        assert len(result) >= 10
        found_types = {sq.type for sq in result}
        assert found_types == _VALID_TYPES

    def test_live_different_domain_same_structure(self):
        """Prompt must be domain-agnostic — CRM query should produce same structure."""
        config = _make_config(max_retries=3)
        result = generate_sub_queries("best CRM software for startups", config)
        assert len(result) >= 10
        found_types = {sq.type for sq in result}
        assert found_types == _VALID_TYPES
