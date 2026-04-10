"""
tests/geo/test_api_fanout.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
API-level tests for POST /api/fanout/generate.

All LLM calls and (where needed) gap-analysis calls are mocked — no API key
or embedding model is required.

Five test groups:

  Case 1 — No existing_content : verifies sub-queries are returned with no
            gap_summary, covered/similarity_score fields absent (None).

  Case 2 — With existing_content : mocks analyse_gaps to inject a realistic
            GapSummary and verifies the response envelope includes gap_summary
            with the correct structure.

  Case 3 — LLM failure (503) : every retry returns bad JSON →
            generate_sub_queries raises LLMUnavailableError → API returns 503
            with the correct error shape.

  Case 4 — Gap analysis failure (non-fatal) : sub-queries succeed but
            analyse_gaps raises → API still returns 200 with sub-queries and
            gap_summary=None.

  Case 5 — Validation errors : blank target_query and missing fields → 422.
"""
from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models.schemas import GapSummary, LLMUnavailableError, SubQuery
from app.services.fanout_engine import _VALID_TYPES

client = TestClient(app, raise_server_exceptions=False)

_ENDPOINT = "/api/fanout/generate"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MOCK_TARGET = "best AI writing tool for SEO"

_EXISTING_CONTENT = (
    "AI writing tools help content teams produce SEO-optimised articles faster. "
    "They use machine learning to suggest keywords, improve readability, and "
    "structure content for featured snippets and AI overviews. "
    "Teams at agencies and in-house marketing departments use these tools daily "
    "to reduce production time while maintaining quality."
)


def _valid_llm_response(n_per_type: int = 2) -> str:
    """Return a JSON string satisfying all structural constraints."""
    items = [
        {"type": t, "query": f"{t} query {i + 1}"}
        for t in sorted(_VALID_TYPES)
        for i in range(n_per_type)
    ]
    return json.dumps({"sub_queries": items})


def _mock_llm(return_value: str):
    """Patch _call_llm in fanout_engine to always return the given string."""
    return patch("app.services.fanout_engine._call_llm", return_value=return_value)


def _sub_queries_with_coverage() -> list[SubQuery]:
    """Sub-queries with gap-analysis annotations already applied."""
    covered_types = ["comparative", "feature_specific", "use_case", "how_to"]
    missing_types = ["trust_signals", "definitional"]
    items = []
    for t in sorted(_VALID_TYPES):
        for i in range(2):
            items.append(
                SubQuery(
                    type=t,  # type: ignore[arg-type]
                    query=f"{t} query {i + 1}",
                    covered=(t in covered_types),
                    similarity_score=0.80 if t in covered_types else 0.45,
                )
            )
    return items


def _mock_gap_summary() -> GapSummary:
    return GapSummary(
        covered=8,
        total=12,
        coverage_percent=66.7,
        covered_types=["comparative", "feature_specific", "use_case", "how_to"],
        missing_types=["trust_signals", "definitional"],
    )


# ---------------------------------------------------------------------------
# Shape assertion helpers
# ---------------------------------------------------------------------------

_VALID_TYPES_SET = set(_VALID_TYPES)


def _assert_sub_query(sq: dict, *, expect_gap_fields: bool) -> None:
    assert "type" in sq,  f"sub_query missing 'type': {sq}"
    assert "query" in sq, f"sub_query missing 'query': {sq}"
    assert sq["type"] in _VALID_TYPES_SET, f"Unknown type: {sq['type']!r}"
    assert isinstance(sq["query"], str) and sq["query"].strip(), "query must be non-empty string"

    if expect_gap_fields:
        assert "covered" in sq,          f"sub_query missing 'covered': {sq}"
        assert "similarity_score" in sq, f"sub_query missing 'similarity_score': {sq}"
        assert isinstance(sq["covered"], bool), "'covered' must be bool"
        assert isinstance(sq["similarity_score"], float), "'similarity_score' must be float"
    else:
        # When no existing_content the fields should be absent or None
        assert sq.get("covered") is None,          "'covered' must be None when no content"
        assert sq.get("similarity_score") is None, "'similarity_score' must be None when no content"


def _assert_envelope(data: dict, *, expect_gap_summary: bool) -> None:
    """Assert the top-level FanOutResponse shape."""
    for field in ("target_query", "model_used", "total_sub_queries", "sub_queries"):
        assert field in data, f"Missing field '{field}' in response"

    assert isinstance(data["target_query"], str)
    assert isinstance(data["model_used"], str)
    assert isinstance(data["total_sub_queries"], int)
    assert data["total_sub_queries"] >= 10, (
        f"total_sub_queries {data['total_sub_queries']} below minimum 10"
    )

    sub_queries = data["sub_queries"]
    assert isinstance(sub_queries, list)
    assert len(sub_queries) == data["total_sub_queries"]

    returned_types = {sq["type"] for sq in sub_queries}
    assert returned_types == _VALID_TYPES_SET, (
        f"Expected types {_VALID_TYPES_SET}, got {returned_types}"
    )

    for sq in sub_queries:
        _assert_sub_query(sq, expect_gap_fields=expect_gap_summary)

    if expect_gap_summary:
        gs = data.get("gap_summary")
        assert gs is not None, "gap_summary must be present when existing_content supplied"
        for field in ("covered", "total", "coverage_percent", "covered_types", "missing_types"):
            assert field in gs, f"gap_summary missing field '{field}'"
        assert isinstance(gs["covered"], int)
        assert isinstance(gs["total"], int)
        assert isinstance(gs["coverage_percent"], float)
        assert isinstance(gs["covered_types"], list)
        assert isinstance(gs["missing_types"], list)
        # covered is a sub-query count (not a type count); must be in [0, total]
        assert 0 <= gs["covered"] <= gs["total"]
        assert 0.0 <= gs["coverage_percent"] <= 100.0
    else:
        assert data.get("gap_summary") is None, (
            "gap_summary must be None when no existing_content"
        )


# ===========================================================================
# Case 1 — No existing_content: sub-queries only, no gap analysis
# ===========================================================================

def test_no_content_returns_sub_queries_only():
    """
    When existing_content is omitted the API must return:
      - HTTP 200
      - All 6 sub-query types present
      - gap_summary is None / absent
      - covered and similarity_score are None on every sub-query
    """
    with _mock_llm(_valid_llm_response()):
        resp = client.post(_ENDPOINT, json={"target_query": _MOCK_TARGET})

    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    data = resp.json()
    _assert_envelope(data, expect_gap_summary=False)
    assert data["target_query"] == _MOCK_TARGET


def test_no_content_reflects_correct_query():
    """target_query is echoed back verbatim in the response."""
    query = "best CRM software for startups"
    with _mock_llm(_valid_llm_response()):
        resp = client.post(_ENDPOINT, json={"target_query": query})

    assert resp.status_code == 200
    assert resp.json()["target_query"] == query


# ===========================================================================
# Case 2 — With existing_content: gap analysis is run and returned
# ===========================================================================

def test_with_content_returns_gap_summary():
    """
    When existing_content is provided the API must:
      - call analyse_gaps (mocked here)
      - return HTTP 200 with gap_summary populated
      - annotate every sub-query with covered + similarity_score
    """
    annotated = _sub_queries_with_coverage()
    gs = _mock_gap_summary()

    with _mock_llm(_valid_llm_response()), \
         patch("app.api.fanout.analyse_gaps", return_value=(annotated, gs)):
        resp = client.post(_ENDPOINT, json={
            "target_query": _MOCK_TARGET,
            "existing_content": _EXISTING_CONTENT,
        })

    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    data = resp.json()
    _assert_envelope(data, expect_gap_summary=True)


def test_gap_summary_covered_types_are_valid():
    """covered_types and missing_types must only contain known sub-query types."""
    annotated = _sub_queries_with_coverage()
    gs = _mock_gap_summary()

    with _mock_llm(_valid_llm_response()), \
         patch("app.api.fanout.analyse_gaps", return_value=(annotated, gs)):
        resp = client.post(_ENDPOINT, json={
            "target_query": _MOCK_TARGET,
            "existing_content": _EXISTING_CONTENT,
        })

    assert resp.status_code == 200
    gap = resp.json()["gap_summary"]
    for t in gap["covered_types"]:
        assert t in _VALID_TYPES_SET, f"Unknown covered_type: {t!r}"
    for t in gap["missing_types"]:
        assert t in _VALID_TYPES_SET, f"Unknown missing_type: {t!r}"


def test_gap_summary_totals_are_consistent():
    """covered + (sub-queries marked not covered) == total_sub_queries."""
    annotated = _sub_queries_with_coverage()
    gs = _mock_gap_summary()

    with _mock_llm(_valid_llm_response()), \
         patch("app.api.fanout.analyse_gaps", return_value=(annotated, gs)):
        resp = client.post(_ENDPOINT, json={
            "target_query": _MOCK_TARGET,
            "existing_content": _EXISTING_CONTENT,
        })

    assert resp.status_code == 200
    data = resp.json()
    manually_covered = sum(1 for sq in data["sub_queries"] if sq["covered"] is True)
    assert manually_covered == data["gap_summary"]["covered"]


# ===========================================================================
# Case 3 — LLM failure → 503 Service Unavailable
# ===========================================================================

def test_llm_failure_returns_503():
    """
    When the LLM never returns valid JSON the engine raises LLMUnavailableError.
    The API must catch this and return 503 with the correct error envelope.
    """
    with patch(
        "app.services.fanout_engine._call_llm",
        side_effect=["bad json", "still bad", "nope"],
    ):
        resp = client.post(_ENDPOINT, json={"target_query": _MOCK_TARGET})

    assert resp.status_code == 503, f"Expected 503, got {resp.status_code}: {resp.text}"
    data = resp.json()
    assert data["error"] == "llm_unavailable"
    assert "message" in data
    assert "detail" in data


def test_llm_failure_error_shape():
    """503 error body must match the ErrorResponse schema exactly."""
    with _mock_llm("not json at all"):
        resp = client.post(_ENDPOINT, json={"target_query": _MOCK_TARGET})

    assert resp.status_code == 503
    data = resp.json()
    # All three ErrorResponse fields must be present
    assert set(data.keys()) >= {"error", "message", "detail"}
    assert isinstance(data["error"],   str)
    assert isinstance(data["message"], str)
    # detail may be None or a string
    assert data["detail"] is None or isinstance(data["detail"], str)


# ===========================================================================
# Case 4 — Gap analysis failure is non-fatal
# ===========================================================================

def test_gap_analysis_error_still_returns_200_without_gap_summary():
    """
    If analyse_gaps raises an exception, the API must:
      - NOT return 5xx
      - Return HTTP 200 with sub-queries intact
      - Return gap_summary=None (non-fatal degradation)
    """
    with _mock_llm(_valid_llm_response()), \
         patch("app.api.fanout.analyse_gaps", side_effect=RuntimeError("model load failed")):
        resp = client.post(_ENDPOINT, json={
            "target_query": _MOCK_TARGET,
            "existing_content": _EXISTING_CONTENT,
        })

    assert resp.status_code == 200, f"Expected 200 on gap failure, got {resp.status_code}"
    data = resp.json()
    assert data["gap_summary"] is None, "gap_summary should be None when gap analysis fails"
    # Sub-queries are still returned
    assert len(data["sub_queries"]) >= 10


def test_gap_analysis_error_sub_queries_have_no_coverage_fields():
    """After a gap analysis failure, covered/similarity_score must remain None."""
    with _mock_llm(_valid_llm_response()), \
         patch("app.api.fanout.analyse_gaps", side_effect=ValueError("embedding error")):
        resp = client.post(_ENDPOINT, json={
            "target_query": _MOCK_TARGET,
            "existing_content": _EXISTING_CONTENT,
        })

    assert resp.status_code == 200
    for sq in resp.json()["sub_queries"]:
        assert sq.get("covered") is None
        assert sq.get("similarity_score") is None


# ===========================================================================
# Case 5 — Validation errors → 422 Unprocessable Entity
# ===========================================================================

def test_blank_target_query_returns_422():
    """Pydantic rejects a blank target_query with HTTP 422."""
    resp = client.post(_ENDPOINT, json={"target_query": "   "})
    assert resp.status_code == 422, (
        f"Expected 422 for blank target_query, got {resp.status_code}"
    )


def test_missing_target_query_returns_422():
    """target_query is required — omitting it must return 422."""
    resp = client.post(_ENDPOINT, json={})
    assert resp.status_code == 422, (
        f"Expected 422 for missing target_query, got {resp.status_code}"
    )


def test_empty_string_target_query_returns_422():
    """An empty string (length 0) is rejected by the min_length=1 constraint."""
    resp = client.post(_ENDPOINT, json={"target_query": ""})
    assert resp.status_code == 422, (
        f"Expected 422 for empty target_query, got {resp.status_code}"
    )


def test_whitespace_only_content_is_treated_as_no_content():
    """
    existing_content of pure whitespace must skip gap analysis —
    the API should return 200 with gap_summary=None (same as omitting the field).
    """
    with _mock_llm(_valid_llm_response()):
        resp = client.post(_ENDPOINT, json={
            "target_query": _MOCK_TARGET,
            "existing_content": "    \n  ",
        })

    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    assert resp.json()["gap_summary"] is None
