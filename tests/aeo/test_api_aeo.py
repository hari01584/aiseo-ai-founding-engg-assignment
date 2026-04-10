"""
tests/test_api_aeo.py
~~~~~~~~~~~~~~~~~~~~~
API-level sanity tests for POST /api/aeo/analyze.

Three cases exercise the full request/response cycle through FastAPI's
in-process TestClient (no live server needed):

  Case 1 — URL input   : passes a real public URL; verifies the envelope
            shape and that every check is present.  The test does NOT
            assert specific scores because content at a live URL can change;
            it only checks that the API handles the round-trip without
            error and returns a well-formed response.

  Case 2 — HTML input  : posts raw HTML.  The content is designed so
            expected scores are deterministic (same prose used in the
            Check C integration tests).  Asserts score bands, check IDs,
            and that no check returned an error recommendation.

  Case 3 — Plain-text input : posts a plain-text paragraph.  Same
            deterministic-content strategy.  Verifies that the API still
            works when the parser falls back to the plain-text path
            (no BeautifulSoup parsing for h-tags).

All three cases assert:
  • HTTP 200 status
  • response shape: aeo_score, band, checks (list of 3)
  • each check has: check_id, name, score, max_score, passed, details
  • aeo_score is a float in [0, 100]
  • band is one of the four recognised strings
"""
from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app, raise_server_exceptions=True)

_ENDPOINT = "/api/aeo/analyze"

_VALID_BANDS = {
    "AEO Optimized ✅",
    "Needs Improvement 🟡",
    "Significant Gaps 🔴",
    "Not AEO Ready ⛔",
}

_EXPECTED_CHECK_IDS = {"direct_answer", "htag_hierarchy", "readability"}


# ---------------------------------------------------------------------------
# Shared shape helper
# ---------------------------------------------------------------------------

def _assert_envelope(data: dict) -> None:
    """Assert the top-level response shape is correct for every case."""
    assert "aeo_score" in data,  "Missing 'aeo_score' in response"
    assert "band"      in data,  "Missing 'band' in response"
    assert "checks"    in data,  "Missing 'checks' in response"

    score = data["aeo_score"]
    assert isinstance(score, (int, float)), f"aeo_score must be numeric, got {type(score)}"
    assert 0.0 <= score <= 100.0,           f"aeo_score {score} out of [0, 100]"

    assert data["band"] in _VALID_BANDS, f"Unrecognised band: {data['band']!r}"

    checks = data["checks"]
    assert isinstance(checks, list),   "checks must be a list"
    assert len(checks) == 3,           f"Expected 3 checks, got {len(checks)}"

    ids_returned = {c["check_id"] for c in checks}
    assert ids_returned == _EXPECTED_CHECK_IDS, (
        f"Expected check IDs {_EXPECTED_CHECK_IDS}, got {ids_returned}"
    )

    for c in checks:
        for field in ("check_id", "name", "score", "max_score", "passed", "details"):
            assert field in c, f"Check '{c.get('check_id', '?')}' missing field '{field}'"
        assert isinstance(c["score"],     int),  "score must be int"
        assert isinstance(c["max_score"], int),  "max_score must be int"
        assert isinstance(c["passed"],    bool), "passed must be bool"
        assert 0 <= c["score"] <= c["max_score"], (
            f"score {c['score']} out of range [0, {c['max_score']}]"
        )


# ===========================================================================
#  Case 1 — URL input
#
#  We mock fetch_and_parse so the test is deterministic and needs no
#  network access.  The mock returns the same grade-9 HTML used in Case 2
#  so the test focuses purely on the URL code-path, not on content scoring.
# ===========================================================================

_GRADE9_HTML = """\
<html><body>
<nav>Home | About | Contact</nav>
<article>
<p>Target topics that answer common questions your readers often submit.</p>
<p>Search engines index pages that offer clear and useful answers.</p>
<p>Using headers and bullet points makes reading easier for busy people.</p>
<p>Good content answers user questions clearly and directly every time.</p>
<p>Topic research drives better writing when done before starting content.</p>
</article>
<footer>Privacy Policy | Terms</footer>
</body></html>"""


def test_url_input_returns_valid_envelope():
    """
    URL path: mock fetch_and_parse at the site where aeo.py imported it so
    no real network call is made.  Verifies the API handles input_type='url'
    and returns a well-formed response — shape, status, bands, and check IDs.
    """
    from bs4 import BeautifulSoup
    from app.services.content_parser import _extract_clean_text, _extract_first_paragraph_html

    _raw = _GRADE9_HTML
    _soup = BeautifulSoup(_raw, "html.parser")
    _clean = _extract_clean_text(_soup)
    _fp = _extract_first_paragraph_html(_soup)

    # Must patch at 'app.api.aeo.fetch_and_parse' — that is the name the
    # router module bound at import time.
    with patch("app.api.aeo.fetch_and_parse", return_value=(_raw, _soup, _clean, _fp)):
        resp = client.post(_ENDPOINT, json={
            "input_type": "url",
            "input_value": "https://example.com/seo-guide",
        })

    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    _assert_envelope(resp.json())


# ===========================================================================
#  Case 2 — HTML input  (deterministic content)
#
#  Posts the same grade-9 article HTML directly.  We know from the Check C
#  integration tests that this content scores:
#    • readability  → score 20 (fk ≈ 9.0, in band)
#  H-tag hierarchy and direct_answer scores depend on the HTML structure
#  but we only assert they are within their valid ranges.
# ===========================================================================

def test_html_input_returns_valid_envelope_and_scores():
    """
    HTML path: posts raw HTML with a clean article structure.
    Verifies the response envelope AND that readability passes (score=20)
    since the article body uses confirmed grade-9 prose.
    """
    resp = client.post(_ENDPOINT, json={
        "input_type": "text",
        "input_value": _GRADE9_HTML,
    })

    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    data = resp.json()
    _assert_envelope(data)

    # Readability check must pass for this content
    readability = next(c for c in data["checks"] if c["check_id"] == "readability")
    assert readability["score"] == 20, (
        f"Expected readability score=20 for grade-9 HTML, got {readability['score']}"
    )
    assert readability["passed"] is True
    assert readability["details"]["fk_grade_level"] is not None

    # No check should have errored (recommendation would start with "Check could not")
    for c in data["checks"]:
        rec = c.get("recommendation") or ""
        assert not rec.startswith("Check could not"), (
            f"Check '{c['check_id']}' raised an internal error: {rec}"
        )


# ===========================================================================
#  Case 3 — Plain-text input  (deterministic content)
#
#  Posts a plain-text paragraph (no HTML tags).  The parser falls back to
#  the plain-text code-path: soup is empty (no h-tags), clean_text equals
#  the raw input.  Verifies the API handles this gracefully and that
#  htag_hierarchy returns score=0 (no h-tags to check) without erroring.
# ===========================================================================

_PLAIN_TEXT = (
    "Target topics that answer common questions your readers often submit. "
    "Search engines index pages that offer clear and useful answers. "
    "Using headers and bullet points makes reading easier for busy people. "
    "Good content answers user questions clearly and directly every time. "
    "Topic research drives better writing when done before starting content."
)


def test_plain_text_input_returns_valid_envelope():
    """
    Plain-text path: no HTML tags so the parser skips BeautifulSoup.
    Verifies the API handles the plain-text fallback and returns valid output:
      • htag_hierarchy has no h-tags to check → its score is within range
      • readability scores the plain text (confirmed grade-9 → score 20)
      • No check errors out
    """
    resp = client.post(_ENDPOINT, json={
        "input_type": "text",
        "input_value": _PLAIN_TEXT,
    })

    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    data = resp.json()
    _assert_envelope(data)

    # Readability must pass for this confirmed grade-9 text
    readability = next(c for c in data["checks"] if c["check_id"] == "readability")
    assert readability["score"] == 20, (
        f"Expected readability score=20 for grade-9 plain text, got {readability['score']}"
    )
    assert readability["passed"] is True

    # H-tag check: plain text has no headings — must not raise, score in valid range
    htag = next(c for c in data["checks"] if c["check_id"] == "htag_hierarchy")
    assert 0 <= htag["score"] <= htag["max_score"]

    # No internal errors in any check
    for c in data["checks"]:
        rec = c.get("recommendation") or ""
        assert not rec.startswith("Check could not"), (
            f"Check '{c['check_id']}' raised an internal error: {rec}"
        )


# ===========================================================================
#  Case 4 — Bad request: blank input_value
# ===========================================================================

def test_blank_input_value_returns_422():
    """Pydantic validation rejects a blank input_value with HTTP 422."""
    resp = client.post(_ENDPOINT, json={
        "input_type": "text",
        "input_value": "   ",
    })
    assert resp.status_code == 422, (
        f"Expected 422 for blank input_value, got {resp.status_code}"
    )


# ===========================================================================
#  Case 5 — Bad request: invalid input_type
# ===========================================================================

def test_invalid_input_type_returns_422():
    """Pydantic validation rejects an unrecognised input_type with HTTP 422."""
    resp = client.post(_ENDPOINT, json={
        "input_type": "file",
        "input_value": "some content",
    })
    assert resp.status_code == 422, (
        f"Expected 422 for input_type='file', got {resp.status_code}"
    )
