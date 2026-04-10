"""
tests/aeo/check_b/test_htag_hierarchy.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Integration tests for HtagHierarchyCheck.run().

Two parametrized tables exercise the full pipeline end-to-end:

  SOUP_CASES  – HTML snippets parsed directly into BeautifulSoup.
                Covers every rule (exactly-one-H1, no-skip, no-pre-H1),
                every score band (20 / 12 / 0), and every recommendation path.

  HTML_CASES  – Full HTML bodies fed through fetch_and_parse(), exactly as the
                live API does.  Cases mirror real article layouts: clean
                hierarchy, single skip, tag before H1, missing H1, 3+ violations.

Each row carries every expected output field so one parametrized test can
assert h_tags_found, violations count, violation substrings, score, passed,
and recommendation in a single shot.
"""
from __future__ import annotations

import pytest
from bs4 import BeautifulSoup

from app.services.aeo_checks.base import ParsedContent
from app.services.aeo_checks.htag_hierarchy import HtagHierarchyCheck
from app.services.content_parser import fetch_and_parse

_check = HtagHierarchyCheck()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_soup(html: str):
    """Build ParsedContent from a raw HTML snippet — bypasses fetch_and_parse."""
    soup = BeautifulSoup(html, "html.parser")
    content = ParsedContent(
        raw_html=html,
        soup=soup,
        clean_text=soup.get_text(),
        first_paragraph="",
    )
    return _check.run(content)


def _run_html(html: str):
    """Parse a full HTML body the same way the live API does."""
    raw, soup, clean, first_para = fetch_and_parse("text", html)
    content = ParsedContent(
        raw_html=raw,
        soup=soup,
        clean_text=clean,
        first_paragraph=first_para,
    )
    return _check.run(content)


# ---------------------------------------------------------------------------
# Schema for every row
#
#   id                – shown in pytest -v output
#   htags             – expected details.h_tags_found  (ordered list of tag names)
#   violation_count   – expected len(details.violations)
#   violation_phrases – list of substrings, every one must appear in at least
#                       one violation string (e.g. "h2" to confirm which level)
#   score             – expected result.score  (0 / 12 / 20)
#   passed            – expected result.passed (True only when score == 20)
#   rec_is_none       – True → recommendation must be None
#   rec_contains      – list of substrings every one of which must appear in rec
#   why               – one-line human note on what this case exercises
# ---------------------------------------------------------------------------


# ===========================================================================
#  SOUP CASES
#  HTML snippets where we control every heading precisely.
#  first_paragraph is left empty — check B only cares about soup.
# ===========================================================================

SOUP_CASES = [
    # ── score 20: zero violations ────────────────────────────────────────────
    dict(
        id="s01_perfect_h1_h2_h3",
        html=(
            "<h1>What Is the Capital of France?</h1>"
            "<h2>Geographic Overview</h2>"
            "<h3>The Seine Basin</h3>"
            "<h2>Political Role</h2>"
            "<h3>National Government</h3>"
            "<h3>International Diplomacy</h3>"
        ),
        # Clean H1→H2→H3 descent, multiple H2/H3 siblings — all valid
        htags=["h1", "h2", "h3", "h2", "h3", "h3"],
        violation_count=0, violation_phrases=[],
        score=20, passed=True,
        rec_is_none=True, rec_contains=[],
        why="Full H1→H2→H3 hierarchy, multiple siblings — zero violations.",
    ),
    dict(
        id="s02_perfect_h1_multiple_h2",
        html=(
            "<h1>Guide to French Cities</h1>"
            "<h2>Paris</h2>"
            "<h2>Lyon</h2>"
            "<h2>Marseille</h2>"
        ),
        # Flat H1+H2s, no depth required — valid
        htags=["h1", "h2", "h2", "h2"],
        violation_count=0, violation_phrases=[],
        score=20, passed=True,
        rec_is_none=True, rec_contains=[],
        why="Single H1 followed by multiple H2 siblings — flat but valid.",
    ),
    dict(
        id="s03_perfect_h1_only",
        html=(
            "<h1>Paris: Capital of France</h1>"
            "<p>Paris is the capital city of France, located in the north of the country "
            "along the Seine river, and has served as the country's political centre for "
            "over a thousand years.</p>"
        ),
        # No sub-headings at all — perfectly valid, no hierarchy to violate
        htags=["h1"],
        violation_count=0, violation_phrases=[],
        score=20, passed=True,
        rec_is_none=True, rec_contains=[],
        why="Only an H1, no sub-headings — nothing to violate.",
    ),
    dict(
        id="s04_perfect_deep_descent",
        html=(
            "<h1>France</h1>"
            "<h2>Regions</h2>"
            "<h3>Île-de-France</h3>"
            "<h4>Paris</h4>"
            "<h4>Versailles</h4>"
            "<h3>Provence</h3>"
            "<h4>Marseille</h4>"
        ),
        # H1→H2→H3→H4 depth used correctly, going back up is fine
        htags=["h1", "h2", "h3", "h4", "h4", "h3", "h4"],
        violation_count=0, violation_phrases=[],
        score=20, passed=True,
        rec_is_none=True, rec_contains=[],
        why="Deep H1→H2→H3→H4 nesting, including valid level-up — zero violations.",
    ),

    # ── score 12: exactly one violation ─────────────────────────────────────
    dict(
        id="s05_one_skip_h1_to_h3",
        html=(
            "<h1>Paris: Capital of France</h1>"
            "<h3>History of Governance</h3>"
            "<p>The city has been capital for over a thousand years of French history.</p>"
        ),
        # H1→H3 skips H2 — one violation
        htags=["h1", "h3"],
        violation_count=1, violation_phrases=["h2"],
        score=12, passed=False,
        rec_is_none=False, rec_contains=["H1", "H2", "H3"],
        why="H1 jumps straight to H3 — missing H2 is the single violation.",
    ),
    dict(
        id="s06_one_skip_h2_to_h4",
        html=(
            "<h1>Paris: Capital of France</h1>"
            "<h2>Political History</h2>"
            "<h4>The Capetian Dynasty</h4>"
            "<p>The Capetians established Paris as the royal seat.</p>"
        ),
        # H2→H4 skips H3 — one violation
        htags=["h1", "h2", "h4"],
        violation_count=1, violation_phrases=["h3"],
        score=12, passed=False,
        rec_is_none=False, rec_contains=["H1", "H2", "H3"],
        why="H2 jumps to H4 — missing H3 is the single violation.",
    ),
    dict(
        id="s07_one_h2_before_h1",
        html=(
            "<h2>Introduction to French Geography</h2>"
            "<h1>Paris: Capital of France</h1>"
            "<h2>Main Sections</h2>"
            "<h3>Political Role</h3>"
        ),
        # H2 appears before H1 — one pre-H1 violation
        htags=["h2", "h1", "h2", "h3"],
        violation_count=1, violation_phrases=["before"],
        score=12, passed=False,
        rec_is_none=False, rec_contains=["H1", "H2", "H3"],
        why="One H2 precedes the H1 — single pre-H1 violation.",
    ),

    # ── score 12: exactly two violations ────────────────────────────────────
    dict(
        id="s08_two_violations_pre_h1_plus_skip",
        html=(
            "<h2>Preamble: France Overview</h2>"
            "<h1>Paris: Capital of France</h1>"
            "<h3>Deep Sub-section on Culture</h3>"
            "<p>Content about Paris culture and arts.</p>"
        ),
        # H2 before H1 (1) + H1→H3 skip (1) = 2 violations → score 12
        htags=["h2", "h1", "h3"],
        violation_count=2, violation_phrases=["before", "h2"],
        score=12, passed=False,
        rec_is_none=False, rec_contains=["H1", "H2", "H3"],
        why="H2 before H1 plus H1→H3 skip — exactly two violations.",
    ),

    # ── score 0: missing H1 ─────────────────────────────────────────────────
    dict(
        id="s09_no_h1_only_h2_h3",
        html=(
            "<h2>About Paris</h2>"
            "<h3>Geography and Location</h3>"
            "<p>Paris is situated in northern France along the Seine river.</p>"
            "<h3>Political Significance</h3>"
            "<p>Paris houses the Élysée Palace and the National Assembly.</p>"
        ),
        # No H1 at all → validate_htags generates 4 violations (no-h1 + 3 pre-H1 tags)
        htags=["h2", "h3", "h3"],
        violation_count=4, violation_phrases=["h1"],
        score=0, passed=False,
        rec_is_none=False, rec_contains=["<h1>"],
        why="No H1 present — missing H1 triggers score-0 branch.",
    ),
    dict(
        id="s10_no_headings_at_all",
        html=(
            "<p>Paris is the capital of France, a status it has held for over a thousand "
            "years. The city serves as the seat of government and cultural centre of the "
            "French republic.</p>"
            "<p>It is located along the Seine river in northern France.</p>"
        ),
        # Zero heading tags — single "no headings found" violation, score 0 (no H1)
        htags=[],
        violation_count=1, violation_phrases=["no heading"],
        score=0, passed=False,
        rec_is_none=False, rec_contains=["<h1>"],
        why="No heading tags at all — score 0.",
    ),

    # ── score 0: 3+ violations ───────────────────────────────────────────────
    dict(
        id="s11_three_violations_chaos",
        html=(
            "<h2>Early Section on Paris Geography</h2>"
            "<h3>Even Earlier Sub-section on History</h3>"
            "<h1>Paris: Capital of France</h1>"
            "<h3>Culture and Arts — Skipping H2</h3>"
            "<p>The Louvre is the world's most-visited art museum.</p>"
        ),
        # H2 before H1 (1) + H3 before H1 (2) + H1→H3 skip (3) = 3 violations
        htags=["h2", "h3", "h1", "h3"],
        violation_count=3, violation_phrases=["before", "h2"],
        score=0, passed=False,
        rec_is_none=False, rec_contains=["multiple issues"],
        why="3 violations — hits the 3+ threshold, score 0.",
    ),
    dict(
        id="s12_two_h1_tags",
        html=(
            "<h1>Paris: Capital of France</h1>"
            "<h2>History</h2>"
            "<h1>Paris: Second Heading</h1>"
            "<h2>Culture</h2>"
        ),
        # Duplicate H1 (1 violation) only — still 1 violation → score 12
        htags=["h1", "h2", "h1", "h2"],
        violation_count=1, violation_phrases=["h1"],
        score=12, passed=False,
        rec_is_none=False, rec_contains=["H1", "H2", "H3"],
        why="Two H1 tags — exactly one duplicate violation, score 12.",
    ),
]


# ===========================================================================
#  HTML CASES
#  Full HTML bodies fed through fetch_and_parse(), exactly as the API does.
#  Tests that the parser + check pipeline works end-to-end with real markup.
# ===========================================================================

HTML_CASES = [
    # ── score 20: real article layout ───────────────────────────────────────
    dict(
        id="h01_perfect_article_layout",
        html="""<!DOCTYPE html>
<html lang="en">
<head><title>Capital of France</title></head>
<body>
  <header><nav>Home | About | Contact</nav></header>
  <main>
    <article>
      <h1>What Is the Capital of France?</h1>
      <p>Paris is the undisputed capital of France, serving as the seat of national
      government and the cultural heart of the country.</p>
      <h2>Geographic Overview</h2>
      <p>Paris lies in the Île-de-France region along the banks of the Seine river.</p>
      <h2>Political Role</h2>
      <h3>National Government</h3>
      <p>The Élysée Palace and the National Assembly are both located in Paris.</p>
      <h3>International Diplomacy</h3>
      <p>UNESCO, OECD, and numerous international bodies are headquartered in Paris.</p>
    </article>
  </main>
  <footer><p>© 2024 Example</p></footer>
</body>
</html>""",
        # nav/header/footer don't affect soup-level h-tag scan — all headings
        # inside <article> are collected correctly
        htags=["h1", "h2", "h2", "h3", "h3"],
        violation_count=0, violation_phrases=[],
        score=20, passed=True,
        rec_is_none=True, rec_contains=[],
        why="Full article layout — H1→H2→H3 used correctly throughout.",
    ),
    dict(
        id="h02_perfect_flat_with_sidebar",
        html="""<html><body>
<aside><p>Related: Lyon, Marseille</p></aside>
<main>
  <h1>Paris: Capital of France</h1>
  <h2>History</h2>
  <p>Paris has been the capital for over twelve centuries.</p>
  <h2>Culture</h2>
  <p>The city is home to the Louvre and the Musée d'Orsay.</p>
  <h2>Transport</h2>
  <p>Paris is served by the Métro, RER, and two international airports.</p>
</main>
</body></html>""",
        # <aside> has no headings; main has a perfect H1+H2s structure
        htags=["h1", "h2", "h2", "h2"],
        violation_count=0, violation_phrases=[],
        score=20, passed=True,
        rec_is_none=True, rec_contains=[],
        why="Sidebar present but has no headings — main H1+H2s structure is perfect.",
    ),

    # ── score 12: single violation ───────────────────────────────────────────
    dict(
        id="h03_one_skip_h1_to_h3",
        html="""<html><body>
<h1>Paris: Capital of France</h1>
<p>Paris is the capital and largest city of France, situated along the Seine.</p>
<h3>History of the French Capital</h3>
<p>Paris became the permanent royal residence under the Capetian kings in the
tenth century, and has served as the national capital ever since.</p>
<h3>Cultural Significance</h3>
<p>The city is home to the Louvre, the Eiffel Tower, and Notre-Dame Cathedral.</p>
</body></html>""",
        # H1→H3 jump — missing H2 is the only violation
        htags=["h1", "h3", "h3"],
        violation_count=1, violation_phrases=["h2"],
        score=12, passed=False,
        rec_is_none=False, rec_contains=["H1", "H2", "H3"],
        why="H1 jumps straight to H3 in a real article — single violation.",
    ),
    dict(
        id="h04_one_h2_before_h1",
        html="""<html><body>
<h2>Introduction to the French Capital</h2>
<p>France is a unitary semi-presidential republic whose capital is Paris.</p>
<h1>Paris: Capital of France</h1>
<h2>Geography</h2>
<p>Paris lies in the north of France along the Seine river.</p>
<h2>Politics</h2>
<p>The Élysée Palace is the official residence of the French President.</p>
</body></html>""",
        # H2 at position 1 before the H1 — one pre-H1 violation
        htags=["h2", "h1", "h2", "h2"],
        violation_count=1, violation_phrases=["before"],
        score=12, passed=False,
        rec_is_none=False, rec_contains=["H1", "H2", "H3"],
        why="H2 introduces the page before H1 — single pre-H1 violation.",
    ),

    # ── score 0: missing H1 ─────────────────────────────────────────────────
    dict(
        id="h05_no_h1_only_h2_h3",
        html="""<html><body>
<h2>About Paris</h2>
<p>Paris is widely recognised as one of the world's great cities, renowned for
its art, architecture, gastronomy, and its central role in European history.</p>
<h3>Geographic Location</h3>
<p>Paris lies in the Île-de-France region in the north of the country.</p>
<h3>Political Status</h3>
<p>Paris is the seat of the French national government and head of state.</p>
<h2>Cultural Heritage</h2>
<p>The city is home to the Louvre, the world's most-visited art museum.</p>
</body></html>""",
        # No H1 → 3 violations (no-H1 + both H2s and H3s before phantom H1)
        htags=["h2", "h3", "h3", "h2"],
        violation_count=5, violation_phrases=["h1"],
        score=0, passed=False,
        rec_is_none=False, rec_contains=["<h1>"],
        why="No H1 at all — score 0 regardless of H2/H3 usage.",
    ),

    # ── score 0: 3+ violations ───────────────────────────────────────────────
    dict(
        id="h06_three_violations_pre_h1_plus_skip",
        html="""<html><body>
<h2>Preamble: Overview of France</h2>
<h3>Background on French Geography</h3>
<h1>Paris: Capital of France</h1>
<h3>Cultural and Historical Significance</h3>
<p>Paris has shaped European civilisation for more than a thousand years,
serving as the centre of art, philosophy, politics, and diplomacy.</p>
</body></html>""",
        # H2 before H1 (1) + H3 before H1 (2) + H1→H3 skip (3) = 3 violations
        htags=["h2", "h3", "h1", "h3"],
        violation_count=3, violation_phrases=["before", "h2"],
        score=0, passed=False,
        rec_is_none=False, rec_contains=["multiple issues"],
        why="Two tags before H1 plus one skip — 3 violations, score 0.",
    ),
]


# ===========================================================================
#  Tests
# ===========================================================================

@pytest.mark.parametrize("case", SOUP_CASES, ids=[c["id"] for c in SOUP_CASES])
def test_soup_input(case):
    """Full pipeline check on direct BeautifulSoup input."""
    r = _run_soup(case["html"])
    _assert_all(case, r)


@pytest.mark.parametrize("case", HTML_CASES, ids=[c["id"] for c in HTML_CASES])
def test_html_input(case):
    """Full pipeline check on HTML input via fetch_and_parse()."""
    r = _run_html(case["html"])
    _assert_all(case, r)


# ---------------------------------------------------------------------------
# Shared assertion helper — checks every output field at once
# ---------------------------------------------------------------------------

def _assert_all(case, r):
    cid = case["id"]

    assert r.details.h_tags_found == case["htags"], (
        f"[{cid}] h_tags_found: got {r.details.h_tags_found}, expected {case['htags']}"
    )
    assert len(r.details.violations) == case["violation_count"], (
        f"[{cid}] violation count: got {len(r.details.violations)}, "
        f"expected {case['violation_count']}\n  violations: {r.details.violations}"
    )
    for phrase in case["violation_phrases"]:
        combined = " ".join(r.details.violations).lower()
        assert phrase.lower() in combined, (
            f"[{cid}] expected '{phrase}' in violations but got: {r.details.violations}"
        )
    assert r.score == case["score"], (
        f"[{cid}] score: got {r.score}, expected {case['score']}"
    )
    assert r.passed == case["passed"], (
        f"[{cid}] passed: got {r.passed}, expected {case['passed']}"
    )

    if case["rec_is_none"]:
        assert r.recommendation is None, (
            f"[{cid}] expected no recommendation, got: {r.recommendation!r}"
        )
    else:
        assert r.recommendation is not None, (
            f"[{cid}] expected a recommendation, got None"
        )
        for phrase in case["rec_contains"]:
            assert phrase in r.recommendation, (
                f"[{cid}] '{phrase}' not found in recommendation: {r.recommendation!r}"
            )
