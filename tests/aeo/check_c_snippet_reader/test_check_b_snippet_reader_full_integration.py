"""
tests/aeo/check_c/test_readability.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Integration tests for ReadabilityCheck.run().

Two parametrized tables exercise the full pipeline end-to-end:

  PLAIN_CASES  – plain-text strings fed directly as clean_text.
                 Sentences are realistic paragraph-length (10-13 words each),
                 covering every scoring band: score 20, 14, 8, and 0 (both
                 too-simple and too-complex).

  HTML_CASES   – full HTML bodies parsed via fetch_and_parse().
                 The check receives a proper ParsedContent built from the
                 parser output, exactly as the live API would do.
                 Cases cover: clean article body (score 20), academic jargon
                 (score 0), and the critical boilerplate-stripping case where
                 nav/footer carry jargon that must NOT influence the FK score.

All FK values were confirmed via textstat probe before being hardcoded.
Each row carries every expected output field so one parametrized test can
assert fk_grade_level (band range), score, passed, recommendation text, and
complex_sentences count in a single shot.
"""
from __future__ import annotations

import pytest
from bs4 import BeautifulSoup

from app.services.aeo_checks.base import ParsedContent
from app.services.aeo_checks.readability import ReadabilityCheck
from app.services.content_parser import fetch_and_parse

_check = ReadabilityCheck()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_plain(text: str):
    """Feed raw text directly as clean_text — bypasses the HTML parser."""
    content = ParsedContent(
        raw_html="",
        soup=BeautifulSoup("", "html.parser"),
        clean_text=text,
        first_paragraph=text[:200],
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
#   id           – shown in pytest -v output
#   fk_lo        – inclusive lower bound for fk_grade_level
#   fk_hi        – inclusive upper bound for fk_grade_level
#   score        – expected result.score  (0 / 8 / 14 / 20)
#   passed       – expected result.passed (True only when score == 20)
#   rec_is_none  – True  → recommendation must be None
#   rec_contains – list of substrings every one of which must appear in rec
#   min_complex  – minimum number of complex_sentences in details (≥ this)
#   why          – one-line human note on what this case exercises
# ---------------------------------------------------------------------------


# ===========================================================================
#  PLAIN-TEXT CASES
#
#  All FK values confirmed via textstat probe.
#  Texts use 10-13-word sentences with controlled syl/word ratios to land
#  reliably in each scoring band without fighting floating-point instability.
# ===========================================================================

PLAIN_CASES = [
    # ── score 20: fk in 7–9 range ───────────────────────────────────────────
    dict(
        id="p01_score20_fk_9",
        text=(
            "Target topics that answer common questions your readers often submit. "
            "Search engines index pages that offer clear and useful answers. "
            "Using headers and bullet points makes reading easier for busy people. "
            "Good content answers user questions clearly and directly every time. "
            "Topic research drives better writing when done before starting content."
        ),
        # Probe confirmed: fk=9.0, score=20
        fk_lo=7.0, fk_hi=9.9,
        score=20, passed=True,
        rec_is_none=True, rec_contains=[],
        min_complex=1,
        why="fk=9.0 — sits in the 7–9 target band, score 20.",
    ),

    # ── score 14: fk in 6.x (one grade below target) ────────────────────────
    dict(
        id="p02_score14_fk_6",
        text=(
            "Content writing helps your website reach more people online each day. "
            "Search engines index pages that offer clear and useful answers. "
            "Using header tags makes your content easy to scan and follow. "
            "Short sentences and active voice make your writing easier to read. "
            "Good content builds trust with your readers over a long period of time."
        ),
        # Probe confirmed: fk=6.3, score=14
        fk_lo=6.0, fk_hi=6.9,
        score=14, passed=False,
        rec_is_none=False, rec_contains=["too simple", "7–9"],
        min_complex=1,
        why="fk=6.3 — one grade below target, score 14, too-simple recommendation.",
    ),

    # ── score 8: fk in 10.x (one grade above target) ────────────────────────
    dict(
        id="p03_score8_fk_10",
        text=(
            "Target topics that answer common questions your readers often submit. "
            "Using headers and bullet points makes reading easier for busy people. "
            "Product pages convert better when they feature customer review sections. "
            "Creating a posting schedule helps content writers publish content often. "
            "Writing better content online requires planning and research each season."
        ),
        # Probe confirmed: fk=10.1, score=8
        fk_lo=10.0, fk_hi=11.0,
        score=8, passed=False,
        rec_is_none=False, rec_contains=["too complex", "7–9"],
        min_complex=1,
        why="fk=10.1 — one grade above target, score 8, too-complex recommendation.",
    ),

    # ── score 0: fk ≤ 4 (very simple) ───────────────────────────────────────
    dict(
        id="p04_score0_very_simple",
        text=(
            "Cats are nice. Dogs are fun. I like pets. They are good. "
            "We play all day. It is fun. I run fast. My dog barks."
        ),
        # Probe confirmed: fk=-2.6, score=0
        fk_lo=-5.0, fk_hi=4.0,
        score=0, passed=False,
        rec_is_none=False, rec_contains=["too simple", "7–9"],
        min_complex=1,
        why="fk=-2.6 — far below any scoring band, score 0.",
    ),

    # ── score 0: fk ≥ 12 (very complex) ─────────────────────────────────────
    dict(
        id="p05_score0_very_complex",
        text=(
            "The epistemological ramifications of poststructuralist discourse analysis necessitate "
            "comprehensive recontextualisation of foundational ontological presuppositions underpinning "
            "contemporary sociocultural hermeneutics and concomitant methodological paradigms. "
            "Theorists must systematically interrogate hegemonic ideological formations perpetuating "
            "neoliberal epistemic hierarchies within institutionalised interdisciplinary frameworks. "
            "The multidimensional intersubjectivity of phenomenological consciousness constitutes "
            "an irreducible locus of sociopolitical contestation within contemporary investigation."
        ),
        # Probe confirmed: fk=34.9, score=0
        fk_lo=12.0, fk_hi=99.0,
        score=0, passed=False,
        rec_is_none=False, rec_contains=["too complex", "7–9"],
        min_complex=3,
        why="fk=34.9 — far above any scoring band, score 0.",
    ),
]


# ===========================================================================
#  HTML CASES
#
#  Full HTML bodies fed through fetch_and_parse(), exactly as the API does.
#  The check's clean_text is whatever _extract_clean_text returns (boilerplate
#  tags nav/header/footer/aside stripped before FK computation).
# ===========================================================================

HTML_CASES = [
    # ── score 20: clean article body ────────────────────────────────────────
    dict(
        id="h01_score20_clean_article",
        html="""\
<html><body>
<nav>Home | About | Blog | Contact</nav>
<article>
<p>Target topics that answer common questions your readers often submit.</p>
<p>Search engines index pages that offer clear and useful answers.</p>
<p>Using headers and bullet points makes reading easier for busy people.</p>
<p>Good content answers user questions clearly and directly every time.</p>
<p>Topic research drives better writing when done before starting content.</p>
</article>
<footer>Privacy Policy | Terms of Service | Sitemap</footer>
</body></html>""",
        # Probe confirmed: clean_text = 5 article sentences, fk=9.0, score=20
        fk_lo=7.0, fk_hi=9.9,
        score=20, passed=True,
        rec_is_none=True, rec_contains=[],
        min_complex=1,
        why="Five grade-9 sentences inside <article>; nav and footer stripped.",
    ),

    # ── score 0: academic jargon in article ─────────────────────────────────
    dict(
        id="h02_score0_jargon_article",
        html="""\
<html><body>
<article>
<p>The epistemological ramifications of poststructuralist discourse analysis necessitate comprehensive recontextualisation of foundational ontological presuppositions underpinning contemporary sociocultural hermeneutics and concomitant methodological paradigms.</p>
<p>Theorists must systematically interrogate hegemonic ideological formations perpetuating neoliberal epistemic hierarchies within institutionalised interdisciplinary frameworks.</p>
<p>The multidimensional intersubjectivity of phenomenological consciousness constitutes an irreducible locus of sociopolitical contestation within contemporary investigation.</p>
</article>
</body></html>""",
        # Probe confirmed: fk=34.9, score=0
        fk_lo=12.0, fk_hi=99.0,
        score=0, passed=False,
        rec_is_none=False, rec_contains=["too complex", "7–9"],
        min_complex=3,
        why="Jargon-heavy <article> body yields fk=34.9, score 0.",
    ),

    # ── boilerplate stripping verification ──────────────────────────────────
    # nav and footer carry the same jargon that would push FK to 34.9 if
    # included.  The article body uses the same grade-9 sentences as h01.
    # If stripping works correctly, FK stays in 7–9 and score=20.
    dict(
        id="h03_boilerplate_stripped_correctly",
        html="""\
<html><body>
<nav>Epistemological ramifications necessitate recontextualisation of ontological presuppositions. Theorists interrogate hegemonic formations perpetuating neoliberal hierarchies institutionalised interdisciplinary frameworks. Multidimensional intersubjectivity constitutes irreducible sociopolitical contestation.</nav>
<article>
<p>Target topics that answer common questions your readers often submit.</p>
<p>Search engines index pages that offer clear and useful answers.</p>
<p>Using headers and bullet points makes reading easier for busy people.</p>
<p>Good content answers user questions clearly and directly every time.</p>
<p>Topic research drives better writing when done before starting content.</p>
</article>
<footer>Epistemological ramifications necessitate recontextualisation of ontological presuppositions. Theorists interrogate hegemonic formations perpetuating neoliberal hierarchies.</footer>
</body></html>""",
        # Probe confirmed: nav+footer jargon stripped, only article text used,
        # fk=9.0, score=20.  If stripping were broken, fk would be ~20+.
        fk_lo=7.0, fk_hi=9.9,
        score=20, passed=True,
        rec_is_none=True, rec_contains=[],
        min_complex=1,
        why="Jargon in nav/footer is stripped; only the grade-9 article body counts.",
    ),
]


# ===========================================================================
#  Tests
# ===========================================================================

@pytest.mark.parametrize("case", PLAIN_CASES, ids=[c["id"] for c in PLAIN_CASES])
def test_plain_text(case):
    """Full pipeline check on plain-text input."""
    r = _run_plain(case["text"])
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

    # FK grade level must fall within the expected band
    fk = r.details.fk_grade_level
    assert case["fk_lo"] <= fk <= case["fk_hi"], (
        f"[{cid}] fk_grade_level {fk} not in [{case['fk_lo']}, {case['fk_hi']}]"
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

    assert len(r.details.complex_sentences) >= case["min_complex"], (
        f"[{cid}] complex_sentences: got {len(r.details.complex_sentences)}, "
        f"expected ≥ {case['min_complex']}"
    )
