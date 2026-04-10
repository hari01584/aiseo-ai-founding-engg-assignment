"""
tests/aeo/check_a_direct_answer_detection/test_scoring.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Integration tests for DirectAnswerCheck.run().

Two parametrized tables exercise the full pipeline end-to-end:

  PLAIN_CASES  – plain-text strings fed directly as first_paragraph.
                 Sentences are realistic paragraph-length (30-50 words),
                 covering every scoring band and failure mode.

  HTML_CASES   – full HTML bodies parsed via fetch_and_parse().
                 The check receives a proper ParsedContent built from the
                 parser output, exactly as the live API would do.
                 Cases cover: clean first <p>, hedge in first <p>, question
                 opening, 61-90 word band (score 8), >90 word band (score 0).

Each row carries every expected output field so one parametrized test can
assert word_count, has_hedge_phrase, is_declarative, score, passed, and
recommendation in a single shot.
"""
from __future__ import annotations

import pytest
from bs4 import BeautifulSoup

from app.services.aeo_checks.base import ParsedContent
from app.services.aeo_checks.direct_answer import DirectAnswerCheck
from app.services.content_parser import fetch_and_parse

_check = DirectAnswerCheck()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_plain(text: str):
    """Feed raw text directly — bypasses the HTML parser."""
    content = ParsedContent(
        raw_html="",
        soup=BeautifulSoup("", "html.parser"),
        clean_text=text,
        first_paragraph=text,
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
#   word_count   – expected details.word_count
#   has_hedge    – expected details.has_hedge_phrase
#   is_decl      – expected details.is_declarative
#   score        – expected result.score  (0 / 8 / 12 / 20)
#   passed       – expected result.passed (True only when score == 20)
#   rec_is_none  – True  → recommendation must be None
#   rec_contains – list of substrings every one of which must appear in rec
#   why          – one-line human note on what this case exercises
# ---------------------------------------------------------------------------


# ===========================================================================
#  PLAIN-TEXT CASES
#  Realistic paragraph-length sentences (30-50 words).
# ===========================================================================

PLAIN_CASES = [
    # ── score 20: declarative, ≤60 words, no hedge ──────────────────────────
    dict(
        id="p01_perfect_declarative",
        text=(
            "Paris is the undisputed capital of France, serving as the seat of the national "
            "government, the home of the country's most celebrated cultural institutions, and "
            "the economic engine that drives the broader French national economy."
        ),
        # wc=35, nsubj='Paris', ROOT VERB='is', no hedge → full score
        word_count=35, has_hedge=False, is_decl=True,
        score=20, passed=True,
        rec_is_none=True, rec_contains=[],
        why="35-word declarative answer — every condition met, perfect score.",
    ),
    dict(
        id="p02_perfect_alt_subject",
        text=(
            "France designates Paris as its official capital, a status the city has held "
            "continuously since the early medieval period and one reinforced by centuries of "
            "royal, republican, and imperial governance concentrated within its historic boundaries."
        ),
        # wc=35, nsubj='France', ROOT VERB='designates', no hedge → full score
        word_count=35, has_hedge=False, is_decl=True,
        score=20, passed=True,
        rec_is_none=True, rec_contains=[],
        why="Different subject ('France'), still clean declarative — full score.",
    ),
    dict(
        id="p03_perfect_mid_length",
        text=(
            "Paris is the capital and largest city of France, functioning as the political, "
            "economic, and cultural heart of the country, home to the national government, "
            "France's finest museums, and some of the world's most iconic architectural landmarks."
        ),
        # wc=37, declarative, no hedge → full score
        word_count=37, has_hedge=False, is_decl=True,
        score=20, passed=True,
        rec_is_none=True, rec_contains=[],
        why="37-word declarative with rich detail — still under 60, full score.",
    ),

    # ── score 12: hedge phrase detected ─────────────────────────────────────
    dict(
        id="p04_hedge_depending_on",
        text=(
            "Paris is widely regarded as the capital of France, though depending on whether "
            "you define capital purely in political terms or also in cultural and economic "
            "terms, scholars argue that a city's status as a true capital is more fluid than "
            "official designations suggest."
        ),
        # wc=44, 'depending on' triggers hedge → score 12
        word_count=44, has_hedge=True, is_decl=True,
        score=12, passed=False,
        rec_is_none=False, rec_contains=["Remove hedge phrases"],
        why="44 words but 'depending on' is a listed hedge — score drops to 12.",
    ),
    dict(
        id="p05_hedge_in_some_cases",
        text=(
            "Paris serves as the capital of France under most conventional definitions, though "
            "in some cases regional cities such as Lyon or Strasbourg are considered equally "
            "significant nodes of political and administrative power within the broader structure "
            "of the French republic."
        ),
        # wc=40, 'in some cases' triggers hedge → score 12
        word_count=40, has_hedge=True, is_decl=True,
        score=12, passed=False,
        rec_is_none=False, rec_contains=["Remove hedge phrases"],
        why="'in some cases' is a scope-limiting hedge — score 12.",
    ),

    # ── score 12: not declarative (question) ────────────────────────────────
    dict(
        id="p06_question_opening",
        text=(
            "Is Paris truly the undisputed capital of France, given its unparalleled role as "
            "the seat of the national government, its cultural dominance over the rest of the "
            "country, and its towering significance in European diplomatic history?"
        ),
        # wc=36, trailing '?' → is_declarative False → score 12
        word_count=36, has_hedge=False, is_decl=False,
        score=12, passed=False,
        rec_is_none=False, rec_contains=["declarative"],
        why="36-word question ending in '?' — interrogative, not declarative.",
    ),

    # ── score 12: not declarative (no subject / fragment) ───────────────────
    dict(
        id="p07_fragment_noun_phrase",
        text=(
            "The historic capital of France, renowned worldwide for its extraordinary art, "
            "elegant Haussmann architecture, world-class gastronomy, and its enduring status "
            "as one of the great cities of European civilisation since the medieval period."
        ),
        # wc=33, no ROOT VERB → is_declarative False → score 12
        word_count=33, has_hedge=False, is_decl=False,
        score=12, passed=False,
        rec_is_none=False, rec_contains=["declarative"],
        why="Noun-phrase fragment — no main verb, not declarative.",
    ),
    dict(
        id="p08_imperative_no_subject",
        text=(
            "Consider Paris as the undisputed capital of France, taking into account its "
            "centuries-long history as the seat of the monarchy and republic, its unrivalled "
            "concentration of national institutions, and its extraordinary influence on "
            "European culture, diplomacy, and the arts."
        ),
        # wc=39, imperative verb 'Consider' with no explicit subject → not declarative
        word_count=39, has_hedge=False, is_decl=False,
        score=12, passed=False,
        rec_is_none=False, rec_contains=["declarative"],
        why="Imperative opening — no explicit subject, not declarative.",
    ),
    dict(
        id="p09_verb_phrase_no_subject",
        text=(
            "Is located in northern France along the Seine river, historically significant as "
            "a royal residence since the Capetian dynasty, and widely recognised today as the "
            "political, administrative, and cultural centre of the entire French nation."
        ),
        # wc=35, verb-phrase fragment, no subject → is_declarative False
        word_count=35, has_hedge=False, is_decl=False,
        score=12, passed=False,
        rec_is_none=False, rec_contains=["declarative"],
        why="Verb-phrase fragment with no subject — not declarative.",
    ),

    # ── score 0: >90 words ───────────────────────────────────────────────────
    dict(
        id="p10_over_90_words",
        text=(
            "Paris is the capital of France and one of the most celebrated cities in the entire "
            "world, drawing visitors from every corner of the globe who come to admire its "
            "extraordinary architecture, explore its unrivalled collection of museums and "
            "galleries, savour its legendary cuisine, stroll along the banks of the Seine, "
            "browse the boutiques of the Marais, immerse themselves in a cultural richness that "
            "has been centuries in the making, and witness firsthand a civilisation that has "
            "shaped the trajectory of human history in ways that continue to resonate across "
            "the modern world."
        ),
        # wc=93, declarative but >90 → hard disqualification, score 0
        word_count=93, has_hedge=False, is_decl=True,
        score=0, passed=False,
        rec_is_none=False, rec_contains=["93 words", "60-word"],
        why="93 words — exceeds 90-word hard cap, score 0 regardless of structure.",
    ),
]


# ===========================================================================
#  HTML CASES
#  Full HTML bodies fed through fetch_and_parse(), exactly as the API does.
#  The check's first_paragraph is whatever _extract_first_paragraph_html
#  returns (first <p> tag in document order).
# ===========================================================================

HTML_CASES = [
    # ── score 20: clean declarative first <p> ───────────────────────────────
    dict(
        id="h01_perfect_single_p",
        html="""<html><body>
<p>Paris is the undisputed capital of France, serving as the seat of its national
government, the home of its most celebrated cultural institutions, and the engine
of its economy, drawing tens of millions of international visitors every year.</p>
<p>Other major French cities include Lyon, Marseille, and Bordeaux.</p>
</body></html>""",
        # First <p> = 38-word declarative, no hedge → score 20
        word_count=38, has_hedge=False, is_decl=True,
        score=20, passed=True,
        rec_is_none=True, rec_contains=[],
        why="Clean first <p> — declarative, under 60 words, no hedge.",
    ),
    dict(
        id="h02_perfect_article_layout",
        html="""<html><body>
<header><nav>Home | About | Contact</nav></header>
<article>
  <h1>What is the capital of France?</h1>
  <p>Paris is the capital of France and has served as the country's political,
  cultural, and administrative centre for more than a thousand years, housing the
  Élysée Palace, the National Assembly, and dozens of the world's most-visited
  cultural landmarks.</p>
  <p>The city lies on the Seine river in northern France.</p>
</article>
</body></html>""",
        # First <p> = 38-word declarative inside <article> → score 20
        word_count=38, has_hedge=False, is_decl=True,
        score=20, passed=True,
        rec_is_none=True, rec_contains=[],
        why="Realistic article layout — first <p> after h1 is clean declarative.",
    ),

    # ── score 12: hedge phrase in first <p> ─────────────────────────────────
    dict(
        id="h03_hedge_depending_on",
        html="""<html><body>
<p>Paris is widely regarded as the capital of France, though depending on whether
you define capital purely in political terms or also in cultural and economic terms,
scholars argue that a city's status as a true capital is more fluid than official
designations suggest.</p>
<p>France is a unitary semi-presidential republic with Paris at its centre.</p>
</body></html>""",
        # First <p> has 'depending on' → hedge detected, score 12
        word_count=44, has_hedge=True, is_decl=True,
        score=12, passed=False,
        rec_is_none=False, rec_contains=["Remove hedge phrases"],
        why="First <p> contains 'depending on' — hedge detected in HTML context.",
    ),
    dict(
        id="h04_hedge_in_some_cases",
        html="""<html><body>
<p>Paris is recognised as the capital of France under most conventional definitions,
though in some cases regional capitals such as Lyon or Strasbourg are considered
equally significant nodes of political and administrative power within the broader
structure of the French republic.</p>
</body></html>""",
        # 'in some cases' triggers hedge → score 12
        word_count=41, has_hedge=True, is_decl=True,
        score=12, passed=False,
        rec_is_none=False, rec_contains=["Remove hedge phrases"],
        why="Scope-limiting hedge 'in some cases' in HTML first paragraph.",
    ),

    # ── score 12: first <p> is a question ───────────────────────────────────
    dict(
        id="h05_question_first_p",
        html="""<html><body>
<p>Is Paris truly the undisputed capital of France, considering its unparalleled
role as the seat of government, its cultural dominance over the rest of the country,
and its towering significance in European and global diplomatic history?</p>
<p>The answer, of course, is yes.</p>
</body></html>""",
        # First <p> is a question → is_declarative False, score 12
        word_count=36, has_hedge=False, is_decl=False,
        score=12, passed=False,
        rec_is_none=False, rec_contains=["declarative"],
        why="First <p> ends with '?' — interrogative opener, score 12.",
    ),

    # ── score 8: first <p> is 61–90 words ───────────────────────────────────
    dict(
        id="h06_score_8_between_60_and_90",
        html="""<html><body>
<p>Paris has been the capital of France for more than twelve centuries, a distinction
it earned through its central role in the Frankish kingdom, its growth as a royal
residence under the Capetian dynasty, its transformation into a modern metropolis
under Haussmann, and its enduring status as the political, cultural, and economic
engine of the French state even in the contemporary era.</p>
</body></html>""",
        # wc=62, declarative, no hedge → 61–90 band → score 8
        word_count=62, has_hedge=False, is_decl=True,
        score=8, passed=False,
        rec_is_none=False, rec_contains=["62 words", "60"],
        why="62-word first <p> — just over the 60-word limit, score 8.",
    ),

    # ── score 0: first <p> is >90 words ─────────────────────────────────────
    dict(
        id="h07_score_0_over_90",
        html="""<html><body>
<p>Paris is the capital of France and one of the most celebrated cities in the entire
world, drawing visitors from every corner of the globe who come to admire its
extraordinary architecture, explore its unrivalled collection of museums and galleries,
savour its legendary cuisine, stroll along the banks of the Seine, browse the boutiques
of the Marais, immerse themselves in a cultural richness that has been centuries in the
making, and witness firsthand a civilisation that has shaped the trajectory of human
history in ways that continue to resonate across the modern world today.</p>
</body></html>""",
        # wc=94, declarative, but >90 → score 0
        word_count=94, has_hedge=False, is_decl=True,
        score=0, passed=False,
        rec_is_none=False, rec_contains=["94 words", "60-word"],
        why="94-word first <p> — exceeds hard cap, score 0.",
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

    assert r.details.word_count == case["word_count"], (
        f"[{cid}] word_count: got {r.details.word_count}, expected {case['word_count']}"
    )
    assert r.details.has_hedge_phrase == case["has_hedge"], (
        f"[{cid}] has_hedge_phrase: got {r.details.has_hedge_phrase}, expected {case['has_hedge']}"
    )
    assert r.details.is_declarative == case["is_decl"], (
        f"[{cid}] is_declarative: got {r.details.is_declarative}, expected {case['is_decl']}"
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