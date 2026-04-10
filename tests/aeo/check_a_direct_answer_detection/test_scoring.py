"""
tests/aeo/check_a_direct_answer_detection/test_scoring.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Integration tests for DirectAnswerCheck.run().

One table of 10 real sentences, each with fully hardcoded expected values
for every output field.  A single parametrized test runs the full pipeline
(count_words → has_hedge_phrase → is_declarative → compute_score →
build_recommendation → CheckResult) and asserts all fields at once.
"""
import pytest
from bs4 import BeautifulSoup

from app.services.aeo_checks.base import ParsedContent
from app.services.aeo_checks.direct_answer import DirectAnswerCheck

_check = DirectAnswerCheck()


def _run(text: str):
    content = ParsedContent(
        raw_html="",
        soup=BeautifulSoup("", "html.parser"),
        clean_text=text,
        first_paragraph=text,
    )
    return _check.run(content)


# ---------------------------------------------------------------------------
# Test table
#
# Each entry is a dict with:
#   id            – shown in pytest output
#   text          – the sentence fed to the check
#   word_count    – expected details.word_count
#   has_hedge     – expected details.has_hedge_phrase
#   is_decl       – expected details.is_declarative
#   score         – expected result.score
#   passed        – expected result.passed  (True only when score == 20)
#   rec_is_none   – whether recommendation must be None
#   rec_contains  – list of substrings that must appear in recommendation
#   why           – one-line explanation of the case
# ---------------------------------------------------------------------------

CASES = [
    dict(
        id="01_perfect",
        text="Paris is the capital of France.",
        # Short, clear subject + verb → full marks
        word_count=6, has_hedge=False, is_decl=True,
        score=20, passed=True,
        rec_is_none=True, rec_contains=[],
        why="Short, clear declarative sentence — perfect score.",
    ),
    dict(
        id="02_hedge_may_vary",
        text="The capital of France is Paris, but it may vary.",
        # 'may vary' is in _HEDGE_PHRASES → hedge detected, score drops to 12
        word_count=10, has_hedge=True, is_decl=True,
        score=12, passed=False,
        rec_is_none=False, rec_contains=["Remove hedge phrases"],
        why="Contains 'may vary' — hedge phrase detected.",
    ),
    dict(
        id="03_question",
        text="Is Paris the capital of France?",
        # Trailing '?' → is_declarative returns False
        word_count=6, has_hedge=False, is_decl=False,
        score=12, passed=False,
        rec_is_none=False, rec_contains=["declarative"],
        why="Ends with '?' — interrogative, not declarative.",
    ),
    dict(
        id="04_fragment_no_verb",
        text="The capital of France.",
        # Noun phrase with no verb → not declarative
        word_count=4, has_hedge=False, is_decl=False,
        score=12, passed=False,
        rec_is_none=False, rec_contains=["declarative"],
        why="Noun fragment, no verb — incomplete sentence.",
    ),
    dict(
        id="05_slightly_long_still_under_60",
        text="Paris is the capital of France and it is a famous city known for culture, food, art, and tourism.",
        # 19 words, no hedge, declarative → still scores 20
        word_count=19, has_hedge=False, is_decl=True,
        score=20, passed=True,
        rec_is_none=True, rec_contains=[],
        why="19 words — comfortably under 60, full score.",
    ),
    dict(
        id="06_too_long_over_90",
        text=(
            "Paris is the capital of France and one of the most visited cities in the world, "
            "renowned for its iconic landmarks such as the Eiffel Tower, the Louvre, and Notre-Dame Cathedral, "
            "as well as its world-class cuisine, vibrant arts scene, rich history, elegant architecture, "
            "and its enduring status as a global centre of fashion, culture, and diplomacy that has shaped "
            "the modern world for many centuries, attracting tens of millions of tourists, students, diplomats, "
            "and artists from every corner of the globe each and every single year without any exception whatsoever."
        ),
        # 91 words → score 0 regardless of declarative/hedge
        word_count=91, has_hedge=False, is_decl=True,
        score=0, passed=False,
        rec_is_none=False, rec_contains=["91 words", "60-word"],
        why="91 words — exceeds 90-word hard limit, score 0.",
    ),
    dict(
        id="07_imperative_no_subject",
        text="Consider Paris as the capital of France.",
        # No explicit subject → spaCy doesn't find nsubj on ROOT → not declarative
        word_count=7, has_hedge=False, is_decl=False,
        score=12, passed=False,
        rec_is_none=False, rec_contains=["declarative"],
        why="Imperative — no explicit subject, not declarative.",
    ),
    dict(
        id="08_missing_subject",
        text="Is located in France.",
        # Verb phrase fragment, no subject at all
        word_count=4, has_hedge=False, is_decl=False,
        score=12, passed=False,
        rec_is_none=False, rec_contains=["declarative"],
        why="Verb phrase with no subject — incomplete.",
    ),
    dict(
        id="09_simple_valid",
        text="France has its capital in Paris.",
        # Clear subject 'France' + verb 'has' → declarative
        word_count=6, has_hedge=False, is_decl=True,
        score=20, passed=True,
        rec_is_none=True, rec_contains=[],
        why="Valid subject + verb, concise declarative — full score.",
    ),
    dict(
        id="10_subtle_hedge_in_some_cases",
        text="The capital of France is Paris in some cases.",
        # 'in some cases' is a listed hedge phrase
        word_count=9, has_hedge=True, is_decl=True,
        score=12, passed=False,
        rec_is_none=False, rec_contains=["it depends", "depending on"],
        why="Contains 'in some cases' — subtle hedge phrase.",
    ),
]


# ---------------------------------------------------------------------------
# Single parametrized test — checks every field at once
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_direct_answer_check(case):
    r = _run(case["text"])

    # ── detail fields ────────────────────────────────────────────────────────
    assert r.details.word_count   == case["word_count"], \
        f"[{case['id']}] word_count: got {r.details.word_count}, expected {case['word_count']}"

    assert r.details.has_hedge_phrase == case["has_hedge"], \
        f"[{case['id']}] has_hedge_phrase: got {r.details.has_hedge_phrase}, expected {case['has_hedge']}"

    assert r.details.is_declarative   == case["is_decl"], \
        f"[{case['id']}] is_declarative: got {r.details.is_declarative}, expected {case['is_decl']}"

    # ── score & passed ────────────────────────────────────────────────────────
    assert r.score  == case["score"],  \
        f"[{case['id']}] score: got {r.score}, expected {case['score']}"

    assert r.passed == case["passed"], \
        f"[{case['id']}] passed: got {r.passed}, expected {case['passed']}"

    # ── recommendation ────────────────────────────────────────────────────────
    if case["rec_is_none"]:
        assert r.recommendation is None, \
            f"[{case['id']}] expected no recommendation, got: {r.recommendation!r}"
    else:
        assert r.recommendation is not None, \
            f"[{case['id']}] expected a recommendation, got None"
        for phrase in case["rec_contains"]:
            assert phrase in r.recommendation, \
                f"[{case['id']}] '{phrase}' not found in recommendation: {r.recommendation!r}"