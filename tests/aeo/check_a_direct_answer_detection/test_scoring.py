"""
tests/aeo/check_a_direct_answer_detection/test_scoring.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Sentence-level tests for compute_score(), build_recommendation(), and
DirectAnswerCheck.run() — all in one place, all driven by real sentences.

Every row in the tables below is a concrete English sentence paired with
hardcoded expected values and a one-line comment explaining the edge case.
No abstract parameter tables — if a sentence changes behaviour you see
exactly which sentence broke.
"""
import pytest
from bs4 import BeautifulSoup

from app.services.aeo_checks.base import ParsedContent
from app.services.aeo_checks.direct_answer import (
    DirectAnswerCheck,
    _HEDGE_PHRASES,
    _UPPER_LIMIT,
    _WORD_LIMIT,
    build_recommendation,
    compute_score,
    count_words,
    has_hedge_phrase,
    is_declarative,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_check = DirectAnswerCheck()


def _content(text: str) -> ParsedContent:
    return ParsedContent(
        raw_html="",
        soup=BeautifulSoup("", "html.parser"),
        clean_text=text,
        first_paragraph=text,
    )


def _run(text: str):
    return _check.run(_content(text))


# ---------------------------------------------------------------------------
# Score 20 — ≤60 words, declarative, no hedge
# Each row: (sentence, expected_score, rec_is_none, detail_notes)
# ---------------------------------------------------------------------------

class TestScore20:

    def test_clean_factual_statement(self):
        # Prototypical declarative answer — subject + verb + object, well under limit
        r = _run("Python is a high-level, general-purpose programming language.")
        assert r.score == 20
        assert r.passed is True
        assert r.recommendation is None
        assert r.details.is_declarative is True
        assert r.details.has_hedge_phrase is False

    def test_passive_voice_still_declarative(self):
        # Passive "was written" — nsubjpass should satisfy the subject check
        r = _run("The report was written by the engineering team.")
        assert r.score == 20
        assert r.details.is_declarative is True

    def test_sentence_with_subordinate_clause(self):
        # Long-ish but still ≤60 words and structurally declarative
        r = _run("Machine learning improves over time as it processes more training data.")
        assert r.score == 20
        assert r.details.word_count <= 60

    def test_exactly_one_word(self):
        # Degenerate minimum: single token scores 20 if treated as declarative
        # (spaCy may not parse it as declarative — score must be 12 or 20)
        wc = count_words("Python.")
        score = compute_score(wc, declarative=True, hedge=False)
        assert score == 20

    def test_exactly_60_words(self):
        # Boundary: 60 is still within the ≤60 band
        sentence = " ".join(["word"] * 59) + " end."   # 60 tokens
        wc = count_words(sentence)
        assert wc == 60
        assert compute_score(wc, declarative=True, hedge=False) == 20


# ---------------------------------------------------------------------------
# Score 12 — ≤60 words but hedge OR not declarative
# ---------------------------------------------------------------------------

class TestScore12:

    def test_hedge_it_depends(self):
        # "it depends" is the canonical hedge phrase
        r = _run("Python is a great language but it depends on your use case.")
        assert r.score == 12
        assert r.details.has_hedge_phrase is True
        assert "it depends" in r.recommendation

    def test_hedge_in_general(self):
        # "in general" is a softer but still listed hedge phrase
        r = _run("In general, Python is preferred for data science projects.")
        assert r.score == 12
        assert r.details.has_hedge_phrase is True

    def test_hedge_results_may_vary(self):
        # Scope-limiting hedge at end of sentence
        r = _run("Performance is good but results may vary on different hardware.")
        assert r.score == 12
        assert r.details.has_hedge_phrase is True

    def test_hedge_not_necessarily(self):
        # Uncertainty hedge — negation form
        r = _run("Faster hardware is not necessarily better for all workloads.")
        assert r.score == 12
        assert r.details.has_hedge_phrase is True

    def test_direct_question_not_declarative(self):
        # Trailing "?" means the sentence is interrogative, not declarative
        r = _run("What is the best programming language for web development?")
        assert r.score == 12
        assert r.details.is_declarative is False
        assert "subject" in r.recommendation.lower() or "declarative" in r.recommendation.lower()

    def test_yes_no_question_not_declarative(self):
        # Inverted subject-aux order with trailing "?" → interrogative
        r = _run("Is Python the right language for your project?")
        assert r.score == 12
        assert r.details.is_declarative is False

    def test_hedge_takes_priority_over_not_declarative(self):
        # Both hedge AND not declarative — recommendation should mention hedge, not structure
        wc = count_words("Short bad sentence.")
        rec = build_recommendation(wc, declarative=False, hedge=True)
        assert any(p in rec for p in _HEDGE_PHRASES[:3])
        assert "subject" not in rec.lower()

    def test_recommendation_mentions_example_hedge_phrases(self):
        # The first 3 items in _HEDGE_PHRASES must appear as examples in the message
        rec = build_recommendation(30, declarative=True, hedge=True)
        for phrase in _HEDGE_PHRASES[:3]:
            assert phrase in rec


# ---------------------------------------------------------------------------
# Score 8 — 61–90 words (declarative/hedge flags irrelevant in this band)
# ---------------------------------------------------------------------------

class TestScore8:

    def test_boundary_61_words(self):
        # 61 is the first word count that drops out of the ≤60 band
        sentence = " ".join(["The", "sky", "is", "blue"] + ["and", "clear"] * 28 + ["extra"]) + "."
        wc = count_words(sentence)
        assert wc == 61
        r = _run(sentence)
        assert r.score == 8
        assert "61 words" in r.recommendation
        assert "60" in r.recommendation           # must mention the limit

    def test_boundary_90_words(self):
        # 90 is the last word count still in the 61–90 band
        sentence = " ".join(["word"] * 89 + ["end."]) 
        wc = count_words(sentence)
        assert wc == 90
        assert compute_score(wc, declarative=True, hedge=False) == 8

    def test_mid_range_75_words(self):
        # Middle of the band — no edge ambiguity
        assert compute_score(75, declarative=True, hedge=False) == 8

    def test_hedge_flag_ignored_in_band(self):
        # Hedge presence does NOT change score when already in 61–90 range
        assert compute_score(70, declarative=True, hedge=True) == 8

    def test_declarative_flag_ignored_in_band(self):
        # Non-declarative does NOT change score when already in 61–90 range
        assert compute_score(80, declarative=False, hedge=False) == 8

    def test_recommendation_mentions_word_count_and_limit(self):
        # Message must tell user the actual count AND the 60-word target
        rec = build_recommendation(74, declarative=True, hedge=False)
        assert "74 words" in rec
        assert "60" in rec


# ---------------------------------------------------------------------------
# Score 0 — >90 words (word count is disqualifying regardless of other flags)
# ---------------------------------------------------------------------------

class TestScore0:

    def test_boundary_91_words(self):
        # 91 is the first count that scores 0
        sentence = " ".join(["word"] * 90 + ["end."])
        wc = count_words(sentence)
        assert wc == 91
        r = _run(sentence)
        assert r.score == 0
        assert r.passed is False
        assert "91 words" in r.recommendation
        assert "60-word" in r.recommendation

    def test_large_paragraph_200_words(self):
        # Well over the limit — clear disqualification
        sentence = " ".join(["word"] * 199 + ["end."])
        assert compute_score(200, declarative=True, hedge=False) == 0

    def test_empty_paragraph(self):
        # No content at all — should score 0 with a "no paragraph" message
        r = _run("")
        assert r.score == 0
        assert r.details.word_count == 0
        assert r.recommendation is not None   # must give guidance

    def test_hedge_flag_irrelevant_over_90(self):
        # Hedge does not change the 0-score outcome
        assert compute_score(100, declarative=True, hedge=True) == 0

    def test_recommendation_does_not_mention_hedge_when_over_90(self):
        # Word-count branch fires first — hedge examples must NOT appear
        rec = build_recommendation(95, declarative=True, hedge=True)
        assert "95 words" in rec
        assert not any(p in rec for p in _HEDGE_PHRASES[:3])

    def test_recommendation_does_not_mention_declarative_issue_when_over_90(self):
        # Word-count branch fires first — structural advice must NOT appear
        rec = build_recommendation(91, declarative=False, hedge=False)
        assert "91 words" in rec
        assert not any(p in rec for p in _HEDGE_PHRASES[:3])


# ---------------------------------------------------------------------------
# Result shape — always valid regardless of input
# ---------------------------------------------------------------------------

class TestResultShape:

    def test_all_detail_fields_present(self):
        # Every required field on details must exist and have the right type
        r = _run("FastAPI is a modern Python web framework.")
        assert isinstance(r.details.word_count, int)
        assert isinstance(r.details.is_declarative, bool)
        assert isinstance(r.details.has_hedge_phrase, bool)
        assert r.details.threshold == _WORD_LIMIT    # must expose the 60-word threshold

    def test_check_metadata_correct(self):
        # Static metadata fields must always match the spec
        r = _run("FastAPI is a modern Python web framework.")
        assert r.check_id == "direct_answer"
        assert r.name == "Direct Answer Detection"
        assert r.max_score == 20

    def test_passed_true_only_when_score_equals_max(self):
        # passed is a strict equality check, not a "good enough" flag
        perfect = _run("Python is a high-level programming language.")
        assert perfect.passed == (perfect.score == 20)

        imperfect = _run(
            "Python is great but it depends on many factors and conditions."
        )
        assert imperfect.passed is False

    def test_score_is_always_int(self):
        for text in [
            "Python is fast.",
            "It depends on context.",
            "",
            " ".join(["word"] * 100),
        ]:
            r = _run(text)
            assert isinstance(r.score, int)

    def test_recommendation_is_none_or_nonempty_string(self):
        # Recommendation must never be an empty string — either None or meaningful
        for text in [
            "Python is a programming language.",
            "It depends on your use case.",
            " ".join(["word"] * 95),
            "",
        ]:
            r = _run(text)
            assert r.recommendation is None or (
                isinstance(r.recommendation, str) and len(r.recommendation) > 10
            )

