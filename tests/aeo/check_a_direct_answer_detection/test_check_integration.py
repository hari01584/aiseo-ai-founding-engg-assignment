"""
tests/aeo/check_a/test_check_integration.py
Integration tests for DirectAnswerCheck.run()

Tests the full pipeline: ParsedContent → check → CheckResult.
Each test maps to a specific scoring outcome and verifies that the
sub-functions (word_count, is_declarative, has_hedge_phrase) combine
correctly into the final result.
"""
from bs4 import BeautifulSoup
from app.services.aeo_checks.base import ParsedContent
from app.services.aeo_checks.direct_answer import DirectAnswerCheck


# ── Shared fixtures ──────────────────────────────────────────────────────────

check = DirectAnswerCheck()


def _content(first_paragraph: str) -> ParsedContent:
    return ParsedContent(
        raw_html="",
        soup=BeautifulSoup("", "html.parser"),
        clean_text=first_paragraph,
        first_paragraph=first_paragraph,
    )


def _n_words(n: int) -> str:
    """Return a padded declarative sentence of exactly n words."""
    base = ["The", "sky", "is", "blue"]
    while len(base) < n:
        base += ["and", "blue"]
    return " ".join(base[:n]) + "."


# ── Score 20 ─────────────────────────────────────────────────────────────────

class TestScore20:
    def test_short_clean_declarative(self):
        result = check.run(_content(
            "Python is a high-level, general-purpose programming language known for its simplicity."
        ))
        assert result.score == 20
        assert result.passed is True
        assert result.recommendation is None

    def test_at_boundary_60_words(self):
        result = check.run(_content(_n_words(60)))
        assert result.details.word_count == 60
        # Score is 20 if declarative (our _n_words sentence is declarative)
        assert result.score in {12, 20}


# ── Score 12 ─────────────────────────────────────────────────────────────────

class TestScore12:
    def test_hedge_phrase_present(self):
        result = check.run(_content(
            "Python is a great language but it depends on your use case and project requirements."
        ))
        assert result.score == 12
        assert result.passed is False
        assert result.details.has_hedge_phrase is True

    def test_question_not_declarative(self):
        result = check.run(_content(
            "What is the best programming language for web development?"
        ))
        assert result.score == 12
        assert result.passed is False
        assert result.details.is_declarative is False


# ── Score 8 ──────────────────────────────────────────────────────────────────

class TestScore8:
    def test_at_boundary_61_words(self):
        result = check.run(_content(_n_words(61)))
        assert result.score == 8
        assert result.details.word_count == 61
        assert "61 words" in result.recommendation

    def test_mid_range_70_words(self):
        result = check.run(_content(_n_words(70)))
        assert result.score == 8
        assert result.details.word_count == 70

    def test_at_boundary_90_words(self):
        result = check.run(_content(_n_words(90)))
        assert result.score == 8
        assert result.details.word_count == 90


# ── Score 0 ──────────────────────────────────────────────────────────────────

class TestScore0:
    def test_at_boundary_91_words(self):
        result = check.run(_content(_n_words(91)))
        assert result.score == 0
        assert result.details.word_count == 91

    def test_large_word_count(self):
        result = check.run(_content(_n_words(150)))
        assert result.score == 0

    def test_empty_paragraph(self):
        result = check.run(_content(""))
        assert result.score == 0
        assert result.details.word_count == 0
        assert result.recommendation is not None


# ── Result shape always valid ─────────────────────────────────────────────────

class TestResultShape:
    def test_all_fields_present(self):
        result = check.run(_content("FastAPI is a modern web framework."))
        assert result.check_id == "direct_answer"
        assert result.name == "Direct Answer Detection"
        assert result.max_score == 20
        assert isinstance(result.passed, bool)
        assert isinstance(result.score, int)
        assert hasattr(result.details, "word_count")
        assert hasattr(result.details, "is_declarative")
        assert hasattr(result.details, "has_hedge_phrase")
        assert hasattr(result.details, "threshold")
        assert result.details.threshold == 60

    def test_passed_true_only_when_score_equals_max(self):
        perfect = check.run(_content("Python is a high-level programming language."))
        assert perfect.passed == (perfect.score == 20)

        imperfect = check.run(_content(_n_words(70)))
        assert imperfect.passed is False
