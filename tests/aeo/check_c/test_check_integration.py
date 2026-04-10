"""
tests/aeo/check_c/test_check_integration.py
Integration tests for ReadabilityCheck.run()

Uses real textstat scoring. FK grade is non-deterministic per text length,
so tests target score bands using known-grade prose rather than exact values.
"""
from bs4 import BeautifulSoup
from app.services.aeo_checks.base import ParsedContent
from app.services.aeo_checks.readability import ReadabilityCheck


check = ReadabilityCheck()


def _content(text: str) -> ParsedContent:
    return ParsedContent(
        raw_html="",
        soup=BeautifulSoup("", "html.parser"),
        clean_text=text,
        first_paragraph=text[:200],
    )


# Grade-calibrated sample texts
# These are representative; textstat may shift slightly but should stay in band.

_GRADE_7_9_TEXT = (
    "Search engine optimization helps websites rank higher in results. "
    "Good content answers user questions clearly and directly. "
    "Writers should use simple words and short sentences. "
    "The goal is to make information easy to find and understand. "
    "Every page should have one clear topic and a strong opening line."
)

_GRADE_12_PLUS_TEXT = (
    "The multifaceted socioeconomic ramifications of contemporary globalization "
    "necessitate comprehensive interdisciplinary investigation into the systemic "
    "perpetuation of epistemic hierarchies within neoliberal frameworks. "
    "Consequently, practitioners must interrogate foundational ontological assumptions "
    "underpinning hegemonic discursive formations that simultaneously obfuscate and "
    "reify structural inequalities in the contemporary geopolitical landscape."
)

_GRADE_3_TEXT = (
    "Cats are nice. Dogs are fun. I like pets. They are good. "
    "We play all day. It is fun. I run and jump. My dog barks."
)


class TestScore20:
    def test_target_range_text(self):
        result = check.run(_content(_GRADE_7_9_TEXT))
        assert result.score == 20
        assert result.passed is True
        assert result.recommendation is None
        assert 7.0 <= result.details.fk_grade_level <= 9.0


class TestScore0:
    def test_very_complex_text(self):
        result = check.run(_content(_GRADE_12_PLUS_TEXT))
        assert result.score == 0
        assert result.passed is False
        assert result.details.fk_grade_level >= 12

    def test_very_simple_text(self):
        result = check.run(_content(_GRADE_3_TEXT))
        assert result.score == 0
        assert result.passed is False
        assert result.details.fk_grade_level <= 4

    def test_empty_text(self):
        result = check.run(_content(""))
        assert result.score == 0
        assert result.passed is False
        assert result.recommendation is not None


class TestResultShape:
    def test_all_fields_present(self):
        result = check.run(_content(_GRADE_7_9_TEXT))
        assert result.check_id == "readability"
        assert result.name == "Snippet Readability"
        assert result.max_score == 20
        assert isinstance(result.passed, bool)
        assert isinstance(result.details.fk_grade_level, float)
        assert result.details.target_range == "7-9"
        assert isinstance(result.details.complex_sentences, list)

    def test_complex_sentences_at_most_3(self):
        result = check.run(_content(_GRADE_7_9_TEXT))
        assert len(result.details.complex_sentences) <= 3

    def test_complex_sentences_are_strings(self):
        result = check.run(_content(_GRADE_12_PLUS_TEXT))
        for s in result.details.complex_sentences:
            assert isinstance(s, str)

    def test_fk_grade_is_rounded_to_1dp(self):
        result = check.run(_content(_GRADE_7_9_TEXT))
        # Round-tripping: value should have at most 1 decimal place
        assert result.details.fk_grade_level == round(result.details.fk_grade_level, 1)
