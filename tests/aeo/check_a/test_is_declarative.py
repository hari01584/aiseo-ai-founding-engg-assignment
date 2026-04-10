"""
tests/aeo/check_a/test_is_declarative.py
Unit tests for the is_declarative() helper.

Uses real spaCy (en_core_web_sm / lg) — no mocking.
Each test targets a specific linguistic pattern the parser must handle.
"""
from app.services.aeo_checks.direct_answer import is_declarative


class TestIsDeclarative:
    # ── Passing: clear declarative sentences ────────────────────────────────
    def test_simple_subject_verb_object(self):
        assert is_declarative("Python is a programming language.") is True

    def test_subject_verb_with_complement(self):
        assert is_declarative("FastAPI enables developers to build APIs quickly.") is True

    def test_complex_declarative(self):
        assert is_declarative(
            "Machine learning models improve over time as they are exposed to more data."
        ) is True

    def test_passive_voice_declarative(self):
        # nsubjpass should be detected
        assert is_declarative("The report was written by the team.") is True

    def test_declarative_with_clause(self):
        assert is_declarative("SEO helps websites rank higher in search results.") is True

    # ── Failing: questions ───────────────────────────────────────────────────
    def test_direct_question(self):
        assert is_declarative("What is the best programming language?") is False

    def test_yes_no_question(self):
        assert is_declarative("Is Python easy to learn?") is False

    # ── Failing: empty / fragment ────────────────────────────────────────────
    def test_empty_string(self):
        assert is_declarative("") is False

    def test_single_noun_fragment(self):
        # No verb → not declarative
        assert is_declarative("Overview.") is False

    def test_noun_phrase_no_verb(self):
        assert is_declarative("A comprehensive guide to SEO.") is False
