"""
tests/aeo/check_a/test_has_hedge_phrase.py
Unit tests for the has_hedge_phrase() helper.
"""
import pytest
from app.services.aeo_checks.direct_answer import has_hedge_phrase


class TestHasHedgePhrase:
    # ── Each phrase is detected ──────────────────────────────────────────────
    @pytest.mark.parametrize("phrase", [
        "it depends",
        "may vary",
        "in some cases",
        "this varies",
        "generally speaking",
    ])
    def test_each_phrase_detected(self, phrase: str):
        assert has_hedge_phrase(f"The answer {phrase} on context.") is True

    # ── Clean sentences ──────────────────────────────────────────────────────
    def test_no_hedge_clean_sentence(self):
        assert has_hedge_phrase("Python is a high-level programming language.") is False

    def test_empty_string(self):
        assert has_hedge_phrase("") is False

    # ── Case insensitivity ───────────────────────────────────────────────────
    def test_uppercase_hedge(self):
        assert has_hedge_phrase("IT DEPENDS on your setup.") is True

    def test_mixed_case_hedge(self):
        assert has_hedge_phrase("Generally Speaking, results differ.") is True

    # ── Phrase position ──────────────────────────────────────────────────────
    def test_hedge_at_start(self):
        assert has_hedge_phrase("Generally speaking, this is the answer.") is True

    def test_hedge_at_end(self):
        assert has_hedge_phrase("Your mileage may vary.") is True  # "may vary" is in the list

    def test_hedge_mid_sentence(self):
        assert has_hedge_phrase("Performance may vary based on hardware.") is True

    # ── Partial word does not trigger ────────────────────────────────────────
    def test_partial_word_no_match(self):
        # "depend" alone ≠ "it depends"
        assert has_hedge_phrase("Results depend on your dataset.") is False

    def test_varies_alone_no_match(self):
        # "varies" alone ≠ "this varies"
        assert has_hedge_phrase("Performance varies across machines.") is False
