"""
tests/aeo/check_a/test_has_hedge_phrase.py
Unit tests for the has_hedge_phrase() helper.

The parametrized detection test is driven directly from _HEDGE_PHRASES so it
automatically covers any phrase added to the list in the future.
"""
import pytest
from app.services.aeo_checks.direct_answer import _HEDGE_PHRASES, has_hedge_phrase


class TestHasHedgePhrase:

    # ── Every phrase in the canonical list is individually detected ──────────
    @pytest.mark.parametrize("phrase", _HEDGE_PHRASES)
    def test_each_phrase_detected(self, phrase: str):
        assert has_hedge_phrase(f"The answer {phrase} on context.") is True

    # ── Clean sentences produce no false positives ───────────────────────────
    def test_no_hedge_clean_sentence(self):
        assert has_hedge_phrase("Python is a high-level programming language.") is False

    def test_empty_string(self):
        assert has_hedge_phrase("") is False

    def test_direct_confident_answer(self):
        assert has_hedge_phrase("FastAPI is the fastest way to build Python APIs.") is False

    # ── Case insensitivity ───────────────────────────────────────────────────
    def test_uppercase(self):
        assert has_hedge_phrase("IT DEPENDS on your setup.") is True

    def test_mixed_case(self):
        assert has_hedge_phrase("Generally Speaking, results differ.") is True

    def test_mixed_case_broadly(self):
        assert has_hedge_phrase("Broadly Speaking, this is true.") is True

    # ── Phrase position ──────────────────────────────────────────────────────
    def test_hedge_at_start(self):
        assert has_hedge_phrase("Generally speaking, this is the answer.") is True

    def test_hedge_at_end(self):
        assert has_hedge_phrase("Your mileage may vary.") is True

    def test_hedge_mid_sentence(self):
        assert has_hedge_phrase("Performance may vary based on hardware.") is True

    def test_hedge_in_longer_paragraph(self):
        assert has_hedge_phrase(
            "Python is widely used. However, results may vary depending on the task."
        ) is True

    # ── Partial words / near-matches do NOT trigger ──────────────────────────
    def test_depend_alone_no_match(self):
        # "depend" ≠ "it depends" / "this depends" / "depending on"
        assert has_hedge_phrase("Results depend on your dataset.") is False

    def test_varies_alone_no_match(self):
        # "varies" alone ≠ "this varies"
        assert has_hedge_phrase("Performance varies across machines.") is False

    def test_general_alone_no_match(self):
        # "general" alone ≠ "in general" / "as a general rule"
        assert has_hedge_phrase("This is a general overview.") is False

    def test_possible_alone_no_match(self):
        # "possible" alone ≠ "it is possible"
        assert has_hedge_phrase("A faster solution is possible.") is False

    def test_chance_alone_no_match(self):
        # "chance" alone ≠ "there is a chance"
        assert has_hedge_phrase("There is very little chance of error.") is False
