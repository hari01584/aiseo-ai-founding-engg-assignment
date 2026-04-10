"""
tests/aeo/check_a/test_count_words.py
Unit tests for the count_words() helper.
"""
import pytest
from app.services.aeo_checks.direct_answer import count_words


class TestCountWords:
    def test_simple_sentence(self):
        assert count_words("The sky is blue.") == 4

    def test_empty_string(self):
        assert count_words("") == 0

    def test_whitespace_only(self):
        assert count_words("   ") == 0

    def test_single_word(self):
        assert count_words("Python") == 1

    def test_exactly_60_words(self):
        assert count_words(" ".join(["word"] * 60)) == 60

    def test_exactly_61_words(self):
        assert count_words(" ".join(["word"] * 61)) == 61

    def test_exactly_90_words(self):
        assert count_words(" ".join(["word"] * 90)) == 90

    def test_exactly_91_words(self):
        assert count_words(" ".join(["word"] * 91)) == 91

    def test_extra_internal_spaces_ignored(self):
        # str.split() collapses any whitespace sequence
        assert count_words("hello   world") == 2

    def test_newlines_count_as_separators(self):
        assert count_words("one\ntwo\nthree") == 3
