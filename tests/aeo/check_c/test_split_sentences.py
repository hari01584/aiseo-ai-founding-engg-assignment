"""
tests/aeo/check_c/test_split_sentences.py
Unit tests for the split_sentences() helper.
"""
from app.services.aeo_checks.readability import split_sentences


class TestSplitSentences:
    def test_single_sentence(self):
        result = split_sentences("Python is a programming language.")
        assert len(result) == 1
        assert result[0] == "Python is a programming language."

    def test_two_sentences(self):
        result = split_sentences("Python is great. It is also easy to learn.")
        assert len(result) == 2

    def test_splits_on_exclamation(self):
        result = split_sentences("This is great! Use it everywhere.")
        assert len(result) == 2

    def test_splits_on_question_mark(self):
        result = split_sentences("What is Python? It is a language.")
        assert len(result) == 2

    def test_filters_short_fragments(self):
        # Fragments with fewer than 3 words are excluded
        result = split_sentences("OK. Python is a great programming language.")
        assert all(len(s.split()) >= 3 for s in result)

    def test_empty_string(self):
        assert split_sentences("") == []

    def test_whitespace_only(self):
        assert split_sentences("   ") == []

    def test_no_punctuation_returns_whole_text(self):
        result = split_sentences("Python is a programming language with clean syntax")
        assert len(result) == 1
