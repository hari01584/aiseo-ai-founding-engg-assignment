"""
tests/aeo/check_c/test_sentence_complexity.py
Unit tests for the sentence_complexity() helper.

complexity = syllable_count / word_count
Higher ratio = harder to read.
"""
from app.services.aeo_checks.readability import sentence_complexity


class TestSentenceComplexity:
    def test_empty_string_returns_zero(self):
        assert sentence_complexity("") == 0.0

    def test_simple_words_lower_complexity(self):
        simple = sentence_complexity("The cat sat on the mat.")
        complex_ = sentence_complexity(
            "The multifaceted socioeconomic ramifications necessitate comprehensive investigation."
        )
        assert simple < complex_

    def test_returns_float(self):
        result = sentence_complexity("Python is easy to use.")
        assert isinstance(result, float)

    def test_higher_syllable_ratio_scores_higher(self):
        # "cat" (1 syl) vs "information" (4 syl) — more syllables per word = higher score
        low = sentence_complexity("The cat ran fast.")
        high = sentence_complexity("The communication infrastructure demonstrated significant deterioration.")
        assert high > low
