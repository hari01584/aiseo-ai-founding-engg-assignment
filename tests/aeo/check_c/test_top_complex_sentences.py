"""
tests/aeo/check_c/test_top_complex_sentences.py
Unit tests for the top_complex_sentences() helper.
"""
from app.services.aeo_checks.readability import top_complex_sentences


class TestTopComplexSentences:
    def test_returns_at_most_n(self):
        text = (
            "The cat sat. "
            "Python is easy. "
            "Dogs run fast. "
            "Birds fly high. "
        )
        result = top_complex_sentences(text, n=3)
        assert len(result) <= 3

    def test_empty_text_returns_empty(self):
        assert top_complex_sentences("") == []

    def test_fewer_sentences_than_n(self):
        result = top_complex_sentences("Python is great.", n=3)
        assert len(result) <= 1

    def test_most_complex_ranked_first(self):
        simple = "The cat sat on the mat."
        complex_ = (
            "The multifaceted socioeconomic ramifications of contemporary "
            "globalization necessitate comprehensive interdisciplinary investigation."
        )
        text = f"{simple} {complex_}"
        result = top_complex_sentences(text, n=2)
        # The complex sentence should appear before the simple one
        assert any("ramification" in s.lower() or "socioeconomic" in s.lower() for s in result[:1])

    def test_long_sentences_truncated_to_200_chars(self):
        long_sentence = "word " * 100 + "."
        result = top_complex_sentences(long_sentence, n=1)
        if result:
            assert len(result[0]) <= 200

    def test_default_n_is_3(self):
        text = " ".join([
            "The cat sat on the mat.",
            "Python is a high-level programming language.",
            "Machine learning models improve with more data.",
            "The socioeconomic ramifications are multifaceted.",
        ])
        result = top_complex_sentences(text)
        assert len(result) <= 3
