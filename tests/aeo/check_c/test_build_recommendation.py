"""
tests/aeo/check_c/test_build_recommendation.py
Unit tests for the readability build_recommendation() helper.
"""
from app.services.aeo_checks.readability import build_recommendation


class TestBuildRecommendation:
    def test_none_when_in_target_range(self):
        assert build_recommendation(7.0) is None
        assert build_recommendation(8.5) is None
        assert build_recommendation(9.0) is None

    def test_too_simple_mentions_grade(self):
        rec = build_recommendation(4.0)
        assert "4.0" in rec or "Grade 4" in rec

    def test_too_simple_says_too_simple(self):
        rec = build_recommendation(3.0)
        assert "simple" in rec.lower() or "complex" in rec.lower() or "detail" in rec.lower()

    def test_too_complex_mentions_grade(self):
        rec = build_recommendation(13.0)
        assert "13.0" in rec or "Grade 13" in rec

    def test_too_complex_says_simplify(self):
        rec = build_recommendation(14.0)
        assert "shorten" in rec.lower() or "jargon" in rec.lower() or "plain" in rec.lower()

    def test_target_range_mentioned_in_both_directions(self):
        rec_low = build_recommendation(3.0)
        rec_high = build_recommendation(14.0)
        assert "7" in rec_low or "7–9" in rec_low
        assert "7" in rec_high or "7–9" in rec_high
