"""
tests/aeo/check_b/test_build_recommendation.py
Unit tests for the htag_hierarchy build_recommendation() helper.
"""
from app.services.aeo_checks.htag_hierarchy import build_recommendation


class TestBuildRecommendation:
    def test_no_recommendation_when_no_violations(self):
        assert build_recommendation([], ["h1", "h2"]) is None

    def test_missing_h1_recommendation(self):
        rec = build_recommendation(["no h1 found"], [])
        assert rec is not None
        assert "h1" in rec.lower()

    def test_multiple_violations_recommendation(self):
        rec = build_recommendation(["v1", "v2", "v3"], ["h1"])
        assert rec is not None
        assert "h1" in rec.lower() or "level" in rec.lower() or "structure" in rec.lower()

    def test_single_violation_generic_recommendation(self):
        rec = build_recommendation(["level skipped"], ["h1", "h2"])
        assert rec is not None
        # Should suggest fixing violations
        assert "violation" in rec.lower() or "heading" in rec.lower()
