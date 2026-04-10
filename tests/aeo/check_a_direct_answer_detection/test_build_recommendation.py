"""
tests/aeo/check_a/test_build_recommendation.py
Unit tests for the build_recommendation() helper.

Verifies the right message is returned for each condition,
and that priority ordering is correct (word_count > hedge > declarative).
"""
import pytest
from app.services.aeo_checks.direct_answer import _HEDGE_PHRASES, build_recommendation


class TestBuildRecommendation:
    # ── Perfect: no recommendation ──────────────────────────────────────────
    def test_none_when_perfect(self):
        assert build_recommendation(30, declarative=True, hedge=False) is None

    def test_none_at_60_word_boundary(self):
        assert build_recommendation(60, declarative=True, hedge=False) is None

    # ── >90 words ────────────────────────────────────────────────────────────
    def test_mentions_actual_word_count_over_90(self):
        rec = build_recommendation(100, declarative=True, hedge=False)
        assert "100 words" in rec

    def test_mentions_60_word_limit_over_90(self):
        rec = build_recommendation(100, declarative=True, hedge=False)
        assert "60-word" in rec or "60 word" in rec

    def test_over_90_takes_priority_over_hedge(self):
        # Word count dominates — hedge advice should NOT appear
        rec = build_recommendation(95, declarative=True, hedge=True)
        assert "95 words" in rec
        assert "hedge" not in rec.lower()

    def test_over_90_takes_priority_over_not_declarative(self):
        rec = build_recommendation(91, declarative=False, hedge=False)
        assert "91 words" in rec
        # hedge-specific phrases must NOT appear — word-count branch took priority
        assert not any(p in rec for p in _HEDGE_PHRASES[:3])

    # ── 61–90 words ──────────────────────────────────────────────────────────
    def test_mentions_actual_word_count_61_to_90(self):
        rec = build_recommendation(74, declarative=True, hedge=False)
        assert "74 words" in rec

    def test_61_to_90_mentions_trim(self):
        rec = build_recommendation(70, declarative=True, hedge=False)
        assert "60 words" in rec or "under 60" in rec

    # ── Hedge phrase ─────────────────────────────────────────────────────────
    def test_hedge_recommendation_content(self):
        rec = build_recommendation(30, declarative=True, hedge=True)
        # Recommendation must mention at least one of the first 3 example phrases
        assert any(p in rec for p in _HEDGE_PHRASES[:3])

    def test_hedge_takes_priority_over_not_declarative(self):
        # When ≤60 words: hedge is checked before declarative
        rec = build_recommendation(30, declarative=False, hedge=True)
        assert any(p in rec for p in _HEDGE_PHRASES[:3])

    # ── Not declarative ──────────────────────────────────────────────────────
    def test_not_declarative_recommendation_content(self):
        rec = build_recommendation(30, declarative=False, hedge=False)
        assert "subject" in rec.lower() or "declarative" in rec.lower()
