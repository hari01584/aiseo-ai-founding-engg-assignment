"""
tests/aeo/check_a/test_compute_score.py
Unit tests for the compute_score() helper.

Covers every scoring branch and all four boundary values (60, 61, 90, 91).
No NLP involved — pure integer logic.
"""
import pytest
from app.services.aeo_checks.direct_answer import compute_score


class TestComputeScore:
    # ── Score 20: ≤60 words, declarative, no hedge ──────────────────────────
    def test_perfect_typical(self):
        assert compute_score(30, declarative=True, hedge=False) == 20

    def test_perfect_at_boundary_60(self):
        assert compute_score(60, declarative=True, hedge=False) == 20

    def test_perfect_at_minimum_1_word(self):
        assert compute_score(1, declarative=True, hedge=False) == 20

    # ── Score 12: ≤60 words but hedge OR not declarative ────────────────────
    def test_score_12_hedge_only(self):
        assert compute_score(40, declarative=True, hedge=True) == 12

    def test_score_12_not_declarative_only(self):
        assert compute_score(40, declarative=False, hedge=False) == 12

    def test_score_12_both_hedge_and_not_declarative(self):
        assert compute_score(40, declarative=False, hedge=True) == 12

    def test_score_12_at_boundary_60_with_hedge(self):
        assert compute_score(60, declarative=True, hedge=True) == 12

    # ── Score 8: 61–90 words ────────────────────────────────────────────────
    def test_score_8_at_boundary_61(self):
        assert compute_score(61, declarative=True, hedge=False) == 8

    def test_score_8_mid_range(self):
        assert compute_score(75, declarative=True, hedge=False) == 8

    def test_score_8_at_boundary_90(self):
        assert compute_score(90, declarative=True, hedge=False) == 8

    def test_score_8_declarative_and_hedge_ignored_in_range(self):
        # Declarative/hedge flags don't affect score when word_count is 61–90
        assert compute_score(75, declarative=False, hedge=True) == 8

    # ── Score 0: >90 words ──────────────────────────────────────────────────
    def test_score_0_at_boundary_91(self):
        assert compute_score(91, declarative=True, hedge=False) == 0

    def test_score_0_large_count(self):
        assert compute_score(500, declarative=True, hedge=False) == 0

    def test_score_0_declarative_and_hedge_ignored_over_90(self):
        assert compute_score(200, declarative=False, hedge=True) == 0
