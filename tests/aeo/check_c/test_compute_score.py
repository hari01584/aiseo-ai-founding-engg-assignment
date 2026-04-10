"""
tests/aeo/check_c/test_compute_score.py
Unit tests for the readability compute_score() helper.

FK Grade → Score mapping:
  7–9   → 20
  6–<7 or >9–10 → 14
  5–<6 or >10–11 → 8
  <5 or >11 → 0
"""
import pytest
from app.services.aeo_checks.readability import compute_score


class TestComputeScore:
    # ── Score 20: grade 7–9 ─────────────────────────────────────────────────
    @pytest.mark.parametrize("fk", [7.0, 7.5, 8.0, 8.5, 9.0])
    def test_score_20_in_target_range(self, fk):
        assert compute_score(fk) == 20

    # ── Score 14: one grade outside ─────────────────────────────────────────
    @pytest.mark.parametrize("fk", [6.0, 6.5, 9.5, 10.0])
    def test_score_14_one_step_outside(self, fk):
        assert compute_score(fk) == 14

    # ── Score 8: two grades outside ─────────────────────────────────────────
    @pytest.mark.parametrize("fk", [5.0, 5.5, 10.5, 11.0])
    def test_score_8_two_steps_outside(self, fk):
        assert compute_score(fk) == 8

    # ── Score 0: far outside ────────────────────────────────────────────────
    @pytest.mark.parametrize("fk", [0.0, 1.0, 4.9, 11.1, 15.0, 20.0])
    def test_score_0_far_outside(self, fk):
        assert compute_score(fk) == 0
