"""
tests/aeo/check_b/test_compute_score.py
Unit tests for the htag_hierarchy compute_score() helper.
"""
from app.services.aeo_checks.htag_hierarchy import compute_score


class TestComputeScore:
    # ── Score 20: 0 violations ───────────────────────────────────────────────
    def test_no_violations(self):
        assert compute_score([], ["h1", "h2"]) == 20

    # ── Score 12: 1–2 violations, H1 present ────────────────────────────────
    def test_one_violation_with_h1(self):
        assert compute_score(["skip"], ["h1", "h2"]) == 12

    def test_two_violations_with_h1(self):
        assert compute_score(["v1", "v2"], ["h1", "h2"]) == 12

    # ── Score 0: no H1 ──────────────────────────────────────────────────────
    def test_no_h1_always_zero(self):
        assert compute_score(["no h1 found"], ["h2", "h3"]) == 0

    def test_no_h1_even_with_one_violation(self):
        assert compute_score(["v1"], []) == 0

    # ── Score 0: 3+ violations ───────────────────────────────────────────────
    def test_three_violations(self):
        assert compute_score(["v1", "v2", "v3"], ["h1"]) == 0

    def test_four_violations(self):
        assert compute_score(["v1", "v2", "v3", "v4"], ["h1"]) == 0
