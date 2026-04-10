"""
tests/aeo/check_b/test_validate_htags.py
Unit tests for the validate_htags() helper.

Each test targets exactly one rule or edge case.
"""
import pytest
from app.services.aeo_checks.htag_hierarchy import validate_htags


class TestValidateHtags:
    # ── Perfect structure ────────────────────────────────────────────────────
    def test_perfect_h1_only(self):
        assert validate_htags(["h1"]) == []

    def test_perfect_h1_h2(self):
        assert validate_htags(["h1", "h2", "h2"]) == []

    def test_perfect_h1_h2_h3(self):
        assert validate_htags(["h1", "h2", "h3", "h2", "h3"]) == []

    def test_perfect_full_depth(self):
        assert validate_htags(["h1", "h2", "h3", "h4", "h5", "h6"]) == []

    # ── Rule 1: exactly one H1 ───────────────────────────────────────────────
    def test_no_h1_gives_violation(self):
        violations = validate_htags(["h2", "h3"])
        assert any("h1" in v.lower() and "no" in v.lower() for v in violations)

    def test_two_h1s_gives_violation(self):
        violations = validate_htags(["h1", "h2", "h1"])
        assert any("2" in v and "h1" in v.lower() for v in violations)

    def test_three_h1s_gives_violation(self):
        violations = validate_htags(["h1", "h1", "h1"])
        assert any("3" in v and "h1" in v.lower() for v in violations)

    # ── Rule 2: no level skipped ─────────────────────────────────────────────
    def test_skip_h1_to_h3(self):
        violations = validate_htags(["h1", "h3"])
        assert any("h2" in v for v in violations)

    def test_skip_h2_to_h4(self):
        violations = validate_htags(["h1", "h2", "h4"])
        assert any("h3" in v for v in violations)

    def test_no_skip_when_going_back_up(self):
        # h3 → h2 is going UP a level — not a skip violation
        violations = validate_htags(["h1", "h2", "h3", "h2"])
        assert violations == []

    # ── Rule 3: no tag before H1 ────────────────────────────────────────────
    def test_h2_before_h1_is_violation(self):
        violations = validate_htags(["h2", "h1"])
        assert any("before" in v for v in violations)

    def test_h3_before_h1_is_violation(self):
        violations = validate_htags(["h3", "h1"])
        assert any("before" in v for v in violations)

    def test_multiple_tags_before_h1(self):
        violations = validate_htags(["h2", "h3", "h1"])
        before_violations = [v for v in violations if "before" in v]
        assert len(before_violations) == 2

    # ── Empty input ──────────────────────────────────────────────────────────
    def test_empty_list_gives_violation(self):
        violations = validate_htags([])
        assert len(violations) == 1
        assert "no heading" in violations[0].lower()

    # ── Violation count ──────────────────────────────────────────────────────
    def test_multiple_violations_accumulate(self):
        # Two H1s + a skip = at least 2 violations
        violations = validate_htags(["h1", "h3", "h1"])
        assert len(violations) >= 2
