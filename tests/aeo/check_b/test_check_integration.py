"""
tests/aeo/check_b/test_check_integration.py
Integration tests for HtagHierarchyCheck.run()

Tests the full pipeline: ParsedContent (with real HTML soup) → check → CheckResult.
"""
from bs4 import BeautifulSoup
from app.services.aeo_checks.base import ParsedContent
from app.services.aeo_checks.htag_hierarchy import HtagHierarchyCheck


check = HtagHierarchyCheck()


def _content(html: str) -> ParsedContent:
    soup = BeautifulSoup(html, "html.parser")
    return ParsedContent(
        raw_html=html,
        soup=soup,
        clean_text=soup.get_text(),
        first_paragraph="",
    )


# ── Score 20 ─────────────────────────────────────────────────────────────────

class TestScore20:
    def test_perfect_h1_h2_h3(self):
        result = check.run(_content("<h1>Title</h1><h2>Sec</h2><h3>Sub</h3>"))
        assert result.score == 20
        assert result.passed is True
        assert result.details.violations == []
        assert result.recommendation is None

    def test_perfect_h1_only(self):
        result = check.run(_content("<h1>Title</h1><p>Content here.</p>"))
        assert result.score == 20
        assert result.passed is True

    def test_perfect_multiple_h2s(self):
        result = check.run(_content(
            "<h1>Title</h1><h2>A</h2><h2>B</h2><h3>C</h3><h2>D</h2>"
        ))
        assert result.score == 20


# ── Score 12 ─────────────────────────────────────────────────────────────────

class TestScore12:
    def test_one_skip_violation(self):
        # H1 → H3 (missing H2)
        result = check.run(_content("<h1>Title</h1><h3>Sub</h3>"))
        assert result.score == 12
        assert result.passed is False
        assert len(result.details.violations) == 1

    def test_one_tag_before_h1(self):
        result = check.run(_content("<h2>Early</h2><h1>Title</h1>"))
        assert result.score == 12
        assert result.passed is False

    def test_two_violations(self):
        # H2 before H1 + H1→H3 skip
        result = check.run(_content("<h2>Early</h2><h1>Title</h1><h3>Sub</h3>"))
        assert result.score == 12
        assert len(result.details.violations) == 2


# ── Score 0 ──────────────────────────────────────────────────────────────────

class TestScore0:
    def test_no_h1(self):
        result = check.run(_content("<h2>Section</h2><h3>Sub</h3>"))
        assert result.score == 0
        assert result.passed is False

    def test_three_or_more_violations(self):
        # Multiple H1s + skip + tag before H1
        result = check.run(_content(
            "<h2>Before</h2><h1>A</h1><h3>Skip</h3><h1>B</h1>"
        ))
        assert result.score == 0

    def test_no_headings_at_all(self):
        result = check.run(_content("<p>No headings here.</p>"))
        assert result.score == 0


# ── Result shape ──────────────────────────────────────────────────────────────

class TestResultShape:
    def test_all_fields_present(self):
        result = check.run(_content("<h1>Title</h1><h2>Section</h2>"))
        assert result.check_id == "htag_hierarchy"
        assert result.name == "H-tag Hierarchy"
        assert result.max_score == 20
        assert isinstance(result.passed, bool)
        assert isinstance(result.details.violations, list)
        assert isinstance(result.details.h_tags_found, list)

    def test_h_tags_found_matches_dom_order(self):
        result = check.run(_content("<h1>T</h1><h2>A</h2><h2>B</h2><h3>C</h3>"))
        assert result.details.h_tags_found == ["h1", "h2", "h2", "h3"]
