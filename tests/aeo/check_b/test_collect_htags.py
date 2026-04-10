"""
tests/aeo/check_b/test_collect_htags.py
Unit tests for the collect_htags() helper.
"""
from bs4 import BeautifulSoup
from app.services.aeo_checks.htag_hierarchy import collect_htags


def _soup(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "html.parser")


class TestCollectHtags:
    def test_no_headings(self):
        assert collect_htags(_soup("<p>Hello</p>")) == []

    def test_single_h1(self):
        assert collect_htags(_soup("<h1>Title</h1>")) == ["h1"]

    def test_dom_order_preserved(self):
        html = "<h1>A</h1><h2>B</h2><h2>C</h2><h3>D</h3>"
        assert collect_htags(_soup(html)) == ["h1", "h2", "h2", "h3"]

    def test_non_heading_tags_excluded(self):
        html = "<h1>Title</h1><p>Para</p><div>Block</div><h2>Section</h2>"
        assert collect_htags(_soup(html)) == ["h1", "h2"]

    def test_all_levels_collected(self):
        html = "<h1>A</h1><h2>B</h2><h3>C</h3><h4>D</h4><h5>E</h5><h6>F</h6>"
        assert collect_htags(_soup(html)) == ["h1", "h2", "h3", "h4", "h5", "h6"]

    def test_empty_html(self):
        assert collect_htags(_soup("")) == []

    def test_h1_after_h2(self):
        # Order is structural — collect_htags just reads DOM order
        html = "<h2>First</h2><h1>Second</h1>"
        assert collect_htags(_soup(html)) == ["h2", "h1"]
