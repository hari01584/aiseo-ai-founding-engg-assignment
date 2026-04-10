"""
tests/aeo/content_parser/test_first_paragraph.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for the "extract first paragraph" behaviour in content_parser.py.

Two internal helpers are tested directly:
  - _extract_first_paragraph_html(soup)   → first <p> with real content
  - _extract_first_paragraph_plain(text)  → first double-newline block

Cases are grouped into:
  TestFirstParagraphHtml  – HTML path
  TestFirstParagraphPlain – plain-text path
  TestFetchAndParseIntegration – end-to-end via the public API
"""
import pytest
from bs4 import BeautifulSoup

from app.services.content_parser import (
    _extract_first_paragraph_html,
    _extract_first_paragraph_plain,
    fetch_and_parse,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _soup(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "html.parser")


# ---------------------------------------------------------------------------
# HTML path
# ---------------------------------------------------------------------------

class TestFirstParagraphHtml:

    def test_single_p_tag_returned(self):
        """A lone <p> with real content is returned verbatim."""
        soup = _soup("<html><body><p>Python is a programming language.</p></body></html>")
        assert _extract_first_paragraph_html(soup) == "Python is a programming language."

    def test_first_p_wins_over_later_ones(self):
        """When several <p> tags exist the first non-trivial one is chosen."""
        soup = _soup(
            "<body>"
            "<p>First paragraph with enough words here.</p>"
            "<p>Second paragraph which should be ignored.</p>"
            "</body>"
        )
        result = _extract_first_paragraph_html(soup)
        assert result == "First paragraph with enough words here."

    def test_no_p_tags_returns_empty_string(self):
        """If there are no <p> tags at all the function returns an empty string."""
        soup = _soup("<body><h1>Just a Heading</h1><div>Some div text</div></body>")
        assert _extract_first_paragraph_html(soup) == ""

    def test_inner_tags_stripped_from_text(self):
        """Text inside child tags (<strong>, <em>, <a>) is included, tags removed."""
        soup = _soup(
            "<body><p>Python is <strong>a high-level</strong> programming language.</p></body>"
        )
        result = _extract_first_paragraph_html(soup)
        assert "high-level" in result
        assert "<strong>" not in result

    def test_empty_html_returns_empty_string(self):
        """Empty document returns empty string."""
        assert _extract_first_paragraph_html(_soup("")) == ""

    def test_whitespace_only_p_skipped(self):
        """A <p> that contains only whitespace does not qualify."""
        soup = _soup(
            "<body>"
            "<p>   </p>"
            "<p>This paragraph has substantial enough content to return.</p>"
            "</body>"
        )
        result = _extract_first_paragraph_html(soup)
        assert "" in result

    def test_p_in_article_tag_returned(self):
        """A <p> inside <article> is valid content, not boilerplate."""
        soup = _soup(
            "<body>"
            "<article><p>Article intro paragraph with genuine content here.</p></article>"
            "</body>"
        )
        result = _extract_first_paragraph_html(soup)
        assert "Article intro" in result

    def test_multiline_p_content_joined(self):
        """Newlines inside a <p> are collapsed via get_text separator."""
        soup = _soup("<body><p>Line one\nline two\nline three of content here.</p></body>")
        result = _extract_first_paragraph_html(soup)
        assert "Line one" in result
        assert "line three" in result


# ---------------------------------------------------------------------------
# Plain-text path
# ---------------------------------------------------------------------------

class TestFirstParagraphPlain:

    def test_single_block_returned(self):
        """Plain text with no blank lines returns the whole string."""
        text = "Python is a high-level programming language used worldwide."
        assert _extract_first_paragraph_plain(text) == text

    def test_first_block_wins_over_second(self):
        """First double-newline block is chosen, not the second."""
        text = "First paragraph with real content.\n\nSecond paragraph ignored."
        result = _extract_first_paragraph_plain(text)
        assert result == "First paragraph with real content."

    def test_leading_blank_lines_ignored(self):
        """Leading newlines before the first block are stripped."""
        text = "\n\nActual first paragraph starts here with content."
        result = _extract_first_paragraph_plain(text)
        assert result == "Actual first paragraph starts here with content."

    def test_multiple_blank_lines_treated_as_separator(self):
        """Three or more newlines still act as a paragraph separator."""
        text = "First paragraph here.\n\n\n\nSecond paragraph here."
        result = _extract_first_paragraph_plain(text)
        assert result == "First paragraph here."

    def test_empty_string_returns_empty(self):
        """Empty input → empty output."""
        assert _extract_first_paragraph_plain("") == ""

    def test_whitespace_only_returns_empty(self):
        """Whitespace-only input → empty string after strip."""
        assert _extract_first_paragraph_plain("   \n\n   ") == ""

    def test_trailing_whitespace_stripped(self):
        """Trailing spaces/newlines in the block are stripped."""
        text = "This is the first paragraph.   "
        result = _extract_first_paragraph_plain(text)
        assert result == "This is the first paragraph."

    def test_returns_string_type(self):
        assert isinstance(_extract_first_paragraph_plain("Hello world today."), str)


# ---------------------------------------------------------------------------
# End-to-end via fetch_and_parse (public API)
# ---------------------------------------------------------------------------

class TestFetchAndParseIntegration:

    def test_html_input_extracts_first_p(self):
        html = (
            "<html><body>"
            "<p>FastAPI is a modern web framework for Python.</p>"
            "<p>Second paragraph is ignored in this call.</p>"
            "</body></html>"
        )
        _, _, _, first_para = fetch_and_parse("text", html)
        assert "FastAPI" in first_para
        assert "Second" not in first_para

    def test_plain_text_input_extracts_first_block(self):
        text = "Machine learning powers modern AI.\n\nDeep learning is a subset."
        _, _, _, first_para = fetch_and_parse("text", text)
        assert "Machine learning" in first_para
        assert "Deep learning" not in first_para

    def test_returns_string_not_none(self):
        _, _, _, first_para = fetch_and_parse("text", "<html><body><p>Hello world today.</p></body></html>")
        assert isinstance(first_para, str)

    def test_no_paragraph_returns_empty_string(self):
        html = "<html><body><h1>Only a Heading Here</h1></body></html>"
        _, _, _, first_para = fetch_and_parse("text", html)
        assert first_para == ""
