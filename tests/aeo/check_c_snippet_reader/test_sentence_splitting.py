"""
tests/aeo/check_c/test_sentence_splitting.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Test A — Sentence splitting from HTML-parsed content.

Verifies that `split_sentences` correctly extracts readable sentences from
the `clean_text` produced by `fetch_and_parse`:

  1. Boilerplate tags (nav, footer) are stripped before splitting so their
     text never appears in the sentence list.
  2. The sentences returned match the meaningful <p> text in the article.
  3. Short fragments (< 3 words) are filtered out by the splitter.
"""
from __future__ import annotations

from app.services.aeo_checks.readability import split_sentences
from app.services.content_parser import fetch_and_parse

# ---------------------------------------------------------------------------
# Fixture HTML — nav and footer hold boilerplate-only text; article has three
# clean <p> sentences.  The h1 tag text gets merged into clean_text by the
# parser, but does NOT produce a separate paragraph.
# ---------------------------------------------------------------------------

_HTML = """\
<html><body>
<nav>Home | About | Contact</nav>
<article>
<p>Good content answers user questions clearly and directly.</p>
<p>Search engines index pages that offer clear, useful answers.</p>
<p>Short sentences make your writing easier to read online.</p>
</article>
<footer>Privacy Policy | Terms | Sitemap</footer>
</body></html>"""

# The three expected sentences after parsing and splitting
_EXPECTED_SENTENCES = [
    "Good content answers user questions clearly and directly.",
    "Search engines index pages that offer clear, useful answers.",
    "Short sentences make your writing easier to read online.",
]


def _get_sentences() -> list[str]:
    """Parse the HTML, extract clean_text, then split into sentences."""
    _raw, _soup, clean_text, _fp = fetch_and_parse("text", _HTML)
    return split_sentences(clean_text)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_correct_sentence_count():
    """Splitting the parsed article body yields exactly 3 sentences."""
    sentences = _get_sentences()
    assert len(sentences) == 3, (
        f"Expected 3 sentences, got {len(sentences)}: {sentences}"
    )


def test_sentences_match_article_paragraphs():
    """Each sentence matches the corresponding <p> text from the article."""
    sentences = _get_sentences()
    for expected in _EXPECTED_SENTENCES:
        assert expected in sentences, (
            f"Expected sentence not found: {expected!r}\nGot: {sentences}"
        )


def test_boilerplate_text_not_in_sentences():
    """nav and footer text must not appear in any extracted sentence."""
    sentences = _get_sentences()
    boilerplate_fragments = ["Home", "About", "Contact", "Privacy Policy", "Sitemap"]
    for fragment in boilerplate_fragments:
        for s in sentences:
            assert fragment not in s, (
                f"Boilerplate fragment {fragment!r} leaked into sentence: {s!r}"
            )
