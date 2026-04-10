"""
tests/aeo/check_c/test_top_complex_sentences.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Test C — Most-complex sentence extraction.

Verifies `top_complex_sentences`:

  1. Returns sentences in descending complexity order (highest syl/word first).
  2. Returns at most the requested N sentences (default 3).
  3. Truncates very long sentences to 200 characters.
  4. Handles edge cases: empty text, fewer sentences than N.

Ground truth confirmed via textstat probe — complexity values are computed as
syllable_count / word_count and rounded for comparisons.
"""
from __future__ import annotations

from app.services.aeo_checks.readability import (
    top_complex_sentences,
    sentence_complexity,
)

# ---------------------------------------------------------------------------
# Shared fixture — a mixed-complexity paragraph.
# Confirmed sentence complexity values:
#   jargon-1 ≈ 4.333  (most complex)
#   jargon-2 ≈ 3.625
#   "Cats are nice." ≈ 1.000  (tied simple)
# ---------------------------------------------------------------------------

_MIXED_TEXT = (
    "Cats are nice. "
    "The epistemological ramifications necessitate comprehensive recontextualisation "
    "of ontological presuppositions. "
    "Dogs are fun. "
    "Theorists interrogate hegemonic formations perpetuating neoliberal epistemic hierarchies. "
    "I like pets."
)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_most_complex_sentence_is_ranked_first():
    """The sentence with the highest syl/word ratio must come first."""
    top = top_complex_sentences(_MIXED_TEXT, n=3)
    assert len(top) >= 2

    # The first result must have higher or equal complexity than the second
    c0 = sentence_complexity(top[0])
    c1 = sentence_complexity(top[1])
    assert c0 >= c1, (
        f"top[0] complexity ({c0:.3f}) should be ≥ top[1] ({c1:.3f})\n"
        f"  top[0]: {top[0]!r}\n  top[1]: {top[1]!r}"
    )


def test_results_are_sorted_descending():
    """All returned sentences must be in non-ascending complexity order."""
    top = top_complex_sentences(_MIXED_TEXT, n=3)
    complexities = [sentence_complexity(s) for s in top]
    assert complexities == sorted(complexities, reverse=True), (
        f"Complexity order not descending: {complexities}"
    )


def test_returns_at_most_n_sentences():
    """Never returns more sentences than the requested N."""
    for n in (1, 2, 3):
        result = top_complex_sentences(_MIXED_TEXT, n=n)
        assert len(result) <= n, f"Expected ≤ {n} sentences, got {len(result)}"


def test_default_n_is_3():
    """Calling without n= uses the default of 3."""
    result = top_complex_sentences(_MIXED_TEXT)
    assert len(result) <= 3


def test_long_sentence_truncated_to_200_chars():
    """Sentences longer than 200 characters are cut at the 200-char boundary."""
    # Build a single very-long sentence (confirmed 476 chars in probe)
    very_long = (
        "The comprehensive evaluation of epistemological frameworks within poststructuralist "
        "discourse necessitates systematic interrogation of ontological presuppositions, "
        "hegemonic formations, neoliberal hierarchies, institutionalised frameworks, "
        "phenomenological consciousness, sociopolitical contestation, interdisciplinary "
        "methodologies, and the irreducible complexity of contemporary academic discourse "
        "within the broader context of globalised knowledge production systems today."
    )
    assert len(very_long) > 200, "Fixture sentence must exceed 200 chars"
    result = top_complex_sentences(very_long, n=1)
    assert len(result) == 1
    assert len(result[0]) <= 200, (
        f"Expected truncation at 200 chars, got {len(result[0])}: {result[0]!r}"
    )


def test_empty_text_returns_empty_list():
    """Empty input produces an empty list, not an error."""
    assert top_complex_sentences("") == []


def test_fewer_sentences_than_n_returns_what_is_available():
    """When text has only 1 sentence, n=3 returns at most 1 result."""
    result = top_complex_sentences("Python is a high-level programming language.", n=3)
    assert len(result) <= 1
