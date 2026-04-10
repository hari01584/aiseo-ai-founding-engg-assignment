"""
tests/aeo/check_c/test_complexity_scoring.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Test B — Easy vs hard snippet complexity.

Verifies that `sentence_complexity` correctly differentiates between simple
and polysyllabic text:

  1. Simple monosyllabic sentences score at the minimum ratio (1.0 syl/word).
  2. Academic jargon-heavy sentences score significantly higher than everyday text.
  3. The FK grade and readability score reflect the difference — an easy
     snippet is rated too-simple (score < 20) and a hard one is rated
     too-complex (score = 0).
"""
from __future__ import annotations

import pytest

from app.services.aeo_checks.readability import (
    sentence_complexity,
    split_sentences,
    compute_score,
)
import textstat

# ---------------------------------------------------------------------------
# Sample texts — ground truth confirmed via textstat probe
# ---------------------------------------------------------------------------

# Very simple: monosyllabic words, very short sentences — fk ≈ -2.6, score=0
_EASY_TEXT = (
    "Cats are nice. Dogs are fun. I like pets. They are good. "
    "We play all day. It is fun. I run fast. My dog barks."
)

# Jargon-heavy academic prose — fk ≈ 34.9, score=0
_HARD_TEXT = (
    "The epistemological ramifications of poststructuralist discourse analysis necessitate "
    "comprehensive recontextualisation of foundational ontological presuppositions underpinning "
    "contemporary sociocultural hermeneutics and concomitant methodological paradigms. "
    "Theorists must systematically interrogate hegemonic ideological formations perpetuating "
    "neoliberal epistemic hierarchies within institutionalised interdisciplinary frameworks. "
    "The multidimensional intersubjectivity of phenomenological consciousness constitutes "
    "an irreducible locus of sociopolitical contestation within contemporary investigation."
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_simple_sentences_have_low_complexity():
    """Monosyllabic sentences like 'Cats are nice.' have complexity ≤ 1.0."""
    sentences = split_sentences(_EASY_TEXT)
    assert len(sentences) > 0, "No sentences extracted from easy text"
    for s in sentences:
        c = sentence_complexity(s)
        assert c <= 1.0, (
            f"Expected complexity ≤ 1.0 for simple sentence, got {c:.3f}: {s!r}"
        )


def test_jargon_sentences_have_high_complexity():
    """Academic jargon sentences have complexity well above simple text."""
    sentences = split_sentences(_HARD_TEXT)
    assert len(sentences) > 0, "No sentences extracted from hard text"
    complexities = [sentence_complexity(s) for s in sentences]
    avg_hard = sum(complexities) / len(complexities)
    # Simple text averages ~1.0; hard text should average significantly higher
    assert avg_hard > 2.5, (
        f"Expected avg complexity > 2.5 for jargon text, got {avg_hard:.3f}"
    )


def test_hard_text_is_harder_than_easy_text():
    """The average complexity of jargon text exceeds that of simple text."""
    easy_sents = split_sentences(_EASY_TEXT)
    hard_sents = split_sentences(_HARD_TEXT)

    avg_easy = sum(sentence_complexity(s) for s in easy_sents) / len(easy_sents)
    avg_hard = sum(sentence_complexity(s) for s in hard_sents) / len(hard_sents)

    assert avg_hard > avg_easy, (
        f"Expected hard avg ({avg_hard:.3f}) > easy avg ({avg_easy:.3f})"
    )


def test_easy_text_score_reflects_too_simple():
    """Very simple text gets score=0 (FK well below 5)."""
    fk = textstat.flesch_kincaid_grade(_EASY_TEXT)
    score = compute_score(fk)
    assert score == 0, f"Expected score=0 for easy text, got {score} (fk={fk:.1f})"
    assert fk < 5, f"Expected FK < 5 for easy text, got {fk:.1f}"


def test_hard_text_score_reflects_too_complex():
    """Jargon text gets score=0 (FK well above 12)."""
    fk = textstat.flesch_kincaid_grade(_HARD_TEXT)
    score = compute_score(fk)
    assert score == 0, f"Expected score=0 for hard text, got {score} (fk={fk:.1f})"
    assert fk >= 12, f"Expected FK ≥ 12 for hard text, got {fk:.1f}"
