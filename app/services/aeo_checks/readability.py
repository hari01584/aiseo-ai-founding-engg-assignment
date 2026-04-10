"""
readability.py
~~~~~~~~~~~~~~
Check C — Snippet Readability Scorer  (max 20 pts)

Steps:
  1. Uses `clean_text` (boilerplate already stripped by content_parser).
  2. Computes Flesch-Kincaid Grade Level via `textstat`.
  3. Identifies the 3 most complex sentences ranked by:
         complexity = syllable_count / max(word_count, 1)

Scoring:
  FK 7–9   → 20
  FK 6 or 10 (i.e. one grade outside range) → 14
  FK 5 or 11 → 8
  FK ≤ 4 or ≥ 12 → 0
"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

import textstat

from app.models.schemas import CheckResult, ReadabilityDetails
from app.services.aeo_checks.base import BaseCheck, ParsedContent

_TARGET_LOW = 7
_TARGET_HIGH = 9
_TOP_N_COMPLEX = 3


# ── Pure helper functions (independently testable) ────────────────────────────


def split_sentences(text: str) -> List[str]:
    """Lightweight sentence splitter — avoids loading a second spaCy pipeline."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip() and len(p.split()) >= 3]


def sentence_complexity(sentence: str) -> float:
    """syllables ÷ word_count — higher means harder to read."""
    word_count = len(sentence.split())
    if word_count == 0:
        return 0.0
    return textstat.syllable_count(sentence) / word_count


def top_complex_sentences(text: str, n: int = _TOP_N_COMPLEX) -> List[str]:
    sentences = split_sentences(text)
    if not sentences:
        return []
    ranked: List[Tuple[float, str]] = sorted(
        ((sentence_complexity(s), s) for s in sentences),
        key=lambda x: x[0],
        reverse=True,
    )
    # Truncate long sentences for display
    return [s[:200] for _, s in ranked[:n]]


def compute_score(fk: float) -> int:
    if _TARGET_LOW <= fk <= _TARGET_HIGH:
        return 20
    if (6 <= fk < _TARGET_LOW) or (_TARGET_HIGH < fk <= 10):
        return 14
    if (5 <= fk < 6) or (10 < fk <= 11):
        return 8
    return 0


def build_recommendation(fk: float) -> Optional[str]:
    if _TARGET_LOW <= fk <= _TARGET_HIGH:
        return None
    if fk < _TARGET_LOW:
        return (
            f"Content reads at Grade {fk:.1f} — too simple. "
            "Add more domain-specific detail and longer, well-structured sentences "
            "to reach the Grade 7–9 target range."
        )
    return (
        f"Content reads at Grade {fk:.1f} — too complex. "
        "Shorten sentences, replace jargon with plain language, "
        "and aim for Grade 7–9 so AI engines can extract clean answers."
    )


# ── Check class ───────────────────────────────────────────────────────────────


class ReadabilityCheck(BaseCheck):
    check_id = "readability"
    name = "Snippet Readability"
    max_score = 20

    def run(self, content: ParsedContent) -> CheckResult:
        text = content.clean_text

        if not text.strip():
            details = ReadabilityDetails(
                fk_grade_level=0.0,
                complex_sentences=[],
            )
            return self._build_result(
                self.check_id,
                self.name,
                score=0,
                max_score=self.max_score,
                details=details,
                recommendation="No readable text found in the content.",
            )

        fk = textstat.flesch_kincaid_grade(text)
        complex_sents = top_complex_sentences(text)
        score = compute_score(fk)
        recommendation = build_recommendation(fk)

        details = ReadabilityDetails(
            fk_grade_level=round(fk, 1),
            complex_sentences=complex_sents,
        )

        return self._build_result(
            self.check_id,
            self.name,
            score=score,
            max_score=self.max_score,
            details=details,
            recommendation=recommendation,
        )
