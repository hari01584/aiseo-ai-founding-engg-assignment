"""
direct_answer.py
~~~~~~~~~~~~~~~~
Check A — Direct Answer Detection  (max 20 pts)

Tests whether the opening paragraph:
  1. Is ≤ 60 words
  2. Is a complete declarative sentence  (spaCy: nsubj + ROOT verb, no trailing ?)
  3. Contains no hedge phrases

Scoring:
  ≤ 60 words + declarative + no hedge  → 20
  ≤ 60 words but hedging OR incomplete → 12
  61–90 words                          →  8
  > 90 words                           →  0
"""
from __future__ import annotations

from typing import Optional

import spacy

from app.models.schemas import CheckResult, DirectAnswerDetails
from app.services.aeo_checks.base import BaseCheck, ParsedContent

# ── spaCy model (lazy-loaded once) ──────────────────────────────────────────
_nlp: Optional[spacy.language.Language] = None


def _load_model(name: str) -> spacy.language.Language:
    """Load a spaCy model, downloading it automatically if not present."""
    try:
        return spacy.load(name)
    except OSError:
        import subprocess
        import sys
        import logging

        logging.getLogger(__name__).info("spaCy model '%s' not found — downloading…", name)
        subprocess.run(
            [sys.executable, "-m", "spacy", "download", name],
            check=True,
        )
        return spacy.load(name)


def _get_nlp() -> spacy.language.Language:
    global _nlp
    if _nlp is None:
        try:
            _nlp = _load_model("en_core_web_lg")
        except Exception:  # noqa: BLE001
            _nlp = _load_model("en_core_web_sm")
    return _nlp


# ── Constants ────────────────────────────────────────────────────────────────

_HEDGE_PHRASES: list[str] = [
    "it depends",
    "may vary",
    "in some cases",
    "this varies",
    "generally speaking",
]

_WORD_LIMIT = 60
_UPPER_LIMIT = 90


# ── Pure helper functions (independently testable) ───────────────────────────


def count_words(text: str) -> int:
    return len(text.split())


def has_hedge_phrase(text: str) -> bool:
    lower = text.lower()
    return any(phrase in lower for phrase in _HEDGE_PHRASES)


def is_declarative(text: str) -> bool:
    """
    Parse the first sentence with spaCy and check:
      - At least one nominal subject  (nsubj / nsubjpass / expl)
      - ROOT token is a VERB or AUX
      - Does not end with '?'  (not a question)
    """
    nlp = _get_nlp()
    doc = nlp(text)

    first_sent = next(iter(doc.sents), None)
    if first_sent is None:
        return False

    has_subject = any(
        tok.dep_ in {"nsubj", "nsubjpass", "expl"} for tok in first_sent
    )
    has_root_verb = any(
        tok.dep_ == "ROOT" and tok.pos_ in {"VERB", "AUX"} for tok in first_sent
    )
    is_question = first_sent.text.strip().endswith("?")

    return has_subject and has_root_verb and not is_question


def compute_score(word_count: int, declarative: bool, hedge: bool) -> int:
    if word_count > _UPPER_LIMIT:
        return 0
    if word_count > _WORD_LIMIT:
        return 8
    # ≤ 60 words
    if declarative and not hedge:
        return 20
    return 12


def build_recommendation(
    word_count: int, declarative: bool, hedge: bool
) -> Optional[str]:
    if word_count > _UPPER_LIMIT:
        return (
            f"Your opening paragraph is {word_count} words — far above the 60-word limit. "
            "Rewrite it as a single direct, declarative sentence that answers the primary query immediately."
        )
    if word_count > _WORD_LIMIT:
        return (
            f"Your opening paragraph is {word_count} words. "
            "Trim it to under 60 words with a direct, declarative answer."
        )
    if hedge:
        return (
            "Remove hedge phrases such as 'it depends', 'may vary', or 'generally speaking'. "
            "State your answer directly and confidently."
        )
    if not declarative:
        return (
            "The opening paragraph lacks a clear subject–verb structure. "
            "Rewrite it as a complete declarative sentence (e.g. 'X is Y because Z.')."
        )
    return None  # perfect score — no recommendation needed


# ── Check class ──────────────────────────────────────────────────────────────


class DirectAnswerCheck(BaseCheck):
    check_id = "direct_answer"
    name = "Direct Answer Detection"
    max_score = 20

    def run(self, content: ParsedContent) -> CheckResult:
        first_para = content.first_paragraph

        if not first_para:
            details = DirectAnswerDetails(
                word_count=0,
                is_declarative=False,
                has_hedge_phrase=False,
            )
            return self._build_result(
                self.check_id,
                self.name,
                score=0,
                max_score=self.max_score,
                details=details,
                recommendation=(
                    "No opening paragraph could be found. "
                    "Ensure your content has a clear first paragraph that directly answers the primary query."
                ),
            )

        wc = count_words(first_para)
        hedge = has_hedge_phrase(first_para)
        decl = is_declarative(first_para)
        score = compute_score(wc, decl, hedge)
        recommendation = build_recommendation(wc, decl, hedge)

        details = DirectAnswerDetails(
            word_count=wc,
            is_declarative=decl,
            has_hedge_phrase=hedge,
        )

        return self._build_result(
            self.check_id,
            self.name,
            score=score,
            max_score=self.max_score,
            details=details,
            recommendation=recommendation,
        )
