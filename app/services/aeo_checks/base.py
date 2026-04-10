"""
base.py
~~~~~~~
Abstract base class every AEO check must implement.

Each check receives a ParsedContent dataclass and returns a CheckResult.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from bs4 import BeautifulSoup

from app.models.schemas import CheckResult


@dataclass
class ParsedContent:
    """Unified context object passed to every check."""

    raw_html: str        # Original HTML/text as received
    soup: BeautifulSoup  # Full parsed HTML tree (may be empty for plain-text input)
    clean_text: str      # Boilerplate-stripped body text
    first_paragraph: str  # First meaningful paragraph text


class BaseCheck(ABC):
    """All AEO checks must inherit from this class."""

    #: snake_case identifier, e.g. "direct_answer"
    check_id: str
    #: Human-readable name, e.g. "Direct Answer Detection"
    name: str
    #: Maximum points this check can award
    max_score: int

    @abstractmethod
    def run(self, content: ParsedContent) -> CheckResult:
        """Execute the check and return a fully-populated CheckResult."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Shared factory helper
    # ------------------------------------------------------------------

    @staticmethod
    def _build_result(
        check_id: str,
        name: str,
        score: int,
        max_score: int,
        details: Any,
        recommendation: str | None,
    ) -> CheckResult:
        return CheckResult(
            check_id=check_id,
            name=name,
            passed=score == max_score,
            score=score,
            max_score=max_score,
            details=details,
            recommendation=recommendation,
        )
