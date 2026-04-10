"""
htag_hierarchy.py
~~~~~~~~~~~~~~~~~
Check B — H-tag Hierarchy Checker  (max 20 pts)

Rules (each breach = one violation):
  1. Exactly one <h1> must be present.
  2. No heading level may be skipped  (H1 → H3 without H2 is a violation).
  3. No H-tag may appear before the first <h1>.

Scoring:
  0 violations          → 20
  1–2 violations        → 12
  3+ violations OR no H1→  0
"""
from __future__ import annotations

from typing import List, Optional

from bs4 import BeautifulSoup

from app.models.schemas import CheckResult, HtagHierarchyDetails
from app.services.aeo_checks.base import BaseCheck, ParsedContent

_HTAG_NAMES = {"h1", "h2", "h3", "h4", "h5", "h6"}


# ── Pure helper functions (independently testable) ────────────────────────────


def collect_htags(soup: BeautifulSoup) -> List[str]:
    """Return h-tag names in DOM order, e.g. ['h1', 'h2', 'h2', 'h3']."""
    return [tag.name for tag in soup.find_all(_HTAG_NAMES)]


def validate_htags(htags: List[str]) -> List[str]:
    violations: List[str] = []

    if not htags:
        violations.append("No heading tags found in the content.")
        return violations

    h1_count = htags.count("h1")

    # Rule 1 — exactly one H1
    if h1_count == 0:
        violations.append(
            "No <h1> tag found. Every page must have exactly one <h1>."
        )
    elif h1_count > 1:
        violations.append(
            f"Found {h1_count} <h1> tags. There must be exactly one <h1> per page."
        )

    # Index of the first h1 (default to end so pre-h1 loop is skipped if missing)
    first_h1_idx = next(
        (i for i, t in enumerate(htags) if t == "h1"), len(htags)
    )

    # Rule 3 — no h-tag before the first H1
    for i, tag in enumerate(htags[:first_h1_idx]):
        violations.append(
            f"<{tag}> at position {i + 1} appears before the <h1>. "
            "All headings should follow the page's single <h1>."
        )

    # Rule 2 — no level skipped (walk from first H1 onward)
    prev_level: Optional[int] = None
    for tag in htags[first_h1_idx:]:
        current = int(tag[1])
        if prev_level is not None and current > prev_level + 1:
            violations.append(
                f"Heading level skipped: <h{prev_level}> followed directly by "
                f"<h{current}> — level <h{prev_level + 1}> is missing."
            )
        prev_level = current

    return violations


def compute_score(violations: List[str], htags: List[str]) -> int:
    h1_count = htags.count("h1")
    n = len(violations)
    if n == 0:
        return 20
    if h1_count == 0 or n >= 3:
        return 0
    return 12  # 1–2 violations


def build_recommendation(violations: List[str], htags: List[str]) -> Optional[str]:
    if not violations:
        return None
    h1_count = htags.count("h1")
    if h1_count == 0:
        return (
            "Add a single <h1> that clearly states the page's main topic. "
            "All other headings should be nested beneath it in logical order (H2 → H3)."
        )
    if len(violations) >= 3:
        return (
            "Your heading structure has multiple issues. "
            "Start with one <h1>, then use <h2> for major sections and <h3> for sub-sections. "
            "Never skip levels or place headings before the <h1>."
        )
    return (
        "Fix the heading violations listed above. "
        "A clean H1 → H2 → H3 hierarchy helps AI engines understand your content structure."
    )


# ── Check class ───────────────────────────────────────────────────────────────


class HtagHierarchyCheck(BaseCheck):
    check_id = "htag_hierarchy"
    name = "H-tag Hierarchy"
    max_score = 20

    def run(self, content: ParsedContent) -> CheckResult:
        htags = collect_htags(content.soup)
        violations = validate_htags(htags)
        score = compute_score(violations, htags)
        recommendation = build_recommendation(violations, htags)

        details = HtagHierarchyDetails(
            violations=violations,
            h_tags_found=htags,
        )

        return self._build_result(
            self.check_id,
            self.name,
            score=score,
            max_score=self.max_score,
            details=details,
            recommendation=recommendation,
        )
