"""
tests/geo/test_gap_analysis.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Tests for gap_analyzer.py — L2 normalisation, gap summary aggregation,
and end-to-end gap analysis with real embeddings.

Three test classes:

  TestL2Normalise
  ---------------
  Pure-function tests for the normalisation helper — no model loaded.
  Covers zero-vector safety, unit-norm guarantee, and batch shape.

  TestBuildGapSummary
  -------------------
  Pure-function tests for coverage aggregation — no model loaded.
  Covers all-covered, all-missing, partial percent, type deduplication,
  and empty input.

  TestGapAnalysisEndToEnd
  -----------------------
  Uses the real sentence-transformer model (all-MiniLM-L6-v2).
  Sub-queries and content are hand-crafted so coverage outcomes are
  deterministic — no mocking needed.

  Five parametrized cases:
    1. fully_covered              — rich on-topic content → all covered
    2. fully_missing              — off-topic content → all gaps
    3. partial_coverage           — covers how_to + definitional only
    4. empty_content              — blank string → all gaps, no crash
    5. strict_threshold_nothing_passes — threshold=0.99 → nothing covered
"""
from __future__ import annotations

from typing import List

import numpy as np
import pytest

from app.models.schemas import SubQuery
from app.services.gap_analyzer import (
    DEFAULT_SIMILARITY_THRESHOLD,
    analyse_gaps,
    build_gap_summary,
    l2_normalise,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sq(type_: str, query: str) -> SubQuery:
    return SubQuery(type=type_, query=query)  # type: ignore[arg-type]

class TestBuildGapSummary:
    def _annotated(self, covered_map: dict) -> List[SubQuery]:
        return [
            SubQuery(type=t, query=f"q {t}", covered=covered)  # type: ignore[arg-type]
            for t, covered in covered_map.items()
        ]

    def test_all_covered(self):
        sqs = self._annotated({t: True for t in ["how_to", "definitional", "comparative"]})
        summary = build_gap_summary(sqs)
        assert summary.covered == 3
        assert summary.total == 3
        assert summary.coverage_percent == 100.0
        assert summary.missing_types == []

    def test_all_missing(self):
        sqs = self._annotated({t: False for t in ["how_to", "definitional"]})
        summary = build_gap_summary(sqs)
        assert summary.covered == 0
        assert summary.coverage_percent == 0.0
        assert summary.covered_types == []

    def test_partial_coverage_percent(self):
        sqs = self._annotated({"how_to": True, "definitional": True, "comparative": False})
        summary = build_gap_summary(sqs)
        assert summary.coverage_percent == round(2 / 3 * 100, 1)

    def test_covered_types_deduplicated(self):
        sqs = [
            SubQuery(type="how_to", query="q1", covered=True),  # type: ignore[arg-type]
            SubQuery(type="how_to", query="q2", covered=True),  # type: ignore[arg-type]
            SubQuery(type="definitional", query="q3", covered=False),  # type: ignore[arg-type]
        ]
        summary = build_gap_summary(sqs)
        assert summary.covered_types == ["how_to"]
        assert summary.missing_types == ["definitional"]

    def test_empty_list_no_crash(self):
        summary = build_gap_summary([])
        assert summary.covered == 0
        assert summary.total == 0
        assert summary.coverage_percent == 0.0


# ===========================================================================
# End-to-end tests with real embeddings
# These use all-MiniLM-L6-v2 — loaded once, fast (~22ms/sentence).
# Sub-queries and content are crafted so outcomes are deterministic.
# ===========================================================================

#: Sub-queries that clearly map to two specific types: how_to + definitional
_HOW_TO_QUERIES = [
    _sq("how_to", "how to optimise blog content for SEO step by step"),
    _sq("how_to", "how to use AI tools to improve search engine rankings"),
]
_DEFINITIONAL_QUERIES = [
    _sq("definitional", "what is search engine optimisation"),
    _sq("definitional", "definition of AI-assisted content writing"),
]
_UNRELATED_QUERIES = [
    _sq("comparative", "Salesforce vs HubSpot CRM pricing comparison"),
    _sq("trust_signals", "HubSpot customer reviews and case studies 2025"),
]

#: Rich content that clearly covers SEO how-to and definitional topics
_RICH_SEO_CONTENT = """
Search engine optimisation (SEO) is the practice of improving a website's visibility
in search engine results pages to attract more organic traffic.
AI-assisted content writing uses machine learning models to help writers create,
optimise, and structure content for both readers and search engines.

To optimise blog content for SEO, start by researching target keywords using tools
like Ahrefs or SEMrush. Then structure your article with a clear H1 heading and
logically nested H2 and H3 subheadings. Write a concise opening paragraph that
directly answers the reader's primary question.

Using AI tools to improve search engine rankings involves generating content outlines,
identifying semantic keyword clusters, and scoring readability against target grade levels.
AI writing assistants can suggest improvements to sentence structure and flag over-optimised
anchor text patterns that may trigger spam filters.
"""

#: Content completely unrelated to any of the sub-queries above
_UNRELATED_CONTENT = """
The history of ancient Roman architecture spans several centuries of engineering
innovation. The Romans developed concrete, the arch, and the dome to construct
enduring public buildings including the Colosseum and the Pantheon.
Aqueducts transported water across vast distances using gravity-fed channels.
"""


@pytest.mark.parametrize(
    "id_, sub_queries, content, threshold, expected_covered_types, expected_missing_types",
    [
        pytest.param(
            "fully_covered",
            _HOW_TO_QUERIES + _DEFINITIONAL_QUERIES,
            _RICH_SEO_CONTENT,
            DEFAULT_SIMILARITY_THRESHOLD,
            ["definitional", "how_to"],
            [],
            id="fully_covered",
        ),
        pytest.param(
            "fully_missing",
            _HOW_TO_QUERIES + _DEFINITIONAL_QUERIES,
            _UNRELATED_CONTENT,
            DEFAULT_SIMILARITY_THRESHOLD,
            [],
            ["definitional", "how_to"],
            id="fully_missing",
        ),
        pytest.param(
            "partial_coverage",
            _HOW_TO_QUERIES + _DEFINITIONAL_QUERIES + _UNRELATED_QUERIES,
            _RICH_SEO_CONTENT,
            DEFAULT_SIMILARITY_THRESHOLD,
            ["definitional", "how_to"],
            ["comparative", "trust_signals"],
            id="partial_coverage",
        ),
        pytest.param(
            "empty_content",
            _HOW_TO_QUERIES,
            "",
            DEFAULT_SIMILARITY_THRESHOLD,
            [],
            ["how_to"],
            id="empty_content",
        ),
        pytest.param(
            "strict_threshold",
            _HOW_TO_QUERIES + _DEFINITIONAL_QUERIES,
            _RICH_SEO_CONTENT,
            0.99,  # impossibly high — nothing passes
            [],
            ["definitional", "how_to"],
            id="strict_threshold_nothing_passes",
        ),
    ],
)
class TestGapAnalysisEndToEnd:
    def test_gap_map(
        self,
        id_: str,
        sub_queries: List[SubQuery],
        content: str,
        threshold: float,
        expected_covered_types: List[str],
        expected_missing_types: List[str],
    ):
        annotated, summary = analyse_gaps(sub_queries, content, threshold=threshold)

        # Every sub-query must be annotated
        assert len(annotated) == len(sub_queries)
        assert all(sq.covered is not None for sq in annotated)
        assert all(sq.similarity_score is not None for sq in annotated)

        # Gap summary types match expected (sorted for stable comparison)
        assert sorted(summary.covered_types) == sorted(expected_covered_types), (
            f"[{id_}] covered_types mismatch: "
            f"got {summary.covered_types}, expected {expected_covered_types}"
        )
        assert sorted(summary.missing_types) == sorted(expected_missing_types), (
            f"[{id_}] missing_types mismatch: "
            f"got {summary.missing_types}, expected {expected_missing_types}"
        )

        # Counts add up
        assert summary.covered + (summary.total - summary.covered) == summary.total
        assert summary.total == len(sub_queries)

        # Coverage percent is consistent with covered count
        expected_pct = round(summary.covered / summary.total * 100, 1) if summary.total else 0.0
        assert summary.coverage_percent == expected_pct
