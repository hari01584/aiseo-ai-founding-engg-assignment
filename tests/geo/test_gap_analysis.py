"""
tests/geo/test_gap_analysis.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Unit and integration tests for gap_analyzer.py.

  TestL2Normalise        — pure math, no model
  TestBuildGapSummary    — coverage aggregation logic, no model
  TestGapAnalysisEndToEnd — full pipeline with real embeddings
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
# Shared test data
# ---------------------------------------------------------------------------

# Queries that should be covered by _RICH_SEO_CONTENT
HOW_TO_QUERIES = [
    SubQuery(type="how_to", query="how to optimise blog content for SEO step by step"),  # type: ignore[arg-type]
    SubQuery(type="how_to", query="how to use AI tools to improve search engine rankings"),  # type: ignore[arg-type]
]
DEFINITIONAL_QUERIES = [
    SubQuery(type="definitional", query="what is search engine optimisation"),  # type: ignore[arg-type]
    SubQuery(type="definitional", query="definition of AI-assisted content writing"),  # type: ignore[arg-type]
]

# Queries on a completely different topic — should NOT be covered by _RICH_SEO_CONTENT
UNRELATED_QUERIES = [
    SubQuery(type="comparative", query="Salesforce vs HubSpot CRM pricing comparison"),  # type: ignore[arg-type]
    SubQuery(type="trust_signals", query="HubSpot customer reviews and case studies 2025"),  # type: ignore[arg-type]
]

# Article that clearly covers SEO how-to and definitional topics
RICH_SEO_CONTENT = """
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

# Article completely unrelated to any of the queries above
UNRELATED_CONTENT = """
The history of ancient Roman architecture spans several centuries of engineering
innovation. The Romans developed concrete, the arch, and the dome to construct
enduring public buildings including the Colosseum and the Pantheon.
Aqueducts transported water across vast distances using gravity-fed channels.
"""

class TestGapAnalysisEndToEnd:

    def _run(self, queries, content, threshold=DEFAULT_SIMILARITY_THRESHOLD):
        return analyse_gaps(queries, content, threshold=threshold)

    # -- annotation completeness ------------------------------------------

    def test_every_subquery_gets_covered_flag_and_score(self):
        annotated, _ = self._run(HOW_TO_QUERIES, RICH_SEO_CONTENT)

        assert all(sq.covered is not None for sq in annotated)
        assert all(sq.similarity_score is not None for sq in annotated)
        assert all(0.0 <= sq.similarity_score <= 1.0 for sq in annotated)

    # -- coverage outcomes ------------------------------------------------

    def test_on_topic_content_covers_all_query_types(self):
        _, summary = self._run(HOW_TO_QUERIES + DEFINITIONAL_QUERIES, RICH_SEO_CONTENT)

        assert sorted(summary.covered_types) == ["definitional", "how_to"]
        assert summary.missing_types == []

    def test_off_topic_content_misses_all_query_types(self):
        _, summary = self._run(HOW_TO_QUERIES + DEFINITIONAL_QUERIES, UNRELATED_CONTENT)

        assert summary.covered_types == []
        assert sorted(summary.missing_types) == ["definitional", "how_to"]

    def test_partial_coverage_when_some_queries_are_unrelated(self):
        all_queries = HOW_TO_QUERIES + DEFINITIONAL_QUERIES + UNRELATED_QUERIES
        _, summary = self._run(all_queries, RICH_SEO_CONTENT)

        assert sorted(summary.covered_types) == ["definitional", "how_to"]
        assert sorted(summary.missing_types) == ["comparative", "trust_signals"]

    def test_empty_content_marks_everything_as_a_gap(self):
        _, summary = self._run(HOW_TO_QUERIES, content="")

        assert summary.covered == 0
        assert summary.missing_types == ["how_to"]

    def test_impossibly_high_threshold_marks_everything_as_a_gap(self):
        _, summary = self._run(
            HOW_TO_QUERIES + DEFINITIONAL_QUERIES,
            RICH_SEO_CONTENT,
            threshold=0.99,
        )

        assert summary.covered == 0
        assert sorted(summary.missing_types) == ["definitional", "how_to"]

    # -- summary arithmetic -----------------------------------------------

    def test_covered_plus_gaps_equals_total(self):
        annotated, summary = self._run(
            HOW_TO_QUERIES + DEFINITIONAL_QUERIES + UNRELATED_QUERIES, RICH_SEO_CONTENT
        )

        assert summary.total == len(annotated)
        gap_count = sum(1 for sq in annotated if not sq.covered)
        assert summary.covered + gap_count == summary.total

    def test_coverage_percent_matches_covered_count(self):
        annotated, summary = self._run(
            HOW_TO_QUERIES + DEFINITIONAL_QUERIES, RICH_SEO_CONTENT
        )

        expected_pct = round(summary.covered / summary.total * 100, 1)
        assert summary.coverage_percent == expected_pct

