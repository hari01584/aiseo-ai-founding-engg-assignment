"""
tests/geo/test_gap_analysis.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Tests for gap_analyzer.py — cosine similarity math, threshold logic,
gap summary aggregation, and end-to-end gap analysis with real embeddings.

Two test classes:

  TestGapAnalyzerHelpers
  ----------------------
  Pure-function tests — no model loaded.  Cover:
    - l2_normalise:    unit vectors, zero vectors, batch shapes
    - cosine_similarity_matrix:  identity, orthogonal, known pair
    - max_similarity_per_query:  correct max extraction, empty chunks
    - apply_threshold:  exact boundary at configurable threshold
    - build_gap_summary:  counts, de-duplication, coverage_percent

  TestGapAnalysisEndToEnd
  -----------------------
  Uses the real sentence-transformer model (all-MiniLM-L6-v2).
  Sub-queries and content are hand-crafted so coverage labels are
  deterministically above or below 0.72 — no mocking needed.

  Five parametrized cases:
    1. fully_covered   — rich on-topic content → all covered
    2. fully_missing   — completely off-topic content → all gaps
    3. partial         — covers how_to + definitional only
    4. empty_content   — blank string → all gaps, no crash
    5. threshold_shift — same content/queries, higher threshold → fewer covered
"""
from __future__ import annotations

from typing import List

import numpy as np
import pytest

from app.models.schemas import GapSummary, SubQuery
from app.services.gap_analyzer import (
    DEFAULT_SIMILARITY_THRESHOLD,
    analyse_gaps,
    apply_threshold,
    build_gap_summary,
    chunk_content,
    cosine_similarity_matrix,
    l2_normalise,
    max_similarity_per_query,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sq(type_: str, query: str) -> SubQuery:
    return SubQuery(type=type_, query=query)  # type: ignore[arg-type]


def _vec(*values: float) -> np.ndarray:
    return np.array(values, dtype=np.float32)


# ===========================================================================
# Pure helper tests
# ===========================================================================


class TestL2Normalise:
    def test_unit_vector_unchanged(self):
        v = _vec(1.0, 0.0, 0.0)
        result = l2_normalise(v)
        np.testing.assert_allclose(result[0], v, atol=1e-6)

    def test_normalised_vector_has_unit_norm(self):
        v = _vec(3.0, 4.0)
        result = l2_normalise(v)
        norm = np.linalg.norm(result[0])
        assert abs(norm - 1.0) < 1e-6

    def test_zero_vector_does_not_crash(self):
        v = _vec(0.0, 0.0, 0.0)
        result = l2_normalise(v)
        # Norm of zero vector stays zero — division is protected
        assert not np.any(np.isnan(result))

    def test_batch_all_rows_unit_norm(self):
        batch = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        result = l2_normalise(batch)
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, [1.0, 1.0], atol=1e-6)

    def test_output_shape_preserved(self):
        batch = np.random.rand(5, 10).astype(np.float32)
        result = l2_normalise(batch)
        assert result.shape == (5, 10)


class TestCosineSimilarityMatrix:
    def test_identical_vectors_score_one(self):
        v = l2_normalise(_vec(1.0, 2.0, 3.0))
        sim = cosine_similarity_matrix(v, v)
        assert abs(sim[0, 0] - 1.0) < 1e-5

    def test_orthogonal_vectors_score_zero(self):
        q = l2_normalise(_vec(1.0, 0.0))
        c = l2_normalise(_vec(0.0, 1.0))
        sim = cosine_similarity_matrix(q, c)
        assert abs(sim[0, 0]) < 1e-5

    def test_output_shape(self):
        q = l2_normalise(np.random.rand(3, 4).astype(np.float32))
        c = l2_normalise(np.random.rand(5, 4).astype(np.float32))
        sim = cosine_similarity_matrix(q, c)
        assert sim.shape == (3, 5)

    def test_known_pair(self):
        # [1,1] and [1,0] normalised → cosine = 1/sqrt(2) ≈ 0.7071
        q = l2_normalise(_vec(1.0, 1.0))
        c = l2_normalise(_vec(1.0, 0.0))
        sim = cosine_similarity_matrix(q, c)
        assert abs(sim[0, 0] - (1.0 / np.sqrt(2))) < 1e-5

    def test_all_values_in_minus_one_to_one(self):
        q = l2_normalise(np.random.rand(4, 8).astype(np.float32))
        c = l2_normalise(np.random.rand(6, 8).astype(np.float32))
        sim = cosine_similarity_matrix(q, c)
        assert np.all(sim >= -1.0 - 1e-5)
        assert np.all(sim <= 1.0 + 1e-5)


class TestMaxSimilarityPerQuery:
    def test_returns_max_not_mean(self):
        q = l2_normalise(_vec(1.0, 0.0))
        chunks = l2_normalise(np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32))
        result = max_similarity_per_query(q, chunks)
        assert abs(result[0] - 1.0) < 1e-5

    def test_empty_chunks_returns_zeros(self):
        q = l2_normalise(_vec(1.0, 0.0))
        empty = np.zeros((0, 2), dtype=np.float32)
        result = max_similarity_per_query(q, empty)
        assert result[0] == 0.0

    def test_output_length_equals_query_count(self):
        q = l2_normalise(np.random.rand(7, 5).astype(np.float32))
        c = l2_normalise(np.random.rand(3, 5).astype(np.float32))
        result = max_similarity_per_query(q, c)
        assert len(result) == 7


class TestApplyThreshold:
    def test_above_threshold_is_covered(self):
        scores = np.array([0.73], dtype=np.float32)
        assert apply_threshold(scores, threshold=0.72) == [True]

    def test_exactly_at_threshold_is_covered(self):
        scores = np.array([0.72], dtype=np.float32)
        assert apply_threshold(scores, threshold=0.72) == [True]

    def test_below_threshold_is_gap(self):
        scores = np.array([0.719], dtype=np.float32)
        assert apply_threshold(scores, threshold=0.72) == [False]

    def test_custom_threshold_respected(self):
        scores = np.array([0.75], dtype=np.float32)
        assert apply_threshold(scores, threshold=0.80) == [False]
        assert apply_threshold(scores, threshold=0.70) == [True]

    def test_mixed_batch(self):
        scores = np.array([0.50, 0.72, 0.90], dtype=np.float32)
        result = apply_threshold(scores, threshold=0.72)
        assert result == [False, True, True]


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
