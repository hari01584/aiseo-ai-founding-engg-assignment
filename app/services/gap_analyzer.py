"""
gap_analyzer.py
~~~~~~~~~~~~~~~
Semantic gap analysis using sentence-transformer embeddings.

Responsibility
--------------
1. Accept a body of text and a list of sub-queries.
2. Chunk the text into sentences.
3. Encode text chunks and sub-queries using a sentence-transformer model.
4. For each sub-query compute max cosine similarity across all chunks.
5. Apply a configurable threshold to label each sub-query covered/gap.
6. Build and return a GapSummary.

What lives HERE
---------------
  - Embedding model (lazy-loaded singleton)
  - Sentence chunking  (re-uses split_sentences from readability.py)
  - Cosine similarity computation (L2-normalised — correct dot product)
  - Threshold application
  - GapSummary aggregation

What does NOT live here
-----------------------
  - LLM calls  →  app/services/fanout_engine.py
  - Prompt text  →  app/services/fanout_prompts.py
  - HTTP routing  →  app/api/fanout.py

Design notes
------------
- Model is loaded once at module level (lazy) and reused across requests.
- Threshold is always injected as a parameter — never hardcoded in logic
  functions.  The DEFAULT_SIMILARITY_THRESHOLD constant is the single source
  of truth used by the API layer; optimization scripts override it freely.
- Cosine similarity is computed on L2-normalised vectors so the dot product
  equals cosine similarity exactly.  This avoids the common bug of using raw
  dot products on non-unit vectors.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from app.models.schemas import GapSummary, SubQuery
from app.services.aeo_checks.readability import split_sentences

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Threshold constant — the API layer reads this; optimization scripts tune it
# ---------------------------------------------------------------------------

DEFAULT_SIMILARITY_THRESHOLD: float = 0.72

# ---------------------------------------------------------------------------
# Embedding model  (lazy singleton)
# ---------------------------------------------------------------------------

_DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
_model = None  # type: ignore[var-annotated]


def _get_model(model_name: str = _DEFAULT_MODEL_NAME):
    """
    Lazily load and cache the sentence-transformer model.

    all-MiniLM-L6-v2 chosen over all-mpnet-base-v2:
      - ~5× faster encoding (22ms vs 110ms per sentence on CPU)
      - Adequate accuracy for sentence-level semantic similarity
      - 384-dim vs 768-dim — lower memory footprint
    In production with GPU, mpnet is worth the switch; documented in README.
    """
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        logger.info("Loading sentence-transformer model: %s", model_name)
        _model = SentenceTransformer(model_name)
        logger.info("Sentence-transformer model loaded ✓")
    return _model


def warm_up_model(model_name: str = _DEFAULT_MODEL_NAME) -> None:
    """Pre-load the embedding model.  Called from main.py lifespan."""
    _get_model(model_name)


# ---------------------------------------------------------------------------
# Pure helper functions  (independently testable — no model, no I/O)
# ---------------------------------------------------------------------------


def l2_normalise(vectors: NDArray[np.float32]) -> NDArray[np.float32]:
    """
    Row-wise L2 normalisation.

    After normalisation, dot(a, b) == cosine_similarity(a, b) exactly.
    Works on both 1-D (single vector) and 2-D (batch) inputs.
    """
    vectors = np.atleast_2d(vectors).astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    # Avoid division by zero for zero vectors
    norms = np.where(norms == 0, 1.0, norms)
    return vectors / norms


def cosine_similarity_matrix(
    query_vecs: NDArray[np.float32],
    chunk_vecs: NDArray[np.float32],
) -> NDArray[np.float32]:
    """
    Compute cosine similarity between every (query, chunk) pair.

    Parameters
    ----------
    query_vecs : (Q, D) array of L2-normalised query embeddings
    chunk_vecs : (C, D) array of L2-normalised chunk embeddings

    Returns
    -------
    (Q, C) similarity matrix where entry [i, j] = cosine_similarity(q_i, c_j)
    """
    # Both inputs must be normalised; dot product == cosine sim
    return np.dot(query_vecs, chunk_vecs.T)


def max_similarity_per_query(
    query_vecs: NDArray[np.float32],
    chunk_vecs: NDArray[np.float32],
) -> NDArray[np.float32]:
    """
    For each query return the maximum cosine similarity across all chunks.

    Returns a 1-D array of shape (Q,).
    """
    if chunk_vecs.shape[0] == 0:
        # No content chunks — every sub-query is a gap
        return np.zeros(query_vecs.shape[0], dtype=np.float32)

    sim_matrix = cosine_similarity_matrix(query_vecs, chunk_vecs)
    return sim_matrix.max(axis=1)


def apply_threshold(
    similarities: NDArray[np.float32],
    threshold: float,
) -> List[bool]:
    """
    Convert similarity scores to covered/gap booleans.

    Parameters
    ----------
    similarities : 1-D array of max cosine similarity scores per sub-query
    threshold    : minimum score to be considered covered (injected, not hardcoded)

    Returns
    -------
    List of booleans — True = covered, False = gap
    """
    return [float(s) >= threshold for s in similarities]


def build_gap_summary(sub_queries: List[SubQuery]) -> GapSummary:
    """
    Aggregate coverage statistics from a fully-annotated sub-query list.

    Each SubQuery must already have ``covered`` set (not None).
    """
    total = len(sub_queries)
    covered_count = sum(1 for sq in sub_queries if sq.covered)

    covered_types = sorted(
        {sq.type for sq in sub_queries if sq.covered}
    )
    missing_types = sorted(
        {sq.type for sq in sub_queries if not sq.covered}
    )
    coverage_pct = round((covered_count / total) * 100, 1) if total > 0 else 0.0

    return GapSummary(
        covered=covered_count,
        total=total,
        coverage_percent=coverage_pct,
        covered_types=covered_types,
        missing_types=missing_types,
    )


# ---------------------------------------------------------------------------
# Content chunking
# ---------------------------------------------------------------------------


def chunk_content(text: str) -> List[str]:
    """
    Split article text into sentence-level chunks for embedding.

    Re-uses split_sentences from readability.py (same regex splitter already
    in production use) so there is no duplicate sentence-splitting logic.
    Returns an empty list if text is blank.
    """
    if not text or not text.strip():
        return []
    return split_sentences(text)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def analyse_gaps(
    sub_queries: List[SubQuery],
    content: str,
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    model_name: str = _DEFAULT_MODEL_NAME,
) -> Tuple[List[SubQuery], GapSummary]:
    """
    Annotate each sub-query with coverage information and return a GapSummary.

    This function is the single entry point the API router calls.  The
    threshold and model_name are parameters so optimization scripts can sweep
    them freely without touching the production default.

    Parameters
    ----------
    sub_queries : output of fanout_engine.generate_sub_queries() — covered/
                  similarity_score are None at this point.
    content     : the article text to check coverage against.
    threshold   : cosine similarity cutoff for covered=True.
    model_name  : sentence-transformer model identifier.

    Returns
    -------
    (annotated_sub_queries, gap_summary)
    """
    model = _get_model(model_name)

    # 1. Chunk content
    chunks = chunk_content(content)
    query_texts = [sq.query for sq in sub_queries]

    # 2. Encode — always in batch for efficiency
    logger.info(
        "Encoding %d sub-queries and %d content chunks",
        len(query_texts),
        len(chunks),
    )
    query_embeddings_raw: NDArray[np.float32] = model.encode(
        query_texts, convert_to_numpy=True, show_progress_bar=False
    )
    query_vecs = l2_normalise(query_embeddings_raw)

    if chunks:
        chunk_embeddings_raw: NDArray[np.float32] = model.encode(
            chunks, convert_to_numpy=True, show_progress_bar=False
        )
        chunk_vecs = l2_normalise(chunk_embeddings_raw)
    else:
        # No content — every sub-query is a gap with score 0.0
        chunk_vecs = np.zeros((0, query_vecs.shape[1]), dtype=np.float32)

    # 3. Similarity + threshold
    max_sims = max_similarity_per_query(query_vecs, chunk_vecs)
    covered_flags = apply_threshold(max_sims, threshold)

    # 4. Annotate sub-queries (return new objects — inputs are treated as immutable)
    annotated: List[SubQuery] = [
        SubQuery(
            type=sq.type,
            query=sq.query,
            covered=covered_flags[i],
            similarity_score=round(float(max_sims[i]), 4),
        )
        for i, sq in enumerate(sub_queries)
    ]

    # 5. Build summary
    summary = build_gap_summary(annotated)

    logger.info(
        "Gap analysis complete: %d/%d covered (%.1f%%)",
        summary.covered,
        summary.total,
        summary.coverage_percent,
    )

    return annotated, summary
