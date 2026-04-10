"""
fanout.py
~~~~~~~~~
POST /api/fanout/generate

Accepts a target query and optional existing content, generates fan-out
sub-queries via the LLM engine, and optionally runs semantic gap analysis.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.models.schemas import (
    ErrorResponse,
    FanOutRequest,
    FanOutResponse,
    LLMUnavailableError,
)
from app.services.fanout_engine import FanOutConfig, generate_sub_queries
from app.services.gap_analyzer import (
    DEFAULT_SIMILARITY_THRESHOLD,
    analyse_gaps,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def _build_config() -> FanOutConfig:
    """Build the FanOutConfig — swappable for tests."""
    return FanOutConfig(model_name="gpt-4o-mini")


@router.post(
    "/generate",
    response_model=FanOutResponse,
    responses={
        503: {"model": ErrorResponse},
    },
    summary="Query Fan-Out Generator",
    description=(
        "Generates **10–15 sub-queries** across 6 query types by decomposing "
        "the ``target_query`` with an LLM.\n\n"
        "If ``existing_content`` is provided, each sub-query is scored against "
        "the content using sentence-transformer embeddings and a configurable "
        "cosine similarity threshold. Returns a gap map showing which query "
        "types are covered and which are missing."
    ),
)
async def generate(request: FanOutRequest) -> JSONResponse:
    # ------------------------------------------------------------------ #
    # 1. Generate sub-queries via LLM                                     #
    # ------------------------------------------------------------------ #
    config = _build_config()

    try:
        sub_queries = generate_sub_queries(
            target_query=request.target_query,
            config=config,
        )
    except LLMUnavailableError as exc:
        logger.error("Fan-out LLM failure: %s", exc)
        error_body = ErrorResponse(
            error="llm_unavailable",
            message=(
                "Fan-out generation failed. "
                f"The LLM returned an invalid response after {config.max_retries} retries."
            ),
            detail=str(exc),
        )
        return JSONResponse(status_code=503, content=error_body.model_dump())
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error during fan-out generation")
        error_body = ErrorResponse(
            error="internal_error",
            message="An unexpected error occurred during sub-query generation.",
            detail=str(exc),
        )
        return JSONResponse(status_code=503, content=error_body.model_dump())

    # ------------------------------------------------------------------ #
    # 2. Gap analysis  (only when existing_content is provided)           #
    # ------------------------------------------------------------------ #
    gap_summary = None

    if request.existing_content and request.existing_content.strip():
        try:
            sub_queries, gap_summary = analyse_gaps(
                sub_queries=sub_queries,
                content=request.existing_content,
                threshold=DEFAULT_SIMILARITY_THRESHOLD,
            )
        except Exception as exc:  # noqa: BLE001
            # Gap analysis failure is non-fatal — return sub-queries without it
            logger.error("Gap analysis failed (returning sub-queries only): %s", exc)

    # ------------------------------------------------------------------ #
    # 3. Build response                                                    #
    # ------------------------------------------------------------------ #
    response = FanOutResponse(
        target_query=request.target_query,
        model_used=config.model_name,
        total_sub_queries=len(sub_queries),
        sub_queries=sub_queries,
        gap_summary=gap_summary,
    )
    return JSONResponse(status_code=200, content=response.model_dump())
