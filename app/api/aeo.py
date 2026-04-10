"""
aeo.py
~~~~~~
POST /api/aeo/analyze

Accepts a URL or pasted content, runs the three AEO checks, and returns an
AEO Readiness Score (0–100) with per-check diagnostics and recommendations.
"""
from __future__ import annotations

import logging
from typing import List

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.models.schemas import (
    AEOAnalyzeRequest,
    AEOAnalyzeResponse,
    CheckResult,
    ErrorResponse,
)
from app.services.aeo_checks import (
    DirectAnswerCheck,
    HtagHierarchyCheck,
    ReadabilityCheck,
)
from app.services.aeo_checks.base import ParsedContent
from app.services.content_parser import fetch_and_parse

logger = logging.getLogger(__name__)

router = APIRouter()

# Registered checks — order determines the response array order
_CHECKS = [
    DirectAnswerCheck(),
    HtagHierarchyCheck(),
    ReadabilityCheck(),
]

_MAX_RAW = sum(c.max_score for c in _CHECKS)  # 60


def _score_band(normalized: float) -> str:
    if normalized >= 85:
        return "AEO Optimized ✅"
    if normalized >= 65:
        return "Needs Improvement 🟡"
    if normalized >= 40:
        return "Significant Gaps 🔴"
    return "Not AEO Ready ⛔"


@router.post(
    "/analyze",
    response_model=AEOAnalyzeResponse,
    responses={
        422: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
    summary="AEO Content Scorer",
    description=(
        "Accepts a URL or raw HTML/text and runs three AEO checks: "
        "Direct Answer Detection, H-tag Hierarchy, and Snippet Readability. "
        "Returns an AEO Readiness Score (0–100) with per-check diagnostics."
    ),
)
async def analyze(request: AEOAnalyzeRequest) -> JSONResponse:
    # ------------------------------------------------------------------ #
    # 1. Fetch & parse content                                             #
    # ------------------------------------------------------------------ #
    try:
        raw_html, soup, clean_text, first_paragraph = fetch_and_parse(
            request.input_type, request.input_value
        )
    except ValueError as exc:
        logger.warning("Content fetch/parse failed: %s", exc)
        error_body = ErrorResponse(
            error="url_fetch_failed",
            message="Could not retrieve content from the provided URL.",
            detail=str(exc),
        )
        return JSONResponse(status_code=422, content=error_body.model_dump())

    parsed = ParsedContent(
        raw_html=raw_html,
        soup=soup,
        clean_text=clean_text,
        first_paragraph=first_paragraph,
    )

    # ------------------------------------------------------------------ #
    # 2. Run all checks — isolate individual failures                     #
    # ------------------------------------------------------------------ #
    results: List[CheckResult] = []
    for check in _CHECKS:
        try:
            results.append(check.run(parsed))
        except Exception as exc:  # noqa: BLE001
            logger.exception("Check %s raised an unexpected error", check.check_id)
            results.append(
                CheckResult(
                    check_id=check.check_id,
                    name=check.name,
                    passed=False,
                    score=0,
                    max_score=check.max_score,
                    details={},
                    recommendation=f"Check could not be completed: {exc}",
                )
            )

    # ------------------------------------------------------------------ #
    # 3. Aggregate score                                                   #
    # ------------------------------------------------------------------ #
    raw_score = sum(r.score for r in results)
    aeo_score = round((raw_score / _MAX_RAW) * 100, 1)
    band = _score_band(aeo_score)

    response = AEOAnalyzeResponse(
        aeo_score=aeo_score,
        band=band,
        checks=results,
    )
    return JSONResponse(status_code=200, content=response.model_dump())
