from __future__ import annotations

from typing import Any, List, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class AEOAnalyzeRequest(BaseModel):
    input_type: str = Field(..., pattern="^(url|text)$", description="'url' or 'text'")
    input_value: str = Field(
        ..., min_length=1, description="A URL or raw HTML/plain-text content"
    )

    @field_validator("input_value")
    @classmethod
    def value_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("input_value must not be blank")
        return v


# ---------------------------------------------------------------------------
# AEO Check detail models
# ---------------------------------------------------------------------------


class DirectAnswerDetails(BaseModel):
    word_count: int
    threshold: int = 60
    is_declarative: bool
    has_hedge_phrase: bool


class HtagHierarchyDetails(BaseModel):
    violations: List[str]
    h_tags_found: List[str]


class ReadabilityDetails(BaseModel):
    fk_grade_level: float
    target_range: str = "7-9"
    complex_sentences: List[str]


class CheckResult(BaseModel):
    check_id: str
    name: str
    passed: bool
    score: int
    max_score: int
    details: Any  # one of the three detail models above
    recommendation: Optional[str] = None


# ---------------------------------------------------------------------------
# AEO Analyze response
# ---------------------------------------------------------------------------


class AEOAnalyzeResponse(BaseModel):
    aeo_score: float
    band: str
    checks: List[CheckResult]


# ---------------------------------------------------------------------------
# Error response
# ---------------------------------------------------------------------------


class ErrorResponse(BaseModel):
    error: str
    message: str
    detail: Optional[str] = None
