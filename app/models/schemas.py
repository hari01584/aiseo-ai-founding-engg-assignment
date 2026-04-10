from __future__ import annotations

from typing import Any, List, Literal, Optional

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


# ---------------------------------------------------------------------------
# Fan-Out / GEO  — request & response models
# ---------------------------------------------------------------------------

#: The six canonical sub-query types the LLM must generate
SubQueryType = Literal[
    "comparative",
    "feature_specific",
    "use_case",
    "trust_signals",
    "how_to",
    "definitional",
]


class FanOutRequest(BaseModel):
    target_query: str = Field(..., min_length=1, description="The query to fan out")
    existing_content: Optional[str] = Field(
        None,
        description="Optional article text to run gap analysis against",
    )

    @field_validator("target_query")
    @classmethod
    def query_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("target_query must not be blank")
        return v.strip()


class SubQuery(BaseModel):
    """A single generated sub-query, optionally annotated with gap-analysis results."""

    type: SubQueryType
    query: str

    # Present only when existing_content was supplied
    covered: Optional[bool] = None
    similarity_score: Optional[float] = None


class GapSummary(BaseModel):
    """Aggregate coverage statistics across all sub-queries."""

    covered: int
    total: int
    coverage_percent: float
    covered_types: List[str]   # de-duplicated query types that are covered
    missing_types: List[str]   # de-duplicated query types with no covered sub-query


class FanOutResponse(BaseModel):
    target_query: str
    model_used: str
    total_sub_queries: int
    sub_queries: List[SubQuery]
    gap_summary: Optional[GapSummary] = None  # None when no existing_content


# ---------------------------------------------------------------------------
# Internal exceptions (not serialised — used for control flow)
# ---------------------------------------------------------------------------


class LLMUnavailableError(RuntimeError):
    """Raised when the LLM fails to return a valid response after all retries."""
