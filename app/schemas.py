from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class AskRequest(BaseModel):
    question: str
    session_id: str = Field(min_length=1, max_length=128)
    top_k: int | None = Field(default=None, ge=1, le=20)

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("question must not be blank")
        return value.strip()

    @field_validator("session_id")
    @classmethod
    def session_id_must_not_be_blank(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("session_id must not be blank")
        return value.strip()


class Source(BaseModel):
    index: int | None = None
    source: str | None = None
    metadata: dict[str, Any] | None = None
    text_preview: str | None = None


class AskResponse(BaseModel):
    question: str
    rewritten_query: str
    query_was_rewritten: bool
    answer: str
    sources: list[Source] = Field(default_factory=list)
    latency_ms: float
    session_id: str
    route: Literal["faq", "rag"] = "rag"
    faq_id: str | None = None
    faq_score: float | None = None
    faq_match_type: str | None = None
    faq_cache_hit: bool = False


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str | None = None


class ErrorResponse(BaseModel):
    detail: str
