from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class AskRequest(BaseModel):
    question: str
    top_k: int | None = Field(default=None, ge=1, le=20)

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("question must not be blank")
        return value.strip()


class Source(BaseModel):
    index: int | None = None
    source: str | None = None
    metadata: dict[str, Any] | None = None
    text_preview: str | None = None


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[Source] = Field(default_factory=list)
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str | None = None


class ErrorResponse(BaseModel):
    detail: str

