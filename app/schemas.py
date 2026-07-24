from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("question must not be blank")
        return value.strip()


class Source(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int | None = None
    source: str | None = None
    metadata: dict[str, JsonValue] | None = None
    text_preview: str | None = None


class Context(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content: str
    source: str | None = None
    chunk_id: str | None = None
    score: float | None = None
    metadata: dict[str, JsonValue] | None = None


class AskResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str
    sources: list[Source]
    contexts: list[Context]
    route: Literal["rag"]
    latency_ms: float = Field(ge=0)
    rag_trace_id: str = Field(min_length=1)


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    service: str
    version: str | None = None


class ErrorResponse(BaseModel):
    detail: str
