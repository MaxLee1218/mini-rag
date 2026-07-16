"""Validated JSON contract for offline RAG badcases."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


def _required_string(data: Mapping[str, object], field: str) -> str:
    value = data.get(field)
    if not isinstance(value, str):
        raise ValueError(f"badcase {field} must be a string")
    return value


def _optional_string(data: Mapping[str, object], field: str) -> str | None:
    value = data.get(field)
    if value is not None and not isinstance(value, str):
        raise ValueError(f"badcase {field} must be a string or null")
    return value


def _string_list(data: Mapping[str, object], field: str) -> list[str]:
    value = data.get(field)
    if not isinstance(value, list) or not all(
        isinstance(item, str) for item in value
    ):
        raise ValueError(f"badcase {field} must be an array of strings")
    return list(value)


@dataclass(frozen=True)
class BadCase:
    """A suspected RAG failure plus optional human analysis."""

    question: str
    answer: str
    contexts: list[str]
    sources: list[str]
    error_type: str
    timestamp: str
    expected_answer: str | None = None
    root_cause: str | None = None
    solution: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Return a stable, JSON-serializable representation."""
        return {
            "question": self.question,
            "answer": self.answer,
            "expected_answer": self.expected_answer,
            "contexts": list(self.contexts),
            "sources": list(self.sources),
            "error_type": self.error_type,
            "root_cause": self.root_cause,
            "solution": self.solution,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> BadCase:
        """Build a badcase from a validated JSON object."""
        if not isinstance(data, Mapping):
            raise ValueError("badcase must be a JSON object")
        return cls(
            question=_required_string(data, "question"),
            answer=_required_string(data, "answer"),
            expected_answer=_optional_string(data, "expected_answer"),
            contexts=_string_list(data, "contexts"),
            sources=_string_list(data, "sources"),
            error_type=_required_string(data, "error_type"),
            root_cause=_optional_string(data, "root_cause"),
            solution=_optional_string(data, "solution"),
            timestamp=_required_string(data, "timestamp"),
        )
