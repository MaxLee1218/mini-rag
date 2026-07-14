from __future__ import annotations

import json
import logging
from hashlib import sha256
from numbers import Real
from typing import Any, Protocol

from app.faq.models import FAQMatch


logger = logging.getLogger(__name__)
_MATCH_TYPES = {"cache", "exact", "alias", "bm25"}


class FAQCache(Protocol):
    def get(
        self, normalized_question: str, index_version: int
    ) -> FAQMatch | None: ...

    def set(
        self,
        normalized_question: str,
        index_version: int,
        match: FAQMatch,
    ) -> None: ...


class NullFAQCache:
    def get(
        self, normalized_question: str, index_version: int
    ) -> FAQMatch | None:
        return None

    def set(
        self,
        normalized_question: str,
        index_version: int,
        match: FAQMatch,
    ) -> None:
        return None


class RedisFAQCache:
    """Positive FAQ cache whose failures never interrupt query routing."""

    def __init__(self, client: Any, *, ttl_seconds: int) -> None:
        if isinstance(ttl_seconds, bool) or not isinstance(ttl_seconds, int):
            raise ValueError("ttl_seconds must be a positive integer")
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be a positive integer")
        self.client = client
        self.ttl_seconds = ttl_seconds

    def get(
        self, normalized_question: str, index_version: int
    ) -> FAQMatch | None:
        key = faq_cache_key(normalized_question, index_version)
        try:
            value = self.client.get(key)
        except Exception as error:
            _warn("get", error)
            return None
        if value is None:
            return None
        try:
            if isinstance(value, bytes):
                value = value.decode("utf-8")
            payload = json.loads(value)
            return _match_from_payload(payload)
        except Exception as error:
            _warn("decode", error)
            self._delete(key)
            return None

    def set(
        self,
        normalized_question: str,
        index_version: int,
        match: FAQMatch,
    ) -> None:
        key = faq_cache_key(normalized_question, index_version)
        value = json.dumps(
            {
                "faq_id": match.faq_id,
                "question": match.question,
                "answer": match.answer,
                "source": match.source,
                "score": match.score,
                "match_type": match.match_type,
            },
            ensure_ascii=False,
        )
        try:
            self.client.setex(key, self.ttl_seconds, value)
        except Exception as error:
            _warn("set", error)

    def _delete(self, key: str) -> None:
        try:
            self.client.delete(key)
        except Exception as error:
            _warn("delete", error)


def faq_cache_key(normalized_question: str, index_version: int) -> str:
    if isinstance(index_version, bool) or not isinstance(index_version, int):
        raise ValueError("index_version must be a non-negative integer")
    if index_version < 0:
        raise ValueError("index_version must be a non-negative integer")
    digest = sha256(normalized_question.encode("utf-8")).hexdigest()
    return f"mini-rag:faq:v{index_version}:query:{digest}"


def _match_from_payload(payload: object) -> FAQMatch:
    if not isinstance(payload, dict):
        raise ValueError("cache payload must be an object")
    for field in ("faq_id", "question", "answer", "match_type"):
        if not isinstance(payload.get(field), str) or not payload[field].strip():
            raise ValueError(f"cache payload field {field} is invalid")
    source = payload.get("source")
    if source is not None and not isinstance(source, str):
        raise ValueError("cache payload source is invalid")
    score = payload.get("score")
    if isinstance(score, bool) or not isinstance(score, Real):
        raise ValueError("cache payload score is invalid")
    if payload["match_type"] not in _MATCH_TYPES:
        raise ValueError("cache payload match_type is invalid")
    return FAQMatch(
        faq_id=payload["faq_id"],
        question=payload["question"],
        answer=payload["answer"],
        source=source,
        score=float(score),
        match_type="cache",
    )


def _warn(operation: str, error: Exception) -> None:
    logger.warning(
        "faq_cache_operation_failed",
        extra={"operation": operation, "error_type": type(error).__name__},
    )
