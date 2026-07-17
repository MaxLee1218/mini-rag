from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from uuid import uuid4
from collections.abc import Mapping
from typing import Any

from fastapi import Depends, FastAPI, HTTPException

from app.config import DEFAULT_TOP_K
from app.conversation.models import ConversationTurn
from app.conversation.store import ConversationStore
from app.dependencies import get_conversation_store
from app.generator import MissingAPIKeyError
from app.logging_utils import RequestLog, log_request
from app.pipeline_factory import (
    VectorStoreNotReadyError,
    get_default_dual_path_pipeline,
)
from app.schemas import AskRequest, AskResponse, HealthResponse, Source


API_VERSION = "0.1.0"
SOURCE_FIELDS = ("source", "file_path", "filename")
TEXT_FIELDS = ("text", "document", "content", "page_content")
TEXT_PREVIEW_CHARS = 200

logger = logging.getLogger(__name__)
app = FastAPI(
    title="Enterprise RAG Engine API",
    description="FastAPI service for the Enterprise RAG Engine project",
    version=API_VERSION,
)
_pipeline: Any | None = None


def get_pipeline() -> Any:
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    try:
        _pipeline = get_default_dual_path_pipeline(top_k=DEFAULT_TOP_K)
    except MissingAPIKeyError as error:
        logger.exception("DeepSeek API key is not configured.")
        raise HTTPException(
            status_code=500,
            detail="DeepSeek API key is not configured.",
        ) from error
    except VectorStoreNotReadyError as error:
        logger.exception("Vector store is not ready.")
        raise HTTPException(
            status_code=503,
            detail="Vector store is not ready. Please run scripts/ingest.py first.",
        ) from error
    except Exception as error:
        logger.exception("Failed to initialize RAG pipeline.")
        raise HTTPException(status_code=500, detail="RAG pipeline failed.") from error

    return _pipeline


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        service="mini-rag-api",
        version=API_VERSION,
    )


@app.post("/ask", response_model=AskResponse)
def ask(
    payload: AskRequest,
    pipeline: Any = Depends(get_pipeline),
    store: ConversationStore = Depends(get_conversation_store),
) -> AskResponse:
    resolved_top_k = DEFAULT_TOP_K if payload.top_k is None else payload.top_k
    original_question = payload.question
    rewritten_query = original_question
    query_was_rewritten = False
    rewrite_reason = "not_rewritten"
    history_turn_count = 0
    started_at = time.perf_counter()
    log_entry: RequestLog = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_id": str(uuid4()),
        "question": original_question[:500],
        "answer": "",
        "sources": [],
        "latency_ms": 0,
        "status": "error",
        "error_type": None,
        "session_id": payload.session_id,
        "original_question": original_question[:500],
        "rewritten_query": rewritten_query[:500],
        "query_was_rewritten": query_was_rewritten,
        "rewrite_reason": rewrite_reason,
        "history_turn_count": history_turn_count,
        "chunk_mode": _pipeline_chunk_mode(pipeline),
        "route": "rag",
        "faq_id": None,
        "faq_score": None,
        "faq_match_type": None,
        "faq_cache_hit": False,
    }

    try:
        result = pipeline.ask(
            original_question,
            top_k=resolved_top_k,
            session_id=payload.session_id,
        )
        rewritten_query = (
            _display_value(_get_value(result, "rewritten_query"))
            or original_question
        )
        query_was_rewritten = bool(
            _get_value(result, "query_was_rewritten") or False
        )
        rewrite_reason = (
            _display_value(_get_value(result, "rewrite_reason"))
            or "not_rewritten"
        )
        raw_history_count = _get_value(result, "history_turn_count")
        history_turn_count = (
            raw_history_count if isinstance(raw_history_count, int) else 0
        )
        response_sources = normalize_sources(result)
        answer = _display_value(_get_value(result, "answer")) or ""
        latency_ms = int(round((time.perf_counter() - started_at) * 1000))
        response = AskResponse(
            question=original_question,
            rewritten_query=rewritten_query,
            query_was_rewritten=query_was_rewritten,
            answer=answer,
            sources=response_sources,
            latency_ms=latency_ms,
            session_id=payload.session_id,
            route=_get_value(result, "route") or "rag",
            faq_id=_get_value(result, "faq_id"),
            faq_score=_get_value(result, "faq_score"),
            faq_match_type=_get_value(result, "faq_match_type"),
            faq_cache_hit=bool(_get_value(result, "faq_cache_hit") or False),
        )
        store.append_turn(
            payload.session_id,
            ConversationTurn(
                user_message=original_question,
                assistant_message=answer,
                created_at=datetime.now(timezone.utc),
            ),
        )
        log_entry["answer"] = answer
        log_entry["sources"] = [
            source.source for source in response_sources if source.source is not None
        ]
        log_entry["status"] = "success"
        log_entry["rewritten_query"] = rewritten_query[:500]
        log_entry["query_was_rewritten"] = query_was_rewritten
        log_entry["rewrite_reason"] = rewrite_reason
        log_entry["history_turn_count"] = history_turn_count
        log_entry["route"] = response.route
        log_entry["faq_id"] = response.faq_id
        log_entry["faq_score"] = response.faq_score
        log_entry["faq_match_type"] = response.faq_match_type
        log_entry["faq_cache_hit"] = response.faq_cache_hit
        return response
    except ValueError as error:
        log_entry["error_type"] = type(error).__name__
        logger.exception("Invalid RAG request.")
        raise HTTPException(status_code=400, detail="Invalid RAG request.") from error
    except MissingAPIKeyError as error:
        log_entry["error_type"] = type(error).__name__
        logger.exception("DeepSeek API key is not configured.")
        raise HTTPException(
            status_code=500,
            detail="DeepSeek API key is not configured.",
        ) from error
    except VectorStoreNotReadyError as error:
        log_entry["error_type"] = type(error).__name__
        logger.exception("Vector store is not ready.")
        raise HTTPException(
            status_code=503,
            detail="Vector store is not ready. Please run scripts/ingest.py first.",
        ) from error
    except Exception as error:
        log_entry["error_type"] = type(error).__name__
        logger.exception("RAG pipeline failed.")
        raise HTTPException(status_code=500, detail="RAG pipeline failed.") from error
    finally:
        log_entry["latency_ms"] = int(round((time.perf_counter() - started_at) * 1000))
        log_request(log_entry)


def normalize_sources(result: Any) -> list[Source]:
    source_items = _as_list(_get_value(result, "sources"))
    if not source_items:
        source_items = _as_list(_get_value(result, "contexts"))

    sources = []
    for index, item in enumerate(source_items, start=1):
        source = _source_from_item(item)
        metadata = _metadata_from_item(item)
        text = _text_from_item(item)
        preview = _preview_text(text, TEXT_PREVIEW_CHARS) if text else None
        sources.append(
            Source(
                index=index,
                source=source,
                metadata=metadata,
                text_preview=preview,
            )
        )
    return sources


def _source_from_item(item: Any) -> str | None:
    if isinstance(item, (str, bytes)):
        return _display_value(item)

    for field_name in SOURCE_FIELDS:
        value = _display_value(_get_value(item, field_name))
        if value:
            return value

    metadata = _metadata_from_item(item)
    if metadata:
        for field_name in SOURCE_FIELDS:
            value = _display_value(metadata.get(field_name))
            if value:
                return value
    return None


def _metadata_from_item(item: Any) -> dict[str, Any] | None:
    if isinstance(item, (str, bytes)):
        return None

    metadata = _get_value(item, "metadata")
    if isinstance(metadata, Mapping):
        return dict(metadata)
    return None


def _text_from_item(item: Any) -> str | None:
    if isinstance(item, (str, bytes)):
        return None

    for field_name in TEXT_FIELDS:
        value = _display_value(_get_value(item, field_name))
        if value:
            return value
    return None


def _get_value(item: Any, field_name: str) -> Any:
    if isinstance(item, Mapping):
        return item.get(field_name)
    return getattr(item, field_name, None)


def _pipeline_chunk_mode(pipeline: Any) -> str:
    """Return the effective retrieval mode without exposing retrieval internals."""
    retriever = getattr(pipeline, "retriever", None)
    mode = getattr(retriever, "mode", None)
    if mode in {"standard", "parent-child"}:
        return mode

    child_retriever = getattr(retriever, "child_retriever", None)
    child_mode = getattr(child_retriever, "mode", None)
    if child_mode in {"standard", "parent-child"}:
        return child_mode
    return "standard"


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, (str, bytes)):
        return [value]
    try:
        return list(value)
    except TypeError:
        return [value]


def _display_value(value: Any) -> str | None:
    if value is None or callable(value):
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        clean_value = value.strip()
        return clean_value or None
    return str(value)


def _preview_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars]
