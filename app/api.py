from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from uuid import uuid4
from collections.abc import Mapping
from typing import Any

from fastapi import Depends, FastAPI, HTTPException

from app.config import DEFAULT_TOP_K
from app.generator import MissingAPIKeyError
from app.logging_utils import RequestLog, log_request
from app.pipeline import RAGPipeline
from app.pipeline_factory import VectorStoreNotReadyError, build_default_pipeline
from app.schemas import AskRequest, AskResponse, HealthResponse, Source


API_VERSION = "0.1.0"
SOURCE_FIELDS = ("source", "file_path", "filename")
TEXT_FIELDS = ("text", "document", "content", "page_content")
TEXT_PREVIEW_CHARS = 200

logger = logging.getLogger(__name__)
app = FastAPI(
    title="mini-rag API",
    description="FastAPI service for the mini-rag project",
    version=API_VERSION,
)
_pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    try:
        _pipeline = build_default_pipeline(top_k=DEFAULT_TOP_K)
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
    request: AskRequest,
    pipeline: RAGPipeline = Depends(get_pipeline),
) -> AskResponse:
    resolved_top_k = DEFAULT_TOP_K if request.top_k is None else request.top_k
    started_at = time.perf_counter()
    log_entry: RequestLog = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_id": str(uuid4()),
        "question": request.question,
        "answer": "",
        "sources": [],
        "latency_ms": 0,
        "status": "error",
        "error_type": None,
    }

    try:
        result = pipeline.ask(request.question, top_k=resolved_top_k)
        response_sources = normalize_sources(result)
        answer = _display_value(_get_value(result, "answer")) or ""
        latency_ms = int(round((time.perf_counter() - started_at) * 1000))
        response = AskResponse(
            question=_display_value(_get_value(result, "question")) or request.question,
            answer=answer,
            sources=response_sources,
            latency_ms=latency_ms,
        )
        log_entry["answer"] = answer
        log_entry["sources"] = [
            source.source for source in response_sources if source.source is not None
        ]
        log_entry["status"] = "success"
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
