from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)
_default_dual_path_pipeline: Any | None = None
_default_dual_path_lock = threading.Lock()


class VectorStoreNotReadyError(RuntimeError):
    """Raised when the local vector database is missing or empty."""


def build_default_dual_path_pipeline(top_k: int | None = None) -> Any:
    """Build FAQ routing without eagerly constructing the deep RAG path."""
    from app import config
    from app.dependencies import query_preparer
    from app.dual_path_pipeline import DualPathPipeline
    from app.faq.cache import NullFAQCache, RedisFAQCache
    from app.faq.matcher import FAQMatcher
    from app.faq.models import FAQMatch
    from app.faq.repository import FAQRepository
    from app.faq.text import normalize_question

    records = []
    index_version = 0
    if config.FAQ_ENABLED:
        try:
            repository = FAQRepository(config.FAQ_DB_PATH)
            repository.ensure_schema()
            records = repository.list_enabled()
            index_version = repository.get_index_version()
        except Exception as error:
            logger.exception(
                "faq_fast_path_initialization_failed",
                extra={"error_type": type(error).__name__},
            )
            records = []
            index_version = 0

    matcher = FAQMatcher(
        records,
        threshold=config.FAQ_MATCH_THRESHOLD,
        margin=config.FAQ_MATCH_MARGIN,
    )
    cache: Any = NullFAQCache()
    if config.FAQ_ENABLED and config.FAQ_CACHE_ENABLED:
        try:
            from redis import Redis

            client = Redis.from_url(
                config.REDIS_URL,
                socket_connect_timeout=config.REDIS_CONNECT_TIMEOUT_SECONDS,
                socket_timeout=config.REDIS_SOCKET_TIMEOUT_SECONDS,
                decode_responses=True,
            )
            cache = RedisFAQCache(
                client,
                ttl_seconds=config.FAQ_CACHE_TTL_SECONDS,
            )
        except Exception as error:
            logger.warning(
                "faq_cache_initialization_failed",
                extra={"error_type": type(error).__name__},
            )
            cache = NullFAQCache()

    if records and config.FAQ_CACHE_PREWARM:
        for record in records:
            surfaces = [
                (record.question, "exact"),
                *((alias, "alias") for alias in record.aliases),
            ]
            for surface, match_type in surfaces:
                cache.set(
                    normalize_question(surface),
                    index_version,
                    FAQMatch(
                        faq_id=record.id,
                        question=record.question,
                        answer=record.answer,
                        source=record.source,
                        score=1.0,
                        match_type=match_type,
                    ),
                )

    return DualPathPipeline(
        faq_matcher=matcher,
        faq_cache=cache,
        rag_pipeline_provider=_lazy_rag_pipeline_provider(top_k),
        faq_index_version=index_version,
        rag_query_preparer=query_preparer,
    )


def get_default_dual_path_pipeline(top_k: int | None = None) -> Any:
    """Return the process-wide immutable FAQ router and lazy RAG provider."""
    global _default_dual_path_pipeline
    if _default_dual_path_pipeline is not None:
        return _default_dual_path_pipeline
    with _default_dual_path_lock:
        if _default_dual_path_pipeline is None:
            _default_dual_path_pipeline = build_default_dual_path_pipeline(top_k)
    return _default_dual_path_pipeline


def reset_default_dual_path_pipeline() -> None:
    """Clear the process singleton for isolated tests."""
    global _default_dual_path_pipeline
    with _default_dual_path_lock:
        _default_dual_path_pipeline = None


def _lazy_rag_pipeline_provider(top_k: int | None) -> Any:
    pipeline: Any | None = None
    lock = threading.Lock()

    def provide() -> Any:
        nonlocal pipeline
        if pipeline is not None:
            return pipeline
        with lock:
            if pipeline is None:
                pipeline = build_default_pipeline(top_k)
        return pipeline

    return provide


def build_default_pipeline(top_k: int | None = None) -> Any:
    """Build the default local RAG pipeline without doing work at import time."""
    from app.bm25_retriever import BM25Retriever
    from app.config import (
        CHUNK_MODE,
        HYBRID_CANDIDATE_MULTIPLIER,
        HYBRID_DENSE_WEIGHT,
        HYBRID_SPARSE_WEIGHT,
        PARENT_STORE_PATH,
        RERANKER_BATCH_SIZE,
        RERANKER_CANDIDATE_K,
        RERANKER_DEVICE,
        RERANKER_ENABLED,
        RERANKER_FAILURE_MODE,
        RERANKER_LOCAL_FILES_ONLY,
        RERANKER_MAX_LENGTH,
        RERANKER_MODEL,
        RERANKER_TOP_K,
        VECTOR_COLLECTION_NAME,
        VECTOR_DB_PATH,
    )
    from app.embeddings import Embedder
    from app.generator import DeepSeekGenerator, load_deepseek_config_from_env
    from app.hybrid_retriever import HybridRetriever
    from app.pipeline import RAGPipeline
    from app.parent_store import SQLiteParentStore
    from app.retriever import Retriever
    from app.reranker import CrossEncoderReranker
    from app.vector_store import ChromaVectorStore

    resolved_top_k = RERANKER_TOP_K if top_k is None else _validate_top_k(top_k)
    config = load_deepseek_config_from_env()
    persist_path = _resolved_vector_db_path(VECTOR_DB_PATH)
    if not persist_path.exists():
        raise VectorStoreNotReadyError("Please run scripts/ingest.py first.")

    embedder = Embedder()
    vector_store = ChromaVectorStore(
        collection_name=VECTOR_COLLECTION_NAME,
        persist_path=str(persist_path),
    )
    parent_store = None
    try:
        count = _get_vector_store_count(vector_store)
        if count is None or count == 0:
            raise VectorStoreNotReadyError("Please run scripts/ingest.py first.")

        if CHUNK_MODE == "parent-child":
            parent_store_path = _resolved_project_path(PARENT_STORE_PATH)
            if not parent_store_path.is_file():
                raise VectorStoreNotReadyError(
                    "Parent store is missing. Please run parent-child ingest first."
                )
            parent_store = SQLiteParentStore(parent_store_path)
            parent_count = getattr(parent_store, "count", None)
            if callable(parent_count) and int(parent_count()) == 0:
                raise VectorStoreNotReadyError(
                    "Parent store is empty. Please run parent-child ingest first."
                )
            retriever = Retriever(
                embedder=embedder,
                vector_store=vector_store,
                default_top_k=resolved_top_k,
                mode="parent-child",
                parent_store=parent_store,
            )
        else:
            dense_retriever = Retriever(
                embedder=embedder,
                vector_store=vector_store,
                default_top_k=resolved_top_k,
            )
            sparse_retriever = BM25Retriever(_load_stored_documents(vector_store))
            retriever = HybridRetriever(
                sparse_retriever=sparse_retriever,
                dense_retriever=dense_retriever,
                sparse_weight=HYBRID_SPARSE_WEIGHT,
                dense_weight=HYBRID_DENSE_WEIGHT,
                candidate_multiplier=HYBRID_CANDIDATE_MULTIPLIER,
            )
        generator = DeepSeekGenerator(config=config)
        reranker = None
        if RERANKER_ENABLED:
            reranker = CrossEncoderReranker(
                RERANKER_MODEL,
                batch_size=RERANKER_BATCH_SIZE,
                max_length=RERANKER_MAX_LENGTH,
                device=RERANKER_DEVICE,
                failure_mode=RERANKER_FAILURE_MODE,
                local_files_only=RERANKER_LOCAL_FILES_ONLY,
            )
        return RAGPipeline(
            retriever=retriever,
            generator=generator,
            reranker=reranker,
            candidate_k=(
                resolved_top_k if CHUNK_MODE == "parent-child" else RERANKER_CANDIDATE_K
            ),
            final_top_k=resolved_top_k,
            expand_retrieval_candidates=CHUNK_MODE != "parent-child",
        )
    except Exception:
        close_parent_store = getattr(parent_store, "close", None)
        if callable(close_parent_store):
            close_parent_store()
        close = getattr(vector_store, "close", None)
        if callable(close):
            close()
        raise


def _validate_top_k(top_k: Any) -> int:
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    return top_k


def _get_vector_store_count(vector_store: Any) -> int | None:
    count = getattr(vector_store, "count", None)
    if callable(count):
        try:
            return int(count())
        except Exception as error:
            raise VectorStoreNotReadyError(
                "Please run scripts/ingest.py first."
            ) from error

    collection = getattr(vector_store, "collection", None)
    collection_count = getattr(collection, "count", None)
    if callable(collection_count):
        try:
            return int(collection_count())
        except Exception as error:
            raise VectorStoreNotReadyError(
                "Please run scripts/ingest.py first."
            ) from error

    return None


def _resolved_vector_db_path(path: str) -> Path:
    return _resolved_project_path(path)


def _resolved_project_path(path: str | Path) -> Path:
    resolved_path = Path(path)
    if resolved_path.is_absolute():
        return resolved_path
    return Path(__file__).resolve().parents[1] / resolved_path


def _load_stored_documents(vector_store: Any) -> list[dict[str, Any]]:
    stored = vector_store.collection.get(include=["documents", "metadatas"])
    ids = stored.get("ids") or []
    documents = stored.get("documents") or []
    metadatas = stored.get("metadatas") or []
    return [
        {
            "id": item_id,
            "text": documents[index],
            "metadata": metadatas[index] or {},
        }
        for index, item_id in enumerate(ids)
    ]
