from __future__ import annotations

from pathlib import Path
from typing import Any


class VectorStoreNotReadyError(RuntimeError):
    """Raised when the local vector database is missing or empty."""


def build_default_pipeline(top_k: int | None = None) -> Any:
    """Build the default local RAG pipeline without doing work at import time."""
    from app.bm25_retriever import BM25Retriever
    from app.config import (
        HYBRID_CANDIDATE_MULTIPLIER,
        HYBRID_DENSE_WEIGHT,
        HYBRID_SPARSE_WEIGHT,
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
    try:
        count = _get_vector_store_count(vector_store)
        if count is None or count == 0:
            raise VectorStoreNotReadyError("Please run scripts/ingest.py first.")

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
            candidate_k=RERANKER_CANDIDATE_K,
            final_top_k=resolved_top_k,
        )
    except Exception:
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
