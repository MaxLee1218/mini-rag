from __future__ import annotations

from pathlib import Path
from typing import Any


class VectorStoreNotReadyError(RuntimeError):
    """Raised when the local vector database is missing or empty."""


def build_default_pipeline(top_k: int | None = None) -> Any:
    """Build the default local RAG pipeline without doing work at import time."""
    from app.config import DEFAULT_TOP_K, VECTOR_COLLECTION_NAME, VECTOR_DB_PATH
    from app.embeddings import Embedder
    from app.generator import DeepSeekGenerator, load_deepseek_config_from_env
    from app.pipeline import RAGPipeline
    from app.retriever import Retriever
    from app.vector_store import ChromaVectorStore

    resolved_top_k = DEFAULT_TOP_K if top_k is None else _validate_top_k(top_k)
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

        retriever = Retriever(
            embedder=embedder,
            vector_store=vector_store,
            default_top_k=resolved_top_k,
        )
        generator = DeepSeekGenerator(config=config)
        return RAGPipeline(retriever=retriever, generator=generator)
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
