from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from app.embeddings import Embedder
from app.vector_store import ChromaVectorStore


class Retriever:
    def __init__(
        self,
        embedder: Any | None = None,
        vector_store: Any | None = None,
        default_top_k: int = 5,
    ):
        self.default_top_k = self._validate_top_k(
            default_top_k,
            field_name="default_top_k",
        )
        self.embedder = embedder if embedder is not None else Embedder()
        self.vector_store = (
            vector_store if vector_store is not None else ChromaVectorStore()
        )

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        where: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        clean_query = self._clean_query(query)
        resolved_top_k = (
            self.default_top_k
            if top_k is None
            else self._validate_top_k(top_k, field_name="top_k")
        )
        query_embedding = self.embedder.embed_query(clean_query)
        return self.vector_store.query(
            query_embedding,
            top_k=resolved_top_k,
            where=where,
        )

    def retrieve_texts(
        self,
        query: str,
        top_k: int | None = None,
        where: Mapping[str, Any] | None = None,
    ) -> list[str]:
        results = self.retrieve(query, top_k=top_k, where=where)
        return [self._result_text(result) for result in results]

    def build_context(
        self,
        results: list[dict[str, Any]],
        separator: str = "\n\n",
    ) -> str:
        if not isinstance(results, list):
            raise ValueError("results must be a list")
        if not isinstance(separator, str):
            raise ValueError("separator must be a string")
        if not results:
            return ""

        blocks = [self._context_block(result) for result in results]
        return separator.join(blocks)

    def retrieve_context(
        self,
        query: str,
        top_k: int | None = None,
        where: Mapping[str, Any] | None = None,
        separator: str = "\n\n",
    ) -> str:
        results = self.retrieve(query, top_k=top_k, where=where)
        return self.build_context(results, separator=separator)

    def _clean_query(self, query: str) -> str:
        if not isinstance(query, str):
            raise ValueError("query must be a string")

        clean_query = query.strip()
        if not clean_query:
            raise ValueError("query must not be blank")

        return clean_query

    def _validate_top_k(self, top_k: int, field_name: str) -> int:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(f"{field_name} must be a positive integer")
        return top_k

    def _context_block(self, result: Mapping[str, Any]) -> str:
        text = self._result_text(result)
        chunk_id = self._result_id(result)
        source = self._result_source(result)

        if source is None:
            header = f"[chunk_id: {chunk_id}]"
        else:
            header = f"[chunk_id: {chunk_id} | source: {source}]"

        return f"{header}\n{text}"

    def _result_text(self, result: Mapping[str, Any]) -> str:
        if not isinstance(result, Mapping):
            raise ValueError("result must be a dictionary")

        text = result.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("result text must not be blank")
        return text

    def _result_id(self, result: Mapping[str, Any]) -> str:
        result_id = result.get("id")
        if result_id is None:
            return "unknown"

        result_id = str(result_id).strip()
        if not result_id:
            return "unknown"
        return result_id

    def _result_source(self, result: Mapping[str, Any]) -> str | None:
        metadata = result.get("metadata") or {}
        if not isinstance(metadata, Mapping):
            return None

        source = metadata.get("source")
        if source is None:
            return None

        source = str(source).strip()
        if not source:
            return None
        return source
