from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from app.embeddings import Embedder
from app.vector_store import ChromaVectorStore


VALID_RETRIEVAL_MODES = ("standard", "parent-child")


class ParentChunkNotFoundError(RuntimeError):
    """Raised when a retrieved child references a missing parent."""


class Retriever:
    def __init__(
        self,
        embedder: Any | None = None,
        vector_store: Any | None = None,
        default_top_k: int = 5,
        mode: str = "standard",
        parent_store: Any | None = None,
    ):
        self.default_top_k = self._validate_top_k(
            default_top_k,
            field_name="default_top_k",
        )
        self.embedder = embedder if embedder is not None else Embedder()
        self.vector_store = (
            vector_store if vector_store is not None else ChromaVectorStore()
        )
        if mode not in VALID_RETRIEVAL_MODES:
            raise ValueError("mode must be one of: standard, parent-child")
        if mode == "parent-child" and parent_store is None:
            raise ValueError("parent_store is required in parent-child mode")
        self.mode = mode
        self.parent_store = parent_store

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
        query_where = (
            _merge_child_filter(where) if self.mode == "parent-child" else where
        )
        results = self.vector_store.query(
            query_embedding,
            top_k=resolved_top_k,
            where=query_where,
        )
        scored_results = [self._with_similarity_score(result) for result in results]
        if self.mode == "parent-child":
            return _resolve_parent_results(scored_results, self.parent_store)
        return scored_results

    def _with_similarity_score(self, result: Mapping[str, Any]) -> dict[str, Any]:
        copied_result = dict(result)
        if "distance" in copied_result:
            copied_result["score"] = 1.0 - float(copied_result["distance"])
        return copied_result

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

    def close(self) -> None:
        """Close injected persistence resources when they expose close()."""
        for resource in (self.parent_store, self.vector_store):
            close = getattr(resource, "close", None)
            if callable(close):
                close()

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


class ParentChildRetriever:
    """Resolve ranked child results from another retriever into parent contexts."""

    def __init__(
        self,
        child_retriever: Any,
        parent_store: Any,
        *,
        strict_parent_lookup: bool = True,
    ) -> None:
        if child_retriever is None:
            raise ValueError("child_retriever is required")
        if parent_store is None:
            raise ValueError("parent_store is required")
        self.child_retriever = child_retriever
        self.parent_store = parent_store
        self.strict_parent_lookup = strict_parent_lookup

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        children = self.child_retriever.retrieve(query, top_k=top_k)
        return _resolve_parent_results(
            list(children),
            self.parent_store,
            strict=self.strict_parent_lookup,
        )

    def build_context(
        self,
        results: list[dict[str, Any]],
        separator: str = "\n\n",
    ) -> str:
        blocks = []
        for result in results:
            metadata = result.get("metadata") or {}
            blocks.append(
                f"[parent_id: {result.get('id', 'unknown')} | "
                f"source: {metadata.get('source', 'unknown')}]\n{result.get('text', '')}"
            )
        return separator.join(blocks)

    def close(self) -> None:
        for resource in (self.parent_store, self.child_retriever):
            close = getattr(resource, "close", None)
            if callable(close):
                close()


def _merge_child_filter(where: Mapping[str, Any] | None) -> dict[str, Any]:
    child_filter = {"chunk_type": "child"}
    if where is None or not where:
        return child_filter
    return {"$and": [dict(where), child_filter]}


def _resolve_parent_results(
    children: list[Mapping[str, Any]],
    parent_store: Any,
    *,
    strict: bool = True,
) -> list[dict[str, Any]]:
    ordered_ids: list[str] = []
    first_child_by_parent: dict[str, Mapping[str, Any]] = {}
    for child in children:
        metadata = child.get("metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError(
                "child metadata must be a dictionary: "
                f"{child.get('id', 'unknown')}"
            )
        parent_id = metadata.get("parent_id")
        if not isinstance(parent_id, str) or not parent_id.strip():
            raise ValueError(
                "child metadata.parent_id must not be blank: "
                f"{child.get('id', 'unknown')}"
            )
        if parent_id not in first_child_by_parent:
            ordered_ids.append(parent_id)
            first_child_by_parent[parent_id] = child

    parents = parent_store.get_many(ordered_ids)
    parents_by_id = {str(parent.get("id")): parent for parent in parents}
    resolved: list[dict[str, Any]] = []
    for parent_id in ordered_ids:
        child = first_child_by_parent[parent_id]
        parent = parents_by_id.get(parent_id)
        if parent is None:
            if strict:
                metadata = child.get("metadata") or {}
                raise ParentChunkNotFoundError(
                    "missing parent_id "
                    f"{parent_id} for child_id {child.get('id', 'unknown')} "
                    f"source {metadata.get('source', 'unknown')}"
                )
            continue
        copied_parent = dict(parent)
        parent_metadata = dict(copied_parent.get("metadata") or {})
        parent_metadata.setdefault("parent_id", parent_id)
        copied_parent["metadata"] = parent_metadata
        child_score = child.get("score")
        if child_score is None and "distance" in child:
            child_score = 1.0 - float(child["distance"])
        copied_parent["retrieval"] = {
            "matched_child_id": child.get("id"),
            "matched_child_text": child.get("text"),
            "child_score": child_score,
        }
        if "distance" in child:
            copied_parent["distance"] = child["distance"]
        if child_score is not None:
            copied_parent["score"] = child_score
        resolved.append(copied_parent)
    return resolved
