from __future__ import annotations

import math
import tempfile
from collections.abc import Mapping, Sequence
from typing import Any

import chromadb
import numpy as np
from chromadb.config import Settings

from app.config import VECTOR_COLLECTION_NAME, VECTOR_DB_PATH


SIMPLE_METADATA_TYPES = (str, int, float, bool)
NUMERIC_EMBEDDING_TYPES = (int, float, np.integer, np.floating)


class ChromaVectorStore:
    def __init__(
        self,
        collection_name: str = VECTOR_COLLECTION_NAME,
        persist_path: str = VECTOR_DB_PATH,
        in_memory: bool = False,
    ):
        if not isinstance(collection_name, str) or not collection_name.strip():
            raise ValueError("collection_name must not be blank")

        self.collection_name = collection_name
        self.persist_path = persist_path
        self.in_memory = in_memory
        self._temp_dir: tempfile.TemporaryDirectory[str] | None = None
        self._closed = False
        self.client = self._create_client()
        self.collection = self._get_or_create_collection()

    def add_chunks(self, chunks: Sequence[Mapping[str, Any]]) -> None:
        if isinstance(chunks, (str, bytes)):
            raise ValueError("chunks must be a sequence of embedded chunk records")

        chunk_list = list(chunks)
        if not chunk_list:
            raise ValueError("chunks must not be empty")

        records = [self._prepare_chunk(chunk) for chunk in chunk_list]
        self.collection.upsert(
            ids=[record["id"] for record in records],
            documents=[record["text"] for record in records],
            metadatas=[record["metadata"] for record in records],
            embeddings=[record["embedding"] for record in records],
        )

    def query(
        self,
        query_embedding: Sequence[float],
        top_k: int = 5,
        where: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if not isinstance(top_k, int) or isinstance(top_k, bool):
            raise ValueError("top_k must be an integer")
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        embedding = self._normalize_embedding(
            query_embedding,
            field_name="query_embedding",
        )
        query_result = self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        return self._format_query_results(query_result)

    def count(self) -> int:
        return self.collection.count()

    def clear(self) -> None:
        result = self.collection.get(include=[])
        ids = result.get("ids", [])
        if ids:
            self.collection.delete(ids=ids)

    def reset(self) -> None:
        self.client.delete_collection(name=self.collection_name)
        self.collection = self._get_or_create_collection()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        close_client = getattr(self.client, "close", None)
        if callable(close_client):
            close_client()

        if self._temp_dir is not None:
            self._temp_dir.cleanup()
            self._temp_dir = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _create_client(self) -> Any:
        settings = Settings(
            anonymized_telemetry=False,
            chroma_api_impl="chromadb.api.segment.SegmentAPI",
        )
        if self.in_memory:
            self._temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
            return chromadb.PersistentClient(
                path=self._temp_dir.name,
                settings=settings,
            )
        return chromadb.PersistentClient(
            path=self.persist_path,
            settings=settings,
        )

    def _get_or_create_collection(self) -> Any:
        try:
            return self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=None,
            )
        except (TypeError, ValueError):
            return self.client.get_or_create_collection(name=self.collection_name)

    def _prepare_chunk(self, chunk: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(chunk, Mapping):
            raise ValueError("each chunk must be a dictionary")

        chunk_id = self._required_text_field(
            chunk,
            field_name="id",
            label="chunk id",
        )
        text = self._required_text_field(
            chunk,
            field_name="text",
            label="chunk text",
        )
        metadata = self._metadata_from_chunk(chunk)
        embedding = self._required_embedding(chunk)

        return {
            "id": chunk_id,
            "text": text,
            "metadata": metadata,
            "embedding": embedding,
        }

    def _required_text_field(
        self,
        chunk: Mapping[str, Any],
        field_name: str,
        label: str,
    ) -> str:
        if field_name not in chunk:
            raise ValueError(f"chunk must contain '{field_name}'")
        value = chunk[field_name]
        if not isinstance(value, str):
            raise ValueError(f"{label} must be a string")
        if not value.strip():
            raise ValueError(f"{label} must not be blank")
        return value

    def _required_embedding(self, chunk: Mapping[str, Any]) -> list[float]:
        if "embedding" not in chunk:
            raise ValueError("chunk must contain 'embedding'")
        return self._normalize_embedding(chunk["embedding"], field_name="embedding")

    def _metadata_from_chunk(self, chunk: Mapping[str, Any]) -> dict[str, Any]:
        for field_name in ("metadata", "embedding_model", "embedding_dimension"):
            if field_name not in chunk:
                raise ValueError(f"chunk must contain '{field_name}'")

        metadata_value = chunk["metadata"]
        if not isinstance(metadata_value, Mapping):
            raise ValueError("chunk metadata must be a dictionary")

        metadata = self._sanitize_metadata(metadata_value)
        embedding_model = chunk["embedding_model"]
        if not isinstance(embedding_model, str):
            raise ValueError("embedding_model must be a string")
        metadata["embedding_model"] = embedding_model
        metadata["embedding_dimension"] = self._normalize_embedding_dimension(
            chunk["embedding_dimension"]
        )
        return metadata

    def _sanitize_metadata(self, metadata: Mapping[str, Any]) -> dict[str, Any]:
        sanitized: dict[str, Any] = {}

        for key, value in metadata.items():
            key = str(key)
            if key == "embedding":
                continue
            if value is None:
                continue
            if isinstance(value, (list, dict, tuple, set)):
                raise ValueError(f"metadata value for '{key}' must be flat")
            if not isinstance(value, SIMPLE_METADATA_TYPES):
                raise ValueError(f"metadata value for '{key}' must be JSON-like")
            sanitized[key] = value

        return sanitized

    def _normalize_embedding_dimension(self, value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, NUMERIC_EMBEDDING_TYPES):
            raise ValueError("embedding_dimension must be an integer")
        dimension = int(value)
        if dimension <= 0:
            raise ValueError("embedding_dimension must be greater than 0")
        return dimension

    def _normalize_embedding(
        self,
        value: Any,
        field_name: str,
    ) -> list[float]:
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if isinstance(value, (str, bytes)):
            raise ValueError(f"{field_name} must be a sequence of numbers")

        try:
            values = list(value)
        except TypeError as error:
            raise ValueError(
                f"{field_name} must be a sequence of numbers"
            ) from error

        if not values:
            raise ValueError(f"{field_name} must not be empty")

        normalized = []
        for item in values:
            if isinstance(item, bool) or not isinstance(item, NUMERIC_EMBEDDING_TYPES):
                raise ValueError(f"{field_name} must contain only numbers")
            number = float(item)
            if math.isnan(number) or math.isinf(number):
                raise ValueError(f"{field_name} must contain only finite numbers")
            normalized.append(number)

        return normalized

    def _format_query_results(self, query_result: Mapping[str, Any]) -> list[dict[str, Any]]:
        ids = self._first_result_list(query_result, "ids")
        documents = self._first_result_list(query_result, "documents")
        metadatas = self._first_result_list(query_result, "metadatas")
        distances = self._first_result_list(query_result, "distances")

        results = []
        for index, item_id in enumerate(ids):
            results.append(
                {
                    "id": item_id,
                    "text": documents[index],
                    "metadata": metadatas[index] or {},
                    "distance": distances[index],
                }
            )
        return results

    def _first_result_list(
        self,
        query_result: Mapping[str, Any],
        key: str,
    ) -> list[Any]:
        values = query_result.get(key) or []
        if not values:
            return []
        return values[0] or []
