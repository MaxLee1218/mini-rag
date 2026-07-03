from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from typing import Any

import numpy as np


DEFAULT_EMBEDDING_MODEL = "models/paraphrase-multilingual-MiniLM-L12-v2"
TEXT_FIELDS = ("text", "content", "chunk")
OUTPUT_KEYS = (
    "id",
    "text",
    "metadata",
    "embedding",
    "embedding_model",
    "embedding_dimension",
)


class Embedder:
    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        batch_size: int = 32,
        normalize: bool = True,
        device: str | None = None,
        model: Any | None = None,
    ):
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must not be blank")
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")

        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = normalize
        self.device = device
        self._model = model
        self._dimension: int | None = None

    @property
    def model(self) -> Any:
        if self._model is None:
            self._model = self._load_model()
        return self._model

    @property
    def dimension(self) -> int:
        if self._dimension is not None:
            return self._dimension

        getter = getattr(self.model, "get_sentence_embedding_dimension", None)
        if callable(getter):
            dimension = getter()
            if dimension is not None:
                self._dimension = int(dimension)
                return self._dimension

        for attribute in ("dimension", "embedding_dimension"):
            dimension = getattr(self.model, attribute, None)
            if dimension is not None:
                self._dimension = int(dimension)
                return self._dimension

        sample_embedding = self.embed_texts(["dimension check"])
        self._dimension = int(sample_embedding.shape[1])
        return self._dimension

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        text_list = self._validate_texts(texts)
        embeddings = self.model.encode(
            text_list,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        embedding_array = np.asarray(embeddings, dtype=np.float32)

        if embedding_array.ndim == 1 and len(text_list) == 1:
            embedding_array = embedding_array.reshape(1, -1)
        if embedding_array.ndim != 2:
            raise ValueError("embedding model must return a 2D array")
        if embedding_array.shape[0] != len(text_list):
            raise ValueError("embedding count must match text count")

        self._dimension = int(embedding_array.shape[1])
        return embedding_array

    def embed_query(self, query: str) -> np.ndarray:
        if not isinstance(query, str):
            raise ValueError("query must be a string")
        return self.embed_texts([query])[0]

    def embed_chunks(self, chunks: Sequence[Any]) -> list[dict[str, Any]]:
        if isinstance(chunks, (str, bytes)):
            raise ValueError("chunks must be a sequence of chunk items")

        chunk_list = list(chunks)
        normalized_chunks = [
            self._normalize_chunk(chunk, index)
            for index, chunk in enumerate(chunk_list)
        ]
        embeddings = self.embed_texts(
            [chunk["text"] for chunk in normalized_chunks]
        )

        return [
            self._build_embedded_chunk(
                chunk=chunk,
                embedding=embeddings[index],
                embedding_dimension=int(embeddings.shape[1]),
            )
            for index, chunk in enumerate(normalized_chunks)
        ]

    def _load_model(self) -> Any:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise ImportError(
                "sentence-transformers is required to load the embedding model"
            ) from error

        if self.device is None:
            return SentenceTransformer(self.model_name)
        return SentenceTransformer(self.model_name, device=self.device)

    def _validate_texts(self, texts: Sequence[str]) -> list[str]:
        if isinstance(texts, (str, bytes)):
            raise ValueError("texts must be a sequence of strings")

        text_list = list(texts)
        if not text_list:
            raise ValueError("texts must not be empty")

        for text in text_list:
            if not isinstance(text, str):
                raise ValueError("all texts must be strings")
            if not text.strip():
                raise ValueError("texts must not contain blank strings")

        return text_list

    def _normalize_chunk(self, chunk: Any, index: int) -> dict[str, Any]:
        if isinstance(chunk, str):
            return {
                "id": self._default_chunk_id(index),
                "text": chunk,
                "metadata": {},
            }

        if is_dataclass(chunk) and not isinstance(chunk, type):
            return self._normalize_mapping(asdict(chunk), index)

        if isinstance(chunk, Mapping):
            return self._normalize_mapping(chunk, index)

        if hasattr(chunk, "_asdict"):
            return self._normalize_mapping(chunk._asdict(), index)

        if hasattr(chunk, "__dict__"):
            return self._normalize_mapping(vars(chunk), index)

        return self._normalize_attribute_chunk(chunk, index)

    def _normalize_mapping(
        self,
        chunk: Mapping[str, Any],
        index: int,
    ) -> dict[str, Any]:
        text_key = self._find_text_key(chunk)
        if text_key is None:
            raise ValueError("chunk must contain text, content, or chunk")

        text = chunk[text_key]
        if not isinstance(text, str):
            raise ValueError("chunk text must be a string")

        metadata = self._metadata_from_mapping(chunk)
        return {
            "id": self._chunk_id_from_mapping(chunk, index),
            "text": text,
            "metadata": metadata,
        }

    def _normalize_attribute_chunk(self, chunk: Any, index: int) -> dict[str, Any]:
        mapping: dict[str, Any] = {}

        for text_field in TEXT_FIELDS:
            if hasattr(chunk, text_field):
                mapping[text_field] = getattr(chunk, text_field)
                break

        for attribute in ("id", "metadata", "source", "chunk_id"):
            if hasattr(chunk, attribute):
                mapping[attribute] = getattr(chunk, attribute)

        if not mapping:
            raise ValueError("chunk must be a string, mapping, or object with text")

        return self._normalize_mapping(mapping, index)

    def _find_text_key(self, chunk: Mapping[str, Any]) -> str | None:
        for key in TEXT_FIELDS:
            if key in chunk:
                return key
        return None

    def _chunk_id_from_mapping(self, chunk: Mapping[str, Any], index: int) -> str:
        chunk_id = chunk.get("id")
        if chunk_id is None:
            return self._default_chunk_id(index)
        return str(chunk_id)

    def _metadata_from_mapping(self, chunk: Mapping[str, Any]) -> dict[str, Any]:
        metadata_value = chunk.get("metadata", {})
        if metadata_value is None:
            metadata: dict[str, Any] = {}
        elif isinstance(metadata_value, Mapping):
            metadata = dict(metadata_value)
        else:
            raise ValueError("chunk metadata must be a dictionary")

        ignored_keys = set(OUTPUT_KEYS) | {"content", "chunk"}
        for key, value in chunk.items():
            if key not in ignored_keys:
                metadata.setdefault(key, value)

        return metadata

    def _build_embedded_chunk(
        self,
        chunk: dict[str, Any],
        embedding: np.ndarray,
        embedding_dimension: int,
    ) -> dict[str, Any]:
        return {
            "id": chunk["id"],
            "text": chunk["text"],
            "metadata": chunk["metadata"],
            "embedding": embedding.tolist(),
            "embedding_model": self.model_name,
            "embedding_dimension": embedding_dimension,
        }

    def _default_chunk_id(self, index: int) -> str:
        return f"chunk-{index}"
