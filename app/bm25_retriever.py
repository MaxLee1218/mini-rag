from __future__ import annotations

from hashlib import sha256
import re
from collections.abc import Mapping, Sequence
from typing import Any

from rank_bm25 import BM25Okapi


_TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    """Split Latin text into words and Chinese text into characters."""
    if not isinstance(text, str):
        return []
    return _TOKEN_PATTERN.findall(text.lower())


class BM25Retriever:
    """A standalone sparse retriever backed by a BM25 index."""

    def __init__(self, documents: Sequence[Mapping[str, Any]]):
        self.documents = list(documents)
        self.tokenized_documents = [
            tokenize(str(document.get("text", ""))) for document in self.documents
        ]
        index_documents = self.tokenized_documents or [["_empty_corpus_"]]
        self.bm25 = BM25Okapi(index_documents)

    def retrieve(
        self, query: str | None, top_k: int = 5
    ) -> list[dict[str, Any]]:
        query_tokens = tokenize(query) if query is not None else []
        if not self.documents or top_k <= 0 or not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)
        ranked_indices = sorted(
            range(len(self.documents)),
            key=lambda index: float(scores[index]),
            reverse=True,
        )

        return [
            {
                "id": self._stable_id(self.documents[index]),
                "text": self.documents[index].get("text", ""),
                "metadata": dict(self.documents[index].get("metadata") or {}),
                "score": float(scores[index]),
            }
            for index in ranked_indices[:top_k]
        ]

    def _stable_id(self, document: Mapping[str, Any]) -> str:
        document_id = document.get("id")
        if document_id is not None and str(document_id).strip():
            return str(document_id).strip()

        metadata = document.get("metadata") or {}
        if not isinstance(metadata, Mapping):
            metadata = {}
        source = str(metadata.get("source") or "").strip()
        chunk_id = metadata.get("chunk_id")
        if source and chunk_id is not None and str(chunk_id).strip():
            return f"{source}:{str(chunk_id).strip()}"

        text = str(document.get("text") or "")
        digest = sha256(f"{source}\0{text}".encode("utf-8")).hexdigest()
        return f"legacy:{digest}"
