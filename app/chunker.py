from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any


DEFAULT_SEPARATORS = ["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]


@dataclass(frozen=True)
class ParentChildChunks:
    """Parent records and their searchable child records."""

    parents: list[dict[str, Any]]
    children: list[dict[str, Any]]


def _validate_chunk_settings(
    chunk_size: int,
    chunk_overlap: int,
    separators: list[str] | tuple[str, ...] | None,
) -> None:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be greater than or equal to 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    if separators is not None and not isinstance(separators, (list, tuple)):
        raise TypeError("separators must be a list or tuple of strings")
    if separators is not None and any(not isinstance(separator, str) for separator in separators):
        raise TypeError("separators must contain only strings")


def _normalize_separators(
    separators: list[str] | tuple[str, ...] | None,
) -> list[str]:
    if separators is None:
        return list(DEFAULT_SEPARATORS)

    normalized = list(separators)
    if "" not in normalized:
        normalized.append("")
    return normalized


def _hard_split_text(text: str, chunk_size: int) -> list[str]:
    return [text[start : start + chunk_size] for start in range(0, len(text), chunk_size)]


def _split_with_separator(text: str, separator: str, chunk_size: int) -> list[str]:
    if separator == "":
        return _hard_split_text(text, chunk_size)

    parts = text.split(separator)
    splits = []

    for index, part in enumerate(parts):
        if index < len(parts) - 1:
            part += separator
        if part:
            splits.append(part)

    return splits


def _recursive_split_text(
    text: str,
    chunk_size: int,
    separators: list[str],
) -> list[str]:
    if len(text) <= chunk_size:
        return [text]
    if not separators:
        return _hard_split_text(text, chunk_size)

    separator = separators[0]
    if separator == "":
        return _hard_split_text(text, chunk_size)

    splits = _split_with_separator(text, separator, chunk_size)
    if len(splits) <= 1:
        return _recursive_split_text(text, chunk_size, separators[1:])

    chunks = []
    for split in splits:
        if len(split) <= chunk_size:
            chunks.append(split)
        else:
            chunks.extend(_recursive_split_text(split, chunk_size, separators[1:]))

    return chunks


def _merge_splits(splits: list[str], chunk_size: int) -> list[str]:
    merged_chunks = []
    current_chunk = ""

    for split in splits:
        if not split.strip():
            continue

        candidate = current_chunk + split
        if len(candidate) <= chunk_size:
            current_chunk = candidate
            continue

        if current_chunk.strip():
            merged_chunks.append(current_chunk)
        current_chunk = split

    if current_chunk.strip():
        merged_chunks.append(current_chunk)

    return merged_chunks


def split_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    separators: list[str] | tuple[str, ...] | None = None,
) -> list[str]:
    """Split text with recursive separators and a final hard-split fallback.

    Args:
        text: Text to split.
        chunk_size: Maximum number of characters in each returned chunk.
        chunk_overlap: Retained and validated for compatibility with the old API.
            The recursive separator chunker does not actively create overlap.
        separators: Ordered separators to try. A final empty separator is always
            used internally as a hard-split fallback.

    Returns:
        Non-empty chunks in original order. Every chunk is at most chunk_size.

    Raises:
        ValueError: If chunk_size or chunk_overlap is invalid.
        TypeError: If separators is not a list or tuple of strings.
    """
    _validate_chunk_settings(chunk_size, chunk_overlap, separators)
    normalized_separators = _normalize_separators(separators)

    text = text.strip()
    if not text:
        return []

    splits = _recursive_split_text(text, chunk_size, normalized_separators)
    merged_chunks = _merge_splits(splits, chunk_size)
    final_chunks = []

    for chunk in merged_chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        if len(chunk) <= chunk_size:
            final_chunks.append(chunk)
        else:
            final_chunks.extend(_hard_split_text(chunk, chunk_size))

    assert all(len(chunk) <= chunk_size for chunk in final_chunks)
    return final_chunks


def _get_document_text(document: dict) -> str:
    if "content" in document:
        return document["content"]
    if "text" in document:
        return document["text"]
    raise ValueError("document must contain 'content' or 'text'")


def split_document(
    document: dict,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    separators: list[str] | tuple[str, ...] | None = None,
) -> list[dict]:
    """Split one document while preserving its source and metadata.

    chunk_overlap is retained and validated for compatibility with the old API.
    The recursive separator chunker does not actively generate overlap.
    """
    if "source" not in document:
        raise ValueError("document must contain 'source'")

    original_metadata = document.get("metadata")
    if original_metadata is not None and not isinstance(original_metadata, dict):
        raise ValueError("metadata must be a dict")

    text_chunks = split_text(
        _get_document_text(document),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
    )
    chunks = []

    for chunk_id, chunk_content in enumerate(text_chunks):
        chunk = {
            "content": chunk_content,
            "source": document["source"],
            "chunk_id": chunk_id,
        }

        if original_metadata is not None:
            copied_metadata = dict(original_metadata)
            if "chunk_id" in copied_metadata:
                copied_metadata["original_chunk_id"] = copied_metadata["chunk_id"]
            copied_metadata["chunk_id"] = chunk_id
            chunk["metadata"] = copied_metadata

        chunks.append(chunk)

    return chunks


def split_documents(
    documents: list[dict],
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    separators: list[str] | tuple[str, ...] | None = None,
) -> list[dict]:
    """Split documents in order, restarting chunk_id for each document.

    chunk_overlap is retained and validated for compatibility with the old API.
    The recursive separator chunker does not actively generate overlap.
    """
    chunks = []

    for document in documents:
        chunks.extend(
            split_document(
                document,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=separators,
            )
        )

    return chunks


def split_document_parent_child(
    document: dict[str, Any],
    parent_chunk_size: int,
    child_chunk_size: int,
    parent_chunk_overlap: int = 0,
    child_chunk_overlap: int = 0,
    separators: list[str] | tuple[str, ...] | None = None,
) -> ParentChildChunks:
    """Split a document into persisted parents and searchable children."""
    _validate_parent_child_settings(
        parent_chunk_size,
        child_chunk_size,
        parent_chunk_overlap,
        child_chunk_overlap,
    )
    if "source" not in document:
        raise ValueError("document must contain 'source'")
    source = str(document["source"])
    if not source.strip():
        raise ValueError("document source must not be blank")
    original_metadata = document.get("metadata")
    if original_metadata is not None and not isinstance(original_metadata, dict):
        raise ValueError("metadata must be a dict")

    text = _get_document_text(document)
    if not isinstance(text, str):
        raise ValueError("document text must be a string")
    document_id = _stable_hash(source)
    parent_texts = _split_text_with_overlap(
        text,
        chunk_size=parent_chunk_size,
        chunk_overlap=parent_chunk_overlap,
        separators=separators,
    )
    parents: list[dict[str, Any]] = []
    children: list[dict[str, Any]] = []

    for parent_index, parent_text in enumerate(parent_texts):
        parent_id = (
            f"parent::{document_id}::{parent_index}::{_stable_hash(parent_text)}"
        )
        parent_metadata = dict(original_metadata or {})
        parent_metadata.update(
            {
                "source": source,
                "chunk_type": "parent",
                "parent_index": parent_index,
                "document_id": document_id,
                "parent_id": parent_id,
            }
        )
        parents.append(
            {"id": parent_id, "text": parent_text, "metadata": parent_metadata}
        )

        child_texts = _split_text_with_overlap(
            parent_text,
            chunk_size=child_chunk_size,
            chunk_overlap=child_chunk_overlap,
            separators=separators,
        )
        for child_index, child_text in enumerate(child_texts):
            child_id = (
                f"child::{document_id}::{parent_index}::{child_index}::"
                f"{_stable_hash(child_text)}"
            )
            child_metadata = dict(original_metadata or {})
            child_metadata.update(
                {
                    "source": source,
                    "chunk_type": "child",
                    "parent_id": parent_id,
                    "parent_index": parent_index,
                    "child_index": child_index,
                    "document_id": document_id,
                }
            )
            children.append(
                {"id": child_id, "text": child_text, "metadata": child_metadata}
            )

    return ParentChildChunks(parents=parents, children=children)


def split_documents_parent_child(
    documents: list[dict[str, Any]],
    parent_chunk_size: int,
    child_chunk_size: int,
    parent_chunk_overlap: int = 0,
    child_chunk_overlap: int = 0,
    separators: list[str] | tuple[str, ...] | None = None,
) -> ParentChildChunks:
    """Split multiple documents while preserving their input order."""
    parents: list[dict[str, Any]] = []
    children: list[dict[str, Any]] = []
    for document in documents:
        split = split_document_parent_child(
            document,
            parent_chunk_size=parent_chunk_size,
            child_chunk_size=child_chunk_size,
            parent_chunk_overlap=parent_chunk_overlap,
            child_chunk_overlap=child_chunk_overlap,
            separators=separators,
        )
        parents.extend(split.parents)
        children.extend(split.children)
    return ParentChildChunks(parents=parents, children=children)


def _validate_parent_child_settings(
    parent_chunk_size: int,
    child_chunk_size: int,
    parent_chunk_overlap: int,
    child_chunk_overlap: int,
) -> None:
    if parent_chunk_size <= 0:
        raise ValueError("parent_chunk_size must be greater than 0")
    if child_chunk_size <= 0:
        raise ValueError("child_chunk_size must be greater than 0")
    if child_chunk_size > parent_chunk_size:
        raise ValueError(
            "child_chunk_size must be smaller than or equal to parent_chunk_size"
        )
    if parent_chunk_overlap < 0:
        raise ValueError("parent_chunk_overlap must be greater than or equal to 0")
    if parent_chunk_overlap >= parent_chunk_size:
        raise ValueError(
            "parent_chunk_overlap must be smaller than parent_chunk_size"
        )
    if child_chunk_overlap < 0:
        raise ValueError("child_chunk_overlap must be greater than or equal to 0")
    if child_chunk_overlap >= child_chunk_size:
        raise ValueError("child_chunk_overlap must be smaller than child_chunk_size")


def _split_text_with_overlap(
    text: str,
    *,
    chunk_size: int,
    chunk_overlap: int,
    separators: list[str] | tuple[str, ...] | None,
) -> list[str]:
    base_chunks = split_text(
        text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
    )
    if chunk_overlap == 0 or len(base_chunks) < 2:
        return base_chunks

    overlapped = [base_chunks[0]]
    for chunk in base_chunks[1:]:
        prefix = overlapped[-1][-chunk_overlap:]
        combined = f"{prefix}{chunk}"
        if len(combined) <= chunk_size:
            overlapped.append(combined)
            continue
        step = chunk_size - chunk_overlap
        start = 0
        while start < len(combined):
            candidate = combined[start : start + chunk_size].strip()
            if candidate and candidate != overlapped[-1]:
                overlapped.append(candidate)
            if start + chunk_size >= len(combined):
                break
            start += step
    return overlapped


def _stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
