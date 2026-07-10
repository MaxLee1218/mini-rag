DEFAULT_SEPARATORS = ["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]


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
