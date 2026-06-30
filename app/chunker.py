def split_text(text: str, chunk_size: int = 500, chunk_overlap: int = 100) -> list[str]:
    """Split text into overlapping character chunks.

    Args:
        text: Text to split.
        chunk_size: Maximum number of characters in each chunk.
        chunk_overlap: Number of characters shared by neighboring chunks.

    Returns:
        A list of non-empty text chunks in original order.

    Raises:
        ValueError: If chunk_size is not positive, or chunk_overlap is negative
            or greater than or equal to chunk_size.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be greater than or equal to 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    text = text.strip()
    if not text:
        return []

    chunks = []
    step = chunk_size - chunk_overlap
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start += step

    return chunks


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
) -> list[dict]:
    if "source" not in document:
        raise ValueError("document must contain 'source'")

    text = _get_document_text(document)
    text_chunks = split_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    return [
        {
            "content": chunk,
            "source": document["source"],
            "chunk_id": chunk_id,
        }
        for chunk_id, chunk in enumerate(text_chunks)
    ]


def split_documents(
    documents: list[dict],
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> list[dict]:
    chunks = []

    for document in documents:
        chunks.extend(
            split_document(
                document,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        )

    return chunks
