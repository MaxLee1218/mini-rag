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
