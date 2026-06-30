import pytest

from app.chunker import split_document, split_documents, split_text


def test_split_text_splits_plain_text_in_order():
    text = "abcdefghijklmnop"

    chunks = split_text(text, chunk_size=6, chunk_overlap=0)

    assert chunks == ["abcdef", "ghijkl", "mnop"]


def test_split_text_returns_empty_list_for_empty_string():
    assert split_text("") == []


def test_split_text_returns_empty_list_for_whitespace_only_text():
    assert split_text(" \n\t  ") == []


def test_split_text_returns_single_chunk_when_text_is_shorter_than_chunk_size():
    assert split_text("short text", chunk_size=50, chunk_overlap=10) == ["short text"]


def test_split_text_strips_full_text_before_splitting():
    text = "   abcdefghij   "

    chunks = split_text(text, chunk_size=5, chunk_overlap=0)

    assert chunks == ["abcde", "fghij"]


def test_split_text_keeps_requested_overlap_between_chunks():
    text = "abcdefghijklmnopqrstuvwxyz"

    chunks = split_text(text, chunk_size=10, chunk_overlap=2)

    assert chunks == [
        "abcdefghij",
        "ijklmnopqr",
        "qrstuvwxyz",
    ]


def test_split_text_raises_value_error_when_chunk_size_is_zero():
    with pytest.raises(ValueError):
        split_text("text", chunk_size=0)


def test_split_text_raises_value_error_when_chunk_size_is_negative():
    with pytest.raises(ValueError):
        split_text("text", chunk_size=-1)


def test_split_text_raises_value_error_when_chunk_overlap_is_negative():
    with pytest.raises(ValueError):
        split_text("text", chunk_overlap=-1)


def test_split_text_raises_value_error_when_chunk_overlap_equals_chunk_size():
    with pytest.raises(ValueError):
        split_text("text", chunk_size=10, chunk_overlap=10)


def test_split_text_raises_value_error_when_chunk_overlap_is_larger_than_chunk_size():
    with pytest.raises(ValueError):
        split_text("text", chunk_size=10, chunk_overlap=11)


def test_split_document_handles_content_field_and_preserves_metadata():
    document = {
        "content": "abcdefghijklmnopqrstuvwxyz",
        "source": "folder/example.md",
    }

    chunks = split_document(document, chunk_size=10, chunk_overlap=0)

    assert chunks == [
        {"content": "abcdefghij", "source": "folder/example.md", "chunk_id": 0},
        {"content": "klmnopqrst", "source": "folder/example.md", "chunk_id": 1},
        {"content": "uvwxyz", "source": "folder/example.md", "chunk_id": 2},
    ]
    assert all(len(chunk["content"]) <= 10 for chunk in chunks)


def test_split_document_handles_legacy_text_field():
    document = {
        "text": "abcdefghijklmnop",
        "source": "example.md",
    }

    chunks = split_document(document, chunk_size=6, chunk_overlap=0)

    assert chunks == [
        {"content": "abcdef", "source": "example.md", "chunk_id": 0},
        {"content": "ghijkl", "source": "example.md", "chunk_id": 1},
        {"content": "mnop", "source": "example.md", "chunk_id": 2},
    ]


def test_split_document_prefers_content_over_text():
    document = {
        "content": "content wins",
        "text": "text loses",
        "source": "example.md",
    }

    chunks = split_document(document, chunk_size=50, chunk_overlap=0)

    assert chunks == [
        {"content": "content wins", "source": "example.md", "chunk_id": 0},
    ]


def test_split_documents_combines_documents_and_restarts_chunk_ids():
    documents = [
        {"content": "abcdefghijkl", "source": "first.md"},
        {"content": "mnopqrstuvwx", "source": "second.md"},
    ]

    chunks = split_documents(documents, chunk_size=6, chunk_overlap=0)

    assert chunks == [
        {"content": "abcdef", "source": "first.md", "chunk_id": 0},
        {"content": "ghijkl", "source": "first.md", "chunk_id": 1},
        {"content": "mnopqr", "source": "second.md", "chunk_id": 0},
        {"content": "stuvwx", "source": "second.md", "chunk_id": 1},
    ]


def test_split_document_raises_value_error_when_content_and_text_are_missing():
    with pytest.raises(ValueError, match="document must contain 'content' or 'text'"):
        split_document({"source": "example.md"})


def test_split_document_raises_value_error_when_source_is_missing():
    with pytest.raises(ValueError, match="document must contain 'source'"):
        split_document({"content": "text"})
