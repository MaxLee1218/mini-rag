import pytest

from app.chunker import split_document, split_documents, split_text


def test_split_text_hard_splits_plain_text_in_order():
    chunks = split_text("abcdefghijklmnop", chunk_size=6, chunk_overlap=0)

    assert chunks == ["abcdef", "ghijkl", "mnop"]


def test_split_text_returns_empty_list_for_empty_or_whitespace_text():
    assert split_text("") == []
    assert split_text(" \n\t  ") == []


def test_split_text_returns_single_chunk_when_text_is_shorter_than_chunk_size():
    assert split_text("short text", chunk_size=50, chunk_overlap=10) == ["short text"]


def test_split_text_strips_full_text_before_splitting():
    chunks = split_text("   abcdefghij   ", chunk_size=5, chunk_overlap=0)

    assert chunks == ["abcde", "fghij"]


def test_split_text_prioritizes_paragraph_boundaries_when_chunks_would_overflow():
    text = "段落一。\n\n段落二。\n\n段落三。"

    chunks = split_text(text, chunk_size=7, chunk_overlap=0)

    assert chunks == ["段落一。", "段落二。", "段落三。"]
    assert all(len(chunk) <= 7 for chunk in chunks)


def test_split_text_merges_small_splits_without_losing_newlines_or_punctuation():
    text = "第一句。\n第二句。\n第三句。"

    chunks = split_text(text, chunk_size=10, chunk_overlap=0)

    assert chunks == ["第一句。\n第二句。", "第三句。"]
    assert all(len(chunk) <= 10 for chunk in chunks)


def test_split_text_preserves_sentence_punctuation():
    assert split_text("第一句。第二句。", chunk_size=4, chunk_overlap=0) == [
        "第一句。",
        "第二句。",
    ]


def test_split_text_preserves_spaces_between_english_words():
    chunks = split_text(
        "machine learning model pipeline",
        chunk_size=24,
        chunk_overlap=0,
        separators=[" "],
    )

    assert chunks == ["machine learning model", "pipeline"]
    assert "machinelearning" not in "".join(chunks)


def test_split_text_recursively_falls_back_for_an_oversized_paragraph():
    text = "第一句。第二句。第三句。第四句。"

    chunks = split_text(text, chunk_size=4, chunk_overlap=0)

    assert chunks == ["第一句。", "第二句。", "第三句。", "第四句。"]
    assert all(len(chunk) <= 4 for chunk in chunks)


def test_split_text_hard_splits_long_text_without_matching_separators():
    chunks = split_text("a" * 32, chunk_size=10, chunk_overlap=0)

    assert chunks == ["a" * 10, "a" * 10, "a" * 10, "a" * 2]
    assert all(len(chunk) <= 10 for chunk in chunks)


def test_split_text_appends_hard_split_fallback_to_custom_separators():
    chunks = split_text(
        "a" * 16,
        chunk_size=5,
        chunk_overlap=0,
        separators=["|"],
    )

    assert chunks == ["a" * 5, "a" * 5, "a" * 5, "a"]
    assert all(len(chunk) <= 5 for chunk in chunks)


def test_split_text_uses_hard_split_when_custom_separators_are_empty():
    assert split_text("abcdefgh", chunk_size=3, chunk_overlap=0, separators=[]) == [
        "abc",
        "def",
        "gh",
    ]


@pytest.mark.parametrize("chunk_size", [0, -1])
def test_split_text_rejects_non_positive_chunk_size(chunk_size):
    with pytest.raises(ValueError):
        split_text("text", chunk_size=chunk_size)


@pytest.mark.parametrize("chunk_overlap", [-1, 10, 11])
def test_split_text_rejects_invalid_chunk_overlap(chunk_overlap):
    with pytest.raises(ValueError):
        split_text("text", chunk_size=10, chunk_overlap=chunk_overlap)


@pytest.mark.parametrize("separators", ["\n", {"\n"}, ["\n", 1], ("\n", None)])
def test_split_text_rejects_invalid_separators(separators):
    with pytest.raises((TypeError, ValueError)):
        split_text("text", separators=separators)


def test_split_document_preserves_key_fields_and_copies_metadata():
    document = {
        "content": "abcdefghij",
        "source": "folder/example.md",
        "metadata": {"category": "guide", "chunk_id": "legacy"},
    }

    chunks = split_document(document, chunk_size=5, chunk_overlap=0)

    assert document["metadata"] == {"category": "guide", "chunk_id": "legacy"}
    assert [chunk["content"] for chunk in chunks] == ["abcde", "fghij"]
    assert [chunk["source"] for chunk in chunks] == ["folder/example.md"] * 2
    assert [chunk["chunk_id"] for chunk in chunks] == [0, 1]
    assert chunks[0]["metadata"] == {
        "category": "guide",
        "original_chunk_id": "legacy",
        "chunk_id": 0,
    }
    assert chunks[1]["metadata"]["chunk_id"] == 1
    assert chunks[0]["metadata"] is not document["metadata"]
    assert chunks[0]["metadata"] is not chunks[1]["metadata"]


def test_split_document_handles_legacy_text_field_without_adding_metadata():
    chunks = split_document(
        {"text": "abcdefghijkl", "source": "example.md"},
        chunk_size=6,
        chunk_overlap=0,
    )

    assert [chunk["content"] for chunk in chunks] == ["abcdef", "ghijkl"]
    assert [chunk["source"] for chunk in chunks] == ["example.md", "example.md"]
    assert [chunk["chunk_id"] for chunk in chunks] == [0, 1]
    assert all("metadata" not in chunk for chunk in chunks)


def test_split_document_prefers_content_over_text():
    chunks = split_document(
        {"content": "content wins", "text": "text loses", "source": "example.md"},
        chunk_size=50,
        chunk_overlap=0,
    )

    assert len(chunks) == 1
    assert chunks[0]["content"] == "content wins"
    assert chunks[0]["source"] == "example.md"
    assert chunks[0]["chunk_id"] == 0


def test_split_documents_restarts_chunk_ids_and_does_not_mutate_inputs():
    documents = [
        {"content": "abcdefghijkl", "source": "first.md", "metadata": {"kind": "a"}},
        {"content": "mnopqrstuvwx", "source": "second.md", "metadata": {"kind": "b"}},
    ]

    chunks = split_documents(documents, chunk_size=6, chunk_overlap=0)

    assert [chunk["content"] for chunk in chunks] == ["abcdef", "ghijkl", "mnopqr", "stuvwx"]
    assert [chunk["source"] for chunk in chunks] == ["first.md", "first.md", "second.md", "second.md"]
    assert [chunk["chunk_id"] for chunk in chunks] == [0, 1, 0, 1]
    assert documents[0]["metadata"] == {"kind": "a"}
    assert documents[1]["metadata"] == {"kind": "b"}
    assert chunks[0]["metadata"] == {"kind": "a", "chunk_id": 0}
    assert chunks[2]["metadata"] == {"kind": "b", "chunk_id": 0}


def test_split_document_raises_value_error_when_content_and_text_are_missing():
    with pytest.raises(ValueError, match="document must contain 'content' or 'text'"):
        split_document({"source": "example.md"})


def test_split_document_raises_value_error_when_source_is_missing():
    with pytest.raises(ValueError, match="document must contain 'source'"):
        split_document({"content": "text"})


def test_split_document_rejects_non_dict_metadata():
    with pytest.raises(ValueError, match="metadata must be a dict"):
        split_document({"content": "text", "source": "example.md", "metadata": "bad"})
