import pytest

from app.chunker import split_text


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
