from __future__ import annotations

import pytest

from app.chunker import split_document_parent_child


def test_parent_child_chunks_are_stable_linked_and_bounded():
    document = {
        "text": "第一部分讲 Python。\n\n第二部分讲 FastAPI。\n\n第三部分讲 RAG 父子块切分和重排序。",
        "source": "docs/example.txt",
    }

    first = split_document_parent_child(
        document,
        parent_chunk_size=28,
        child_chunk_size=12,
        parent_chunk_overlap=4,
        child_chunk_overlap=2,
    )
    second = split_document_parent_child(
        document,
        parent_chunk_size=28,
        child_chunk_size=12,
        parent_chunk_overlap=4,
        child_chunk_overlap=2,
    )

    assert first == second
    assert first.parents
    assert first.children
    parent_ids = {parent["id"] for parent in first.parents}
    assert all(parent["metadata"]["chunk_type"] == "parent" for parent in first.parents)
    assert all(len(parent["text"]) <= 28 for parent in first.parents)
    for child in first.children:
        assert child["metadata"]["parent_id"] in parent_ids
        assert child["metadata"]["chunk_type"] == "child"
        assert child["metadata"]["source"] == "docs/example.txt"
        assert len(child["text"]) <= 12


def test_parent_child_handles_short_blank_and_same_named_sources():
    short = split_document_parent_child(
        {"text": "短文", "source": "a/same.txt"},
        parent_chunk_size=100,
        child_chunk_size=20,
    )
    other = split_document_parent_child(
        {"text": "短文", "source": "b/same.txt"},
        parent_chunk_size=100,
        child_chunk_size=20,
    )
    blank = split_document_parent_child(
        {"text": " \n ", "source": "empty.txt"},
        parent_chunk_size=100,
        child_chunk_size=20,
    )

    assert len(short.parents) == len(short.children) == 1
    assert short.parents[0]["id"] != other.parents[0]["id"]
    assert blank.parents == []
    assert blank.children == []


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"parent_chunk_size": 0, "child_chunk_size": 1}, "parent_chunk_size"),
        ({"parent_chunk_size": 10, "child_chunk_size": 0}, "child_chunk_size"),
        ({"parent_chunk_size": 10, "child_chunk_size": 11}, "child_chunk_size"),
        (
            {"parent_chunk_size": 10, "child_chunk_size": 5, "parent_chunk_overlap": -1},
            "parent_chunk_overlap",
        ),
        (
            {"parent_chunk_size": 10, "child_chunk_size": 5, "parent_chunk_overlap": 10},
            "parent_chunk_overlap",
        ),
        (
            {"parent_chunk_size": 10, "child_chunk_size": 5, "child_chunk_overlap": 5},
            "child_chunk_overlap",
        ),
    ],
)
def test_parent_child_rejects_invalid_settings(kwargs, message):
    with pytest.raises(ValueError, match=message):
        split_document_parent_child({"text": "abc", "source": "a.txt"}, **kwargs)
