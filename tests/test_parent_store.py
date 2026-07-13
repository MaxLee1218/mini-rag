from __future__ import annotations

from app.parent_store import SQLiteParentStore


def make_parent(parent_id: str = "parent::one", text: str = "full parent") -> dict:
    return {
        "id": parent_id,
        "text": text,
        "metadata": {
            "source": "docs/example.md",
            "chunk_type": "parent",
            "document_id": "doc-1",
            "parent_index": 0,
        },
    }


def test_parent_store_persists_upserts_and_preserves_input_order(tmp_path):
    path = tmp_path / "parents.sqlite3"
    store = SQLiteParentStore(path)
    store.upsert([make_parent(), make_parent("parent::two", "second")])
    store.upsert([make_parent(text="updated parent")])
    store.close()

    reopened = SQLiteParentStore(path)
    assert reopened.count() == 2
    assert reopened.get("parent::one")["text"] == "updated parent"
    assert [item["id"] for item in reopened.get_many(["parent::two", "missing", "parent::one"])] == [
        "parent::two",
        "parent::one",
    ]
    reopened.reset()
    assert reopened.count() == 0
    reopened.close()


def test_parent_store_round_trip_links_every_child(tmp_path):
    from app.chunker import split_document_parent_child

    chunks = split_document_parent_child(
        {"text": "alpha beta gamma delta epsilon", "source": "docs/a.txt"},
        parent_chunk_size=18,
        child_chunk_size=8,
    )
    store = SQLiteParentStore(tmp_path / "parents.sqlite3")
    store.upsert(chunks.parents)

    for child in chunks.children:
        parent = store.get(child["metadata"]["parent_id"])
        assert parent is not None
        assert parent["id"] == child["metadata"]["parent_id"]
        assert parent["metadata"]["source"] == child["metadata"]["source"]
