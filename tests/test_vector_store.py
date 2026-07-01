from numbers import Real
from uuid import uuid4

import pytest

from app.vector_store import ChromaVectorStore


def unique_collection_name() -> str:
    return f"test_{uuid4().hex}"


def make_store() -> ChromaVectorStore:
    return ChromaVectorStore(
        collection_name=unique_collection_name(),
        in_memory=True,
    )


def make_chunk(
    chunk_id: str = "chunk-a",
    text: str = "alpha text",
    embedding: list[float] | None = None,
    metadata: dict | None = None,
) -> dict:
    return {
        "id": chunk_id,
        "text": text,
        "metadata": metadata if metadata is not None else {"source": "alpha.md"},
        "embedding": embedding if embedding is not None else [1.0, 0.0, 0.0],
        "embedding_model": "fake-model",
        "embedding_dimension": 3,
    }


def make_chunks() -> list[dict]:
    return [
        make_chunk(
            chunk_id="chunk-a",
            text="alpha text",
            embedding=[1.0, 0.0, 0.0],
            metadata={"source": "alpha.md"},
        ),
        make_chunk(
            chunk_id="chunk-b",
            text="beta text",
            embedding=[0.0, 1.0, 0.0],
            metadata={"source": "beta.md"},
        ),
    ]


def test_vector_store_can_be_created():
    store = make_store()

    assert store.count() == 0


def test_add_chunks_upserts_records_and_count_returns_total():
    store = make_store()

    store.add_chunks(make_chunks())

    assert store.count() == 2


def test_add_chunks_updates_duplicate_ids_without_crashing():
    store = make_store()
    store.add_chunks([make_chunk(text="original text")])

    store.add_chunks([make_chunk(text="updated text")])
    results = store.query([1.0, 0.0, 0.0], top_k=1)

    assert store.count() == 1
    assert results[0]["id"] == "chunk-a"
    assert results[0]["text"] == "updated text"


def test_query_returns_clean_results_with_expected_top_result_and_key_order():
    store = make_store()
    store.add_chunks(make_chunks())

    results = store.query([1.0, 0.0, 0.0], top_k=2)

    assert len(results) == 2
    assert results[0]["id"] == "chunk-a"
    assert list(results[0].keys()) == ["id", "text", "metadata", "distance"]
    assert isinstance(results[0]["distance"], Real)


def test_query_supports_metadata_filter():
    store = make_store()
    store.add_chunks(make_chunks())

    results = store.query(
        [1.0, 0.0, 0.0],
        top_k=2,
        where={"source": "beta.md"},
    )

    assert len(results) == 1
    assert results[0]["id"] == "chunk-b"
    assert results[0]["metadata"]["source"] == "beta.md"


def test_embedding_metadata_is_stored_as_flat_metadata_values():
    store = make_store()
    store.add_chunks([make_chunk()])

    results = store.query([1.0, 0.0, 0.0], top_k=1)

    assert results[0]["metadata"]["embedding_model"] == "fake-model"
    assert results[0]["metadata"]["embedding_dimension"] == 3
    assert "embedding" not in results[0]["metadata"]


def test_add_chunks_rejects_empty_input():
    store = make_store()

    with pytest.raises(ValueError, match="chunks must not be empty"):
        store.add_chunks([])


def test_add_chunks_rejects_missing_embedding():
    store = make_store()
    chunk = make_chunk()
    del chunk["embedding"]

    with pytest.raises(ValueError, match="chunk must contain 'embedding'"):
        store.add_chunks([chunk])


def test_add_chunks_rejects_missing_id():
    store = make_store()
    chunk = make_chunk()
    del chunk["id"]

    with pytest.raises(ValueError, match="chunk must contain 'id'"):
        store.add_chunks([chunk])


def test_add_chunks_rejects_blank_id():
    store = make_store()

    with pytest.raises(ValueError, match="chunk id must not be blank"):
        store.add_chunks([make_chunk(chunk_id="   ")])


def test_add_chunks_rejects_missing_text():
    store = make_store()
    chunk = make_chunk()
    del chunk["text"]

    with pytest.raises(ValueError, match="chunk must contain 'text'"):
        store.add_chunks([chunk])


def test_add_chunks_rejects_blank_text():
    store = make_store()

    with pytest.raises(ValueError, match="chunk text must not be blank"):
        store.add_chunks([make_chunk(text="   ")])


def test_add_chunks_rejects_nested_metadata():
    store = make_store()

    with pytest.raises(ValueError, match="metadata value for 'tags'"):
        store.add_chunks([make_chunk(metadata={"tags": ["rag"]})])


def test_query_rejects_empty_query_embedding():
    store = make_store()

    with pytest.raises(ValueError, match="query_embedding must not be empty"):
        store.query([], top_k=1)


def test_query_rejects_non_positive_top_k():
    store = make_store()

    with pytest.raises(ValueError, match="top_k must be greater than 0"):
        store.query([1.0, 0.0, 0.0], top_k=0)


def test_clear_empties_records_and_keeps_store_usable():
    store = make_store()
    store.add_chunks(make_chunks())

    store.clear()
    store.add_chunks([make_chunk()])
    results = store.query([1.0, 0.0, 0.0], top_k=1)

    assert store.count() == 1
    assert results[0]["id"] == "chunk-a"


def test_clear_on_empty_collection_keeps_store_usable():
    store = make_store()

    store.clear()
    store.add_chunks([make_chunk()])

    assert store.count() == 1


def test_reset_recreates_collection_and_keeps_store_usable():
    store = make_store()
    store.add_chunks(make_chunks())

    store.reset()
    store.add_chunks([make_chunk()])
    results = store.query([1.0, 0.0, 0.0], top_k=1)

    assert store.count() == 1
    assert results[0]["id"] == "chunk-a"
