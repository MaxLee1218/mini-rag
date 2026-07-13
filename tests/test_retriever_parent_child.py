from __future__ import annotations

import pytest

from app.retriever import ParentChildRetriever, ParentChunkNotFoundError, Retriever


PARENTS = {
    "parent::1": {
        "id": "parent::1",
        "text": "局部关键词以及完整背景信息的长文本",
        "metadata": {"source": "docs/example.md", "chunk_type": "parent"},
    },
    "parent::2": {
        "id": "parent::2",
        "text": "第二个父块",
        "metadata": {"source": "docs/other.md", "chunk_type": "parent"},
    },
}


class FakeEmbedder:
    def embed_query(self, query):
        return [1.0]


class FakeVectorStore:
    def __init__(self, results):
        self.results = results
        self.where = None

    def query(self, query_embedding, top_k=5, where=None):
        self.where = where
        return self.results[:top_k]


class FakeParentStore:
    def __init__(self, parents=None):
        self.parents = PARENTS if parents is None else parents

    def get_many(self, parent_ids):
        return [self.parents[parent_id] for parent_id in parent_ids if parent_id in self.parents]


def child(child_id, parent_id, score=0.9):
    return {
        "id": child_id,
        "text": "局部关键词",
        "metadata": {
            "source": "docs/example.md",
            "chunk_type": "child",
            "parent_id": parent_id,
        },
        "score": score,
    }


def test_dense_parent_child_filter_merges_user_where():
    store = FakeVectorStore([child("child::1", "parent::1")])
    retriever = Retriever(FakeEmbedder(), store, mode="parent-child", parent_store=FakeParentStore())

    results = retriever.retrieve("question", where={"source": "docs/example.md"})

    assert store.where == {
        "$and": [
            {"source": "docs/example.md"},
            {"chunk_type": "child"},
        ]
    }
    assert results[0]["id"] == "parent::1"


def test_parent_child_deduplicates_in_first_child_rank_order_and_keeps_debug_data():
    child_retriever = type(
        "ChildRetriever",
        (),
        {"retrieve": lambda self, query, top_k=None: [
            child("child::a", "parent::1", 0.95),
            child("child::b", "parent::1", 0.90),
            child("child::c", "parent::2", 0.80),
        ]},
    )()
    retriever = ParentChildRetriever(child_retriever, FakeParentStore())

    results = retriever.retrieve("question", top_k=3)

    assert [result["id"] for result in results] == ["parent::1", "parent::2"]
    assert results[0]["text"] == PARENTS["parent::1"]["text"]
    assert results[0]["retrieval"] == {
        "matched_child_id": "child::a",
        "matched_child_text": "局部关键词",
        "child_score": 0.95,
    }


def test_parent_child_missing_parent_is_explicit_and_identifies_child():
    child_retriever = type(
        "ChildRetriever",
        (),
        {"retrieve": lambda self, query, top_k=None: [child("child::lost", "parent::missing")]},
    )()
    retriever = ParentChildRetriever(child_retriever, FakeParentStore({}))

    with pytest.raises(ParentChunkNotFoundError, match=r"parent::missing.*child::lost.*docs/example.md"):
        retriever.retrieve("question")


def test_standard_retriever_remains_backward_compatible():
    store = FakeVectorStore([
        {"id": "legacy", "text": "legacy text", "metadata": {"source": "old.md"}, "distance": 0.1}
    ])
    retriever = Retriever(FakeEmbedder(), store)

    assert retriever.retrieve("question")[0]["id"] == "legacy"
    assert store.where is None
