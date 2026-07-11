import pytest

from app.hybrid_retriever import HybridRetriever


class FakeRoute:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def retrieve(self, query, top_k=5):
        self.calls.append((query, top_k))
        return self.results[:top_k]


def record(item_id, score, source=None):
    return {
        "id": item_id,
        "text": f"text for {item_id}",
        "metadata": {"source": source or f"{item_id}.md"},
        "score": score,
    }


def test_retrieve_expands_candidates_merges_and_fuses_independent_scores():
    sparse = FakeRoute([record("shared", 10), record("sparse-only", 0)])
    dense = FakeRoute([record("dense-only", 0.9), record("shared", 0.7)])
    retriever = HybridRetriever(
        sparse_retriever=sparse,
        dense_retriever=dense,
        sparse_weight=0.4,
        dense_weight=0.6,
        candidate_multiplier=2,
    )

    results = retriever.retrieve(" question ", top_k=2)

    assert sparse.calls == [("question", 4)]
    assert dense.calls == [("question", 4)]
    assert [result["id"] for result in results] == ["dense-only", "shared"]
    shared = results[1]
    assert shared["sparse_score"] == 10
    assert shared["dense_score"] == 0.7
    assert shared["normalized_sparse_score"] == 1.0
    assert shared["normalized_dense_score"] == 0.0
    assert shared["score"] == pytest.approx(0.4)
    dense_only = results[0]
    assert dense_only["normalized_sparse_score"] == 0.0
    assert dense_only["normalized_dense_score"] == 1.0
    assert dense_only["score"] == pytest.approx(0.6)


def test_weights_are_normalized_and_equal_scores_remain_relevant():
    sparse = FakeRoute([record("b", 3), record("a", 3)])
    dense = FakeRoute([])
    retriever = HybridRetriever(sparse, dense, sparse_weight=4, dense_weight=6)

    results = retriever.retrieve("query", top_k=5)

    assert [result["id"] for result in results] == ["a", "b"]
    assert [result["score"] for result in results] == [0.4, 0.4]


def test_empty_routes_return_empty_results():
    retriever = HybridRetriever(FakeRoute([]), FakeRoute([]))

    assert retriever.retrieve("query") == []


@pytest.mark.parametrize("query", ["", "   ", None])
def test_retrieve_rejects_blank_query(query):
    retriever = HybridRetriever(FakeRoute([]), FakeRoute([]))

    with pytest.raises(ValueError, match="query must not be blank"):
        retriever.retrieve(query)


@pytest.mark.parametrize("top_k", [0, -1, True, 1.5])
def test_retrieve_rejects_invalid_top_k(top_k):
    retriever = HybridRetriever(FakeRoute([]), FakeRoute([]))

    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        retriever.retrieve("query", top_k=top_k)


@pytest.mark.parametrize(
    ("sparse_weight", "dense_weight"),
    [(-1, 1), (1, -1), (0, 0), (float("inf"), 1), ("1", 1), (True, 1)],
)
def test_constructor_rejects_invalid_weights(sparse_weight, dense_weight):
    with pytest.raises(ValueError, match="weights must be finite non-negative numbers"):
        HybridRetriever(
            FakeRoute([]),
            FakeRoute([]),
            sparse_weight=sparse_weight,
            dense_weight=dense_weight,
        )


@pytest.mark.parametrize("multiplier", [0, -1, True, 1.5])
def test_constructor_rejects_invalid_candidate_multiplier(multiplier):
    with pytest.raises(ValueError, match="candidate_multiplier must be a positive integer"):
        HybridRetriever(
            FakeRoute([]),
            FakeRoute([]),
            candidate_multiplier=multiplier,
        )


def test_retrieve_rejects_malformed_result_record():
    retriever = HybridRetriever(
        FakeRoute([{"text": "missing id", "metadata": {}, "score": 1.0}]),
        FakeRoute([]),
    )

    with pytest.raises(ValueError, match="result id must not be blank"):
        retriever.retrieve("query")
