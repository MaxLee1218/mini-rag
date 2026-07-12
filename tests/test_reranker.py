from types import SimpleNamespace

import numpy as np
import pytest
import torch

from app.reranker import CrossEncoderReranker


class FakeCrossEncoder:
    def __init__(self, scores=None, error=None):
        self.scores = [0.0] if scores is None else scores
        self.error = error
        self.calls = []

    def predict(self, pairs, **kwargs):
        self.calls.append((pairs, kwargs))
        if self.error:
            raise self.error
        return self.scores


class RecordingLoader:
    def __init__(self, model=None, error=None):
        self.model = model or FakeCrossEncoder()
        self.error = error
        self.calls = []

    def __call__(self, model_name_or_path, **kwargs):
        self.calls.append((model_name_or_path, kwargs))
        if self.error:
            raise self.error
        return self.model


def docs():
    return [
        {
            "id": "a",
            "text": "alpha",
            "score": 0.8,
            "metadata": {"source": "a.md"},
        },
        {
            "id": "b",
            "content": "beta",
            "score": 0.7,
            "metadata": {"source": "b.md"},
        },
        {
            "id": "c",
            "document": "gamma",
            "score": 0.6,
            "metadata": {"source": "c.md"},
        },
    ]


def make_reranker(scores, **kwargs):
    model = FakeCrossEncoder(scores)
    loader = RecordingLoader(model)
    return CrossEncoderReranker("model-id", model_loader=loader, **kwargs), model, loader


def test_empty_documents_return_without_loading():
    loader = RecordingLoader()
    reranker = CrossEncoderReranker("model-id", model_loader=loader)
    assert reranker.rerank("query", []) == []
    assert loader.calls == []


@pytest.mark.parametrize("query", [None, "", "   "])
def test_blank_query_raises(query):
    with pytest.raises(ValueError, match="query must not be blank"):
        CrossEncoderReranker("model-id", model_loader=RecordingLoader()).rerank(
            query, []
        )


@pytest.mark.parametrize("top_k", [0, -1, True, 1.5, "2"])
def test_invalid_top_k_raises(top_k):
    with pytest.raises(ValueError, match="top_k must be a positive integer or None"):
        CrossEncoderReranker("model-id", model_loader=RecordingLoader()).rerank("q", [], top_k)


def test_rejects_non_sequence_documents():
    with pytest.raises(ValueError, match="documents must be a sequence"):
        CrossEncoderReranker("model-id").rerank("q", {"text": "bad"})


def test_batches_correct_pairs_and_predict_options_once():
    reranker, model, _ = make_reranker([0.1, 0.3, 0.2], batch_size=7)
    reranker.rerank(" query ", docs())
    assert model.calls == [
        (
            [("query", "alpha"), ("query", "beta"), ("query", "gamma")],
            {"batch_size": 7, "show_progress_bar": False, "convert_to_numpy": True},
        )
    ]


def test_sorts_descending_stably_and_applies_top_k():
    reranker, _, _ = make_reranker([0.5, 0.8, 0.8])
    result = reranker.rerank("q", docs(), top_k=2)
    assert [item["id"] for item in result] == ["b", "c"]


def test_none_and_oversized_top_k_return_all():
    reranker, _, _ = make_reranker([0.3, 0.2, 0.1])
    assert len(reranker.rerank("q", docs(), top_k=None)) == 3
    assert len(reranker.rerank("q", docs(), top_k=99)) == 3


def test_preserves_fields_copies_metadata_and_does_not_mutate_input():
    original = docs()
    original[0]["dense_score"] = 0.4
    original[0]["sparse_score"] = 2.0
    reranker, _, _ = make_reranker([1.0, 0.0, -1.0])
    result = reranker.rerank("q", original)
    assert result[0]["score"] == 0.8
    assert result[0]["dense_score"] == 0.4
    assert result[0]["sparse_score"] == 2.0
    assert result[0]["metadata"] == {"source": "a.md"}
    assert result[0]["metadata"] is not original[0]["metadata"]
    assert result[0]["rerank_score"] == 1.0
    assert type(result[0]["rerank_score"]) is float
    assert all("rerank_score" not in item for item in original)


@pytest.mark.parametrize(
    "scores",
    [np.array([0.1, 0.2, 0.3], dtype=np.float32), torch.tensor([0.1, 0.2, 0.3])],
)
def test_numpy_and_torch_scores_become_python_floats(scores):
    reranker, _, _ = make_reranker(scores)
    result = reranker.rerank("q", docs())
    assert all(type(item["rerank_score"]) is float for item in result)


def test_blank_text_is_not_scored_and_is_ranked_last():
    values = docs()
    values.insert(1, {"id": "blank", "page_content": "  ", "score": 99.0})
    reranker, model, _ = make_reranker([0.1, 0.3, 0.2])
    result = reranker.rerank("q", values)
    assert len(model.calls[0][0]) == 3
    assert result[-1]["id"] == "blank"
    assert result[-1]["rerank_score"] == float("-inf")


def test_all_blank_text_preserves_order_without_loading():
    loader = RecordingLoader()
    reranker = CrossEncoderReranker("model-id", model_loader=loader)
    values = [{"id": "a", "text": None}, {"id": "b", "content": " "}]
    result = reranker.rerank("q", values, top_k=1)
    assert [item["id"] for item in result] == ["a"]
    assert loader.calls == []
    assert result[0]["rerank_score"] == float("-inf")


def test_nan_is_demoted_and_infinities_are_deterministic():
    reranker, _, _ = make_reranker([float("nan"), float("inf"), float("-inf")])
    result = reranker.rerank("q", docs())
    assert [item["id"] for item in result] == ["b", "a", "c"]
    assert result[1]["rerank_score"] == float("-inf")


@pytest.mark.parametrize("failure", ["count", "load", "predict"])
def test_model_runtime_failures_fallback_without_fabricated_scores(failure):
    if failure == "count":
        loader = RecordingLoader(FakeCrossEncoder([0.1]))
    elif failure == "load":
        loader = RecordingLoader(error=OSError("missing"))
    else:
        loader = RecordingLoader(FakeCrossEncoder(error=RuntimeError("predict")))
    reranker = CrossEncoderReranker("model-id", model_loader=loader)
    result = reranker.rerank("q", docs(), top_k=2)
    assert [item["id"] for item in result] == ["a", "b"]
    assert all("rerank_score" not in item for item in result)
    assert result[0]["score"] == 0.8


def test_model_loads_once_and_is_reused():
    reranker, _, loader = make_reranker([0.1, 0.2, 0.3])
    reranker.rerank("q", docs())
    reranker.rerank("q2", docs())
    assert len(loader.calls) == 1


def test_loader_receives_model_device_max_length_and_local_only():
    reranker, _, loader = make_reranker(
        [0.1, 0.2, 0.3], max_length=256, device="cpu", local_files_only=True
    )
    reranker.rerank("q", docs())
    assert loader.calls == [
        ("model-id", {"device": "cpu", "max_length": 256, "local_files_only": True})
    ]


def torch_state(cuda, mps_marker):
    backends = SimpleNamespace()
    if mps_marker is not None:
        backends.mps = SimpleNamespace(is_available=lambda: mps_marker)
    return SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: cuda), backends=backends
    )


@pytest.mark.parametrize(
    ("requested", "torch_module", "expected"),
    [
        ("auto", torch_state(True, True), "cuda"),
        ("auto", torch_state(False, True), "mps"),
        ("auto", torch_state(False, False), "cpu"),
        ("cuda", torch_state(False, True), "cpu"),
        ("mps", torch_state(False, False), "cpu"),
        ("auto", torch_state(False, None), "cpu"),
    ],
)
def test_device_selection(requested, torch_module, expected):
    reranker, _, loader = make_reranker(
        [0.1, 0.2, 0.3], device=requested, torch_module=torch_module
    )
    reranker.rerank("q", docs())
    assert loader.calls[0][1]["device"] == expected


def test_cpu_construction_does_not_touch_torch_or_load_model():
    class ExplodingTorch:
        def __getattr__(self, name):
            raise AssertionError(name)

    loader = RecordingLoader()
    CrossEncoderReranker(
        "model-id",
        device="cpu",
        torch_module=ExplodingTorch(),
        model_loader=loader,
    )
    assert loader.calls == []
