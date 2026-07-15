from __future__ import annotations

import sys
from collections.abc import Mapping
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from evaluation.models import (
    EvaluationRecord,
    EvaluationSample,
    LatencyObservation,
)
from evaluation.ragas_evaluator import (
    CollectionsBindings,
    LegacyBindings,
    LocalRagasEmbeddings,
    RagasConfig,
    RagasEvaluator,
    default_ragas_config,
    _import_ragas_module,
)


METRIC_NAMES = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
)


@dataclass
class FakeMetric:
    values: list[object]

    def __post_init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def score(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        value = self.values.pop(0)
        if isinstance(value, Exception):
            raise value
        return SimpleNamespace(value=value)


class FakeLegacyResult:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._rows = rows

    def to_pandas(self) -> object:
        return SimpleNamespace(to_dict=lambda orient: self._rows)


def _config() -> RagasConfig:
    return RagasConfig(
        api_key="test-key",
        provider="deepseek",
        base_url="https://deepseek.test",
        model="test-judge",
        embedding_model="test-embedding",
        timeout_seconds=5.0,
    )


class FakeEmbedder:
    def embed_query(self, text: str) -> object:
        assert text == "one"
        return SimpleNamespace(tolist=lambda: [1.0, 2.0])

    def embed_texts(self, texts: list[str]) -> object:
        assert texts == ["one", "two"]
        return SimpleNamespace(tolist=lambda: [[1.0], [2.0]])


def test_local_ragas_embeddings_support_sync_and_async_interfaces() -> None:
    embeddings = LocalRagasEmbeddings(FakeEmbedder())

    assert embeddings.embed_text("one") == [1.0, 2.0]
    assert embeddings.embed_texts(["one", "two"]) == [[1.0], [2.0]]

    import asyncio

    assert asyncio.run(embeddings.aembed_text("one")) == [1.0, 2.0]
    assert asyncio.run(embeddings.aembed_texts(["one", "two"])) == [
        [1.0],
        [2.0],
    ]


def test_default_ragas_config_uses_deepseek_credentials(monkeypatch) -> None:
    import app.config as app_config

    monkeypatch.setattr(app_config, "EVALUATION_RAGAS_PROVIDER", "deepseek")
    monkeypatch.setattr(app_config, "EVALUATION_RAGAS_MODEL", "deepseek-judge")
    monkeypatch.setattr(app_config, "EVALUATION_RAGAS_EMBEDDING_MODEL", "local")
    monkeypatch.setattr(app_config, "EVALUATION_RAGAS_TIMEOUT", 9.0)
    monkeypatch.setattr(app_config, "DEEPSEEK_BASE_URL", "https://deepseek.test")
    monkeypatch.setattr(app_config, "require_deepseek_api_key", lambda: "ds-key")

    config = default_ragas_config()

    assert config.provider == "deepseek"
    assert config.api_key == "ds-key"
    assert config.base_url == "https://deepseek.test"


def _record(question: str = "q") -> EvaluationRecord:
    return EvaluationRecord(
        sample=EvaluationSample(
            question=question,
            ground_truth="reference",
            reference_contexts=("context",),
        ),
        answer="answer",
        contexts=["context"],
        sources=["source.txt"],
        route="rag",
        latency=LatencyObservation(1.0, 2.0, 3.0, 6.0),
        retrieval_hit=True,
        abstention_correct=None,
    )


def _collections(values: Mapping[str, list[object]]) -> CollectionsBindings:
    return CollectionsBindings(
        version="0.4.3",
        scorers={name: FakeMetric(list(values[name])) for name in METRIC_NAMES},
    )


def test_collections_api_populates_records_and_aggregates() -> None:
    bindings = _collections(
        {
            "faithfulness": [0.8, 1.0],
            "answer_relevancy": [0.7, 0.9],
            "context_precision": [0.6, 1.0],
            "context_recall": [0.5, 1.0],
        }
    )
    records = [_record("q1"), _record("q2")]

    summary = RagasEvaluator(
        _config(), bindings_loader=lambda config: bindings
    ).evaluate(records)

    assert records[0].ragas.faithfulness == 0.8
    assert records[1].ragas.context_recall == 1.0
    assert summary.metrics["faithfulness"] == {"score": 0.9, "count": 2}
    assert summary.status == "completed"
    assert summary.version == "0.4.3"
    assert summary.model == "test-judge"
    assert summary.embedding_model == "test-embedding"


def test_collections_api_uses_metric_specific_arguments() -> None:
    bindings = _collections({name: [1.0] for name in METRIC_NAMES})
    RagasEvaluator(
        _config(), bindings_loader=lambda config: bindings
    ).evaluate([_record()])

    scorers = bindings.scorers
    assert set(scorers["faithfulness"].calls[0]) == {
        "user_input",
        "response",
        "retrieved_contexts",
    }
    assert set(scorers["answer_relevancy"].calls[0]) == {
        "user_input",
        "response",
    }
    assert set(scorers["context_precision"].calls[0]) == {
        "user_input",
        "reference",
        "retrieved_contexts",
    }
    assert set(scorers["context_recall"].calls[0]) == {
        "user_input",
        "reference",
        "retrieved_contexts",
    }


def test_metric_failure_is_partial_and_does_not_fabricate_zero() -> None:
    values = {name: [0.8] for name in METRIC_NAMES}
    values["context_recall"] = [RuntimeError("private provider payload")]
    records = [_record()]

    summary = RagasEvaluator(
        _config(), bindings_loader=lambda config: _collections(values)
    ).evaluate(records)

    assert records[0].ragas.context_recall is None
    assert records[0].ragas.faithfulness == 0.8
    assert summary.metrics["context_recall"] == {"score": None, "count": 0}
    assert summary.status == "partial"
    assert "private provider payload" not in " ".join(summary.errors)
    assert records[0].ragas.errors == ["context_recall: RuntimeError"]


@pytest.mark.parametrize("bad_value", [True, float("nan"), float("inf"), -0.1, 1.1, "0.5"])
def test_invalid_metric_value_is_reported_as_partial(bad_value: object) -> None:
    values = {name: [0.8] for name in METRIC_NAMES}
    values["faithfulness"] = [bad_value]
    records = [_record()]

    summary = RagasEvaluator(
        _config(), bindings_loader=lambda config: _collections(values)
    ).evaluate(records)

    assert records[0].ragas.faithfulness is None
    assert summary.metrics["faithfulness"] == {"score": None, "count": 0}
    assert summary.status == "partial"


def test_missing_ragas_returns_unavailable_without_raising() -> None:
    def missing(config: RagasConfig) -> CollectionsBindings:
        raise ModuleNotFoundError("No module named 'ragas'", name="ragas")

    summary = RagasEvaluator(_config(), bindings_loader=missing).evaluate([_record()])

    assert summary.status == "unavailable"
    assert summary.metrics == {
        name: {"score": None, "count": 0} for name in METRIC_NAMES
    }
    assert "pip install ragas" in summary.errors[0]


def test_legacy_api_maps_each_metric_by_record_index() -> None:
    calls: list[tuple[object, list[object]]] = []
    metrics = {name: object() for name in METRIC_NAMES}

    def evaluate(dataset: object, *, metrics: list[object], **kwargs: object) -> object:
        calls.append((dataset, metrics))
        name = next(key for key, value in bindings.metrics.items() if value is metrics[0])
        return FakeLegacyResult([{name: 0.4}, {name: 0.8}])

    bindings = LegacyBindings(
        version="0.2.0",
        evaluate=evaluate,
        dataset_factory=lambda rows: rows,
        metrics=metrics,
        llm=object(),
        embeddings=object(),
    )
    records = [_record("q1"), _record("q2")]

    summary = RagasEvaluator(
        _config(), bindings_loader=lambda config: bindings
    ).evaluate(records)

    assert len(calls) == 4
    assert records[0].ragas.answer_relevancy == 0.4
    assert records[1].ragas.context_precision == 0.8
    assert summary.metrics["context_recall"] == {"score": 0.6, "count": 2}
    assert summary.status == "completed"


def test_known_vertexai_import_installs_only_local_type_shim(monkeypatch) -> None:
    module_name = "langchain_community.chat_models.vertexai"
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    returned = object()
    calls = 0

    def importer(name: str) -> object:
        nonlocal calls
        assert name == "ragas"
        calls += 1
        if calls == 1:
            error = ModuleNotFoundError(f"No module named '{module_name}'")
            error.name = module_name
            raise error
        assert hasattr(sys.modules[module_name], "ChatVertexAI")
        return returned

    assert _import_ragas_module(import_module=importer) is returned
    assert calls == 2


def test_unrelated_import_error_is_not_shimmed(monkeypatch) -> None:
    module_name = "langchain_community.chat_models.vertexai"
    monkeypatch.delitem(sys.modules, module_name, raising=False)

    def importer(name: str) -> object:
        error = ModuleNotFoundError("No module named 'different_dependency'")
        error.name = "different_dependency"
        raise error

    with pytest.raises(ModuleNotFoundError) as error_info:
        _import_ragas_module(import_module=importer)
    assert error_info.value.name == "different_dependency"
    assert module_name not in sys.modules
