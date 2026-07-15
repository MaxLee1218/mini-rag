"""Version-tolerant, failure-isolated RAGAS metric evaluation."""

from __future__ import annotations

import importlib
import inspect
import math
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Real
from types import ModuleType
from typing import Any

from evaluation.models import EvaluationRecord


METRIC_NAMES = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
)
_VERTEXAI_MODULE = "langchain_community.chat_models.vertexai"


@dataclass(frozen=True)
class RagasConfig:
    """Provider configuration used only by the offline RAGAS evaluator."""

    api_key: str
    provider: str
    base_url: str | None
    model: str
    embedding_model: str
    timeout_seconds: float


@dataclass(frozen=True)
class CollectionsBindings:
    """RAGAS collections scorers for the current per-sample API."""

    version: str
    scorers: Mapping[str, Any]
    api_kind: str = "collections"
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class LegacyBindings:
    """RAGAS batch functions and metrics for legacy releases."""

    version: str
    evaluate: Callable[..., Any]
    dataset_factory: Callable[[list[dict[str, Any]]], Any]
    metrics: Mapping[str, Any]
    llm: Any
    embeddings: Any
    api_kind: str = "legacy"
    warnings: tuple[str, ...] = ()


@dataclass
class RagasEvaluationSummary:
    """Aggregate RAGAS results and safe operational diagnostics."""

    status: str
    version: str | None
    model: str
    embedding_model: str
    metrics: dict[str, dict[str, float | int | None]]
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


Bindings = CollectionsBindings | LegacyBindings
BindingsLoader = Callable[[RagasConfig], Bindings]


class LocalRagasEmbeddings:
    """Adapt the project's local embedder to RAGAS' modern embedding contract."""

    def __init__(self, embedder: Any) -> None:
        self._embedder = embedder

    def embed_text(self, text: str, **kwargs: Any) -> list[float]:
        """Embed one text synchronously using the project embedder."""
        del kwargs
        return list(self._embedder.embed_query(text).tolist())

    def embed_texts(
        self,
        texts: list[str],
        **kwargs: Any,
    ) -> list[list[float]]:
        """Embed a text batch synchronously using one model invocation."""
        del kwargs
        return list(self._embedder.embed_texts(texts).tolist())

    async def aembed_text(self, text: str, **kwargs: Any) -> list[float]:
        """Expose RAGAS' async contract for the synchronous local model."""
        return self.embed_text(text, **kwargs)

    async def aembed_texts(
        self,
        texts: list[str],
        **kwargs: Any,
    ) -> list[list[float]]:
        """Expose RAGAS' async batch contract for the synchronous local model."""
        return self.embed_texts(texts, **kwargs)

    def embed_query(self, text: str) -> list[float]:
        """Expose the legacy LangChain single-query embedding contract."""
        return self.embed_text(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Expose the legacy LangChain document embedding contract."""
        return self.embed_texts(texts)

    async def aembed_query(self, text: str) -> list[float]:
        """Expose the legacy asynchronous single-query contract."""
        return await self.aembed_text(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """Expose the legacy asynchronous document embedding contract."""
        return await self.aembed_texts(texts)


class RagasEvaluator:
    """Evaluate records with RAGAS without aborting on individual failures."""

    def __init__(
        self,
        config: RagasConfig,
        *,
        bindings_loader: BindingsLoader = lambda config: load_ragas_bindings(config),
    ) -> None:
        self.config = config
        self.bindings_loader = bindings_loader

    def evaluate(
        self,
        records: Sequence[EvaluationRecord],
    ) -> RagasEvaluationSummary:
        """Populate per-record scores and return valid-score aggregates."""
        try:
            bindings = self.bindings_loader(self.config)
        except ModuleNotFoundError as error:
            return self._unavailable(_missing_dependency_error(error))
        except Exception as error:
            return self._unavailable(
                f"RAGAS initialization failed: {type(error).__name__}"
            )

        errors: list[str] = []
        if isinstance(bindings, CollectionsBindings):
            self._evaluate_collections(records, bindings, errors)
        else:
            self._evaluate_legacy(records, bindings, errors)

        metrics = _aggregate_metrics(records)
        return RagasEvaluationSummary(
            status="partial" if errors else "completed",
            version=bindings.version,
            model=self.config.model,
            embedding_model=self.config.embedding_model,
            metrics=metrics,
            warnings=list(bindings.warnings),
            errors=errors,
        )

    def _evaluate_collections(
        self,
        records: Sequence[EvaluationRecord],
        bindings: CollectionsBindings,
        errors: list[str],
    ) -> None:
        for index, record in enumerate(records):
            inputs = _metric_inputs(record)
            for metric_name in METRIC_NAMES:
                try:
                    scorer = bindings.scorers[metric_name]
                    result = scorer.score(**inputs[metric_name])
                    score = _validated_score(result, metric_name)
                except Exception as error:
                    _record_metric_error(record, metric_name, error, index, errors)
                    continue
                setattr(record.ragas, metric_name, score)

    def _evaluate_legacy(
        self,
        records: Sequence[EvaluationRecord],
        bindings: LegacyBindings,
        errors: list[str],
    ) -> None:
        rows = [
            {
                "question": record.sample.question,
                "answer": record.answer,
                "contexts": list(record.contexts),
                "ground_truth": record.sample.ground_truth,
            }
            for record in records
        ]
        for metric_name in METRIC_NAMES:
            try:
                dataset = bindings.dataset_factory(rows)
                result = _call_legacy_evaluate(
                    bindings,
                    dataset,
                    bindings.metrics[metric_name],
                )
                result_rows = _legacy_result_rows(result)
                if len(result_rows) != len(records):
                    raise ValueError("legacy result row count mismatch")
            except Exception as error:
                for index, record in enumerate(records):
                    _record_metric_error(
                        record,
                        metric_name,
                        error,
                        index,
                        errors,
                    )
                continue

            for index, (record, result_row) in enumerate(zip(records, result_rows)):
                try:
                    score = _validated_score(result_row, metric_name)
                except Exception as error:
                    _record_metric_error(
                        record,
                        metric_name,
                        error,
                        index,
                        errors,
                    )
                    continue
                setattr(record.ragas, metric_name, score)

    def _unavailable(self, error: str) -> RagasEvaluationSummary:
        return RagasEvaluationSummary(
            status="unavailable",
            version=None,
            model=self.config.model,
            embedding_model=self.config.embedding_model,
            metrics=_empty_metrics(),
            errors=[error],
        )


def default_ragas_config() -> RagasConfig:
    """Load RAGAS evaluator configuration through the project config module."""
    from app.config import (
        DEEPSEEK_BASE_URL,
        EVALUATION_RAGAS_PROVIDER,
        EVALUATION_RAGAS_EMBEDDING_MODEL,
        EVALUATION_RAGAS_MODEL,
        EVALUATION_RAGAS_TIMEOUT,
        require_deepseek_api_key,
        require_openai_api_key,
    )

    if EVALUATION_RAGAS_PROVIDER == "deepseek":
        api_key = require_deepseek_api_key()
        base_url: str | None = DEEPSEEK_BASE_URL
    else:
        api_key = require_openai_api_key()
        base_url = None
    return RagasConfig(
        api_key=api_key,
        provider=EVALUATION_RAGAS_PROVIDER,
        base_url=base_url,
        model=EVALUATION_RAGAS_MODEL,
        embedding_model=EVALUATION_RAGAS_EMBEDDING_MODEL,
        timeout_seconds=EVALUATION_RAGAS_TIMEOUT,
    )


def load_ragas_bindings(config: RagasConfig | None = None) -> Bindings:
    """Load current RAGAS scorers, falling back to the legacy evaluate API."""
    resolved_config = config or default_ragas_config()
    ragas_module = _import_ragas_module()
    version = str(getattr(ragas_module, "__version__", "unknown"))

    try:
        metrics_module = importlib.import_module("ragas.metrics.collections")
        llms_module = importlib.import_module("ragas.llms")
        openai_module = importlib.import_module("openai")
        client_kwargs: dict[str, Any] = dict(
            api_key=resolved_config.api_key,
            timeout=resolved_config.timeout_seconds,
        )
        if resolved_config.base_url is not None:
            client_kwargs["base_url"] = resolved_config.base_url
        client = openai_module.AsyncOpenAI(**client_kwargs)
        # DeepSeek exposes an OpenAI-compatible API. Using RAGAS' OpenAI
        # adapter selects JSON structured output while the client base URL
        # still routes every request to DeepSeek.
        llm = llms_module.llm_factory(
            resolved_config.model,
            provider="openai",
            client=client,
        )
        embeddings = _load_collections_embeddings(resolved_config, client)
        return CollectionsBindings(
            version=version,
            scorers={
                "faithfulness": metrics_module.Faithfulness(llm=llm),
                "answer_relevancy": metrics_module.AnswerRelevancy(
                    llm=llm,
                    embeddings=embeddings,
                ),
                "context_precision": metrics_module.ContextPrecision(llm=llm),
                "context_recall": metrics_module.ContextRecall(llm=llm),
            },
        )
    except (AttributeError, ImportError):
        return _load_legacy_bindings(resolved_config, version)


def _import_ragas_module(
    *,
    import_module: Callable[[str], Any] = importlib.import_module,
) -> Any:
    """Import RAGAS with a narrow shim for its removed VertexAI type import."""
    try:
        return import_module("ragas")
    except ModuleNotFoundError as error:
        if error.name != _VERTEXAI_MODULE:
            raise

    installed_shim = _VERTEXAI_MODULE not in sys.modules
    if installed_shim:
        shim = ModuleType(_VERTEXAI_MODULE)
        shim.ChatVertexAI = type("ChatVertexAI", (), {})
        sys.modules[_VERTEXAI_MODULE] = shim
    try:
        return import_module("ragas")
    except Exception:
        if installed_shim:
            sys.modules.pop(_VERTEXAI_MODULE, None)
        raise


def _load_legacy_bindings(config: RagasConfig, version: str) -> LegacyBindings:
    metrics_module = importlib.import_module("ragas.metrics")
    datasets_module = importlib.import_module("datasets")
    langchain_openai = importlib.import_module("langchain_openai")
    ragas_module = importlib.import_module("ragas")
    client_kwargs: dict[str, Any] = dict(
        api_key=config.api_key,
        model=config.model,
        timeout=config.timeout_seconds,
    )
    if config.base_url is not None:
        client_kwargs["base_url"] = config.base_url
    llm = langchain_openai.ChatOpenAI(**client_kwargs)
    if config.provider == "deepseek":
        embeddings = _local_embeddings(config.embedding_model)
    else:
        embeddings = langchain_openai.OpenAIEmbeddings(
            api_key=config.api_key,
            model=config.embedding_model,
            timeout=config.timeout_seconds,
        )
    return LegacyBindings(
        version=version,
        evaluate=ragas_module.evaluate,
        dataset_factory=datasets_module.Dataset.from_list,
        metrics={name: getattr(metrics_module, name) for name in METRIC_NAMES},
        llm=llm,
        embeddings=embeddings,
    )


def _load_collections_embeddings(config: RagasConfig, client: Any) -> Any:
    """Create provider embeddings without sending unsupported calls to DeepSeek."""
    if config.provider == "deepseek":
        return _modern_local_embeddings(config.embedding_model)
    embeddings_module = importlib.import_module("ragas.embeddings.base")
    return embeddings_module.embedding_factory(
        "openai",
        model=config.embedding_model,
        client=client,
        interface="modern",
    )


def _local_embeddings(model_name: str) -> LocalRagasEmbeddings:
    """Construct the lazily loaded project embedding adapter."""
    from app.embeddings import Embedder

    selected_model = None if model_name == "local" else model_name
    return LocalRagasEmbeddings(Embedder(model_name=selected_model))


def _modern_local_embeddings(model_name: str) -> LocalRagasEmbeddings:
    """Wrap local embeddings in RAGAS' lazily imported nominal base type."""
    embeddings_module = importlib.import_module("ragas.embeddings.base")
    base_class = embeddings_module.BaseRagasEmbedding

    class ModernLocalRagasEmbeddings(LocalRagasEmbeddings, base_class):
        """Runtime adapter satisfying RAGAS collections' nominal type check."""

        def __init__(self, embedder: Any) -> None:
            base_class.__init__(self)
            LocalRagasEmbeddings.__init__(self, embedder)

    from app.embeddings import Embedder

    selected_model = None if model_name == "local" else model_name
    return ModernLocalRagasEmbeddings(Embedder(model_name=selected_model))


def _metric_inputs(record: EvaluationRecord) -> dict[str, dict[str, Any]]:
    common = {"user_input": record.sample.question}
    return {
        "faithfulness": {
            **common,
            "response": record.answer,
            "retrieved_contexts": list(record.contexts),
        },
        "answer_relevancy": {
            **common,
            "response": record.answer,
        },
        "context_precision": {
            **common,
            "reference": record.sample.ground_truth,
            "retrieved_contexts": list(record.contexts),
        },
        "context_recall": {
            **common,
            "reference": record.sample.ground_truth,
            "retrieved_contexts": list(record.contexts),
        },
    }


def _validated_score(result: Any, metric_name: str) -> float:
    value = result.get(metric_name) if isinstance(result, Mapping) else getattr(
        result,
        "value",
        result,
    )
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("metric score must be numeric")
    score = float(value)
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError("metric score must be finite and between zero and one")
    return score


def _record_metric_error(
    record: EvaluationRecord,
    metric_name: str,
    error: Exception,
    index: int,
    summary_errors: list[str],
) -> None:
    safe_error = f"{metric_name}: {type(error).__name__}"
    record.ragas.errors.append(safe_error)
    summary_errors.append(f"sample {index + 1}: {safe_error}")


def _aggregate_metrics(
    records: Sequence[EvaluationRecord],
) -> dict[str, dict[str, float | int | None]]:
    metrics: dict[str, dict[str, float | int | None]] = {}
    for metric_name in METRIC_NAMES:
        values = [
            float(value)
            for record in records
            if (value := getattr(record.ragas, metric_name)) is not None
        ]
        metrics[metric_name] = {
            "score": round(sum(values) / len(values), 6) if values else None,
            "count": len(values),
        }
    return metrics


def _empty_metrics() -> dict[str, dict[str, float | int | None]]:
    return {name: {"score": None, "count": 0} for name in METRIC_NAMES}


def _missing_dependency_error(error: ModuleNotFoundError) -> str:
    if error.name == "ragas" or (error.name or "").startswith("ragas."):
        return "RAGAS is unavailable; install it with: pip install ragas"
    return f"RAGAS dependency unavailable: {error.name or type(error).__name__}"


def _call_legacy_evaluate(
    bindings: LegacyBindings,
    dataset: Any,
    metric: Any,
) -> Any:
    kwargs = {
        "metrics": [metric],
        "llm": bindings.llm,
        "embeddings": bindings.embeddings,
        "raise_exceptions": False,
    }
    signature = inspect.signature(bindings.evaluate)
    accepts_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if not accepts_kwargs:
        kwargs = {
            key: value
            for key, value in kwargs.items()
            if key in signature.parameters
        }
    return bindings.evaluate(dataset, **kwargs)


def _legacy_result_rows(result: Any) -> list[dict[str, Any]]:
    to_pandas = getattr(result, "to_pandas", None)
    if callable(to_pandas):
        frame = to_pandas()
        rows = frame.to_dict(orient="records")
        if isinstance(rows, list):
            return rows
    raise ValueError("legacy RAGAS result does not expose row scores")
