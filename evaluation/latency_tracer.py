"""Evaluation-only latency instrumentation for a pipeline call."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter_ns
from typing import Any

from evaluation.models import LatencyObservation


Clock = Callable[[], int]


class PipelineTraceError(RuntimeError):
    """Wrap a pipeline failure with the latency observed before it failed."""

    def __init__(
        self,
        original_exception: Exception,
        latency: LatencyObservation,
        warnings: list[str] | None = None,
    ) -> None:
        super().__init__("pipeline call failed while tracing latency")
        self.original_exception = original_exception
        self.latency = latency
        self.warnings = list(warnings or [])


@dataclass
class _Elapsed:
    embedding_ns: int | None = None
    retrieval_ns: int | None = None
    generation_ns: int | None = None

    def add(self, stage: str, elapsed_ns: int) -> None:
        attribute = f"{stage}_ns"
        current = getattr(self, attribute)
        setattr(self, attribute, elapsed_ns if current is None else current + elapsed_ns)

    def observation(self, total_ns: int) -> LatencyObservation:
        embedding = _milliseconds(self.embedding_ns)
        retrieval_elapsed = _milliseconds(self.retrieval_ns)
        retrieval = (
            max(0.0, retrieval_elapsed - (embedding or 0.0))
            if retrieval_elapsed is not None
            else None
        )
        return LatencyObservation(
            embedding=embedding,
            retrieval=retrieval,
            generation=_milliseconds(self.generation_ns),
            total=_milliseconds(total_ns),
        )


class _TimedProxy:
    def __init__(self, target: Any, elapsed: _Elapsed, clock_ns: Clock) -> None:
        self._target = target
        self._elapsed = elapsed
        self._clock_ns = clock_ns

    def __getattr__(self, name: str) -> Any:
        return getattr(self._target, name)

    def _call(
        self,
        stage: str,
        function: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        started_ns = self._clock_ns()
        try:
            return function(*args, **kwargs)
        finally:
            self._elapsed.add(stage, self._clock_ns() - started_ns)


class _RetrieverProxy(_TimedProxy):
    def retrieve(self, *args: Any, **kwargs: Any) -> Any:
        return self._call("retrieval", self._target.retrieve, *args, **kwargs)


class _EmbedderProxy(_TimedProxy):
    def embed_query(self, *args: Any, **kwargs: Any) -> Any:
        return self._call("embedding", self._target.embed_query, *args, **kwargs)

    def embed_texts(self, *args: Any, **kwargs: Any) -> Any:
        return self._call("embedding", self._target.embed_texts, *args, **kwargs)

    def embed_chunks(self, *args: Any, **kwargs: Any) -> Any:
        return self._call("embedding", self._target.embed_chunks, *args, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._call("embedding", self._target, *args, **kwargs)


class _GeneratorProxy(_TimedProxy):
    def generate(self, *args: Any, **kwargs: Any) -> Any:
        generate = getattr(self._target, "generate", None)
        function = generate if callable(generate) else self._target
        return self._call("generation", function, *args, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._call("generation", self._target, *args, **kwargs)


def trace_pipeline_call(
    pipeline: Any,
    question: str,
    *,
    top_k: int | None = None,
    clock_ns: Clock = perf_counter_ns,
) -> tuple[Any, LatencyObservation, list[str]]:
    """Call ``pipeline.ask`` while measuring exclusive RAG stage latency."""
    elapsed = _Elapsed()
    warnings: list[str] = []
    replacements: list[tuple[Any, str, Any]] = []

    try:
        retriever = pipeline.retriever
        generator = pipeline.generator
        embedder_location = _find_embedder(retriever)
        if embedder_location is None:
            warnings.append("embedding timer unavailable for custom retriever")
        else:
            owner, embedder = embedder_location
            _replace(
                owner,
                "embedder",
                _EmbedderProxy(embedder, elapsed, clock_ns),
                replacements,
            )

        _replace(
            pipeline,
            "retriever",
            _RetrieverProxy(retriever, elapsed, clock_ns),
            replacements,
        )
        _replace(
            pipeline,
            "generator",
            _GeneratorProxy(generator, elapsed, clock_ns),
            replacements,
        )

        total_started_ns = clock_ns()
        try:
            if top_k is None:
                result = pipeline.ask(question)
            else:
                result = pipeline.ask(question, top_k=top_k)
        except Exception as error:
            observation = elapsed.observation(clock_ns() - total_started_ns)
            raise PipelineTraceError(error, observation, warnings) from error

        observation = elapsed.observation(clock_ns() - total_started_ns)
        return result, observation, warnings
    finally:
        for owner, attribute, original in reversed(replacements):
            setattr(owner, attribute, original)


def _find_embedder(retriever: Any) -> tuple[Any, Any] | None:
    embedder = getattr(retriever, "embedder", None)
    if embedder is not None:
        return retriever, embedder

    dense_retriever = getattr(retriever, "dense_retriever", None)
    if dense_retriever is not None:
        embedder = getattr(dense_retriever, "embedder", None)
        if embedder is not None:
            return dense_retriever, embedder

    child_retriever = getattr(retriever, "child_retriever", None)
    if child_retriever is None:
        return None
    embedder = getattr(child_retriever, "embedder", None)
    if embedder is None:
        return None
    return child_retriever, embedder


def _replace(
    owner: Any,
    attribute: str,
    replacement: Any,
    replacements: list[tuple[Any, str, Any]],
) -> None:
    original = getattr(owner, attribute)
    setattr(owner, attribute, replacement)
    replacements.append((owner, attribute, original))


def _milliseconds(value_ns: int | None) -> float | None:
    return None if value_ns is None else value_ns / 1_000_000
