"""Run the existing RAG pipeline against offline evaluation samples."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from app.prompt_builder import NOT_FOUND_MESSAGE, extract_context_text
from evaluation.latency_tracer import PipelineTraceError, trace_pipeline_call
from evaluation.models import EvaluationRecord, EvaluationSample, LatencyObservation


TraceCallable = Callable[..., tuple[Any, LatencyObservation, list[str]]]
_MAX_ERROR_LENGTH = 200
_NULL_LATENCY = LatencyObservation(None, None, None, None)


class EvaluationRunner:
    """Evaluate samples through one traced pipeline call per sample."""

    def __init__(
        self,
        pipeline: Any,
        *,
        top_k: int | None = None,
        trace: TraceCallable = trace_pipeline_call,
    ) -> None:
        self.pipeline = pipeline
        self.top_k = top_k
        self.trace = trace

    def evaluate(
        self,
        samples: Sequence[EvaluationSample],
    ) -> list[EvaluationRecord]:
        """Evaluate every sample, retaining a failure record when one call fails."""
        records: list[EvaluationRecord] = []
        for sample in samples:
            try:
                result, latency, warnings = self.trace(
                    self.pipeline,
                    sample.question,
                    top_k=self.top_k,
                )
                contexts = [
                    text
                    for context in (getattr(result, "contexts", None) or [])
                    if (text := extract_context_text(context))
                ]
                answer = result.answer
                records.append(
                    EvaluationRecord(
                        sample=sample,
                        answer=answer,
                        contexts=contexts,
                        sources=list(getattr(result, "sources", None) or []),
                        route=getattr(result, "route", "rag"),
                        latency=latency,
                        retrieval_hit=_retrieval_hit(sample, contexts),
                        abstention_correct=_abstention_correct(sample, answer),
                        warnings=list(warnings),
                    )
                )
            except Exception as error:
                original_error, latency, warnings = _error_details(error)
                records.append(
                    EvaluationRecord(
                        sample=sample,
                        answer="",
                        contexts=[],
                        sources=[],
                        route="",
                        latency=latency,
                        retrieval_hit=None,
                        abstention_correct=None,
                        errors=[_bounded_pipeline_error(original_error)],
                        warnings=warnings,
                    )
                )
        return records


def summarize_retrieval(
    records: Sequence[EvaluationRecord],
) -> dict[str, float | int | None]:
    """Aggregate retrieval and abstention outcomes with separate denominators."""
    retrieval_outcomes = [
        record.retrieval_hit
        for record in records
        if record.retrieval_hit is not None
    ]
    abstention_outcomes = [
        record.abstention_correct
        for record in records
        if record.abstention_correct is not None
    ]
    retrieval_hits = sum(retrieval_outcomes)
    correct_abstentions = sum(abstention_outcomes)

    return {
        "retrieval_hit_rate": _rate(retrieval_hits, len(retrieval_outcomes)),
        "retrieval_hits": retrieval_hits,
        "retrieval_evaluable_samples": len(retrieval_outcomes),
        "abstention_accuracy": _rate(
            correct_abstentions,
            len(abstention_outcomes),
        ),
        "correct_abstentions": correct_abstentions,
        "abstention_samples": len(abstention_outcomes),
    }


def _retrieval_hit(
    sample: EvaluationSample,
    contexts: Sequence[str],
) -> bool | None:
    if sample.should_abstain:
        return None

    normalized_contexts = [_normalize_text(context) for context in contexts]
    return any(
        normalized_reference
        and normalized_reference in normalized_context
        for reference in sample.reference_contexts
        if (normalized_reference := _normalize_text(reference))
        for normalized_context in normalized_contexts
    )


def _abstention_correct(
    sample: EvaluationSample,
    answer: str,
) -> bool | None:
    if not sample.should_abstain:
        return None
    return answer.strip() == NOT_FOUND_MESSAGE


def _normalize_text(value: str) -> str:
    return " ".join(value.casefold().split())


def _error_details(
    error: Exception,
) -> tuple[Exception, LatencyObservation, list[str]]:
    if isinstance(error, PipelineTraceError):
        return error.original_exception, error.latency, list(error.warnings)
    return error, _NULL_LATENCY, []


def _bounded_pipeline_error(error: Exception) -> str:
    message = " ".join(str(error).split())
    detail = f"pipeline: {type(error).__name__}"
    if message:
        detail = f"{detail}: {message}"
    if len(detail) <= _MAX_ERROR_LENGTH:
        return detail
    return detail[: _MAX_ERROR_LENGTH - 3].rstrip() + "..."


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator
