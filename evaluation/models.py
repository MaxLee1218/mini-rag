"""Shared data contracts for offline RAG evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EvaluationSample:
    """A validated offline evaluation input."""

    question: str
    ground_truth: str
    reference_contexts: tuple[str, ...] = ()
    should_abstain: bool = False


@dataclass(frozen=True)
class LatencyObservation:
    """Exclusive stage latency values in milliseconds."""

    embedding: float | None
    retrieval: float | None
    generation: float | None
    total: float | None


@dataclass
class RagasScores:
    """Optional RAGAS scores for one sample."""

    faithfulness: float | None = None
    answer_relevancy: float | None = None
    context_precision: float | None = None
    context_recall: float | None = None
    errors: list[str] = field(default_factory=list)


@dataclass
class EvaluationRecord:
    """Pipeline outputs and diagnostics for one sample."""

    sample: EvaluationSample
    answer: str
    contexts: list[str]
    sources: list[str]
    route: str
    latency: LatencyObservation
    retrieval_hit: bool | None
    abstention_correct: bool | None
    ragas: RagasScores = field(default_factory=RagasScores)
    errors: list[str] = field(default_factory=list)
