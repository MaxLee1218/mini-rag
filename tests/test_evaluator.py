from types import SimpleNamespace

from app.prompt_builder import NOT_FOUND_MESSAGE
from evaluation.evaluator import EvaluationRunner, summarize_retrieval
from evaluation.latency_tracer import PipelineTraceError
from evaluation.models import EvaluationRecord, EvaluationSample, LatencyObservation


SUCCESS_LATENCY = LatencyObservation(1.0, 2.0, 3.0, 7.0)
NULL_LATENCY = LatencyObservation(None, None, None, None)


class FakePipeline:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes) if isinstance(outcomes, list) else [outcomes]
        self.calls = []

    def ask(self, question, top_k=None):
        self.calls.append((question, top_k))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def successful_result(
    *,
    answer="answer",
    contexts=None,
    sources=None,
    route="rag",
):
    return SimpleNamespace(
        answer=answer,
        contexts=contexts
        if contexts is not None
        else [
            {
                "content": (
                    "RAG retrieves relevant context before generating an answer."
                ),
                "source": "rag_notes.txt",
            }
        ],
        sources=sources if sources is not None else ["rag_notes.txt"],
        route=route,
    )


def answerable_sample(question="What does RAG retrieve?"):
    return EvaluationSample(
        question=question,
        ground_truth="Relevant context.",
        reference_contexts=(
            "RAG retrieves relevant context before generating an answer.",
        ),
    )


def abstention_sample(question="Unknown?"):
    return EvaluationSample(
        question=question,
        ground_truth=NOT_FOUND_MESSAGE,
        should_abstain=True,
    )


def record(*, retrieval_hit=None, abstention_correct=None):
    sample = abstention_sample() if abstention_correct is not None else answerable_sample()
    return EvaluationRecord(
        sample=sample,
        answer="answer",
        contexts=[],
        sources=[],
        route="rag",
        latency=NULL_LATENCY,
        retrieval_hit=retrieval_hit,
        abstention_correct=abstention_correct,
    )


def direct_trace(pipeline, question, *, top_k=None):
    return pipeline.ask(question, top_k=top_k), SUCCESS_LATENCY, []


def warning_trace(pipeline, question, *, top_k=None):
    return (
        pipeline.ask(question, top_k=top_k),
        SUCCESS_LATENCY,
        ["embedding timer unavailable for custom retriever"],
    )


def test_runner_calls_ask_once_and_preserves_sources():
    pipeline = FakePipeline(successful_result())

    records = EvaluationRunner(pipeline, top_k=5, trace=direct_trace).evaluate(
        [answerable_sample()]
    )

    assert pipeline.calls == [("What does RAG retrieve?", 5)]
    assert records[0].contexts == [
        "RAG retrieves relevant context before generating an answer."
    ]
    assert records[0].sources == ["rag_notes.txt"]
    assert records[0].retrieval_hit is True
    assert records[0].latency == SUCCESS_LATENCY


def test_runner_preserves_successful_trace_warnings():
    result = EvaluationRunner(
        FakePipeline(successful_result()),
        trace=warning_trace,
    ).evaluate([answerable_sample()])[0]

    assert result.warnings == ["embedding timer unavailable for custom retriever"]


def test_retrieval_match_casefolds_and_collapses_whitespace():
    pipeline = FakePipeline(
        successful_result(
            contexts=[{"page_content": "Prefix STRASSE\n  context suffix"}]
        )
    )
    sample = EvaluationSample(
        question="q",
        ground_truth="a",
        reference_contexts=("Strasse context",),
    )

    result = EvaluationRunner(pipeline, trace=direct_trace).evaluate([sample])[0]

    assert result.contexts == ["Prefix STRASSE\n  context suffix"]
    assert result.retrieval_hit is True


def test_runner_scores_only_exact_stripped_abstention_message():
    exact_pipeline = FakePipeline(
        successful_result(answer=f"  {NOT_FOUND_MESSAGE}\n", contexts=[], sources=[])
    )
    extra_pipeline = FakePipeline(
        successful_result(
            answer=f"{NOT_FOUND_MESSAGE}\nSources:", contexts=[], sources=[]
        )
    )

    exact = EvaluationRunner(exact_pipeline, trace=direct_trace).evaluate(
        [abstention_sample()]
    )[0]
    extra = EvaluationRunner(extra_pipeline, trace=direct_trace).evaluate(
        [abstention_sample()]
    )[0]

    assert exact.retrieval_hit is None
    assert exact.abstention_correct is True
    assert extra.abstention_correct is False


def test_retrieval_summary_excludes_abstention_from_hit_rate():
    summary = summarize_retrieval(
        [
            record(retrieval_hit=True),
            record(retrieval_hit=False),
            record(abstention_correct=True),
        ]
    )

    assert summary == {
        "retrieval_hit_rate": 0.5,
        "retrieval_hits": 1,
        "retrieval_evaluable_samples": 2,
        "abstention_accuracy": 1.0,
        "correct_abstentions": 1,
        "abstention_samples": 1,
    }


def test_retrieval_summary_uses_null_rates_for_empty_denominators():
    assert summarize_retrieval([]) == {
        "retrieval_hit_rate": None,
        "retrieval_hits": 0,
        "retrieval_evaluable_samples": 0,
        "abstention_accuracy": None,
        "correct_abstentions": 0,
        "abstention_samples": 0,
    }


def test_runner_continues_after_one_pipeline_error():
    pipeline = FakePipeline([RuntimeError("boom"), successful_result()])

    records = EvaluationRunner(pipeline, trace=direct_trace).evaluate(
        [answerable_sample("q1"), answerable_sample("q2")]
    )

    assert pipeline.calls == [("q1", None), ("q2", None)]
    assert records[0].answer == ""
    assert records[0].contexts == []
    assert records[0].sources == []
    assert records[0].latency == NULL_LATENCY
    assert records[0].retrieval_hit is None
    assert records[0].errors == ["pipeline: RuntimeError: boom"]
    assert records[1].answer == "answer"


def test_runner_preserves_partial_latency_and_original_trace_error():
    partial = LatencyObservation(2.0, 4.0, None, 8.0)

    def failing_trace(pipeline, question, *, top_k=None):
        raise PipelineTraceError(
            ValueError("bad provider"),
            partial,
            warnings=["embedding timer unavailable for custom retriever"],
        )

    result = EvaluationRunner(FakePipeline([]), trace=failing_trace).evaluate(
        [answerable_sample()]
    )[0]

    assert result.latency == partial
    assert result.errors == ["pipeline: ValueError: bad provider"]
    assert result.warnings == ["embedding timer unavailable for custom retriever"]


def test_runner_bounds_and_flattens_pipeline_error_messages():
    def failing_trace(pipeline, question, *, top_k=None):
        raise RuntimeError("secret\n" + "x" * 500)

    result = EvaluationRunner(FakePipeline([]), trace=failing_trace).evaluate(
        [answerable_sample()]
    )[0]

    assert result.latency == NULL_LATENCY
    assert "\n" not in result.errors[0]
    assert len(result.errors[0]) <= 200
