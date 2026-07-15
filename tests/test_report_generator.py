from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.models import (
    EvaluationRecord,
    EvaluationSample,
    LatencyObservation,
    RagasScores,
)
from evaluation.ragas_evaluator import RagasEvaluationSummary
from evaluation.report_generator import build_report, write_evaluation_reports


def _record(
    *,
    hit: bool | None = True,
    abstention: bool | None = None,
    contexts: list[str] | None = None,
    faithfulness: float | None = 0.9,
    context_recall: float | None = 0.9,
    answer: str = "A grounded answer.",
    errors: list[str] | None = None,
    warnings: list[str] | None = None,
) -> EvaluationRecord:
    return EvaluationRecord(
        sample=EvaluationSample(
            question="What is RAG?",
            ground_truth="RAG retrieves relevant context.",
            reference_contexts=("RAG retrieves relevant context.",),
            should_abstain=abstention is not None,
        ),
        answer=answer,
        contexts=["private full context"] if contexts is None else contexts,
        sources=["data/raw/sample/rag_notes.txt"],
        route="rag",
        latency=LatencyObservation(1.0, 2.0, 3.0, 7.0),
        retrieval_hit=hit,
        abstention_correct=abstention,
        ragas=RagasScores(
            faithfulness=faithfulness,
            answer_relevancy=0.8,
            context_precision=0.7,
            context_recall=context_recall,
        ),
        errors=[] if errors is None else errors,
        warnings=[] if warnings is None else warnings,
    )


def _ragas_summary(status: str = "completed") -> RagasEvaluationSummary:
    return RagasEvaluationSummary(
        status=status,
        version="0.4.3",
        model="gpt-4o-mini",
        embedding_model="text-embedding-3-small",
        metrics={
            "faithfulness": {"score": 0.9, "count": 1},
            "answer_relevancy": {"score": 0.8, "count": 1},
            "context_precision": {"score": 0.7, "count": 1},
            "context_recall": {"score": 0.9, "count": 1},
        },
    )


def _metadata(samples: int = 1) -> dict[str, object]:
    return {
        "generated_at": "2026-07-15T12:00:00+08:00",
        "dataset": {
            "samples": samples,
            "path": "evaluation/dataset/eval_dataset.json",
            "sha256": "abc123",
        },
        "configuration": {"top_k": 5, "generator_model": "deepseek-v4-flash"},
    }


def _retrieval() -> dict[str, float | int | None]:
    return {
        "retrieval_hit_rate": 1.0,
        "retrieval_hits": 1,
        "retrieval_evaluable_samples": 1,
        "abstention_accuracy": None,
        "correct_abstentions": 0,
        "abstention_samples": 0,
    }


def _latency() -> dict[str, dict[str, float | int | None]]:
    return {
        stage: {"p50": float(index), "p95": float(index + 1), "count": 1}
        for index, stage in enumerate(
            ("embedding", "retrieval", "generation", "total"),
            1,
        )
    }


def _build(records: list[EvaluationRecord]) -> dict[str, object]:
    return build_report(
        records=records,
        retrieval_summary=_retrieval(),
        ragas_summary=_ragas_summary(),
        latency_summary=_latency(),
        metadata=_metadata(len(records)),
        faithfulness_threshold=0.7,
        context_recall_threshold=0.7,
        question_preview_chars=100,
        answer_preview_chars=100,
    )


def test_build_report_contains_stable_schema_and_no_full_contexts() -> None:
    report = _build([_record()])

    assert report["schema_version"] == "1.0"
    assert report["status"] == "completed"
    assert report["dataset"]["samples"] == 1
    assert report["retrieval"]["retrieval_hit_rate"] == 1.0
    assert report["ragas"]["version"] == "0.4.3"
    assert report["latency"]["total"]["p95"] == 5.0
    serialized = json.dumps(report)
    assert "private full context" not in serialized


def test_failure_classification_uses_only_available_measurements() -> None:
    record = _record(hit=False, faithfulness=0.6, context_recall=None)

    report = _build([record])

    failed = report["failed_examples"][0]
    assert failed["reasons"] == ["retrieval miss", "hallucination"]
    assert "insufficient context" not in failed["reasons"]
    assert "contexts" not in failed
    assert failed["context_count"] == 1


def test_empty_context_is_insufficient_and_incorrect_abstention_is_hallucination() -> None:
    answerable = _record(contexts=[], context_recall=None)
    negative = _record(
        hit=None,
        abstention=False,
        faithfulness=None,
        context_recall=None,
        answer="An unsupported claim",
    )

    report = _build([answerable, negative])

    assert report["failed_examples"][0]["reasons"] == ["insufficient context"]
    assert report["failed_examples"][1]["reasons"] == ["hallucination"]


def test_pipeline_error_is_reported_without_private_context_or_unbounded_answer() -> None:
    record = _record(
        answer="x" * 200,
        errors=["pipeline: RuntimeError: secret provider payload"],
        warnings=["embedding timer unavailable for custom retriever"],
    )

    report = _build([record, _record()])

    failed = report["failed_examples"][0]
    assert "pipeline error" in failed["reasons"]
    assert len(failed["answer"]) <= 100
    assert report["status"] == "partial"
    assert report["warnings"] == ["embedding timer unavailable for custom retriever"]
    assert "secret provider payload" not in json.dumps(report)
    assert failed["errors"] == ["pipeline: RuntimeError"]


def test_all_pipeline_failures_set_failed_status() -> None:
    report = _build([_record(errors=["pipeline: RuntimeError"])])
    assert report["status"] == "failed"


def test_partial_ragas_status_sets_partial_report_status() -> None:
    report = build_report(
        records=[_record()],
        retrieval_summary=_retrieval(),
        ragas_summary=_ragas_summary("partial"),
        latency_summary=_latency(),
        metadata=_metadata(),
        faithfulness_threshold=0.7,
        context_recall_threshold=0.7,
        question_preview_chars=100,
        answer_preview_chars=100,
    )
    assert report["status"] == "partial"


def test_write_reports_creates_json_and_required_markdown(tmp_path: Path) -> None:
    json_path = tmp_path / "reports" / "evaluation_report.json"
    markdown_path = tmp_path / "reports" / "evaluation_report.md"

    write_evaluation_reports(_build([_record()]), json_path, markdown_path)

    parsed = json.loads(json_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")
    assert parsed["schema_version"] == "1.0"
    for heading in (
        "# RAG Evaluation Report",
        "## Dataset",
        "## Retrieval Metrics",
        "## RAGAS Metrics",
        "## Latency",
        "## Failed Examples",
    ):
        assert heading in markdown
    assert "| Embedding |" in markdown
    assert "p50 describes typical latency" in markdown
    assert "p95 highlights slow-request bottlenecks" in markdown


def test_atomic_write_failure_preserves_existing_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    json_path = tmp_path / "report.json"
    markdown_path = tmp_path / "report.md"
    json_path.write_text("old-json", encoding="utf-8")
    markdown_path.write_text("old-markdown", encoding="utf-8")

    def fail_replace(self: Path, target: Path) -> Path:
        raise OSError("disk failed")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="disk failed"):
        write_evaluation_reports(_build([_record()]), json_path, markdown_path)

    assert json_path.read_text(encoding="utf-8") == "old-json"
    assert markdown_path.read_text(encoding="utf-8") == "old-markdown"
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "report.json",
        "report.md",
    ]
