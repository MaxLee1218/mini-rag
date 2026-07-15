"""Build privacy-bounded evaluation reports and write them atomically."""

from __future__ import annotations

import json
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from evaluation.models import EvaluationRecord
from evaluation.ragas_evaluator import RagasEvaluationSummary


def build_report(
    *,
    records: Sequence[EvaluationRecord],
    retrieval_summary: Mapping[str, float | int | None],
    ragas_summary: RagasEvaluationSummary,
    latency_summary: Mapping[str, Mapping[str, float | int | None]],
    metadata: Mapping[str, Any],
    faithfulness_threshold: float,
    context_recall_threshold: float,
    question_preview_chars: int,
    answer_preview_chars: int,
) -> dict[str, Any]:
    """Create the versioned JSON report shared by both output formats."""
    _validate_preview_limit(question_preview_chars, "question_preview_chars")
    _validate_preview_limit(answer_preview_chars, "answer_preview_chars")
    _validate_threshold(faithfulness_threshold, "faithfulness_threshold")
    _validate_threshold(context_recall_threshold, "context_recall_threshold")

    record_errors = [
        _safe_error_label(error) for record in records for error in record.errors
    ]
    warnings = _dedupe(
        [
            *[warning for record in records for warning in record.warnings],
            *ragas_summary.warnings,
            *_string_list(metadata.get("warnings")),
        ]
    )
    errors = _dedupe(
        [
            *record_errors,
            *ragas_summary.errors,
            *_string_list(metadata.get("errors")),
        ]
    )
    successful_records = sum(not record.errors for record in records)
    if not records or successful_records == 0:
        status = "failed"
    elif (
        successful_records != len(records)
        or ragas_summary.status != "completed"
        or warnings
        or errors
    ):
        status = "partial"
    else:
        status = "completed"

    failed_examples = [
        failure
        for record in records
        if (
            failure := _failed_example(
                record,
                faithfulness_threshold=faithfulness_threshold,
                context_recall_threshold=context_recall_threshold,
                question_preview_chars=question_preview_chars,
                answer_preview_chars=answer_preview_chars,
            )
        )
    ]

    return {
        "schema_version": "1.0",
        "status": status,
        "generated_at": metadata.get("generated_at"),
        "dataset": dict(metadata.get("dataset") or {}),
        "configuration": dict(metadata.get("configuration") or {}),
        "thresholds": {
            "faithfulness": faithfulness_threshold,
            "context_recall": context_recall_threshold,
        },
        "retrieval": dict(retrieval_summary),
        "ragas": {
            "status": ragas_summary.status,
            "version": ragas_summary.version,
            "model": ragas_summary.model,
            "embedding_model": ragas_summary.embedding_model,
            "metrics": ragas_summary.metrics,
            "warnings": list(ragas_summary.warnings),
            "errors": list(ragas_summary.errors),
        },
        "latency": {
            stage: dict(values) for stage, values in latency_summary.items()
        },
        "failed_examples": failed_examples,
        "warnings": warnings,
        "errors": errors,
    }


def write_evaluation_reports(
    report: Mapping[str, Any],
    json_path: Path,
    markdown_path: Path,
) -> None:
    """Write UTF-8 JSON and Markdown through same-directory temporary files."""
    json_target = Path(json_path)
    markdown_target = Path(markdown_path)
    json_content = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    markdown_content = _render_markdown(report)
    staged: list[tuple[Path, Path]] = []
    try:
        staged.append((_stage_text(json_target, json_content), json_target))
        staged.append((_stage_text(markdown_target, markdown_content), markdown_target))
        for temporary_path, target_path in staged:
            temporary_path.replace(target_path)
    finally:
        for temporary_path, _ in staged:
            temporary_path.unlink(missing_ok=True)


def _failed_example(
    record: EvaluationRecord,
    *,
    faithfulness_threshold: float,
    context_recall_threshold: float,
    question_preview_chars: int,
    answer_preview_chars: int,
) -> dict[str, Any] | None:
    reasons: list[str] = []
    if record.retrieval_hit is False:
        reasons.append("retrieval miss")
    if (
        not (record.sample.should_abstain and record.abstention_correct is True)
        and record.ragas.faithfulness is not None
        and record.ragas.faithfulness < faithfulness_threshold
    ):
        reasons.append("hallucination")
    if record.sample.should_abstain and record.abstention_correct is False:
        _append_once(reasons, "hallucination")
    if not record.sample.should_abstain and (
        not record.contexts
        or (
            record.ragas.context_recall is not None
            and record.ragas.context_recall < context_recall_threshold
        )
    ):
        reasons.append("insufficient context")
    if record.errors:
        reasons.append("pipeline error")
    if not reasons:
        return None
    return {
        "question": _bounded_preview(record.sample.question, question_preview_chars),
        "answer": _bounded_preview(record.answer, answer_preview_chars),
        "sources": list(record.sources),
        "context_count": len(record.contexts),
        "route": record.route,
        "reasons": reasons,
        "metrics": {
            "faithfulness": record.ragas.faithfulness,
            "answer_relevancy": record.ragas.answer_relevancy,
            "context_precision": record.ragas.context_precision,
            "context_recall": record.ragas.context_recall,
        },
        "errors": [_safe_error_label(error) for error in record.errors],
    }


def _render_markdown(report: Mapping[str, Any]) -> str:
    dataset = report.get("dataset") or {}
    retrieval = report.get("retrieval") or {}
    ragas = report.get("ragas") or {}
    metrics = ragas.get("metrics") or {}
    latency = report.get("latency") or {}
    lines = [
        "# RAG Evaluation Report",
        "",
        "## Dataset",
        "",
        f"- Samples: {dataset.get('samples', 'N/A')}",
        f"- Date: {report.get('generated_at') or 'N/A'}",
        f"- Status: {report.get('status') or 'N/A'}",
        f"- Dataset: {dataset.get('path', 'N/A')}",
        "",
        "## Retrieval Metrics",
        "",
        "| Metric | Score | Valid Samples |",
        "| - | -: | -: |",
        (
            "| Hit Rate | "
            f"{_format_score(retrieval.get('retrieval_hit_rate'))} | "
            f"{retrieval.get('retrieval_evaluable_samples', 0)} |"
        ),
        (
            "| Abstention Accuracy | "
            f"{_format_score(retrieval.get('abstention_accuracy'))} | "
            f"{retrieval.get('abstention_samples', 0)} |"
        ),
        "",
        "## RAGAS Metrics",
        "",
        "| Metric | Score | Valid Samples |",
        "| - | -: | -: |",
    ]
    for name, label in (
        ("faithfulness", "Faithfulness"),
        ("answer_relevancy", "Answer Relevancy"),
        ("context_precision", "Context Precision"),
        ("context_recall", "Context Recall"),
    ):
        item = metrics.get(name) or {}
        lines.append(
            f"| {label} | {_format_score(item.get('score'))} | {item.get('count', 0)} |"
        )
    lines.extend(
        [
            "",
            "## Latency",
            "",
            "| Stage | p50 (ms) | p95 (ms) | Samples |",
            "| - | -: | -: | -: |",
        ]
    )
    for stage, label in (
        ("embedding", "Embedding"),
        ("retrieval", "Retrieval"),
        ("generation", "Generation"),
        ("total", "Total"),
    ):
        item = latency.get(stage) or {}
        lines.append(
            f"| {label} | {_format_number(item.get('p50'))} | "
            f"{_format_number(item.get('p95'))} | {item.get('count', 0)} |"
        )
    lines.extend(
        [
            "",
            "p50 describes typical latency for normal user experience; "
            "p95 highlights slow-request bottlenecks.",
            "",
            "## Failed Examples",
            "",
        ]
    )
    failed_examples = report.get("failed_examples") or []
    if not failed_examples:
        lines.append("None.")
    for index, example in enumerate(failed_examples, 1):
        lines.extend(
            [
                f"### Example {index}",
                "",
                "问题：",
                "",
                str(example.get("question") or "N/A"),
                "",
                "预测答案：",
                "",
                str(example.get("answer") or "N/A"),
                "",
                "失败原因：",
                "",
                *[f"- {reason}" for reason in example.get("reasons") or []],
                "",
            ]
        )
    warnings = report.get("warnings") or []
    errors = report.get("errors") or []
    if warnings:
        lines.extend(["## Warnings", "", *[f"- {item}" for item in warnings], ""])
    if errors:
        lines.extend(["## Errors", "", *[f"- {item}" for item in errors], ""])
    return "\n".join(lines).rstrip() + "\n"


def _stage_text(target: Path, content: str) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary:
        temporary.write(content)
        return Path(temporary.name)


def _bounded_preview(value: str, limit: int) -> str:
    clean = " ".join(value.split())
    if len(clean) <= limit:
        return clean
    if limit <= 3:
        return clean[:limit]
    return clean[: limit - 3].rstrip() + "..."


def _format_score(value: Any) -> str:
    return "N/A" if value is None else f"{float(value):.4f}"


def _format_number(value: Any) -> str:
    return "N/A" if value is None else f"{float(value):.3f}"


def _validate_preview_limit(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _validate_threshold(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be between zero and one")
    if not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{name} must be between zero and one")


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [str(item) for item in value]


def _dedupe(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _append_once(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)


def _safe_error_label(error: str) -> str:
    """Retain an error category and type without persisting exception text."""
    parts = [part.strip() for part in error.split(":", 2)]
    if len(parts) >= 2:
        return f"{parts[0]}: {parts[1]}"
    return parts[0][:100]
