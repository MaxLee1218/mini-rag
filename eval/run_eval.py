"""Run the configured mini-RAG pipeline against the offline evaluation set."""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import config
from app.pipeline_factory import build_default_pipeline
from evaluation.dataset_manager import load_evaluation_dataset
from evaluation.evaluator import EvaluationRunner, summarize_retrieval
from evaluation.latency_analyzer import analyze_latencies
from evaluation.models import EvaluationRecord, EvaluationSample
from evaluation.ragas_evaluator import (
    METRIC_NAMES,
    RagasEvaluationSummary,
    RagasEvaluator,
    default_ragas_config,
)
from evaluation.report_generator import build_report, write_evaluation_reports
from scripts.ask import close_pipeline_resources


@dataclass(frozen=True)
class RunnerDependencies:
    """Injectable evaluation operations for deterministic orchestration tests."""

    load_dataset: Callable[[Path], list[EvaluationSample]]
    build_pipeline: Callable[[int], Any]
    evaluate_pipeline: Callable[
        [Any, list[EvaluationSample], int],
        list[EvaluationRecord],
    ]
    evaluate_ragas: Callable[
        [list[EvaluationRecord]],
        RagasEvaluationSummary,
    ]
    summarize_retrieval: Callable[
        [list[EvaluationRecord]],
        Mapping[str, float | int | None],
    ]
    analyze_latency: Callable[
        [list[Any]],
        Mapping[str, Mapping[str, float | int | None]],
    ]
    build_report: Callable[..., dict[str, Any]]
    write_reports: Callable[[Mapping[str, Any], Path, Path], None]
    close_pipeline: Callable[[Any], None]
    now: Callable[[], datetime]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse evaluation dataset, report path, and retrieval options."""
    parser = argparse.ArgumentParser(
        description="Run offline quality and latency evaluation for mini-rag.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=config.EVALUATION_DATASET_PATH,
        help="Evaluation dataset JSON path.",
    )
    parser.add_argument(
        "--json-report",
        type=Path,
        default=config.EVALUATION_JSON_REPORT_PATH,
        help="Generated JSON report path.",
    )
    parser.add_argument(
        "--markdown-report",
        type=Path,
        default=config.EVALUATION_MARKDOWN_REPORT_PATH,
        help="Generated Markdown report path.",
    )
    parser.add_argument(
        "--top-k",
        type=_positive_int,
        default=config.EVALUATION_TOP_K,
        help="Number of final retrieved contexts per question.",
    )
    return parser.parse_args(argv)


def run_evaluation(
    args: argparse.Namespace,
    *,
    dependencies: RunnerDependencies | Any | None = None,
) -> dict[str, Any]:
    """Execute every evaluation layer and return the generated report model."""
    operations = dependencies or _default_dependencies()
    pipeline: Any | None = None
    try:
        samples = operations.load_dataset(Path(args.dataset))
        pipeline = operations.build_pipeline(args.top_k)
        records = operations.evaluate_pipeline(
            pipeline,
            samples,
            args.top_k,
        )
        ragas_summary = operations.evaluate_ragas(records)
        retrieval_summary = operations.summarize_retrieval(records)
        latency_summary = operations.analyze_latency(
            [record.latency for record in records]
        )
        report = operations.build_report(
            records=records,
            retrieval_summary=retrieval_summary,
            ragas_summary=ragas_summary,
            latency_summary=latency_summary,
            metadata=_run_metadata(
                Path(args.dataset),
                len(samples),
                args.top_k,
                operations.now(),
                ragas_summary,
            ),
            faithfulness_threshold=config.EVALUATION_FAITHFULNESS_THRESHOLD,
            context_recall_threshold=config.EVALUATION_CONTEXT_RECALL_THRESHOLD,
            question_preview_chars=config.EVALUATION_QUESTION_PREVIEW_CHARS,
            answer_preview_chars=config.EVALUATION_ANSWER_PREVIEW_CHARS,
        )
        operations.write_reports(
            report,
            Path(args.json_report),
            Path(args.markdown_report),
        )
        return report
    finally:
        if pipeline is not None:
            operations.close_pipeline(pipeline)


def main(
    argv: list[str] | None = None,
    *,
    dependencies: RunnerDependencies | Any | None = None,
) -> int:
    """Run evaluation and translate expected failures to a stable exit code."""
    try:
        args = parse_args(argv)
    except SystemExit as error:
        return int(error.code) if isinstance(error.code, int) else 2
    try:
        report = run_evaluation(args, dependencies=dependencies)
    except Exception as error:
        print(
            f"Evaluation failed: {type(error).__name__}: {_safe_message(error)}",
            file=sys.stderr,
        )
        return 1

    print(f"Evaluation status: {report.get('status', 'unknown')}")
    print(f"Samples: {(report.get('dataset') or {}).get('samples', 0)}")
    print(f"JSON report: {Path(args.json_report)}")
    print(f"Markdown report: {Path(args.markdown_report)}")
    return 0


def _default_dependencies() -> RunnerDependencies:
    return RunnerDependencies(
        load_dataset=load_evaluation_dataset,
        build_pipeline=build_default_pipeline,
        evaluate_pipeline=_evaluate_pipeline,
        evaluate_ragas=_evaluate_ragas,
        summarize_retrieval=summarize_retrieval,
        analyze_latency=analyze_latencies,
        build_report=build_report,
        write_reports=write_evaluation_reports,
        close_pipeline=close_pipeline_resources,
        now=lambda: datetime.now().astimezone(),
    )


def _evaluate_pipeline(
    pipeline: Any,
    samples: list[EvaluationSample],
    top_k: int,
) -> list[EvaluationRecord]:
    return EvaluationRunner(pipeline, top_k=top_k).evaluate(samples)


def _evaluate_ragas(
    records: list[EvaluationRecord],
) -> RagasEvaluationSummary:
    try:
        ragas_config = default_ragas_config()
    except Exception as error:
        return RagasEvaluationSummary(
            status="unavailable",
            version=None,
            model=config.EVALUATION_RAGAS_MODEL,
            embedding_model=config.EVALUATION_RAGAS_EMBEDDING_MODEL,
            metrics={
                name: {"score": None, "count": 0} for name in METRIC_NAMES
            },
            errors=[
                "RAGAS configuration unavailable: "
                f"{type(error).__name__}; configure the selected evaluator provider"
            ],
        )
    return RagasEvaluator(ragas_config).evaluate(records)


def _run_metadata(
    dataset_path: Path,
    samples: int,
    top_k: int,
    generated_at: datetime,
    ragas_summary: RagasEvaluationSummary,
) -> dict[str, Any]:
    return {
        "generated_at": generated_at.isoformat(timespec="seconds"),
        "dataset": {
            "samples": samples,
            "path": _display_path(dataset_path),
            "sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        },
        "configuration": {
            "top_k": top_k,
            "chunk_mode": config.CHUNK_MODE,
            "retrieval_strategy": (
                "parent-child" if config.CHUNK_MODE == "parent-child" else "hybrid"
            ),
            "reranker_enabled": config.RERANKER_ENABLED,
            "generator_model": config.DEEPSEEK_MODEL,
            "ragas_provider": config.EVALUATION_RAGAS_PROVIDER,
            "ragas_model": ragas_summary.model,
            "ragas_embedding_model": ragas_summary.embedding_model,
        },
    }


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _safe_message(error: Exception) -> str:
    message = " ".join(str(error).split())
    return (message[:197].rstrip() + "...") if len(message) > 200 else message


if __name__ == "__main__":
    raise SystemExit(main())
