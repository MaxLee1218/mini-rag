from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from eval import run_eval
from evaluation.dataset_manager import DatasetValidationError
from evaluation.models import (
    EvaluationRecord,
    EvaluationSample,
    LatencyObservation,
)
from evaluation.ragas_evaluator import RagasEvaluationSummary


def _record() -> EvaluationRecord:
    return EvaluationRecord(
        sample=EvaluationSample("question", "ground truth", ("context",)),
        answer="answer",
        contexts=["context"],
        sources=["source.txt"],
        route="rag",
        latency=LatencyObservation(1.0, 2.0, 3.0, 7.0),
        retrieval_hit=True,
        abstention_correct=None,
    )


def _ragas() -> RagasEvaluationSummary:
    return RagasEvaluationSummary(
        status="completed",
        version="0.4.3",
        model="judge",
        embedding_model="embedding",
        metrics={
            name: {"score": 1.0, "count": 1}
            for name in (
                "faithfulness",
                "answer_relevancy",
                "context_precision",
                "context_recall",
            )
        },
    )


def _dependencies(tmp_path: Path) -> tuple[run_eval.RunnerDependencies, SimpleNamespace]:
    state = SimpleNamespace(
        pipeline=object(),
        pipeline_builds=[],
        evaluations=[],
        ragas_calls=0,
        report_calls=[],
        closed=False,
    )
    sample = _record().sample
    records = [_record()]

    def build_pipeline(top_k: int) -> object:
        state.pipeline_builds.append(top_k)
        return state.pipeline

    def evaluate_pipeline(
        pipeline: object,
        samples: list[EvaluationSample],
        top_k: int,
    ) -> list[EvaluationRecord]:
        state.evaluations.append((pipeline, samples, top_k))
        return records

    def evaluate_ragas(rows: list[EvaluationRecord]) -> RagasEvaluationSummary:
        state.ragas_calls += 1
        assert rows is records
        return _ragas()

    def build_report(**kwargs: object) -> dict[str, object]:
        state.report_calls.append(kwargs)
        return {
            "schema_version": "1.0",
            "status": "completed",
            "dataset": {"samples": 1},
        }

    def write_reports(
        report: object,
        json_path: Path,
        markdown_path: Path,
    ) -> None:
        json_path.write_text("{}", encoding="utf-8")
        markdown_path.write_text("report", encoding="utf-8")

    dependencies = run_eval.RunnerDependencies(
        load_dataset=lambda path: [sample],
        build_pipeline=build_pipeline,
        evaluate_pipeline=evaluate_pipeline,
        evaluate_ragas=evaluate_ragas,
        summarize_retrieval=lambda rows: {"retrieval_hit_rate": 1.0},
        analyze_latency=lambda rows: {"total": {"p50": 7.0, "p95": 7.0, "count": 1}},
        build_report=build_report,
        write_reports=write_reports,
        close_pipeline=lambda pipeline: setattr(state, "closed", True),
        now=lambda: datetime(2026, 7, 15, 12, 0, tzinfo=timezone.utc),
    )
    return dependencies, state


def test_main_orchestrates_layers_once_and_closes_pipeline(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dependencies, state = _dependencies(tmp_path)
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text("[]", encoding="utf-8")
    json_path = tmp_path / "report.json"
    markdown_path = tmp_path / "report.md"

    code = run_eval.main(
        [
            "--dataset",
            str(dataset_path),
            "--json-report",
            str(json_path),
            "--markdown-report",
            str(markdown_path),
            "--top-k",
            "3",
        ],
        dependencies=dependencies,
    )

    assert code == 0
    assert state.pipeline_builds == [3]
    assert state.evaluations[0][0] is state.pipeline
    assert state.evaluations[0][2] == 3
    assert state.ragas_calls == 1
    assert len(state.report_calls) == 1
    assert state.closed is True
    assert json_path.is_file()
    assert markdown_path.is_file()
    output = capsys.readouterr().out
    assert "completed" in output
    assert str(json_path) in output


def test_main_returns_one_for_dataset_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dependencies, state = _dependencies(tmp_path)

    def fail(path: Path) -> list[EvaluationSample]:
        raise DatasetValidationError("bad dataset")

    dependencies = SimpleNamespace(**{**dependencies.__dict__, "load_dataset": fail})

    assert run_eval.main([], dependencies=dependencies) == 1
    assert "DatasetValidationError: bad dataset" in capsys.readouterr().err
    assert state.closed is False


def test_main_closes_pipeline_when_report_write_fails(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dependencies, state = _dependencies(tmp_path)

    def fail(*args: object) -> None:
        raise OSError("disk full")

    dependencies = SimpleNamespace(**{**dependencies.__dict__, "write_reports": fail})

    assert run_eval.main([], dependencies=dependencies) == 1
    assert "OSError: disk full" in capsys.readouterr().err
    assert state.closed is True


def test_parse_args_uses_central_defaults() -> None:
    from app import config

    args = run_eval.parse_args([])

    assert args.dataset == config.EVALUATION_DATASET_PATH
    assert args.json_report == config.EVALUATION_JSON_REPORT_PATH
    assert args.markdown_report == config.EVALUATION_MARKDOWN_REPORT_PATH
    assert args.top_k == config.EVALUATION_TOP_K


def test_main_returns_two_for_invalid_top_k(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert run_eval.main(["--top-k", "0"]) == 2
    assert "positive integer" in capsys.readouterr().err
