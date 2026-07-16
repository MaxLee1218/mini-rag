from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.badcase_schema import BadCase
from evaluation.dataset_manager import load_evaluation_dataset
from scripts import import_badcases_to_eval as importer


def _case(**overrides: object) -> BadCase:
    payload: dict[str, object] = {
        "question": "question",
        "answer": "Not found in knowledge base.",
        "expected_answer": None,
        "contexts": [],
        "sources": ["source.txt"],
        "error_type": "retrieval_failure",
        "root_cause": None,
        "solution": None,
        "timestamp": "2026-07-16T12:00:00+00:00",
    }
    payload.update(overrides)
    return BadCase.from_dict(payload)


def _write_badcases(path: Path, cases: list[BadCase]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([case.to_dict() for case in cases], ensure_ascii=False),
        encoding="utf-8",
    )


def test_expected_answer_wins_and_contexts_map_to_reference_contexts() -> None:
    row = importer.badcase_to_evaluation_row(
        _case(
            expected_answer=" expected ",
            solution="implementation fix",
            contexts=["evidence"],
        )
    )

    assert row == {
        "question": "question",
        "ground_truth": "expected",
        "reference_contexts": ["evidence"],
        "metadata": {
            "error_type": "retrieval_failure",
            "root_cause": None,
            "solution": "implementation fix",
            "timestamp": "2026-07-16T12:00:00+00:00",
        },
    }


def test_solution_is_fallback_when_expected_answer_is_blank() -> None:
    row = importer.badcase_to_evaluation_row(
        _case(expected_answer="  ", solution=" fallback answer ")
    )

    assert row is not None
    assert row["ground_truth"] == "fallback answer"
    assert "reference_contexts" not in row


def test_unresolved_case_is_skipped() -> None:
    assert importer.badcase_to_evaluation_row(_case()) is None


def test_import_is_additive_and_compatible_with_dataset_loader(
    tmp_path: Path,
) -> None:
    badcase_path = tmp_path / "eval" / "badcases.json"
    dataset_path = tmp_path / "evaluation" / "dataset.json"
    existing_row = {
        "question": "Existing Question?",
        "ground_truth": "existing answer",
        "metadata": {"owner": "baseline"},
    }
    dataset_path.parent.mkdir(parents=True)
    dataset_path.write_text(json.dumps([existing_row]), encoding="utf-8")
    _write_badcases(
        badcase_path,
        [
            _case(
                question="new question",
                expected_answer="new answer",
                contexts=["new evidence"],
                timestamp="new",
            ),
            _case(
                question="  existing   QUESTION? ",
                expected_answer="replacement",
                timestamp="duplicate",
            ),
            _case(question="unresolved question", timestamp="unresolved"),
        ],
    )

    summary = importer.import_badcases(badcase_path, dataset_path)
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))

    assert summary == importer.ImportSummary(
        scanned=3,
        added=1,
        duplicates=1,
        unresolved=1,
    )
    assert payload[0] == existing_row
    assert len(load_evaluation_dataset(dataset_path)) == 2
    assert payload[1]["reference_contexts"] == ["new evidence"]


def test_import_creates_missing_dataset_and_parent(tmp_path: Path) -> None:
    badcase_path = tmp_path / "badcases.json"
    dataset_path = tmp_path / "nested" / "dataset.json"
    _write_badcases(
        badcase_path,
        [_case(expected_answer="answer", timestamp="new")],
    )

    summary = importer.import_badcases(badcase_path, dataset_path)

    assert summary.added == 1
    assert load_evaluation_dataset(dataset_path)[0].ground_truth == "answer"


def test_import_rejects_missing_badcase_input(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="could not read badcase file"):
        importer.import_badcases(
            tmp_path / "missing.json",
            tmp_path / "dataset.json",
        )


@pytest.mark.parametrize(
    ("source_payload", "dataset_payload", "message"),
    [
        ({}, [], "badcase file top level must be a list"),
        ([], {}, "evaluation dataset top level must be a list"),
        ([], [{"ground_truth": "answer"}], "row 1 question must be a string"),
        (
            [],
            [{"question": " ", "ground_truth": "answer"}],
            "row 1 question must not be blank",
        ),
    ],
)
def test_import_rejects_invalid_top_level_or_existing_rows(
    tmp_path: Path,
    source_payload: object,
    dataset_payload: object,
    message: str,
) -> None:
    badcase_path = tmp_path / "badcases.json"
    dataset_path = tmp_path / "dataset.json"
    badcase_path.write_text(json.dumps(source_payload), encoding="utf-8")
    dataset_path.write_text(json.dumps(dataset_payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        importer.import_badcases(badcase_path, dataset_path)


def test_parse_args_uses_repository_defaults() -> None:
    args = importer.parse_args([])

    assert args.badcase_path == importer.DEFAULT_BADCASE_PATH
    assert args.dataset_path == importer.DEFAULT_DATASET_PATH


def test_main_prints_counts_without_badcase_content(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    badcase_path = tmp_path / "badcases.json"
    dataset_path = tmp_path / "dataset.json"
    _write_badcases(
        badcase_path,
        [_case(question="private question", expected_answer="private answer")],
    )

    result = importer.main(
        [
            "--badcase-path",
            str(badcase_path),
            "--dataset-path",
            str(dataset_path),
        ]
    )

    assert result == 0
    output = capsys.readouterr().out
    assert "scanned=1 added=1 duplicates=0 unresolved=0" in output
    assert "private question" not in output
    assert "private answer" not in output


def test_main_returns_nonzero_for_invalid_input(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = importer.main(
        [
            "--badcase-path",
            str(tmp_path / "missing.json"),
            "--dataset-path",
            str(tmp_path / "dataset.json"),
        ]
    )

    assert result == 1
    assert "ERROR: ValueError:" in capsys.readouterr().err
