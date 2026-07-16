"""Import human-resolved badcases into the existing evaluation dataset."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.badcase_schema import BadCase
from evaluation.dataset_manager import load_evaluation_dataset


DEFAULT_BADCASE_PATH = PROJECT_ROOT / "eval" / "badcases.json"
DEFAULT_DATASET_PATH = (
    PROJECT_ROOT / "evaluation" / "dataset" / "eval_dataset.json"
)


@dataclass(frozen=True)
class ImportSummary:
    """Counts produced by one evaluation import."""

    scanned: int
    added: int
    duplicates: int
    unresolved: int


def badcase_to_evaluation_row(case: BadCase) -> dict[str, object] | None:
    """Convert one resolved badcase to the repository evaluation schema."""
    if not case.question.strip():
        raise ValueError("badcase question must not be blank")
    ground_truth = next(
        (
            value.strip()
            for value in (case.expected_answer, case.solution)
            if value is not None and value.strip()
        ),
        None,
    )
    if ground_truth is None:
        return None
    if any(not context.strip() for context in case.contexts):
        raise ValueError("badcase contexts must not contain blank strings")

    row: dict[str, object] = {
        "question": case.question,
        "ground_truth": ground_truth,
        "metadata": {
            "error_type": case.error_type,
            "root_cause": case.root_cause,
            "solution": case.solution,
            "timestamp": case.timestamp,
        },
    }
    if case.contexts:
        row["reference_contexts"] = list(case.contexts)
    return row


def import_badcases(
    badcase_path: Path,
    dataset_path: Path,
) -> ImportSummary:
    """Append resolved, nonduplicate badcases to an evaluation dataset."""
    cases = _load_badcases(badcase_path)
    existing_rows = _load_evaluation_rows(dataset_path)
    known_questions = {
        _normalize_question(_existing_question(row, row_number))
        for row_number, row in enumerate(existing_rows, start=1)
    }
    merged_rows = list(existing_rows)
    added = 0
    duplicates = 0
    unresolved = 0

    for row_number, case in enumerate(cases, start=1):
        try:
            row = badcase_to_evaluation_row(case)
        except ValueError as error:
            raise ValueError(f"badcase row {row_number}: {error}") from error
        if row is None:
            unresolved += 1
            continue

        normalized_question = _normalize_question(case.question)
        if normalized_question in known_questions:
            duplicates += 1
            continue
        merged_rows.append(row)
        known_questions.add(normalized_question)
        added += 1

    _write_validated_dataset(dataset_path, merged_rows)
    return ImportSummary(len(cases), added, duplicates, unresolved)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse optional badcase input and evaluation dataset paths."""
    parser = argparse.ArgumentParser(
        description="Import resolved badcases into the evaluation dataset.",
    )
    parser.add_argument(
        "--badcase-path",
        type=Path,
        default=DEFAULT_BADCASE_PATH,
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=DEFAULT_DATASET_PATH,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run import and translate data errors to a stable exit code."""
    args = parse_args(argv)
    try:
        summary = import_badcases(args.badcase_path, args.dataset_path)
    except Exception as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        return 1

    print(
        f"scanned={summary.scanned} added={summary.added} "
        f"duplicates={summary.duplicates} unresolved={summary.unresolved}"
    )
    print(f"dataset={args.dataset_path}")
    return 0


def _load_badcases(path: Path) -> list[BadCase]:
    payload = _read_json(path, "badcase file")
    if not isinstance(payload, list):
        raise ValueError("badcase file top level must be a list")

    cases: list[BadCase] = []
    for row_number, row in enumerate(payload, start=1):
        try:
            cases.append(BadCase.from_dict(row))
        except ValueError as error:
            raise ValueError(f"badcase row {row_number}: {error}") from error
    return cases


def _load_evaluation_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = _read_json(path, "evaluation dataset")
    if not isinstance(payload, list):
        raise ValueError("evaluation dataset top level must be a list")

    rows: list[dict[str, Any]] = []
    known_questions: set[str] = set()
    for row_number, row in enumerate(payload, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"evaluation row {row_number} must be a JSON object")
        question = _existing_question(row, row_number)
        normalized_question = _normalize_question(question)
        if normalized_question in known_questions:
            raise ValueError(f"evaluation row {row_number} duplicate question")
        known_questions.add(normalized_question)
        rows.append(row)
    return rows


def _existing_question(row: Mapping[str, object], row_number: int) -> str:
    question = row.get("question")
    if not isinstance(question, str):
        raise ValueError(f"evaluation row {row_number} question must be a string")
    if not question.strip():
        raise ValueError(f"evaluation row {row_number} question must not be blank")
    return question


def _read_json(path: Path, label: str) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read {label} {path}: {error}") from error


def _normalize_question(question: str) -> str:
    return " ".join(question.casefold().split())


def _write_validated_dataset(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary_path.write_text(
            json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        load_evaluation_dataset(temporary_path)
        temporary_path.replace(path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
