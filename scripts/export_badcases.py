"""Export suspected RAG failures from request logs for human review."""

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


DEFAULT_LOG_PATH = PROJECT_ROOT / "logs" / "rag_requests.jsonl"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "eval" / "badcases.json"
ABSTENTION_MARKERS = ("Not found in knowledge base.", "未找到相关信息")


@dataclass(frozen=True)
class ExportSummary:
    """Counts produced by one export run."""

    scanned: int
    candidates: int
    added: int
    duplicates: int


def classify_log_entry(entry: Mapping[str, object]) -> str | None:
    """Return the first matching badcase rule for a request log entry."""
    answer = entry.get("answer")
    if not isinstance(answer, str):
        raise ValueError("log answer must be a string")
    if not answer.strip():
        return "empty_answer"

    if "contexts" in entry:
        contexts = entry["contexts"]
        if not isinstance(contexts, list) or not all(
            isinstance(item, str) for item in contexts
        ):
            raise ValueError("log contexts must be an array of strings")
        if not contexts:
            return "retrieval_failure"

    if any(marker in answer for marker in ABSTENTION_MARKERS):
        return "generation_or_retrieval_failure"
    return None


def export_badcases(log_path: Path, output_path: Path) -> ExportSummary:
    """Append new suspected failures without replacing reviewed badcases."""
    existing_rows = _load_existing_badcases(output_path)
    known_timestamps = {
        BadCase.from_dict(row).timestamp for row in existing_rows
    }
    merged_rows = list(existing_rows)
    scanned = 0
    candidates = 0
    added = 0
    duplicates = 0

    for line_number, entry in _iter_request_logs(log_path):
        scanned += 1
        try:
            error_type = classify_log_entry(entry)
            if error_type is None:
                continue
            candidates += 1
            case = _badcase_from_log(entry, error_type)
        except ValueError as error:
            raise ValueError(f"log line {line_number}: {error}") from error

        if case.timestamp in known_timestamps:
            duplicates += 1
            continue
        merged_rows.append(case.to_dict())
        known_timestamps.add(case.timestamp)
        added += 1

    _write_json_array(output_path, merged_rows)
    return ExportSummary(scanned, candidates, added, duplicates)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse optional request-log and badcase output paths."""
    parser = argparse.ArgumentParser(
        description="Export suspected RAG badcases from request logs.",
    )
    parser.add_argument("--log-path", type=Path, default=DEFAULT_LOG_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run export and translate data errors to a stable exit code."""
    args = parse_args(argv)
    try:
        summary = export_badcases(args.log_path, args.output_path)
    except Exception as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        return 1

    print(
        f"scanned={summary.scanned} candidates={summary.candidates} "
        f"added={summary.added} duplicates={summary.duplicates}"
    )
    print(f"badcases={args.output_path}")
    return 0


def _iter_request_logs(log_path: Path) -> list[tuple[int, Mapping[str, object]]]:
    if not log_path.exists():
        return []
    try:
        lines = log_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise ValueError(f"could not read request log {log_path}: {error}") from error

    entries: list[tuple[int, Mapping[str, object]]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"request log line {line_number} contains invalid JSON"
            ) from error
        if not isinstance(entry, Mapping):
            raise ValueError(
                f"request log line {line_number} must be a JSON object"
            )
        entries.append((line_number, entry))
    return entries


def _badcase_from_log(
    entry: Mapping[str, object],
    error_type: str,
) -> BadCase:
    contexts = entry.get("contexts", [])
    return BadCase.from_dict(
        {
            "question": entry.get("question"),
            "answer": entry.get("answer"),
            "expected_answer": None,
            "contexts": contexts,
            "sources": entry.get("sources"),
            "error_type": error_type,
            "root_cause": None,
            "solution": None,
            "timestamp": entry.get("timestamp"),
        }
    )


def _load_existing_badcases(output_path: Path) -> list[dict[str, Any]]:
    if not output_path.exists():
        return []
    try:
        payload = json.loads(output_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read badcase file {output_path}: {error}") from error
    if not isinstance(payload, list):
        raise ValueError("badcase file top level must be a list")

    rows: list[dict[str, Any]] = []
    for row_number, row in enumerate(payload, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"badcase row {row_number} must be a JSON object")
        try:
            BadCase.from_dict(row)
        except ValueError as error:
            raise ValueError(f"badcase row {row_number}: {error}") from error
        rows.append(row)
    return rows


def _write_json_array(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
