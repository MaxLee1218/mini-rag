from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import export_badcases


def _log_entry(
    *,
    timestamp: str,
    answer: str = "A grounded answer.",
    contexts: list[str] | None = None,
    include_contexts: bool = False,
) -> dict[str, object]:
    entry: dict[str, object] = {
        "timestamp": timestamp,
        "request_id": f"request-{timestamp}",
        "question": f"question-{timestamp}",
        "answer": answer,
        "sources": ["source.txt"],
    }
    if include_contexts:
        entry["contexts"] = [] if contexts is None else contexts
    return entry


def _write_jsonl(path: Path, entries: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(entry, ensure_ascii=False) + "\n" for entry in entries),
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    ("entry", "expected"),
    [
        ({"answer": "   ", "contexts": ["evidence"]}, "empty_answer"),
        ({"answer": "answer", "contexts": []}, "retrieval_failure"),
        (
            {"answer": "Not found in knowledge base."},
            "generation_or_retrieval_failure",
        ),
        (
            {"answer": "未找到相关信息，请换个问题"},
            "generation_or_retrieval_failure",
        ),
        ({"answer": "answer"}, None),
    ],
)
def test_classify_log_entry(
    entry: dict[str, object],
    expected: str | None,
) -> None:
    assert export_badcases.classify_log_entry(entry) == expected


def test_empty_answer_has_priority_over_empty_contexts() -> None:
    assert (
        export_badcases.classify_log_entry({"answer": "", "contexts": []})
        == "empty_answer"
    )


def test_missing_contexts_does_not_trigger_retrieval_failure() -> None:
    assert export_badcases.classify_log_entry({"answer": "answer"}) is None


@pytest.mark.parametrize(
    "entry",
    [
        {"answer": None},
        {"answer": "answer", "contexts": None},
        {"answer": "answer", "contexts": [1]},
    ],
)
def test_classify_log_entry_rejects_invalid_types(entry: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        export_badcases.classify_log_entry(entry)


def test_export_adds_candidates_and_preserves_existing_annotations(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "logs" / "requests.jsonl"
    output_path = tmp_path / "eval" / "badcases.json"
    existing = {
        "question": "already reviewed",
        "answer": "Not found in knowledge base.",
        "expected_answer": "reviewed answer",
        "contexts": [],
        "sources": [],
        "error_type": "generation_or_retrieval_failure",
        "root_cause": "human diagnosis",
        "solution": "human solution",
        "timestamp": "t2",
    }
    output_path.parent.mkdir(parents=True)
    output_path.write_text(json.dumps([existing]), encoding="utf-8")
    _write_jsonl(
        log_path,
        [
            _log_entry(timestamp="t1"),
            _log_entry(timestamp="t2", answer="Not found in knowledge base."),
            _log_entry(timestamp="t3", include_contexts=True),
            _log_entry(timestamp="t4", answer="   "),
        ],
    )

    summary = export_badcases.export_badcases(log_path, output_path)
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert summary == export_badcases.ExportSummary(
        scanned=4,
        candidates=3,
        added=2,
        duplicates=1,
    )
    assert payload[0] == existing
    assert payload[0]["root_cause"] == "human diagnosis"
    assert [item["error_type"] for item in payload[1:]] == [
        "retrieval_failure",
        "empty_answer",
    ]


def test_export_creates_empty_output_when_log_is_missing(tmp_path: Path) -> None:
    output_path = tmp_path / "nested" / "badcases.json"

    summary = export_badcases.export_badcases(tmp_path / "missing.jsonl", output_path)

    assert summary == export_badcases.ExportSummary(0, 0, 0, 0)
    assert json.loads(output_path.read_text(encoding="utf-8")) == []


def test_export_deduplicates_repeated_timestamps_in_one_log(tmp_path: Path) -> None:
    log_path = tmp_path / "requests.jsonl"
    output_path = tmp_path / "badcases.json"
    entry = _log_entry(timestamp="same", answer="Not found in knowledge base.")
    _write_jsonl(log_path, [entry, entry])

    summary = export_badcases.export_badcases(log_path, output_path)

    assert summary == export_badcases.ExportSummary(2, 2, 1, 1)
    assert len(json.loads(output_path.read_text(encoding="utf-8"))) == 1


def test_export_reports_invalid_jsonl_line_number(tmp_path: Path) -> None:
    log_path = tmp_path / "requests.jsonl"
    log_path.write_text("\n{\n", encoding="utf-8")

    with pytest.raises(ValueError, match="line 2 contains invalid JSON"):
        export_badcases.export_badcases(log_path, tmp_path / "badcases.json")


def test_export_rejects_invalid_existing_output(tmp_path: Path) -> None:
    output_path = tmp_path / "badcases.json"
    output_path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="top level must be a list"):
        export_badcases.export_badcases(tmp_path / "missing.jsonl", output_path)


def test_parse_args_uses_repository_defaults() -> None:
    args = export_badcases.parse_args([])

    assert args.log_path == export_badcases.DEFAULT_LOG_PATH
    assert args.output_path == export_badcases.DEFAULT_OUTPUT_PATH


def test_main_prints_counts_without_request_content(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    log_path = tmp_path / "requests.jsonl"
    output_path = tmp_path / "badcases.json"
    _write_jsonl(
        log_path,
        [_log_entry(timestamp="private-time", answer="Not found in knowledge base.")],
    )

    result = export_badcases.main(
        ["--log-path", str(log_path), "--output-path", str(output_path)]
    )

    assert result == 0
    output = capsys.readouterr().out
    assert "scanned=1 candidates=1 added=1 duplicates=0" in output
    assert "question-private-time" not in output


def test_main_returns_nonzero_for_invalid_input(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    log_path = tmp_path / "requests.jsonl"
    log_path.write_text("{", encoding="utf-8")

    result = export_badcases.main(
        [
            "--log-path",
            str(log_path),
            "--output-path",
            str(tmp_path / "badcases.json"),
        ]
    )

    assert result == 1
    assert "ERROR: ValueError:" in capsys.readouterr().err
