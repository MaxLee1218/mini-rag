from __future__ import annotations

import json
from pathlib import Path

from app.logging_utils import RequestLog, log_request


def _request_log(*, status: str = "success", error_type: str | None = None) -> RequestLog:
    return {
        "timestamp": "2026-07-10T10:00:00+00:00",
        "request_id": "00000000-0000-4000-8000-000000000001",
        "question": "What is RAG?",
        "answer": "Retrieval-Augmented Generation.",
        "sources": ["rag_notes.md"],
        "latency_ms": 1234,
        "status": status,
        "error_type": error_type,
        "session_id": "session-1",
        "original_question": "What is RAG?",
        "rewritten_query": "What is RAG?",
        "query_was_rewritten": False,
        "rewrite_reason": "independent_query",
        "history_turn_count": 0,
    }


def test_log_request_creates_parseable_jsonl_entry(tmp_path):
    log_file = tmp_path / "logs" / "rag_requests.jsonl"
    entry = _request_log()

    log_request(entry, log_file=log_file)

    assert log_file.exists()
    assert json.loads(log_file.read_text(encoding="utf-8").strip()) == entry


def test_log_request_appends_multiple_entries(tmp_path):
    log_file = tmp_path / "logs" / "rag_requests.jsonl"
    success_entry = _request_log()
    error_entry = _request_log(status="error", error_type="RuntimeError")
    error_entry["answer"] = ""
    error_entry["sources"] = []

    log_request(success_entry, log_file=log_file)
    log_request(error_entry, log_file=log_file)

    lines = log_file.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in lines] == [success_entry, error_entry]


def test_log_request_keeps_stable_error_fields(tmp_path):
    log_file = tmp_path / "logs" / "rag_requests.jsonl"
    entry = _request_log(status="error", error_type="ValueError")
    entry["answer"] = ""
    entry["sources"] = []

    log_request(entry, log_file=log_file)

    logged_entry = json.loads(log_file.read_text(encoding="utf-8").strip())
    assert set(logged_entry) == set(RequestLog.__annotations__)
    assert logged_entry["status"] == "error"
    assert logged_entry["error_type"] == "ValueError"
    assert logged_entry["answer"] == ""
    assert logged_entry["sources"] == []


def test_log_request_swallows_write_errors(monkeypatch, tmp_path):
    log_file = tmp_path / "logs" / "rag_requests.jsonl"

    def fail_open(*args, **kwargs):
        raise OSError("write failed")

    monkeypatch.setattr(Path, "open", fail_open)

    log_request(_request_log(), log_file=log_file)
