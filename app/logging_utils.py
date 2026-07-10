from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, TypedDict

from app.config import LOG_FILE


class RequestLog(TypedDict):
    timestamp: str
    request_id: str
    question: str
    answer: str
    sources: list[str]
    latency_ms: int
    status: Literal["success", "error"]
    error_type: str | None


def log_request(log_entry: RequestLog, log_file: Path | None = None) -> None:
    """Append one request log entry without affecting the RAG request."""
    try:
        destination = LOG_FILE if log_file is None else log_file
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("a", encoding="utf-8") as file:
            json.dump(log_entry, file, ensure_ascii=False)
            file.write("\n")
    except Exception:
        return
