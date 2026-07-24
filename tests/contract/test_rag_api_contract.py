from __future__ import annotations

from types import SimpleNamespace
from uuid import UUID

from app.schemas import AskResponse, HealthResponse
from tests.asgi_client import asgi_request


class ContractPipeline:
    def ask(
        self,
        question: str,
        top_k: int | None = None,
        *,
        retrieval_query: str | None = None,
        session_id: str | None = None,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            answer="RAG is retrieval-augmented generation.",
            sources=["docs/rag.md"],
            contexts=[
                {
                    "content": "RAG retrieves evidence before generation.",
                    "metadata": {
                        "source": "docs/rag.md",
                        "chunk_id": "rag-1",
                    },
                    "score": 0.9,
                }
            ],
            route="rag",
        )


def test_health_returns_health_response() -> None:
    from app.api import app

    response = asgi_request(app, "GET", "/health")

    assert response.status_code == 200
    HealthResponse.model_validate(response.json())


def test_ask_accepts_request_and_returns_frozen_contract() -> None:
    from app.api import app, get_pipeline

    app.dependency_overrides[get_pipeline] = ContractPipeline
    try:
        response = asgi_request(
            app,
            "POST",
            "/ask",
            headers={"X-Trace-ID": "copilot-request-123"},
            json={"question": "What is RAG?"},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    AskResponse.model_validate(body)
    assert set(body) == {
        "answer",
        "sources",
        "contexts",
        "route",
        "latency_ms",
        "rag_trace_id",
    }
    assert body["answer"] == "RAG is retrieval-augmented generation."
    assert body["sources"]
    assert body["contexts"]
    assert body["route"] == "rag"
    assert body["latency_ms"] >= 0
    assert body["rag_trace_id"] == "copilot-request-123"


def test_ask_generates_trace_id_when_header_is_absent() -> None:
    from app.api import app, get_pipeline

    app.dependency_overrides[get_pipeline] = ContractPipeline
    try:
        response = asgi_request(
            app,
            "POST",
            "/ask",
            json={"question": "What is RAG?"},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert UUID(response.json()["rag_trace_id"]).version == 4
