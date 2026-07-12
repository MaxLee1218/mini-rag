from __future__ import annotations

import importlib
import builtins
from types import SimpleNamespace

from fastapi.testclient import TestClient


def test_importing_api_does_not_build_real_pipeline(monkeypatch):
    calls = []

    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        blocked_modules = {"app.embeddings", "app.vector_store"}
        if name in blocked_modules:
            raise AssertionError(f"{name} should not be imported by app.api")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    import app.config as app_config

    import app.pipeline_factory as pipeline_factory

    monkeypatch.setattr(
        app_config,
        "require_deepseek_api_key",
        lambda: (_ for _ in ()).throw(
            AssertionError("DeepSeek key should not be checked on app.api import")
        ),
    )

    def fail_if_called(top_k):
        calls.append(top_k)
        raise AssertionError("build_default_pipeline should not run on import")

    monkeypatch.setattr(pipeline_factory, "build_default_pipeline", fail_if_called)

    import app.api as api_module

    importlib.reload(api_module)

    assert calls == []
    assert api_module._pipeline is None


def test_health_returns_service_status():
    from app.api import app, get_pipeline

    app.dependency_overrides[get_pipeline] = lambda: _FakePipeline()
    try:
        response = TestClient(app).get("/health")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "mini-rag-api",
        "version": "0.1.0",
    }


def test_ask_returns_answer_sources_and_latency():
    from app.api import app, get_pipeline

    fake_pipeline = _FakePipeline(
        SimpleNamespace(
            question="RAG是什么？",
            answer="RAG 是检索增强生成。",
            sources=["rag_notes.md"],
            contexts=[],
        )
    )
    app.dependency_overrides[get_pipeline] = lambda: fake_pipeline
    try:
        response = TestClient(app).post("/ask", json={"question": "RAG是什么？"})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["question"] == "RAG是什么？"
    assert body["answer"] == "RAG 是检索增强生成。"
    assert body["sources"] == [
        {
            "index": 1,
            "source": "rag_notes.md",
            "metadata": None,
            "text_preview": None,
        }
    ]
    assert isinstance(body["latency_ms"], float)
    assert body["latency_ms"] >= 0
    assert fake_pipeline.calls == [{"question": "RAG是什么？", "top_k": 5}]


def test_ask_passes_requested_top_k_to_pipeline():
    from app.api import app, get_pipeline

    fake_pipeline = _FakePipeline()
    app.dependency_overrides[get_pipeline] = lambda: fake_pipeline
    try:
        response = TestClient(app).post(
            "/ask",
            json={"question": "RAG是什么？", "top_k": 3},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert fake_pipeline.calls == [{"question": "RAG是什么？", "top_k": 3}]


def test_ask_rejects_blank_question():
    from app.api import app, get_pipeline

    app.dependency_overrides[get_pipeline] = lambda: _FakePipeline()
    try:
        response = TestClient(app).post("/ask", json={"question": "   "})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 422


def test_ask_rejects_invalid_top_k():
    from app.api import app, get_pipeline

    app.dependency_overrides[get_pipeline] = lambda: _FakePipeline()
    try:
        response = TestClient(app).post(
            "/ask",
            json={"question": "RAG是什么？", "top_k": 21},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 422


def test_ask_returns_safe_error_when_pipeline_fails():
    from app.api import app, get_pipeline

    app.dependency_overrides[get_pipeline] = lambda: _FailingPipeline()
    try:
        response = TestClient(app).post("/ask", json={"question": "RAG是什么？"})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 500
    assert response.json() == {"detail": "RAG pipeline failed."}


def test_normalize_sources_supports_context_fallback_and_truncates_preview():
    from app.api import normalize_sources

    long_text = "正文" * 150
    result = SimpleNamespace(
        sources=[],
        contexts=[
            {
                "document": long_text,
                "metadata": {"file_path": "docs/rag.md", "chunk_id": 2},
            },
            SimpleNamespace(
                page_content="对象正文",
                metadata={"filename": "object.md"},
            ),
        ],
    )

    sources = normalize_sources(result)

    assert [source.source for source in sources] == ["docs/rag.md", "object.md"]
    assert sources[0].index == 1
    assert sources[0].metadata == {"file_path": "docs/rag.md", "chunk_id": 2}
    assert sources[0].text_preview is not None
    assert len(sources[0].text_preview) == 200
    assert sources[1].text_preview == "对象正文"


def test_normalize_sources_supports_source_dicts_and_objects():
    from app.api import normalize_sources

    source_object = SimpleNamespace(
        source="object.md",
        metadata={"chunk_id": 3},
        text="object text",
    )
    result = SimpleNamespace(
        sources=[
            {"source": "dict.md", "metadata": {"chunk_id": 1}, "text": "dict text"},
            source_object,
        ],
        contexts=[],
    )

    sources = normalize_sources(result)

    assert sources[0].source == "dict.md"
    assert sources[0].metadata == {"chunk_id": 1}
    assert sources[0].text_preview == "dict text"
    assert sources[1].source == "object.md"
    assert sources[1].metadata == {"chunk_id": 3}
    assert sources[1].text_preview == "object text"


class _FakePipeline:
    def __init__(self, result=None):
        self.result = result or SimpleNamespace(
            question="RAG是什么？",
            answer="fake answer",
            sources=[],
            contexts=[],
        )
        self.calls = []

    def ask(self, question, top_k=None):
        self.calls.append({"question": question, "top_k": top_k})
        return self.result


class _FailingPipeline:
    def ask(self, question, top_k=None):
        raise RuntimeError(
            "boom with key sk-secret and path C:\\Users\\mingx\\project\\.env"
        )


def test_ask_logs_success(monkeypatch):
    from uuid import UUID

    import app.api as api_module

    logged_entries = []
    monkeypatch.setattr(api_module, "log_request", logged_entries.append)
    api_module.app.dependency_overrides[api_module.get_pipeline] = lambda: _FakePipeline(
        SimpleNamespace(
            question="What is RAG?",
            answer="RAG means Retrieval-Augmented Generation.",
            sources=["rag_notes.md"],
            contexts=[],
        )
    )
    try:
        response = TestClient(api_module.app).post(
            "/ask", json={"question": "What is RAG?"}
        )
    finally:
        api_module.app.dependency_overrides.clear()

    assert response.status_code == 200
    assert len(logged_entries) == 1
    log_entry = logged_entries[0]
    assert log_entry["question"] == "What is RAG?"
    assert log_entry["answer"] == "RAG means Retrieval-Augmented Generation."
    assert log_entry["sources"] == ["rag_notes.md"]
    assert log_entry["status"] == "success"
    assert log_entry["error_type"] is None
    assert isinstance(log_entry["latency_ms"], int)
    assert UUID(log_entry["request_id"]).version == 4


def test_ask_logs_error_without_sensitive_exception_details(monkeypatch):
    import json

    import app.api as api_module

    logged_entries = []
    monkeypatch.setattr(api_module, "log_request", logged_entries.append)
    api_module.app.dependency_overrides[api_module.get_pipeline] = _FailingPipeline
    try:
        response = TestClient(api_module.app).post(
            "/ask", json={"question": "What is RAG?"}
        )
    finally:
        api_module.app.dependency_overrides.clear()

    assert response.status_code == 500
    assert response.json() == {"detail": "RAG pipeline failed."}
    assert len(logged_entries) == 1
    log_entry = logged_entries[0]
    assert log_entry["question"] == "What is RAG?"
    assert log_entry["answer"] == ""
    assert log_entry["sources"] == []
    assert log_entry["status"] == "error"
    assert log_entry["error_type"] == "RuntimeError"
    assert "sk-secret" not in json.dumps(log_entry)
