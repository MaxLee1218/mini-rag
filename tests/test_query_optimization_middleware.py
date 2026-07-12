import asyncio
from datetime import datetime, timezone

import httpx
from fastapi import FastAPI, Request
from pydantic import BaseModel

from app.conversation.memory_store import InMemoryConversationStore
from app.conversation.models import ConversationTurn
from app.middleware.query_optimization import QueryOptimizationMiddleware
from app.query_rewriter.base import QueryRewriteResult


class Body(BaseModel):
    question: str
    session_id: str


class FakeQueryRewriter:
    def __init__(self, rewritten_query="独立查询", error=None):
        self.rewritten_query = rewritten_query
        self.error = error
        self.calls = []

    def rewrite(self, query, history):
        self.calls.append((query, list(history)))
        if self.error:
            raise self.error
        return QueryRewriteResult(query, self.rewritten_query, True, "fake")


def make_app(store, rewriter, *, enabled=True):
    app = FastAPI()
    app.add_middleware(
        QueryOptimizationMiddleware,
        conversation_store=store,
        query_rewriter=rewriter,
        history_limit=5,
        enabled=enabled,
        paths={"/ask"},
    )

    @app.post("/ask")
    async def ask(body: Body, request: Request):
        return {
            "question": body.question,
            "session_id": body.session_id,
            "original_question": request.state.original_question,
            "rewritten_query": request.state.rewritten_query,
            "was_rewritten": request.state.query_was_rewritten,
            "history_count": len(request.state.conversation_history),
        }

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app


def request(app, method, path, **kwargs):
    async def send():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.request(method, path, **kwargs)

    return asyncio.run(send())


def test_middleware_writes_rewrite_result_to_request_state():
    rewriter = FakeQueryRewriter("Middleware 有什么作用？")
    response = request(
        make_app(InMemoryConversationStore(), rewriter),
        "POST",
        "/ask",
        json={"question": "它有什么作用？", "session_id": "session-a"},
    )
    assert response.status_code == 200
    assert response.json()["rewritten_query"] == "Middleware 有什么作用？"
    assert response.json()["original_question"] == "它有什么作用？"


def test_middleware_reads_only_requested_session_history():
    store = InMemoryConversationStore()
    expected = ConversationTurn("什么是 Middleware？", "中间层", datetime.now(timezone.utc))
    store.append_turn("session-a", expected)
    store.append_turn(
        "session-b",
        ConversationTurn("德国在哪里？", "欧洲", datetime.now(timezone.utc)),
    )
    rewriter = FakeQueryRewriter()
    request(
        make_app(store, rewriter),
        "POST",
        "/ask",
        json={"question": "它是什么？", "session_id": "session-a"},
    )
    assert rewriter.calls == [("它是什么？", [expected])]


def test_different_session_does_not_receive_other_history():
    store = InMemoryConversationStore()
    store.append_turn(
        "session-a",
        ConversationTurn("什么是 Middleware？", "中间层", datetime.now(timezone.utc)),
    )
    rewriter = FakeQueryRewriter()
    request(
        make_app(store, rewriter),
        "POST",
        "/ask",
        json={"question": "它是什么？", "session_id": "session-b"},
    )
    assert rewriter.calls == [("它是什么？", [])]


def test_non_target_path_does_not_call_rewriter():
    rewriter = FakeQueryRewriter()
    response = request(
        make_app(InMemoryConversationStore(), rewriter), "GET", "/health"
    )
    assert response.status_code == 200
    assert rewriter.calls == []


def test_rewriter_exception_falls_back_to_original_question():
    rewriter = FakeQueryRewriter(error=RuntimeError("boom"))
    response = request(
        make_app(InMemoryConversationStore(), rewriter),
        "POST",
        "/ask",
        json={"question": "它是什么？", "session_id": "session-a"},
    )
    assert response.status_code == 200
    assert response.json()["rewritten_query"] == "它是什么？"
    assert response.json()["was_rewritten"] is False


def test_disabled_middleware_does_not_call_rewriter():
    rewriter = FakeQueryRewriter()
    response = request(
        make_app(InMemoryConversationStore(), rewriter, enabled=False),
        "POST",
        "/ask",
        json={"question": "它是什么？", "session_id": "session-a"},
    )
    assert response.status_code == 200
    assert response.json()["rewritten_query"] == "它是什么？"
    assert rewriter.calls == []


def test_request_body_remains_available_to_pydantic_route():
    response = request(
        make_app(InMemoryConversationStore(), FakeQueryRewriter()),
        "POST",
        "/ask",
        json={"question": "问题", "session_id": "session-a"},
    )
    assert response.status_code == 200
    assert response.json()["question"] == "问题"
    assert response.json()["session_id"] == "session-a"


def test_invalid_body_is_left_to_fastapi_validation():
    app = make_app(InMemoryConversationStore(), FakeQueryRewriter())
    assert request(
        app,
        "POST",
        "/ask",
        content="not-json",
        headers={"content-type": "application/json"},
    ).status_code == 422
    assert request(
        app, "POST", "/ask", json={"session_id": "session-a"}
    ).status_code == 422
