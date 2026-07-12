from types import SimpleNamespace

from app.query_rewriter.llm_rewriter import LLMQueryRewriter
from tests.asgi_client import asgi_request


class FakePipeline:
    def __init__(self):
        self.calls = []
        self.fail = False

    def ask(self, question, top_k=None, *, retrieval_query=None):
        self.calls.append(
            {
                "question": question,
                "top_k": top_k,
                "retrieval_query": retrieval_query,
            }
        )
        if self.fail:
            raise RuntimeError("boom")
        return SimpleNamespace(
            question=question,
            answer=f"answer: {question}",
            sources=[],
            contexts=[],
        )


def setup_api(completion_call=None):
    import app.api as api_module

    assert isinstance(api_module.query_rewriter, LLMQueryRewriter)
    original_completion_call = api_module.query_rewriter._completion_call
    rewrite_prompts = []

    def fake_completion(prompt):
        rewrite_prompts.append(prompt)
        current_question = prompt.split("当前问题：\n", 1)[1].split("\n\n", 1)[0]
        if (
            current_question == "它为什么可以优化查询？"
            and "用户：什么是 Middleware？" in prompt
        ):
            return "Middleware 为什么可以优化查询？"
        return current_question

    api_module.query_rewriter._completion_call = completion_call or fake_completion
    store = api_module.get_conversation_store()
    for session_id in ("session-1", "session-2", "session-evict", "session-fail"):
        store.clear_session(session_id)
    pipeline = FakePipeline()
    pipeline.rewrite_prompts = rewrite_prompts
    pipeline.original_completion_call = original_completion_call
    api_module.app.dependency_overrides[api_module.get_pipeline] = lambda: pipeline
    return api_module, store, pipeline


def teardown_api(api_module, pipeline):
    api_module.query_rewriter._completion_call = pipeline.original_completion_call
    api_module.app.dependency_overrides.clear()


def test_first_and_second_round_use_history_and_preserve_original_question():
    api_module, store, pipeline = setup_api()
    try:
        first = asgi_request(
            api_module.app,
            "POST",
            "/ask",
            json={"session_id": "session-1", "question": "什么是 Middleware？"},
        )
        second = asgi_request(
            api_module.app,
            "POST",
            "/ask",
            json={"session_id": "session-1", "question": "它为什么可以优化查询？"},
        )
    finally:
        teardown_api(api_module, pipeline)

    assert first.status_code == 200
    assert first.json()["query_was_rewritten"] is False
    assert len(store.get_recent_turns("session-1", 5)) == 2
    assert len(pipeline.rewrite_prompts) == 2
    assert "用户：什么是 Middleware？" in pipeline.rewrite_prompts[1]
    assert "Middleware" in pipeline.calls[1]["retrieval_query"]
    assert second.json()["question"] == "它为什么可以优化查询？"
    assert second.json()["session_id"] == "session-1"


def test_new_session_cannot_read_old_session_history():
    api_module, store, pipeline = setup_api()
    try:
        asgi_request(
            api_module.app,
            "POST",
            "/ask",
            json={"session_id": "session-1", "question": "什么是 Middleware？"},
        )
        response = asgi_request(
            api_module.app,
            "POST",
            "/ask",
            json={"session_id": "session-2", "question": "它有什么作用？"},
        )
    finally:
        teardown_api(api_module, pipeline)

    assert response.status_code == 200
    assert pipeline.calls[-1]["retrieval_query"] == "它有什么作用？"
    assert response.json()["query_was_rewritten"] is False
    assert len(store.get_recent_turns("session-2", 5)) == 1


def test_session_keeps_only_five_successful_turns():
    api_module, store, _ = setup_api()
    try:
        for number in range(1, 7):
            response = asgi_request(
                api_module.app,
                "POST",
                "/ask",
                json={
                    "session_id": "session-evict",
                    "question": f"独立问题 {number}",
                },
            )
            assert response.status_code == 200
    finally:
        teardown_api(api_module, _)

    turns = store.get_recent_turns("session-evict", 5)
    assert len(turns) == 5
    assert [turn.user_message for turn in turns] == [f"独立问题 {n}" for n in range(2, 7)]


def test_failed_question_is_not_stored():
    api_module, store, pipeline = setup_api()
    pipeline.fail = True
    try:
        response = asgi_request(
            api_module.app,
            "POST",
            "/ask",
            json={"session_id": "session-fail", "question": "问题"},
        )
    finally:
        teardown_api(api_module, pipeline)
    assert response.status_code == 500
    assert store.get_recent_turns("session-fail", 5) == []


def test_blank_question_and_session_are_rejected_and_health_is_unchanged():
    api_module, _, pipeline = setup_api()
    try:
        blank_question = asgi_request(
            api_module.app,
            "POST",
            "/ask",
            json={"session_id": "session-1", "question": "   "},
        )
        blank_session = asgi_request(
            api_module.app,
            "POST",
            "/ask",
            json={"session_id": "   ", "question": "问题"},
        )
        health = asgi_request(api_module.app, "GET", "/health")
    finally:
        teardown_api(api_module, pipeline)
    assert blank_question.status_code == 422
    assert blank_session.status_code == 422
    assert health.status_code == 200
    assert pipeline.calls == []


def test_query_rewrite_timeout_falls_back_without_failing_request():
    def timeout(_prompt):
        raise TimeoutError("timed out")

    api_module, store, pipeline = setup_api(completion_call=timeout)
    try:
        response = asgi_request(
            api_module.app,
            "POST",
            "/ask",
            json={"session_id": "session-1", "question": "how old is he?"},
        )
    finally:
        teardown_api(api_module, pipeline)

    assert response.status_code == 200
    assert pipeline.calls[-1]["retrieval_query"] == "how old is he?"
    assert response.json()["query_was_rewritten"] is False
    assert len(store.get_recent_turns("session-1", 5)) == 1
