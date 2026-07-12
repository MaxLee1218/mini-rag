from datetime import datetime, timezone

import pytest

from app.conversation.models import ConversationTurn
from app.generator import DeepSeekConfig
from app.query_rewriter.llm_rewriter import (
    DeepSeekQueryRewriteClient,
    LLMQueryRewriter,
)
from app.query_rewriter.rule_based import RuleBasedQueryRewriter


def history(user: str, assistant: str = "回答") -> list[ConversationTurn]:
    return [ConversationTurn(user, assistant, datetime.now(timezone.utc))]


def test_independent_question_is_not_rewritten():
    result = RuleBasedQueryRewriter().rewrite(
        "什么是 FastAPI Middleware？",
        history("什么是 RAG？"),
    )
    assert result.rewritten_query == result.original_query
    assert result.was_rewritten is False


def test_pronoun_query_is_rewritten_with_middleware_topic():
    result = RuleBasedQueryRewriter().rewrite(
        "它为什么可以优化查询？",
        history("什么是 Middleware？", "Middleware 是中间处理层。"),
    )
    assert "Middleware" in result.rewritten_query
    assert result.rewritten_query != result.original_query
    assert result.was_rewritten is True


def test_this_reference_is_rewritten_with_hybrid_retrieval_topic():
    result = RuleBasedQueryRewriter().rewrite(
        "这个有什么缺点？",
        history("什么是混合检索？"),
    )
    assert "混合检索" in result.rewritten_query
    assert result.was_rewritten is True


def test_context_dependent_query_without_history_is_not_invented():
    result = RuleBasedQueryRewriter().rewrite("它有什么作用？", [])
    assert result.rewritten_query == result.original_query
    assert result.was_rewritten is False
    assert "history" in result.reason


def test_age_follow_up_adds_background_without_answering():
    result = RuleBasedQueryRewriter().rewrite(
        "我三年后几岁？",
        history("我今年 24 岁。", "你今年 24 岁。"),
    )
    assert "24" in result.rewritten_query
    assert "三年后" in result.rewritten_query
    assert result.rewritten_query.strip() != "27 岁"
    assert result.was_rewritten is True


def test_rewritten_query_is_bounded_and_does_not_copy_full_history():
    long_message = "超长历史主题" * 200
    result = RuleBasedQueryRewriter(max_query_chars=500).rewrite(
        "这个有什么缺点？",
        history(long_message),
    )
    assert len(result.rewritten_query) <= 500
    assert result.rewritten_query != long_message + "这个有什么缺点？"


def test_topic_is_not_repeated_when_query_already_contains_it():
    result = RuleBasedQueryRewriter().rewrite(
        "Middleware 这个组件有什么缺点？",
        history("什么是 Middleware？"),
    )
    assert result.rewritten_query.count("Middleware") == 1


@pytest.mark.parametrize("query", ["", "   "])
def test_blank_query_is_rejected(query):
    with pytest.raises(ValueError, match="query must not be blank"):
        RuleBasedQueryRewriter().rewrite(query, [])


def test_llm_rewriter_calls_model_for_independent_question():
    prompts = []
    rewriter = LLMQueryRewriter(
        lambda prompt: prompts.append(prompt) or "什么是 RAG？"
    )

    result = rewriter.rewrite("什么是 RAG？", [])

    assert len(prompts) == 1
    assert "什么是 RAG？" in prompts[0]
    assert result.rewritten_query == "什么是 RAG？"
    assert result.was_rewritten is False
    assert result.reason == "llm_unchanged"


def test_llm_rewriter_passes_history_and_returns_contextual_rewrite():
    prompts = []
    rewriter = LLMQueryRewriter(
        lambda prompt: prompts.append(prompt) or "How old is laojingqiao?"
    )

    result = rewriter.rewrite(
        "how old is he?",
        history("who is my teacher", "Your teacher is laojingqiao."),
    )

    assert "who is my teacher" in prompts[0]
    assert "Your teacher is laojingqiao." in prompts[0]
    assert result.rewritten_query == "How old is laojingqiao?"
    assert result.was_rewritten is True
    assert result.reason == "llm_rewrite"


@pytest.mark.parametrize(
    "provider_output",
    [
        "",
        "   ",
        "query one\nquery two",
        "x" * 501,
        "**How old is laojingqiao?**",
        '"How old is laojingqiao?"',
        "1. How old is laojingqiao?",
    ],
)
def test_llm_rewriter_invalid_output_falls_back_to_original(provider_output):
    result = LLMQueryRewriter(lambda prompt: provider_output).rewrite(
        "how old is he?",
        history("who is my teacher", "Your teacher is laojingqiao."),
    )

    assert result.rewritten_query == "how old is he?"
    assert result.was_rewritten is False
    assert result.reason == "llm_rewrite_failed"


def test_llm_rewriter_provider_exception_falls_back_to_original():
    def fail(_prompt):
        raise TimeoutError("timed out")

    result = LLMQueryRewriter(fail).rewrite("how old is he?", [])

    assert result.rewritten_query == "how old is he?"
    assert result.was_rewritten is False
    assert result.reason == "llm_rewrite_failed"


def test_llm_rewriter_bounds_history_and_prompt_length():
    turns = [
        ConversationTurn(
            user_message=f"old-user-{number}-" + "u" * 800,
            assistant_message=f"old-assistant-{number}-" + "a" * 800,
            created_at=datetime.now(timezone.utc),
        )
        for number in range(6)
    ]
    prompts = []
    rewriter = LLMQueryRewriter(
        lambda prompt: prompts.append(prompt) or "current question"
    )

    rewriter.rewrite("current question", turns)

    assert len(prompts[0]) <= 6000
    assert "old-user-0" not in prompts[0]
    assert "old-user-1" in prompts[0]
    assert "old-user-5" in prompts[0]


@pytest.mark.parametrize("query", ["", "   "])
def test_llm_rewriter_rejects_blank_query(query):
    with pytest.raises(ValueError, match="query must not be blank"):
        LLMQueryRewriter(lambda prompt: "unused").rewrite(query, [])


def test_deepseek_rewrite_client_uses_rewrite_specific_settings(monkeypatch):
    captured = {}
    base_config = DeepSeekConfig(
        api_key="test-key",
        base_url="https://deepseek.test",
        model="test-model",
        timeout=30.0,
    )

    monkeypatch.setattr(
        "app.query_rewriter.llm_rewriter.load_deepseek_config_from_env",
        lambda: base_config,
    )

    def fake_generate(prompt, *, config, temperature, max_tokens):
        captured.update(
            prompt=prompt,
            config=config,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return "rewritten question"

    monkeypatch.setattr(
        "app.query_rewriter.llm_rewriter.generate_answer",
        fake_generate,
    )

    result = DeepSeekQueryRewriteClient(timeout=7.5)("rewrite prompt")

    assert result == "rewritten question"
    assert captured["prompt"] == "rewrite prompt"
    assert captured["config"] == DeepSeekConfig(
        api_key="test-key",
        base_url="https://deepseek.test",
        model="test-model",
        timeout=7.5,
    )
    assert captured["temperature"] == 0.0
    assert captured["max_tokens"] == 128
