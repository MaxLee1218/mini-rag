from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import replace
from numbers import Real

from app.conversation.models import ConversationTurn
from app.generator import generate_answer, load_deepseek_config_from_env
from app.query_rewriter.base import QueryRewriter, QueryRewriteResult


logger = logging.getLogger(__name__)

DEFAULT_MAX_HISTORY_TURNS = 5
DEFAULT_MAX_PROMPT_CHARS = 6000
DEFAULT_MAX_QUERY_CHARS = 500
DEFAULT_REWRITE_MAX_TOKENS = 128
MESSAGE_PREVIEW_CHARS = 400
CURRENT_QUERY_PREVIEW_CHARS = 1000

REWRITE_SYSTEM_INSTRUCTIONS = """你是一个查询重写器。

请根据最近的对话历史，将当前问题转换成适合检索、可以脱离对话独立理解的问题。

规则：
1. 当前问题依赖历史时才改写；已经独立完整时必须原样返回。
2. 保持用户原有语言、意图、范围、实体和约束。
3. 不回答问题，不推理最终答案，不添加用户没有询问的内容。
4. 只输出一个纯文本问题，不输出解释、Markdown、引号或候选列表。
5. 输出必须为单行。"""


def build_query_rewrite_prompt(
    query: str,
    history: list[ConversationTurn],
    *,
    max_history_turns: int = DEFAULT_MAX_HISTORY_TURNS,
    max_prompt_chars: int = DEFAULT_MAX_PROMPT_CHARS,
) -> str:
    """Build a bounded prompt for one context-aware query rewrite."""
    clean_query = _validate_query(query)
    if not isinstance(max_history_turns, int) or max_history_turns <= 0:
        raise ValueError("max_history_turns must be a positive integer")
    if not isinstance(max_prompt_chars, int) or max_prompt_chars <= 0:
        raise ValueError("max_prompt_chars must be a positive integer")

    history_lines: list[str] = []
    for turn in list(history)[-max_history_turns:]:
        history_lines.append(
            f"用户：{_preview(turn.user_message, MESSAGE_PREVIEW_CHARS)}"
        )
        history_lines.append(
            f"助手：{_preview(turn.assistant_message, MESSAGE_PREVIEW_CHARS)}"
        )

    history_text = "\n".join(history_lines) if history_lines else "（无历史对话）"
    prompt = (
        f"{REWRITE_SYSTEM_INSTRUCTIONS}\n\n"
        f"最近对话：\n{history_text}\n\n"
        f"当前问题：\n{_preview(clean_query, CURRENT_QUERY_PREVIEW_CHARS)}\n\n"
        "只输出最终的单个问题："
    )
    if len(prompt) > max_prompt_chars:
        raise ValueError("query rewrite prompt exceeds max_prompt_chars")
    return prompt


class DeepSeekQueryRewriteClient:
    """Call DeepSeek with query-rewrite-specific timeout and token settings."""

    def __init__(
        self,
        *,
        timeout: float,
        max_tokens: int = DEFAULT_REWRITE_MAX_TOKENS,
    ) -> None:
        if isinstance(timeout, bool) or not isinstance(timeout, Real) or timeout <= 0:
            raise ValueError("timeout must be a positive number")
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        self.timeout = float(timeout)
        self.max_tokens = max_tokens

    def __call__(self, prompt: str) -> str:
        config = replace(
            load_deepseek_config_from_env(),
            timeout=self.timeout,
        )
        return generate_answer(
            prompt,
            config=config,
            temperature=0.0,
            max_tokens=self.max_tokens,
        )


class LLMQueryRewriter(QueryRewriter):
    """Rewrite retrieval queries through an injected LLM completion call."""

    def __init__(
        self,
        completion_call: Callable[[str], str],
        *,
        max_history_turns: int = DEFAULT_MAX_HISTORY_TURNS,
        max_prompt_chars: int = DEFAULT_MAX_PROMPT_CHARS,
        max_query_chars: int = DEFAULT_MAX_QUERY_CHARS,
    ) -> None:
        if not callable(completion_call):
            raise ValueError("completion_call must be callable")
        if not isinstance(max_query_chars, int) or max_query_chars <= 0:
            raise ValueError("max_query_chars must be a positive integer")
        self._completion_call = completion_call
        self.max_history_turns = max_history_turns
        self.max_prompt_chars = max_prompt_chars
        self.max_query_chars = max_query_chars

    def rewrite(
        self,
        query: str,
        history: list[ConversationTurn],
    ) -> QueryRewriteResult:
        clean_query = _validate_query(query)
        try:
            prompt = build_query_rewrite_prompt(
                clean_query,
                list(history),
                max_history_turns=self.max_history_turns,
                max_prompt_chars=self.max_prompt_chars,
            )
            output = self._completion_call(prompt)
            rewritten = _validate_provider_output(output, self.max_query_chars)
        except Exception as error:
            logger.warning(
                "query_rewrite_provider_failed",
                extra={"error_type": type(error).__name__},
            )
            return QueryRewriteResult(
                clean_query,
                clean_query,
                False,
                "llm_rewrite_failed",
            )

        was_rewritten = rewritten != clean_query
        return QueryRewriteResult(
            clean_query,
            rewritten,
            was_rewritten,
            "llm_rewrite" if was_rewritten else "llm_unchanged",
        )


def _validate_query(query: str) -> str:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must not be blank")
    return query.strip()


def _validate_provider_output(output: str, max_query_chars: int) -> str:
    if not isinstance(output, str):
        raise ValueError("LLM rewrite must be a string")
    rewritten = output.strip()
    if not rewritten:
        raise ValueError("LLM rewrite must not be blank")
    if "\n" in rewritten or "\r" in rewritten:
        raise ValueError("LLM rewrite must be one line")
    if len(rewritten) > max_query_chars:
        raise ValueError("LLM rewrite exceeds max_query_chars")
    if (
        rewritten.startswith(("#", "- ", "* ", "**", "`"))
        or re.match(r"^\d+[.)]\s", rewritten)
        or (
            len(rewritten) >= 2
            and rewritten[0] == rewritten[-1]
            and rewritten[0] in {'"', "'", "`"}
        )
    ):
        raise ValueError("LLM rewrite must be plain text without formatting")
    return rewritten


def _preview(value: str, limit: int) -> str:
    clean_value = " ".join(value.split())
    return clean_value[:limit]
