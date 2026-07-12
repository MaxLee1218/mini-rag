from __future__ import annotations

import re

from app.conversation.models import ConversationTurn
from app.query_rewriter.base import QueryRewriter, QueryRewriteResult


_REFERENCE_MARKERS = (
    "这种方法",
    "上一个",
    "这个",
    "那个",
    "它们",
    "他们",
    "她们",
    "它",
    "他",
    "她",
)
_CONTINUATION_MARKERS = ("继续", "上面", "刚才", "前面", "那为什么", "然后呢")
_QUESTION_PREFIXES = (
    "请介绍",
    "请解释",
    "什么是",
    "什么叫",
    "如何理解",
)


class RuleBasedQueryRewriter(QueryRewriter):
    """Conservatively replace clear contextual references with a recent topic."""

    def __init__(self, max_history_turns: int = 5, max_query_chars: int = 500) -> None:
        if not 1 <= max_history_turns <= 5:
            raise ValueError("max_history_turns must be between 1 and 5")
        if max_query_chars <= 0:
            raise ValueError("max_query_chars must be positive")
        self.max_history_turns = max_history_turns
        self.max_query_chars = max_query_chars

    def rewrite(
        self,
        query: str,
        history: list[ConversationTurn],
    ) -> QueryRewriteResult:
        clean_query = _validate_query(query)
        relevant_history = list(history[-self.max_history_turns :])
        contextual = any(marker in clean_query for marker in _REFERENCE_MARKERS)
        contextual = contextual or any(
            marker in clean_query for marker in _CONTINUATION_MARKERS
        )
        numeric_follow_up = bool(
            re.search(r"我.*(?:年后|以后).*几岁", clean_query)
        )
        if not contextual and not numeric_follow_up:
            return _unchanged(clean_query, "independent_query")
        if not relevant_history:
            return _unchanged(clean_query, "context_required_but_no_history")

        previous_user_message = relevant_history[-1].user_message.strip()
        if numeric_follow_up:
            rewritten = f"{previous_user_message.rstrip('。！？?')}，{clean_query}"
        else:
            topic = _extract_topic(previous_user_message)
            if not topic:
                return _unchanged(clean_query, "no_reliable_history_topic")
            rewritten = clean_query
            replaced = False
            for marker in _REFERENCE_MARKERS:
                if marker not in rewritten:
                    continue
                marker_index = rewritten.index(marker)
                replacement = "" if topic in rewritten else topic
                following_index = marker_index + len(marker)
                if (
                    replacement
                    and following_index < len(rewritten)
                    and not rewritten[following_index].isspace()
                ):
                    replacement += " "
                rewritten = rewritten.replace(marker, replacement, 1)
                replaced = True
                break
            if not replaced:
                rewritten = f"关于{topic}，{clean_query}"

        rewritten = re.sub(r"\s+", " ", rewritten).strip()[: self.max_query_chars]
        if not rewritten or rewritten == clean_query:
            return _unchanged(clean_query, "rewrite_not_needed")
        return QueryRewriteResult(clean_query, rewritten, True, "context_reference")


def _validate_query(query: str) -> str:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must not be blank")
    return query.strip()


def _extract_topic(message: str) -> str:
    topic = message.strip()
    for prefix in _QUESTION_PREFIXES:
        if topic.startswith(prefix):
            topic = topic[len(prefix) :]
            break
    topic = topic.strip(" ，,。！？?!：:")
    if not topic:
        return ""
    first_clause = re.split(r"[，,。！？?!；;]", topic, maxsplit=1)[0].strip()
    return first_clause[:200]


def _unchanged(query: str, reason: str) -> QueryRewriteResult:
    return QueryRewriteResult(query, query, False, reason)
