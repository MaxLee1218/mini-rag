from __future__ import annotations

from app.config import (
    CONVERSATION_HISTORY_LIMIT,
    QUERY_REWRITE_PROVIDER,
    QUERY_REWRITE_TIMEOUT,
)
from app.conversation.memory_store import InMemoryConversationStore
from app.conversation.store import ConversationStore
from app.query_rewriter.base import QueryRewriter
from app.query_rewriter.llm_rewriter import (
    DeepSeekQueryRewriteClient,
    LLMQueryRewriter,
)


conversation_store: ConversationStore = InMemoryConversationStore(
    max_turns=CONVERSATION_HISTORY_LIMIT
)

if QUERY_REWRITE_PROVIDER == "deepseek":
    query_rewriter: QueryRewriter = LLMQueryRewriter(
        completion_call=DeepSeekQueryRewriteClient(
            timeout=QUERY_REWRITE_TIMEOUT,
        ),
        max_history_turns=CONVERSATION_HISTORY_LIMIT,
    )
else:  # Configuration validation should make this unreachable.
    raise RuntimeError(f"Unsupported query rewrite provider: {QUERY_REWRITE_PROVIDER}")


def get_conversation_store() -> ConversationStore:
    """Return the process-wide conversation store dependency."""
    return conversation_store
