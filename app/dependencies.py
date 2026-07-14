from __future__ import annotations

from app.config import (
    CONVERSATION_HISTORY_LIMIT,
    QUERY_REWRITE_ENABLED,
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
from app.query_preparation import ConversationQueryPreparer


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

query_preparer = ConversationQueryPreparer(
    store=conversation_store,
    rewriter=query_rewriter,
    history_limit=CONVERSATION_HISTORY_LIMIT,
    enabled=QUERY_REWRITE_ENABLED,
)


def get_conversation_store() -> ConversationStore:
    """Return the process-wide conversation store dependency."""
    return conversation_store
