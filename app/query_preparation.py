from __future__ import annotations

import logging
from typing import Any

from app.dual_path_pipeline import PreparedRAGQuery


logger = logging.getLogger(__name__)


class ConversationQueryPreparer:
    """Prepare a conversation-aware retrieval query only for RAG misses."""

    def __init__(
        self,
        *,
        store: Any,
        rewriter: Any,
        history_limit: int,
        enabled: bool = True,
    ) -> None:
        if not isinstance(history_limit, int) or history_limit <= 0:
            raise ValueError("history_limit must be positive")
        self.store = store
        self.rewriter = rewriter
        self.history_limit = history_limit
        self.enabled = enabled

    def prepare(
        self, question: str, session_id: str | None
    ) -> PreparedRAGQuery:
        history = []
        if session_id:
            try:
                history = list(
                    self.store.get_recent_turns(session_id, self.history_limit)
                )
            except Exception as error:
                logger.warning(
                    "conversation_history_lookup_failed",
                    extra={"error_type": type(error).__name__},
                )
        if not self.enabled:
            return PreparedRAGQuery(
                question, False, "query_rewrite_disabled", len(history)
            )
        try:
            result = self.rewriter.rewrite(question, history)
        except Exception as error:
            logger.warning(
                "query_rewrite_failed",
                extra={"error_type": type(error).__name__},
            )
            return PreparedRAGQuery(
                question, False, "query_rewriter_error", len(history)
            )
        return PreparedRAGQuery(
            retrieval_query=result.rewritten_query,
            query_was_rewritten=result.was_rewritten,
            rewrite_reason=result.reason,
            history_turn_count=len(history),
        )
