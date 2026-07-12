from __future__ import annotations

import json
import logging
import time
from collections.abc import Collection
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from app.conversation.store import ConversationStore
from app.query_rewriter.base import QueryRewriter, QueryRewriteResult


logger = logging.getLogger(__name__)
LOG_TEXT_LIMIT = 500


class QueryOptimizationMiddleware(BaseHTTPMiddleware):
    """Load session history and expose a safe retrieval query in request state."""

    def __init__(
        self,
        app: Any,
        *,
        conversation_store: ConversationStore,
        query_rewriter: QueryRewriter,
        history_limit: int,
        enabled: bool = True,
        paths: Collection[str] = frozenset({"/ask"}),
    ) -> None:
        super().__init__(app)
        if history_limit <= 0:
            raise ValueError("history_limit must be positive")
        self.conversation_store = conversation_store
        self.query_rewriter = query_rewriter
        self.history_limit = history_limit
        self.enabled = enabled
        self.paths = frozenset(paths)

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        if request.method != "POST" or request.url.path not in self.paths:
            return await call_next(request)

        payload = await _read_json_body(request)
        if not _has_usable_fields(payload):
            return await call_next(request)

        original_question = payload["question"].strip()
        session_id = payload["session_id"].strip()
        started_at = time.perf_counter()
        history = []
        try:
            history = self.conversation_store.get_recent_turns(
                session_id,
                self.history_limit,
            )
        except Exception:
            logger.exception(
                "conversation history lookup failed",
                extra={"session_id": session_id},
            )

        if self.enabled:
            try:
                result = self.query_rewriter.rewrite(original_question, history)
            except Exception:
                logger.exception(
                    "query rewrite failed; using original question",
                    extra={"session_id": session_id},
                )
                result = QueryRewriteResult(
                    original_question,
                    original_question,
                    False,
                    "rewriter_error",
                )
        else:
            result = QueryRewriteResult(
                original_question,
                original_question,
                False,
                "query_rewrite_disabled",
            )

        request.state.original_question = original_question
        request.state.rewritten_query = result.rewritten_query
        request.state.query_was_rewritten = result.was_rewritten
        request.state.query_rewrite_reason = result.reason
        request.state.conversation_history = list(history)
        request.state.session_id = session_id
        logger.info(
            "query_optimization_completed",
            extra={
                "session_id": session_id,
                "original_question": original_question[:LOG_TEXT_LIMIT],
                "rewritten_query": result.rewritten_query[:LOG_TEXT_LIMIT],
                "query_was_rewritten": result.was_rewritten,
                "rewrite_reason": result.reason,
                "history_turn_count": len(history),
                "rewrite_latency_ms": round(
                    (time.perf_counter() - started_at) * 1000, 3
                ),
            },
        )
        return await call_next(request)


async def _read_json_body(request: Request) -> Any:
    try:
        body = await request.body()
        return json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None


def _has_usable_fields(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    return all(
        isinstance(payload.get(field), str) and payload[field].strip()
        for field in ("question", "session_id")
    )
