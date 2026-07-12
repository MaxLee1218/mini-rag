from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.conversation.models import ConversationTurn


@dataclass(frozen=True)
class QueryRewriteResult:
    """Structured outcome of one query rewrite decision."""

    original_query: str
    rewritten_query: str
    was_rewritten: bool
    reason: str


class QueryRewriter(ABC):
    """Replaceable interface for context-aware retrieval-query rewriting."""

    @abstractmethod
    def rewrite(
        self,
        query: str,
        history: list[ConversationTurn],
    ) -> QueryRewriteResult:
        """Return a retrieval query without answering the user's question."""
        raise NotImplementedError
