from __future__ import annotations

from abc import ABC, abstractmethod

from app.conversation.models import ConversationTurn


class ConversationStore(ABC):
    """Replaceable persistence boundary for session-scoped conversation turns."""

    @abstractmethod
    def get_recent_turns(
        self,
        session_id: str,
        limit: int,
    ) -> list[ConversationTurn]:
        """Return up to ``limit`` newest completed turns for one session."""
        raise NotImplementedError

    @abstractmethod
    def append_turn(self, session_id: str, turn: ConversationTurn) -> None:
        """Append one completed turn to a session."""
        raise NotImplementedError

    @abstractmethod
    def clear_session(self, session_id: str) -> None:
        """Delete all turns belonging to one session."""
        raise NotImplementedError
