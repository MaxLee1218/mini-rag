from __future__ import annotations

from threading import RLock

from app.conversation.models import ConversationTurn
from app.conversation.store import ConversationStore


class InMemoryConversationStore(ConversationStore):
    """Thread-safe, process-local conversation storage with bounded sessions."""

    def __init__(self, max_turns: int = 5) -> None:
        if (
            isinstance(max_turns, bool)
            or not isinstance(max_turns, int)
            or not 3 <= max_turns <= 5
        ):
            raise ValueError("max_turns must be between 3 and 5")
        self._max_turns = max_turns
        self._turns: dict[str, list[ConversationTurn]] = {}
        self._lock = RLock()

    def get_recent_turns(
        self,
        session_id: str,
        limit: int,
    ) -> list[ConversationTurn]:
        clean_session_id = _validate_session_id(session_id)
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be a positive integer")
        with self._lock:
            return list(self._turns.get(clean_session_id, ())[-limit:])

    def append_turn(self, session_id: str, turn: ConversationTurn) -> None:
        clean_session_id = _validate_session_id(session_id)
        if not isinstance(turn, ConversationTurn):
            raise TypeError("turn must be a ConversationTurn")
        with self._lock:
            turns = self._turns.setdefault(clean_session_id, [])
            turns.append(turn)
            if len(turns) > self._max_turns:
                del turns[: len(turns) - self._max_turns]

    def clear_session(self, session_id: str) -> None:
        clean_session_id = _validate_session_id(session_id)
        with self._lock:
            self._turns.pop(clean_session_id, None)


def _validate_session_id(session_id: str) -> str:
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("session_id must not be blank")
    return session_id.strip()
