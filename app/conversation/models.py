from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass(frozen=True)
class ConversationTurn:
    """One completed user and assistant exchange stored with a UTC timestamp."""

    user_message: str
    assistant_message: str
    created_at: datetime

    def __post_init__(self) -> None:
        for field_name in ("user_message", "assistant_message"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must not be blank")
        if (
            not isinstance(self.created_at, datetime)
            or self.created_at.tzinfo is None
            or self.created_at.utcoffset() != timedelta(0)
        ):
            raise ValueError("created_at must use UTC")
