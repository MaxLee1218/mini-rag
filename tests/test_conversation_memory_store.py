from datetime import datetime, timedelta, timezone

import pytest

from app.conversation.memory_store import InMemoryConversationStore
from app.conversation.models import ConversationTurn


def turn(number: int) -> ConversationTurn:
    return ConversationTurn(
        user_message=f"question-{number}",
        assistant_message=f"answer-{number}",
        created_at=datetime(2026, 1, number, tzinfo=timezone.utc),
    )


def test_empty_session_returns_empty_list():
    assert InMemoryConversationStore().get_recent_turns("session-a", 5) == []


def test_append_and_read_one_turn():
    store = InMemoryConversationStore()
    expected = turn(1)
    store.append_turn("session-a", expected)
    assert store.get_recent_turns("session-a", 5) == [expected]


def test_sessions_are_isolated():
    store = InMemoryConversationStore()
    store.append_turn("session-a", turn(1))
    store.append_turn("session-b", turn(2))
    assert store.get_recent_turns("session-a", 5) == [turn(1)]
    assert store.get_recent_turns("session-b", 5) == [turn(2)]


def test_old_turns_are_evicted_at_capacity():
    store = InMemoryConversationStore(max_turns=5)
    for number in range(1, 7):
        store.append_turn("session-a", turn(number))
    assert store.get_recent_turns("session-a", 5) == [turn(n) for n in range(2, 7)]


def test_read_limit_returns_only_most_recent_turns():
    store = InMemoryConversationStore()
    for number in range(1, 6):
        store.append_turn("session-a", turn(number))
    assert store.get_recent_turns("session-a", 3) == [turn(n) for n in range(3, 6)]


def test_read_returns_a_copy():
    store = InMemoryConversationStore()
    store.append_turn("session-a", turn(1))
    received = store.get_recent_turns("session-a", 5)
    received.clear()
    assert store.get_recent_turns("session-a", 5) == [turn(1)]


def test_clear_session_does_not_affect_other_sessions():
    store = InMemoryConversationStore()
    store.append_turn("session-a", turn(1))
    store.append_turn("session-b", turn(2))
    store.clear_session("session-a")
    assert store.get_recent_turns("session-a", 5) == []
    assert store.get_recent_turns("session-b", 5) == [turn(2)]


@pytest.mark.parametrize("session_id", ["", "   "])
def test_blank_session_id_is_rejected(session_id):
    store = InMemoryConversationStore()
    with pytest.raises(ValueError, match="session_id must not be blank"):
        store.get_recent_turns(session_id, 5)
    with pytest.raises(ValueError, match="session_id must not be blank"):
        store.append_turn(session_id, turn(1))
    with pytest.raises(ValueError, match="session_id must not be blank"):
        store.clear_session(session_id)


@pytest.mark.parametrize("limit", [0, -1])
def test_non_positive_read_limit_is_rejected(limit):
    with pytest.raises(ValueError, match="limit must be a positive integer"):
        InMemoryConversationStore().get_recent_turns("session-a", limit)


@pytest.mark.parametrize("max_turns", [2, 6, True])
def test_invalid_store_capacity_is_rejected(max_turns):
    with pytest.raises(ValueError, match="max_turns must be between 3 and 5"):
        InMemoryConversationStore(max_turns=max_turns)


def test_append_requires_conversation_turn():
    with pytest.raises(TypeError, match="turn must be a ConversationTurn"):
        InMemoryConversationStore().append_turn("session-a", {})


@pytest.mark.parametrize("field", ["user_message", "assistant_message"])
def test_turn_rejects_blank_messages(field):
    values = {
        "user_message": "question",
        "assistant_message": "answer",
        "created_at": datetime.now(timezone.utc),
    }
    values[field] = "  "
    with pytest.raises(ValueError, match=f"{field} must not be blank"):
        ConversationTurn(**values)


def test_turn_requires_utc_timestamp():
    with pytest.raises(ValueError, match="created_at must use UTC"):
        ConversationTurn("q", "a", datetime.now())
    with pytest.raises(ValueError, match="created_at must use UTC"):
        ConversationTurn("q", "a", datetime.now(timezone(timedelta(hours=8))))
