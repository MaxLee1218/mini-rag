from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.conversation.memory_store import InMemoryConversationStore
from app.conversation.models import ConversationTurn
from app.query_rewriter.llm_rewriter import LLMQueryRewriter


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exercise conversation memory and query rewriting offline."
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use deterministic local answers without providers or vector storage.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.mock:
        print("Use --mock; real-provider conversation smoke is not enabled.", file=sys.stderr)
        return 2

    session_id = "demo-session"
    store = InMemoryConversationStore(max_turns=5)
    rewriter = _build_mock_rewriter()
    questions = ("什么是 Middleware？", "它为什么可以优化查询？")

    print(f"Session: {session_id}")
    for number, question in enumerate(questions, start=1):
        history = store.get_recent_turns(session_id, 5)
        rewrite = rewriter.rewrite(question, history)
        answer = _mock_answer(number)
        store.append_turn(
            session_id,
            ConversationTurn(question, answer, datetime.now(timezone.utc)),
        )
        print()
        print(f"Round {number}")
        print(f"Original question: {question}")
        print(f"Rewritten query: {rewrite.rewritten_query}")
        print(f"Was rewritten: {rewrite.was_rewritten}")
        print(f"Answer: {answer}")

    print()
    print(f"Stored turns: {len(store.get_recent_turns(session_id, 5))}")
    return 0


def _mock_answer(round_number: int) -> str:
    if round_number == 1:
        return "Middleware 是请求进入业务流程前的中间处理层。"
    return "它可以在检索前把依赖上下文的问题转换为独立查询。"


def _build_mock_rewriter() -> LLMQueryRewriter:
    """Build an offline fake-LLM rewriter for the deterministic smoke test."""

    def complete(prompt: str) -> str:
        current_question = prompt.split("当前问题：\n", 1)[1].split("\n\n", 1)[0]
        if (
            current_question == "它为什么可以优化查询？"
            and "用户：什么是 Middleware？" in prompt
        ):
            return "Middleware 为什么可以优化查询？"
        return current_question

    return LLMQueryRewriter(completion_call=complete)


if __name__ == "__main__":
    raise SystemExit(main())
