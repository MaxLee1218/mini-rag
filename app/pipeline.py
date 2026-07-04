from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from app.prompt_builder import (
    append_sources_to_answer,
    build_prompt,
    build_sources_section,
)


@dataclass(frozen=True)
class RAGResult:
    question: str
    answer: str
    contexts: list[Any]
    sources: list[str]
    prompt: str | None = None


class RAGPipeline:
    """Coordinate retrieval, prompt building, generation, and source attachment."""

    def __init__(
        self,
        *,
        retriever: Any,
        generator: Any,
        prompt_builder: Any = build_prompt,
        source_appender: Callable[..., str] = append_sources_to_answer,
    ) -> None:
        self.retriever = retriever
        self.generator = generator
        self.prompt_builder = prompt_builder
        self.source_appender = source_appender

    def ask(self, question: str, top_k: int = 4) -> RAGResult:
        clean_question = _validate_question(question)
        resolved_top_k = _validate_top_k(top_k)

        contexts = _normalize_contexts(
            self.retriever.retrieve(clean_question, top_k=resolved_top_k)
        )
        prompt = self._build_prompt(clean_question, contexts)
        raw_answer = self._generate(prompt)

        final_answer = raw_answer
        if contexts:
            final_answer = self.source_appender(
                raw_answer,
                contexts,
                max_sources=resolved_top_k,
            )

        return RAGResult(
            question=clean_question,
            answer=final_answer,
            contexts=contexts,
            sources=_extract_sources_from_contexts(
                contexts,
                max_sources=resolved_top_k,
            ),
            prompt=prompt,
        )

    def _build_prompt(self, question: str, contexts: list[Any]) -> str:
        build = getattr(self.prompt_builder, "build", None)
        if callable(build):
            return build(question, contexts)
        if callable(self.prompt_builder):
            return self.prompt_builder(question, contexts)
        raise ValueError("prompt_builder must be callable or provide build()")

    def _generate(self, prompt: str) -> str:
        generate = getattr(self.generator, "generate", None)
        if callable(generate):
            return generate(prompt)
        if callable(self.generator):
            return self.generator(prompt)
        raise ValueError("generator must be callable or provide generate()")


def _validate_question(question: Any) -> str:
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must not be blank")
    return question.strip()


def _validate_top_k(top_k: Any) -> int:
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    return top_k


def _normalize_contexts(contexts: Any) -> list[Any]:
    if contexts is None:
        return []
    return list(contexts)


def _extract_sources_from_contexts(
    contexts: list[Any],
    *,
    max_sources: int,
) -> list[str]:
    sources_section = build_sources_section(contexts, max_sources=max_sources)
    if not sources_section:
        return []

    sources = []
    seen = set()
    for line in sources_section.splitlines():
        source = _source_from_source_line(line)
        if source is None or source in seen:
            continue
        sources.append(source)
        seen.add(source)
    return sources


def _source_from_source_line(line: str) -> str | None:
    line = line.strip()
    if not line.startswith("- ["):
        return None

    marker_end = line.find("]")
    if marker_end < 0:
        return None

    source = line[marker_end + 1 :].strip()
    if not source:
        return None
    return source
