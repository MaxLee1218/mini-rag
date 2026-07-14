from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from app.prompt_builder import (
    append_sources_to_answer,
    build_prompt,
    build_sources_section,
)


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RAGResult:
    question: str
    answer: str
    contexts: list[Any]
    sources: list[str]
    prompt: str | None = None
    route: Literal["faq", "rag"] = "rag"
    faq_id: str | None = None
    faq_score: float | None = None
    faq_match_type: str | None = None
    faq_cache_hit: bool = False
    rewritten_query: str | None = None
    query_was_rewritten: bool = False
    rewrite_reason: str = "not_rewritten"
    history_turn_count: int = 0


class RAGPipeline:
    """Coordinate retrieval, prompt building, generation, and source attachment."""

    def __init__(
        self,
        *,
        retriever: Any,
        generator: Any,
        prompt_builder: Any = build_prompt,
        source_appender: Callable[..., str] = append_sources_to_answer,
        reranker: Any | None = None,
        candidate_k: int = 10,
        final_top_k: int = 5,
        expand_retrieval_candidates: bool = True,
    ) -> None:
        self.retriever = retriever
        self.generator = generator
        self.prompt_builder = prompt_builder
        self.source_appender = source_appender
        self.reranker = reranker
        self.candidate_k = _validate_named_top_k(candidate_k, "candidate_k")
        self.final_top_k = _validate_named_top_k(final_top_k, "final_top_k")
        if not isinstance(expand_retrieval_candidates, bool):
            raise ValueError("expand_retrieval_candidates must be a boolean")
        self.expand_retrieval_candidates = expand_retrieval_candidates

    def ask(
        self,
        question: str,
        top_k: int | None = None,
        *,
        retrieval_query: str | None = None,
    ) -> RAGResult:
        clean_question = validate_question(question)
        query_for_retrieval = _resolve_retrieval_query(
            retrieval_query,
            clean_question,
        )
        resolved_top_k = (
            self.final_top_k if top_k is None else _validate_top_k(top_k)
        )
        retrieval_top_k = (
            max(self.candidate_k, resolved_top_k)
            if self.expand_retrieval_candidates
            else resolved_top_k
        )

        candidates = _normalize_contexts(
            self.retriever.retrieve(query_for_retrieval, top_k=retrieval_top_k)
        )[:retrieval_top_k]
        contexts = self._select_contexts(
            query_for_retrieval, candidates, top_k=resolved_top_k
        )
        generation_question = _build_generation_question(
            clean_question,
            query_for_retrieval,
        )
        prompt = self._build_prompt(generation_question, contexts)
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

    def _select_contexts(
        self, question: str, candidates: list[Any], *, top_k: int
    ) -> list[Any]:
        if self.reranker is None:
            return candidates[:top_k]
        try:
            return _normalize_contexts(
                self.reranker.rerank(question, candidates, top_k=top_k)
            )[:top_k]
        except Exception:
            logger.exception(
                "reranker failed; falling back to original retrieval order"
            )
            return candidates[:top_k]

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


def validate_question(question: Any) -> str:
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must not be blank")
    return question.strip()


def _resolve_retrieval_query(retrieval_query: Any, question: str) -> str:
    if retrieval_query is None:
        return question
    if not isinstance(retrieval_query, str):
        raise ValueError("retrieval_query must be a string")
    return retrieval_query.strip() or question


def _build_generation_question(
    original_question: str,
    resolved_question: str,
) -> str:
    if resolved_question == original_question:
        return original_question
    return (
        f"Original question:\n{original_question}\n\n"
        "Resolved question for disambiguation:\n"
        f"{resolved_question}\n\n"
        "Answer the original question. Use the resolved question only to "
        "identify references from the conversation."
    )


def _validate_top_k(top_k: Any) -> int:
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    return top_k


def _validate_named_top_k(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


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
