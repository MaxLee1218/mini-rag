from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Any

from app.faq.text import normalize_question
from app.pipeline import RAGResult, validate_question


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreparedRAGQuery:
    retrieval_query: str
    query_was_rewritten: bool
    rewrite_reason: str
    history_turn_count: int


class DualPathPipeline:
    """Route maintained FAQ answers before lazily entering the RAG path."""

    def __init__(
        self,
        *,
        faq_matcher: Any,
        faq_cache: Any,
        rag_pipeline_provider: Any,
        faq_index_version: int,
        rag_query_preparer: Any | None = None,
    ) -> None:
        if not callable(rag_pipeline_provider):
            raise ValueError("rag_pipeline_provider must be callable")
        self.faq_matcher = faq_matcher
        self.faq_cache = faq_cache
        self.rag_pipeline_provider = rag_pipeline_provider
        self.faq_index_version = faq_index_version
        self.rag_query_preparer = rag_query_preparer

    def ask(
        self,
        question: str,
        top_k: int | None = None,
        *,
        retrieval_query: str | None = None,
        session_id: str | None = None,
    ) -> RAGResult:
        clean_question = validate_question(question)
        normalized = normalize_question(clean_question)
        match = self._cache_get(normalized)
        cache_hit = match is not None
        if match is None:
            match = self._match(clean_question)
            if match is not None and _has_answer(match):
                self._cache_set(normalized, match)
        if match is not None and _has_answer(match):
            return RAGResult(
                question=clean_question,
                answer=match.answer,
                contexts=[],
                sources=[match.source] if match.source else [],
                prompt="",
                route="faq",
                faq_id=match.faq_id,
                faq_score=match.score,
                faq_match_type="cache" if cache_hit else match.match_type,
                faq_cache_hit=cache_hit,
                rewritten_query=clean_question,
                query_was_rewritten=False,
                rewrite_reason="faq_fast_path",
                history_turn_count=0,
            )

        prepared = self._prepare_rag_query(
            clean_question,
            retrieval_query=retrieval_query,
            session_id=session_id,
        )
        rag_pipeline = self.rag_pipeline_provider()
        result = rag_pipeline.ask(
            clean_question,
            top_k=top_k,
            retrieval_query=prepared.retrieval_query,
        )
        return replace(
            result,
            route="rag",
            faq_id=None,
            faq_score=None,
            faq_match_type=None,
            faq_cache_hit=False,
            rewritten_query=prepared.retrieval_query,
            query_was_rewritten=prepared.query_was_rewritten,
            rewrite_reason=prepared.rewrite_reason,
            history_turn_count=prepared.history_turn_count,
        )

    def _cache_get(self, normalized: str) -> Any | None:
        try:
            return self.faq_cache.get(normalized, self.faq_index_version)
        except Exception as error:
            logger.warning(
                "faq_cache_lookup_failed",
                extra={"error_type": type(error).__name__},
            )
            return None

    def _cache_set(self, normalized: str, match: Any) -> None:
        try:
            self.faq_cache.set(normalized, self.faq_index_version, match)
        except Exception as error:
            logger.warning(
                "faq_cache_write_failed",
                extra={"error_type": type(error).__name__},
            )

    def _match(self, question: str) -> Any | None:
        try:
            return self.faq_matcher.match(question)
        except Exception as error:
            logger.warning(
                "faq_matcher_failed",
                extra={"error_type": type(error).__name__},
            )
            return None

    def _prepare_rag_query(
        self,
        question: str,
        *,
        retrieval_query: str | None,
        session_id: str | None,
    ) -> PreparedRAGQuery:
        if retrieval_query is not None:
            resolved = (
                retrieval_query.strip()
                if isinstance(retrieval_query, str)
                else retrieval_query
            )
            return PreparedRAGQuery(
                retrieval_query=resolved or question,
                query_was_rewritten=bool(resolved and resolved != question),
                rewrite_reason="explicit_retrieval_query",
                history_turn_count=0,
            )
        if self.rag_query_preparer is None:
            return PreparedRAGQuery(question, False, "no_query_preparer", 0)
        prepare = getattr(self.rag_query_preparer, "prepare", None)
        if callable(prepare):
            return prepare(question, session_id)
        if callable(self.rag_query_preparer):
            return self.rag_query_preparer(question, session_id)
        raise ValueError("rag_query_preparer must be callable or provide prepare()")


def _has_answer(match: Any) -> bool:
    return isinstance(getattr(match, "answer", None), str) and bool(
        match.answer.strip()
    )
