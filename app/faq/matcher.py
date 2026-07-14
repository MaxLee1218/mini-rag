from __future__ import annotations

from collections import defaultdict
from numbers import Real

from rank_bm25 import BM25Okapi

from app.faq.models import FAQMatch, FAQRecord
from app.faq.text import normalize_question, tokenize_question


class FAQMatcher:
    """Read-only exact and BM25 matcher over enabled FAQ surface forms."""

    def __init__(
        self,
        records: list[FAQRecord] | tuple[FAQRecord, ...],
        *,
        threshold: float,
        margin: float,
    ) -> None:
        self.threshold = _nonnegative_number(threshold, "threshold")
        self.margin = _nonnegative_number(margin, "margin")
        self.records = tuple(record for record in records if record.enabled)
        self._records_by_id = {record.id: record for record in self.records}
        self._canonical: dict[str, str] = {}
        self._aliases: dict[str, str] = {}
        self._surface_owners: dict[str, set[str]] = defaultdict(set)
        surface_faq_ids: list[str] = []
        tokenized_surfaces: list[list[str]] = []

        for record in self.records:
            canonical = normalize_question(record.question)
            self._add_exact(self._canonical, canonical, record.id)
            self._surface_owners[canonical].add(record.id)
            surface_faq_ids.append(record.id)
            tokenized_surfaces.append(tokenize_question(canonical))
            for alias in record.aliases:
                normalized_alias = normalize_question(alias)
                self._add_exact(self._aliases, normalized_alias, record.id)
                self._surface_owners[normalized_alias].add(record.id)
                surface_faq_ids.append(record.id)
                tokenized_surfaces.append(tokenize_question(normalized_alias))

        self._surface_faq_ids = tuple(surface_faq_ids)
        self._bm25 = BM25Okapi(tokenized_surfaces) if tokenized_surfaces else None

    def match(self, question: str) -> FAQMatch | None:
        normalized = normalize_question(question)
        if not normalized:
            return None
        if len(self._surface_owners.get(normalized, ())) > 1:
            return None

        canonical_id = self._canonical.get(normalized)
        if canonical_id:
            return self._match_for(canonical_id, 1.0, "exact")
        alias_id = self._aliases.get(normalized)
        if alias_id:
            return self._match_for(alias_id, 1.0, "alias")

        query_tokens = tokenize_question(normalized)
        if self._bm25 is None or not query_tokens:
            return None
        scores = self._bm25.get_scores(query_tokens)
        by_faq: dict[str, float] = {}
        for index, faq_id in enumerate(self._surface_faq_ids):
            score = float(scores[index])
            by_faq[faq_id] = max(by_faq.get(faq_id, float("-inf")), score)
        ranked = sorted(by_faq.items(), key=lambda item: (-item[1], item[0]))
        if not ranked:
            return None
        top_faq_id, top_score = ranked[0]
        if top_score <= 0 or top_score < self.threshold:
            return None
        if len(ranked) > 1 and top_score - ranked[1][1] < self.margin:
            return None
        return self._match_for(top_faq_id, top_score, "bm25")

    @staticmethod
    def _add_exact(target: dict[str, str], surface: str, faq_id: str) -> None:
        if surface and surface not in target:
            target[surface] = faq_id

    def _match_for(
        self, faq_id: str, score: float, match_type: str
    ) -> FAQMatch:
        record = self._records_by_id[faq_id]
        return FAQMatch(
            faq_id=record.id,
            question=record.question,
            answer=record.answer,
            source=record.source,
            score=float(score),
            match_type=match_type,
        )


def _nonnegative_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or value < 0:
        raise ValueError(f"{name} must be a non-negative number")
    return float(value)
