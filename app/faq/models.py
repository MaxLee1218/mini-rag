from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class FAQRecord:
    id: str
    question: str
    answer: str
    aliases: tuple[str, ...] = ()
    source: str | None = None
    enabled: bool = True


@dataclass(frozen=True)
class FAQMatch:
    faq_id: str
    question: str
    answer: str
    source: str | None
    score: float
    match_type: Literal["cache", "exact", "alias", "bm25"]


@dataclass(frozen=True)
class FAQImportSummary:
    inserted: int
    updated: int
    unchanged: int
    index_version: int
