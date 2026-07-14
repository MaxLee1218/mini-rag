from __future__ import annotations

import re
import unicodedata

import jieba


_TRAILING_PUNCTUATION = re.compile(r"[?？！!.。]+$")
_TOKEN_SPANS = re.compile(r"[a-z0-9]+|[㐀-鿿]+")


def normalize_question(text: str) -> str:
    """Normalize FAQ surfaces without removing semantic characters."""
    if not isinstance(text, str):
        return ""
    normalized = unicodedata.normalize("NFKC", text).strip().lower()
    normalized = " ".join(normalized.split())
    normalized = _TRAILING_PUNCTUATION.sub("", normalized).rstrip()
    return normalized


def tokenize_question(text: str) -> list[str]:
    """Tokenize normalized Chinese, English, and numeric FAQ text."""
    normalized = normalize_question(text)
    if not normalized:
        return []
    tokens: list[str] = []
    for span in _TOKEN_SPANS.findall(normalized):
        if span.isascii():
            tokens.append(span)
        else:
            tokens.extend(token.strip() for token in jieba.cut(span) if token.strip())
    return tokens
