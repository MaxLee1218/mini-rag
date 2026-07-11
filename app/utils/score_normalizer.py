from __future__ import annotations


def min_max_normalize(scores: list[float]) -> list[float]:
    """Normalize one retrieval route's scores to the inclusive 0..1 range."""
    if not scores:
        return []

    minimum = min(scores)
    maximum = max(scores)
    if maximum == minimum:
        return [1.0] * len(scores)

    score_range = maximum - minimum
    return [(score - minimum) / score_range for score in scores]
