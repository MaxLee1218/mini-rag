"""Aggregate latency observations for offline evaluation reports."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from evaluation.models import LatencyObservation


def analyze_latencies(
    observations: Sequence[LatencyObservation],
) -> dict[str, dict[str, float | int | None]]:
    """Calculate p50, p95, and valid count for every stage."""
    result: dict[str, dict[str, float | int | None]] = {}
    for stage in ("embedding", "retrieval", "generation", "total"):
        values = [
            float(value)
            for row in observations
            if (value := getattr(row, stage)) is not None
        ]
        result[stage] = {
            "p50": round(float(np.percentile(values, 50)), 3) if values else None,
            "p95": round(float(np.percentile(values, 95)), 3) if values else None,
            "count": len(values),
        }
    return result
