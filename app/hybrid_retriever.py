from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Real
from typing import Any

from app.utils.score_normalizer import min_max_normalize


class HybridRetriever:
    """Fuse independently normalized sparse and dense retrieval results."""

    def __init__(
        self,
        sparse_retriever: Any,
        dense_retriever: Any,
        sparse_weight: float = 0.5,
        dense_weight: float = 0.5,
        candidate_multiplier: int = 2,
    ) -> None:
        self.sparse_retriever = sparse_retriever
        self.dense_retriever = dense_retriever
        self.sparse_weight, self.dense_weight = self._normalize_weights(
            sparse_weight, dense_weight
        )
        self.candidate_multiplier = self._validate_candidate_multiplier(
            candidate_multiplier
        )

    def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        clean_query = self._validate_query(query)
        resolved_top_k = self._validate_top_k(top_k)
        candidate_count = resolved_top_k * self.candidate_multiplier

        sparse_results = list(
            self.sparse_retriever.retrieve(clean_query, top_k=candidate_count)
        )
        dense_results = list(
            self.dense_retriever.retrieve(clean_query, top_k=candidate_count)
        )
        sparse_scores = [self._result_score(result) for result in sparse_results]
        dense_scores = [self._result_score(result) for result in dense_results]

        merged: dict[str, dict[str, Any]] = {}
        self._merge_route(
            merged,
            sparse_results,
            sparse_scores,
            min_max_normalize(sparse_scores),
            route="sparse",
        )
        self._merge_route(
            merged,
            dense_results,
            dense_scores,
            min_max_normalize(dense_scores),
            route="dense",
        )

        for result in merged.values():
            result.setdefault("normalized_sparse_score", 0.0)
            result.setdefault("normalized_dense_score", 0.0)
            result["score"] = (
                self.sparse_weight * result["normalized_sparse_score"]
                + self.dense_weight * result["normalized_dense_score"]
            )

        ranked = sorted(
            merged.values(),
            key=lambda item: (-item["score"], item["id"]),
        )
        return ranked[:resolved_top_k]

    def _merge_route(
        self,
        merged: dict[str, dict[str, Any]],
        results: list[Mapping[str, Any]],
        raw_scores: list[float],
        normalized_scores: list[float],
        *,
        route: str,
    ) -> None:
        for source_result, raw_score, normalized_score in zip(
            results, raw_scores, normalized_scores
        ):
            result_id = self._result_id(source_result)
            if result_id not in merged:
                merged[result_id] = dict(source_result)
            else:
                for key, value in source_result.items():
                    merged[result_id].setdefault(key, value)
            merged[result_id][f"{route}_score"] = raw_score
            merged[result_id][f"normalized_{route}_score"] = normalized_score

    def _result_id(self, result: Mapping[str, Any]) -> str:
        if not isinstance(result, Mapping):
            raise ValueError("result must be a dictionary")
        result_id = result.get("id")
        if result_id is None or not str(result_id).strip():
            raise ValueError("result id must not be blank")
        return str(result_id).strip()

    def _result_score(self, result: Mapping[str, Any]) -> float:
        self._result_id(result)
        score = result.get("score")
        if isinstance(score, bool) or not isinstance(score, Real):
            raise ValueError("result score must be a finite number")
        number = float(score)
        if not math.isfinite(number):
            raise ValueError("result score must be a finite number")
        return number

    def _normalize_weights(
        self,
        sparse_weight: Any,
        dense_weight: Any,
    ) -> tuple[float, float]:
        weights = (sparse_weight, dense_weight)
        if any(
            isinstance(weight, bool)
            or not isinstance(weight, Real)
            or not math.isfinite(float(weight))
            or float(weight) < 0
            for weight in weights
        ):
            raise ValueError("weights must be finite non-negative numbers")
        total = float(sparse_weight) + float(dense_weight)
        if total <= 0:
            raise ValueError(
                "weights must be finite non-negative numbers with a positive sum"
            )
        return float(sparse_weight) / total, float(dense_weight) / total

    def _validate_candidate_multiplier(self, value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("candidate_multiplier must be a positive integer")
        return value

    def _validate_query(self, query: Any) -> str:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must not be blank")
        return query.strip()

    def _validate_top_k(self, top_k: Any) -> int:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        return top_k
