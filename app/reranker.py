from __future__ import annotations

import copy
import logging
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass, replace
from numbers import Real
from typing import Any, Callable

from app.prompt_builder import extract_context_text


logger = logging.getLogger(__name__)
ModelLoader = Callable[..., Any]
VALID_DEVICES = ("auto", "cpu", "cuda", "mps")


class CrossEncoderReranker:
    """Rerank retrieved candidates with a lazily loaded Cross-Encoder."""

    def __init__(
        self,
        model_name_or_path: str,
        *,
        batch_size: int = 16,
        max_length: int = 256,
        device: str = "cpu",
        failure_mode: str = "fallback",
        local_files_only: bool = False,
        model_loader: ModelLoader | None = None,
        torch_module: Any | None = None,
    ) -> None:
        self.model_name_or_path = _nonblank(
            model_name_or_path, "model_name_or_path"
        )
        self.batch_size = _positive_int(batch_size, "batch_size")
        self.max_length = _positive_int(max_length, "max_length")
        if not isinstance(device, str) or device.strip().lower() not in VALID_DEVICES:
            raise ValueError(f"device must be one of: {', '.join(VALID_DEVICES)}")
        self.device = device.strip().lower()
        if failure_mode != "fallback":
            raise ValueError("failure_mode must be fallback")
        if not isinstance(local_files_only, bool):
            raise ValueError("local_files_only must be a boolean")
        self.failure_mode = failure_mode
        self.local_files_only = local_files_only
        self._model_loader = model_loader or _default_model_loader
        self._torch_module = torch_module
        self._model: Any | None = None
        self._resolved_device: str | None = None
        logger.info(
            "reranker configured model=%s device=%s local_files_only=%s "
            "batch_size=%d max_length=%d",
            self.model_name_or_path,
            self.device,
            self.local_files_only,
            self.batch_size,
            self.max_length,
        )

    def rerank(
        self,
        query: str,
        documents: Sequence[Any],
        top_k: int | None = None,
    ) -> list[Any]:
        """Return safe candidate copies ordered by raw Cross-Encoder score."""
        clean_query = _nonblank(query, "query")
        limit = _optional_positive_int(top_k)
        if isinstance(documents, (str, bytes, Mapping)) or not isinstance(
            documents, Sequence
        ):
            raise ValueError("documents must be a sequence")

        originals = list(documents)
        fallback = [_copy_result(item) for item in originals]
        if not originals:
            return []

        ranked_copies = [_copy_result(item) for item in originals]
        valid_indices: list[int] = []
        pairs: list[tuple[str, str]] = []
        for index, document in enumerate(originals):
            text = extract_context_text(document)
            if text:
                valid_indices.append(index)
                pairs.append((clean_query, text))
            else:
                logger.warning("reranker candidate has no usable text index=%d", index)
                ranked_copies[index] = _with_score(
                    ranked_copies[index], float("-inf")
                )

        if not pairs:
            return _limit(ranked_copies, limit)

        started_at = time.perf_counter()
        try:
            model = self._load_model()
            raw_scores = model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            scores = _score_list(raw_scores)
            if len(scores) != len(valid_indices):
                raise RuntimeError(
                    "reranker score count does not match valid document count"
                )
            for index, score in zip(valid_indices, scores):
                if math.isnan(score):
                    logger.warning(
                        "reranker returned NaN index=%d; treating as lowest", index
                    )
                    score = float("-inf")
                ranked_copies[index] = _with_score(ranked_copies[index], score)
        except Exception:
            logger.exception(
                "reranker failed; falling back to original retrieval order"
            )
            return _limit(fallback, limit)

        ranked = sorted(
            ranked_copies,
            key=lambda item: _get_score(item),
            reverse=True,
        )
        result = _limit(ranked, limit)
        logger.info(
            "reranker completed candidates=%d valid_text=%d returned=%d duration_ms=%.2f",
            len(originals),
            len(pairs),
            len(result),
            (time.perf_counter() - started_at) * 1000,
        )
        return result

    def _load_model(self) -> Any:
        if self._model is None:
            resolved_device = self._resolve_device()
            logger.info(
                "loading reranker model=%s device=%s local_files_only=%s",
                self.model_name_or_path,
                resolved_device,
                self.local_files_only,
            )
            self._model = self._model_loader(
                self.model_name_or_path,
                device=resolved_device,
                max_length=self.max_length,
                local_files_only=self.local_files_only,
            )
            logger.info("reranker model loaded model=%s", self.model_name_or_path)
        return self._model

    def _resolve_device(self) -> str:
        if self._resolved_device is not None:
            return self._resolved_device
        if self.device == "cpu":
            self._resolved_device = "cpu"
            return "cpu"
        torch_module = self._torch_module
        if torch_module is None:
            import torch as torch_module

        cuda_available = bool(torch_module.cuda.is_available())
        mps_backend = getattr(getattr(torch_module, "backends", None), "mps", None)
        mps_check = getattr(mps_backend, "is_available", None)
        mps_available = bool(mps_check()) if callable(mps_check) else False
        if self.device == "auto":
            resolved = "cuda" if cuda_available else "mps" if mps_available else "cpu"
        elif self.device == "cuda" and cuda_available:
            resolved = "cuda"
        elif self.device == "mps" and mps_available:
            resolved = "mps"
        else:
            logger.warning(
                "requested reranker device %s is unavailable; falling back to cpu",
                self.device,
            )
            resolved = "cpu"
        self._resolved_device = resolved
        return resolved


def _default_model_loader(model_name_or_path: str, **kwargs: Any) -> Any:
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name_or_path, **kwargs)


def _copy_result(item: Any) -> Any:
    if isinstance(item, Mapping):
        copied = dict(item)
        metadata = copied.get("metadata")
        if isinstance(metadata, Mapping):
            copied["metadata"] = dict(metadata)
        return copied
    if is_dataclass(item) and not isinstance(item, type):
        names = {field.name for field in fields(item)}
        if "rerank_score" not in names:
            raise TypeError("dataclass result must define rerank_score")
        return replace(item)
    try:
        copied = copy.copy(item)
        setattr(copied, "rerank_score", getattr(copied, "rerank_score", None))
        return copied
    except (AttributeError, TypeError) as error:
        raise TypeError("document result cannot be safely copied") from error


def _with_score(item: Any, score: float) -> Any:
    if isinstance(item, dict):
        item["rerank_score"] = float(score)
        return item
    if is_dataclass(item):
        return replace(item, rerank_score=float(score))
    try:
        setattr(item, "rerank_score", float(score))
        return item
    except (AttributeError, TypeError) as error:
        raise TypeError("document result does not support rerank_score") from error


def _get_score(item: Any) -> float:
    value = (
        item.get("rerank_score")
        if isinstance(item, Mapping)
        else getattr(item, "rerank_score")
    )
    return float(value)


def _score_list(value: Any) -> list[float]:
    if isinstance(value, Real) and not isinstance(value, bool):
        return [float(value)]
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "reshape"):
        value = value.reshape(-1)
    try:
        return [float(score) for score in value]
    except (TypeError, ValueError) as error:
        raise RuntimeError("reranker returned invalid scores") from error


def _nonblank(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must not be blank")
    return value.strip()


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _optional_positive_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("top_k must be a positive integer or None")
    return value


def _limit(items: list[Any], limit: int | None) -> list[Any]:
    return items if limit is None else items[:limit]
