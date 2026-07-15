"""Offline RAG evaluation contracts and utilities."""

from evaluation.dataset_manager import DatasetValidationError, load_evaluation_dataset
from evaluation.models import (
    EvaluationRecord,
    EvaluationSample,
    LatencyObservation,
    RagasScores,
)

__all__ = [
    "DatasetValidationError",
    "EvaluationRecord",
    "EvaluationSample",
    "LatencyObservation",
    "RagasScores",
    "load_evaluation_dataset",
]
