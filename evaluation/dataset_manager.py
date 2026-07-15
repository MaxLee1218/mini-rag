"""Loading and validation for version-controlled evaluation datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evaluation.models import EvaluationSample


class DatasetValidationError(ValueError):
    """Raised when an evaluation dataset cannot be read or validated."""


def _required_string(row: dict[str, Any], name: str, row_number: int) -> str:
    value = row.get(name)
    if not isinstance(value, str):
        raise DatasetValidationError(f"row {row_number} {name} must be a string")
    if not value.strip():
        raise DatasetValidationError(f"row {row_number} {name} must not be blank")
    return value


def _reference_contexts(row: dict[str, Any], row_number: int) -> tuple[str, ...]:
    if "reference_contexts" not in row:
        return ()

    contexts = row["reference_contexts"]
    if not isinstance(contexts, list) or not contexts:
        raise DatasetValidationError(
            f"row {row_number} reference_contexts must be a non-empty list"
        )

    validated: list[str] = []
    for item_number, context in enumerate(contexts, start=1):
        if not isinstance(context, str):
            raise DatasetValidationError(
                f"row {row_number} reference_contexts item {item_number} must be a string"
            )
        if not context.strip():
            raise DatasetValidationError(
                f"row {row_number} reference_contexts item {item_number} must not be blank"
            )
        validated.append(context)
    return tuple(validated)


def _should_abstain(row: dict[str, Any], row_number: int) -> bool:
    value = row.get("should_abstain", False)
    if not isinstance(value, bool):
        raise DatasetValidationError(
            f"row {row_number} should_abstain must be a boolean"
        )
    return value


def load_evaluation_dataset(path: Path) -> list[EvaluationSample]:
    """Load and validate evaluation samples from a UTF-8 JSON file."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise DatasetValidationError(
            f"could not read evaluation dataset {path}: {error}"
        ) from error
    except (UnicodeError, json.JSONDecodeError) as error:
        raise DatasetValidationError(
            f"evaluation dataset {path} contains invalid JSON: {error}"
        ) from error

    if not isinstance(payload, list):
        raise DatasetValidationError("evaluation dataset top level must be a list")
    if not payload:
        raise DatasetValidationError("evaluation dataset must contain at least one sample")

    samples: list[EvaluationSample] = []
    normalized_questions: set[str] = set()
    for row_number, row in enumerate(payload, start=1):
        if not isinstance(row, dict):
            raise DatasetValidationError(f"row {row_number} must be an object")

        question = _required_string(row, "question", row_number)
        ground_truth = _required_string(row, "ground_truth", row_number)
        normalized_question = " ".join(question.casefold().split())
        if normalized_question in normalized_questions:
            raise DatasetValidationError(f"row {row_number} duplicate question")
        normalized_questions.add(normalized_question)

        samples.append(
            EvaluationSample(
                question=question,
                ground_truth=ground_truth,
                reference_contexts=_reference_contexts(row, row_number),
                should_abstain=_should_abstain(row, row_number),
            )
        )

    return samples
