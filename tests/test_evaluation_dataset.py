import json

import pytest

from evaluation.dataset_manager import DatasetValidationError, load_evaluation_dataset
from evaluation.models import EvaluationSample


def write_dataset(tmp_path, payload):
    path = tmp_path / "eval.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_load_dataset_returns_typed_sample(tmp_path):
    path = write_dataset(
        tmp_path,
        [
            {
                "question": "What does RAG retrieve?",
                "ground_truth": "RAG retrieves relevant context.",
                "reference_contexts": [
                    "RAG retrieves relevant context before generating an answer."
                ],
                "should_abstain": False,
            }
        ],
    )

    sample = load_evaluation_dataset(path)[0]

    assert isinstance(sample, EvaluationSample)
    assert sample.question == "What does RAG retrieve?"
    assert sample.reference_contexts == (
        "RAG retrieves relevant context before generating an answer.",
    )
    assert sample.should_abstain is False


def test_load_dataset_applies_optional_field_defaults(tmp_path):
    path = write_dataset(tmp_path, [{"question": "Question?", "ground_truth": "Answer."}])

    assert load_evaluation_dataset(path) == [
        EvaluationSample(question="Question?", ground_truth="Answer.")
    ]


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ([], "at least one sample"),
        ({}, "top level must be a list"),
        (["row"], "row 1 must be an object"),
        ([{"ground_truth": "a"}], "row 1 question must be a string"),
        ([{"question": " ", "ground_truth": "a"}], "row 1 question must not be blank"),
        ([{"question": "q"}], "row 1 ground_truth must be a string"),
        ([{"question": "q", "ground_truth": " "}], "row 1 ground_truth must not be blank"),
        (
            [{"question": "q", "ground_truth": "a", "reference_contexts": []}],
            "row 1 reference_contexts must be a non-empty list",
        ),
        (
            [{"question": "q", "ground_truth": "a", "reference_contexts": [" "]}],
            "row 1 reference_contexts item 1 must not be blank",
        ),
        (
            [{"question": "q", "ground_truth": "a", "should_abstain": "yes"}],
            "row 1 should_abstain must be a boolean",
        ),
    ],
)
def test_load_dataset_rejects_invalid_rows(tmp_path, payload, message):
    path = write_dataset(tmp_path, payload)

    with pytest.raises(DatasetValidationError, match=message):
        load_evaluation_dataset(path)


def test_load_dataset_rejects_normalized_duplicate_questions(tmp_path):
    path = write_dataset(
        tmp_path,
        [
            {"question": "What is RAG?", "ground_truth": "a"},
            {"question": " what  IS rag? ", "ground_truth": "b"},
        ],
    )

    with pytest.raises(DatasetValidationError, match="row 2 duplicate question"):
        load_evaluation_dataset(path)


def test_load_dataset_translates_missing_file_error(tmp_path):
    path = tmp_path / "missing.json"

    with pytest.raises(DatasetValidationError, match="could not read evaluation dataset"):
        load_evaluation_dataset(path)


def test_load_dataset_translates_invalid_json_error(tmp_path):
    path = tmp_path / "eval.json"
    path.write_text("not json", encoding="utf-8")

    with pytest.raises(DatasetValidationError, match="invalid JSON"):
        load_evaluation_dataset(path)


def test_checked_in_dataset_contains_ten_valid_samples():
    from app.config import EVALUATION_DATASET_PATH

    samples = load_evaluation_dataset(EVALUATION_DATASET_PATH)

    assert len(samples) == 10
    assert sum(sample.should_abstain for sample in samples) == 2
    assert all(sample.reference_contexts for sample in samples if not sample.should_abstain)
