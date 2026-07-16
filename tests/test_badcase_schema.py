from __future__ import annotations

import pytest

from eval.badcase_schema import BadCase


def _payload() -> dict[str, object]:
    return {
        "question": "他多少岁？",
        "answer": "Not found in knowledge base.",
        "expected_answer": "laojingqiao 34 years old",
        "contexts": ["laojingqiao is 34 years old"],
        "sources": ["personal.txt"],
        "error_type": "generation_failure",
        "root_cause": "rewritten query not passed into generator",
        "solution": "add rewritten_query into prompt",
        "timestamp": "2026-07-16T12:00:00+00:00",
    }


def test_badcase_round_trip_preserves_all_fields() -> None:
    case = BadCase.from_dict(_payload())

    assert BadCase.from_dict(case.to_dict()) == case
    assert list(case.to_dict()) == [
        "question",
        "answer",
        "expected_answer",
        "contexts",
        "sources",
        "error_type",
        "root_cause",
        "solution",
        "timestamp",
    ]


def test_from_dict_defaults_optional_annotations_to_none() -> None:
    payload = _payload()
    for field in ("expected_answer", "root_cause", "solution"):
        payload.pop(field)

    case = BadCase.from_dict(payload)

    assert (case.expected_answer, case.root_cause, case.solution) == (
        None,
        None,
        None,
    )


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ([], "badcase must be a JSON object"),
        ({**_payload(), "question": None}, "question must be a string"),
        ({**_payload(), "answer": 3}, "answer must be a string"),
        ({**_payload(), "contexts": "context"}, "contexts must be an array"),
        ({**_payload(), "contexts": [1]}, "contexts must be an array"),
        ({**_payload(), "sources": None}, "sources must be an array"),
        ({**_payload(), "error_type": False}, "error_type must be a string"),
        ({**_payload(), "timestamp": None}, "timestamp must be a string"),
        (
            {**_payload(), "expected_answer": 4},
            "expected_answer must be a string or null",
        ),
        ({**_payload(), "root_cause": []}, "root_cause must be a string or null"),
        ({**_payload(), "solution": {}}, "solution must be a string or null"),
    ],
)
def test_from_dict_rejects_invalid_field_types(
    payload: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        BadCase.from_dict(payload)  # type: ignore[arg-type]
