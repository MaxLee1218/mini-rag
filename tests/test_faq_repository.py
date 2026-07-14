from dataclasses import replace

import pytest

from app.faq.models import FAQRecord
from app.faq.repository import FAQRepository


def _record(**changes):
    values = {
        "id": "faq-password",
        "question": "如何重置密码？",
        "answer": "打开安全设置。",
        "aliases": ("忘记密码怎么办", "密码找回方法"),
        "source": None,
        "enabled": True,
    }
    values.update(changes)
    return FAQRecord(**values)


def test_schema_is_repeatable_and_initial_version_is_zero(tmp_path):
    repository = FAQRepository(tmp_path / "nested" / "faq.db")

    repository.ensure_schema()
    repository.ensure_schema()

    assert repository.get_index_version() == 0


def test_import_is_idempotent_and_versions_effective_changes_only(tmp_path):
    repository = FAQRepository(tmp_path / "faq.db")
    item = _record()

    first = repository.import_records([item])
    second = repository.import_records([item])
    changed = repository.import_records(
        [replace(item, answer="使用找回入口。", aliases=("找回密码",))]
    )

    assert (first.inserted, first.updated, first.unchanged) == (1, 0, 0)
    assert (second.inserted, second.updated, second.unchanged) == (0, 0, 1)
    assert (changed.inserted, changed.updated, changed.unchanged) == (0, 1, 0)
    assert first.index_version == second.index_version == 1
    assert changed.index_version == 2
    assert repository.list_enabled() == [
        replace(item, answer="使用找回入口。", aliases=("找回密码",))
    ]


def test_aliases_are_replaced_and_duplicate_normalized_aliases_are_deduplicated(
    tmp_path,
):
    repository = FAQRepository(tmp_path / "faq.db")
    repository.import_records([_record(aliases=("RAG 是什么？", "rag 是什么?"))])

    stored = repository.list_all()[0]

    assert stored.aliases == ("RAG 是什么？",)


def test_disabled_records_are_excluded_and_enabled_change_versions(tmp_path):
    repository = FAQRepository(tmp_path / "faq.db")
    disabled = _record(enabled=False)

    first = repository.import_records([disabled])
    second = repository.import_records([replace(disabled, enabled=True)])

    assert repository.list_enabled() == [replace(disabled, enabled=True)]
    assert first.index_version == 1
    assert second.index_version == 2


def test_source_can_be_none(tmp_path):
    repository = FAQRepository(tmp_path / "faq.db")

    repository.import_records([_record(source=None)])

    assert repository.list_all()[0].source is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("id", ""),
        ("question", " "),
        ("answer", ""),
        ("enabled", 1),
        ("aliases", ("",)),
    ],
)
def test_invalid_record_is_rejected_without_partial_write(
    tmp_path, field, value
):
    repository = FAQRepository(tmp_path / "faq.db")
    valid = _record(id="valid")
    invalid = replace(_record(id="invalid"), **{field: value})

    with pytest.raises(ValueError):
        repository.import_records([valid, invalid])

    assert repository.list_all() == []


def test_delete_via_replace_import_is_not_supported_implicitly(tmp_path):
    repository = FAQRepository(tmp_path / "faq.db")
    repository.import_records([_record()])

    summary = repository.import_records([])

    assert summary.unchanged == 0
    assert summary.index_version == 1
    assert repository.list_all() == [_record()]
