import json

from scripts import faq_admin


def _record():
    return {
        "id": "faq-rag",
        "question": "什么是 RAG？",
        "aliases": ["RAG 是什么？", "解释一下 RAG", "什么叫检索增强生成？"],
        "answer": "RAG 是检索增强生成。",
        "source": "README.md",
        "enabled": True,
    }


def test_admin_init_import_and_list_are_repeatable(tmp_path, capsys):
    db_path = tmp_path / "nested" / "faq.db"
    source = tmp_path / "faqs.json"
    source.write_text(json.dumps([_record()], ensure_ascii=False), encoding="utf-8")

    assert faq_admin.main(["--db-path", str(db_path), "init"]) == 0
    assert faq_admin.main(["--db-path", str(db_path), "init"]) == 0
    capsys.readouterr()
    assert faq_admin.main(
        ["--db-path", str(db_path), "import", str(source)]
    ) == 0
    first_output = capsys.readouterr().out
    assert "inserted=1 updated=0 unchanged=0" in first_output
    assert faq_admin.main(
        ["--db-path", str(db_path), "import", str(source)]
    ) == 0
    assert "inserted=0 updated=0 unchanged=1" in capsys.readouterr().out

    assert faq_admin.main(["--db-path", str(db_path), "list"]) == 0
    output = capsys.readouterr().out
    assert "faq-rag" in output
    assert "什么是 RAG？" in output
    assert "enabled=true" in output
    assert "aliases=3" in output
    assert _record()["answer"] not in output


def test_admin_invalid_json_returns_nonzero(tmp_path, capsys):
    source = tmp_path / "broken.json"
    source.write_text("{", encoding="utf-8")

    result = faq_admin.main(
        ["--db-path", str(tmp_path / "faq.db"), "import", str(source)]
    )

    assert result == 1
    assert "ERROR:" in capsys.readouterr().err


def test_admin_invalid_or_unknown_fields_return_nonzero(tmp_path, capsys):
    invalid_records = [
        {**_record(), "enabled": "true"},
        {**_record(), "question": ""},
        {**_record(), "unknown": "field"},
    ]
    for index, record in enumerate(invalid_records):
        source = tmp_path / f"invalid-{index}.json"
        source.write_text(json.dumps([record], ensure_ascii=False), encoding="utf-8")
        assert faq_admin.main(
            ["--db-path", str(tmp_path / "faq.db"), "import", str(source)]
        ) == 1
        capsys.readouterr()
