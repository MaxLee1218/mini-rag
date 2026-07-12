from types import SimpleNamespace

from scripts import ask as ask_script


class FakePipeline:
    def __init__(self, result=None):
        self.result = result or SimpleNamespace(
            question="RAG是什么？",
            answer="RAG 是检索增强生成。",
            contexts=[{"text": "RAG 是检索增强生成。", "source": "docs/rag.md"}],
            sources=[],
        )
        self.calls = []

    def ask(self, question, top_k=None):
        self.calls.append({"question": question, "top_k": top_k})
        return self.result


def test_format_result_outputs_question_and_answer():
    result = SimpleNamespace(
        question="RAG是什么？",
        answer="RAG 是检索增强生成。",
        contexts=[],
        sources=[],
    )

    output = ask_script.format_result(result)

    assert "问题：" in output
    assert "RAG是什么？" in output
    assert "回答：" in output
    assert "RAG 是检索增强生成。" in output


def test_single_question_mode_calls_fake_pipeline_once_and_returns_zero(
    monkeypatch,
    capsys,
):
    fake_pipeline = FakePipeline()
    monkeypatch.setattr(
        ask_script,
        "build_default_pipeline",
        lambda top_k: fake_pipeline,
    )

    exit_code = ask_script.main(["RAG是什么？"])

    assert exit_code == 0
    assert fake_pipeline.calls == [{"question": "RAG是什么？", "top_k": 5}]
    assert "RAG 是检索增强生成。" in capsys.readouterr().out


def test_single_question_mode_rejects_blank_question(monkeypatch, capsys):
    monkeypatch.setattr(
        ask_script,
        "build_default_pipeline",
        lambda top_k: FakePipeline(),
    )

    exit_code = ask_script.main([""])

    captured = capsys.readouterr()
    assert exit_code != 0
    assert "问题不能为空" in captured.err


def test_interactive_mode_exits_on_q(monkeypatch, capsys):
    fake_pipeline = FakePipeline()
    monkeypatch.setattr(
        ask_script,
        "build_default_pipeline",
        lambda top_k: fake_pipeline,
    )
    monkeypatch.setattr("builtins.input", lambda prompt="": "q")

    exit_code = ask_script.main([])

    assert exit_code == 0
    assert fake_pipeline.calls == []
    assert "请输入问题" in capsys.readouterr().out


def test_interactive_mode_asks_one_question_then_exits(monkeypatch, capsys):
    fake_pipeline = FakePipeline()
    answers = iter(["RAG是什么？", "Q"])
    monkeypatch.setattr(
        ask_script,
        "build_default_pipeline",
        lambda top_k: fake_pipeline,
    )
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))

    exit_code = ask_script.main([])

    assert exit_code == 0
    assert fake_pipeline.calls == [{"question": "RAG是什么？", "top_k": 5}]
    assert "RAG 是检索增强生成。" in capsys.readouterr().out


def test_show_context_displays_document_key_summary():
    long_text = "文档正文" * 80
    result = SimpleNamespace(
        question="问题",
        answer="回答",
        contexts=[
            {
                "document": long_text,
                "metadata": {"source": "docs/rag.md"},
            }
        ],
        sources=[],
    )

    output = ask_script.format_result(result, show_context=True)

    assert "检索上下文：" in output
    assert "[1] docs/rag.md" in output
    assert "文档正文" in output
    assert long_text not in output


def test_answer_with_sources_marker_does_not_append_duplicate_sources():
    result = SimpleNamespace(
        question="问题",
        answer="回答 [1]\n\n来源：\n- [1] docs/rag.md",
        contexts=[],
        sources=["docs/rag.md"],
    )

    output = ask_script.format_result(result)

    assert output.count("来源：") == 1
    assert output.count("docs/rag.md") == 1


def test_no_sources_does_not_print_result_sources():
    result = SimpleNamespace(
        question="问题",
        answer="回答",
        contexts=[],
        sources=["docs/rag.md"],
    )

    output = ask_script.format_result(result, show_sources=False)

    assert "来源：" not in output
    assert "docs/rag.md" not in output
