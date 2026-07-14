from app.pipeline import RAGResult
from scripts import smoke_pipeline_api as smoke


class FakeDualPipeline:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def ask(self, question, top_k=None):
        self.calls.append((question, top_k))
        return self.result


def _faq_result():
    return RAGResult(
        question="RAG 是什么？",
        answer="标准答案",
        contexts=[],
        sources=["README.md"],
        prompt="",
        route="faq",
        faq_id="faq-rag",
        faq_score=1.0,
        faq_match_type="exact",
        faq_cache_hit=False,
    )


def _rag_result():
    return RAGResult(
        question="深度问题",
        answer="生成答案",
        contexts=[{"text": "context"}],
        sources=["doc.md"],
        prompt="prompt",
        route="rag",
    )


def test_faq_smoke_prints_route_fields(monkeypatch, capsys):
    pipeline = FakeDualPipeline(_faq_result())
    monkeypatch.setattr(smoke, "build_pipeline", lambda: pipeline)

    result = smoke.main(
        [
            "--route",
            "faq",
            "--question",
            "RAG 是什么？",
            "--expect-route",
            "faq",
        ]
    )

    output = capsys.readouterr().out
    assert result == 0
    assert "route: faq" in output
    assert "faq_id: faq-rag" in output
    assert "faq_score: 1.0" in output
    assert "faq_match_type: exact" in output
    assert "faq_cache_hit: false" in output
    assert "latency_ms:" in output
    assert "Retriever returned 0 contexts" not in output
    assert pipeline.calls == [("RAG 是什么？", 5)]


def test_rag_smoke_prints_answer_and_sources(monkeypatch, capsys):
    monkeypatch.setattr(
        smoke, "build_pipeline", lambda: FakeDualPipeline(_rag_result())
    )

    result = smoke.main(
        ["--route", "rag", "--question", "深度问题", "--expect-route", "rag"]
    )

    output = capsys.readouterr().out
    assert result == 0
    assert "route: rag" in output
    assert "生成答案" in output
    assert "doc.md" in output


def test_smoke_route_mismatch_returns_nonzero(monkeypatch, capsys):
    monkeypatch.setattr(
        smoke, "build_pipeline", lambda: FakeDualPipeline(_rag_result())
    )

    result = smoke.main(
        ["--route", "faq", "--question", "问题", "--expect-route", "faq"]
    )

    assert result == 1
    assert "expected route faq" in capsys.readouterr().err
