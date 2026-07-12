from app.query_rewriter.llm_rewriter import LLMQueryRewriter
from scripts.smoke_conversation import _build_mock_rewriter, main


def test_mock_smoke_uses_llm_rewriter_with_offline_completion():
    assert isinstance(_build_mock_rewriter(), LLMQueryRewriter)


def test_mock_smoke_runs_two_contextual_rounds_without_network(capsys):
    assert main(["--mock"]) == 0
    output = capsys.readouterr().out
    assert "Session: demo-session" in output
    assert "Original question: 什么是 Middleware？" in output
    assert "Rewritten query: 什么是 Middleware？" in output
    assert "Was rewritten: False" in output
    assert "Original question: 它为什么可以优化查询？" in output
    assert "Rewritten query: Middleware 为什么可以优化查询？" in output
    assert "Was rewritten: True" in output
    assert "Stored turns: 2" in output
