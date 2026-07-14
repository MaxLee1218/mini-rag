from datetime import datetime, timezone

from app.conversation.models import ConversationTurn
from app.dual_path_pipeline import DualPathPipeline, PreparedRAGQuery
from app.faq.models import FAQMatch
from app.pipeline import RAGResult
from app.query_preparation import ConversationQueryPreparer
from app.query_rewriter.base import QueryRewriteResult


def _faq_match(match_type="exact"):
    return FAQMatch(
        faq_id="faq-rag",
        question="什么是 RAG？",
        answer="maintained answer",
        source="README.md",
        score=1.0,
        match_type=match_type,
    )


def _rag_result():
    return RAGResult(
        question="它是什么？",
        answer="rag answer",
        contexts=[{"text": "context", "metadata": {"source": "doc.md"}}],
        sources=["doc.md"],
        prompt="rag prompt",
    )


class FakeCache:
    def __init__(self):
        self.values = {}

    def get(self, question, version):
        return self.values.get((question, version))

    def set(self, question, version, match):
        self.values[(question, version)] = FAQMatch(
            faq_id=match.faq_id,
            question=match.question,
            answer=match.answer,
            source=match.source,
            score=match.score,
            match_type="cache",
        )


class StubMatcher:
    def __init__(self, match):
        self.result = match
        self.calls = []
        self.fail_if_called = False

    def match(self, question):
        if self.fail_if_called:
            raise AssertionError("matcher should not be called")
        self.calls.append(question)
        return self.result


class SpyRAG:
    def __init__(self):
        self.calls = []

    def ask(self, question, top_k=None, *, retrieval_query=None):
        self.calls.append((question, top_k, retrieval_query))
        return _rag_result()


class SpyProvider:
    def __init__(self, rag):
        self.rag = rag
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return self.rag


class SpyPreparer:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def prepare(self, question, session_id):
        self.calls.append((question, session_id))
        return self.result


def test_faq_hit_does_not_create_rag_or_rewrite_query():
    def forbidden(*args, **kwargs):
        raise AssertionError("deep path must not be touched")

    pipeline = DualPathPipeline(
        faq_matcher=StubMatcher(_faq_match()),
        faq_cache=FakeCache(),
        rag_pipeline_provider=forbidden,
        faq_index_version=3,
        rag_query_preparer=forbidden,
    )

    result = pipeline.ask(" RAG 是什么？ ", top_k=9, session_id="s1")

    assert result.route == "faq"
    assert result.faq_id == "faq-rag"
    assert result.answer == "maintained answer"
    assert result.sources == ["README.md"]
    assert result.contexts == []
    assert result.prompt == ""
    assert result.faq_cache_hit is False
    assert result.rewritten_query == "RAG 是什么？"
    assert result.rewrite_reason == "faq_fast_path"


def test_second_request_hits_cache_and_skips_matcher():
    matcher = StubMatcher(_faq_match())
    pipeline = DualPathPipeline(
        faq_matcher=matcher,
        faq_cache=FakeCache(),
        rag_pipeline_provider=lambda: SpyRAG(),
        faq_index_version=4,
    )

    first = pipeline.ask("RAG 是什么？")
    matcher.fail_if_called = True
    second = pipeline.ask("RAG 是什么？")

    assert first.faq_cache_hit is False
    assert second.faq_cache_hit is True
    assert second.faq_match_type == "cache"


def test_cache_index_version_is_part_of_routing_lookup():
    cache = FakeCache()
    cache.set("rag 是什么", 1, _faq_match())
    matcher = StubMatcher(None)
    rag = SpyRAG()
    pipeline = DualPathPipeline(
        faq_matcher=matcher,
        faq_cache=cache,
        rag_pipeline_provider=lambda: rag,
        faq_index_version=2,
    )

    result = pipeline.ask("RAG 是什么？", retrieval_query="RAG 是什么？")

    assert result.route == "rag"
    assert matcher.calls == ["RAG 是什么？"]


def test_miss_prepares_query_and_calls_existing_rag_once():
    rag = SpyRAG()
    provider = SpyProvider(rag)
    preparer = SpyPreparer(
        PreparedRAGQuery("resolved", True, "llm_rewrite", 2)
    )
    pipeline = DualPathPipeline(
        faq_matcher=StubMatcher(None),
        faq_cache=FakeCache(),
        rag_pipeline_provider=provider,
        faq_index_version=1,
        rag_query_preparer=preparer,
    )

    result = pipeline.ask("它是什么？", top_k=7, session_id="session-1")

    assert provider.calls == 1
    assert preparer.calls == [("它是什么？", "session-1")]
    assert rag.calls == [("它是什么？", 7, "resolved")]
    assert result.route == "rag"
    assert result.answer == "rag answer"
    assert result.contexts == _rag_result().contexts
    assert result.sources == ["doc.md"]
    assert result.prompt == "rag prompt"
    assert result.rewritten_query == "resolved"
    assert result.query_was_rewritten is True
    assert result.faq_id is None


def test_explicit_retrieval_query_skips_preparer():
    rag = SpyRAG()

    def forbidden(*args, **kwargs):
        raise AssertionError("preparer should not be called")

    pipeline = DualPathPipeline(
        faq_matcher=StubMatcher(None),
        faq_cache=FakeCache(),
        rag_pipeline_provider=lambda: rag,
        faq_index_version=1,
        rag_query_preparer=forbidden,
    )

    result = pipeline.ask("原问题", top_k=3, retrieval_query="检索问题")

    assert rag.calls == [("原问题", 3, "检索问题")]
    assert result.rewritten_query == "检索问题"


def test_matcher_exception_degrades_to_rag():
    class BrokenMatcher:
        def match(self, question):
            raise RuntimeError("matcher failed")

    rag = SpyRAG()
    pipeline = DualPathPipeline(
        faq_matcher=BrokenMatcher(),
        faq_cache=FakeCache(),
        rag_pipeline_provider=lambda: rag,
        faq_index_version=1,
    )

    assert pipeline.ask("问题", retrieval_query="问题").route == "rag"


def test_conversation_query_preparer_uses_history_and_rewriter():
    turn = ConversationTurn(
        user_message="什么是 Middleware？",
        assistant_message="它是中间件。",
        created_at=datetime.now(timezone.utc),
    )

    class Store:
        def get_recent_turns(self, session_id, limit):
            assert (session_id, limit) == ("s1", 5)
            return [turn]

    class Rewriter:
        def rewrite(self, question, history):
            assert history == [turn]
            return QueryRewriteResult(
                question, "Middleware 为什么有用？", True, "llm_rewrite"
            )

    preparer = ConversationQueryPreparer(
        store=Store(), rewriter=Rewriter(), history_limit=5, enabled=True
    )

    result = preparer.prepare("它为什么有用？", "s1")

    assert result == PreparedRAGQuery(
        "Middleware 为什么有用？", True, "llm_rewrite", 1
    )


def test_conversation_query_preparer_failure_uses_original_question():
    class Store:
        def get_recent_turns(self, session_id, limit):
            raise RuntimeError("store unavailable")

    class Rewriter:
        def rewrite(self, question, history):
            raise RuntimeError("provider unavailable")

    preparer = ConversationQueryPreparer(
        store=Store(), rewriter=Rewriter(), history_limit=5, enabled=True
    )

    result = preparer.prepare("原问题", "s1")

    assert result.retrieval_query == "原问题"
    assert result.query_was_rewritten is False
    assert result.history_turn_count == 0
