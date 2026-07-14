from types import SimpleNamespace

from app.hybrid_retriever import HybridRetriever
from app.faq.models import FAQRecord
from app.faq.repository import FAQRepository
from app.pipeline import RAGResult
from app.pipeline_factory import build_default_pipeline
from app.retriever import Retriever


def test_default_pipeline_builds_hybrid_retriever_from_stored_chunks(monkeypatch, tmp_path):
    import app.config as config
    import app.embeddings as embeddings_module
    import app.generator as generator_module
    import app.reranker as reranker_module
    import app.vector_store as vector_store_module

    class FakeCollection:
        def get(self, include=None):
            assert include == ["documents", "metadatas"]
            return {
                "ids": ["chunk-a"],
                "documents": ["alpha text"],
                "metadatas": [{"source": "alpha.md", "chunk_id": 0}],
            }

    class FakeVectorStore:
        def __init__(self, **kwargs):
            self.collection = FakeCollection()

        def count(self):
            return 1

    fake_embedder = object()
    monkeypatch.setattr(config, "VECTOR_DB_PATH", str(tmp_path))
    monkeypatch.setattr(config, "HYBRID_SPARSE_WEIGHT", 0.4)
    monkeypatch.setattr(config, "HYBRID_DENSE_WEIGHT", 0.6)
    monkeypatch.setattr(config, "HYBRID_CANDIDATE_MULTIPLIER", 3)
    monkeypatch.setattr(config, "RERANKER_ENABLED", True)
    monkeypatch.setattr(config, "RERANKER_MODEL", "./models/test-reranker")
    monkeypatch.setattr(config, "RERANKER_TOP_K", 5)
    monkeypatch.setattr(config, "RERANKER_CANDIDATE_K", 10)
    monkeypatch.setattr(config, "RERANKER_BATCH_SIZE", 16)
    monkeypatch.setattr(config, "RERANKER_MAX_LENGTH", 256)
    monkeypatch.setattr(config, "RERANKER_DEVICE", "cpu")
    monkeypatch.setattr(config, "RERANKER_FAILURE_MODE", "fallback")
    monkeypatch.setattr(config, "RERANKER_LOCAL_FILES_ONLY", True)
    reranker_calls = []

    class FakeReranker:
        def __init__(self, model_name_or_path, **kwargs):
            reranker_calls.append((model_name_or_path, kwargs))

        def rerank(self, query, documents, top_k=None):
            return list(documents)[:top_k]

    monkeypatch.setattr(reranker_module, "CrossEncoderReranker", FakeReranker)
    monkeypatch.setattr(embeddings_module, "Embedder", lambda: fake_embedder)
    monkeypatch.setattr(vector_store_module, "ChromaVectorStore", FakeVectorStore)
    monkeypatch.setattr(
        generator_module, "load_deepseek_config_from_env", lambda: SimpleNamespace()
    )
    monkeypatch.setattr(
        generator_module, "DeepSeekGenerator", lambda config: ("generator", config)
    )

    pipeline = build_default_pipeline()

    assert isinstance(pipeline.retriever, HybridRetriever)
    assert pipeline.retriever.sparse_retriever.documents == [
        {
            "id": "chunk-a",
            "text": "alpha text",
            "metadata": {"source": "alpha.md", "chunk_id": 0},
        }
    ]
    assert pipeline.retriever.dense_retriever.embedder is fake_embedder
    assert pipeline.retriever.sparse_weight == 0.4
    assert pipeline.retriever.dense_weight == 0.6
    assert pipeline.retriever.candidate_multiplier == 3
    assert pipeline.candidate_k == 10
    assert pipeline.final_top_k == 5
    assert isinstance(pipeline.reranker, FakeReranker)
    assert reranker_calls == [
        (
            "./models/test-reranker",
            {
                "batch_size": 16,
                "max_length": 256,
                "device": "cpu",
                "failure_mode": "fallback",
                "local_files_only": True,
            },
        )
    ]


def test_pipeline_source_does_not_hardcode_tinybert_model():
    from pathlib import Path

    assert "ms-marco-TinyBERT" not in Path("app/pipeline.py").read_text()


def test_default_pipeline_parent_child_mode_builds_resolving_retriever(
    monkeypatch, tmp_path
):
    import app.config as config
    import app.embeddings as embeddings_module
    import app.generator as generator_module
    import app.parent_store as parent_store_module
    import app.vector_store as vector_store_module

    class FakeVectorStore:
        def __init__(self, **kwargs):
            self.collection = SimpleNamespace(get=lambda include=None: {})

        def count(self):
            return 2

    fake_parent_store = object()
    (tmp_path / "parents.sqlite3").touch()
    monkeypatch.setattr(config, "VECTOR_DB_PATH", str(tmp_path))
    monkeypatch.setattr(config, "CHUNK_MODE", "parent-child")
    monkeypatch.setattr(config, "PARENT_STORE_PATH", str(tmp_path / "parents.sqlite3"))
    monkeypatch.setattr(config, "RERANKER_ENABLED", False)
    monkeypatch.setattr(config, "RERANKER_TOP_K", 4)
    monkeypatch.setattr(embeddings_module, "Embedder", lambda: "embedder")
    monkeypatch.setattr(vector_store_module, "ChromaVectorStore", FakeVectorStore)
    monkeypatch.setattr(parent_store_module, "SQLiteParentStore", lambda path: fake_parent_store)
    monkeypatch.setattr(
        generator_module, "load_deepseek_config_from_env", lambda: SimpleNamespace()
    )
    monkeypatch.setattr(generator_module, "DeepSeekGenerator", lambda config: "generator")

    pipeline = build_default_pipeline()

    assert isinstance(pipeline.retriever, Retriever)
    assert pipeline.retriever.mode == "parent-child"
    assert pipeline.retriever.parent_store is fake_parent_store
    assert pipeline.candidate_k == 4
    assert pipeline.expand_retrieval_candidates is False


def _seed_faq(path):
    repository = FAQRepository(path)
    repository.import_records(
        [
            FAQRecord(
                id="faq-rag",
                question="什么是 RAG？",
                aliases=("RAG 是什么？",),
                answer="标准答案",
                source="README.md",
            )
        ]
    )


def _fake_rag_result(question="深度问题"):
    return RAGResult(
        question=question,
        answer="rag answer",
        contexts=[],
        sources=[],
        prompt="prompt",
    )


def test_dual_factory_faq_hit_does_not_build_deep_pipeline(
    monkeypatch, tmp_path
):
    import app.config as config
    import app.pipeline_factory as factory

    db_path = tmp_path / "faq.db"
    _seed_faq(db_path)
    monkeypatch.setattr(config, "FAQ_ENABLED", True)
    monkeypatch.setattr(config, "FAQ_DB_PATH", db_path)
    monkeypatch.setattr(config, "FAQ_CACHE_ENABLED", False)

    def forbidden(*args, **kwargs):
        raise AssertionError("deep pipeline must stay lazy")

    monkeypatch.setattr(factory, "build_default_pipeline", forbidden)

    pipeline = factory.build_default_dual_path_pipeline()
    result = pipeline.ask("RAG 是什么？")

    assert result.route == "faq"
    assert result.answer == "标准答案"


def test_dual_factory_disabled_faq_calls_deep_pipeline_lazily(monkeypatch):
    import app.config as config
    import app.pipeline_factory as factory

    calls = []

    class FakeRAG:
        def ask(self, question, top_k=None, *, retrieval_query=None):
            return _fake_rag_result(question)

    monkeypatch.setattr(config, "FAQ_ENABLED", False)
    monkeypatch.setattr(
        factory,
        "build_default_pipeline",
        lambda top_k=None: calls.append(top_k) or FakeRAG(),
    )
    pipeline = factory.build_default_dual_path_pipeline(top_k=6)
    assert calls == []

    assert pipeline.ask("深度问题", retrieval_query="深度问题").route == "rag"
    assert calls == [6]


def test_dual_factory_faq_database_failure_degrades_to_rag(
    monkeypatch, tmp_path
):
    import app.config as config
    import app.pipeline_factory as factory

    db_directory = tmp_path / "not-a-database"
    db_directory.mkdir()

    class FakeRAG:
        def ask(self, question, top_k=None, *, retrieval_query=None):
            return _fake_rag_result(question)

    monkeypatch.setattr(config, "FAQ_ENABLED", True)
    monkeypatch.setattr(config, "FAQ_DB_PATH", db_directory)
    monkeypatch.setattr(config, "FAQ_CACHE_ENABLED", False)
    monkeypatch.setattr(factory, "build_default_pipeline", lambda top_k=None: FakeRAG())

    pipeline = factory.build_default_dual_path_pipeline()

    assert pipeline.ask("深度问题", retrieval_query="深度问题").route == "rag"


def test_default_dual_pipeline_is_a_process_singleton(monkeypatch):
    import app.config as config
    import app.pipeline_factory as factory

    monkeypatch.setattr(config, "FAQ_ENABLED", False)
    factory.reset_default_dual_path_pipeline()
    try:
        first = factory.get_default_dual_path_pipeline()
        second = factory.get_default_dual_path_pipeline()
    finally:
        factory.reset_default_dual_path_pipeline()

    assert first is second
