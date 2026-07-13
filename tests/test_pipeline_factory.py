from types import SimpleNamespace

from app.hybrid_retriever import HybridRetriever
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
