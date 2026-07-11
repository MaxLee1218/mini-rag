from types import SimpleNamespace

from app.hybrid_retriever import HybridRetriever
from app.pipeline_factory import build_default_pipeline


def test_default_pipeline_builds_hybrid_retriever_from_stored_chunks(monkeypatch, tmp_path):
    import app.config as config
    import app.embeddings as embeddings_module
    import app.generator as generator_module
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
