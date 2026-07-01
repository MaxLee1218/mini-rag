from uuid import uuid4

import numpy as np

from app.embeddings import Embedder
from app.retriever import Retriever
from app.vector_store import ChromaVectorStore


class FakeEmbeddingModel:
    def get_sentence_embedding_dimension(self) -> int:
        return 3

    def encode(
        self,
        texts,
        batch_size,
        normalize_embeddings,
        convert_to_numpy,
        show_progress_bar,
    ):
        vectors = [self._vector_for_text(text) for text in texts]
        return np.array(vectors, dtype=np.float32)

    def _vector_for_text(self, text):
        lowered = text.lower()
        if "rag" in lowered or "retrieval" in lowered:
            return [1.0, 0.0, 0.0]
        if "embedding" in lowered:
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]


def test_retriever_uses_real_embedder_and_vector_store_with_fake_model():
    embedder = Embedder(model_name="fake-model", model=FakeEmbeddingModel())
    vector_store = ChromaVectorStore(
        collection_name=f"test_{uuid4().hex}",
        in_memory=True,
    )
    chunks = [
        {
            "id": "chunk-rag",
            "text": "RAG uses retrieval to find relevant context.",
            "metadata": {"source": "rag.md"},
        },
        {
            "id": "chunk-embedding",
            "text": "Embeddings turn text into vectors.",
            "metadata": {"source": "embeddings.md"},
        },
        {
            "id": "chunk-weather",
            "text": "Weather forecasts describe rain and sunshine.",
            "metadata": {"source": "weather.md"},
        },
    ]

    try:
        vector_store.add_chunks(embedder.embed_chunks(chunks))
        retriever = Retriever(embedder=embedder, vector_store=vector_store)

        results = retriever.retrieve("What is RAG?", top_k=2)
        context = retriever.retrieve_context("What is RAG?", top_k=2)

        assert len(results) >= 1
        assert results[0]["id"] == "chunk-rag"
        assert list(results[0].keys()) == ["id", "text", "metadata", "distance"]
        assert "RAG uses retrieval" in context
        assert "chunk-rag" in context
        assert "rag.md" in context
    finally:
        vector_store.close()
