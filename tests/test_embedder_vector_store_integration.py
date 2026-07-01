import numpy as np
from uuid import uuid4

from app.embeddings import Embedder
from app.vector_store import ChromaVectorStore


class FakeEmbeddingModel:
    def __init__(self, dimension: int = 3):
        self.dimension = dimension

    def get_sentence_embedding_dimension(self) -> int:
        return self.dimension

    def encode(
        self,
        texts,
        batch_size,
        normalize_embeddings,
        convert_to_numpy,
        show_progress_bar,
    ):
        values = []
        for text in texts:
            if "alpha" in text:
                values.append([1.0, 0.0, 0.0])
            else:
                values.append([0.0, 1.0, 0.0])
        return np.array(values, dtype=np.float32)


def test_embedder_output_can_be_inserted_and_queried_in_vector_store():
    embedder = Embedder(model_name="fake-model", model=FakeEmbeddingModel())
    chunks = [
        {
            "id": "chunk-alpha",
            "text": "alpha text",
            "metadata": {"source": "alpha.md"},
        },
        {
            "id": "chunk-beta",
            "text": "beta text",
            "metadata": {"source": "beta.md"},
        },
    ]
    embedded_chunks = embedder.embed_chunks(chunks)
    store = ChromaVectorStore(
        collection_name=f"test_{uuid4().hex}",
        in_memory=True,
    )

    store.add_chunks(embedded_chunks)
    results = store.query([1.0, 0.0, 0.0], top_k=1)

    assert store.count() == 2
    assert len(results) == 1
    assert results[0]["id"] == "chunk-alpha"
    assert list(results[0].keys()) == ["id", "text", "metadata", "distance"]
