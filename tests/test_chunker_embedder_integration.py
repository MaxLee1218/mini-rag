import numpy as np

from app.chunker import split_document
from app.embeddings import Embedder


class FakeEmbeddingModel:
    def __init__(self, dimension: int = 4):
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
        return np.array(
            [
                [float(row + column) for column in range(self.dimension)]
                for row, _ in enumerate(texts)
            ],
            dtype=np.float32,
        )


def test_embedder_consumes_chunks_from_real_chunker():
    chunks = split_document(
        {
            "content": "RAG retrieves context before generating an answer.",
            "source": "sample.md",
        },
        chunk_size=20,
        chunk_overlap=0,
    )
    embedder = Embedder(model_name="fake-model", model=FakeEmbeddingModel())

    embedded_chunks = embedder.embed_chunks(chunks)

    assert len(embedded_chunks) == len(chunks)
    assert embedded_chunks[0]["id"] == "chunk-0"
    assert embedded_chunks[0]["text"] == chunks[0]["content"]
    assert embedded_chunks[0]["metadata"] == {
        "source": "sample.md",
        "chunk_id": 0,
    }
    assert list(embedded_chunks[0].keys()) == [
        "id",
        "text",
        "metadata",
        "embedding",
        "embedding_model",
        "embedding_dimension",
    ]
    assert len(embedded_chunks[0]["embedding"]) == 4
