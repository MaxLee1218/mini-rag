from dataclasses import dataclass

import numpy as np
import pytest

from app.embeddings import Embedder


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
        return np.array(
            [
                [float(len(text) + column) for column in range(self.dimension)]
                for text in texts
            ],
            dtype=np.float32,
        )


def make_embedder(dimension: int = 3) -> Embedder:
    return Embedder(model_name="fake-model", model=FakeEmbeddingModel(dimension))


def test_embed_texts_returns_numpy_array_with_expected_shape():
    embedder = make_embedder(dimension=4)

    embeddings = embedder.embed_texts(["hello", "world"])

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (2, 4)


def test_embed_query_returns_single_embedding_vector():
    embedder = make_embedder(dimension=5)

    embedding = embedder.embed_query("hello")

    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (5,)


def test_embed_texts_rejects_empty_input():
    embedder = make_embedder()

    with pytest.raises(ValueError, match="texts must not be empty"):
        embedder.embed_texts([])


def test_embed_texts_rejects_blank_text():
    embedder = make_embedder()

    with pytest.raises(ValueError, match="texts must not contain blank strings"):
        embedder.embed_texts(["valid text", "   "])


def test_embed_chunks_preserves_id_text_and_metadata():
    embedder = make_embedder(dimension=2)
    chunks = [
        {
            "id": "chunk-a",
            "text": "hello",
            "metadata": {"source": "notes.md", "chunk_id": 0},
        }
    ]

    embedded_chunks = embedder.embed_chunks(chunks)

    assert embedded_chunks[0]["id"] == "chunk-a"
    assert embedded_chunks[0]["text"] == "hello"
    assert embedded_chunks[0]["metadata"] == {"source": "notes.md", "chunk_id": 0}


def test_embed_chunks_adds_embedding_fields_and_keeps_key_order():
    embedder = make_embedder(dimension=2)

    embedded_chunks = embedder.embed_chunks(["hello"])

    assert list(embedded_chunks[0].keys()) == [
        "id",
        "text",
        "metadata",
        "embedding",
        "embedding_model",
        "embedding_dimension",
    ]
    assert embedded_chunks[0]["id"] == "chunk-0"
    assert embedded_chunks[0]["metadata"] == {}
    assert embedded_chunks[0]["embedding"] == [5.0, 6.0]
    assert embedded_chunks[0]["embedding_model"] == "fake-model"
    assert embedded_chunks[0]["embedding_dimension"] == 2


@dataclass
class ObjectChunk:
    content: str
    source: str
    chunk_id: int


def test_embed_chunks_supports_object_chunks_and_preserves_extra_fields_as_metadata():
    embedder = make_embedder(dimension=2)
    chunk = ObjectChunk(content="object text", source="object.md", chunk_id=3)

    embedded_chunks = embedder.embed_chunks([chunk])

    assert embedded_chunks[0]["text"] == "object text"
    assert embedded_chunks[0]["metadata"] == {"source": "object.md", "chunk_id": 3}
