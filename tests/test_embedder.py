from dataclasses import dataclass

import numpy as np
import pytest

import app.embeddings as embeddings_module
from app.embeddings import Embedder


EXPECTED_REQUIRED_LOCAL_MODEL_FILES = (
    "model.safetensors",
    "config.json",
    "modules.json",
    "config_sentence_transformers.json",
    "sentence_bert_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "sentencepiece.bpe.model",
    "1_Pooling/config.json",
)


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


def configure_test_model_path(monkeypatch, tmp_path):
    model_dir = tmp_path / "local-model"
    monkeypatch.setattr(
        embeddings_module,
        "LOCAL_EMBEDDING_MODEL_PATH",
        model_dir,
        raising=False,
    )
    monkeypatch.setattr(
        embeddings_module,
        "LOCAL_EMBEDDING_MODEL",
        str(model_dir),
        raising=False,
    )
    return model_dir


def create_required_model_files(model_dir):
    for file_name in embeddings_module.REQUIRED_LOCAL_MODEL_FILES:
        file_path = model_dir / file_name
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text("test", encoding="utf-8")


def test_required_local_model_files_cover_sentence_transformer_files():
    assert hasattr(embeddings_module, "REQUIRED_LOCAL_MODEL_FILES")
    assert (
        embeddings_module.REQUIRED_LOCAL_MODEL_FILES
        == EXPECTED_REQUIRED_LOCAL_MODEL_FILES
    )


def test_local_embedding_model_unavailable_when_directory_is_missing(
    monkeypatch,
    tmp_path,
):
    assert hasattr(embeddings_module, "_local_embedding_model_available")
    configure_test_model_path(monkeypatch, tmp_path)

    assert embeddings_module._local_embedding_model_available() is False


def test_local_embedding_model_unavailable_when_required_file_is_missing(
    monkeypatch,
    tmp_path,
):
    assert hasattr(embeddings_module, "_local_embedding_model_available")
    model_dir = configure_test_model_path(monkeypatch, tmp_path)
    create_required_model_files(model_dir)
    (model_dir / "model.safetensors").unlink()

    assert embeddings_module._local_embedding_model_available() is False


def test_local_embedding_model_available_when_required_files_exist(
    monkeypatch,
    tmp_path,
):
    assert hasattr(embeddings_module, "_local_embedding_model_available")
    model_dir = configure_test_model_path(monkeypatch, tmp_path)
    create_required_model_files(model_dir)

    assert embeddings_module._local_embedding_model_available() is True


def test_default_embedding_model_uses_remote_model_when_local_is_unavailable(
    monkeypatch,
    tmp_path,
):
    assert hasattr(embeddings_module, "_default_embedding_model")
    configure_test_model_path(monkeypatch, tmp_path)

    assert (
        embeddings_module._default_embedding_model()
        == embeddings_module.REMOTE_EMBEDDING_MODEL
    )


def test_default_embedding_model_uses_local_model_when_local_is_available(
    monkeypatch,
    tmp_path,
):
    assert hasattr(embeddings_module, "_default_embedding_model")
    model_dir = configure_test_model_path(monkeypatch, tmp_path)
    create_required_model_files(model_dir)

    assert embeddings_module._default_embedding_model() == str(model_dir)


def test_embedder_without_model_name_uses_runtime_default(monkeypatch):
    assert hasattr(embeddings_module, "_default_embedding_model")
    monkeypatch.setattr(
        embeddings_module,
        "_default_embedding_model",
        lambda: "runtime-default-model",
    )

    embedder = Embedder(model_name=None, model=FakeEmbeddingModel())

    assert embedder.model_name == "runtime-default-model"


def test_embedder_explicit_model_name_overrides_default_selection(monkeypatch):
    assert hasattr(embeddings_module, "_default_embedding_model")
    monkeypatch.setattr(
        embeddings_module,
        "_default_embedding_model",
        lambda: "runtime-default-model",
    )

    embedder = Embedder(model_name="custom-model", model=FakeEmbeddingModel())

    assert embedder.model_name == "custom-model"


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
