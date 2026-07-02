from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from scripts import ingest as ingest_script


class FakeEmbedder:
    def __init__(self):
        self.received_chunks = None

    def embed_chunks(self, chunks):
        self.received_chunks = list(chunks)
        return [
            {
                **chunk,
                "embedding": [1.0, 0.0],
                "embedding_model": "fake-model",
                "embedding_dimension": 2,
            }
            for chunk in self.received_chunks
        ]


class FakeVectorStore:
    def __init__(self):
        self.reset_called = False
        self.added_chunks = None

    def reset(self):
        self.reset_called = True

    def add_chunks(self, chunks):
        self.added_chunks = list(chunks)

    def count(self):
        if self.added_chunks is None:
            return 0
        return len(self.added_chunks)


class ExplodingComponent:
    def __getattr__(self, name):
        raise AssertionError(f"dry run should not access {name}")


@dataclass
class ObjectChunk:
    content: str
    source: str
    chunk_id: int


def test_normalize_extensions_accepts_common_forms():
    assert ingest_script.normalize_extensions("txt,md") == {".txt", ".md"}
    assert ingest_script.normalize_extensions(".txt,.md") == {".txt", ".md"}
    assert ingest_script.normalize_extensions(" .TXT , .Md ") == {".txt", ".md"}


def test_iter_input_files_supports_single_file_case_insensitively(tmp_path):
    raw_file = tmp_path / "Notes.TXT"
    raw_file.write_text("alpha", encoding="utf-8")

    files = ingest_script.iter_input_files(raw_file, {".txt"})

    assert files == [raw_file]


def test_iter_input_files_recurses_in_sorted_order_and_skips_noise(tmp_path):
    first = tmp_path / "a" / "first.md"
    second = tmp_path / "b" / "second.TXT"
    ignored = tmp_path / "b" / "ignored.json"
    cache_file = tmp_path / ".pytest_cache" / "cached.txt"
    chroma_file = tmp_path / "data" / "chroma" / "db.txt"
    pycache_file = tmp_path / "__pycache__" / "compiled.txt"
    for file_path in [first, second, ignored, cache_file, chroma_file, pycache_file]:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text("content", encoding="utf-8")

    files = ingest_script.iter_input_files(tmp_path, {".txt", ".md"})

    assert files == [first, second]


def test_iter_input_files_skips_data_chroma_even_when_input_is_chroma(tmp_path):
    chroma_file = tmp_path / "data" / "chroma" / "db.txt"
    chroma_file.parent.mkdir(parents=True)
    chroma_file.write_text("database file", encoding="utf-8")

    with pytest.raises(ValueError, match="no supported files found"):
        ingest_script.iter_input_files(chroma_file.parent, {".txt"})
    with pytest.raises(ValueError, match="no supported files found"):
        ingest_script.iter_input_files(chroma_file, {".txt"})


def test_iter_input_files_raises_when_no_supported_files_exist(tmp_path):
    raw_file = tmp_path / "image.png"
    raw_file.write_text("not text", encoding="utf-8")

    with pytest.raises(ValueError, match="no supported files found"):
        ingest_script.iter_input_files(tmp_path, {".txt", ".md"})


def test_single_file_ingestion_returns_summary_and_stores_embedded_chunks(tmp_path):
    raw_file = tmp_path / "RAG Notes.txt"
    raw_file.write_text("Retrieval augmented generation uses context.", encoding="utf-8")
    embedder = FakeEmbedder()
    vector_store = FakeVectorStore()

    summary = ingest_script.ingest(
        input_path=raw_file,
        collection="test_collection",
        persist_path=str(tmp_path / "chroma"),
        embedder=embedder,
        vector_store=vector_store,
    )

    assert summary["input_path"] == str(raw_file)
    assert summary["collection"] == "test_collection"
    assert summary["persist_path"] == str(tmp_path / "chroma")
    assert summary["reset"] is False
    assert summary["dry_run"] is False
    assert summary["files_indexed"] == 1
    assert summary["chunks_created"] == 1
    assert summary["chunks_stored"] == 1
    assert embedder.received_chunks is not None
    stored_chunk = vector_store.added_chunks[0]
    prepared_chunk = embedder.received_chunks[0]
    assert prepared_chunk["id"] == stored_chunk["id"]
    assert prepared_chunk["id"].startswith("rag_notes-")
    assert prepared_chunk["id"].endswith("-chunk-0")
    assert prepared_chunk["metadata"]["filename"] == "RAG Notes.txt"
    assert prepared_chunk["metadata"]["relative_path"] == "RAG Notes.txt"
    assert prepared_chunk["metadata"]["chunk_index"] == 0


def test_directory_ingestion_uses_stable_distinct_project_relative_ids(tmp_path):
    first = tmp_path / "sample" / "same.txt"
    second = tmp_path / "other" / "same.txt"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_text("alpha content", encoding="utf-8")
    second.write_text("beta content", encoding="utf-8")
    first_store = FakeVectorStore()
    second_store = FakeVectorStore()

    ingest_script.ingest(
        input_path=tmp_path,
        collection="test_collection",
        persist_path=str(tmp_path / "chroma"),
        embedder=FakeEmbedder(),
        vector_store=first_store,
    )
    ingest_script.ingest(
        input_path=tmp_path,
        collection="test_collection",
        persist_path=str(tmp_path / "chroma"),
        embedder=FakeEmbedder(),
        vector_store=second_store,
    )

    first_ids = [chunk["id"] for chunk in first_store.added_chunks]
    second_ids = [chunk["id"] for chunk in second_store.added_chunks]
    assert first_ids == second_ids
    assert len(first_ids) == len(set(first_ids))
    assert {
        chunk["metadata"]["relative_path"] for chunk in first_store.added_chunks
    } == {"other/same.txt", "sample/same.txt"}


def test_ingest_raises_for_empty_document(tmp_path):
    raw_file = tmp_path / "empty.md"
    raw_file.write_text("   \n\t", encoding="utf-8")

    with pytest.raises(ValueError, match="document is empty"):
        ingest_script.ingest(
            input_path=raw_file,
            collection="test_collection",
            persist_path=str(tmp_path / "chroma"),
            embedder=FakeEmbedder(),
            vector_store=FakeVectorStore(),
        )


def test_metadata_merge_order_and_sanitization(tmp_path):
    raw_file = tmp_path / "notes.txt"
    raw_file.write_text("alpha", encoding="utf-8")
    embedder = FakeEmbedder()
    vector_store = FakeVectorStore()

    def loader(file_path, base_path=None):
        return {
            "text": "alpha",
            "metadata": {
                "source": "loader-source",
                "filename": "loader-name",
                "relative_path": "loader/path.txt",
                "tags": ["rag", "notes"],
                "none_value": None,
            },
        }

    def chunker(document):
        return [
            {
                "content": document["text"],
                "metadata": {
                    "source": "chunker-source",
                    "filename": "chunker-name",
                    "relative_path": "chunker/path.txt",
                    "chunk_index": 99,
                    "extra": {"level": 1},
                },
            }
        ]

    ingest_script.ingest(
        input_path=raw_file,
        collection="test_collection",
        persist_path=str(tmp_path / "chroma"),
        embedder=embedder,
        vector_store=vector_store,
        loader_func=loader,
        chunker_func=chunker,
    )

    metadata = embedder.received_chunks[0]["metadata"]
    assert metadata["source"] == "notes.txt"
    assert metadata["filename"] == "notes.txt"
    assert metadata["relative_path"] == "notes.txt"
    assert metadata["chunk_index"] == 0
    assert metadata["tags"] == "['rag', 'notes']"
    assert metadata["extra"] == "{'level': 1}"
    assert "none_value" not in metadata


def test_prepare_chunk_rejects_blank_object_chunk(tmp_path):
    raw_file = tmp_path / "notes.txt"
    raw_file.write_text("alpha", encoding="utf-8")

    def chunker(document):
        return [ObjectChunk(content="   ", source="notes.txt", chunk_id=0)]

    with pytest.raises(ValueError, match="chunk text must not be blank"):
        ingest_script.ingest(
            input_path=raw_file,
            collection="test_collection",
            persist_path=str(tmp_path / "chroma"),
            embedder=FakeEmbedder(),
            vector_store=FakeVectorStore(),
            chunker_func=chunker,
        )


def test_dry_run_ignores_embedder_vector_store_and_reset(tmp_path):
    raw_file = tmp_path / "notes.md"
    raw_file.write_text("dry run content", encoding="utf-8")

    summary = ingest_script.ingest(
        input_path=raw_file,
        collection="test_collection",
        persist_path=str(tmp_path / "chroma"),
        reset=True,
        dry_run=True,
        embedder=ExplodingComponent(),
        vector_store=ExplodingComponent(),
    )

    assert summary["dry_run"] is True
    assert summary["reset"] is True
    assert summary["files_indexed"] == 1
    assert summary["chunks_created"] == 1
    assert summary["chunks_stored"] == 0


def test_reset_runs_before_add_chunks(tmp_path):
    raw_file = tmp_path / "notes.txt"
    raw_file.write_text("reset content", encoding="utf-8")
    vector_store = FakeVectorStore()

    ingest_script.ingest(
        input_path=raw_file,
        collection="test_collection",
        persist_path=str(tmp_path / "chroma"),
        reset=True,
        embedder=FakeEmbedder(),
        vector_store=vector_store,
    )

    assert vector_store.reset_called is True
    assert vector_store.added_chunks is not None


def test_main_prints_summary_from_ingest(tmp_path, capsys):
    raw_file = tmp_path / "notes.txt"
    raw_file.write_text("cli content", encoding="utf-8")

    exit_code = ingest_script.main(
        [
            "--input",
            str(raw_file),
            "--collection",
            "test_collection",
            "--persist-path",
            str(tmp_path / "chroma"),
            "--dry-run",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Input path:" in captured.out
    assert "Dry run: yes" in captured.out
    assert "Chunks stored: 0" in captured.out


def test_main_prints_readable_errors(tmp_path, capsys):
    exit_code = ingest_script.main(
        [
            "--input",
            str(tmp_path / "missing.txt"),
            "--collection",
            "test_collection",
            "--persist-path",
            str(tmp_path / "chroma"),
            "--dry-run",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Error:" in captured.err
