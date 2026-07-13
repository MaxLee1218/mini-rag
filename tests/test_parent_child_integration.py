from __future__ import annotations

from app.parent_store import SQLiteParentStore
from app.pipeline import RAGPipeline
from app.retriever import Retriever
from app.vector_store import ChromaVectorStore
from scripts.ingest import ingest


class KeywordEmbedder:
    model_name = "offline-keyword"

    def embed_chunks(self, chunks):
        records = []
        for chunk in chunks:
            vector = [1.0, 0.0] if "父子块切分" in chunk["text"] else [0.0, 1.0]
            records.append(
                {
                    **chunk,
                    "embedding": vector,
                    "embedding_model": self.model_name,
                    "embedding_dimension": 2,
                }
            )
        return records

    def embed_query(self, query):
        return [1.0, 0.0]


class FakeGenerator:
    def generate(self, prompt):
        assert "向量检索、父子块切分和重排序" in prompt
        return "小 child 用于精准检索，大 parent 用于完整生成 [1]。"


def test_offline_parent_child_ingest_retrieve_and_pipeline(tmp_path):
    source_file = tmp_path / "data" / "example.txt"
    source_file.parent.mkdir()
    source_file.write_text(
        "第一部分讲述 Python 的基础语法、变量、函数和模块。\n"
        "第二部分讲述 FastAPI 的路由、请求模型、依赖注入和异常处理。\n"
        "第三部分讲述 RAG 中的向量检索、父子块切分和重排序。",
        encoding="utf-8",
    )
    vector_store = ChromaVectorStore(
        collection_name="parent_child_test",
        persist_path=str(tmp_path / "chroma"),
    )
    parent_store = SQLiteParentStore(tmp_path / "parents.sqlite3")
    kwargs = {
        "input_path": tmp_path,
        "collection": "parent_child_test",
        "persist_path": str(tmp_path / "chroma"),
        "extensions": ".txt",
        "chunk_mode": "parent-child",
        "parent_chunk_size": 42,
        "parent_chunk_overlap": 0,
        "child_chunk_size": 16,
        "child_chunk_overlap": 0,
        "embedder": KeywordEmbedder(),
        "vector_store": vector_store,
        "parent_store": parent_store,
    }

    first_summary = ingest(**kwargs)
    first_parent_count = parent_store.count()
    first_child_count = vector_store.count()
    second_summary = ingest(**kwargs)

    assert first_summary["stored_parent_chunks"] == first_parent_count
    assert first_summary["stored_child_chunks"] == first_child_count
    assert second_summary["stored_parent_chunks"] == first_parent_count
    assert second_summary["stored_child_chunks"] == first_child_count

    retriever = Retriever(
        KeywordEmbedder(),
        vector_store,
        mode="parent-child",
        parent_store=parent_store,
    )
    contexts = retriever.retrieve(
        "父子块切分有什么作用？",
        top_k=3,
        where={"source": "data/example.txt"},
    )

    assert contexts[0]["metadata"]["chunk_type"] == "parent"
    assert "向量检索、父子块切分和重排序" in contexts[0]["text"]
    assert contexts[0]["retrieval"]["matched_child_id"].startswith("child::")
    assert "父子块切分" in contexts[0]["retrieval"]["matched_child_text"]

    result = RAGPipeline(
        retriever=retriever,
        generator=FakeGenerator(),
        candidate_k=3,
        final_top_k=3,
    ).ask("父子块切分有什么作用？")
    assert result.contexts[0]["id"].startswith("parent::")
    assert result.sources == ["data/example.txt"]
    assert "data/example.txt" in result.answer

    parent_store.close()
    vector_store.close()
