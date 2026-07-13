from __future__ import annotations

from app.pipeline import RAGPipeline
from app.retriever import ParentChildRetriever


class ChildRetriever:
    def __init__(self):
        self.top_k = None

    def retrieve(self, query, top_k=None):
        self.top_k = top_k
        return [
            {
                "id": "child::rag",
                "text": "父子块切分",
                "metadata": {
                    "source": "data/example.txt",
                    "chunk_type": "child",
                    "parent_id": "parent::rag",
                },
                "score": 0.98,
            }
        ]


class ParentStore:
    def get_many(self, parent_ids):
        return [
            {
                "id": "parent::rag",
                "text": "第三部分讲述 RAG 中的向量检索、父子块切分和重排序。",
                "metadata": {
                    "source": "data/example.txt",
                    "chunk_type": "parent",
                    "parent_id": "parent::rag",
                },
            }
        ]


class FakeGenerator:
    def __init__(self):
        self.prompt = None

    def generate(self, prompt):
        self.prompt = prompt
        return "父子块使用小块检索、大块生成 [1]。"


def test_pipeline_prompt_context_and_sources_use_restored_parent():
    generator = FakeGenerator()
    child_retriever = ChildRetriever()
    pipeline = RAGPipeline(
        retriever=ParentChildRetriever(child_retriever, ParentStore()),
        generator=generator,
        candidate_k=10,
        final_top_k=1,
        expand_retrieval_candidates=False,
    )

    result = pipeline.ask("父子块切分有什么作用？")

    assert result.contexts[0]["id"] == "parent::rag"
    assert "向量检索、父子块切分和重排序" in result.contexts[0]["text"]
    assert "向量检索、父子块切分和重排序" in generator.prompt
    assert result.contexts[0]["retrieval"]["matched_child_id"] == "child::rag"
    assert result.sources == ["data/example.txt"]
    assert child_retriever.top_k == 1
