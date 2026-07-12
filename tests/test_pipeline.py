import pytest

from app.pipeline import RAGPipeline, RAGResult


NOT_FOUND = "Not found in knowledge base."


class FakeRetriever:
    def __init__(self, contexts):
        self.contexts = contexts
        self.received_question = None
        self.received_top_k = None

    def retrieve(self, question, top_k=None):
        self.received_question = question
        self.received_top_k = top_k
        return self.contexts


class FakeGenerator:
    def __init__(self, answer, call_log=None):
        self.answer = answer
        self.call_log = call_log
        self.received_prompt = None

    def generate(self, prompt):
        if self.call_log is not None:
            self.call_log.append("generate")
        self.received_prompt = prompt
        return self.answer


class FailingGenerator:
    def generate(self, prompt):
        raise RuntimeError("boom")


class PromptBuilderObject:
    def __init__(self):
        self.received_question = None
        self.received_contexts = None

    def build(self, question, contexts):
        self.received_question = question
        self.received_contexts = contexts
        return "OBJECT PROMPT"


class FakeReranker:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = []

    def rerank(self, question, candidates, top_k=None):
        self.calls.append((question, candidates, top_k))
        if self.error:
            raise self.error
        return self.result if self.result is not None else list(reversed(candidates))[:top_k]


def test_ask_runs_retrieval_prompt_generation_and_returns_result():
    call_log = []
    contexts = [
        {
            "text": "Python 是一种编程语言。",
            "metadata": {"source": "python_intro.md"},
        }
    ]
    retriever = FakeRetriever(contexts)
    generator = FakeGenerator("Python 是一种编程语言 [1]。", call_log=call_log)

    def fake_prompt_builder(question, received_contexts):
        call_log.append("build_prompt")
        assert question == "Python 是什么？"
        assert received_contexts == contexts
        return "PROMPT"

    original_retrieve = retriever.retrieve

    def retrieve_with_log(question, top_k=None):
        call_log.append("retrieve")
        return original_retrieve(question, top_k=top_k)

    retriever.retrieve = retrieve_with_log
    pipeline = RAGPipeline(
        retriever=retriever,
        generator=generator,
        prompt_builder=fake_prompt_builder,
    )

    result = pipeline.ask("  Python 是什么？  ")

    assert call_log == ["retrieve", "build_prompt", "generate"]
    assert isinstance(result, RAGResult)
    assert result.question == "Python 是什么？"
    assert result.contexts == contexts
    assert result.sources == ["python_intro.md"]
    assert result.answer == (
        "Python 是一种编程语言 [1]。\n\n来源：\n- [1] python_intro.md"
    )
    assert generator.received_prompt == "PROMPT"
    assert result.prompt == "PROMPT"


def test_ask_passes_context_body_and_question_to_generator_prompt():
    contexts = [
        {
            "text": "max is my name. 我今年114514岁了。",
            "metadata": {"source": "private/private_notes.txt"},
        }
    ]
    retriever = FakeRetriever(contexts)
    generator = FakeGenerator("你今年114514岁。")
    pipeline = RAGPipeline(retriever=retriever, generator=generator)

    result = pipeline.ask("我今年几岁？")

    assert "max is my name" in generator.received_prompt
    assert "114514" in generator.received_prompt
    assert "我今年几岁？" in generator.received_prompt
    assert "114514" in result.answer
    assert result.prompt == generator.received_prompt


def test_sources_are_extracted_from_contexts_not_answer_text():
    contexts = [{"text": "真实上下文", "metadata": {"source": "real.md"}}]
    retriever = FakeRetriever(contexts)
    generator = FakeGenerator(
        "答案：\nRAG 是检索增强生成 [1]。\n\n来源：\n- [1] fake.md"
    )
    pipeline = RAGPipeline(retriever=retriever, generator=generator)

    result = pipeline.ask("什么是 RAG？")

    assert result.sources == ["real.md"]
    assert "real.md" in result.answer
    assert "fake.md" not in result.answer


def test_ask_passes_top_k_to_retriever():
    retriever = FakeRetriever([])
    generator = FakeGenerator(NOT_FOUND)
    pipeline = RAGPipeline(retriever=retriever, generator=generator)

    pipeline.ask("问题", top_k=2)

    assert retriever.received_top_k == 10


def test_pipeline_reranks_candidates_before_prompt_and_limits_results():
    contexts = [
        {"id": str(index), "text": f"text {index}", "metadata": {"source": f"{index}.md"}}
        for index in range(12)
    ]
    retriever = FakeRetriever(contexts)
    reranked = list(reversed(contexts[:10]))[:3]
    reranker = FakeReranker(result=reranked)
    prompt_builder = PromptBuilderObject()
    pipeline = RAGPipeline(
        retriever=retriever,
        generator=FakeGenerator("answer"),
        prompt_builder=prompt_builder,
        reranker=reranker,
        candidate_k=10,
        final_top_k=3,
    )

    result = pipeline.ask("question", top_k=None)

    assert retriever.received_top_k == 10
    assert reranker.calls == [("question", contexts[:10], 3)]
    assert prompt_builder.received_contexts == reranked
    assert result.contexts == reranked


def test_pipeline_without_reranker_keeps_fused_order_and_final_limit():
    contexts = [{"id": str(index), "text": str(index)} for index in range(8)]
    pipeline = RAGPipeline(
        retriever=FakeRetriever(contexts),
        generator=FakeGenerator(NOT_FOUND),
        candidate_k=6,
        final_top_k=2,
    )
    result = pipeline.ask("q", top_k=None)
    assert [item["id"] for item in result.contexts] == ["0", "1"]


def test_pipeline_reranker_runtime_failure_falls_back_and_generates():
    contexts = [{"id": str(index), "text": str(index)} for index in range(4)]
    pipeline = RAGPipeline(
        retriever=FakeRetriever(contexts),
        generator=FakeGenerator("answer"),
        reranker=FakeReranker(error=RuntimeError("boom")),
        candidate_k=4,
        final_top_k=2,
    )
    result = pipeline.ask("q", top_k=None)
    assert [item["id"] for item in result.contexts] == ["0", "1"]
    assert result.answer.startswith("answer")


@pytest.mark.parametrize("question", ["", "   ", None])
def test_ask_rejects_blank_question(question):
    pipeline = RAGPipeline(
        retriever=FakeRetriever([]),
        generator=FakeGenerator(NOT_FOUND),
    )

    with pytest.raises(ValueError, match="question must not be blank"):
        pipeline.ask(question)


@pytest.mark.parametrize("top_k", [0, -1, True, 1.5, "4"])
def test_ask_rejects_invalid_top_k(top_k):
    pipeline = RAGPipeline(
        retriever=FakeRetriever([]),
        generator=FakeGenerator(NOT_FOUND),
    )

    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        pipeline.ask("问题", top_k=top_k)


def test_ask_handles_empty_contexts_stably():
    retriever = FakeRetriever([])
    generator = FakeGenerator(NOT_FOUND)
    pipeline = RAGPipeline(retriever=retriever, generator=generator)

    result = pipeline.ask("没有资料的问题")

    assert result.question == "没有资料的问题"
    assert result.answer == NOT_FOUND
    assert result.contexts == []
    assert result.sources == []
    assert generator.received_prompt is not None


def test_retriever_none_result_becomes_empty_contexts():
    retriever = FakeRetriever(None)
    generator = FakeGenerator(NOT_FOUND)
    pipeline = RAGPipeline(retriever=retriever, generator=generator)

    result = pipeline.ask("问题")

    assert result.contexts == []
    assert result.sources == []


def test_retriever_tuple_result_becomes_list_without_mutating_original():
    original_contexts = (
        {"text": "A", "metadata": {"source": "a.md"}},
        {"text": "B", "metadata": {"source": "b.md"}},
    )
    retriever = FakeRetriever(original_contexts)
    generator = FakeGenerator("A [1]。")
    pipeline = RAGPipeline(retriever=retriever, generator=generator)

    result = pipeline.ask("问题")

    assert result.contexts == list(original_contexts)
    assert isinstance(result.contexts, list)
    assert original_contexts[0]["text"] == "A"


def test_generator_exception_is_not_swallowed():
    pipeline = RAGPipeline(
        retriever=FakeRetriever([{"text": "A", "metadata": {"source": "a.md"}}]),
        generator=FailingGenerator(),
    )

    with pytest.raises(RuntimeError, match="boom"):
        pipeline.ask("问题")


def test_pipeline_accepts_callable_generator():
    contexts = [{"text": "A", "metadata": {"source": "a.md"}}]

    def callable_generator(prompt):
        return "A [1]。"

    pipeline = RAGPipeline(
        retriever=FakeRetriever(contexts),
        generator=callable_generator,
    )

    result = pipeline.ask("问题")

    assert result.answer.endswith("来源：\n- [1] a.md")
    assert result.sources == ["a.md"]


def test_pipeline_accepts_prompt_builder_object():
    contexts = [{"text": "A", "metadata": {"source": "a.md"}}]
    prompt_builder = PromptBuilderObject()
    generator = FakeGenerator("A [1]。")
    pipeline = RAGPipeline(
        retriever=FakeRetriever(contexts),
        generator=generator,
        prompt_builder=prompt_builder,
    )

    result = pipeline.ask("问题")

    assert prompt_builder.received_question == "问题"
    assert prompt_builder.received_contexts == contexts
    assert generator.received_prompt == "OBJECT PROMPT"
    assert result.sources == ["a.md"]
