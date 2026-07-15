from types import SimpleNamespace

import pytest

from app.retriever import ParentChildRetriever
from evaluation.latency_tracer import PipelineTraceError, trace_pipeline_call


class FakeClock:
    def __init__(self, values):
        self._values = iter(values)

    def __call__(self):
        return next(self._values)

    @classmethod
    def expected_sequence(cls):
        milliseconds = (0, 10, 20, 40, 60, 70, 90, 100)
        return cls(value * 1_000_000 for value in milliseconds)


class FakeEmbedder:
    def __init__(self):
        self.calls = []

    def embed_query(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return [0.1]


class FakeDenseRetriever:
    retriever_kind = "dense"

    def __init__(self):
        self.embedder = FakeEmbedder()
        self.calls = []

    def retrieve(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        self.embedder.embed_query(args[0], purpose="query")
        return [{"content": "context", "source": "source.md"}]


class FakeGenerator:
    generator_kind = "fake"

    def __init__(self):
        self.calls = []

    def generate(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return "answer"


class FakeDensePipeline:
    def __init__(self):
        self.retriever = FakeDenseRetriever()
        self.generator = FakeGenerator()
        self.seen_retriever_kind = None
        self.seen_generator_kind = None

    def ask(self, question, top_k=None):
        self.seen_retriever_kind = self.retriever.retriever_kind
        self.seen_generator_kind = self.generator.generator_kind
        contexts = self.retriever.retrieve(question, top_k=top_k)
        answer = self.generator.generate("prompt", temperature=0)
        return SimpleNamespace(answer=answer, contexts=contexts)


class FakeSparseRetriever:
    def retrieve(self, *args, **kwargs):
        return []


class FakeHybridRetriever:
    def __init__(self):
        self.sparse_retriever = FakeSparseRetriever()
        self.dense_retriever = FakeDenseRetriever()

    def retrieve(self, *args, **kwargs):
        self.sparse_retriever.retrieve(*args, **kwargs)
        return self.dense_retriever.retrieve(*args, **kwargs)


class FakeHybridPipeline(FakeDensePipeline):
    def __init__(self):
        super().__init__()
        self.retriever = FakeHybridRetriever()

    def ask(self, question, top_k=None):
        contexts = self.retriever.retrieve(question, top_k=top_k)
        answer = self.generator.generate("prompt")
        return SimpleNamespace(answer=answer, contexts=contexts)


class FakeParentChildDenseRetriever(FakeDenseRetriever):
    def retrieve(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        self.embedder.embed_query(args[0], purpose="query")
        return [
            {
                "id": "child::1",
                "text": "matching child",
                "metadata": {
                    "source": "source.md",
                    "parent_id": "parent::1",
                },
                "score": 0.9,
            }
        ]


class FakeParentStore:
    def get_many(self, parent_ids):
        return [
            {
                "id": "parent::1",
                "text": "parent context",
                "metadata": {"source": "source.md"},
            }
        ]


class FakeParentChildPipeline(FakeHybridPipeline):
    def __init__(self):
        super().__init__()
        self.retriever = ParentChildRetriever(
            FakeParentChildDenseRetriever(),
            FakeParentStore(),
        )


class FakeCustomRetriever:
    def retrieve(self, *args, **kwargs):
        return []


class FakeCustomPipeline(FakeHybridPipeline):
    def __init__(self):
        super().__init__()
        self.retriever = FakeCustomRetriever()


class RaisingPipeline(FakeDensePipeline):
    def __init__(self):
        super().__init__()
        self.failure = RuntimeError("pipeline failed")

    def ask(self, question, top_k=None):
        self.retriever.retrieve(question, top_k=top_k)
        raise self.failure


def test_trace_measures_exclusive_retrieval_and_restores_components():
    pipeline = FakeDensePipeline()
    original_retriever = pipeline.retriever
    original_embedder = original_retriever.embedder
    original_generator = pipeline.generator

    result, latency, warnings = trace_pipeline_call(
        pipeline,
        "question",
        top_k=3,
        clock_ns=FakeClock.expected_sequence(),
    )

    assert result.answer == "answer"
    assert latency.embedding == 20.0
    assert latency.retrieval == 30.0
    assert latency.generation == 20.0
    assert latency.total == 100.0
    assert warnings == []
    assert pipeline.retriever is original_retriever
    assert original_retriever.embedder is original_embedder
    assert pipeline.generator is original_generator
    assert original_retriever.calls == [(("question",), {"top_k": 3})]
    assert original_embedder.calls == [(("question",), {"purpose": "query"})]
    assert original_generator.calls == [(("prompt",), {"temperature": 0})]
    assert pipeline.seen_retriever_kind == "dense"
    assert pipeline.seen_generator_kind == "fake"


def test_trace_restores_components_when_pipeline_raises():
    pipeline = RaisingPipeline()
    original_retriever = pipeline.retriever
    original_embedder = original_retriever.embedder
    original_generator = pipeline.generator

    with pytest.raises(PipelineTraceError) as error_info:
        trace_pipeline_call(pipeline, "question")

    assert pipeline.retriever is original_retriever
    assert original_retriever.embedder is original_embedder
    assert pipeline.generator is original_generator
    assert error_info.value.__cause__ is pipeline.failure
    assert error_info.value.original_exception is pipeline.failure
    assert error_info.value.latency.embedding is not None
    assert error_info.value.latency.retrieval is not None
    assert error_info.value.latency.generation is None
    assert error_info.value.latency.total is not None


def test_trace_discovers_hybrid_dense_embedder():
    pipeline = FakeHybridPipeline()
    original_embedder = pipeline.retriever.dense_retriever.embedder

    _, latency, warnings = trace_pipeline_call(pipeline, "question")

    assert latency.embedding is not None
    assert warnings == []
    assert pipeline.retriever.dense_retriever.embedder is original_embedder


def test_trace_discovers_and_restores_parent_child_embedder():
    pipeline = FakeParentChildPipeline()
    original_retriever = pipeline.retriever
    original_embedder = original_retriever.child_retriever.embedder

    _, latency, warnings = trace_pipeline_call(
        pipeline,
        "question",
        clock_ns=FakeClock.expected_sequence(),
    )

    assert latency.embedding == 20.0
    assert latency.retrieval == 30.0
    assert warnings == []
    assert pipeline.retriever is original_retriever
    assert original_retriever.child_retriever.embedder is original_embedder


def test_trace_marks_unknown_embedding_unavailable():
    _, latency, warnings = trace_pipeline_call(FakeCustomPipeline(), "question")

    assert latency.embedding is None
    assert warnings == ["embedding timer unavailable for custom retriever"]


def test_trace_omits_top_k_when_not_provided():
    class Pipeline(FakeDensePipeline):
        def __init__(self):
            super().__init__()
            self.ask_calls = []

        def ask(self, *args, **kwargs):
            self.ask_calls.append((args, kwargs))
            return super().ask(*args, **kwargs)

    pipeline = Pipeline()

    trace_pipeline_call(pipeline, "question")

    assert pipeline.ask_calls == [(("question",), {})]
