import pytest

from app.retriever import Retriever


class FakeEmbedder:
    def __init__(self):
        self.received_query = None

    def embed_query(self, query):
        self.received_query = query
        return [1.0, 0.0, 0.0]


class FakeVectorStore:
    def __init__(self, results=None):
        self.results = results if results is not None else make_results()
        self.received_query_embedding = None
        self.received_top_k = None
        self.received_where = None

    def query(self, query_embedding, top_k=5, where=None):
        self.received_query_embedding = query_embedding
        self.received_top_k = top_k
        self.received_where = where
        return self.results


class VectorStoreError(Exception):
    pass


class FailingVectorStore:
    def query(self, query_embedding, top_k=5, where=None):
        raise VectorStoreError("vector store failed")


def make_results():
    return [
        {
            "id": "chunk-a",
            "text": "first retrieved chunk",
            "metadata": {"source": "alpha.md"},
            "distance": 0.1,
        },
        {
            "id": "chunk-b",
            "text": "second retrieved chunk",
            "metadata": {"source": "beta.md"},
            "distance": 0.2,
        },
    ]


def make_retriever(results=None, default_top_k=5):
    embedder = FakeEmbedder()
    vector_store = FakeVectorStore(results)
    retriever = Retriever(
        embedder=embedder,
        vector_store=vector_store,
        default_top_k=default_top_k,
    )
    return retriever, embedder, vector_store


def test_constructor_uses_injected_dependencies():
    embedder = FakeEmbedder()
    vector_store = FakeVectorStore()

    retriever = Retriever(embedder=embedder, vector_store=vector_store)

    assert retriever.embedder is embedder
    assert retriever.vector_store is vector_store


@pytest.mark.parametrize("default_top_k", [0, -1, True])
def test_constructor_rejects_invalid_default_top_k(default_top_k):
    with pytest.raises(ValueError, match="default_top_k must be a positive integer"):
        Retriever(
            embedder=FakeEmbedder(),
            vector_store=FakeVectorStore(),
            default_top_k=default_top_k,
        )


def test_retrieve_calls_embedder_with_cleaned_query():
    retriever, embedder, _ = make_retriever()

    retriever.retrieve("  What is RAG?  ")

    assert embedder.received_query == "What is RAG?"


def test_retrieve_passes_query_embedding_top_k_and_where_to_vector_store():
    retriever, _, vector_store = make_retriever()
    where = {"source": "alpha.md"}

    retriever.retrieve("What is RAG?", top_k=2, where=where)

    assert vector_store.received_query_embedding == [1.0, 0.0, 0.0]
    assert vector_store.received_top_k == 2
    assert vector_store.received_where is where


def test_retrieve_uses_default_top_k_when_top_k_is_not_provided():
    retriever, _, vector_store = make_retriever(default_top_k=3)

    retriever.retrieve("What is RAG?")

    assert vector_store.received_top_k == 3


def test_retrieve_returns_exact_vector_store_result_object_unchanged():
    results = make_results()
    retriever, _, _ = make_retriever(results=results)

    retrieved = retriever.retrieve("What is RAG?")

    assert retrieved is results
    assert retrieved == make_results()


def test_retrieve_preserves_vector_store_error():
    retriever = Retriever(
        embedder=FakeEmbedder(),
        vector_store=FailingVectorStore(),
    )

    with pytest.raises(VectorStoreError, match="vector store failed"):
        retriever.retrieve("What is RAG?")


@pytest.mark.parametrize("query", ["", "   "])
def test_retrieve_rejects_blank_query(query):
    retriever, _, _ = make_retriever()

    with pytest.raises(ValueError, match="query must not be blank"):
        retriever.retrieve(query)


def test_retrieve_rejects_non_string_query():
    retriever, _, _ = make_retriever()

    with pytest.raises(ValueError, match="query must be a string"):
        retriever.retrieve(123)


@pytest.mark.parametrize("top_k", [0, -1, True, 1.5, "2"])
def test_retrieve_rejects_invalid_top_k(top_k):
    retriever, _, _ = make_retriever()

    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        retriever.retrieve("What is RAG?", top_k=top_k)


def test_retrieve_texts_returns_text_fields_in_order():
    retriever, _, _ = make_retriever()

    texts = retriever.retrieve_texts("What is RAG?")

    assert texts == ["first retrieved chunk", "second retrieved chunk"]


@pytest.mark.parametrize(
    "results",
    [
        [{"id": "chunk-a", "metadata": {}, "distance": 0.1}],
        [{"id": "chunk-a", "text": "  ", "metadata": {}, "distance": 0.1}],
    ],
)
def test_retrieve_texts_rejects_missing_or_blank_text(results):
    retriever, _, _ = make_retriever(results=results)

    with pytest.raises(ValueError, match="result text must not be blank"):
        retriever.retrieve_texts("What is RAG?")


def test_build_context_returns_empty_string_for_empty_results():
    retriever, _, _ = make_retriever()

    assert retriever.build_context([]) == ""


def test_build_context_includes_chunk_id_source_and_text():
    retriever, _, _ = make_retriever()

    context = retriever.build_context([make_results()[0]])

    assert context == "[chunk_id: chunk-a | source: alpha.md]\nfirst retrieved chunk"


def test_build_context_uses_unknown_for_missing_id():
    retriever, _, _ = make_retriever()

    context = retriever.build_context(
        [{"text": "retrieved chunk", "metadata": {"source": "alpha.md"}}]
    )

    assert context == "[chunk_id: unknown | source: alpha.md]\nretrieved chunk"


def test_build_context_omits_source_when_metadata_or_source_is_missing():
    retriever, _, _ = make_retriever()

    context = retriever.build_context(
        [
            {"id": "chunk-a", "text": "first retrieved chunk"},
            {"id": "chunk-b", "text": "second retrieved chunk", "metadata": {}},
        ],
        separator="\n---\n",
    )

    assert context == (
        "[chunk_id: chunk-a]\nfirst retrieved chunk"
        "\n---\n"
        "[chunk_id: chunk-b]\nsecond retrieved chunk"
    )


@pytest.mark.parametrize(
    "results",
    [
        [{"id": "chunk-a", "metadata": {}}],
        [{"id": "chunk-a", "text": "  ", "metadata": {}}],
    ],
)
def test_build_context_rejects_missing_or_blank_text(results):
    retriever, _, _ = make_retriever()

    with pytest.raises(ValueError, match="result text must not be blank"):
        retriever.build_context(results)


def test_build_context_rejects_non_list_results():
    retriever, _, _ = make_retriever()

    with pytest.raises(ValueError, match="results must be a list"):
        retriever.build_context({"text": "not a list"})


def test_build_context_rejects_non_string_separator():
    retriever, _, _ = make_retriever()

    with pytest.raises(ValueError, match="separator must be a string"):
        retriever.build_context([], separator=123)


def test_retrieve_context_returns_formatted_context():
    retriever, _, _ = make_retriever()

    context = retriever.retrieve_context("What is RAG?", top_k=1)

    assert "[chunk_id: chunk-a | source: alpha.md]" in context
    assert "first retrieved chunk" in context
