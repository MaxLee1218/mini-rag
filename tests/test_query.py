import pytest

from scripts import query as query_script


def make_results():
    return [
        {
            "id": "rag-a1b2c3d4-chunk-0",
            "text": "RAG retrieves relevant context before answering.",
            "metadata": {
                "source": "data/raw/sample/rag_notes.txt",
                "filename": "rag_notes.txt",
                "relative_path": "sample/rag_notes.txt",
            },
            "distance": 0.123456,
        },
        {
            "id": "embedding-e5f6g7h8-chunk-0",
            "text": "Embeddings convert text into vectors.",
            "metadata": {
                "source": "data/raw/sample/embedding_notes.txt",
                "filename": "embedding_notes.txt",
                "relative_path": "sample/embedding_notes.txt",
            },
            "distance": 0.5,
        },
    ]


class FakeRetriever:
    def __init__(self, results=None):
        self.results = make_results() if results is None else results
        self.received_query = None
        self.received_top_k = None
        self.received_where = None
        self.received_context_results = None

    def retrieve(self, query, top_k=None, where=None):
        self.received_query = query
        self.received_top_k = top_k
        self.received_where = where
        return self.results

    def build_context(self, results):
        self.received_context_results = results
        return "formatted context"


class FailingContextRetriever(FakeRetriever):
    def build_context(self, results):
        raise ValueError("blank result text")


class NoContextRetriever(FakeRetriever):
    build_context = None


class ExplodingFactory:
    def __init__(self, *args, **kwargs):
        raise AssertionError("real component should not be instantiated")


def test_parse_where_filter_returns_none_for_none():
    assert query_script.parse_where_filter(None) is None


def test_parse_where_filter_parses_valid_json_object():
    assert query_script.parse_where_filter('{ "filename": "rag_notes.txt" }') == {
        "filename": "rag_notes.txt"
    }


def test_parse_where_filter_accepts_common_powershell_unquoted_object():
    assert query_script.parse_where_filter("{ filename: rag_notes.txt }") == {
        "filename": "rag_notes.txt"
    }


def test_parse_where_filter_rejects_invalid_json():
    with pytest.raises(ValueError, match="invalid JSON"):
        query_script.parse_where_filter("{not json}")


def test_parse_where_filter_rejects_non_object_json():
    with pytest.raises(ValueError, match="must be a JSON object"):
        query_script.parse_where_filter('["rag_notes.txt"]')


def test_validate_query_strips_query_for_display_and_retrieval():
    retriever = FakeRetriever(results=[])

    output = query_script.run_query(
        query="  What is RAG?  ",
        collection="local_test_docs",
        persist_path="data/chroma",
        top_k=3,
        retriever=retriever,
    )

    assert retriever.received_query == "What is RAG?"
    assert output.startswith("Query: What is RAG?\n")


@pytest.mark.parametrize("query", ["", "   ", None])
def test_run_query_rejects_blank_query(query):
    with pytest.raises(ValueError, match="query must not be blank"):
        query_script.run_query(
            query=query,
            collection="local_test_docs",
            persist_path="data/chroma",
            top_k=3,
            retriever=FakeRetriever(),
        )


@pytest.mark.parametrize("collection", ["", "   ", None])
def test_run_query_rejects_blank_collection(collection):
    with pytest.raises(ValueError, match="collection must not be blank"):
        query_script.run_query(
            query="What is RAG?",
            collection=collection,
            persist_path="data/chroma",
            top_k=3,
            retriever=FakeRetriever(),
        )


@pytest.mark.parametrize("persist_path", ["", "   ", None])
def test_run_query_rejects_blank_persist_path(persist_path):
    with pytest.raises(ValueError, match="persist_path must not be blank"):
        query_script.run_query(
            query="What is RAG?",
            collection="local_test_docs",
            persist_path=persist_path,
            top_k=3,
            retriever=FakeRetriever(),
        )


@pytest.mark.parametrize("top_k", [0, -1, True, 1.5, "3"])
def test_run_query_rejects_invalid_top_k(top_k):
    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        query_script.run_query(
            query="What is RAG?",
            collection="local_test_docs",
            persist_path="data/chroma",
            top_k=top_k,
            retriever=FakeRetriever(),
        )


@pytest.mark.parametrize("max_chars", [0, -1, True, 1.5, "10"])
def test_run_query_rejects_invalid_max_chars(max_chars):
    with pytest.raises(ValueError, match="max_chars must be a positive integer"):
        query_script.run_query(
            query="What is RAG?",
            collection="local_test_docs",
            persist_path="data/chroma",
            top_k=3,
            max_chars=max_chars,
            retriever=FakeRetriever(),
        )


def test_truncate_text_keeps_short_and_boundary_text_unchanged():
    assert query_script.truncate_text("short", 10) == "short"
    assert query_script.truncate_text("exact", 5) == "exact"


def test_truncate_text_truncates_long_text_and_appends_ellipsis():
    assert query_script.truncate_text("alpha beta gamma", 10) == "alpha beta..."


def test_extract_source_info_returns_stable_shape_with_unknowns():
    assert query_script.extract_source_info({"metadata": {"source": "alpha.md"}}) == {
        "source": "alpha.md",
        "filename": "unknown",
        "relative_path": "unknown",
    }
    assert query_script.extract_source_info({"metadata": None}) == {
        "source": "unknown",
        "filename": "unknown",
        "relative_path": "unknown",
    }


def test_format_result_includes_required_fields_with_one_based_index():
    formatted = query_script.format_result(make_results()[0], index=1, max_chars=1000)

    assert "[1]" in formatted
    assert "Source: data/raw/sample/rag_notes.txt" in formatted
    assert "Filename: rag_notes.txt" in formatted
    assert "Relative path: sample/rag_notes.txt" in formatted
    assert "Chunk ID: rag-a1b2c3d4-chunk-0" in formatted
    assert "Distance: 0.1235" in formatted
    assert "Text:\nRAG retrieves relevant context before answering." in formatted


def test_format_result_handles_unknown_metadata_distance_and_empty_text():
    formatted = query_script.format_result(
        {"id": None, "text": "   ", "metadata": "not metadata", "distance": "far"},
        index=2,
        max_chars=1000,
    )

    assert "[2]" in formatted
    assert "Source: unknown" in formatted
    assert "Filename: unknown" in formatted
    assert "Relative path: unknown" in formatted
    assert "Chunk ID: unknown" in formatted
    assert "Distance: unknown" in formatted
    assert "Text:\n[empty text]" in formatted


def test_format_result_handles_missing_text():
    formatted = query_script.format_result(
        {"id": "chunk-a", "metadata": {}},
        index=1,
        max_chars=1000,
    )

    assert "Text:\n[empty text]" in formatted


def test_format_sources_preserves_order_deduplicates_and_uses_fallbacks():
    results = [
        {"metadata": {"source": "alpha.md"}},
        {"metadata": {"source": "alpha.md"}},
        {"metadata": {"relative_path": "beta.md"}},
        {"metadata": {"filename": "gamma.md"}},
        {"metadata": {}},
        {"metadata": None},
    ]

    assert query_script.format_sources(results) == (
        "Sources:\n"
        "- alpha.md\n"
        "- beta.md\n"
        "- gamma.md\n"
        "- unknown"
    )


def test_run_query_calls_retriever_and_preserves_result_order():
    retriever = FakeRetriever(results=make_results())
    where = {"filename": "rag_notes.txt"}

    output = query_script.run_query(
        query="What does RAG retrieve?",
        collection="local_test_docs",
        persist_path="data/chroma",
        top_k=3,
        where=where,
        retriever=retriever,
    )

    assert retriever.received_query == "What does RAG retrieve?"
    assert retriever.received_top_k == 3
    assert retriever.received_where is where
    assert "Query: What does RAG retrieve?" in output
    assert "Collection: local_test_docs" in output
    assert "Top k: 3" in output
    assert "Retrieved results: 2" in output
    assert output.index("[1]") < output.index("[2]")
    assert output.index("rag_notes.txt") < output.index("embedding_notes.txt")


def test_run_query_with_injected_retriever_does_not_create_real_components(monkeypatch):
    monkeypatch.setattr(query_script, "Embedder", ExplodingFactory)
    monkeypatch.setattr(query_script, "ChromaVectorStore", ExplodingFactory)
    monkeypatch.setattr(query_script, "Retriever", ExplodingFactory)

    output = query_script.run_query(
        query="What is RAG?",
        collection="local_test_docs",
        persist_path="data/chroma",
        top_k=3,
        retriever=FakeRetriever(results=[]),
    )

    assert "Retrieved results: 0" in output


def test_run_query_empty_results_includes_hints_and_no_sources_section():
    output = query_script.run_query(
        query="What is RAG?",
        collection="local_test_docs",
        persist_path="data/chroma",
        top_k=3,
        retriever=FakeRetriever(results=[]),
    )

    assert "Retrieved results: 0" in output
    assert "No retrieved results returned." in output
    assert "- Make sure you have run scripts/ingest.py first." in output
    assert "- Make sure you are querying the same collection name used during ingestion." in output
    assert "- Try increasing --top-k." in output
    assert "Sources:" not in output


def test_run_query_show_context_uses_retriever_build_context():
    retriever = FakeRetriever(results=make_results())

    output = query_script.run_query(
        query="What is RAG?",
        collection="local_test_docs",
        persist_path="data/chroma",
        top_k=3,
        show_context=True,
        retriever=retriever,
    )

    assert retriever.received_context_results == make_results()
    assert "Context:\nformatted context" in output
    assert "Sources:" in output


def test_run_query_show_context_handles_context_errors():
    output = query_script.run_query(
        query="What is RAG?",
        collection="local_test_docs",
        persist_path="data/chroma",
        top_k=3,
        show_context=True,
        retriever=FailingContextRetriever(results=make_results()),
    )

    assert "Context:\n[context unavailable: blank result text]" in output
    assert "[1]" in output


def test_run_query_show_context_handles_missing_context_builder():
    output = query_script.run_query(
        query="What is RAG?",
        collection="local_test_docs",
        persist_path="data/chroma",
        top_k=3,
        show_context=True,
        retriever=NoContextRetriever(results=make_results()),
    )

    assert "Context:\n[context unavailable: retriever has no build_context method]" in output


def test_run_query_no_sources_summary_still_shows_individual_sources():
    output = query_script.run_query(
        query="What is RAG?",
        collection="local_test_docs",
        persist_path="data/chroma",
        top_k=3,
        include_sources_summary=False,
        retriever=FakeRetriever(results=make_results()),
    )

    assert "Source: data/raw/sample/rag_notes.txt" in output
    assert "\nSources:\n" not in output


def test_main_prints_output_and_returns_zero(capsys):
    exit_code = query_script.main(
        [
            "--query",
            "What is RAG?",
            "--collection",
            "local_test_docs",
            "--persist-path",
            "data/chroma",
            "--top-k",
            "3",
            "--where",
            '{ "filename": "rag_notes.txt" }',
        ],
        retriever=FakeRetriever(results=[]),
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Query: What is RAG?" in captured.out
    assert "Retrieved results: 0" in captured.out


def test_main_prints_readable_error_and_returns_one(capsys):
    exit_code = query_script.main(
        [
            "--query",
            "   ",
            "--collection",
            "local_test_docs",
            "--persist-path",
            "data/chroma",
        ],
        retriever=FakeRetriever(results=[]),
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Error: query must not be blank" in captured.err
