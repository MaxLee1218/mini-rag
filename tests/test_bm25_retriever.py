from app.bm25_retriever import BM25Retriever, tokenize


def sample_documents():
    return [
        {
            "id": "rag-0",
            "text": "RAG使用embedding检索",
            "metadata": {"source": "rag.md", "chunk_id": 0},
        },
        {
            "text": "数据库用于保存数据",
            "metadata": {"source": "database.md", "chunk_id": 1},
        },
        {"text": "Python是一种编程语言", "metadata": {"source": "python.md"}},
    ]


def test_bm25_initializes_index():
    retriever = BM25Retriever(sample_documents())
    assert retriever.bm25 is not None
    assert len(retriever.tokenized_documents) == 3


def test_retrieve_returns_keyword_match_first():
    results = BM25Retriever(sample_documents()).retrieve("Python")
    assert results[0]["metadata"]["source"] == "python.md"
    assert isinstance(results[0]["score"], float)
    assert results[0]["score"] > 0


def test_retrieve_resolves_stable_ids_without_changing_raw_scores():
    retriever = BM25Retriever(sample_documents())

    first = retriever.retrieve("RAG", top_k=3)
    second = retriever.retrieve("RAG", top_k=3)

    results_by_source = {result["metadata"]["source"]: result for result in first}
    assert results_by_source["rag.md"]["id"] == "rag-0"
    assert results_by_source["database.md"]["id"] == "database.md:1"
    assert results_by_source["python.md"]["id"] == next(
        result["id"]
        for result in second
        if result["metadata"]["source"] == "python.md"
    )
    raw_scores = retriever.bm25.get_scores(["rag"])
    expected_by_source = {
        document["metadata"]["source"]: float(raw_scores[index])
        for index, document in enumerate(retriever.documents)
    }
    assert all(
        result["score"] == expected_by_source[result["metadata"]["source"]]
        for result in first
    )


def test_retrieve_respects_top_k():
    results = BM25Retriever(sample_documents()).retrieve("Python", top_k=1)
    assert len(results) == 1


def test_retrieve_returns_empty_list_for_empty_query():
    retriever = BM25Retriever(sample_documents())
    assert retriever.retrieve("") == []
    assert retriever.retrieve(None) == []


def test_empty_document_collection_can_be_retrieved_safely():
    retriever = BM25Retriever([])
    assert retriever.bm25 is not None
    assert retriever.retrieve("Python") == []


def test_non_positive_top_k_returns_no_results():
    retriever = BM25Retriever(sample_documents())
    assert retriever.retrieve("Python", top_k=0) == []
    assert retriever.retrieve("Python", top_k=-1) == []


def test_chinese_tokenization_supports_keyword_retrieval():
    documents = [
        {"text": "苹果是一种水果", "metadata": {"source": "apple.md"}},
        {"text": "汽车需要燃料", "metadata": {"source": "car.md"}},
        {"text": "编程需要逻辑", "metadata": {"source": "code.md"}},
    ]
    results = BM25Retriever(documents).retrieve("苹果", top_k=1)
    assert tokenize("苹果")
    assert results[0]["metadata"]["source"] == "apple.md"
