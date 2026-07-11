from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.bm25_retriever import BM25Retriever


def main() -> None:
    documents = [
        {"text": "RAG通过embedding进行语义检索", "metadata": {"source": "rag.md"}},
        {"text": "FastAPI用于构建Python API服务", "metadata": {"source": "fastapi.md"}},
        {"text": "BM25适合进行关键词检索", "metadata": {"source": "bm25.md"}},
    ]
    query = "embedding"
    results = BM25Retriever(documents).retrieve(query)

    print(f"Query:\n{query}\n")
    print("Top results:\n")
    for position, result in enumerate(results, start=1):
        print(f"{position}.")
        print(f"source:\n{result['metadata'].get('source', '')}\n")
        print(f"score:\n{result['score']}\n")
        print(f"text:\n{result['text']}\n")


if __name__ == "__main__":
    main()
