from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.chunker import split_document
from app.embeddings import Embedder
from app.vector_store import ChromaVectorStore


def main() -> None:
    document = {
        "content": (
            "Retrieval-Augmented Generation retrieves relevant context before "
            "answering a question. The chunker splits source documents into "
            "small pieces. The embedder converts each chunk into a vector. "
            "The vector store keeps those precomputed vectors and returns "
            "similar chunks for a query embedding."
        ),
        "source": "manual_vector_store_sample.md",
    }

    chunks = split_document(document, chunk_size=90, chunk_overlap=10)
    embedder = Embedder()
    embedded_chunks = embedder.embed_chunks(chunks)
    store = ChromaVectorStore(collection_name="manual_vector_store_test")

    try:
        store.reset()
        store.add_chunks(embedded_chunks)

        query_embedding = embedder.embed_query("How does the vector store help RAG?")
        results = store.query(query_embedding, top_k=2)

        print(f"Stored chunks: {store.count()}")
        print(f"Query results: {len(results)}")

        if results:
            print(f"First result keys: {results[0].keys()}")
            print(f"First result text: {results[0]['text']}")
        else:
            print("No query results returned.")
    finally:
        store.close()


if __name__ == "__main__":
    main()
