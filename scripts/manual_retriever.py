from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.chunker import split_document
from app.embeddings import Embedder
from app.retriever import Retriever
from app.vector_store import ChromaVectorStore


def main() -> None:
    document = {
        "content": (
            "Retrieval-Augmented Generation, or RAG, uses retrieval to find "
            "relevant context before generating an answer. Embeddings convert "
            "text into numeric vectors so related ideas can be compared. "
            "Vector search uses those vectors to find chunks that are close "
            "to a user query."
        ),
        "source": "manual_retriever_sample.md",
    }

    chunks = split_document(document, chunk_size=100, chunk_overlap=10)
    embedder = Embedder()
    vector_store = ChromaVectorStore(collection_name="manual_retriever_test")

    try:
        vector_store.reset()
        vector_store.add_chunks(embedder.embed_chunks(chunks))

        retriever = Retriever(embedder=embedder, vector_store=vector_store)
        results = retriever.retrieve("What does RAG use retrieval for?", top_k=2)
        context = retriever.build_context(results)

        print(f"Stored chunks: {vector_store.count()}")
        print(f"Retrieved results: {len(results)}")

        if results:
            print(f"First result keys: {results[0].keys()}")
            print(f"First result text: {results[0]['text']}")
            print("Context:")
            print(context)
        else:
            print("No retrieved results returned.")
    finally:
        vector_store.close()


if __name__ == "__main__":
    main()
