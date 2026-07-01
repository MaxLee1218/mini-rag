from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.chunker import split_document
from app.embeddings import Embedder


def main() -> None:
    document = {
        "content": (
            "Retrieval-Augmented Generation retrieves relevant local context "
            "before generating an answer."
        ),
        "source": "manual_sample.md",
    }

    chunks = split_document(document, chunk_size=80, chunk_overlap=10)
    embedder = Embedder()
    embedded_chunks = embedder.embed_chunks(chunks)

    first_chunk = embedded_chunks[0]
    print(first_chunk.keys())
    print(len(first_chunk["embedding"]))


if __name__ == "__main__":
    main()
