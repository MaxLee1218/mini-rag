"""Opt-in real-model smoke check for the configured Cross-Encoder reranker."""

from __future__ import annotations

import logging

from app.config import (
    RERANKER_BATCH_SIZE,
    RERANKER_DEVICE,
    RERANKER_FAILURE_MODE,
    RERANKER_LOCAL_FILES_ONLY,
    RERANKER_MAX_LENGTH,
    RERANKER_MODEL,
)
from app.reranker import CrossEncoderReranker


logger = logging.getLogger(__name__)


def main() -> int:
    """Run an English-first real Cross-Encoder smoke check."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    query = "What is retrieval-augmented generation?"
    passages = [
        "Retrieval-augmented generation retrieves external documents before generating an answer.",
        "A convolutional neural network is commonly used for image recognition.",
        "A retrieval system searches a document collection for information relevant to a query.",
    ]
    documents = [
        {"id": index, "text": text, "original_index": index}
        for index, text in enumerate(passages, start=1)
    ]
    reranker = CrossEncoderReranker(
        RERANKER_MODEL,
        batch_size=RERANKER_BATCH_SIZE,
        max_length=RERANKER_MAX_LENGTH,
        device=RERANKER_DEVICE,
        failure_mode=RERANKER_FAILURE_MODE,
        local_files_only=RERANKER_LOCAL_FILES_ONLY,
    )
    results = reranker.rerank(query, documents)
    if not results or "rerank_score" not in results[0]:
        logger.error("Real reranker smoke check fell back; inspect the preceding logs.")
        return 1
    for reranked_index, result in enumerate(results, start=1):
        logger.info(
            "original=%d reranked=%d rerank_score=%.6f text=%s",
            result["original_index"],
            reranked_index,
            result["rerank_score"],
            result["text"][:100],
        )
    logger.info("TinyBERT is primarily trained for English retrieval.")
    logger.info("Chinese scores are for engineering verification only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
