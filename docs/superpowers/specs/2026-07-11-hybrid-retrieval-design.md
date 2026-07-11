# Hybrid Retrieval Design

## Goal

Add enterprise-style hybrid retrieval to mini-rag by combining BM25 sparse
retrieval and Chroma dense retrieval. Each route keeps its original score, is
normalized independently, and contributes to a weighted final ranking.

The change must preserve the current dictionary-based result format and the
public `RAGPipeline.ask()` interface. It must not change the embedding
implementation or Chroma vector-store internals.

## Chosen Approach

Use a compatibility-focused incremental integration:

- Keep retrieval records as dictionaries rather than introducing a new
  `Document` class.
- Extend dense retrieval results with a similarity score while retaining the
  original Chroma distance.
- Reuse the existing `app/bm25_retriever.py` module.
- Add a focused score-normalization utility and a `HybridRetriever` orchestration
  module.
- Assemble hybrid retrieval in `pipeline_factory.py` without changing API, CLI,
  prompt-building, or generation interfaces.

This approach keeps the change small and follows the repository's existing data
model. A dense adapter was rejected because it would add a thin extra layer, and
a full typed-result refactor was rejected because it would affect unrelated
modules and break backward compatibility.

## Retrieval Flow

```text
RAGPipeline.ask(query, top_k)
  -> HybridRetriever.retrieve(query, top_k)
       -> BM25Retriever.retrieve(query, top_k * 2)
       -> Retriever.retrieve(query, top_k * 2)
       -> normalize BM25 scores independently
       -> normalize dense scores independently
       -> merge matching chunks by stable identifier
       -> weighted score fusion
       -> deterministic descending ranking
       -> final top_k results
  -> prompt builder
  -> generator
```

The default candidate multiplier is `2`. For a requested final `top_k` of five,
each retrieval route may therefore contribute up to ten candidates before
fusion.

## Result Format

All retrievers continue to use dictionaries containing the current core fields:

```python
{
    "id": "chunk-id",
    "text": "chunk content",
    "metadata": {"source": "file.md", "chunk_id": 0},
}
```

Dense results retain `distance` and add `score = 1 - distance`. Hybrid results
include the following diagnostic fields where applicable:

```python
{
    "score": 0.82,
    "sparse_score": 8.2,
    "dense_score": 0.91,
    "normalized_sparse_score": 1.0,
    "normalized_dense_score": 0.7,
}
```

`score` on the hybrid result is the final fused score. A route that did not
retrieve a chunk contributes a normalized score of `0.0` for that chunk.

## Components

### Score normalizer

Add `app/utils/score_normalizer.py` with:

```python
def min_max_normalize(scores: list[float]) -> list[float]:
    ...
```

It returns an empty list for empty input. It applies
`(score - minimum) / (maximum - minimum)` normally. If all input scores are
equal, it returns `1.0` for every item to avoid division by zero and to preserve
the fact that all candidates are equally relevant within that route.

### BM25 retriever

Update the existing `app/bm25_retriever.py`; do not add a duplicate sparse
retriever module. Preserve its indexing and tokenization responsibilities and
its raw BM25 scores. Results use `id`, `text`, `metadata`, and `score`.

The stable identifier is resolved in this order:

1. Existing top-level `id`.
2. The pair `metadata.source` and `metadata.chunk_id`.
3. A deterministic fallback derived from source and text when legacy records do
   not provide a chunk ID.

The fallback exists for backward compatibility. Normal ingestion is expected to
provide chunk IDs.

### Dense retriever

Update `app/retriever.py` to copy each result returned by the vector store and
add `score = 1 - distance`. Preserve the original `distance` value. Do not alter
the vector-store result objects in place, and do not modify `app/vector_store.py`.

### Hybrid retriever

Add `app/hybrid_retriever.py` with a `HybridRetriever` that accepts injected
sparse and dense retrievers, weights, and a candidate multiplier. Its public
method remains:

```python
def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
    ...
```

The retriever:

1. Validates the query and `top_k` through clear input checks.
2. Requests `top_k * candidate_multiplier` results from each route.
3. Normalizes each route's scores independently.
4. Merges records by their stable chunk identifier.
5. Computes
   `sparse_weight * normalized_sparse + dense_weight * normalized_dense`.
6. Sorts by final score descending, with a stable identifier as a deterministic
   tie-breaker.
7. Returns at most `top_k` records.

Weights must be finite, non-negative numbers, and their sum must be greater than
zero. The class normalizes them internally, so values such as `4` and `6` behave
like `0.4` and `0.6`. The candidate multiplier must be a positive integer.

### Configuration

Add these defaults to `app/config.py`:

```python
HYBRID_SPARSE_WEIGHT = 0.5
HYBRID_DENSE_WEIGHT = 0.5
HYBRID_TOP_K = 5
HYBRID_CANDIDATE_MULTIPLIER = 2
```

These values centralize future tuning and avoid hard-coded retrieval behavior.

### Pipeline integration

Update `app/pipeline_factory.py` to load the stored chunk texts and metadata from
the existing Chroma collection, build `BM25Retriever`, create the current dense
`Retriever`, and inject a configured `HybridRetriever` into `RAGPipeline`.

The factory continues to validate that the vector database exists and is not
empty. `RAGPipeline.ask()`, API request/response models, and CLI invocation remain
unchanged. An explicitly supplied top-k value continues to control the final
number of contexts.

## Error Handling

Configuration and input errors raise clear `ValueError` or configuration errors.
Failures from BM25, embeddings, or Chroma propagate rather than silently
degrading to one retrieval route; silent fallback could conceal an unhealthy
production index. Empty result lists are valid and fuse to an empty list, which
preserves the existing `Not found in knowledge base.` behavior.

Malformed result records fail with a clear error instead of being merged under
an ambiguous identifier.

## Testing

Add `tests/test_score_normalizer.py` covering:

- `[10, 20, 30]` normalizing to `[0.0, 0.5, 1.0]`.
- Equal scores normalizing to all `1.0`.
- Empty, negative, and floating-point inputs.

Add `tests/test_hybrid_retriever.py` covering:

- Both routes receive `top_k * 2`.
- Independent normalization and weighted fusion.
- Correct merge behavior for overlapping and route-only chunks.
- Correct final order and top-k truncation.
- Empty routes, equal scores, invalid weights, and invalid candidate multiplier.

Update BM25 tests to verify stable IDs and preserved raw scores. Update dense
retriever tests to verify both `distance` and `score`, and to prove that source
results are not mutated. Update pipeline-factory and pipeline integration tests
to verify that the default pipeline uses hybrid retrieval while public API and
CLI behavior remain unchanged.

Tests use fakes for embeddings, Chroma, and generation. No external API is
required for unit tests. Verification runs focused tests first, followed by the
complete `pytest` suite. `python scripts/ask.py` is an optional local smoke test
when a populated vector store and valid environment configuration are present.

## Scope Boundaries

This work does not:

- Introduce a second document model.
- Change embedding behavior.
- Change Chroma storage or query internals.
- Add reranking, query rewriting, web retrieval, or other retrieval routes.
- Refactor unrelated modules.

## Success Criteria

The feature is complete when sparse and dense candidates are retrieved with a
two-times candidate pool, independently normalized, merged without duplicate
chunks, fused with configurable weights, and returned in deterministic final
rank order; the existing API and CLI continue to work; and all relevant and
full-project tests pass.
