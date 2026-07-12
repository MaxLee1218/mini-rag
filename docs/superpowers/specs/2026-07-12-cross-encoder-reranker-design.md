# Cross-Encoder Reranker Design

## Scope

Add a configurable, optional second-stage Cross-Encoder reranker to the existing modular RAG pipeline. The reranker operates only on already fused sparse and dense retrieval candidates. It does not replace first-stage retrieval, alter persisted indexes, or change stored metadata contracts.

The default model is `cross-encoder/ms-marco-TinyBERT-L2-v2`. It is selected for a lightweight CPU-capable engineering loop and English retrieval use. The design makes no unmeasured claim about Chinese ranking quality and permits model replacement through configuration alone.

## Existing Contracts

`HybridRetriever.retrieve(query, top_k)` returns ranked dictionaries. Current results contain `id`, `text`, `metadata`, and fused `score`, with route-specific fields such as `sparse_score`, `dense_score`, and normalized component scores when available. The fused `score` is preserved unchanged.

`app.prompt_builder.extract_context_text()` is the shared text adapter. It supports strings, mappings with `text`, `content`, `page_content`, `document`, or `chunk`, and objects with supported text attributes. The reranker will reuse it instead of introducing a second extraction contract.

`RAGPipeline` currently owns retrieve → prompt → generate orchestration. `build_default_pipeline()` constructs the shared pipeline used by FastAPI and `scripts/ask.py`; it is the correct composition point for a reusable reranker instance. The low-level `scripts/query.py` remains a retrieval diagnostic rather than an answer-generation pipeline entrypoint.

## Architecture

Create `app/reranker.py` with a generic `CrossEncoderReranker`. Its constructor accepts model name/path, batch size, maximum tokenizer length, requested device, local-files-only behavior, failure mode, and an injectable model loader. Construction validates configuration but does not import or load the model eagerly.

The first `rerank()` call containing at least one nonblank document body resolves the device and loads one `sentence_transformers.CrossEncoder`. Later calls reuse the same instance. Tests inject a fake loader/model, so pytest never downloads a model or requires accelerator hardware.

`RAGPipeline` receives an optional reranker and two independent limits:

- `candidate_k`: number of fused candidates requested from the retriever and offered to the reranker.
- `final_top_k`: number of contexts passed to the prompt builder and generator.

The default pipeline flow is:

```text
sparse retrieval + dense retrieval
  -> hybrid normalization, fusion, deduplication, and top 10 candidates
  -> optional Cross-Encoder batch reranking
  -> top 5 contexts
  -> prompt construction
  -> generation and source normalization
```

When reranking is disabled, the first five fused candidates are used without changing their order. The existing public `RAGPipeline.ask(question, top_k=...)` interface remains valid: an explicit call-level `top_k` continues to control the final context count, while configured `candidate_k` remains the maximum first-stage candidate count and is raised to at least the requested final count when necessary.

## Reranking Behavior

`rerank(query, documents, top_k=None)` validates the query, document sequence, and optional positive `top_k`. Empty documents return immediately without model loading.

Each candidate is safely copied. Mapping results use a shallow top-level copy plus a copied nested `metadata` mapping. Dataclass values use `dataclasses.replace()` when they expose a `rerank_score` field. Other copyable objects use `copy.copy()` and must support assignment of `rerank_score`; clearly unsupported caller types raise a direct type error rather than enter model fallback.

The shared text extractor converts supported body values to stripped strings. Blank bodies are excluded from inference, receive Python `float("-inf")`, and remain after all valid bodies. If every body is blank, the model is not loaded and candidates retain input order subject to `top_k`.

All valid `(query, document_text)` pairs are passed to one `model.predict()` call using the configured batch size, `show_progress_bar=False`, and `convert_to_numpy=True`. Scores are flattened from supported list, NumPy, or Torch-like outputs and converted individually to Python `float`. A scalar is valid only for one effective document. A score-count mismatch is a runtime reranker failure.

Sorting uses raw Cross-Encoder logits descending. Python's stable sort preserves original order for ties. NaN is converted to negative infinity with a warning; positive and negative infinity remain deterministic Python floats. The new `rerank_score` never replaces `score` or route-specific retrieval scores.

## Devices and Loading

Supported requested devices are `auto`, `cpu`, `cuda`, and `mps`. `auto` selects CUDA, then MPS, then CPU. Device detection guards missing `torch.backends` or `torch.backends.mps`. Explicit unavailable CUDA or MPS requests log a warning and fall back to CPU.

The default model loader imports `torch` and `sentence_transformers.CrossEncoder` only when inference is first needed. It passes model name/path, resolved device, `max_length`, and `local_files_only`. A Hugging Face model name may download on first use; a local path works with `local_files_only=true`.

## Failure Handling

Only model device resolution, loading, output conversion, and inference failures enter fallback. The reranker logs the exception with the message `reranker failed; falling back to original retrieval order`, does not add fabricated scores, preserves candidate order and original fields, and applies the final `top_k` limit.

Invalid query, invalid `top_k`, invalid constructor configuration, invalid document collection type, and unsupported result-copy types are caller/configuration errors and propagate directly.

The Pipeline also guards an injected reranker's unexpected runtime exception so a third-party compatible reranker cannot stop answer generation. It logs fallback and uses the original candidate order. This boundary does not catch input validation errors raised before retrieval.

## Configuration

`app/config.py` adds validated defaults and environment parsing for:

```env
RERANKER_ENABLED=true
RERANKER_MODEL=cross-encoder/ms-marco-TinyBERT-L2-v2
RERANKER_TOP_K=5
RERANKER_CANDIDATE_K=10
RERANKER_BATCH_SIZE=16
RERANKER_MAX_LENGTH=256
RERANKER_DEVICE=cpu
RERANKER_FAILURE_MODE=fallback
RERANKER_LOCAL_FILES_ONLY=false
```

Positive integers reject booleans and values below one. Boolean parsing accepts only explicit conventional true/false spellings. Device and failure-mode values are normalized and checked against their fixed sets. `top_k` may exceed `candidate_k`; actual output is bounded by available candidates.

`build_default_pipeline()` constructs one `CrossEncoderReranker` only when enabled and injects it into `RAGPipeline`. Model names do not appear in pipeline, API, CLI, or smoke business logic.

## Observability and Privacy

Initialization logs bounded configuration facts: model identifier/path, requested and resolved device, local-only mode, and numeric limits. First load success is logged once. Each rerank records candidate count, valid-text count, returned count, inference duration, and fallback reason without logging full queries or document bodies.

## Tests and Verification

Tests are written before production behavior and use fake models/loaders. Reranker tests cover validation, one-call batching, extraction, stable sorting, copies, score conversion, blank bodies, NaN/infinity, lazy reuse, device selection, loading arguments, output mismatch, and load/inference fallback.

Pipeline tests cover candidate/final limits, reranker ordering, disabled behavior, runtime fallback, prompt inputs, and replaceable reranker interfaces. Factory/config tests cover defaults, environment validation, shared API/CLI construction, local paths, and construction without model loading.

Documentation and dependency updates cover `requirements.txt`, `.env.example`, `.gitignore`, README, and an opt-in English-first smoke script if it fits the existing manual smoke pattern. Verification runs focused tests during TDD, repository formatting/lint commands where configured, `python -m pytest -q`, `git diff --check`, and full diff inspection.

## Compatibility and Limitations

No persisted vector, identifier, chunk, collection, embedding, or metadata meaning changes. Existing retrieval fields remain intact. Reranking adds query latency and may initiate a model download on first use. TinyBERT is primarily trained on English MS MARCO data; Chinese or multilingual production use requires an independent evaluation set and may require changing `RERANKER_MODEL`.

The implementation preserves enough information for later baseline versus model comparisons: reranking can be disabled, original scores remain available, successful outputs contain raw `rerank_score`, model selection is configurable, and rerank latency is observable. It does not invent evaluation measurements or add an evaluation subsystem in this change.
