# Enterprise RAG Engine Specification

This document defines the behavioral contracts, architectural boundaries, quality requirements, and staged evolution of this RAG system. Follow it together with `AGENTS.md`.

The system is a local-first modular monolith that is evolving toward enterprise readiness. Current capabilities must remain understandable and testable while security, traceability, evaluation, observability, and operational maturity are added incrementally.

## 1. Goals and Invariants

The system must:

- ingest supported local documents while preserving provenance;
- split documents into stable, traceable chunks;
- generate embeddings and persist searchable vector data;
- support vector, BM25 lexical, and hybrid retrieval;
- build bounded prompts from retrieved evidence;
- generate answers grounded only in that evidence;
- abstain when evidence is insufficient;
- return traceable source citations;
- expose consistent CLI and FastAPI interfaces;
- produce safe operational diagnostics;
- remain modular, testable, and evolvable.

The following invariants are non-negotiable:

1. Metadata and provenance must survive every stage.
2. Query execution must not silently ingest or rebuild indexes.
3. Retrieval must not generate the final answer.
4. Generation must not retrieve or mutate stored documents.
5. Citations must refer only to retrieved evidence.
6. Secrets and private content must not be exposed through source code, responses, or logs.
7. Persisted contract changes require compatibility analysis.

## 2. Architectural Strategy

The default architecture is a modular monolith. Components have explicit responsibilities and are composed inside the process. This keeps the system understandable, locally runnable, and easy to test while preserving boundaries that could support later extraction.

Service extraction is not a default milestone. It requires evidence that scale, isolation, security, ownership, availability, compliance, or independent deployment needs cannot be met cleanly inside the modular monolith.

### Current module map

| Responsibility | Current modules |
|---|---|
| Configuration | `app/config.py` |
| Document loading | `app/document_loader.py` |
| Chunking | `app/chunker.py` |
| Embeddings | `app/embeddings.py` |
| Vector persistence | `app/vector_store.py` |
| Vector retrieval | `app/retriever.py` |
| Lexical retrieval | `app/bm25_retriever.py` |
| Hybrid fusion | `app/hybrid_retriever.py`, `app/utils/score_normalizer.py` |
| Prompt construction and citations | `app/prompt_builder.py` |
| LLM generation | `app/generator.py` |
| RAG orchestration | `app/pipeline.py`, `app/pipeline_factory.py` |
| API transport and schemas | `app/api.py`, `app/schemas.py` |
| CLI transport | `scripts/ask.py`, `scripts/query.py` |
| Ingestion entrypoint | `scripts/ingest.py` |
| Operational logging | `app/logging_utils.py` |
| Manual smoke checks | `scripts/smoke_bm25.py`, `scripts/smoke_pipeline_api.py` |

Entrypoints validate and translate external input, then call orchestration. Orchestration composes retrieval, prompt construction, generation, and source normalization. Retrieval uses index abstractions. Ingestion is the only normal flow that writes document indexes.

## 3. Primary Data Flows

### 3.1 Offline ingestion

```text
discover supported files
  -> load and normalize documents
  -> split while preserving provenance
  -> embed chunk content
  -> persist content, vectors, and metadata
  -> report counts, skips, failures, and index location
```

`scripts/ingest.py` is the current ingestion entrypoint. It must make index-changing behavior explicit and provide useful summaries. Partial failure behavior must be visible; failed items must not be reported as successfully stored.

### 3.2 Online query

```text
CLI or API request
  -> validate request
  -> construct configured pipeline
  -> retrieve vector and/or lexical candidates
  -> normalize and fuse scores when using hybrid retrieval
  -> build bounded grounded context
  -> generate answer
  -> validate and normalize sources
  -> return result and safe operational log
```

Online query paths must be read-only with respect to source documents and indexes unless a future API explicitly defines a separate write operation.

## 4. Document Loading

The current loader supports:

- `.txt`
- `.md`
- `.pdf`
- `.docx`

Supported formats must be determined by the loader interface rather than duplicated across unrelated modules.

Loading requirements:

- Accept an explicit input path; do not hardcode it inside core loading logic.
- Use a relative, stable source path when an input root is known.
- Normalize text without erasing meaningful content.
- Skip or report empty and unsupported files clearly.
- Preserve source identity and additive metadata.
- Isolate format-specific parsing so one parser failure can be diagnosed.
- Treat document content as untrusted input. Loading a document must not execute embedded code or follow unexpected network references.

## 5. Chunking

Chunking must preserve meaning and provenance while producing inputs suitable for embedding and retrieval.

Current defaults are configured centrally; values such as chunk size and overlap are operational settings, not permanent architecture constants.

Each chunk requires:

- text content;
- source identity;
- stable chunk identity or index within the source;
- any metadata required to trace the chunk to its origin.

Chunk identity must be deterministic enough for ingestion diagnostics and index lifecycle operations. Changing chunking rules can invalidate stored identifiers and retrieval baselines, so it requires an index compatibility decision and evaluation when relevant.

## 6. Embeddings and Vector Persistence

Embeddings convert chunk content into vectors. Metadata must not be embedded as if it were document content unless a deliberate, evaluated strategy says otherwise.

Embedding requirements:

- expose the model through a focused wrapper;
- make model name and provider configurable;
- validate output dimensions and item counts;
- preserve correspondence between each vector and its chunk;
- avoid live model or API requirements in unit tests;
- batch work when justified without losing per-item error visibility.

Vector persistence requirements:

- store chunk content, embedding, and provenance metadata together;
- expose focused operations for insertion and similarity search;
- keep database internals out of orchestration and transport layers;
- document whether returned numbers are distances or similarities;
- detect incompatible embedding dimensions or collections clearly;
- require controlled rebuild or migration when persisted contracts become incompatible.

Chroma is the current vector store. It is a replaceable implementation choice, not permission for callers to depend on Chroma internals.

## 7. Canonical Data Contracts

The exact Python representation may be a mapping, dataclass, Pydantic model, or adapter. Semantic fields must remain consistent across boundaries.

### 7.1 Source document

```python
{
    "content": "document text",
    "source": "relative/path/document.md",
    "metadata": {}
}
```

`metadata` may be flattened by existing adapters. The source identity and content remain required.

### 7.2 Chunk

```python
{
    "content": "chunk text",
    "source": "relative/path/document.md",
    "chunk_id": "stable-or-source-local-id",
    "metadata": {}
}
```

### 7.3 Retrieved context

```python
{
    "content": "retrieved text",
    "source": "relative/path/document.md",
    "chunk_id": "stable-or-source-local-id",
    "score": 0.82,
    "score_type": "normalized_similarity",
    "retrieval_strategy": "hybrid",
    "metadata": {}
}
```

Current implementations may omit new optional fields such as `score_type` or `retrieval_strategy`. New code should add them compatibly when consumers need unambiguous score semantics.

### 7.4 Pipeline result

The pipeline result must expose:

- the answer text;
- retrieved contexts used to build the prompt;
- normalized sources derived from those contexts;
- optional diagnostic data that is safe for its caller.

### 7.5 API source

The FastAPI response uses normalized source objects defined in `app/schemas.py`. A source must identify the originating document and may include chunk or preview information when the public schema allows it. API normalization must tolerate supported internal context representations without inventing provenance.

### 7.6 Request log

Operational request logs should contain structured, bounded fields such as timestamp, request identifier, duration, status, error category, retrieval strategy, result counts, and source identifiers. Questions, contexts, prompts, and answers must not be logged in full by default.

### Contract evolution

- Additive optional metadata is preferred.
- Required-field additions need defaults, adapters, or a versioning decision.
- Removing, renaming, or reinterpreting persisted fields requires a migration plan.
- Producers and consumers must be updated together or protected by compatibility adapters.

## 8. Retrieval

The system currently supports vector retrieval, BM25 retrieval, and hybrid retrieval.

Every retrieval strategy must:

- accept a validated, non-empty query;
- honor a validated positive `top_k`;
- return a consistent collection of retrieved contexts;
- preserve provenance and metadata;
- return an empty result rather than fabricate evidence;
- avoid modifying documents or indexes;
- define score direction and meaning.

### Score semantics

These values must not be treated as interchangeable:

- vector distance: lower may be better;
- vector similarity: higher is better;
- raw BM25 score: higher is better but query- and corpus-dependent;
- normalized component score: comparable only under the documented normalization method;
- fused hybrid score: meaningful only with its fusion weights and candidate sets.

Hybrid retrieval must normalize component scores before weighted fusion, apply deterministic tie-breaking, and preserve enough diagnostic information to explain which retrievers contributed when diagnostics are enabled.

Metadata filtering, reranking, query rewriting, multi-query retrieval, and caching are valid future extensions. They must integrate through retrieval or orchestration boundaries and must be justified by evaluation or documented failure cases.

## 9. Prompt Construction, Generation, and Citations

Prompt construction must:

- use only retrieved contexts selected for the request;
- label context entries so citations can be traced deterministically;
- bound total context size;
- preserve source identity through truncation and formatting;
- include clear grounded-answer and abstention instructions;
- treat retrieved content as data, not as trusted system instructions.

Generation must:

- consume the constructed prompt through a focused generator interface;
- use explicit model, timeout, temperature, and token settings;
- translate provider failures into safe application errors;
- never expose credentials in errors;
- avoid silently using outside knowledge when the contract requires grounded answers.

If retrieved evidence is empty or insufficient, the default compatibility response is exactly:

```text
Not found in knowledge base.
```

A future configurable or localized abstention policy must be versioned or introduced compatibly.

Sources must be derived only from retrieved contexts. Citation parsing, validation, deduplication, and final source formatting should be deterministic outside the LLM whenever practical. A model citation that does not map to a retrieved context must not become a reported source.

## 10. API and CLI Contracts

The current external interfaces are:

- FastAPI health and ask endpoints in `app/api.py`;
- request and response models in `app/schemas.py`;
- formal question-answering CLI in `scripts/ask.py`;
- retrieval inspection CLI in `scripts/query.py`.

External interfaces must:

- validate blank questions, invalid limits, and malformed filters;
- return or print predictable answer and source structures;
- distinguish configuration, readiness, validation, provider, and internal failures;
- keep debug context opt-in;
- avoid exposing sensitive internals;
- close owned resources cleanly.

Public field removal or semantic change requires an explicit compatibility or versioning decision.

## 11. Configuration

Configuration is centralized in `app/config.py` and focused provider configuration objects.

Configuration requirements:

- load secrets from environment variables or approved secret providers;
- provide safe defaults only for non-secret settings;
- validate numeric ranges, paths, model names, and required credentials;
- fail near startup or component construction with actionable errors;
- keep CLI or API overrides explicit and bounded;
- never log secret values;
- document new environment variables in `.env.example` and user-facing documentation.

Environment-specific paths must be configurable. Core logic must accept explicit dependencies or settings rather than reading global environment state repeatedly.

## 12. Error Handling and Resilience

Errors should be classified at stable application boundaries:

- invalid request or configuration;
- missing or unreadable source data;
- ingestion or parsing failure;
- embedding failure;
- index unavailable or incompatible;
- retrieval failure;
- prompt or context construction failure;
- provider timeout or provider response failure;
- unexpected internal failure.

Errors must include enough context for action without leaking secrets or private data. Partial ingestion must report failed and successful counts accurately. Retries, when added, must be bounded, observable, and safe for the operation. Non-idempotent writes require an idempotency strategy before automatic retry.

## 13. Security and Privacy

- Never commit credentials, `.env`, private documents, local vector data, model artifacts, or sensitive logs.
- Treat documents, queries, retrieved text, and model output as potentially sensitive and untrusted.
- Do not log full content by default; use identifiers or bounded, redacted previews.
- Prevent retrieved prompt-injection text from overriding system-level grounding and safety instructions.
- Apply file type, path, and size controls appropriate to the ingestion environment.
- Do not expose stack traces, internal paths, credentials, or raw provider responses through the API.
- Document which external provider receives which data before introducing or changing a provider.
- Add authentication, authorization, tenant isolation, encryption, audit retention, and deletion controls when deployment requirements introduce those boundaries; do not claim they exist before implementation and verification.

## 14. Observability

The current system provides request logging. It should evolve toward structured operational events with:

- stable request or correlation identifiers;
- stage names and durations;
- ingestion and retrieval counts;
- selected retrieval strategy and configuration identifiers;
- source identifiers rather than full source content;
- normalized error categories;
- bounded field sizes and documented retention.

Metrics should cover request rate, latency, failure rate, ingestion throughput, retrieval result counts, provider latency, and resource saturation when deployment needs them. Distributed tracing is appropriate only after requests cross meaningful process or service boundaries.

## 15. Testing and Evaluation

### Automated testing

The test strategy includes:

- unit tests for validation and component behavior;
- contract tests for result shapes, score semantics, and metadata preservation;
- integration tests for ingestion, retrieval, pipeline, CLI, and FastAPI composition;
- regression tests for fixed defects;
- opt-in smoke tests for configured local models and real providers.

Unit and normal integration tests must not require live APIs, network access, or model downloads.

### RAG evaluation

Representative, versioned evaluation data should measure:

- retrieval relevance, recall, and ranking quality;
- groundedness or faithfulness to supplied context;
- citation correctness and source traceability;
- abstention correctness when evidence is insufficient;
- answer usefulness where grounded evidence exists;
- latency, provider failure rate, and resource cost.

Chunking, embedding, retrieval, fusion, reranking, or prompt changes that can affect quality should be compared with the relevant baseline before promotion. Evaluation results must record dataset version, configuration, model identifiers, and retrieval strategy so results are reproducible.

## 16. Release and Change Gates

A behavior change is ready when:

1. Its requirement and affected contracts are explicit.
2. Relevant unit and regression tests pass.
3. Affected integration boundaries pass.
4. Persisted-data and API compatibility are evaluated.
5. Quality-sensitive changes are compared against available evaluation baselines.
6. Security, privacy, configuration, and observability effects are reviewed.
7. Documentation and examples match the behavior.
8. Remaining risks and rollback or rebuild requirements are recorded.

Production readiness is deployment-specific. Passing repository tests alone does not prove availability, capacity, compliance, backup, recovery, or operational readiness.

## 17. Enterprise Evolution Roadmap

### Stage 1: Stabilize contracts

- Align documentation with implemented behavior.
- Strengthen typed boundaries and compatibility adapters.
- Preserve metadata and deterministic identifiers.
- Standardize errors and automated contract tests.

### Stage 2: Measure and improve quality

- Establish representative evaluation datasets and baselines.
- Improve chunking, fusion, filtering, prompts, and citation validation using evidence.
- Add reranking or query transformation only when it improves measured outcomes.

### Stage 3: Harden security and observability

- Add structured events, metrics, request correlation, privacy controls, retention policies, dependency checks, and provider resilience appropriate to the deployment.
- Introduce authentication, authorization, or tenant isolation only when the product boundary requires them.

### Stage 4: Improve performance and delivery

- Add batching, caching, controlled concurrency, background ingestion, deployment packaging, readiness checks, backup, and index lifecycle management based on measured needs.
- Define capacity targets and service objectives before optimizing architecture around them.

### Stage 5: Extract infrastructure with evidence

- Consider queues, separate workers, metadata databases, object storage, independent services, or orchestration platforms only when concrete scale, isolation, ownership, compliance, availability, or deployment requirements justify them.
- Preserve contracts and observability across any extracted boundary.

## 18. Explicit Non-Goals

The enterprise direction does not automatically require:

- microservices;
- Kubernetes;
- a message queue;
- a new relational or distributed database;
- multi-tenancy or user accounts;
- a complex frontend;
- an agent framework;
- a specific cloud provider;
- replacing every mapping with a new domain class;
- speculative scalability work without measurements.

Correctness, security, measurable quality, and safe evolution take priority over infrastructure count and feature count.
