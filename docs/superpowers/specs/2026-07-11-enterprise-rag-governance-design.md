# Enterprise RAG Governance and Specification Design

## 1. Purpose

The repository has grown beyond its original minimal RAG v1 scope. It now includes document ingestion, recursive chunking, embeddings, Chroma storage, vector and BM25 retrieval, hybrid retrieval, prompt construction, answer generation, pipeline orchestration, CLI and HTTP entrypoints, request logging, and broad automated test coverage.

The existing `AGENTS.md` and `rag_spec.md` still describe a fixed learning-only pipeline and prohibit several capabilities that already exist. They will be rewritten to describe the current system accurately and guide it toward a mature enterprise RAG system without prematurely introducing distributed architecture.

## 2. Chosen Direction

The project will remain a modular monolith by default. Enterprise capabilities will be introduced incrementally behind clear module boundaries and stable contracts. A service may be extracted only when measured scale, reliability, security, ownership, or deployment requirements justify the added operational cost.

“Enterprise-grade” means disciplined engineering rather than infrastructure volume. The target qualities are correctness, security, data integrity, traceability, maintainability, measurable retrieval and answer quality, observability, predictable failure behavior, and safe evolution.

## 3. Document Responsibilities

### `AGENTS.md`

`AGENTS.md` will govern how coding agents work in this repository. It will define:

- instruction precedence and the relationship to `rag_spec.md`;
- scope control and evidence-based changes;
- current architectural boundaries and dependency direction;
- compatibility and data migration expectations;
- test and verification requirements;
- security, privacy, secret handling, and logging restrictions;
- dependency and infrastructure adoption rules;
- debugging and incident-safe workflows;
- documentation expectations and completion criteria.

It will avoid freezing the project to specific filenames where a responsibility-based boundary is more durable. It may still list current modules to help agents navigate the repository.

### `rag_spec.md`

`rag_spec.md` will define what the RAG system does and the qualities it must preserve. It will cover:

- current architecture and module map;
- offline ingestion and online query flows;
- canonical data contracts and metadata requirements;
- retrieval interfaces and hybrid retrieval behavior;
- grounded generation, abstention, and citation traceability;
- configuration and environment handling;
- error taxonomy and safe degradation;
- structured logging, metrics, and future tracing boundaries;
- retrieval and answer evaluation;
- security and privacy requirements;
- testing levels and release gates;
- phased enterprise evolution.

Implementation details that are likely to change will be expressed as defaults or current choices, not timeless architectural requirements.

## 4. Architecture Model

The specification will describe two primary flows.

### Offline ingestion

```text
source files
  -> document loading
  -> normalization and chunking
  -> embedding
  -> vector/index persistence
  -> ingestion summary and diagnostics
```

### Online query

```text
API or CLI request
  -> request validation
  -> pipeline construction
  -> vector and/or lexical retrieval
  -> score normalization and fusion
  -> prompt construction
  -> grounded generation
  -> source normalization
  -> response and structured request log
```

The modular monolith will preserve explicit boundaries between transport, orchestration, retrieval, generation, persistence, and configuration. Entry points may compose modules but must not absorb core RAG logic.

## 5. Data and Interface Contracts

The rewritten specification will retain the essential fields used by the current system while allowing metadata to evolve.

Core contracts will distinguish:

- source documents;
- chunks prepared for indexing;
- retrieved contexts with provenance and scores;
- generated results with answer, contexts, and normalized sources;
- API request and response schemas;
- structured request log records.

Required provenance such as source identity and chunk identity must survive ingestion, retrieval, prompt construction, generation, and response formatting. New metadata fields may be added compatibly. Renaming, changing meaning, or removing persisted fields requires an explicit compatibility or migration plan.

Typed models may coexist with mapping-based adapters while the codebase evolves. The specification will define semantic requirements rather than force an immediate repository-wide conversion.

## 6. Retrieval and Generation Quality

Vector, BM25, and hybrid retrieval are current supported strategies. Retrieval implementations must return a consistent result shape and must not generate answers. Score semantics must be documented because distances, similarities, lexical scores, and normalized fusion scores are not interchangeable.

Reranking, query rewriting, filters, multi-query retrieval, and caching are permitted future extensions when backed by a concrete requirement and evaluation. They must integrate through retrieval or orchestration boundaries rather than bypass the pipeline.

Generation must remain grounded in retrieved context. When context is missing or insufficient, the system must abstain using the configured fallback behavior. Sources must be derived only from retrieved contexts and remain traceable to indexed content. Citation formatting and validation must be deterministic outside the LLM whenever practical.

## 7. Safety and Operational Quality

The documents will establish these rules:

- secrets come from environment-backed configuration and are never committed or logged;
- private document content, prompts, and answers are not logged in full by default;
- logs use bounded previews or identifiers when content is necessary for diagnostics;
- external providers must have explicit timeouts and clear error translation;
- unit tests do not require live APIs or model downloads;
- destructive index rebuilds or data migrations must be explicit and recoverable;
- API errors must not expose credentials, internal paths, or provider payloads unnecessarily;
- configuration is validated near startup or component construction.

The modular monolith should support structured logs and stable request identifiers first. Metrics and distributed traces may be added when deployment requirements justify them.

## 8. Testing and Evaluation

Changes will be verified at the narrowest useful level and then at affected integration boundaries.

The target test model includes:

- unit tests for individual modules and validation rules;
- contract tests for result shapes and metadata preservation;
- integration tests for ingestion, retrieval, pipeline, CLI, and API composition;
- smoke tests for manually configured real providers;
- regression tests for every fixed defect;
- offline evaluation sets for retrieval relevance, citation correctness, groundedness, and abstention behavior.

Live-provider tests remain opt-in. A change is not complete until relevant automated tests pass and documentation or configuration examples are updated when public behavior changes.

## 9. Enterprise Evolution Stages

### Stage 1: Stabilize contracts

Keep the modular monolith, align documentation with current behavior, strengthen typed boundaries, metadata preservation, deterministic errors, and automated tests.

### Stage 2: Measure and improve quality

Create representative evaluation datasets and baselines. Improve chunking, retrieval fusion, filtering, reranking, prompts, and citation validation only when metrics or documented failure cases justify the change.

### Stage 3: Harden security and observability

Add structured operational events, metrics, request correlation, privacy controls, retention rules, dependency checks, and provider resilience appropriate to the deployment environment.

### Stage 4: Improve performance and delivery

Introduce batching, caching, background ingestion, concurrency controls, deployment packaging, health/readiness checks, and index lifecycle management based on measured needs.

### Stage 5: Extract infrastructure only with evidence

Consider queues, separate workers, dedicated metadata stores, object storage, independent services, multi-tenancy, or orchestration platforms only when concrete scale, isolation, ownership, compliance, or availability requirements cannot be met cleanly by the modular monolith.

## 10. Explicit Non-Goals

The rewrite itself will not:

- change application code or runtime behavior;
- introduce microservices, Kubernetes, queues, databases, or agent frameworks;
- require immediate multi-tenancy or authentication;
- replace all dictionaries with new domain classes;
- prescribe a specific cloud provider;
- claim production readiness without deployment-specific evidence;
- preserve obsolete v1 restrictions that conflict with the current repository.

## 11. Acceptance Criteria

The documentation update is complete when:

1. `AGENTS.md` accurately governs safe, incremental work on the current repository.
2. `rag_spec.md` accurately represents the current modules and both primary data flows.
3. The documents consistently favor a modular monolith and evidence-based evolution.
4. Current hybrid retrieval, API, CLI, logging, and tests are no longer described as forbidden future work.
5. Grounding, provenance, citation, privacy, compatibility, testing, and evaluation requirements are explicit.
6. Contradictory v1-only rules and stale filename references are removed or reframed as historical/current defaults.
7. No application code, dependencies, runtime data, or secrets are modified.
