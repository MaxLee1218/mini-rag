# RAG Project Agent Rules

This file defines repository-wide rules for AI coding agents working on this project.

The project is an evolving Retrieval-Augmented Generation system. Its direction is a mature, enterprise-capable RAG platform developed through safe, incremental changes. Enterprise quality means correctness, security, data integrity, traceability, maintainability, observability, measurable quality, and predictable failure behavior. It does not mean adding infrastructure without evidence.

For RAG behavior, data contracts, and quality requirements, also follow `rag_spec.md`.

## 1. Instruction Precedence

Use this precedence when instructions or evidence disagree:

```text
explicit user request
  -> repository AGENTS.md
  -> rag_spec.md for RAG behavior and contracts
  -> current code and tests as evidence of implemented behavior
```

- `AGENTS.md` governs how changes are investigated, implemented, tested, and delivered.
- `rag_spec.md` governs system behavior, architectural invariants, data contracts, and target qualities.
- Existing code and tests describe the current implementation, but do not override an explicit requirement.
- If instructions conflict or a requested change would violate a safety invariant, report the conflict and resolve it explicitly. Do not silently choose an interpretation that risks data loss, security, or compatibility.

## 2. Project Direction

The default architecture is a modular monolith. Preserve clear boundaries and stable interfaces while expanding capability.

The system may progressively add better chunking, metadata filters, hybrid retrieval, reranking, query transformation, evaluation, caching, background ingestion, additional providers, and deployment features when a concrete requirement justifies them.

Do not introduce microservices, queues, new databases, orchestration platforms, multi-tenancy, or complex agent frameworks only because the target is “enterprise-grade.” Extract infrastructure or services only when measured scale, reliability, security, ownership, compliance, or deployment requirements justify the operational cost.

## 3. Current Responsibility Boundaries

The current repository is organized by responsibility:

```text
configuration
  app/config.py

ingestion
  app/document_loader.py
  app/chunker.py
  app/embeddings.py
  app/vector_store.py
  scripts/ingest.py

retrieval
  app/retriever.py
  app/bm25_retriever.py
  app/hybrid_retriever.py
  app/utils/score_normalizer.py

generation
  app/prompt_builder.py
  app/generator.py

orchestration
  app/pipeline.py
  app/pipeline_factory.py

transport and schemas
  app/api.py
  app/schemas.py
  scripts/ask.py
  scripts/query.py

operations and diagnostics
  app/logging_utils.py
  scripts/smoke_bm25.py
  scripts/smoke_pipeline_api.py
```

These filenames describe the current structure, not a prohibition on adding focused modules.

### Boundary rules

- Transport layers validate and translate external requests; they must not absorb retrieval, prompt, generation, or persistence logic.
- Orchestration composes components through explicit interfaces; it must not depend on vector database internals.
- Retrieval returns relevant contexts and scores; it must not generate the final answer or mutate source documents.
- Generation consumes a bounded prompt or context; it must not perform ingestion or directly query persistence.
- Ingestion owns document discovery, chunk creation, embedding, and index writes. Query paths must not silently rebuild indexes.
- Configuration remains centralized in `app/config.py` or focused configuration objects. Core functions must not hardcode secrets or environment-specific paths.
- Shared schemas and metadata semantics must remain consistent across ingestion, retrieval, pipeline, CLI, API, and logging boundaries.
- Prefer small, explicit interfaces and dependency injection over hidden globals or construction inside core logic.

## 4. Safe Change Rules

Before editing:

1. Inspect the affected module, its callers, tests, configuration, and persisted data contracts.
2. Identify whether the change affects ingestion, online queries, public API/CLI behavior, stored metadata, or provider integration.
3. Define the narrowest change that satisfies the request.
4. Determine the verification needed before implementation.

While editing:

- Keep changes scoped to the requested behavior.
- Preserve unrelated user changes and avoid broad opportunistic refactors.
- Prefer backward-compatible additions to breaking changes.
- Keep functions and classes focused, readable, and independently testable.
- Use structured inputs and outputs where multiple values or metadata cross a boundary.
- Do not discard provenance, chunk identity, retrieval scores, or other required metadata.
- Do not add speculative abstraction. Introduce an interface when there is an actual boundary, alternative implementation, or testing need.
- Update documentation and configuration examples when public behavior or operational requirements change.

## 5. Compatibility and Data Integrity

Changes to persisted vectors, identifiers, metadata, collection names, embedding dimensions, distance metrics, or chunking rules can invalidate existing indexes.

- Additive metadata changes are preferred.
- Renaming, removing, or changing the meaning of a persisted field requires an explicit compatibility or migration plan.
- Embedding-model or dimension changes require index compatibility checks and normally a controlled rebuild.
- Destructive rebuilds, migrations, and cleanup must be explicit, recoverable where practical, and never triggered as an incidental query side effect.
- Stable API and CLI fields must not be removed or reinterpreted without a versioning or migration decision.
- Source and chunk provenance must remain traceable from raw input through the final response.

## 6. RAG Quality Rules

- Generated answers must be grounded in retrieved context.
- If evidence is missing or insufficient, the system must abstain according to `rag_spec.md`.
- Sources must be derived only from retrieved contexts and must never be fabricated.
- Retrieval strategies must document score meaning. Distance, similarity, lexical score, normalized score, and fused score are not interchangeable.
- Changes to chunking, retrieval, fusion, prompts, or citation behavior require tests and, when evaluation data exists, comparison against the relevant baseline.
- Quality improvements should be supported by failure examples or measurements rather than intuition alone.

## 7. Testing and Verification

Use the narrowest useful test first, followed by affected integration boundaries.

- Add or update unit tests for changed behavior.
- Add regression tests for defects.
- Test data contracts and metadata preservation when boundary shapes change.
- Test ingestion, retrieval, pipeline, CLI, and API composition when their integration changes.
- Mock or inject external APIs, embedding models, clocks, filesystems, and stores in basic tests where practical.
- Unit tests must not require live provider calls, network access, or model downloads.
- Real-provider and full-chain smoke tests must be explicit and opt-in.
- Do not claim a change is complete or passing without fresh relevant verification output.
- If the full suite cannot run, report exactly what was run, what was not run, and why.

Documentation-only changes do not require invented runtime tests. They do require reference checks, consistency review, `git diff --check`, and a complete diff inspection.

## 8. Security and Privacy

- Load secrets from environment variables or approved secret providers. Never hardcode or commit credentials.
- Never commit `.env`, private source documents, local indexes, model artifacts, or runtime logs containing sensitive data.
- Do not log API keys, authorization headers, full provider payloads, or unredacted exceptions that may contain secrets.
- Do not log full private documents, prompts, contexts, questions, or answers by default. Use identifiers, structured fields, or bounded and redacted previews when diagnostics require content.
- External calls require explicit timeouts and errors translated into safe, actionable application errors.
- API responses must not expose secrets, unnecessary internal paths, stack traces, or sensitive provider details.
- Validate configuration at startup or component construction so failures occur predictably.
- Treat new dependencies and providers as supply-chain and data-boundary decisions; document why they are needed and what data they receive.

## 9. Observability and Operations

- Prefer structured logs with stable event names and request correlation identifiers.
- Record useful operational facts such as stage, duration, counts, selected retrieval strategy, source identifiers, and error category without exposing sensitive content.
- Bound log fields and previews to prevent uncontrolled storage growth.
- Keep health checks lightweight; distinguish process health from dependency or index readiness when those concepts are introduced.
- Add metrics, tracing, retention policies, retries, circuit breakers, or background workers only when operational requirements call for them.
- Retries must be bounded and safe. Do not retry non-idempotent writes without an idempotency strategy.

## 10. Dependencies and Infrastructure

Before adding a package, framework, service, datastore, or deployment component:

1. State the requirement it satisfies.
2. Check whether current dependencies or standard-library facilities already satisfy it.
3. Evaluate security, maintenance, licensing, data exposure, and operational cost.
4. Define how it will be tested and configured.
5. Avoid combining the adoption with unrelated refactoring.

Do not invent APIs or write pseudo-code in place of a requested implementation.

## 11. Debugging Workflow

Identify the failing flow before proposing a fix.

```text
offline ingestion
  discovery -> loading -> chunking -> embedding -> index write

online query
  validation -> pipeline construction -> retrieval -> fusion
  -> prompt building -> generation -> citation/source normalization
  -> transport response and logging
```

Inspect the error, reproduce it at the smallest boundary, trace data and metadata through that boundary, form a testable hypothesis, and add a regression test before or with the fix. Do not randomly rewrite the full pipeline.

## 12. Development Workflow

For each requested change:

1. Inspect repository instructions and relevant specifications.
2. Confirm current behavior from code and tests.
3. Define scope, contracts, compatibility risks, and acceptance criteria.
4. Add or update focused tests when behavior changes.
5. Implement the smallest coherent change.
6. Run focused verification, then affected integration checks.
7. Review the full diff for unintended edits, secrets, and stale documentation.
8. Summarize what changed, verification evidence, remaining risks, and follow-up work.

Do not delete files, rewrite persisted data, change public contracts, or perform destructive Git operations unless explicitly authorized.

## 13. Priority Order

Use this order when trade-offs are unavoidable:

```text
Security and correctness
  > Data integrity and traceability
  > Compatibility and maintainability
  > Observability and operability
  > Performance
  > Feature count
```

The goal is steady, evidence-based improvement of a trustworthy RAG system—not maximum complexity or maximum feature count.
