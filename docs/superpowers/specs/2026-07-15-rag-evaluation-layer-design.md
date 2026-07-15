# RAG Evaluation Layer Design

## Purpose

Add an offline evaluation and performance-analysis layer to mini-RAG without
changing the behavior or public interfaces of ingestion, retrieval, generation,
the RAG pipeline, CLI commands, or API endpoints. The evaluation layer must use
the existing `RAGPipeline.ask()` method, measure retrieval and generation quality,
capture stage latency, and produce reproducible JSON and Markdown reports.

The first checked-in dataset and report target the repository's current public
sample documents. The resulting scores are a dated snapshot, not a general claim
about production quality.

## Scope and Constraints

- Keep evaluation independent from the online query and ingestion flows.
- Do not copy pipeline orchestration or modify core retriever, generator, or
  pipeline interfaces.
- Centralize paths, thresholds, models, and runtime settings in `app/config.py`.
- Preserve source provenance while preventing reports from storing full retrieved
  contexts or unbounded generated content.
- Make metric failures explicit. Missing or failed measurements are `null`, never
  fabricated zeros.
- Keep unit tests offline and deterministic. Live provider evaluation is an
  explicit, authorized final verification step.

## Architecture

```text
eval/run_eval.py
  -> DatasetManager.load()
  -> build_default_pipeline()
  -> EvaluationRunner
       -> LatencyTracer.wrap(pipeline)
       -> pipeline.ask(question)
       -> context/source normalization
       -> RetrievalEvaluator.hit()
  -> RagasEvaluator.evaluate(samples)
  -> LatencyAnalyzer.percentiles()
  -> ReportGenerator.write(json, markdown)
```

The implementation uses evaluation-side composition. The tracer temporarily
wraps the components on the evaluation-only pipeline instance and restores them
even when a request fails. Subclassing `RAGPipeline` is rejected because it would
duplicate orchestration. Adding hooks to the core pipeline is rejected for this
increment because it would change a stable interface solely for offline tooling.

## Files and Responsibilities

### `evaluation/models.py`

Defines typed dataclasses or equivalent structured contracts shared by evaluation
modules: dataset samples, per-sample pipeline observations, latency values, metric
values, failures, aggregate results, and run metadata. Optional measurements use
`None`; their serialization becomes JSON `null`.

### `evaluation/dataset_manager.py`

Loads UTF-8 JSON from a configured or explicitly supplied path. It validates that
the top level is a non-empty list and that every row has nonblank `question` and
`ground_truth` strings. It also validates optional `reference_contexts` as a
non-empty list of nonblank strings and optional `should_abstain` as a boolean.
Duplicate normalized questions are rejected because they distort aggregate
statistics.

### `evaluation/evaluator.py`

Coordinates evaluation of dataset samples against an injected pipeline. Each
sample is executed through `pipeline.ask(question)` exactly once. The evaluator
normalizes supported current context representations into text and source IDs,
records route and sources, computes deterministic retrieval hit and abstention
outcomes, and captures sample-level errors without aborting later samples.

Retrieval hit applies only to answerable samples. A hit occurs when any normalized
`reference_contexts` entry is contained in any normalized retrieved context.
Normalization is limited to case folding and whitespace collapsing; it does not
claim semantic equivalence. Samples with `should_abstain=true` are excluded from
the hit-rate denominator and instead contribute to a separate abstention accuracy
measurement. Correct abstention uses the existing exact pipeline contract:
`Not found in knowledge base.` after removing any normalized source suffix.

### `evaluation/latency_tracer.py`

Provides an evaluation-only context manager that wraps:

- the top-level retriever's `retrieve()` call;
- the query embedder's `embed_query()` call when discoverable in standard,
  parent-child, or hybrid retrievers;
- the generator's `generate()` call;
- the complete `pipeline.ask()` wall-clock call.

All measurements use `time.perf_counter_ns()` and are converted to milliseconds.
Embedding time is accumulated because a retriever may invoke it more than once.
`retrieval` is exclusive time: total `retrieve()` duration minus accumulated
embedding duration, floored at zero to absorb timer precision. `generation` is
the generator call duration, and `total` is the complete ask duration. Reranking,
prompt construction, and source normalization remain visible in `total` but are
not incorrectly attributed to another stage.

If an injected custom retriever does not expose a recognized embedder,
`embedding` is `null` with a warning. Other supported measurements still run.
Every temporary replacement is restored in a `finally` path.

The tracer accepts a small span/event sink protocol so future Langfuse or
OpenTelemetry exporters can consume measurements without changing evaluation
logic. This release supplies an in-memory sink only.

### `evaluation/ragas_evaluator.py`

Adapts evaluation observations to the four required RAGAS metrics:

- Faithfulness
- Answer Relevancy
- Context Precision
- Context Recall

The preferred path supports the RAGAS 0.4 collections API and uses an explicitly
constructed OpenAI client, configured evaluation LLM, and configured embedding
model. A legacy adapter supports older `ragas.evaluate()` and legacy metric
imports when collections are unavailable. Imports are lazy so normal application
startup never depends on RAGAS.

RAGAS 0.4.3 has a known unconditional import of the removed
`langchain_community.chat_models.vertexai` module. When and only when that exact
module is absent, the adapter installs a minimal in-memory `ChatVertexAI` type
shim before retrying the RAGAS import. It does not modify site-packages, install a
VertexAI provider, or downgrade LangChain. Other import failures are surfaced
normally.

The evaluator returns per-sample optional scores, aggregate means over valid
scores, valid score counts, version/model metadata, warnings, and structured
errors. A failed metric or sample is `null`; evaluation of independent metrics and
the remainder of the report continues where the installed RAGAS API permits it.
If RAGAS is absent, status is `unavailable` with an actionable installation
message. Provider or partial metric failures produce `partial` rather than a
fabricated complete result.

### `evaluation/latency_analyzer.py`

Consumes multiple latency observations and uses `numpy.percentile` to calculate
p50 and p95 for embedding, retrieval, generation, and total stages. Null values
are ignored, and each stage includes a valid observation count. A stage with no
valid observations reports null percentiles.

p50 represents the typical user experience. p95 exposes the slow-request tail
and is the primary bottleneck indicator in this small offline run.

### `evaluation/report_generator.py`

Builds a single structured report model and renders it to JSON and Markdown. Both
files are UTF-8 and written atomically by replacing a temporary file in the same
directory. Parent directories are created as needed.

The Markdown report contains dataset size/date, retrieval hit rate, abstention
accuracy, the four RAGAS aggregates, the four-stage p50/p95 latency table, run
warnings, and failed examples. Failed examples include a bounded question and
answer preview, sources, and one or more reason labels. Full retrieved contexts
are not persisted.

### `eval/run_eval.py`

Provides the requested `python eval/run_eval.py` entrypoint. It resolves the
repository root when run as a file, accepts optional dataset/report path and
`top_k` overrides, constructs the existing default RAG pipeline, invokes the
evaluation modules, and prints a concise summary with output paths.

Initialization-level dataset, configuration, index, provider, or report-write
failures return a nonzero exit code with an actionable error. A single pipeline
sample or individual RAGAS metric failure is recorded and does not abort the
entire run.

## Dataset

`evaluation/dataset/eval_dataset.json` contains at least ten samples grounded in:

- `data/raw/sample/rag_notes.txt`
- `data/raw/sample/embedding_notes.txt`

Answerable samples use factual ground truths and exact reference excerpts from
these documents. They cover direct facts, explanations requiring multiple
sentences, and questions whose answer depends on retrieved context. Negative
samples ask for facts not present in either document, use the existing abstention
text as ground truth, and set `should_abstain=true`. No answer or reference
context is invented.

## Configuration

Focused evaluation settings are added to `app/config.py`, with environment
overrides documented in `.env.example`:

- dataset and JSON/Markdown report paths relative to `PROJECT_ROOT` by default;
- evaluation `top_k`;
- RAGAS LLM and embedding model names;
- RAGAS request timeout;
- faithfulness and context-recall failure thresholds, both defaulting to `0.7`;
- maximum failed-example question and answer preview lengths.

Paths are normalized through project-root-aware helpers. Secrets continue to use
the existing OpenAI environment configuration and are never placed in reports.

## Failure Classification

A sample may have multiple failure reasons:

- `retrieval miss`: an answerable sample has no deterministic reference-context
  match;
- `hallucination`: a valid faithfulness score is below `0.7` or its configured
  replacement;
- `insufficient context`: an answerable sample has no contexts, or a valid
  context-recall score is below `0.7` or its configured replacement.

A missing RAGAS score creates a metric error, not a quality failure. This keeps
operational failure distinct from measured poor quality.

## Report Contract

The JSON report includes:

- schema version and status (`completed`, `partial`, or `failed`);
- timestamp, dataset path/hash, sample counts, pipeline settings, RAGAS version,
  and evaluator model names;
- retrieval hit rate with numerator and denominator;
- abstention accuracy with numerator and denominator;
- each RAGAS aggregate with valid score count;
- p50/p95 and valid count for each latency stage;
- bounded failed-example details and structured run warnings/errors.

The Markdown report presents the requested human-readable tables and explanations
of p50 and p95. README embeds the aggregates from one authorized live run with its
date and model configuration and links to the generated reports. It explicitly
labels the values as a sample snapshot.

## Testing Strategy

Implementation follows red-green-refactor cycles. Focused offline tests cover:

1. dataset loading, optional fields, duplicate detection, and validation errors;
2. context normalization, retrieval hit denominator, and abstention accuracy;
3. standard, parent-child, and hybrid embedder discovery;
4. exclusive latency arithmetic and restoration after success or exception;
5. NumPy p50/p95 calculation, null filtering, and empty stages;
6. RAGAS collections, legacy, missing-package, upstream shim, and partial metric
   failure paths using injected fake APIs;
7. JSON/Markdown structure, escaping, bounded previews, and atomic replacement;
8. CLI orchestration and exit behavior with fake pipelines and evaluators.

After focused tests, existing pipeline and factory regression tests run, followed
by the full offline suite. Final checks include `git diff --check`, complete diff
inspection, and the authorized live evaluation. Live provider calls and model
downloads are never part of unit tests.

## Acceptance Criteria

- `python eval/run_eval.py` evaluates every valid dataset row through the existing
  `pipeline.ask()` method and writes both reports.
- The required four RAGAS aggregates and per-sample optional scores are present.
- Retrieval hit rate and negative-sample abstention accuracy have explicit,
  non-misleading denominators.
- Embedding, retrieval, generation, and total latency p50/p95 values are reported
  without double-counting embedding inside retrieval.
- Missing RAGAS or individual evaluation failures remain diagnosable and do not
  crash unrelated application imports or erase non-RAGAS results.
- Existing ingest/query/ask/API behavior and core component interfaces remain
  unchanged.
- README contains the authorized real-run snapshot only after a successful live
  execution; no result is fabricated.
