# Badcase Feedback Loop Design

## Goal

Add a lightweight offline feedback loop that discovers likely RAG failures from
request logs, preserves human analysis, and incrementally feeds resolved cases
into the existing evaluation dataset. The feature remains independent of the
online RAG path and introduces no database, service, or provider dependency.

## Scope and Constraints

- Do not modify retriever, generator, pipeline, or API request behavior.
- Do not add full retrieved contexts to the current request log contract.
- Use JSONL request logs and JSON files as the only inputs and outputs.
- Preserve existing badcases, human annotations, and evaluation samples.
- Keep generated evaluation data compatible with `python eval/run_eval.py`.
- Add focused automated tests for every new behavior.

## Data Contract

`eval/badcase_schema.py` defines a `BadCase` dataclass with these fields in
stable serialization order:

```text
question: str
answer: str
expected_answer: str | None
contexts: list[str]
sources: list[str]
error_type: str
root_cause: str | None
solution: str | None
timestamp: str
```

`to_dict()` returns a JSON-serializable mapping. `from_dict()` validates
required strings and string lists while treating absent optional annotation
fields as `None`. The badcase record does not persist `request_id`; the
exporter uses timestamp as the stable deduplication key allowed by the feature
requirements.

## Export Flow

`scripts/export_badcases.py` reads `logs/rag_requests.jsonl` and writes an array
to `eval/badcases.json`. It exposes importable functions for classification,
parsing, merging, and file orchestration, with a small CLI entry point.

Classification assigns at most one error type per log entry, using this
precedence:

1. A blank `answer` becomes `empty_answer`.
2. An explicitly present empty `contexts` list becomes `retrieval_failure`.
3. An answer containing `Not found in knowledge base.` or `未找到相关信息`
   becomes `generation_or_retrieval_failure`.

A missing `contexts` field means the retrieval result is unknown and does not
trigger the second rule. This prevents all current request logs from being
misclassified and avoids changing the privacy-sensitive logging contract.

Existing badcases are loaded before new candidates. Existing records and human
annotations remain unchanged, and candidates with a timestamp already present
in the output are skipped. The output directory and output file are created
when absent. A missing input log produces a valid empty badcase file when no
existing output exists. Invalid JSON or invalid record shapes fail with an
actionable message instead of silently discarding evidence.

## Import Flow

`scripts/import_badcases_to_eval.py` reads `eval/badcases.json` and
incrementally updates the repository's actual evaluation dataset,
`evaluation/dataset/eval_dataset.json`.

For each badcase, `ground_truth` is selected from the first nonblank value in
this order:

1. `expected_answer`
2. `solution`

Records without either value remain in the badcase file and are skipped during
import so the strict evaluation dataset contract is never made invalid.
Nonempty `contexts` become `reference_contexts`. A `metadata` object retains
`error_type`, `root_cause`, `solution`, and `timestamp` for traceability; the
current evaluation loader safely ignores this additive field.

Existing evaluation rows are preserved. Questions are normalized by Unicode
case folding and whitespace collapsing, matching the current dataset loader's
duplicate semantics. A badcase whose normalized question is already in the
evaluation dataset is skipped rather than replacing the trusted baseline.
Missing parent directories and a missing destination file are created as
needed. Invalid source or destination JSON fails explicitly.

## CLI Behavior

Both scripts accept optional path arguments so tests and operators can work
with isolated files. With no arguments, the commands are:

```bash
python scripts/export_badcases.py
python scripts/import_badcases_to_eval.py
python eval/run_eval.py
```

Each tool prints a bounded summary containing discovered, added, duplicate,
and skipped counts as applicable. Expected data errors return a nonzero exit
code without exposing document content.

## Testing

Tests cover:

- complete `BadCase` serialization round trips and optional-field defaults;
- validation failures for malformed badcase records;
- all three classification rules and their precedence;
- missing versus explicitly empty contexts;
- creation of missing output paths;
- preservation of existing annotations and timestamp deduplication;
- answer precedence during import;
- unresolved-case skipping;
- context and metadata mapping;
- normalized-question deduplication without overwrites;
- compatibility of the merged dataset with the existing dataset loader;
- CLI defaults and error exit behavior.

Verification runs the focused new tests first, then the full test suite,
`git diff --check`, and a complete diff review.

## Compatibility and Privacy

The change adds offline files and scripts only. It does not alter persisted
vectors, chunk identifiers, API schemas, log producers, or online request
latency. Existing evaluation rows retain their current fields and meanings.
The exporter consumes full questions and answers already present in the local,
Git-ignored log; operators remain responsible for keeping generated badcase
files free of private production content before committing them.
