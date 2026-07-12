# Query Optimization Middleware Design

## Purpose

Add session-scoped conversation memory and context-aware query rewriting before
retrieval without coupling HTTP request state to the RAG pipeline. The feature
must preserve the user's original question for prompting, logging, and API
responses while supplying a separate rewritten query to retrieval.

`session_id` becomes a required `/ask` request field. This is an explicitly
approved API contract change. Direct Pipeline and CLI calls remain compatible
and do not require a session.

## Architecture

The implementation remains a modular monolith with four explicit boundaries:

- Conversation models and storage own validated conversation turns and
  session-isolated persistence.
- Query rewriters decide whether a query depends on history and return a
  structured rewrite result.
- HTTP middleware coordinates request-scoped history lookup and rewriting and
  exposes the result through `request.state`.
- The API invokes the framework-independent Pipeline and appends a completed
  conversation turn only after successful answer generation.

The application constructs one store and one rewriter at startup and injects
them into the middleware. Neither middleware nor routes access the in-memory
store's internal dictionary. A future MySQL store implements the same
`ConversationStore` interface without changing middleware behavior.

## Data Contracts

`ConversationTurn` is a frozen dataclass containing nonblank `user_message`,
nonblank `assistant_message`, and a timezone-aware UTC `created_at`. Invalid
turns raise `ValueError` at construction.

`ConversationStore` exposes:

```python
def get_recent_turns(session_id: str, limit: int) -> list[ConversationTurn]: ...
def append_turn(session_id: str, turn: ConversationTurn) -> None: ...
def clear_session(session_id: str) -> None: ...
```

`QueryRewriter.rewrite(query, history)` returns a frozen
`QueryRewriteResult` with `original_query`, `rewritten_query`,
`was_rewritten`, and `reason`. A rewriter never returns an answer or mutates
history.

The request-scoped fields are:

```text
request.state.original_question
request.state.rewritten_query
request.state.query_was_rewritten
request.state.query_rewrite_reason
request.state.conversation_history
request.state.session_id
```

## Conversation Storage

`InMemoryConversationStore` uses an `RLock` around all access to a dictionary
whose keys are validated session identifiers and whose values are lists of
immutable turns. Each append automatically retains only the configured most
recent turns. Reads return a new list, so callers cannot mutate internal
storage. Clearing one session does not affect any other session.

The configured capacity defaults to 5 and must be between 3 and 5 inclusive.
Invalid configured values fail during configuration loading. Per-read limits
must be positive; a read can request fewer turns than the store capacity.

This implementation is process-local. It does not provide persistence across
restarts or sharing across worker processes.

## Rule-Based Query Rewriting

The first production rewriter is deterministic and requires no network calls.
It rewrites only when the current query has a clear context-dependent signal,
such as Chinese pronouns (`它`, `这个`, `那个`, `这种方法`) or continuation
phrases (`继续`, `上面`, `刚才`, `前面`, `那为什么`). Independent questions
remain byte-for-byte equivalent after surrounding whitespace normalization.

When rewriting is needed, the rewriter examines at most the configured recent
history window and extracts a bounded topic from the most recent useful user
message. It replaces the contextual reference with that topic instead of
concatenating complete history. Duplicate sentences are removed and the final
query is bounded to 500 characters.

If history is empty or no reliable topic can be extracted, the original query
is returned with `was_rewritten=False` and a diagnostic reason. Age or numeric
background may be included to make a retrieval query self-contained, but the
rewriter must not calculate or emit the final answer.

An `LLMQueryRewriter` adapter boundary is reserved for later provider-backed
implementations. No live model call or new dependency is introduced in this
change.

## Middleware Flow

`QueryOptimizationMiddleware` handles only configured question-answer paths,
initially `POST /ask`. Health, documentation, OpenAPI, and all other requests
pass through unchanged.

For a target request, middleware reads the JSON body and restores it so FastAPI
and Pydantic can consume it normally. If JSON is malformed or required fields
are absent, incorrectly typed, or blank, middleware skips optimization and
allows the existing schema layer to return 422. It does not duplicate the
public request schema.

For a structurally usable request, middleware loads only that session's recent
turns. When rewriting is disabled, it uses the original question without
calling the rewriter. Store read failures fall back to empty history. Rewriter
failures fall back to the original question. Both failures are logged with
bounded, structured diagnostics and do not turn an otherwise valid RAG request
into a 500 response.

## Pipeline and Prompt Semantics

The Pipeline entry point becomes:

```python
def ask(
    self,
    question: str,
    top_k: int | None = None,
    *,
    retrieval_query: str | None = None,
) -> RAGResult: ...
```

The original `question` remains the prompt question and `RAGResult.question`.
The validated `retrieval_query` is passed to Retriever and Reranker. A missing
or whitespace-only retrieval query falls back to the original question for
backward compatibility. The Pipeline never imports or reads a FastAPI
`Request`.

## API Contract and Persistence Timing

`AskRequest` requires a stripped, nonblank `session_id` of at most 128
characters in addition to the existing validated question and `top_k`.
`AskResponse` adds `session_id`, `rewritten_query`, and
`query_was_rewritten`; its `question` remains the original question.

The route passes `request.state.rewritten_query` as `retrieval_query`. Only
after Pipeline success does it append a `ConversationTurn` containing the
original question and final answer. Pipeline or generator failures do not
create incomplete or fabricated turns.

## Configuration and Composition

Central configuration adds:

```text
CONVERSATION_HISTORY_LIMIT=5
QUERY_REWRITE_ENABLED=true
QUERY_REWRITE_PROVIDER=rule_based
QUERY_REWRITE_TIMEOUT=10.0
```

The history limit is validated as an integer from 3 through 5. The provider is
validated against the currently supported `rule_based` value. Timeout remains
centralized for a future LLM implementation and must be positive.

Application dependencies are constructed once in a focused composition
location. Tests may construct an isolated FastAPI application with injected
fake Pipeline, QueryRewriter, and ConversationStore instances.

## Logging and Privacy

Query optimization diagnostics include request ID when available, session ID,
bounded original question, bounded rewritten query, rewrite flag and reason,
history turn count, and rewrite latency. Question fields are capped at 500
characters. Logs never contain full history, authorization headers, provider
payloads, secrets, or unbounded prompts and answers.

Existing request logging is extended compatibly where practical. Error
responses remain generic and do not reveal provider or filesystem details.

## Testing Strategy

Implementation follows test-driven development in this order:

1. Conversation model and store validation, isolation, eviction, copying,
   clearing, and invalid inputs.
2. Rule-based rewriting for independent questions, pronouns, missing history,
   numeric context, output bounds, and blank input.
3. Middleware state, correct-session history, path filtering, disabled mode,
   exception fallback, and reusable request body behavior.
4. Pipeline retrieval-query routing, original-question preservation, legacy
   behavior, reranker behavior, and blank retrieval-query fallback.
5. API two-round integration, session isolation, five-turn eviction,
   success-only persistence, validation, and unaffected health behavior.
6. Offline mock smoke execution and README command examples.

All unit and API tests use fakes and must not call external APIs, download
models, connect to MySQL, or depend on a populated Chroma database. Completion
requires focused test runs, `python -m pytest -q`, mock smoke execution,
`git diff --check`, and full diff inspection.

## Compatibility and Known Limits

- `/ask` requests without `session_id` now return 422 by approved design.
- Direct Pipeline calls and existing CLI flows retain their current signatures
  and behavior unless they opt into `retrieval_query`.
- Conversation data is lost on process restart.
- Multiple worker processes do not share conversation history.
- Cross-process persistence requires a future MySQL or Redis implementation of
  the stable store interface.
