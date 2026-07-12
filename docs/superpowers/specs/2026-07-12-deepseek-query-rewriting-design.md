# DeepSeek Query Rewriting Design

## Purpose

Replace the production rule-based query rewriting path with an LLM-based
rewriter that calls the project's existing DeepSeek API before retrieval. The
LLM receives the current question and the session's recent conversation turns,
then returns one retrieval-ready question. The user's original question remains
unchanged for generation, logging, persistence, and the API response.

This design updates only the FastAPI `POST /ask` query-optimization path. Direct
Pipeline calls and the current CLI continue to bypass conversation middleware.

## Selected Approach

Use a dedicated DeepSeek query-rewrite adapter built on the existing generator
client boundary. It shares `DEEPSEEK_API_KEY`, `DEEPSEEK_BASE_URL`, and
`DEEPSEEK_MODEL`, but has a separate prompt, timeout, temperature, and small
output-token limit. The adapter is injected into `LLMQueryRewriter`; neither the
middleware nor the route creates an API client.

This is preferred over injecting the answer generator directly because rewrite
parameters must remain independently configurable. Calling DeepSeek directly
from middleware is rejected because it would couple HTTP orchestration to a
provider and make offline testing harder.

## Request Flow

For every structurally valid request while `QUERY_REWRITE_ENABLED=true`:

```text
POST /ask
  -> QueryOptimizationMiddleware
  -> ConversationStore.get_recent_turns(session_id, limit)
  -> LLMQueryRewriter.rewrite(original_question, history)
  -> DeepSeek rewrite call
  -> request.state.rewritten_query
  -> route
  -> pipeline.ask(question=original_question,
                  retrieval_query=rewritten_query)
  -> Retriever
```

The LLM is called even for an independent question. It decides whether the
question needs rewriting and returns the original question unchanged when it
does not. No rule-based pre-classification runs before the LLM call.

When rewriting is disabled, middleware uses the original question and does not
call DeepSeek.

## Prompt Contract

The rewrite prompt contains at most the configured 3–5 recent turns, ordered
oldest to newest, followed by the current question. Each turn is labeled as user
or assistant content. The prompt instructs the model to:

1. produce a context-independent retrieval query only when history is needed;
2. preserve the user's language, intent, scope, entities, and constraints;
3. return the current question unchanged when it is already independent;
4. never answer the question or add unrequested content;
5. output exactly one plain-text question without Markdown, explanation,
   alternatives, or surrounding quotes.

Prompt and history sizes are bounded. The provider response must be a nonblank,
single-line string within the configured rewrite output limit. Invalid output is
treated as rewrite failure rather than sent to retrieval.

## Provider Integration

The DeepSeek rewrite call reuses the existing provider configuration and HTTP
implementation. Rewrite-specific settings are:

```text
QUERY_REWRITE_PROVIDER=deepseek
QUERY_REWRITE_TIMEOUT=10.0
temperature=0.0
max_tokens=128
```

`QUERY_REWRITE_PROVIDER` supports only `deepseek` for the production path and
defaults to it. API-key validation remains lazy: importing `app.api` must not
require a configured key. The key is read when an enabled rewrite request makes
the provider call.

The provider adapter receives its call dependency through construction so unit
and API tests can use fakes without network access. No new third-party package
is required.

## Results and Failure Behavior

`LLMQueryRewriter` continues returning `QueryRewriteResult`:

```python
QueryRewriteResult(
    original_query=original_question,
    rewritten_query=provider_output_or_original,
    was_rewritten=provider_output_or_original != original_question,
    reason="llm_rewrite" | "llm_unchanged" | "llm_rewrite_failed",
)
```

DeepSeek timeouts, HTTP errors, missing credentials, malformed responses, empty
responses, multiline responses, or adapter exceptions all fall back to the
original question with `was_rewritten=False`. Query rewriting must never prevent
the existing RAG request from continuing. The rule-based rewriter is not used as
a production fallback because the requested behavior is LLM-based.

Failures are logged through the existing bounded logging path without API keys,
authorization headers, full provider payloads, or full conversation history.

## Composition and Boundaries

`app/dependencies.py` creates one process-wide conversation store and one
LLM-based query rewriter. It wires the provider adapter into
`LLMQueryRewriter`. Middleware depends only on `ConversationStore` and
`QueryRewriter`; it remains unaware of DeepSeek.

The Pipeline remains framework- and provider-independent. Retriever receives
only `retrieval_query`; Generator receives the existing grounded-answer prompt
whose user task remains the original question.

The existing `RuleBasedQueryRewriter` may remain as an unused implementation for
compatibility and isolated tests, but it is removed from default composition and
is never called by the DeepSeek production path.

## Testing

Implementation follows test-driven development. Tests must first establish:

- default configuration selects `deepseek` and rejects unsupported providers;
- the DeepSeek adapter builds a bounded prompt containing history and the
  current question and uses rewrite-specific timeout/token settings;
- independent and contextual questions both invoke the injected fake LLM;
- unchanged output produces `was_rewritten=False`;
- rewritten output produces `was_rewritten=True`;
- empty, multiline, malformed, timeout, and provider-error output falls back to
  the original question;
- missing API credentials do not fail module import and do not turn `/ask` into
  a rewrite-originated 500 response;
- disabled rewriting does not call the provider;
- Middleware, API, Pipeline, health, and conversation persistence behavior stay
  compatible.

All automated tests use fake provider calls. They must not make network
requests, download models, connect to MySQL, or require Chroma data. Completion
requires focused tests, the full offline test suite, mock conversation smoke
execution, `git diff --check`, and full diff inspection.

## Compatibility and Limits

- Enabled `/ask` requests incur one additional DeepSeek call before retrieval.
- Rewrite latency and DeepSeek usage cost therefore increase per request.
- A rewrite failure safely degrades to retrieval with the original question.
- Conversation history remains process-local and is lost on restart.
- Multiple worker processes do not share history.
- CLI conversation rewriting remains outside this change and requires a
  separate transport integration if desired.
