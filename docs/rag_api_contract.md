# Enterprise RAG Engine HTTP API Contract

This document freezes the HTTP contract consumed by
`Agentic-Enterprise-Knowledge-Copilot`.

## Health

### Request

```http
GET /health
```

### Response

```json
{
  "status": "ok",
  "service": "mini-rag-api",
  "version": "0.1.0"
}
```

The response conforms to `HealthResponse`.

## Ask

### Request

```http
POST /ask
Content-Type: application/json
X-Trace-ID: copilot-request-123
```

`X-Trace-ID` is optional. When it is present and non-empty, the service preserves
it in `rag_trace_id`. When it is absent or blank, the service generates a UUID.

The JSON body contains exactly one field:

```json
{
  "question": "What is retrieval-augmented generation?"
}
```

| Field | Type | Required | Description |
|---|---|---:|---|
| `question` | string | yes | Non-blank question. Surrounding whitespace is removed. |

Unknown request fields are rejected.

### Response

```json
{
  "answer": "Retrieval-augmented generation grounds an answer in retrieved evidence.",
  "sources": [
    {
      "index": 1,
      "source": "docs/rag.md",
      "metadata": {
        "chunk_id": "rag-1"
      },
      "text_preview": "Retrieval-augmented generation..."
    }
  ],
  "contexts": [
    {
      "content": "Retrieval-augmented generation...",
      "source": "docs/rag.md",
      "chunk_id": "rag-1",
      "score": 0.92,
      "metadata": {
        "chunk_id": "rag-1"
      }
    }
  ],
  "route": "rag",
  "latency_ms": 12,
  "rag_trace_id": "copilot-request-123"
}
```

| Field | Type | Required | Description |
|---|---|---:|---|
| `answer` | string | yes | Grounded answer returned by the RAG pipeline. |
| `sources` | array of `Source` | yes | Normalized sources derived from pipeline output. |
| `contexts` | array of `Context` | yes | Retrieved contexts used by the pipeline. |
| `route` | literal `"rag"` | yes | Stable route identifier for this API. |
| `latency_ms` | number | yes | Server-side request latency in milliseconds. |
| `rag_trace_id` | string | yes | Preserved `X-Trace-ID` or a generated UUID. |

No response fields outside this schema are returned.

### Source

| Field | Type | Required |
|---|---|---:|
| `index` | integer or null | yes |
| `source` | string or null | yes |
| `metadata` | object or null | yes |
| `text_preview` | string or null | yes |

### Context

| Field | Type | Required |
|---|---|---:|
| `content` | string | yes |
| `source` | string or null | yes |
| `chunk_id` | string or null | yes |
| `score` | number or null | yes |
| `metadata` | object or null | yes |
