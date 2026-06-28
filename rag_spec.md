# RAG System Specification

This document defines the technical design for the minimal RAG system.

It should be followed together with `agents.md`.

---

## 1. System Overview

This is a local Retrieval-Augmented Generation (RAG) system.

The goal is to answer user questions based on local documents.

The system pipeline is:

```text
User Query
  -> Retriever
  -> Relevant document chunks
  -> LLM prompt with context
  -> Generated answer
  -> Answer with citations
```

The first version should prioritize a clear and correct end-to-end pipeline.

---

## 2. Supported Document Formats

### v1

Support:

- `.txt`
- `.md`

### Future versions

Add support for:

- `.pdf`
- `.docx`
- `.html`
- `.csv`

Do not implement future formats unless explicitly requested.

---

## 3. Document Loading

The document loader should read files from:

```text
data/raw/
```

Each loaded document should use this structure:

```python
{
    "content": str,
    "source": str
}
```

Example:

```python
{
    "content": "RAG means Retrieval-Augmented Generation.",
    "source": "rag_notes.md"
}
```

Rules:

- Empty files should be skipped or reported clearly.
- Unsupported file types should be ignored.
- File paths should not be hardcoded inside core functions.
- Metadata must be preserved.

---

## 4. Chunking Strategy

Documents should be split into smaller chunks before embedding.

### Default settings

```text
chunk_size = 800
chunk_overlap = 120
```

These values may be adjusted later.

### Purpose of chunking

Chunking helps the retriever find relevant parts of long documents.

### Purpose of overlap

Overlap prevents important context from being lost at chunk boundaries.

### Chunk format

Each chunk must use this structure:

```python
{
    "content": str,
    "source": str,
    "chunk_id": int
}
```

Example:

```python
{
    "content": "RAG combines retrieval and generation.",
    "source": "rag_notes.md",
    "chunk_id": 0
}
```

---

## 5. Embedding Strategy

Embeddings convert text into numerical vectors.

### v1 recommendation

Use one of the following:

- OpenAI embedding API
- Any OpenAI-compatible embedding API

### Future option

A local embedding model may be added later.

### Rules

- Only embed `chunk["content"]`.
- Do not embed metadata.
- Do not lose the connection between content and metadata.
- Embedding code should be isolated in `embeddings.py`.

---

## 6. Vector Database

### Recommended database

Use ChromaDB for the first version.

### Storage path

```text
chroma_db/
```

This directory should be ignored by Git.

### Stored item format

Each stored item should include:

```python
{
    "document": chunk["content"],
    "embedding": [...],
    "metadata": {
        "source": chunk["source"],
        "chunk_id": chunk["chunk_id"]
    }
}
```

### Responsibilities

The vector database should:

- Store chunks
- Store metadata
- Support similarity search
- Return top-k relevant chunks

---

## 7. Retrieval Strategy

The retriever takes a user question and returns relevant chunks.

### Default settings

```text
top_k = 5
similarity = cosine similarity
```

### Retrieved result format

```python
[
    {
        "content": "...",
        "source": "...",
        "chunk_id": 0,
        "score": 0.82
    }
]
```

Rules:

- Retrieval should not generate the final answer.
- Retrieval should not modify documents.
- Retrieval should preserve source information.
- Retrieval should return an empty list if no result is found.

---

## 8. RAG Prompt Template

The RAG chain should use a strict grounded prompt.

### System prompt

```text
You are a helpful assistant.

Answer the user's question using only the provided context.

If the answer is not contained in the context, say exactly:
"Not found in knowledge base."

Do not use outside knowledge.

Always include sources at the end of the answer.
```

### User prompt template

```text
Context:
{retrieved_context}

Question:
{user_question}

Answer:
```

---

## 9. Answer Format

The final answer should include:

```text
Answer:
...

Sources:
- source_file.md (chunk 0)
- source_file.md (chunk 3)
```

If no useful context is found, the answer should be:

```text
Answer:
Not found in knowledge base.

Sources:
None
```

---

## 10. Ingestion Flow

The ingestion script should be:

```text
scripts/ingest.py
```

It should perform:

```text
Load documents
  -> Split into chunks
  -> Generate embeddings
  -> Store in ChromaDB
```

The script should print useful progress information:

- Number of files loaded
- Number of chunks created
- Number of chunks stored
- Vector database path

---

## 11. Question Answering Flow

The question script should be:

```text
scripts/ask.py
```

It should perform:

```text
Read user question
  -> Retrieve top-k chunks
  -> Build prompt
  -> Call LLM
  -> Print answer and sources
```

---

## 12. API Design

A future FastAPI endpoint may be added.

### Endpoint

```text
POST /ask
```

### Request

```json
{
    "question": "What is RAG?"
}
```

### Response

```json
{
    "answer": "...",
    "sources": [
        {
            "source": "rag_notes.md",
            "chunk_id": 0
        }
    ]
}
```

Do not implement this API until the command-line version works.

---

## 13. UI Design

A future Streamlit UI may be added.

Minimum UI features:

- Text input for question
- Submit button
- Answer display
- Source display
- Optional retrieved context display

Do not build the UI before the command-line version works.

---

## 14. Error Handling

Handle these cases clearly:

| Case | Expected behavior |
|---|---|
| No `.env` file | Tell user to configure environment |
| No API key | Tell user API key is missing |
| `data/raw/` is empty | Tell user to add documents |
| No chunks created | Stop ingestion and report |
| Vector database is empty | Tell user to run ingestion first |
| No retrieval result | Return `Not found in knowledge base.` |
| LLM call fails | Return a clear error message |

---

## 15. Logging

The system should eventually log:

- Loaded files
- Skipped files
- Number of chunks
- Embedding errors
- Vector database writes
- User questions
- Retrieved sources
- LLM errors

Simple `print()` logging is acceptable in the first version.

A proper logger can be added later.

---

## 16. Evaluation Philosophy

The project should be evaluated by:

- Whether the full RAG pipeline works
- Whether retrieved chunks are relevant
- Whether answers are grounded in retrieved context
- Whether sources are correctly preserved
- Whether the code is understandable

The project should not be evaluated by:

- UI beauty
- Number of frameworks
- Amount of code
- Production-level scalability

---

## 17. Future Upgrade Ideas

After v1 works, possible upgrades include:

- PDF support
- Better chunking
- Hybrid search
- Reranking
- Query rewriting
- Multi-query retrieval
- Streaming answers
- Web UI
- Docker deployment
- Evaluation dataset
- Local embedding model
- Local LLM support

Do not implement these until the basic version works.

---

## 18. Version 1 Completion Criteria

v1 is complete when:

1. Documents can be loaded from `data/raw/`.
2. Text can be split into chunks.
3. Chunks can be embedded.
4. Chunks can be stored in ChromaDB.
5. A user question can retrieve relevant chunks.
6. The LLM can answer using retrieved context.
7. The answer includes sources.
8. The project has a clear README.
9. Basic tests pass.

---

End of `rag_spec.md`.
