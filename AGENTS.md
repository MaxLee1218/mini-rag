# RAG Project Agent Rules (Codex Guidance)

This file defines strict rules for any AI coding agent, such as Codex or GPT, working in this repository.

The goal is to ensure a minimal, correct, and modular RAG system for learning purposes.

---

## 1. Project Goal

This project is a minimal Retrieval-Augmented Generation (RAG) system.

It must:

- Load local documents
- Support `.txt` and `.md` in v1
- Add `.pdf` support later
- Split documents into chunks
- Generate embeddings for chunks
- Store embeddings in a vector database
- Retrieve relevant chunks for a user query
- Generate answers using an LLM with grounded context
- Return answers with source citations

---

## 2. Non-Goals for v1

Do not implement the following in the first version:

- Authentication or user accounts
- Multi-user systems
- Cloud deployment
- Distributed systems
- Web scraping or crawling
- Complex frontend frameworks such as React
- Agent frameworks beyond basic RAG
- Complex database systems
- Microservice architecture

The first version should focus only on a working end-to-end RAG pipeline.

---

## 3. System Architecture

All implementations must follow this pipeline:

```text
document_loader
    -> text_splitter
    -> embeddings
    -> vector_store
    -> retriever
    -> rag_chain
    -> ui or api layer
```

No module should bypass this pipeline.

---

## 4. File Responsibility Rules

Each file must have a single responsibility.

### `document_loader.py`

Responsible only for:

- Loading raw files from disk
- Reading supported document formats
- Returning structured document objects

It must not:

- Split text
- Generate embeddings
- Store vectors
- Call the LLM

---

### `text_splitter.py`

Responsible only for:

- Splitting documents into chunks
- Preserving metadata such as source and chunk ID

It must not:

- Read files directly
- Generate embeddings
- Store vectors
- Call the LLM

---

### `embeddings.py`

Responsible only for:

- Converting text into embeddings
- Wrapping the embedding model or embedding API

It must not:

- Read files
- Split documents
- Store vectors
- Perform retrieval
- Call the chat LLM

---

### `vector_store.py`

Responsible only for:

- Creating or loading the vector database
- Storing documents, embeddings, and metadata
- Querying similar vectors

It must not:

- Read raw files
- Split documents
- Decide the final answer
- Call the chat LLM

---

### `retriever.py`

Responsible only for:

- Taking a user query
- Searching the vector database
- Returning top-k relevant chunks

It must not:

- Generate the final answer
- Modify source documents
- Rebuild the entire database unless explicitly asked

---

### `rag_chain.py`

Responsible only for:

- Combining retrieval and LLM prompting
- Building the final context
- Calling the LLM
- Returning an answer with sources

It must not:

- Load raw files
- Split documents
- Directly manipulate the vector database internals

---

### `main.py`

Responsible only for:

- API entry point
- Connecting external requests to the RAG chain

It must not contain core RAG logic.

---

## 5. Data Format Rules

All loaded documents must use this structure:

```python
{
    "content": "document text here",
    "source": "filename.md"
}
```

All chunks must use this structure:

```python
{
    "content": "text chunk here",
    "source": "filename.md",
    "chunk_id": 0
}
```

Retrieved chunks should use this structure:

```python
{
    "content": "retrieved text chunk",
    "source": "filename.md",
    "chunk_id": 0,
    "score": 0.82
}
```

Metadata must never be lost.

---

## 6. Coding Rules

Follow these rules strictly:

- Keep functions small and testable.
- Avoid unnecessary global variables.
- Do not mix responsibilities across modules.
- Return structured data such as dictionaries or lists.
- Do not return unstructured raw strings when structured output is needed.
- Do not hardcode file paths inside core logic functions.
- Prefer explicit function inputs and outputs.
- Prefer simple code over clever code.
- Write readable names for functions, variables, and classes.
- Avoid over-engineering.

---

## 7. Testing Rules

When adding or changing a module:

- Add or update tests when appropriate.
- Test one module at a time.
- Prefer small unit tests.
- Do not rely on external APIs in basic unit tests unless explicitly required.
- Use sample files under `tests/fixtures/` when needed.

Minimum tests should cover:

- Document loading
- Text splitting
- Metadata preservation
- Retrieval output format
- RAG answer output format

---

## 8. RAG Answering Rules

When generating answers:

- Answer only using retrieved context.
- Do not hallucinate missing facts.
- If the context is insufficient, respond exactly:

```text
Not found in knowledge base.
```

- Always include sources used in the answer.
- Do not cite sources that were not retrieved.
- Do not pretend a source contains information that it does not contain.

---

## 9. Forbidden Behaviors

The coding agent must not:

- Merge all logic into one file.
- Skip the vector database step.
- Invent fake APIs.
- Write pseudo-code instead of real implementation.
- Add unnecessary frameworks.
- Over-engineer the system.
- Implement features not explicitly requested.
- Refactor unrelated files without permission.
- Delete files unless explicitly asked.
- Store API keys in source code.
- Commit `.env` files or secrets.

---

## 10. Configuration Rules

Configuration should be centralized.

Use `app/config.py` for:

- API key loading
- Model names
- Chunk size
- Chunk overlap
- Top-k retrieval setting
- Vector database path

Secrets must be loaded from environment variables or `.env`.

Never hardcode API keys.

---

## 11. Development Workflow

When implementing features:

1. Modify one module at a time.
2. Write minimal working code first.
3. Test locally before expanding.
4. Do not refactor unrelated files.
5. Keep changes incremental.
6. Make code understandable for a beginner.
7. Explain important design decisions in comments only when necessary.

---

## 12. Debugging Rules

If something breaks:

1. Inspect the error message.
2. Identify which stage failed.
3. Verify the data flow step by step:

```text
loader -> splitter -> embeddings -> vector_store -> retriever -> rag_chain
```

Do not randomly rewrite the entire system.

---

## 13. Priority Rule

Use this priority order:

```text
Correctness > Simplicity > Readability > Performance > Features
```

---

## 14. Codex Instruction Template

When asking Codex to work on this repository, use this style:

```text
Follow agents.md and rag_spec.md strictly.

Implement only the requested module.
Do not modify unrelated files.
Keep the implementation simple and beginner-friendly.
Add or update tests if needed.
```

---

End of `agents.md`.
