# mini-rag

A minimal RAG project built from scratch for learning and experimentation.

This project implements a complete basic Retrieval-Augmented Generation pipeline, including document loading, chunking, embedding, vector storage, retrieval, prompt construction, LLM generation, source citation, smoke testing, and a command-line question-answering entrypoint.

## Features

- Load raw documents from `data/raw/`
- Split documents into chunks with metadata preservation
- Generate embeddings with a multilingual SentenceTransformer model
- Store vectors in ChromaDB
- Retrieve relevant contexts by semantic similarity
- Build RAG prompts with context and source constraints
- Generate answers with DeepSeek API
- Append sources to answers as a fallback
- Run real full-chain smoke tests
- Ask questions through a formal CLI entrypoint
- Unit tests for core modules and CLI behavior

## Project Structure

```text
mini-rag/
├── app/
│   ├── chunker.py              # Split documents into chunks
│   ├── config.py               # Project configuration and .env loading
│   ├── document_loader.py      # Load raw text documents
│   ├── embeddings.py           # Embedding model wrapper
│   ├── generator.py            # DeepSeek LLM generator
│   ├── pipeline.py             # Main RAG orchestration layer
│   ├── prompt_builder.py       # Build prompts and handle sources
│   ├── retriever.py            # Retrieve relevant contexts
│   └── vector_store.py         # Chroma vector store wrapper
│
├── data/
│   ├── raw/                    # Raw documents
│   └── chroma/                 # Local Chroma vector database
│
├── models/                     # Optional local embedding models
│
├── scripts/
│   ├── ingest.py               # Build or update vector database
│   ├── query.py                # Basic retrieval query script
│   ├── smoke_pipeline_api.py   # Real API full-chain smoke test
│   └── ask.py                  # Formal CLI question-answering entrypoint
│
├── tests/                      # Unit tests
├── .env.example                # Environment variable template
├── .gitignore
├── README.md
└── requirements.txt
```

## RAG Pipeline

The main workflow is:

```text
Raw Documents
    ↓
Document Loader
    ↓
Chunker
    ↓
Embedder
    ↓
Chroma Vector Store
    ↓
Retriever
    ↓
Prompt Builder
    ↓
DeepSeek Generator
    ↓
RAG Pipeline Result
```

The main runtime entrypoint is `app/pipeline.py`.

`RAGPipeline` connects retrieval, prompt building, generation, and source handling into one stable interface.

## Requirements

Recommended Python version:

```bash
python 3.11
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If you are using a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file in the project root using `.env.example` as a template:

```env
DEEPSEEK_API_KEY=your_deepseek_api_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-flash
DEEPSEEK_TIMEOUT=30
```

The `.env` file is ignored by Git.

The project automatically loads it through `python-dotenv` when `app.config` is imported.

Do not commit your real API key.

## Prepare Documents

Put your raw text documents into:

```text
data/raw/
```

Example:

```text
data/raw/
├── note1.txt
├── note2.txt
└── profile.txt
```

After adding or changing documents, rebuild the vector database.

## Build the Vector Database

Run:

```bash
python scripts/ingest.py
```

This script loads documents from `data/raw/`, chunks them, embeds them, and stores them into ChromaDB under `data/chroma/`.

Important:

```text
scripts/ingest.py updates the vector database.
scripts/ask.py only asks questions.
scripts/ask.py does not automatically ingest new documents.
```

If you modify files in `data/raw/`, run `scripts/ingest.py` again.

## Ask Questions from CLI

Single-question mode:

```bash
python scripts/ask.py "RAG是什么？"
```

Interactive mode:

```bash
python scripts/ask.py
```

In interactive mode, type:

```text
exit
quit
q
```

to exit.

Use a custom retrieval count:

```bash
python scripts/ask.py "RAG是什么？" --top-k 3
```

Show retrieved context for debugging:

```bash
python scripts/ask.py "RAG是什么？" --show-context
```

Hide the extra source section:

```bash
python scripts/ask.py "RAG是什么？" --no-sources
```

## Run Full-Chain Smoke Test

After building the vector database and configuring `.env`, run:

```bash
python scripts/smoke_pipeline_api.py
```

This script verifies the real end-to-end chain:

```text
Embedder
    ↓
ChromaVectorStore
    ↓
Retriever
    ↓
Prompt Builder
    ↓
DeepSeekGenerator
    ↓
RAGPipeline
```

The smoke script is only for manual verification.

It does not automatically ingest documents, rebuild the database, delete the database, or hardcode API keys.

## Run Tests

Run all tests:

```bash
python -m pytest
```

Run a specific test file:

```bash
python -m pytest tests/test_ask_cli.py
```

The CLI tests use fake pipelines and do not call the real DeepSeek API or real Chroma vector database.

## Git Ignore Policy

The following files or directories should not be committed:

```text
.venv/
.env
__pycache__/
.pytest_cache/
data/chroma/
data/raw/private/
models/
```

Recommended `.gitignore` entries:

```gitignore
.venv/
.env
__pycache__/
.pytest_cache/
data/chroma/
data/raw/private/
models/
```

## Common Commands

Install dependencies:

```bash
pip install -r requirements.txt
```

Build vector database:

```bash
python scripts/ingest.py
```

Ask one question:

```bash
python scripts/ask.py "你的问题"
```

Start interactive question answering:

```bash
python scripts/ask.py
```

Run real API smoke test:

```bash
python scripts/smoke_pipeline_api.py
```

Run tests:

```bash
python -m pytest
```

## Troubleshooting

### Missing DeepSeek API key

If you see an error about `DEEPSEEK_API_KEY`, check that your `.env` file exists in the project root:

```text
mini-rag/.env
```

and contains:

```env
DEEPSEEK_API_KEY=your_deepseek_api_key_here
```

### Vector database is missing or empty

If the system says no context was found or asks you to run ingest, run:

```bash
python scripts/ingest.py
```

Then ask again:

```bash
python scripts/ask.py "你的问题"
```

### New documents are not reflected in answers

After changing files in `data/raw/`, run:

```bash
python scripts/ingest.py
```

The query scripts do not update the vector database automatically.

### Import error: No module named app

Make sure you run commands from the project root:

```bash
cd mini-rag
python scripts/ask.py "RAG是什么？"
```

## Current Status

Implemented:

- Document loading
- Text chunking
- Embedding generation
- Chroma vector storage
- Semantic retrieval
- Prompt building
- DeepSeek generation
- RAG pipeline orchestration
- Source citation fallback
- Real API smoke test
- Formal CLI question-answering entrypoint
- Unit tests

Not implemented yet:

- Web frontend
- FastAPI backend service
- Evaluation dataset
- Reranking
- Hybrid retrieval
- Multi-turn conversation memory
- Docker deployment
- CI pipeline

## Roadmap

Possible next steps:

```text
1. Add a Streamlit frontend
2. Add a FastAPI /ask endpoint
3. Add evaluation questions and an eval script
4. Add logging and debug mode
5. Add query rewriting
6. Add reranking
7. Add hybrid search
8. Add Docker support
9. Add GitHub Actions CI
```

## Purpose

This project is designed as a learning-oriented mini RAG system.

The goal is not to rely on a high-level RAG framework immediately, but to understand and implement the core components manually:

```text
loading → chunking → embedding → vector storage → retrieval → prompting → generation → sources → user entrypoint
```
