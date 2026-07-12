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
- Expose `GET /health` and `POST /ask` through FastAPI
- Unit tests for core modules and CLI behavior

## Project Structure

```text
mini-rag/
├── app/
│   ├── chunker.py              # Split documents into chunks
│   ├── api.py                  # FastAPI HTTP service layer
│   ├── config.py               # Project configuration and .env loading
│   ├── document_loader.py      # Load raw text documents
│   ├── embeddings.py           # Embedding model wrapper
│   ├── generator.py            # DeepSeek LLM generator
│   ├── pipeline.py             # Main RAG orchestration layer
│   ├── pipeline_factory.py     # Build the default RAG pipeline
│   ├── schemas.py              # API request and response models
│   ├── prompt_builder.py       # Build prompts and handle sources
│   ├── retriever.py            # Retrieve relevant contexts
│   ├── reranker.py             # Optional Cross-Encoder second-stage reranking
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
Cross-Encoder Reranker (optional)
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

## Reranking

The first retrieval stage uses BM25 and vector retrieval to recall candidates quickly, then hybrid retrieval normalizes, fuses, and deduplicates them. The optional second stage uses `cross-encoder/ms-marco-TinyBERT-L2-v2` to jointly encode each query and candidate document, produce a raw relevance score, and reorder only those recalled candidates.

The default flow is:

```text
Recall and fuse 10 candidates
→ TinyBERT Cross-Encoder reranking
→ Keep 5 documents for the prompt
```

Configure it in `.env`:

```env
RERANKER_ENABLED=true
RERANKER_MODEL=cross-encoder/ms-marco-TinyBERT-L2-v2
RERANKER_TOP_K=5
RERANKER_CANDIDATE_K=10
RERANKER_BATCH_SIZE=16
RERANKER_MAX_LENGTH=256
RERANKER_DEVICE=cpu
RERANKER_FAILURE_MODE=fallback
RERANKER_LOCAL_FILES_ONLY=false
```

If `RERANKER_MODEL` is a Hugging Face model name, the first real reranking call may download the TinyBERT model. Later runs reuse the Hugging Face cache. To use a model downloaded into this project instead:

```env
RERANKER_MODEL=./models/ms-marco-TinyBERT-L2-v2
RERANKER_LOCAL_FILES_ONLY=true
```

The model is loaded lazily and reused by the pipeline. If loading or inference fails, the pipeline logs the failure, keeps the original fused order, applies the final Top-K limit, and continues generation. Set `RERANKER_ENABLED=false` to retain a no-reranker baseline. Any `sentence-transformers.CrossEncoder`-compatible model can be selected through `RERANKER_MODEL` without changing Pipeline code.

TinyBERT is a lightweight Cross-Encoder suitable for CPU environments, but Cross-Encoder inference still adds query latency. Reduce `RERANKER_CANDIDATE_K`, `RERANKER_MAX_LENGTH`, or `RERANKER_BATCH_SIZE` when lower latency is more important.

`cross-encoder/ms-marco-TinyBERT-L2-v2` is primarily trained on English MS MARCO data. It is appropriate for lightweight engineering verification and English retrieval scenarios. Projects dominated by Chinese documents should validate it with an independent evaluation set and retain the ability to switch to a Chinese or multilingual reranker. This project does not claim an unmeasured Chinese retrieval improvement.

Unit tests use fake models and do not download TinyBERT:

```bash
python -m pytest tests/test_reranker.py -q
python -m pytest -q
```

## Conversation Memory and Query Rewriting

`POST /ask` uses `QueryOptimizationMiddleware` before the RAG pipeline. The
middleware loads the requested session's recent completed turns and calls the
configured DeepSeek model once to produce a retrieval-ready question. The LLM
returns an independent rewrite when the question depends on history and returns
the original question when no rewrite is needed. The original question is
preserved for the final prompt, response, and request log; only
`rewritten_query` is passed to retrieval and reranking.

Every `/ask` request requires a nonblank `session_id` of at most 128
characters. Reuse the same value for follow-up questions:

```bash
curl -X POST "http://127.0.0.1:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "demo-user-001",
    "question": "什么是 Middleware？"
  }'
```

Second round:

```bash
curl -X POST "http://127.0.0.1:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "demo-user-001",
    "question": "它为什么可以优化查询？"
  }'
```

Conversation history currently uses `InMemoryConversationStore` and keeps the
latest five completed turns per session by default. History is lost when the
process restarts and is not shared between multiple worker processes. A future
MySQL implementation can implement the stable `ConversationStore` interface
without changing middleware or API orchestration.

Configuration:

```env
CONVERSATION_HISTORY_LIMIT=5
QUERY_REWRITE_ENABLED=true
QUERY_REWRITE_PROVIDER=deepseek
QUERY_REWRITE_TIMEOUT=10.0
```

`CONVERSATION_HISTORY_LIMIT` must be between 3 and 5. Disabling rewriting keeps
the session contract but sends the original question to retrieval without an
extra DeepSeek call. When rewriting is enabled, each valid `/ask` request adds
one DeepSeek API call and therefore adds latency and provider usage. A missing
key, timeout, provider error, or invalid rewrite output safely falls back to the
original question instead of failing the RAG request.

The current CLI entrypoints call `RAGPipeline` directly and do not pass through
the FastAPI middleware, so their questions are not conversation-rewritten.

Run focused and complete offline tests with the project virtual environment:

```bash
.venv/bin/python -m pytest tests/test_conversation_memory_store.py -q
.venv/bin/python -m pytest tests/test_query_rewriter.py -q
.venv/bin/python -m pytest tests/test_query_optimization_middleware.py -q
.venv/bin/python -m pytest tests/test_api_conversation.py -q
.venv/bin/python -m pytest -q
```

Run the deterministic conversation smoke check without an API key, model
download, vector database, or network access:

```bash
.venv/bin/python scripts/smoke_conversation.py --mock
```

An opt-in real-model check uses English retrieval examples:

```bash
python -m scripts.smoke_reranker
```

Future model comparisons can disable reranking for a baseline and retain both the original retrieval scores and successful `rerank_score` values. A production evaluation should compare no reranker, TinyBERT, and relevant Chinese or multilingual models using Hit Rate@K, Recall@K, MRR@K, NDCG@K, mean rerank latency, and P95 rerank latency. No evaluation numbers are fabricated here.

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


## Ask Questions from FastAPI

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the local API service:

```bash
uvicorn app.api:app --reload
```

Open the generated API docs:

```text
http://127.0.0.1:8000/docs
```

Check service health:

```bash
curl http://127.0.0.1:8000/health
```

Ask a question:

```bash
curl -X POST "http://127.0.0.1:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "你的问题"}'
```

The API default `top_k` is `5`, matching the CLI and reranker default. You can override it per request:

```json
{
  "question": "RAG是什么？",
  "top_k": 3
}
```

Important:

```text
POST /ask only answers from the existing vector database.
After adding, deleting, or changing files in data/raw/, run python scripts/ingest.py again.
GET /health only means the API service started. It does not verify the DeepSeek API key or vector database data.
```

## Run Tests

Run all tests:

```bash
python -m pytest
```

Run a specific test file:

```bash
python -m pytest tests/test_ask_cli.py
python -m pytest tests/test_api.py
```

The CLI and API tests use fake pipelines and do not call the real DeepSeek API or real Chroma vector database.

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
- FastAPI backend service
- Unit tests

Not implemented yet:

- Web frontend
- Evaluation dataset
- Reranking
- Hybrid retrieval
- Docker deployment
- CI pipeline

## Roadmap

Possible next steps:

```text
1. Add a Streamlit frontend
2. Add evaluation questions and an eval script
3. Add logging and debug mode
4. Add Docker support
5. Add GitHub Actions CI
```

## Purpose

This project is designed as a learning-oriented mini RAG system.

The goal is not to rely on a high-level RAG framework immediately, but to understand and implement the core components manually:

```text
loading → chunking → embedding → vector storage → retrieval → prompting → generation → sources → user entrypoint
```


## Request Logs

`POST /ask` appends one JSON object per request to `logs/rag_requests.jsonl`.
The file is ignored by Git because it may contain user questions and model answers.

A synthetic success and error example is available at
`docs/example_rag_request_logs.jsonl`.

View recent local logs in PowerShell:

```powershell
Get-Content logs/rag_requests.jsonl
```

## BM25 Retrieval

BM25 is a sparse retrieval algorithm that ranks document chunks by keyword relevance. It runs independently from the existing embedding and Chroma-based dense retrieval pipeline.

The project currently supports:

- Dense Retrieval (Embedding + Chroma)
- Sparse Retrieval (BM25)

Run the standalone BM25 smoke test from the project root:

```bash
python scripts/smoke_bm25.py
```

The smoke test uses local in-memory chunks and does not call the API, an LLM, or ChromaDB.
