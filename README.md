# Enterprise RAG Engine

Production-oriented Retrieval-Augmented Generation system for enterprise knowledge management.

**An AI-powered knowledge assistant that retrieves, grounds, and generates answers from enterprise documents.**

Enterprise RAG Engine is an enterprise knowledge intelligence system powered by Retrieval-Augmented Generation. It provides a modular foundation for building an Enterprise Knowledge Base Assistant across quality, engineering, procurement, and operations workflows while keeping source provenance and measurable RAG quality at the center of the system.

## Problem

Traditional enterprise knowledge systems often make critical information difficult to use:

- policies, procedures, and technical knowledge are scattered across PDFs and other documents;
- manual search depends on exact wording and requires employees to inspect multiple files;
- slow knowledge retrieval delays operational and engineering decisions;
- inconsistent interpretations lead to inconsistent answers across teams.

Enterprise RAG Engine combines semantic retrieval, keyword retrieval, document grounding, and LLM generation. It finds evidence by both meaning and terminology, builds a bounded prompt from retrieved content, and returns an answer with traceable sources.

## Use Case

### Enterprise Quality Management Knowledge Base

The target scenario is an internal knowledge assistant that helps manufacturing engineers, quality engineers, procurement teams, and operations teams query controlled enterprise knowledge.

Representative data sources include the following document groups.

#### 1. SOP Documents

Examples:

- `Supplier Quality Manual.pdf`
- `Incoming Inspection Procedure.pdf`
- `Production Quality Standard.pdf`
- `ISO9001 Procedure.pdf`

Suitable source material can come from ISO public documents, publicly available enterprise quality manuals, appropriately licensed Kaggle datasets, and synthetic enterprise documents. Verify licensing, confidentiality, and data-handling requirements before ingestion.

#### 2. Technical Documentation

Examples:

- Machine Maintenance Manual
- Failure Analysis Report
- Engineering Specification

#### 3. Enterprise Policies

Examples:

- Procurement Policy
- Risk Management Guideline
- Safety Regulation

## Example Query

The following illustrates the expected grounded-answer experience for a quality-management knowledge base. The named document and page are demo content, not a claim about files included in this repository.

**User**

> What is the procedure when supplier quality deviation occurs?

**System answer**

> According to the Supplier Quality Manual:
>
> 1. Create a deviation report.
> 2. Perform root cause analysis.
> 3. Submit a supplier corrective action request.
>
> **Source:** Supplier Quality Manual.pdf, page 24

The system generates the answer from retrieved enterprise documents and derives source references from the retrieved contexts. If the available evidence is insufficient, the grounded generation contract requires the system to abstain instead of fabricating an answer.

## Architecture

Enterprise RAG Engine remains a modular monolith with explicit boundaries between ingestion, retrieval, generation, orchestration, and transport.

```text
Enterprise Documents
        ↓
Document Processing
        ↓
Chunking
        ↓
Embedding + BM25 Index
        ↓
Hybrid Retrieval
        ↓
Optional Reranking
        ↓
Prompt Construction
        ↓
LLM Generation
        ↓
Answer + Sources
```

The current system supports dense retrieval, sparse retrieval, hybrid search with score normalization, optional Cross-Encoder reranking, citation generation, and an offline evaluation pipeline. Query execution reads existing indexes; document ingestion remains an explicit offline operation.

## Features

### Document Intelligence

- PDF, DOCX, Markdown, and TXT ingestion
- recursive document discovery with configurable extensions
- standard and optional parent-child chunking
- metadata and source-provenance preservation
- persistent Chroma vectors and SQLite parent-chunk lookup

### Retrieval System

- embedding-based semantic retrieval with Sentence Transformers
- BM25 keyword retrieval for exact enterprise terminology
- normalized hybrid retrieval and deterministic score fusion
- configurable retrieval depth and result deduplication
- optional Cross-Encoder reranking with safe fallback behavior

### Generation

- context-grounded generation through the DeepSeek API
- source citation and source-normalization support
- bounded prompt construction with abstention instructions
- hallucination reduction through evidence-only prompting
- FAQ fast path for maintained answers and a RAG deep path for complex questions
- conversation-aware query rewriting with safe fallback to the original query

### Evaluation

- versioned offline evaluation dataset
- retrieval hit rate and abstention evaluation
- RAGAS faithfulness, answer relevance, context precision, and context recall
- per-stage latency tracing with p50 and p95 analysis
- structured JSON and human-readable Markdown reports
- bad-case export and resolved-case import workflow

### API Service

- FastAPI backend
- `GET /health` health endpoint
- `POST /ask` REST API endpoint
- request validation with stable response schemas
- structured operational logging and request correlation

## Technical Stack

| Component | Technology |
|---|---|
| Language | Python 3.11 |
| API | FastAPI |
| Vector Database | ChromaDB |
| Embedding | Sentence Transformers |
| Sparse Retrieval | BM25 |
| LLM | DeepSeek API |
| Evaluation | RAGAS |
| Testing | Pytest |

## Project Structure

```text
Enterprise-RAG-Engine/
├── app/
│   ├── conversation/          # Conversation models and in-memory history
│   ├── faq/                   # FAQ fast path, SQLite repository, and cache
│   ├── middleware/            # Query optimization orchestration
│   ├── query_rewriter/        # Rule-based and LLM query rewriting
│   ├── utils/                 # Shared retrieval utilities
│   ├── api.py                 # FastAPI transport layer
│   ├── chunker.py             # Standard and parent-child chunking
│   ├── document_loader.py     # PDF, DOCX, Markdown, and TXT loading
│   ├── embeddings.py          # Sentence Transformer embedding wrapper
│   ├── generator.py           # DeepSeek generation adapter
│   ├── hybrid_retriever.py    # Dense and sparse score fusion
│   ├── pipeline.py            # Core RAG orchestration
│   ├── pipeline_factory.py    # Configured component composition
│   ├── prompt_builder.py      # Grounded prompt and source construction
│   ├── reranker.py            # Optional Cross-Encoder reranking
│   ├── retriever.py           # Vector retrieval
│   └── vector_store.py        # ChromaDB persistence adapter
├── scripts/
│   ├── ingest.py              # Explicit document ingestion
│   ├── query.py               # Retrieval diagnostics
│   ├── ask.py                 # Command-line question answering
│   └── smoke_pipeline_api.py  # Opt-in full-chain smoke test
├── eval/
│   └── run_eval.py            # Offline evaluation entrypoint
├── evaluation/                # Dataset, metrics, tracing, and reporting
├── tests/                     # Offline unit and integration tests
├── data/
│   ├── raw/                   # Source documents
│   ├── chroma/                # Local vector index
│   └── parents/               # Parent-chunk SQLite store
├── reports/                   # Generated evaluation reports
├── .env.example               # Configuration template
├── rag_spec.md                # RAG contracts and quality requirements
├── requirements.txt
└── README.md
```

## Installation

### 1. Clone the repository

The destination argument gives the existing remote repository the new local project directory name:

```bash
git clone https://github.com/MaxLee1218/mini-rag.git Enterprise-RAG-Engine
cd Enterprise-RAG-Engine
```

### 2. Create a virtual environment and install dependencies

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure the environment

Copy the provided template and keep the resulting `.env` file out of version control:

```bash
cp .env.example .env
```

At minimum, configure the DeepSeek provider:

```env
DEEPSEEK_API_KEY=your_deepseek_api_key
DEEPSEEK_MODEL=deepseek-v4-flash
```

Additional retrieval, reranking, parent-child chunking, FAQ, conversation, and evaluation settings are documented in `.env.example`. Never commit real credentials or private enterprise documents.

## Run the System

### Ingest Documents

Place supported documents under `data/raw/`, then build or update the configured indexes explicitly:

```bash
python scripts/ingest.py
```

To preview document processing without writing embeddings or indexes:

```bash
python scripts/ingest.py --dry-run
```

Changing embedding models, dimensions, chunking rules, or parent-child settings can invalidate persisted indexes. Treat rebuilds as controlled operations and use `--reset` only when the affected data is understood.

### Query from the CLI

Ask one question:

```bash
python scripts/ask.py "What is the incoming inspection procedure?"
```

Or start interactive mode:

```bash
python scripts/ask.py
```

For retrieval-only diagnostics:

```bash
python scripts/query.py --query "supplier corrective action"
```

### Start the API

```bash
uvicorn app.api:app --reload
```

Check process health:

```bash
curl http://127.0.0.1:8000/health
```

Ask a question:

```bash
curl -X POST "http://127.0.0.1:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "quality-demo-001",
    "question": "What is the procedure when supplier quality deviation occurs?"
  }'
```

The RAG route requires a populated vector index and a valid DeepSeek configuration. The optional FAQ route uses SQLite as its persistent source of truth and can use Redis as a non-critical positive-match cache.

## Evaluation

Enterprise RAG quality must be measured at retrieval, generation, and operational boundaries rather than judged from isolated demo answers.

The offline evaluation pipeline supports:

- **Retrieval Hit Rate:** whether answerable questions retrieve known reference evidence;
- **Faithfulness:** whether generated claims are supported by retrieved contexts;
- **Answer Relevance:** whether the answer addresses the question;
- **Context Precision:** whether highly ranked contexts are useful;
- **Context Recall:** whether the retrieved contexts contain the required evidence;
- **Latency p50/p95:** typical and tail latency for embedding, retrieval, generation, and total execution.

Run evaluation explicitly after configuring the provider and ingesting the target corpus:

```bash
python eval/run_eval.py
```

No production or portfolio benchmark is claimed until the enterprise demo corpus and evaluation protocol are finalized.

| Metric | Score |
|---|---:|
| Hit Rate | TBD |
| Faithfulness | TBD |
| Answer Relevance | TBD |
| Context Precision | TBD |
| Context Recall | TBD |
| Latency p50/p95 | TBD |

## Engineering Design

### Why Hybrid Retrieval?

Dense retrieval captures semantic meaning and can match questions whose wording differs from the source material. BM25 captures exact enterprise terminology such as part numbers, policy names, procedure codes, defect categories, and engineering abbreviations. Normalizing and fusing both result sets improves coverage without confusing vector distance, lexical score, and fused relevance score semantics.

### Why Parent-Child Chunking?

Small child chunks improve retrieval precision because each embedding represents a focused passage. Larger parent chunks preserve the surrounding context needed for grounded generation. In parent-child mode, the system retrieves child embeddings, resolves the linked parent from SQLite, and sends the parent context to the prompt while retaining source provenance.

### Why an Evaluation Pipeline?

Enterprise RAG requires measurable quality, not only functional demos. Separating retrieval metrics, generation metrics, abstention behavior, bad-case analysis, and stage latency makes failures diagnosable and allows changes to chunking, retrieval, prompts, or models to be compared against a baseline.

## Roadmap

### Completed

- [x] Modular RAG pipeline
- [x] FastAPI service
- [x] Structured logging
- [x] Hybrid retrieval
- [x] Optional reranking
- [x] Parent-child chunking
- [x] Offline evaluation pipeline
- [x] Query rewriting
- [x] Conversation memory

### Future

- [ ] Online feedback loop
- [ ] Production Docker deployment
- [ ] Cloud deployment reference architecture

Roadmap items will be introduced only when concrete quality, scale, security, or operational requirements justify them.

## Testing

Unit and integration tests use injected or mocked external dependencies and do not require live provider calls:

```bash
pytest
```

Real-provider and full-chain smoke tests remain explicit and opt-in.

## GitHub Repository Description

> Enterprise RAG Engine - A production-oriented Retrieval-Augmented Generation system for enterprise knowledge management.
