# RoboSathi RAG Engine 🤖

A local-first, production-ready RAG (Retrieval-Augmented Generation) system that answers questions from your document library with verified citations. Runs on your machine — no cloud dependency required.

---

## What It Does

Upload PDFs, HTML files, Markdown docs, or source code. Ask questions in natural language. The engine retrieves the most relevant passages, generates a cited answer, and validates every claim against the source material.

**Example:**
```
You: Explain Viktor Frankl's concept of the meaning of life.
Engine: Frankl argues that meaning is not abstract but unique to each individual...
[Doc-0, p. 97] [Doc-0, p. 121]
```

---

## Key Features

| Feature | Description |
|---|---|
| **Query Router** | Classifies each input — greetings and small talk bypass the vector store entirely, saving compute |
| **Multi-format Ingestion** | PDF, HTML, Markdown, plain text, source code — parse and chunk them all |
| **Semantic Chunking** | Splits documents at natural topic boundaries using sentence embedding distance spikes |
| **Hybrid Search** | Combines BM25 keyword matching + dense vector cosine similarity via RRF fusion |
| **Cross-Encoder Reranking** | Neural reranker re-scores top candidates for maximum relevance |
| **Citation Guardrails** | Validates every `[Doc-X, p. Y]` reference and checks claim faithfulness |
| **Self-Correction Loop** | Up to 3 rewrite attempts if the answer fails faithfulness checks |
| **Citation Map** | Parses citations in the answer and maps them back to source filenames and pages |
| **LLM Metrics** | Tracks per-call token usage and timing for every LLM interaction |
| **Persistent Vector Store** | Optional ChromaDB mode with file-change tracking via SHA-256 ledger |
| **Deduplication** | Auto-removes duplicate chunks on startup; manual `/dedup` command available |
| **Conversation Memory** | Sliding-window chat history for multi-turn context |
| **RAGAS Evaluation** | Offline quality metrics: faithfulness, relevancy, precision, recall |
| **3 Interfaces** | CLI REPL, Streamlit dashboard, FastAPI REST API |

---

## Quick Start

### Prerequisites

- Python 3.12+
- `uv` package manager ([install guide](https://docs.astral.sh/uv/))

### Install

```bash
uv sync
```

### Run

```bash
# CLI (interactive terminal)
uv run python cli_app.py

# Streamlit dashboard
uv run streamlit run ui_app.py

# FastAPI server
uv run uvicorn api:app --reload --port 8000
```

### Configure

Edit `config/rag_config.toml` for RAG parameters:
- `[chunking]` — max chunk size, semantic threshold
- `[retrieval]` — top-N results, RRF weights, HyDE toggle
- `[generation]` — LLM temperature, max tokens
- `[guardrails]` — max self-correction attempts
- `[vector_store]` — mode (`in-memory` or `persist`), ChromaDB path
- `[logging]` — log level (`DEBUG`, `INFO`, `WARNING`, `ERROR`)

Edit `config/llm_config.toml` for LLM providers.

Set API keys as lowercase environment variables:
```bash
export nvidia_api_key="nvapi-..."
export openai_api_key="sk-..."
# (all lowercase: hf_api_key, gemini_api_key, anthropic_api_key, etc.)
```

### Ingest Documents

```bash
# CLI
/ingest ./input_files/my_book.pdf

# API
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"path": "./input_files/my_book.pdf"}'
```

### Query

```bash
# CLI
User > Explain the concept of meaning of life as per the uploaded texts.

# API
curl http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Explain the concept of meaning of life"}'
```

---

## Architecture Overview

```
User Input
    │
    ▼
QueryRouter ──► DIRECT_LLM (greetings, small talk → direct LLM)
    │
    ▼ RAG_RETRIEVAL
RAGCoreEngine.search()
    ├── BM25 sparse retrieval
    ├── Dense vector cosine similarity
    ├── RRF fusion
    └── Cross-encoder reranking
    │
    ▼
GuardrailsManager.generate_faithful_answer()
    ├── Generate answer with [Doc-X, p. Y] citations
    ├── Validate citations
    ├── Check faithfulness (JSON-based)
    └── Self-correct up to 3 attempts
    │
    ▼
Output: cited answer + citation map + metrics
```

### Project Structure

```
rag_agent_v2/
├── config/                    # TOML configuration files
│   ├── llm_config.toml        # LLM providers & model settings
│   └── rag_config.toml        # RAG pipeline parameters
├── input_files/               # Place your documents here
├── local_models/              # Auto-downloaded embedding/reranker models
├── logs/                      # Query telemetry logs
├── chroma_db/                 # Persistent vector store (auto-created in persist mode)
├── rag_engine/
│   ├── __init__.py            # Module exports + ColoredFormatter + configure_logging
│   ├── cli.py                 # CLI REPL loop (/help, /ingest, /model, /dedup, /clean-db)
│   ├── core.py                # Parsers, semantic chunker, hybrid search, vector store ops
│   ├── evaluation.py          # RAGAS offline evaluation suite
│   ├── guardrails.py          # Citation validation, faithfulness check, self-correction
│   ├── ingestion.py           # IngestionCoordinator: directory scanning, SHA-256 tracking
│   ├── llm.py                 # LiteLLM client with multi-provider cloud fallback
│   ├── memory.py              # ConversationMemory with sliding window
│   ├── metrics.py             # LLMMetricsCollector: per-call token/time tracking
│   ├── prompts.py             # All system/generation prompt templates
│   ├── query_router.py        # QueryRouter: classifies input intent
│   ├── ui.py                  # Streamlit rendering components
│   ├── utils/
│   │   └── logger.py          # Async JSONL query logger
│   └── vector_store.py        # ChromaDB lifecycle, ingestion ledger, dedup
├── tests/                     # Pytest test suite
├── api.py                     # FastAPI application
├── cli_app.py                 # CLI entry point
├── ui_app.py                  # Streamlit entry point
└── pyproject.toml             # Dependencies managed by uv
```

---

## CLI Commands

| Command | Description |
|---|---|
| `/help` | Show available commands |
| `/model <name>` | Switch LLM model (e.g., `local-llm`) |
| `/ingest <path>` | Ingest a file or directory |
| `/dedup` | Remove duplicate chunks |
| `/clean-db` | Wipe the entire vector store |
| `/clear` | Clear terminal |
| `/exit` | Quit |

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/ingest` | Ingest a file or directory |
| `POST` | `/query` | Ask a question |
| `GET` | `/eval` | Run offline RAGAS evaluation |

---

## Running Tests

```bash
uv run pytest
```

---

## Telemetry & Evaluation

Query logs are stored in `logs/query_log.jsonl`. Run offline RAGAS evaluation:

```bash
# Start the API server
uv run uvicorn api:app --reload --port 8000

# Trigger evaluation
curl http://127.0.0.1:8000/eval
```

This computes faithfulness, answer relevancy, context precision, and context recall from your query logs.

---

## License

MIT
