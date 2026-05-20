# Enterprise RAG Engine (`rag_engine`)

A modular, plug-and-play, enterprise-grade RAG library designed for stateless execution, robust local fallback, programmatically verified citations, and offline/online telemetry evaluation feedback loops.

## Architecture

- **`rag_engine.prompts`**: Centralized prompt templates (HyDE, Citations, Faithfulness, Self-correction).
- **`rag_engine.llm`**: Router wrapping `litellm` with local models fallback (`SentenceTransformer`, character hash embeddings) and reranking.
- **`rag_engine.core`**: Multi-format document parser, Semantic Chunker (embedding distance spikes), parent-child node database, BM25 + Vector hybrid search with Reciprocal Rank Fusion (RRF), and HyDE expansion.
- **`rag_engine.guardrails`**: Programmatic citation validator and async self-correction loop checking faithfulness.
- **`rag_engine.cli`**: Extensible click REPL CLI command manager.
- **`rag_engine.ui`**: Streamlit visualization layout fragments.
- **`rag_engine.utils.logger`**: Non-blocking async structured JSONL query logger.
- **`rag_engine.evaluation`**: Telemetry log parser supporting RAGAS and heuristic offline metric scores.

## Installation

Ensure `uv` is installed, then run:
```bash
uv sync
```

## Running the Applications

### 1. Interactive REPL Terminal App
```bash
uv run python cli_app.py
```

### 2. Streamlit Dashboard App
```bash
uv run streamlit run ui_app.py
```

### 3. FastAPI Web Server
```bash
uv run uvicorn api:app --reload
```

## Running Verification Tests
```bash
uv run pytest
```
