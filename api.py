import os
import time
import logging
import sys
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from rag_engine import LiteLLMClient, RAGCoreEngine, GuardrailsManager, QueryLogger, ColoredFormatter
from rag_engine.evaluation import RagasEvaluator


# Configure root logging with colored formatter
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
if not root_logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(ColoredFormatter(datefmt="%H:%M:%S"))
    root_logger.addHandler(handler)
logger = logging.getLogger(__name__)


app = FastAPI(title="Enterprise RAG Engine API", version="0.1.0")


# Instantiate components
llm_client = LiteLLMClient()
core_engine = RAGCoreEngine(llm_client)
guardrails = GuardrailsManager(llm_client)
query_logger = QueryLogger()
evaluator = RagasEvaluator(llm_client=llm_client)


class IngestRequest(BaseModel):

    path: str


class QueryRequest(BaseModel):

    query: str
    model: str = "local-llm"
    top_n: int = 3
    use_hyde: bool = False
    history: Optional[List[Dict[str, str]]] = None


@app.post("/ingest")
async def ingest_document(payload: IngestRequest):

    path = payload.path
    if not os.path.exists(path):

        raise HTTPException(status_code=404, detail=f"Path not found: {path}")

    start_time = time.time()
    try:

        if os.path.isdir(path):

            count = 0
            for root, _, files in os.walk(path):

                for file in files:

                    file_path = os.path.join(root, file)
                    ext = os.path.splitext(file_path)[1].lower()
                    if ext in [".pdf", ".html", ".htm", ".md", ".markdown", ".txt", ".py", ".js"]:

                        doc_start = time.time()
                        await core_engine.ingest_file(file_path)
                        doc_duration = time.time() - doc_start
                        logger.info(f"API Ingest: Finished file '{file}' in {doc_duration:.3f}s")
                        count += 1
            total_duration = time.time() - start_time
            logger.info(f"API Ingest: Directory {path} ingestion complete. {count} files in {total_duration:.3f}s")
            return {
                "status": "success",
                "message": f"Ingested {count} files from directory.",
                "duration_seconds": total_duration,
                "path": path
            }
        else:

            logger.info(f"API Request: Ingesting file {path}")
            doc_start = time.time()
            doc_id = await core_engine.ingest_file(path)
            doc_duration = time.time() - doc_start
            logger.info(f"API Ingest: Finished file '{os.path.basename(path)}' in {doc_duration:.3f}s")
            return {
                "status": "success",
                "message": "Ingested file successfully.",
                "doc_id": doc_id,
                "duration_seconds": doc_duration,
                "path": path
            }
    except Exception as e:

        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query")
async def query_engine(payload: QueryRequest):

    start_time = time.time()
    try:

        # 1. Search core retrieve engine
        retrieve_start = time.time()
        contexts = await core_engine.search(
            query=payload.query,
            top_n=payload.top_n,
            use_hyde=payload.use_hyde
        )
        retrieve_duration = time.time() - retrieve_start
        
        # 2. Format conversation history if provided
        chat_history = ""
        if payload.history:
            lines = ["Previous conversation:"]
            for msg in payload.history:
                role = "User" if msg.get("role") == "user" else "Assistant"
                lines.append(f"{role}: {msg.get('content', '')}")
            chat_history = "\n".join(lines)

        # 3. Programmatic guardrails verification & citation check
        gen_start = time.time()
        result = await guardrails.generate_faithful_answer(
            query=payload.query,
            contexts=contexts,
            model=payload.model,
            chat_history=chat_history
        )
        gen_duration = time.time() - gen_start
        
        # 3. Log query execution
        latency = (time.time() - start_time) * 1000
        await query_logger.log_query(
            query=payload.query,
            response=result["answer"],
            retrieved_nodes=contexts,
            latency_ms=latency,
            metadata={"model": payload.model, "api_call": True}
        )
        
        # Format response nodes for output
        clean_contexts = []
        for ctx in contexts:

            clean_contexts.append({
                "text": ctx.get("text"),
                "page_number": ctx.get("page_number"),
                "source": ctx.get("source"),
                "filename": ctx.get("filename"),
                "score": ctx.get("score")
            })

        logger.info(f"API Query: completed in {latency/1000:.3f}s (Retrieve: {retrieve_duration:.3f}s, Gen/Verify: {gen_duration:.3f}s)")
        return {
            "query": payload.query,
            "answer": result["answer"],
            "faithful": result.get("faithful", True),
            "retries": result.get("attempts", 1),
            "latency_ms": latency,
            "retrieve_latency_ms": retrieve_duration * 1000,
            "generation_latency_ms": gen_duration * 1000,
            "retrieved_nodes": clean_contexts
        }
    except Exception as e:

        raise HTTPException(status_code=500, detail=str(e))


@app.get("/eval")
async def run_evaluation():

    try:

        scores = await evaluator.evaluate_ragas()
        return scores
    except Exception as e:

        raise HTTPException(status_code=500, detail=str(e))
