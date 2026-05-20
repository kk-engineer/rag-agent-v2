import os
import time
import asyncio
import click
import logging
import sys
from rag_engine import LiteLLMClient, RAGCoreEngine, GuardrailsManager, QueryLogger, ColoredFormatter, ConversationMemory, configure_logging, build_citation_map, QueryRouter
from rag_engine.prompts import DIRECT_LLM_SYSTEM_PROMPT
from rag_engine.cli import REPLManager


# Load memory config
_memory_window = 5
try:
    with open("config/rag_config.toml", "rb") as f:
        import tomllib
        _memory_window = tomllib.load(f).get("memory", {}).get("context_window", 5)
except Exception:
    pass

memory = ConversationMemory(window_size=_memory_window)


# Configure root logging with colored formatter
root_logger = logging.getLogger()
if not root_logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(ColoredFormatter(datefmt="%H:%M:%S"))
    root_logger.addHandler(handler)
configure_logging()
logger = logging.getLogger(__name__)



llm_client = LiteLLMClient()
core_engine = RAGCoreEngine(llm_client)
guardrails = GuardrailsManager(llm_client)
query_logger = QueryLogger()
query_router = QueryRouter(llm_client)
selected_model = "local-llm"


async def query_callback(query: str):

    start_time = time.time()

    # 0. Route the query
    route_result = await query_router.route_query(query)

    if route_result["route"] == "DIRECT_LLM":

        messages = [
            {"role": "system", "content": DIRECT_LLM_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]
        response = await llm_client.acompletion(messages, temperature=0.7)
        answer = response.choices[0].message.content
        memory.add_turn(query, answer)

        latency = (time.time() - start_time) * 1000
        await query_logger.log_query(
            query=query,
            response=answer,
            retrieved_nodes=[],
            latency_ms=latency,
            metadata={"model": selected_model, "routing": "direct_llm"}
        )

        chunk_size = 5
        for i in range(0, len(answer), chunk_size):
            yield answer[i:i + chunk_size]
            await asyncio.sleep(0.01)

        yield (
            f"\n\n\033[33m💬 Direct LLM response (no document search)"
            f" — Latency: {latency:.1f} ms\033[0m"
        )
        return

    # 1. Search core engine
    start_retrieve = time.time()
    contexts = await core_engine.search(query, use_hyde=False)
    latency_retrieve = (time.time() - start_retrieve) * 1000
    
    # 2. Guardrails generate faithful answer
    start_gen = time.time()
    result = await guardrails.generate_faithful_answer(
        query, contexts, model=selected_model,
        chat_history=memory.format_history()
    )
    latency_gen = (time.time() - start_gen) * 1000
    answer = result["answer"]
    memory.add_turn(query, answer)
    
    # 3. Log query execution
    latency = (time.time() - start_time) * 1000
    await query_logger.log_query(
        query=query,
        response=answer,
        retrieved_nodes=contexts,
        latency_ms=latency,
        metadata={"model": selected_model, "faithful": result.get("faithful", True)}
    )
    
    # 4. Stream response to console
    chunk_size = 5
    for i in range(0, len(answer), chunk_size):

        yield answer[i:i+chunk_size]
        await asyncio.sleep(0.01)

    # 5. Yield citation map (sources actually cited in the answer)
    citation_map = build_citation_map(answer, contexts)
    if citation_map:
        cited_str = "\n\n\033[33m📖 Cited Sources:\033[0m\n"
        for ref, info in citation_map.items():
            cited_str += f"  {ref} → {info['filename']} (p. {info['page']})\n"
        yield cited_str

    # 6. Yield source references mapping
    if contexts:
        sources_str = "\n\n📚 Sources:\n"
        for idx, ctx in enumerate(contexts):
            sources_str += (
                f"  \033[1m[{idx}]\033[0m {ctx.get('filename')} "
                f"(p. {ctx.get('page_number')}) - Score: {ctx.get('score', 0.0):.4f}\n"
            )
        yield sources_str

    # 7. Format and yield metrics block at the end of stream
    num_docs = len(contexts)
    scores = [c.get("score", 0.0) for c in contexts]
    top_score = max(scores) if scores else 0.0
    avg_score = sum(scores) / num_docs if num_docs > 0 else 0.0
    
    metrics_str = (
        f"\n\n"
        f"\033[94m--- 📊 Retrieval & Search Metrics ---\033[0m\n"
        f"• \033[1mRetrieved Nodes:\033[0m {num_docs} chunks\n"
        f"• \033[1mRetrieval Latency:\033[0m {latency_retrieve:.1f} ms ({latency_retrieve/1000:.3f} s)\n"
        f"• \033[1mAnswer Generation Latency:\033[0m {latency_gen:.1f} ms ({latency_gen/1000:.3f} s)\n"
        f"• \033[1mTotal End-to-End Latency:\033[0m {latency:.1f} ms ({latency/1000:.3f} s)\n"
        f"• \033[1mMax Match Score:\033[0m {top_score:.4f}\n"
        f"• \033[1mAvg Match Score:\033[0m {avg_score:.4f}\n"
        f"\033[94m------------------------------------\033[0m\n"
    )

    llm_metrics = result.get("llm_metrics")
    if llm_metrics:
        metrics_str += (
            f"\n"
            f"\033[95m--- LLM Metrics ---\033[0m\n"
            f"• \033[1mTotal LLM Calls:\033[0m {llm_metrics['total_calls']}\n"
            f"• \033[1mTotal Prompt Tokens:\033[0m {llm_metrics['total_prompt_tokens']}\n"
            f"• \033[1mTotal Completion Tokens:\033[0m {llm_metrics['total_completion_tokens']}\n"
            f"• \033[1mTotal Tokens:\033[0m {llm_metrics['total_tokens']}\n"
            f"• \033[1mTotal LLM Time:\033[0m {llm_metrics['total_llm_time_ms']/1000:.3f}s\n"
            f"• \033[1mPer-call breakdown:\033[0m {llm_metrics['per_call_breakdown_str']}\n"
            f"\033[95m------------------------------------\033[0m\n"
        )

    # Log structured metrics
    logger.info(
        f"[RAG Metrics] Retrieval: nodes={num_docs}, "
        f"retrieve={latency_retrieve:.1f}ms, generate={latency_gen:.1f}ms, "
        f"total={latency:.1f}ms, max_score={top_score:.4f}, avg_score={avg_score:.4f}"
    )
    if llm_metrics:
        logger.info(
            f"[RAG Metrics] LLM: calls={llm_metrics['total_calls']}, "
            f"prompt_tks={llm_metrics['total_prompt_tokens']}, "
            f"completion_tks={llm_metrics['total_completion_tokens']}, "
            f"total_tks={llm_metrics['total_tokens']}, "
            f"time={llm_metrics['total_llm_time_ms']/1000:.3f}s, "
            f"breakdown: {llm_metrics['per_call_breakdown_str']}"
        )

    yield metrics_str


def model_callback(model_name: str):

    global selected_model
    selected_model = model_name


async def ingest_callback(path: str):

    start_time = time.time()
    if os.path.isdir(path):

        click.secho(f"Starting directory scan & ingestion: {path}", fg="cyan")
        count = 0
        for root, _, files in os.walk(path):

            for file in files:

                file_path = os.path.join(root, file)
                ext = os.path.splitext(file_path)[1].lower()
                if ext in [".pdf", ".html", ".htm", ".md", ".markdown", ".txt", ".py", ".js"]:

                    click.secho(f"Ingesting file: {file_path}", fg="blue")
                    doc_start = time.time()
                    await core_engine.ingest_file(file_path)
                    doc_duration = time.time() - doc_start
                    click.secho(f"Finished ingesting '{file}' in {doc_duration:.3f}s", fg="green")
                    count += 1
        total_duration = time.time() - start_time
        click.secho(f"Successfully completed folder ingestion. Ingested {count} files in {total_duration:.3f}s", fg="green")
    else:

        click.secho(f"Ingesting file: {path}", fg="blue")
        doc_start = time.time()
        await core_engine.ingest_file(path)
        doc_duration = time.time() - doc_start
        click.secho(f"Finished ingesting '{os.path.basename(path)}' in {doc_duration:.3f}s", fg="green")


@click.command()
@click.option("--ingest-path", default=None, help="Initial file or folder path to ingest")
def main(ingest_path):

    loop = asyncio.get_event_loop()
    
    if ingest_path:

        click.secho(f"Pre-ingesting documents from: {ingest_path}", fg="cyan")
        try:

            loop.run_until_complete(ingest_callback(ingest_path))
        except Exception as e:

            click.secho(f"Pre-ingestion failed: {e}", fg="red")

    repl = REPLManager(
        query_callback=query_callback,
        model_selection_callback=model_callback,
        ingest_callback=ingest_callback,
        core_engine=core_engine,
    )
    
    try:

        loop.run_until_complete(repl.start_interactive_loop())
    except KeyboardInterrupt:

        click.secho("\nGoodbye!", fg="red")


if __name__ == "__main__":

    main()
