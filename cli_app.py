import os
import time
import asyncio
import click
import logging
import sys
from rag_engine import LiteLLMClient, RAGCoreEngine, GuardrailsManager, QueryLogger, ColoredFormatter, ConversationMemory, configure_logging, QueryRouter, LLMMetricsCollector
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
    llm_metrics = LLMMetricsCollector()

    # 0. Route the query
    route_result = await query_router.route_query(query, metrics_collector=llm_metrics, model=selected_model)

    if route_result["route"] == "DIRECT_LLM":

        messages = [
            {"role": "system", "content": DIRECT_LLM_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]
        direct_start = time.time()
        response = await llm_client.acompletion(
            messages, temperature=0.7,
            metrics_collector=llm_metrics,
            metrics_purpose="Direct LLM"
        )
        answer = response.choices[0].message.content
        direct_elapsed = time.time() - direct_start
        usage = getattr(response, "usage", None)
        pt = usage.prompt_tokens if usage else 0
        ct = usage.completion_tokens if usage else 0
        tt = usage.total_tokens if usage else 0
        memory.add_turn(query, answer)

        latency = (time.time() - start_time) * 1000
        await query_logger.log_query(
            query=query,
            response=answer,
            retrieved_nodes=[],
            latency_ms=latency,
            metadata={"model": selected_model, "routing": "direct_llm"}
        )

        logger.info(
            f"\033[1;34m[Direct LLM]\033[0m "
            f"\033[1;33m{selected_model}\033[0m "
            f"\033[1;32m[Tokens: {tt} (In={pt}, Out={ct})]\033[0m "
            f"time: {direct_elapsed:.3f}s"
        )

        chunk_size = 5
        for i in range(0, len(answer), chunk_size):
            yield answer[i:i + chunk_size]
            await asyncio.sleep(0.01)

        metrics_output = llm_metrics.format_pretty_block()
        yield (
            f"\n\n\033[33m💬 Direct LLM response (no document search)"
            f" — Latency: {latency:.1f} ms\033[0m"
        )
        if metrics_output:
            yield f"\n\n{metrics_output}\n"
        return

    # 1. Search core engine
    start_retrieve = time.time()
    contexts = await core_engine.search(query, use_hyde=False)
    latency_retrieve = (time.time() - start_retrieve) * 1000
    
    # 2. Guardrails generate faithful answer
    start_gen = time.time()
    result = await guardrails.generate_faithful_answer(
        query, contexts, model=selected_model,
        chat_history=memory.format_history(),
        metrics_collector=llm_metrics
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
    citation_map = result.get("citation_map", {})
    if citation_map:
        cited_str = "\n\n\033[33m📖 Cited Sources:\033[0m\n"
        for ref, info in citation_map.items():
            cited_str += f"  [{ref}] {info['filename']} (p. {info['page_number']})\n"
        yield cited_str

    # 6. Yield source references mapping using citation_map
    citation_map = result.get("citation_map", {})
    if citation_map:
        sources_str = "\n\n📚 Sources:\n"
        for ref, info in citation_map.items():
            sources_str += (
                f"  \033[1m[{ref}]\033[0m {info.get('filename')} "
                f"(p. {info.get('page_number')}) - Chunk: {info.get('chunk_id', 'N/A')}\n"
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

    metrics_block = llm_metrics.format_pretty_block()
    if metrics_block:
        metrics_str += f"\n{metrics_block}\n"

    # Log structured metrics
    logger.info(
        f"\033[1;37m[Retrieval]\033[0m nodes={num_docs} | "
        f"retrieve={latency_retrieve:.1f}ms | generate={latency_gen:.1f}ms | "
        f"total={latency:.1f}ms | max_score={top_score:.4f} | avg_score={avg_score:.4f}"
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
