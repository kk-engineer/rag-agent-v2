import os
import warnings
import logging
from rag_engine import configure_logging

# Suppress Hugging Face / Transformers warning messages and path lookup alerts
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*Accessing.*__path__.*")
warnings.filterwarnings("ignore", message=".*zoedepth.*")

# Silence transformers module logging
logging.getLogger("transformers").setLevel(logging.ERROR)
configure_logging()


class ZoedepthWarningFilter(logging.Filter):

    def filter(self, record):

        try:

            msg = record.getMessage()
            if "zoedepth" in msg or "__path__" in msg:

                return False
        except Exception:

            pass
        return True


zoedepth_filter = ZoedepthWarningFilter()
logging.getLogger("transformers").addFilter(zoedepth_filter)
logging.getLogger("py.warnings").addFilter(zoedepth_filter)
logging.getLogger().addFilter(zoedepth_filter)

import streamlit as st
import asyncio
import time
import logging
from rag_engine import LiteLLMClient, RAGCoreEngine, GuardrailsManager, QueryLogger, ConversationMemory

logger = logging.getLogger(__name__)
from rag_engine.ui import (
    render_chat_history,
    render_reasoning_step,
    render_retrieved_sources,
    render_streaming_response
)


st.set_page_config(
    page_title="Enterprise RAG Engine Dashboard",
    page_icon="🤖",
    layout="wide"
)

# Inject custom Meta Blue CSS style overrides
st.markdown("""
<style>
/* Custom Scrollbar */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}
::-webkit-scrollbar-track {
    background: #f0f2f5;
}
::-webkit-scrollbar-thumb {
    background: #0064e0;
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
    background: #004baf;
}

/* Chat Input Boundary */
div[data-testid="stChatInput"] {
    border: 2px solid #0064e0 !important;
    border-radius: 12px !important;
    box-shadow: 0 2px 6px rgba(0, 100, 224, 0.15) !important;
}

/* Textarea focus within chat input */
textarea[data-testid="stChatInputTextArea"] {
    border: none !important;
}

/* Chat Avatar Background and circular frames for user & assistant */
div[data-testid="stChatMessageAvatar"] {
    background-color: #e8f0fe !important;
    border: 2px solid #0064e0 !important;
    border-radius: 50% !important;
    width: 40px !important;
    height: 40px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    box-shadow: 0 2px 4px rgba(0, 100, 224, 0.15) !important;
    font-size: 1.25rem !important;
}

/* Chat message window boundaries */
div[data-testid="stChatMessage"] {
    border: 1px solid #0064e0 !important;
    border-radius: 12px !important;
    padding: 12px !important;
    margin-bottom: 12px !important;
    background-color: #f7f9fc !important;
    box-shadow: 0 1px 3px rgba(0, 100, 224, 0.05) !important;
}

/* Style user specific message background differently but within meta theme */
div[data-testid="stChatMessage"][data-test-user="true"],
.st-emotion-cache-janw5y { /* user cached class fallback */
    border-left: 5px solid #0064e0 !important;
}

/* Status step border and background */
div[data-testid="stStatusWidget"] {
    border: 1.5px solid #0064e0 !important;
    border-radius: 8px !important;
    background-color: #f0f2f5 !important;
}

/* Sidebar styling integration */
section[data-testid="stSidebar"] {
    border-right: 2px solid #0064e0 !important;
    background-color: #f0f2f5 !important;
}

/* Button overrides */
.stButton > button {
    background-color: #0064e0 !important;
    color: white !important;
    border: 1.5px solid #004baf !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease-in-out !important;
}
.stButton > button:hover {
    background-color: #004baf !important;
    border-color: #003680 !important;
    box-shadow: 0 4px 8px rgba(0, 100, 224, 0.25) !important;
    transform: translateY(-1px) !important;
}

/* Expander custom styling */
div[data-testid="stExpander"] {
    border: 1px solid #0064e0 !important;
    border-radius: 8px !important;
    background-color: #ffffff !important;
    box-shadow: 0 1px 4px rgba(0, 100, 224, 0.05) !important;
}
</style>
""", unsafe_allow_html=True)


# Initialize session state objects
if "llm_client" not in st.session_state:

    st.session_state.llm_client = LiteLLMClient()
    st.session_state.core_engine = RAGCoreEngine(st.session_state.llm_client)
    st.session_state.guardrails = GuardrailsManager(st.session_state.llm_client)
    st.session_state.query_logger = QueryLogger()
    st.session_state.chat_history = []
    st.session_state.ingested_files = []
    st.session_state.generating = False
    st.session_state.pending_query = None
    st.session_state.memory = ConversationMemory()

if "memory" not in st.session_state:

    st.session_state.memory = ConversationMemory()


def run_async(coro):

    try:

        loop = asyncio.get_event_loop()
    except RuntimeError:

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
    return loop.run_until_complete(coro)


def render_directory_browser():

    if "browse_path" not in st.session_state:

        st.session_state.browse_path = os.path.abspath(".")
    
    current_path = st.session_state.browse_path
    st.write(f"📁 **Location:** `{current_path}`")
    
    try:

        entries = os.listdir(current_path)
        entries = [e for e in entries if not e.startswith(".")]
        entries.sort()
    except Exception as e:

        st.error(f"Cannot read directory: {e}")
        entries = []
        
    dirs = [".."]
    files = []
    for entry in entries:

        full_path = os.path.join(current_path, entry)
        if os.path.isdir(full_path):

            dirs.append(entry)
        else:

            files.append(entry)
            
    selected_dir = st.selectbox(
        "Subdirectories",
        dirs,
        index=0,
        key="dir_navigator"
    )
    
    if st.button("Go Into Folder"):

        if selected_dir == "..":

            st.session_state.browse_path = os.path.dirname(current_path)
        else:

            st.session_state.browse_path = os.path.join(current_path, selected_dir)
        st.rerun()
        
    selected_file = st.selectbox(
        "Select File (optional)",
        ["[Entire Directory]"] + files,
        key="file_navigator"
    )
    
    if selected_file == "[Entire Directory]":

        return current_path
    else:

        return os.path.join(current_path, selected_file)


# Sidebar layout for settings
with st.sidebar:

    st.title("Settings & Ingestion")
    
    # Model name is resolved from default configs loaded at startup
    model_name = "cloud-llm" if st.session_state.llm_client.execution_mode == "cloud" else "local-llm"
    
    st.write("---")
    st.subheader("Document Ingestion")
    
    # Render interactive file system directory browser
    path_to_ingest = render_directory_browser()
    st.caption(f"Target: `{path_to_ingest}`")
    
    if st.button("Ingest Selected Path"):

        if path_to_ingest:

            with st.spinner(f"Ingesting {path_to_ingest}..."):

                try:

                    start_time = time.time()
                    if os.path.isdir(path_to_ingest):

                        st.info(f"Scanning and ingesting directory: `{path_to_ingest}`")
                        count = 0
                        for root, _, files in os.walk(path_to_ingest):

                            for file in files:

                                file_path = os.path.join(root, file)
                                ext = os.path.splitext(file_path)[1].lower()
                                if ext in [".pdf", ".html", ".htm", ".md", ".markdown", ".txt", ".py", ".js"]:

                                    st.write(f"⏳ Ingesting `{file}`...")
                                    doc_start = time.time()
                                    run_async(st.session_state.core_engine.ingest_file(file_path))
                                    doc_duration = time.time() - doc_start
                                    st.write(f"✅ Ingested `{file}` in {doc_duration:.3f}s")
                                    count += 1
                        total_duration = time.time() - start_time
                        st.success(f"Successfully ingested {count} files in {total_duration:.3f}s")
                    else:

                        st.write(f"⏳ Ingesting `{os.path.basename(path_to_ingest)}`...")
                        doc_start = time.time()
                        run_async(st.session_state.core_engine.ingest_file(path_to_ingest))
                        doc_duration = time.time() - doc_start
                        st.success(f"Successfully ingested file `{os.path.basename(path_to_ingest)}` in {doc_duration:.3f}s")
                    st.session_state.ingested_files.append(path_to_ingest)
                except Exception as e:

                    st.error(f"Error during ingestion: {e}")
        else:

            st.warning("Please enter a valid path.")
            
    if st.session_state.ingested_files:

        st.markdown("**Ingested Sources:**")
        for f in st.session_state.ingested_files:

            st.caption(f)

    st.write("---")
    st.subheader("Evaluation Suite")
    if st.button("Run Offline Evaluation"):

        with st.spinner("Calculating RAG evaluation scores from telemetry logs..."):

            from rag_engine.evaluation import RagasEvaluator
            evaluator = RagasEvaluator()
            scores = run_async(evaluator.evaluate_ragas())
            
            if "status" in scores:

                st.info(scores["status"])
            else:

                st.markdown("**Offline Metrics:**")
                def _fmt(val):
                    return "N/A" if val is None else f"{val:.4f}"
                col1, col2 = st.columns(2)
                col1.metric("Faithfulness", _fmt(scores.get('faithfulness')))
                col2.metric("Answer Relevancy", _fmt(scores.get('answer_relevancy')))
                col3, col4 = st.columns(2)
                col3.metric("Context Precision", _fmt(scores.get('context_precision')))
                col4.metric("Context Recall", _fmt(scores.get('context_recall')))
                st.caption(f"Evaluation Method: {scores.get('evaluation_type', 'RAGAS')}")


# Main Content Area
st.title("🤖 RoboSathi RAG Engine")
st.caption("Upload documents - Ask Questions")

# Render Chat UI
render_chat_history(st.session_state.chat_history)

# Query text entry
if st.session_state.generating:

    # Show a disabled chat input when generating
    st.chat_input("Agent is processing the answer...", disabled=True)
else:

    if user_query := st.chat_input("Enter your question..."):

        st.session_state.pending_query = user_query
        st.session_state.generating = True
        st.rerun()

# If there is a pending query, process it
if st.session_state.generating and st.session_state.pending_query:

    user_query = st.session_state.pending_query
    
    # Render user message instantly
    st.chat_message("user", avatar="👤").write(user_query)
    st.session_state.chat_history.append({"role": "user", "content": user_query})

    # Start measuring latency
    start_time = time.time()

    # Create status card for the Agent's thought process
    with st.status("Agent is thinking...", expanded=True) as status_box:

        def show_thought(text: str):

            st.write(text)

        # 1. Search sparse/dense index
        show_thought("🔍 Querying sparse and dense vector indexes...")
        search_start = time.time()
        retrieved_contexts = run_async(st.session_state.core_engine.search(user_query))
        latency_retrieve = (time.time() - search_start) * 1000
        
        # 2. Reasoning Telemetry
        reasoning_text = f"📄 Retrieved {len(retrieved_contexts)} context chunks.\n"
        if retrieved_contexts:

            reasoning_text += "\n".join(f"- Doc: `{c.get('filename')}` (Page {c.get('page_number')})" for c in retrieved_contexts)
        show_thought(reasoning_text)

        # 3. Call guardrails validation loop with conversation memory
        show_thought("🛡️ Grounding answer and executing citation verification guardrails...")
        result = run_async(st.session_state.guardrails.generate_faithful_answer(
            user_query, retrieved_contexts, model=model_name,
            on_thought=show_thought,
            chat_history=st.session_state.memory.format_history()
        ))
        
        status_box.update(label="Verification finished!", state="complete", expanded=False)

    # Process Assistant Response
    with st.chat_message("assistant", avatar="✨"):

        answer = result["answer"]
        
        # Log structured metrics
        latency_total = (time.time() - start_time) * 1000
        num_docs = len(retrieved_contexts)
        scores = [c.get("score", 0.0) for c in retrieved_contexts]
        top_score = max(scores) if scores else 0.0
        avg_score = sum(scores) / num_docs if num_docs > 0 else 0.0
        logger.info(
            f"[RAG Metrics] Retrieval: nodes={num_docs}, "
            f"retrieve={latency_retrieve:.1f}ms, "
            f"total={latency_total:.1f}ms, max_score={top_score:.4f}, avg_score={avg_score:.4f}"
        )
        llm_metrics = result.get("llm_metrics")
        if llm_metrics:
            logger.info(
                f"[RAG Metrics] LLM: calls={llm_metrics['total_calls']}, "
                f"prompt_tks={llm_metrics['total_prompt_tokens']}, "
                f"completion_tks={llm_metrics['total_completion_tokens']}, "
                f"total_tks={llm_metrics['total_tokens']}, "
                f"time={llm_metrics['total_llm_time_ms']/1000:.3f}s, "
                f"breakdown: {llm_metrics['per_call_breakdown_str']}"
            )

        run_async(st.session_state.query_logger.log_query(
            query=user_query,
            response=answer,
            retrieved_nodes=retrieved_contexts,
            latency_ms=latency_total,
            metadata={"model": model_name, "faithful_verified": result.get("faithful", True)}
        ))

        # Stream assistant response to UI
        async def response_stream():

            chunk_size = 5
            for i in range(0, len(answer), chunk_size):

                yield answer[i:i+chunk_size]
                await asyncio.sleep(0.01)

        final_answer = render_streaming_response(response_stream())
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": final_answer,
            "retrieved_contexts": retrieved_contexts,
            "latency_retrieve": latency_retrieve,
            "llm_metrics": result.get("llm_metrics")
        })

    st.session_state.memory.add_turn(user_query, final_answer)

    # Done generating, reset state and rerun to update the UI (enabling chat input again)
    st.session_state.pending_query = None
    st.session_state.generating = False
    st.rerun()
