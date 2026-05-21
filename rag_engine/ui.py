import asyncio
from typing import List, Dict, Any, Union, Generator, AsyncGenerator
from rag_engine.guardrails import build_citation_map


def render_chat_history(messages: List[Dict[str, Any]]):

    import streamlit as st
    st.markdown(
        """
        <style>
        /* 1. Target paragraphs inside chat messages */
        div[data-testid="stChatMessage"] p { 
            font-size: 1.2rem !important; 
        }

        /* 2. Target the text block inside the metric label wrapper */
        div[data-testid="stMetricLabel"] p { 
            font-size: 0.7rem !important; 
        }

        /* 3. Target the inner text container of the metric value */
        div[data-testid="stMetricValue"] > div { 
            font-size: 0.85rem !important; 
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    if "rag_metrics_css" not in st.session_state:
        st.session_state.rag_metrics_css = True
    for msg in messages:

        role = msg["role"]
        content = msg["content"]
        if role == "user":

            with st.chat_message("user", avatar="👤"):

                st.write(content)
        else:

            with st.chat_message("assistant", avatar="✨"):

                st.write(content)
                
                retrieved_contexts = msg.get("retrieved_contexts")
                latency_retrieve = msg.get("latency_retrieve")
                
                if retrieved_contexts is not None:

                    st.markdown("---")
                    st.markdown("##### 📊 Retrieval & Search Metrics")
                    
                    num_docs = len(retrieved_contexts)
                    scores = [c.get("score", 0.0) for c in retrieved_contexts]
                    top_score = max(scores) if scores else 0.0
                    avg_score = sum(scores) / num_docs if num_docs > 0 else 0.0
                    
                    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                    col_m1.metric("Retrieved Nodes", f"{num_docs} chunks")
                    if latency_retrieve is not None:

                        col_m2.metric("Retrieval Latency", f"{latency_retrieve:.1f} ms")
                    else:

                        col_m2.metric("Retrieval Latency", "N/A")
                    col_m3.metric("Max Match Score", f"{top_score:.4f}")
                    col_m4.metric("Avg Match Score", f"{avg_score:.4f}")

                    llm_metrics = msg.get("llm_metrics")
                    if llm_metrics:
                        st.markdown("##### 🤖 LLM Metrics")
                        col_l1, col_l2, col_l3, col_l4 = st.columns(4)
                        col_l1.metric("Total LLM Calls", f"{llm_metrics['total_calls']}")
                        col_l2.metric("Total Prompt Tokens", f"{llm_metrics['total_prompt_tokens']}")
                        col_l3.metric("Total Completion Tokens", f"{llm_metrics['total_completion_tokens']}")
                        col_l4.metric("Total LLM Time", f"{llm_metrics['total_llm_time_ms']/1000:.3f}s")
                        with st.expander("Per-call breakdown"):
                            for call in llm_metrics["per_call_breakdown"]:
                                st.write(
                                    f"• **{call['purpose']}**: "
                                    f"{call['elapsed_ms']/1000:.1f}s, "
                                    f"{call['total_tokens']} tokens"
                                )
                    
                    citation_map = msg.get("citation_map") or build_citation_map(content, retrieved_contexts)
                    if citation_map:
                        st.markdown("##### 📄 Sources Cited in Answer")
                        for ref, info in citation_map.items():
                            ref_display = ref if ref.startswith('[') else f"[{ref}]"
                            pn = info.get('page_number', info.get('page', '?'))
                            score = info.get('score', 0.0)
                            title = f"{ref_display} {info.get('filename', 'Unknown')} • p. {pn} — Score: {score:.4f}"
                            with st.expander(title):
                                st.markdown(f"**File:** {info.get('filename', 'Unknown')}")
                                st.markdown(f"**Page:** {pn}")
                                st.markdown(f"**Score:** {score:.4f}")
                                if info.get('chunk_id'):
                                    st.markdown(f"**Chunk ID:** `{info['chunk_id']}`")
                                st.markdown("---")
                                st.markdown(info.get('text', ''))


def render_reasoning_step(title: str, content: str, expanded: bool = False):

    import streamlit as st
    with st.expander(title, expanded=expanded):

        st.markdown(content)


def render_retrieved_sources(sources: List[Dict[str, Any]]):

    import streamlit as st
    if not sources:

        return

    st.markdown("#### Retrieved Context Sources")
    for idx, src in enumerate(sources):

        filename = src.get("filename", "Unknown Source")
        page = src.get("page_number", 1)
        score = src.get("score", 0.0)
        
        with st.expander(f"[Doc-{idx}] {filename} (Page {page}) - Score: {score:.4f} (0.0-lowest, 1.0-highest)"):

            st.markdown(f"**Filename:** {filename}")
            st.markdown(f"**Source Path:** `{src.get('source')}`")
            st.markdown(f"**Page:** {page}")
            st.markdown(f"**Similarity/Rerank Score:** {score:.4f} (0.0 - lowest, 1.0 - highest)")
            st.markdown("---")
            st.markdown(src.get("text", ""))


def render_streaming_response(
    generator_or_async_generator: Union[Generator[str, None, None], AsyncGenerator[str, None]]
) -> str:

    import streamlit as st
    placeholder = st.empty()
    full_response = ""
    
    # Check if async generator
    if hasattr(generator_or_async_generator, "__anext__"):

        try:

            loop = asyncio.get_event_loop()
        except RuntimeError:

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        async def consume():

            nonlocal full_response
            async for token in generator_or_async_generator:

                full_response += token
                placeholder.markdown(full_response + "▌")
                
        loop.run_until_complete(consume())
    else:

        for token in generator_or_async_generator:

            full_response += token
            placeholder.markdown(full_response + "▌")
            
    placeholder.markdown(full_response)
    return full_response
