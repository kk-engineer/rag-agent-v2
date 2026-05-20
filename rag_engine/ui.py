import asyncio
from typing import List, Dict, Any, Union, Generator, AsyncGenerator


def render_chat_history(messages: List[Dict[str, Any]]):

    import streamlit as st
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
                    
                    render_retrieved_sources(retrieved_contexts)


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
