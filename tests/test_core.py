import os
import pytest
from unittest.mock import MagicMock
from rag_engine import LiteLLMClient, RAGCoreEngine


@pytest.fixture
def temp_files(tmp_path):

    html_file = tmp_path / "doc.html"
    html_file.write_text("<html><body><h1>Hello World</h1><p>This is a RAG test.</p></body></html>")
    
    md_file = tmp_path / "doc.md"
    md_file.write_text("# Markdown Title\nThis is a md document.")
    
    return html_file, md_file


@pytest.mark.asyncio
async def test_parsers(temp_files):

    html_path, md_path = temp_files
    
    llm_client = LiteLLMClient(config_path="nonexistent_config.yaml")
    engine = RAGCoreEngine(llm_client)
    
    # Test HTML Parser
    text_html, pages_html = await engine.parse_file(str(html_path))
    assert "Hello World" in text_html
    assert "RAG test" in text_html
    assert len(pages_html) == 1
    
    # Test Markdown Parser
    text_md, pages_md = await engine.parse_file(str(md_path))
    assert "Markdown Title" in text_md
    assert "md document" in text_md
    assert len(pages_md) == 1


@pytest.mark.asyncio
async def test_semantic_chunking(monkeypatch):

    llm_client = LiteLLMClient(config_path="nonexistent_config.yaml")
    
    async def mock_aembedding(texts, *args, **kwargs):

        ret = []
        for text in texts:

            if "dog" in text:

                ret.append([1.0, 0.0, 0.0])
            else:

                ret.append([0.0, 1.0, 0.0])
        return ret

    monkeypatch.setattr(llm_client, "aembedding", mock_aembedding)
    engine = RAGCoreEngine(llm_client)
    
    pages = [{
        "page_number": 1,
        "text": "This is a dog. The dog barks. Cats are completely different. Cats meow."
    }]
    
    chunks = await engine.chunk_semantically(pages, k=0.1)
    assert len(chunks) >= 2
    assert any("dog" in c["text"] for c in chunks)
    assert any("Cats" in c["text"] for c in chunks)


@pytest.mark.asyncio
async def test_rag_config_and_score_scaling(monkeypatch, tmp_path):

    toml_file = tmp_path / "rag_config.toml"
    toml_file.write_text("""
[chunking]
max_chunk_size = 999
k = 0.5

[retrieval]
top_n = 2
use_hyde = false
rrf_k = 42
""")

    llm_client = LiteLLMClient(config_path="nonexistent_config.yaml")
    engine = RAGCoreEngine(llm_client, config_path=str(toml_file))

    assert engine.config["chunking"]["max_chunk_size"] == 999
    assert engine.config["chunking"]["k"] == 0.5
    assert engine.config["retrieval"]["top_n"] == 2
    assert engine.config["retrieval"]["rrf_k"] == 42

    engine.child_nodes = [{
        "child_id": "c1",
        "parent_id": "p1",
        "text": "sample document",
        "page_number": 1,
        "embedding": [1.0, 0.0, 0.0],
        "source": "dummy.txt",
        "filename": "dummy.txt"
    }]
    
    from rag_engine.core import Document
    engine.parent_store["p1"] = Document("p1", "full parent content", {})

    async def mock_aembedding(texts, *args, **kwargs):

        return [[1.0, 0.0, 0.0]]
    monkeypatch.setattr(llm_client, "aembedding", mock_aembedding)

    def mock_rerank(query, docs, top_n=5):

        return [
            {"document": docs[0], "index": 0, "score": -1.4340}
        ]
    monkeypatch.setattr(llm_client, "rerank", mock_rerank)

    results = await engine.search("dummy query", top_k_retrieval=1, top_k_llm=1)
    assert len(results) == 1
    score = results[0]["score"]
    assert 0.19 <= score <= 0.20
