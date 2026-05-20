import os
import json
import pytest
from rag_engine import QueryLogger


@pytest.mark.asyncio
async def test_query_logger(tmp_path):

    log_file = tmp_path / "test_logs.jsonl"
    logger_instance = QueryLogger(log_path=str(log_file))
    
    retrieved = [
        {"text": "python text content", "page_number": 2, "source": "abc.txt", "score": 0.95}
    ]
    
    await logger_instance.log_query(
        query="What is python?",
        response="Python is coding",
        retrieved_nodes=retrieved,
        latency_ms=150.5,
        metadata={"test": True}
    )
    
    assert os.path.exists(log_file)
    with open(log_file, "r") as f:

        lines = f.readlines()
        assert len(lines) == 1
        
        data = json.loads(lines[0].strip())
        assert data["query"] == "What is python?"
        assert data["response"] == "Python is coding"
        assert len(data["retrieved_nodes"]) == 1
        assert data["retrieved_nodes"][0]["page_number"] == 2
        assert data["latency_ms"] == 150.5
        assert data["metadata"]["test"] is True
