import os
import pytest
from rag_engine.evaluation import RagasEvaluator
from rag_engine import QueryLogger


@pytest.mark.asyncio
async def test_ragas_evaluator_fallback(tmp_path):

    log_file = tmp_path / "query_log.jsonl"
    
    logger_instance = QueryLogger(log_path=str(log_file))
    retrieved = [
        {"text": "Python is an interpreted programming language.", "page_number": 1, "source": "python.txt"}
    ]
    
    await logger_instance.log_query(
        query="What is python?",
        response="Python is a programming language.",
        retrieved_nodes=retrieved,
        latency_ms=10.0
    )
    
    evaluator = RagasEvaluator(log_path=str(log_file))
    scores = await evaluator.evaluate_ragas()
    
    assert scores["evaluation_type"] == "ragas"
    assert scores["faithfulness"] > 0.0
    assert scores["answer_relevancy"] > 0.0


@pytest.mark.asyncio
async def test_ragas_evaluator_success(tmp_path, monkeypatch):

    log_file = tmp_path / "query_log.jsonl"
    logger_instance = QueryLogger(log_path=str(log_file))
    retrieved = [
        {"text": "Python is an interpreted programming language.", "page_number": 1, "source": "python.txt"}
    ]
    await logger_instance.log_query(
        query="What is python?",
        response="Python is a programming language.",
        retrieved_nodes=retrieved,
        latency_ms=10.0,
        metadata={"ground_truth": "Python is a programming language."}
    )

    evaluated_args = {}
    async def mock_aevaluate(dataset, metrics, llm, embeddings):

        evaluated_args["dataset"] = dataset
        evaluated_args["metrics"] = metrics
        evaluated_args["llm"] = llm
        evaluated_args["embeddings"] = embeddings
        return {"faithfulness": [0.95], "answer_relevancy": [0.9], "context_precision": [0.85], "context_recall": [0.8]}

    monkeypatch.setattr("ragas.evaluation.aevaluate", mock_aevaluate)

    class FakeLiteLLMClient:

        toml_config = {
            "llm": {
                "mode": "cloud",
                "cloud": {
                    "provider_order": ["openai"],
                    "openai": {
                        "model": "gpt-4o-mini",
                        "api_key": "fake-openai-key"
                    }
                }
            }
        }
        
        async def aembedding(self, texts):

            return [[0.1] * 1536]

    fake_client = FakeLiteLLMClient()
    evaluator = RagasEvaluator(log_path=str(log_file), llm_client=fake_client)
    scores = await evaluator.evaluate_ragas()

    assert scores["faithfulness"] == 0.95
    assert scores["answer_relevancy"] == 0.9
    assert scores["context_precision"] == 0.85
    assert scores["context_recall"] == 0.8
    assert scores["evaluation_type"] == "ragas"
    assert "dataset" in evaluated_args
