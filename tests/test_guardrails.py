import pytest
from unittest.mock import AsyncMock, MagicMock
from rag_engine import LiteLLMClient, GuardrailsManager


def test_validate_citations():

    llm_client = LiteLLMClient(config_path="nonexistent_config.yaml")
    guardrails = GuardrailsManager(llm_client)
    
    contexts = [
        {"text": "Python is a programming language.", "page_number": 1},
        {"text": "RAG helps ground LLMs.", "page_number": 3}
    ]
    
    # Test valid citations format (N numeric tokens, 1-based)
    response_valid = "Python is standard [1] and RAG is awesome [2]."
    is_ok, errs = guardrails.validate_citations(response_valid, contexts)
    assert is_ok
    assert len(errs) == 0
    
    # Test invalid index (3 > len=2)
    response_invalid_idx = "Python is standard [3]"
    is_ok, errs = guardrails.validate_citations(response_invalid_idx, contexts)
    assert not is_ok
    assert len(errs) == 1
    assert "range" in errs[0]["reason"]


@pytest.mark.asyncio
async def test_guardrails_self_correction_loop(monkeypatch):

    llm_client = LiteLLMClient(config_path="nonexistent_config.yaml")
    guardrails = GuardrailsManager(llm_client)
    
    contexts = [
        {"text": "Python is a programming language.", "page_number": 1}
    ]
    
    completion_calls = 0
    
    async def mock_acompletion(messages, *args, **kwargs):

        nonlocal completion_calls
        completion_calls += 1
        
        mock_response = MagicMock()
        mock_choice = MagicMock()
        
        if completion_calls == 1:

            mock_choice.message.content = "Initial answer [1]"
        elif completion_calls == 2:

            mock_choice.message.content = '{"faithful": false, "claims": [{"claim": "fact", "supported": false}]}'
        elif completion_calls == 3:

            mock_choice.message.content = "Corrected answer [1]"
        else:

            mock_choice.message.content = '{"faithful": true, "claims": []}'
            
        mock_response.choices = [mock_choice]
        return mock_response

    monkeypatch.setattr(llm_client, "acompletion", mock_acompletion)
    
    res = await guardrails.generate_faithful_answer("query text", contexts, max_attempts=3)
    assert res["faithful"]
    assert "Corrected answer" in res["answer"]
    assert res["attempts"] == 2


@pytest.mark.asyncio
async def test_guardrails_config_override(tmp_path):

    toml_file = tmp_path / "rag_config.toml"
    toml_file.write_text("""
[generation]
temperature = 0.85
max_tokens = 99

[guardrails]
max_attempts = 5
""")

    llm_client = LiteLLMClient(config_path="nonexistent_config.toml")
    guardrails = GuardrailsManager(llm_client, config_path=str(toml_file))

    assert guardrails.config["generation"]["temperature"] == 0.85
    assert guardrails.config["generation"]["max_tokens"] == 99
    assert guardrails.config["guardrails"]["max_attempts"] == 5


def test_strip_citations():

    from rag_engine.guardrails import strip_citations
    
    text = "Python is standard [1] and RAG is awesome [2]."
    assert strip_citations(text) == "Python is standard and RAG is awesome."
    
    text_simple = "Another sentence [1]."
    assert strip_citations(text_simple) == "Another sentence."
    
    text_no_space = "Text[1]"
    assert strip_citations(text_no_space) == "Text"
    
    assert strip_citations("") == ""
    assert strip_citations(None) == ""
