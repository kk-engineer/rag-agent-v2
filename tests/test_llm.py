import pytest
from unittest.mock import AsyncMock, MagicMock
from rag_engine import LiteLLMClient


@pytest.mark.asyncio
async def test_litellm_client_fallback_embedding():

    client = LiteLLMClient(config_path="nonexistent_config.toml")
    
    embeddings = await client.aembedding("test text string")
    assert len(embeddings) == 1
    assert len(embeddings[0]) == 384


@pytest.mark.asyncio
async def test_litellm_client_acompletion_mock(monkeypatch):

    client = LiteLLMClient(config_path="nonexistent_config.toml")
    
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "mocked response content"
    mock_response.choices = [mock_choice]
    
    async def mock_acompletion(*args, **kwargs):

        return mock_response

    monkeypatch.setattr(client.router, "acompletion", mock_acompletion)
    
    res = await client.acompletion(messages=[{"role": "user", "content": "hello"}])
    assert res.choices[0].message.content == "mocked response content"


def test_litellm_client_rerank():

    client = LiteLLMClient(config_path="nonexistent_config.toml")
    docs = ["This is a dog document", "That is a cat document", "Computer programming"]
    query = "cat document"
    
    results = client.rerank(query, docs)
    assert len(results) > 0
    assert "cat" in results[0]["document"]


@pytest.mark.asyncio
async def test_litellm_client_cloud_fallback(monkeypatch):

    client = LiteLLMClient(config_path="nonexistent_config.toml")
    client.execution_mode = "cloud"
    client.toml_config = {
        "llm": {
            "mode": "cloud",
            "cloud": {
                "timeout": 0.1,
                "provider_order": ["nvidia", "gemini"],
                "nvidia": {
                    "model": "meta/llama-3.1-8b-instruct",
                    "base_url": "https://integrate.api.nvidia.com/v1"
                },
                "gemini": {
                    "model": "gemini-2.0-flash",
                    "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/"
                }
            }
        }
    }

    call_history = []

    async def mock_acompletion(*args, **kwargs):

        model = kwargs.get("model")
        call_history.append(model)
        if "llama" in model:

            raise RuntimeError("Nvidia failed")
            
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Gemini answer"
        mock_response.choices = [mock_choice]
        return mock_response

    monkeypatch.setattr("litellm.acompletion", mock_acompletion)

    res = await client.acompletion(messages=[{"role": "user", "content": "hi"}])
    assert res.choices[0].message.content == "Gemini answer"
    assert len(call_history) == 2
    assert "llama" in call_history[0]
    assert "gemini" in call_history[1]


@pytest.mark.asyncio
async def test_litellm_client_use_local_hf_embedder():

    client = LiteLLMClient(config_path="nonexistent_config.toml")
    client.rag_config = {
        "retrieval": {
            "use_local_hf_embedder": True
        }
    }
    
    # Mock self._local_embedder
    mock_embedder = MagicMock()
    mock_emb = MagicMock()
    mock_emb.tolist.return_value = [0.1] * 384
    mock_embedder.encode.return_value = [mock_emb]
    client._local_embedder = mock_embedder
    
    embeddings = await client.aembedding("hello world")
    assert mock_embedder.encode.called
    assert embeddings == [[0.1] * 384]


@pytest.mark.asyncio
async def test_litellm_client_lowercase_env_vars(monkeypatch):

    import os
    client = LiteLLMClient(config_path="nonexistent_config.toml")
    client.execution_mode = "cloud"
    client.toml_config = {
        "llm": {
            "mode": "cloud",
            "cloud": {
                "timeout": 10,
                "provider_order": ["nvidia"],
                "nvidia": {
                    "model": "meta/llama-3.1-8b-instruct",
                    "base_url": "https://integrate.api.nvidia.com/v1"
                }
            }
        }
    }

    # Set lowercase env variable
    monkeypatch.setenv("nvidia_api_key", "test-key-nvidia-lowercased")
    
    # Ensure standard env var starts clean
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    async def mock_acompletion(*args, **kwargs):

        assert kwargs.get("api_key") == "test-key-nvidia-lowercased"
        assert os.environ.get("NVIDIA_API_KEY") == "test-key-nvidia-lowercased"
        assert os.environ.get("OPENAI_API_KEY") == "test-key-nvidia-lowercased"
        
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Success"
        mock_response.choices = [mock_choice]
        return mock_response

    monkeypatch.setattr("litellm.acompletion", mock_acompletion)
    
    res = await client.acompletion(messages=[{"role": "user", "content": "hi"}])
    assert res.choices[0].message.content == "Success"


@pytest.mark.asyncio
async def test_litellm_client_cloud_embedding_fallback(monkeypatch):

    client = LiteLLMClient(config_path="nonexistent_config.toml")
    client.toml_config = {
        "embedding": {
            "mode": "cloud",
            "use_local_hf_embedder": False,
            "cloud": {
                "timeout": 0.1,
                "provider_order": ["openai", "nvidia"],
                "openai": {
                    "model": "text-embedding-3-small"
                },
                "nvidia": {
                    "model": "nvidia/embeddings-nv-embed-qa-4",
                    "base_url": "https://integrate.api.nvidia.com/v1"
                }
            }
        }
    }

    call_history = []

    async def mock_aembedding(*args, **kwargs):

        model = kwargs.get("model")
        call_history.append(model)
        if "openai" in model or "text-embedding" in model:

            raise RuntimeError("OpenAI failed")
            
        return {
            "data": [
                {"embedding": [0.2] * 384}
            ]
        }

    monkeypatch.setattr("litellm.aembedding", mock_aembedding)
    monkeypatch.setenv("nvidia_api_key", "test-nvidia-key")

    res = await client.aembedding("test text")
    assert res == [[0.2] * 384]
    assert len(call_history) == 2
    assert "text-embedding-3-small" in call_history[0]
    assert "nvidia/embeddings-nv-embed-qa-4" in call_history[1]
