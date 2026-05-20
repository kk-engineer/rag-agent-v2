import os
import json
import logging
from typing import List, Dict, Any, Optional
from rag_engine.core import tokenize


logger = logging.getLogger(__name__)


class RagasEvaluator:

    def __init__(self, log_path: str = "logs/query_log.jsonl", llm_client: Optional[Any] = None):

        self.log_path = log_path
        self.llm_client = llm_client


    def load_dataset(self) -> List[Dict[str, Any]]:

        if not os.path.exists(self.log_path):

            logger.warning(f"Log file not found at {self.log_path}")
            return []

        dataset = []
        with open(self.log_path, "r", encoding="utf-8") as f:

            for line in f:

                if not line.strip():

                    continue

                try:

                    data = json.loads(line)
                    contexts = [node.get("text", "") for node in data.get("retrieved_nodes", [])]
                    
                    dataset.append({
                        "question": data.get("query", ""),
                        "answer": data.get("response", ""),
                        "contexts": contexts,
                        "ground_truth": data.get("metadata", {}).get("ground_truth", "")
                    })
                except Exception as e:

                    logger.error(f"Error parsing log line: {e}")
                    
        return dataset


    async def evaluate_ragas(self) -> Dict[str, Any]:

        data = self.load_dataset()
        if not data:

            return {"status": "No data found to evaluate."}

        try:

            from datasets import Dataset
            from ragas.evaluation import aevaluate
            from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
            from ragas.llms import LangchainLLMWrapper
            from ragas.embeddings import LangchainEmbeddingsWrapper
            from langchain_openai import ChatOpenAI
            from langchain_core.embeddings import Embeddings
            
            if not self.llm_client:
                from rag_engine.llm import LiteLLMClient
                self.llm_client = LiteLLMClient()
                
            toml_config = self.llm_client.toml_config
            
            def get_active_llm_config(toml):
                if not toml:
                    return {"model": "gpt-4o-mini", "api_base": None, "api_key": None}
                mode = toml.get("llm", {}).get("mode", "local")
                if mode == "local":
                    local_section = toml.get("llm", {}).get("local", {})
                    model_list = local_section.get("model_list", [])
                    for entry in model_list:
                        if entry.get("model_name") != "local-embedding":
                            params = entry.get("litellm_params", {})
                            return {
                                "model": params.get("model"),
                                "api_base": params.get("api_base"),
                                "api_key": params.get("api_key")
                            }
                else:
                    cloud_config = toml.get("llm", {}).get("cloud", {})
                    provider_order = cloud_config.get("provider_order", [])
                    env_key_names = {
                        "nvidia": "NVIDIA_API_KEY",
                        "gemini": "GEMINI_API_KEY",
                        "openrouter": "OPENROUTER_API_KEY",
                        "huggingface": "HF_TOKEN",
                        "deepseek": "DEEPSEEK_API_KEY",
                        "openai": "OPENAI_API_KEY",
                        "anthropic": "ANTHROPIC_API_KEY"
                    }
                    for provider in provider_order:
                        provider_settings = cloud_config.get(provider, {})
                        if not provider_settings:
                            continue
                        model = provider_settings.get("model")
                        base_url = provider_settings.get("base_url")
                        api_key = provider_settings.get("api_key")
                        
                        env_var_upper = env_key_names.get(provider.lower())
                        if not api_key and env_var_upper:
                            api_key = os.environ.get(env_var_upper) or os.environ.get(env_var_upper.lower())
                        if not api_key and provider.lower() == "huggingface":
                            api_key = os.environ.get("hf_api_key") or os.environ.get("HF_API_KEY") or os.environ.get("hf_token")
                            
                        if api_key:
                            return {
                                "model": model,
                                "api_base": base_url,
                                "api_key": api_key,
                                "provider": provider.lower()
                            }
                return {"model": "gpt-4o-mini", "api_base": None, "api_key": os.environ.get("OPENAI_API_KEY")}
                
            llm_config = get_active_llm_config(toml_config)
            model_name = llm_config.get("model")
            api_base = llm_config.get("api_base")
            api_key = llm_config.get("api_key")
            provider = llm_config.get("provider", "")
            
            if provider == "anthropic" and not api_base:
                try:
                    from langchain_anthropic import ChatAnthropic
                    chat_model = ChatAnthropic(model=model_name, api_key=api_key)
                except ImportError:
                    try:
                        from langchain_community.chat_models import ChatAnthropic
                        chat_model = ChatAnthropic(model=model_name, anthropic_api_key=api_key)
                    except ImportError:
                        chat_model = ChatOpenAI(model=model_name, openai_api_key=api_key)
            else:
                if api_base:
                    chat_model = ChatOpenAI(
                        model=model_name,
                        openai_api_base=api_base,
                        openai_api_key=api_key
                    )
                else:
                    chat_model = ChatOpenAI(
                        model=model_name,
                        openai_api_key=api_key
                    )
                    
            class SyncRagasEmbeddings(Embeddings):
                def __init__(self, client):
                    self.client = client
                    
                def embed_documents(self, texts: List[str]) -> List[List[float]]:
                    if self.client._local_embedder is not None:
                        return [emb.tolist() for emb in self.client._local_embedder.encode(texts)]
                    import asyncio, concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor(1) as pool:
                        return pool.submit(asyncio.run, self.client.aembedding(texts)).result()
                        
                def embed_query(self, text: str) -> List[float]:
                    res = self.embed_documents([text])
                    return res[0] if res else []
                    
            custom_embeddings = SyncRagasEmbeddings(self.llm_client)
            
            ragas_llm = LangchainLLMWrapper(chat_model)
            ragas_embeddings = LangchainEmbeddingsWrapper(custom_embeddings)
            
            formatted_data = {
                "question": [item["question"] for item in data],
                "user_input": [item["question"] for item in data],
                "answer": [item["answer"] for item in data],
                "response": [item["answer"] for item in data],
                "contexts": [item["contexts"] for item in data],
                "retrieved_contexts": [item["contexts"] for item in data],
            }
            
            has_ground_truth = all(item["ground_truth"] for item in data)
            if has_ground_truth:

                formatted_data["ground_truth"] = [item["ground_truth"] for item in data]
                formatted_data["reference"] = [item["ground_truth"] for item in data]
                metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
            else:

                metrics = [faithfulness, answer_relevancy]

            dataset = Dataset.from_dict(formatted_data)
            
            result = await aevaluate(
                dataset,
                metrics=metrics,
                llm=ragas_llm,
                embeddings=ragas_embeddings
            )
            
            scores = {}
            for m in metrics:
                vals = result[m.name]
                scores[m.name] = sum(vals) / len(vals) if vals else 0.0
            scores["evaluation_type"] = "ragas"
            all_metric_names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
            for name in all_metric_names:
                if name not in scores:
                    scores[name] = None
            return scores
        except Exception as e:

            logger.warning(f"Ragas evaluation failed (falling back to heuristics): {e}")
            return self._evaluate_heuristics(data)


    def _evaluate_heuristics(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:

        total_faithfulness = 0.0
        total_relevancy = 0.0
        total_precision = 0.0
        total_recall = 0.0
        count = len(dataset)

        for item in dataset:

            q_words = set(tokenize(item["question"]))
            a_words = set(tokenize(item["answer"]))
            
            context_text = " ".join(item["contexts"])
            c_words = set(tokenize(context_text))

            # 1. Faithfulness Proxy: overlap of answer words in context
            if a_words:

                faithfulness_score = len(a_words.intersection(c_words)) / len(a_words)
            else:

                faithfulness_score = 1.0

            # 2. Relevancy Proxy: overlap of answer words with question words
            if q_words:

                relevancy_score = len(a_words.intersection(q_words)) / len(q_words)
            else:

                relevancy_score = 1.0

            # 3. Context Precision Proxy: overlap of context words with question words
            if q_words:

                precision_score = len(c_words.intersection(q_words)) / len(q_words)
            else:

                precision_score = 1.0

            # 4. Context Recall Proxy: overlap of context words with ground truth/question words
            target_words = set(tokenize(item["ground_truth"])) if item["ground_truth"] else q_words
            if target_words:

                recall_score = len(c_words.intersection(target_words)) / len(target_words)
            else:

                recall_score = 1.0

            total_faithfulness += faithfulness_score
            total_relevancy += relevancy_score
            total_precision += precision_score
            total_recall += recall_score

        if count > 0:

            return {
                "faithfulness": total_faithfulness / count,
                "answer_relevancy": total_relevancy / count,
                "context_precision": total_precision / count,
                "context_recall": total_recall / count,
                "evaluation_type": "heuristics_fallback"
            }
        else:

            return {
                "faithfulness": 0.0,
                "answer_relevancy": 0.0,
                "context_precision": 0.0,
                "context_recall": 0.0,
                "evaluation_type": "heuristics_fallback"
            }
