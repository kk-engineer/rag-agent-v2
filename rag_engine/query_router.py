import json
import logging
from typing import Dict, Any
from rag_engine.prompts import QUERY_ROUTER_PROMPT
from rag_engine.llm import LiteLLMClient


logger = logging.getLogger(__name__)


class QueryRouter:

    def __init__(self, llm_client: LiteLLMClient):

        self.llm_client = llm_client

    async def route_query(self, user_query: str) -> Dict[str, Any]:

        prompt = QUERY_ROUTER_PROMPT.format(user_query=user_query)
        messages = [{"role": "user", "content": prompt}]

        try:
            kwargs = {"temperature": 0.0, "max_tokens": 128}
            response = await self.llm_client.acompletion(
                messages, model="local-llm", **kwargs
            )
            raw = response.choices[0].message.content.strip()
            result = json.loads(raw)
            route = result.get("route", "RAG_RETRIEVAL")
            if route not in ("RAG_RETRIEVAL", "DIRECT_LLM"):
                route = "RAG_RETRIEVAL"
            logger.info(f"Query route: {route} — {result.get('reasoning', '')}")
            return {"route": route, "reasoning": result.get("reasoning", "")}
        except Exception as e:
            logger.warning(
                f"Router failed for query='{user_query[:60]}...': {e} — defaulting to RAG_RETRIEVAL"
            )
            return {"route": "RAG_RETRIEVAL", "reasoning": "Fallback: router error"}
