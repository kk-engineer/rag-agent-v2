import json
import logging
import time

try:
    from opensmith import trace
except ImportError:
    def trace(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]
        def decorator(f):
            return f
        return decorator
from typing import Dict, Any, Optional
from rag_engine.prompts import QUERY_ROUTER_PROMPT
from rag_engine.llm import LiteLLMClient


logger = logging.getLogger(__name__)


class QueryRouter:

    def __init__(self, llm_client: LiteLLMClient):

        self.llm_client = llm_client

    @trace(tags=["query_router", "route_query"])
    async def route_query(
        self,
        user_query: str,
        metrics_collector: Optional[Any] = None,
        model: str = "local-llm",
    ) -> Dict[str, Any]:

        prompt = QUERY_ROUTER_PROMPT.format(user_query=user_query)
        messages = [{"role": "user", "content": prompt}]

        try:
            kwargs = {"temperature": 0.0, "max_tokens": 128}
            route_start = time.time()
            logger.debug(f"\033[1;33m[QUERY ROUTER]\033[0m Input: user_query='{user_query}'")
            response = await self.llm_client.acompletion(
                messages, model=model,
                metrics_collector=metrics_collector,
                metrics_purpose="QUERY ROUTER",
                **kwargs
            )
            route_elapsed = time.time() - route_start
            usage = getattr(response, "usage", None)
            pt = usage.prompt_tokens if usage else 0
            ct = usage.completion_tokens if usage else 0
            tt = usage.total_tokens if usage else 0
            raw = response.choices[0].message.content.strip()
            result = json.loads(raw)
            route = result.get("route", "RAG_RETRIEVAL")
            if route not in ("RAG_RETRIEVAL", "DIRECT_LLM"):
                route = "RAG_RETRIEVAL"
            logger.debug(f"\033[1;33m[QUERY ROUTER]\033[0m Output: route={route}, reasoning='{result.get('reasoning', '')}', raw={raw}")
            logger.info(
                f"\033[1;33m[QUERY ROUTER]\033[0m "
                f"\033[1;33m{model}\033[0m "
                f"\033[1;32m[Tokens: {tt} (In={pt}, Out={ct})]\033[0m "
                f"{route} — {result.get('reasoning', '')} | time: {route_elapsed:.3f}s"
            )
            return {"route": route, "reasoning": result.get("reasoning", "")}
        except Exception as e:
            logger.warning(
                f"\033[1;33m[QUERY ROUTER]\033[0m Failed for query='{user_query[:60]}...': {e} — defaulting to RAG_RETRIEVAL"
            )
            return {"route": "RAG_RETRIEVAL", "reasoning": "Fallback: router error"}
