import os
import json
import datetime
import asyncio
from typing import List, Dict, Any, Optional


class QueryLogger:

    def __init__(self, log_path: str = "logs/query_log.jsonl"):

        self.log_path = log_path
        self.lock = asyncio.Lock()
        
        # Ensure log folder exists
        log_dir = os.path.dirname(self.log_path)
        if log_dir:

            os.makedirs(log_dir, exist_ok=True)


    async def log_query(
        self,
        query: str,
        response: str,
        retrieved_nodes: List[Dict[str, Any]],
        latency_ms: float,
        metadata: Optional[Dict[str, Any]] = None
    ):

        log_entry = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "query": query,
            "response": response,
            "retrieved_nodes": [
                {
                    "text": node.get("text"),
                    "page_number": node.get("page_number"),
                    "source": node.get("source"),
                    "score": node.get("score")
                }
                for node in retrieved_nodes
            ],
            "latency_ms": latency_ms,
            "metadata": metadata or {}
        }
        
        async with self.lock:

            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._append_to_file, log_entry)


    def _append_to_file(self, entry: Dict[str, Any]):

        with open(self.log_path, "a", encoding="utf-8") as f:

            f.write(json.dumps(entry) + "\n")
