import os
import time
import re
import asyncio
import uuid
import logging

try:
    from opensmith import trace
except ImportError:
    def trace(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]
        def decorator(f):
            return f
        return decorator
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Callable
from bs4 import BeautifulSoup
import markdown
from pypdf import PdfReader
from rank_bm25 import BM25Okapi

from rag_engine.prompts import HYDE_GENERATION_PROMPT
from rag_engine.llm import LiteLLMClient


logger = logging.getLogger(__name__)


def clean_html(html_content: str) -> str:

    soup = BeautifulSoup(html_content, "html.parser")
    for script_or_style in soup(["script", "style"]):

        script_or_style.decompose()
    return soup.get_text(separator=" ")


def clean_markdown(md_content: str) -> str:

    html = markdown.markdown(md_content)
    return clean_html(html)


def tokenize(text: str) -> List[str]:

    return re.findall(r"\w+", text.lower())


class Document:

    def __init__(self, doc_id: str, content: str, metadata: Dict[str, Any]):

        self.doc_id = doc_id
        self.content = content
        self.metadata = metadata


class RAGCoreEngine:

    def __init__(self, llm_client: LiteLLMClient, config_path: str = "config/rag_config.toml"):

        self.llm_client = llm_client
        self.parent_store: Dict[str, Document] = {}
        self.child_nodes: List[Dict[str, Any]] = []
        self.bm25: Optional[BM25Okapi] = None
        self.lock = asyncio.Lock()
        
        # Load RAG configuration
        self.config = {
            "chunking": {
                "max_chunk_size": 1500,
                "k": 1.0
            },
            "retrieval": {
                "top_n": 5,
                "use_hyde": False,
                "rrf_k": 60,
                "rrf_weight_sparse": 1.0,
                "rrf_weight_dense": 1.0
            },
            "generation": {
                "temperature": 0.0,
                "max_tokens": 512
            },
            "guardrails": {
                "max_attempts": 3
            },
            "models": {
                "embedder_repo_id": "sentence-transformers/all-MiniLM-L6-v2",
                "embedder_local_dir": "local_models/all-MiniLM-L6-v2",
                "reranker_repo_id": "cross-encoder/ms-marco-MiniLM-L-6-v2",
                "reranker_local_dir": "local_models/ms-marco-MiniLM-L-6-v2"
            }
        }
        
        if os.path.exists(config_path):

            try:

                with open(config_path, "rb") as f:

                    import tomllib
                    loaded = tomllib.load(f)
                    for section in ["chunking", "retrieval", "generation", "guardrails", "models"]:

                        if section in loaded:

                            self.config[section].update(loaded[section])
                logger.info(f"Loaded RAG configuration from {config_path}")
            except Exception as e:

                logger.warning(f"Failed to load RAG config: {e}")


    async def parse_file(self, file_path: str) -> Tuple[str, List[Dict[str, Any]]]:

        # Parses file and extracts page-level text
        if not os.path.exists(file_path):

            raise FileNotFoundError(f"File not found: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()
        full_text = ""
        pages = []

        if ext == ".pdf":

            reader = PdfReader(file_path)
            for idx, page in enumerate(reader.pages):

                page_text = page.extract_text() or ""
                full_text += page_text + "\n"
                pages.append({
                    "page_number": idx + 1,
                    "text": page_text
                })

        elif ext in [".html", ".htm"]:

            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:

                content = f.read()
                cleaned = clean_html(content)
                full_text = cleaned
                pages.append({
                    "page_number": 1,
                    "text": cleaned
                })

        elif ext in [".md", ".markdown"]:

            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:

                content = f.read()
                cleaned = clean_markdown(content)
                full_text = cleaned
                pages.append({
                    "page_number": 1,
                    "text": cleaned
                })

        else:

            # Treat as source code or plain text
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:

                content = f.read()
                full_text = content
                pages.append({
                    "page_number": 1,
                    "text": content
                })

        return full_text, pages


    async def chunk_semantically(
        self,
        pages: List[Dict[str, Any]],
        k: Optional[float] = None,
        max_chunk_size: Optional[int] = None
    ) -> List[Dict[str, Any]]:

        if k is None:

            k = self.config["chunking"]["k"]
        if max_chunk_size is None:

            max_chunk_size = self.config["chunking"]["max_chunk_size"]

        semantic_chunks = []
        
        for page in pages:

            page_number = page["page_number"]
            text = page["text"]
            
            # Split text into sentences
            sentence_splits = re.split(r"(?<=[.!?])\s+", text)
            sentences = [s.strip() for s in sentence_splits if s.strip()]
            if not sentences:

                continue

            if len(sentences) == 1:

                semantic_chunks.append({
                    "text": sentences[0],
                    "page_number": page_number
                })
                continue

            # Embed each sentence
            embeddings = await self.llm_client.aembedding(sentences)
            
            # Compute cosine distances between adjacent sentence embeddings
            distances = []
            for i in range(len(embeddings) - 1):

                v1 = np.array(embeddings[i])
                v2 = np.array(embeddings[i+1])
                n1 = np.linalg.norm(v1)
                n2 = np.linalg.norm(v2)
                if n1 > 0 and n2 > 0:

                    cos_sim = np.dot(v1, v2) / (n1 * n2)
                    distances.append(1.0 - cos_sim)
                else:

                    distances.append(1.0)

            # Detect spikes in distance (threshold = mean + k * std)
            if distances:

                mean_dist = np.mean(distances)
                std_dist = np.std(distances)
                threshold = mean_dist + k * std_dist
            else:

                threshold = 1.0

            # Group sentences into chunks based on distance spikes
            current_chunk = [sentences[0]]
            for i, dist in enumerate(distances):

                current_text = " ".join(current_chunk)
                # Split if distance spikes, or if the chunk starts to exceed size limit
                if dist > threshold or len(current_text) > max_chunk_size:

                    semantic_chunks.append({
                        "text": current_text,
                        "page_number": page_number
                    })
                    current_chunk = [sentences[i+1]]
                else:

                    current_chunk.append(sentences[i+1])

            if current_chunk:

                semantic_chunks.append({
                    "text": " ".join(current_chunk),
                    "page_number": page_number
                })

        return semantic_chunks


    @trace(tags=["core_engine"])
    async def ingest_file(self, file_path: str) -> str:

        async with self.lock:

            doc_start_time = time.time()
            filename = os.path.basename(file_path)
            logger.info(f"Starting ingestion for file: {filename}")
            
            # Step 1: Parse file
            parse_start = time.time()
            full_text, pages = await self.parse_file(file_path)
            parse_duration = time.time() - parse_start
            logger.info(f"  [Step 1/4] Parsed file '{filename}' into {len(pages)} pages. Time taken: {parse_duration:.3f}s")
            
            doc_id = str(uuid.uuid4())
            # Save parent document
            doc = Document(
                doc_id=doc_id,
                content=full_text,
                metadata={"source": file_path, "filename": filename}
            )
            self.parent_store[doc_id] = doc

            # Step 2: Semantic Chunking
            chunk_start = time.time()
            chunks = await self.chunk_semantically(pages)
            chunk_duration = time.time() - chunk_start
            logger.info(f"  [Step 2/4] Semantically chunked '{filename}' into {len(chunks)} chunks. Time taken: {chunk_duration:.3f}s")
            
            # Step 3: Embed child chunks and save them
            embed_duration = 0.0
            chunk_texts = [c["text"] for c in chunks]
            if chunk_texts:

                embed_start = time.time()
                embeddings = await self.llm_client.aembedding(chunk_texts)
                embed_duration = time.time() - embed_start
                logger.info(f"  [Step 3/4] Generated embeddings for {len(chunk_texts)} chunks of '{filename}'. Time taken: {embed_duration:.3f}s")
                
                for idx, chunk in enumerate(chunks):

                    child_id = str(uuid.uuid4())
                    self.child_nodes.append({
                        "child_id": child_id,
                        "parent_id": doc_id,
                        "text": chunk["text"],
                        "page_number": chunk["page_number"],
                        "embedding": embeddings[idx],
                        "source": file_path,
                        "filename": filename
                    })

            # Step 4: Update BM25 Index
            bm25_start = time.time()
            self._update_bm25_index()
            bm25_duration = time.time() - bm25_start
            logger.info(f"  [Step 4/4] Updated BM25 index. Time taken: {bm25_duration:.3f}s")
            
            total_duration = time.time() - doc_start_time
            logger.info(f"Successfully ingested file '{filename}' in {total_duration:.3f}s (Parsed: {parse_duration:.3f}s, Chunked: {chunk_duration:.3f}s, Embedded: {embed_duration:.3f}s, BM25 Indexed: {bm25_duration:.3f}s)")
            
            return doc_id


    def _update_bm25_index(self):

        if not self.child_nodes:

            self.bm25 = None
            return

        corpus = [tokenize(node["text"]) for node in self.child_nodes]
        self.bm25 = BM25Okapi(corpus)


    async def retrieve_hyde_query_embedding(self, query: str) -> List[float]:

        # HyDE expansion: Generate hypothetical document
        prompt = HYDE_GENERATION_PROMPT.format(query=query)
        messages = [{"role": "user", "content": prompt}]
        response = await self.llm_client.acompletion(
            messages=messages,
            model="local-llm",
            temperature=0.7,
            max_tokens=256
        )
        hyde_doc = response.choices[0].message.content
        
        # Embed the generated hypothetical document
        hyde_embeddings = await self.llm_client.aembedding(hyde_doc)
        return hyde_embeddings[0]


    @trace(tags=["core_engine"])
    async def search(
        self,
        query: str,
        top_n: Optional[int] = None,
        use_hyde: Optional[bool] = None,
        rrf_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:

        if not self.child_nodes:

            return []

        search_start = time.time()
        logger.info(f"Initiating multi-vector search for query: '{query}'")

        if top_n is None:

            top_n = self.config["retrieval"]["top_n"]
        if use_hyde is None:

            use_hyde = self.config["retrieval"]["use_hyde"]
        if rrf_k is None:

            rrf_k = self.config["retrieval"]["rrf_k"]

        # 1. Sparse (BM25) Retrieve
        sparse_start = time.time()
        bm25_scores = []
        if self.bm25:

            tokenized_query = tokenize(query)
            bm25_scores = self.bm25.get_scores(tokenized_query)
        sparse_duration = time.time() - sparse_start
        logger.info(f"  [Search Step 1/4] Sparse BM25 retrieval completed. Time taken: {sparse_duration:.3f}s")

        # 2. Dense (Vector) Retrieve
        dense_start = time.time()
        if use_hyde:

            hyde_start = time.time()
            query_vector = await self.retrieve_hyde_query_embedding(query)
            hyde_duration = time.time() - hyde_start
            logger.info(f"    - Generated HyDE query expansion & embedding. Time taken: {hyde_duration:.3f}s")
        else:

            query_vectors = await self.llm_client.aembedding(query)
            query_vector = query_vectors[0]

        vector_scores = []
        qv = np.array(query_vector)
        q_norm = np.linalg.norm(qv)
        
        for node in self.child_nodes:

            nv = np.array(node["embedding"])
            n_norm = np.linalg.norm(nv)
            if q_norm > 0 and n_norm > 0:

                cos_sim = np.dot(qv, nv) / (q_norm * n_norm)
                vector_scores.append(float(cos_sim))
            else:

                vector_scores.append(0.0)
        dense_duration = time.time() - dense_start
        logger.info(f"  [Search Step 2/4] Dense vector similarity retrieval completed. Time taken: {dense_duration:.3f}s")

        # 3. Reciprocal Rank Fusion (RRF)
        rrf_start = time.time()
        # Ranks from sparse
        sparse_ranks = np.argsort(np.argsort(-np.array(bm25_scores))) if len(bm25_scores) > 0 else []
        # Ranks from dense
        dense_ranks = np.argsort(np.argsort(-np.array(vector_scores)))

        rrf_weight_sparse = self.config["retrieval"].get("rrf_weight_sparse", 1.0)
        rrf_weight_dense = self.config["retrieval"].get("rrf_weight_dense", 1.0)

        rrf_scores = {}
        for idx, node in enumerate(self.child_nodes):

            child_id = node["child_id"]
            rrf_score = 0.0
            
            # Sparse rank contribution
            if len(sparse_ranks) > 0:

                rank_s = sparse_ranks[idx] + 1  # 1-based rank
                rrf_score += rrf_weight_sparse / (rrf_k + rank_s)
                
            # Dense rank contribution
            rank_d = dense_ranks[idx] + 1
            rrf_score += rrf_weight_dense / (rrf_k + rank_d)
            
            rrf_scores[child_id] = rrf_score

        # Combine results, sort by RRF score
        scored_nodes = []
        for node in self.child_nodes:

            cid = node["child_id"]
            scored_nodes.append({
                "node": node,
                "rrf_score": rrf_scores[cid]
            })

        scored_nodes.sort(key=lambda x: x["rrf_score"], reverse=True)
        top_nodes = scored_nodes[:top_n]
        rrf_duration = time.time() - rrf_start
        logger.info(f"  [Search Step 3/4] RRF fusion complete (Top {len(top_nodes)} candidate nodes). Time taken: {rrf_duration:.3f}s")

        # 4. Rerank nodes using LLM interface rerank
        rerank_start = time.time()
        node_texts = [item["node"]["text"] for item in top_nodes]
        reranked = self.llm_client.rerank(query, node_texts, top_n=top_n)
        rerank_duration = time.time() - rerank_start
        logger.info(f"  [Search Step 4/4] Cohere/LiteLLM Reranker completed. Time taken: {rerank_duration:.3f}s")

        # Construct final output list with Parent-Document contexts
        final_results = []
        for rank_item in reranked:

            original_idx = rank_item["index"]
            matched_node = top_nodes[original_idx]["node"]
            parent_id = matched_node["parent_id"]
            parent_doc = self.parent_store[parent_id]
            
            raw_score = rank_item["score"]
            # Scale score to 0.0-1.0 using sigmoid if it is logit-based (outside [0, 1] or negative)
            if 0.0 <= raw_score <= 1.0:

                scaled_score = raw_score
            else:

                import math
                try:

                    scaled_score = 1.0 / (1.0 + math.exp(-raw_score))
                except Exception:

                    scaled_score = 0.0

            final_results.append({
                "text": matched_node["text"],
                "page_number": matched_node["page_number"],
                "source": matched_node["source"],
                "filename": matched_node["filename"],
                "score": float(scaled_score),
                "parent_content": parent_doc.content,
                "parent_id": parent_id
            })

        search_total_duration = time.time() - search_start
        logger.info(f"Successfully completed multi-vector search in {search_total_duration:.3f}s (Sparse: {sparse_duration:.3f}s, Dense: {dense_duration:.3f}s, RRF: {rrf_duration:.3f}s, Rerank: {rerank_duration:.3f}s)")
        
        return final_results
