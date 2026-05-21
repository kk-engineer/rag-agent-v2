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
from rag_engine.ingestion import sha256_file, get_mtime


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
        self.vector_store = None
        self._ingestion = None
        
        # Load RAG configuration
        self.config = {
            "chunking": {
                "max_chunk_size": 1500,
                "k": 1.0
            },
            "retrieval": {
                "top_k_retrieval": 35,
                "top_k_llm": 5,
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
            },
            "vector_store": {
                "mode": "in-memory",
                "persist_dir": "chroma_db",
                "collection_name": "rag_documents"
            }
        }
        
        if os.path.exists(config_path):

            try:

                with open(config_path, "rb") as f:

                    import tomllib
                    loaded = tomllib.load(f)
                    for section in ["chunking", "retrieval", "generation", "guardrails", "models", "vector_store"]:

                        if section in loaded:

                            self.config[section].update(loaded[section])
                logger.info(f"Loaded RAG configuration from {config_path}")
            except Exception as e:

                logger.warning(f"Failed to load RAG config: {e}")

        self.vs_mode = self.config.get("vector_store", {}).get("mode", "in-memory")
        if self.vs_mode == "persist":

            self._init_persist()


    def _init_persist(self):

        vs_cfg = self.config.get("vector_store", {})
        persist_dir = vs_cfg.get("persist_dir", "chroma_db")
        collection_name = vs_cfg.get("collection_name", "rag_documents")

        from rag_engine.vector_store import VectorStore

        self.vector_store = VectorStore(
            persist_dir=persist_dir, collection_name=collection_name
        )

        model_name = self.config.get("models", {}).get(
            "embedder_repo_id", "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vector_store.initialize(embedding_model=model_name)

        if not self.vector_store.is_data_valid(model_name):

            logger.warning(
                f"Embedding model mismatch in persisted store. "
                f"Run IngestionCoordinator.full_reindex() to rebuild."
            )

        self.child_nodes = self.vector_store.get_all_chunks()

        # defensive dedup: keep first occurrence of (text, source, page_number)
        seen = set()
        unique = []
        dup_ids = []
        for node in self.child_nodes:
            key = (node.get("text"), node.get("source"), node.get("page_number"))
            if key in seen:
                cid = node.get("child_id")
                if cid:
                    dup_ids.append(cid)
            else:
                seen.add(key)
                unique.append(node)
        if dup_ids:
            logger.warning(
                f"Found {len(dup_ids)} duplicate chunks in persisted store "
                f"— keeping first occurrence, removing {len(dup_ids)} from ChromaDB"
            )
            self.child_nodes = unique
            try:
                self.vector_store.collection.delete(ids=dup_ids)
            except Exception as e:
                logger.warning(f"Failed to clean duplicates from ChromaDB: {e}")

        parents_data = self.vector_store.load_parents()
        self.parent_store = {}
        for pid, pdata in parents_data.items():

            from rag_engine.core import Document
            self.parent_store[pid] = Document(
                pid, pdata["content"], pdata.get("metadata", {})
            )

        self._update_bm25_index()
        logger.info(
            f"Persist mode initialized: {len(self.child_nodes)} chunks, "
            f"{len(self.parent_store)} parents"
        )


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


    @trace(tags=["core_engine", "semantic_chunking"])
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


    def _purge_file(self, file_path: str):

        before = len(self.child_nodes)
        self.child_nodes = [
            n for n in self.child_nodes if n.get("source") != file_path
        ]
        self.parent_store = {
            pid: doc for pid, doc in self.parent_store.items()
            if doc.metadata.get("source") != file_path
        }
        removed = before - len(self.child_nodes)
        if removed:
            logger.info(f"Purged {removed} stale chunks for '{os.path.basename(file_path)}'")
            self._update_bm25_index()

    def deduplicate(self) -> dict:

        seen = set()
        keep = []
        dup_ids = []
        for node in self.child_nodes:
            key = (node.get("text"), node.get("source"), node.get("page_number"))
            if key in seen:
                dup_ids.append(node.get("child_id"))
            else:
                seen.add(key)
                keep.append(node)

        before = len(self.child_nodes)
        self.child_nodes = keep

        if dup_ids and self.vs_mode == "persist" and self.vector_store:
            try:
                self.vector_store.collection.delete(ids=dup_ids)
                logger.info(f"Deleted {len(dup_ids)} duplicate IDs from ChromaDB")
            except Exception as e:
                logger.warning(f"Failed to delete duplicates from ChromaDB: {e}")

        if dup_ids:
            self._update_bm25_index()

        result = {"removed": len(dup_ids), "remaining": len(self.child_nodes)}
        logger.info(f"Deduplication complete: removed={result['removed']}, remaining={result['remaining']}")
        return result

    def clean_database(self) -> dict:

        before = len(self.child_nodes)
        self.child_nodes = []
        self.parent_store = {}
        self.bm25 = None

        if self.vs_mode == "persist" and self.vector_store:
            self.vector_store.delete_all()

        logger.info(f"Database cleaned: removed {before} chunks")
        return {"cleared": before}

    @trace(tags=["core_engine"])
    async def ingest_file(self, file_path: str) -> str:

        if self.vs_mode == "persist":

            return await self._ingest_file_persist(file_path)

        async with self.lock:

            self._purge_file(file_path)

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

                    child_id = f"{doc_id}#chunk_{idx}"
                    self.child_nodes.append({
                        "child_id": child_id,
                        "chunk_id": child_id,
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


    @trace(tags=["core_engine", "persist_ingestion"])
    async def _ingest_file_persist(self, file_path: str) -> str:

        async with self.lock:

            ledger = self.vector_store.get_ledger()
            entry = ledger.get(str(file_path))
            if entry:
                old_id = entry["file_id"]
                self.vector_store.delete_document(old_id)
                self.vector_store.remove_ledger_entry(str(file_path))
                self.child_nodes = [
                    n for n in self.child_nodes
                    if n.get("parent_id") != old_id and n.get("file_id") != old_id
                ]
                self.parent_store.pop(old_id, None)
                logger.info(f"Removed stale persist data for file_id={old_id}")

            doc_start_time = time.time()
            filename = os.path.basename(file_path)
            logger.info(f"[Persist] Starting ingestion for file: {filename}")
            doc_id = str(uuid.uuid4())

            parse_start = time.time()
            full_text, pages = await self.parse_file(file_path)
            parse_duration = time.time() - parse_start

            chunk_start = time.time()
            chunks = await self.chunk_semantically(pages)
            chunk_duration = time.time() - chunk_start

            embed_start = time.time()
            chunk_texts = [c["text"] for c in chunks]
            embeddings = await self.llm_client.aembedding(chunk_texts)
            embed_duration = time.time() - embed_start

            chunk_ids = [f"{doc_id}#chunk_{i}" for i in range(len(chunks))]
            self.vector_store.add_chunks_batch(
                chunks=chunks,
                chunk_ids=chunk_ids,
                embeddings=embeddings,
                file_id=doc_id,
                source=file_path,
                filename=filename,
            )

            doc = Document(
                doc_id=doc_id,
                content=full_text,
                metadata={"source": file_path, "filename": filename},
            )
            self.parent_store[doc_id] = doc
            self.vector_store.save_parents(self.parent_store)

            for i, chunk in enumerate(chunks):
                self.child_nodes.append({
                    "child_id": chunk_ids[i],
                    "parent_id": doc_id,
                    "file_id": doc_id,
                    "text": chunk["text"],
                    "page_number": chunk["page_number"],
                    "embedding": embeddings[i],
                    "source": file_path,
                    "filename": filename,
                })

            self.vector_store.update_ledger_entry(
                file_path=str(file_path),
                file_id=doc_id,
                sha256=sha256_file(file_path),
                last_modified=get_mtime(file_path),
                chunk_ids=chunk_ids,
            )

            bm25_start = time.time()
            self._update_bm25_index()
            bm25_duration = time.time() - bm25_start

            total_duration = time.time() - doc_start_time
            logger.info(
                f"[Persist] Ingested '{filename}' in {total_duration:.3f}s "
                f"(Parse: {parse_duration:.3f}s, Chunk: {chunk_duration:.3f}s, "
                f"Embed: {embed_duration:.3f}s, BM25: {bm25_duration:.3f}s)"
            )
            return doc_id


    def _update_bm25_index(self):

        if not self.child_nodes:

            self.bm25 = None
            return

        corpus = [tokenize(node["text"]) for node in self.child_nodes]
        self.bm25 = BM25Okapi(corpus)


    @trace(tags=["core_engine", "hyde_embedding"])
    async def retrieve_hyde_query_embedding(self, query: str, metrics_collector: Optional[Any] = None) -> List[float]:

        # HyDE expansion: Generate hypothetical document
        prompt = HYDE_GENERATION_PROMPT.format(query=query)
        messages = [{"role": "user", "content": prompt}]
        logger.debug(f"\033[1;33m[HYDE GENERATION]\033[0m Input: query='{query}'")
        response = await self.llm_client.acompletion(
            messages=messages,
            model="local-llm",
            temperature=0.7,
            max_tokens=256,
            metrics_collector=metrics_collector,
            metrics_purpose="HYDE GENERATION"
        )
        hyde_doc = response.choices[0].message.content
        logger.debug(f"\033[1;33m[HYDE GENERATION]\033[0m Output: hypothetical_doc='{hyde_doc[:200]}...'")

        # Embed the generated hypothetical document
        hyde_embeddings = await self.llm_client.aembedding(hyde_doc)
        return hyde_embeddings[0]


    @trace(tags=["core_engine"])
    async def search(
        self,
        query: str,
        top_k_retrieval: Optional[int] = None,
        top_k_llm: Optional[int] = None,
        use_hyde: Optional[bool] = None,
        rrf_k: Optional[int] = None,
        metrics_collector: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:

        if not self.child_nodes:

            return []

        search_start = time.time()
        logger.debug(f"\033[1;33m[RETRIEVAL]\033[0m Input: query='{query}'")

        if top_k_retrieval is None:

            top_k_retrieval = self.config["retrieval"]["top_k_retrieval"]
        if top_k_llm is None:

            top_k_llm = self.config["retrieval"]["top_k_llm"]
        if use_hyde is None:

            use_hyde = self.config["retrieval"]["use_hyde"]
        if rrf_k is None:

            rrf_k = self.config["retrieval"]["rrf_k"]

        logger.info(
            f"\033[1;33m[RETRIEVAL START]\033[0m"
            f"top_k={top_k_retrieval} | top_n={top_k_llm} | "
            f"hyde={use_hyde} | db_chunks={len(self.child_nodes)}"
        )

        # 1. Sparse (BM25) Retrieve
        sparse_start = time.time()
        bm25_scores = []
        if self.bm25:

            tokenized_query = tokenize(query)
            bm25_scores = self.bm25.get_scores(tokenized_query)
        sparse_duration = time.time() - sparse_start
        logger.info(f"\033[1;33m[SPARSE BM25 RETRIEVAL]\033[0m Scored {len(bm25_scores)} chunks | time: {sparse_duration:.3f}s")
        if len(bm25_scores) > 0:
            top_bm25_indices = np.argsort(bm25_scores)[-20:][::-1]
            logger.debug(
                f"\033[1;33m[SPARSE BM25 RETRIEVAL]\033[0m tokenized_query={tokenized_query} | "
                "top-20 (idx:score): " + ", ".join(
                    f"{i}:{bm25_scores[i]:.4f}" for i in top_bm25_indices
                )
            )
        else:
            logger.debug(f"\033[1;33m[SPARSE BM25 RETRIEVAL]\033[0m No BM25 index available")

        # 2. Dense (Vector) Retrieve
        dense_start = time.time()
        if use_hyde:

            query_vector = await self.retrieve_hyde_query_embedding(query, metrics_collector=metrics_collector)
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
        logger.info(f"\033[1;33m[DENSE VECTOR RETRIEVAL]\033[0m Scored {len(vector_scores)} chunks | time: {dense_duration:.3f}s")
        logger.debug(
            f"\033[1;33m[DENSE VECTOR RETRIEVAL]\033[0m query_vector[:5]={qv[:5].tolist()} | "
            "top-20: " + ", ".join(
                f"idx={i}:{vector_scores[i]:.4f}"
                for i in np.argsort(vector_scores)[-20:][::-1]
            )
        )

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
        top_nodes = scored_nodes[:top_k_retrieval]
        rrf_duration = time.time() - rrf_start
        logger.info(f"\033[1;33m[RRF FUSION]\033[0m Merged {len(scored_nodes)} chunks → top {len(top_nodes)} candidates | time: {rrf_duration:.3f}s")
        logger.debug(
            f"\033[1;33m[RRF FUSION]\033[0m weights: sparse={rrf_weight_sparse} dense={rrf_weight_dense} k={rrf_k} | "
            "top candidates:\n" + "\n".join(
                f"  [{i}] {item['node'].get('child_id','?')[:40]}... "
                f"rrf={item['rrf_score']:.4f} "
                f"file={item['node'].get('filename','?')} "
                f"pg={item['node'].get('page_number','?')}"
                for i, item in enumerate(top_nodes[:20])
            )
        )

        # 4. Rerank nodes using LLM interface rerank
        rerank_start = time.time()
        node_texts = [item["node"]["text"] for item in top_nodes]
        logger.debug(f"\033[1;34m[RERANKER]\033[0m Input: {len(node_texts)} texts to rerank against query='{query}'")
        reranked = self.llm_client.rerank(query, node_texts, top_n=top_k_llm)
        rerank_duration = time.time() - rerank_start
        logger.info(f"\033[1;33m[RERANKER]\033[0m Re-ranked {len(node_texts)} → {len(reranked)} chunks | time: {rerank_duration:.3f}s")
        logger.debug(
            f"\033[1;33m[RERANKER]\033[0m Output scores: " + ", ".join(
                f"[{r['index']}] score={r['score']:.4f}" for r in reranked
            )
        )

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
                "chunk_id": matched_node.get("chunk_id") or matched_node.get("child_id"),
                "text": matched_node["text"],
                "page_number": matched_node["page_number"],
                "source": matched_node["source"],
                "filename": matched_node["filename"],
                "score": float(scaled_score),
                "parent_content": parent_doc.content,
                "parent_id": parent_id
            })

        search_total_duration = time.time() - search_start
        logger.info(
            f"\033[1;33m[RETRIEVAL END]\033[0m Completed in {search_total_duration:.3f}s | "
            f"sparse={sparse_duration:.3f}s dense={dense_duration:.3f}s "
            f"rrf={rrf_duration:.3f}s rerank={rerank_duration:.3f}s | "
            f"results={len(final_results)}"
        )
        logger.debug(
            f"\033[1;33m[RETRIEVAL]\033[0m Output: {len(final_results)} results:\n"
            + "\n".join(
                f"  [{i}] chunk_id={r['chunk_id'][:40]}... "
                f"file={r['filename']} pg={r['page_number']} "
                f"score={r['score']:.4f}"
                for i, r in enumerate(final_results)
            )
        )
        
        return final_results


    @staticmethod
    def prepare_context_and_citations(search_results: list) -> tuple[str, dict]:
        context_start = time.time()
        citation_map = {}
        context_blocks = []

        for index, match in enumerate(search_results, start=1):
            str_idx = str(index)
            citation_map[str_idx] = {
                "chunk_id": match.get("chunk_id"),
                "filename": match.get("filename", "Unknown"),
                "page_number": match.get("page_number"),
                "score": float(match.get("score", 0.0)),
                "text": match.get("text", ""),
            }
            block = f"--- Document [{str_idx}] ---\n{match['text']}\n"
            context_blocks.append(block)

        formatted = "\n".join(context_blocks)
        context_duration = time.time() - context_start
        logger.info(f"\033[1;33m[CONTEXT PREPARATION]\033[0m Prepared {len(search_results)} chunks, {len(formatted)} chars | time: {context_duration:.3f}s")
        logger.debug(f"\033[1;33m[CONTEXT PREPARATION]\033[0m citation_map={citation_map}")
        return formatted, citation_map
