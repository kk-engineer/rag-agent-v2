import json
import os
import hashlib
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple
import chromadb
from chromadb.config import Settings


logger = logging.getLogger(__name__)


class VectorStore:

    def __init__(
        self,
        persist_dir: str = "chroma_db",
        collection_name: str = "rag_documents",
    ):

        os.makedirs(persist_dir, exist_ok=True)
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.meta_path = os.path.join(persist_dir, "store_meta.json")
        self.parent_path = os.path.join(persist_dir, "parent_store.json")
        self.ledger_path = os.path.join(persist_dir, "chroma_ingestion_ledger.json")

        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = None

    # ── Collection Lifecycle ──────────────────────────────────────

    def initialize(self, embedding_model: Optional[str] = None):

        try:

            self.collection = self.client.get_collection(self.collection_name)
            logger.info(
                f"Loaded existing ChromaDB collection '{self.collection_name}' "
                f"({self.collection.count()} chunks)"
            )
        except (ValueError, chromadb.errors.NotFoundError):

            create_kwargs = {"name": self.collection_name}
            if embedding_model:

                create_kwargs["metadata"] = {
                    "hnsw:space": "cosine",
                    "embedding_model": embedding_model,
                }
            else:

                create_kwargs["metadata"] = {"hnsw:space": "cosine"}
            self.collection = self.client.create_collection(**create_kwargs)
            logger.info(
                f"Created new ChromaDB collection '{self.collection_name}' "
                f"with hnsw:space=cosine"
            )

    def count(self) -> int:

        if self.collection is None:
            return 0
        return self.collection.count()

    def delete_all(self):

        try:

            self.client.delete_collection(self.collection_name)
        except ValueError:
            pass
        self.collection = None
        for p in [self.meta_path, self.parent_path, self.ledger_path]:
            if os.path.exists(p):
                os.remove(p)

    # ── Embedding Model Tracking ──────────────────────────────────

    def get_embedding_model(self) -> Optional[str]:

        if self.collection is None:
            return None
        meta = getattr(self.collection, "metadata", None) or {}
        return meta.get("embedding_model")

    def set_embedding_model(self, model_name: str):

        if self.collection is None:
            return
        current = dict(self.collection.metadata) if self.collection.metadata else {}
        current["embedding_model"] = model_name
        self.collection.modify(metadata=current)
        logger.info(f"Updated collection metadata: embedding_model={model_name}")

    def is_data_valid(self, embedding_model: str) -> bool:

        stored = self.get_embedding_model()
        if stored is None:
            return False
        if stored != embedding_model:
            logger.info(
                f"Embedding model changed: was '{stored}', now '{embedding_model}'"
            )
            return False
        return True

    # ── Chunk Operations ──────────────────────────────────────────

    def add_chunks(self, chunks: List[Dict[str, Any]]):

        if not chunks:
            return
        ids = [c["child_id"] for c in chunks]
        embeddings = [c["embedding"] for c in chunks]
        metadatas = [
            {
                "text": c["text"],
                "page_number": c["page_number"],
                "source": c["source"],
                "filename": c["filename"],
                "parent_id": c["parent_id"],
                "file_id": c.get("file_id", c["parent_id"]),
            }
            for c in chunks
        ]
        self.collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas)

    def add_chunks_batch(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]],
        file_id: str,
        source: str,
        filename: str,
    ) -> List[str]:

        if not chunks:
            return []
        ids = [f"{file_id}#chunk_{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "text": chunks[i]["text"],
                "page_number": chunks[i]["page_number"],
                "source": source,
                "filename": filename,
                "file_id": file_id,
                "parent_id": file_id,
            }
            for i in range(len(chunks))
        ]
        self.collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas)
        logger.info(f"Inserted {len(chunks)} chunks for file_id={file_id}")
        return ids

    def delete_document(self, file_id: str):

        try:

            self.collection.delete(where={"file_id": file_id})
            logger.info(f"Deleted all chunks for file_id={file_id}")
        except Exception as e:

            logger.warning(f"Failed to delete document {file_id}: {e}")

    def get_all_chunks(self) -> List[Dict[str, Any]]:

        if self.collection is None or self.collection.count() == 0:
            return []
        all_data = self.collection.get(include=["embeddings", "metadatas"])
        chunks = []
        for i, cid in enumerate(all_data["ids"]):
            meta = all_data["metadatas"][i]
            chunks.append(
                {
                    "child_id": cid,
                    "text": meta["text"],
                    "page_number": meta["page_number"],
                    "source": meta["source"],
                    "filename": meta["filename"],
                    "parent_id": meta.get("parent_id", meta.get("file_id", "")),
                    "file_id": meta.get("file_id", ""),
                    "embedding": all_data["embeddings"][i],
                }
            )
        logger.info(
            f"Loaded {len(chunks)} chunks from ChromaDB collection '{self.collection_name}'"
        )
        return chunks

    def get_all_texts(self) -> List[Tuple[str, str, str]]:

        if self.collection is None or self.collection.count() == 0:
            return []
        all_data = self.collection.get(include=["metadatas"])
        texts = []
        for i, cid in enumerate(all_data["ids"]):
            meta = all_data["metadatas"][i]
            texts.append((cid, meta["text"], meta.get("file_id", "")))
        return texts

    # ── Parent Store ──────────────────────────────────────────────

    def save_parents(self, parents: Dict[str, Any]):

        serializable = {}
        for doc_id, doc in parents.items():
            serializable[doc_id] = {
                "content": doc.content,
                "metadata": doc.metadata,
            }
        with open(self.parent_path, "w") as f:
            json.dump(serializable, f, indent=2)

    def load_parents(self) -> Dict[str, Dict[str, Any]]:

        if not os.path.exists(self.parent_path):
            return {}
        try:
            with open(self.parent_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load parent store: {e}")
            return {}

    # ── Ingestion Ledger ──────────────────────────────────────────

    def _load_ledger(self) -> Dict[str, Any]:

        if not os.path.exists(self.ledger_path):
            return {}
        try:
            with open(self.ledger_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load ingestion ledger: {e}")
            return {}

    def _save_ledger(self, ledger: Dict[str, Any]):

        with open(self.ledger_path, "w") as f:
            json.dump(ledger, f, indent=2)

    def get_ledger(self) -> Dict[str, Any]:

        return self._load_ledger()

    def update_ledger_entry(
        self,
        file_path: str,
        file_id: str,
        sha256: str,
        last_modified: str,
        chunk_ids: List[str],
    ):

        ledger = self._load_ledger()
        ledger[file_path] = {
            "file_id": file_id,
            "sha256": sha256,
            "last_modified": last_modified,
            "chunk_ids": chunk_ids,
        }
        self._save_ledger(ledger)

    def remove_ledger_entry(self, file_path: str):

        ledger = self._load_ledger()
        ledger.pop(file_path, None)
        self._save_ledger(ledger)

    def clear_ledger(self):

        if os.path.exists(self.ledger_path):
            os.remove(self.ledger_path)

    # ── Collection Swap / Migration ───────────────────────────────

    def create_new_collection(
        self, name: str, embedding_model: str
    ) -> Any:

        try:

            self.client.delete_collection(name)
        except ValueError:
            pass
        new_collection = self.client.create_collection(
            name=name,
            metadata={
                "hnsw:space": "cosine",
                "embedding_model": embedding_model,
            },
        )
        logger.info(f"Created new collection '{name}' with model={embedding_model}")
        return new_collection

    def drop_collection(self, name: str):

        try:

            self.client.delete_collection(name)
            logger.info(f"Dropped collection '{name}'")
        except ValueError:
            logger.warning(f"Collection '{name}' does not exist, skipping drop")

    def swap_collection(self, new_name: str):

        old_name = self.collection_name
        self.collection_name = new_name
        self.collection = self.client.get_collection(new_name)
        logger.info(f"Swapped active collection from '{old_name}' to '{new_name}'")

    # ── Metadata (legacy compatibility) ───────────────────────────

    def get_metadata(self) -> Dict[str, Any]:

        if os.path.exists(self.meta_path):
            try:
                with open(self.meta_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load store metadata: {e}")
        return {}

    def save_metadata(self, embedding_model: str, file_manifest: Dict[str, int]):

        meta = {"embedding_model": embedding_model, "file_manifest": file_manifest}
        with open(self.meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        logger.info(
            f"Saved store metadata: model={embedding_model}, files={len(file_manifest)}"
        )
