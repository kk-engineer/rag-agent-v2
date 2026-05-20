import json
import os
import logging
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings


logger = logging.getLogger(__name__)


class VectorStore:

    def __init__(self, persist_dir: str = "local_models/chromadb", collection_name: str = "rag_chunks"):
        os.makedirs(persist_dir, exist_ok=True)
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.meta_path = os.path.join(persist_dir, "store_meta.json")
        self.parent_path = os.path.join(persist_dir, "parent_store.json")

        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = None

    def initialize(self):
        try:
            self.collection = self.client.get_collection(self.collection_name)
            logger.info(f"Loaded existing ChromaDB collection '{self.collection_name}' ({self.collection.count()} chunks)")
        except Exception:
            self.collection = self.client.create_collection(self.collection_name)
            logger.info(f"Created new ChromaDB collection '{self.collection_name}'")

    def count(self) -> int:
        if self.collection is None:
            return 0
        return self.collection.count()

    def get_metadata(self) -> Dict[str, Any]:
        if os.path.exists(self.meta_path):
            try:
                with open(self.meta_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load store metadata: {e}")
        return {}

    def save_metadata(self, embedding_model: str, file_manifest: Dict[str, int]):
        meta = {
            "embedding_model": embedding_model,
            "file_manifest": file_manifest
        }
        with open(self.meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        logger.info(f"Saved store metadata: model={embedding_model}, files={len(file_manifest)}")

    def is_data_valid(self, embedding_model: str) -> bool:
        meta = self.get_metadata()
        if not meta:
            return False
        if meta.get("embedding_model") != embedding_model:
            logger.info(f"Embedding model changed: was '{meta.get('embedding_model')}', now '{embedding_model}'")
            return False
        return True

    def add_chunks(self, chunks: List[Dict[str, Any]]):
        if not chunks:
            return
        ids = [c["child_id"] for c in chunks]
        embeddings = [c["embedding"] for c in chunks]
        metadatas = [{
            "text": c["text"],
            "page_number": c["page_number"],
            "source": c["source"],
            "filename": c["filename"],
            "parent_id": c["parent_id"]
        } for c in chunks]
        self.collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas)

    def get_all_chunks(self) -> List[Dict[str, Any]]:
        if self.collection is None or self.collection.count() == 0:
            return []
        all_data = self.collection.get(include=["embeddings", "metadatas"])
        chunks = []
        for i, cid in enumerate(all_data["ids"]):
            meta = all_data["metadatas"][i]
            chunks.append({
                "child_id": cid,
                "text": meta["text"],
                "page_number": meta["page_number"],
                "source": meta["source"],
                "filename": meta["filename"],
                "parent_id": meta["parent_id"],
                "embedding": all_data["embeddings"][i]
            })
        logger.info(f"Loaded {len(chunks)} chunks from ChromaDB collection '{self.collection_name}'")
        return chunks

    def delete_all(self):
        try:
            self.client.delete_collection(self.collection_name)
        except ValueError:
            pass
        self.collection = self.client.create_collection(self.collection_name)
        logger.info(f"Cleared ChromaDB collection '{self.collection_name}'")
        for p in [self.meta_path, self.parent_path]:
            if os.path.exists(p):
                os.remove(p)

    def save_parents(self, parents: Dict[str, Any]):
        serializable = {}
        for doc_id, doc in parents.items():
            serializable[doc_id] = {
                "content": doc.content,
                "metadata": doc.metadata
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
