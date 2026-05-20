import os
import time
import hashlib
import uuid
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Set
from rag_engine.vector_store import VectorStore
from rag_engine.llm import LiteLLMClient


logger = logging.getLogger(__name__)


def sha256_file(file_path: str) -> str:

    h = hashlib.sha256()
    with open(file_path, "rb") as f:

        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def get_mtime(file_path: str) -> str:

    return datetime.fromtimestamp(
        os.path.getmtime(file_path), tz=timezone.utc
    ).isoformat()


_SUPPORTED_EXTENSIONS = {".pdf", ".html", ".htm", ".md", ".markdown", ".txt", ".py", ".js"}


class IngestionCoordinator:

    def __init__(
        self,
        vector_store: VectorStore,
        core_engine: Any,
        llm_client: LiteLLMClient,
        config: Dict[str, Any],
    ):

        self.vector_store = vector_store
        self.core_engine = core_engine
        self.llm_client = llm_client
        self.config = config

    def _is_supported(self, file_path: str) -> bool:

        ext = os.path.splitext(file_path)[1].lower()
        return ext in _SUPPORTED_EXTENSIONS

    def _scan_directory(self, dir_path: str) -> List[str]:

        files = []
        for root, _, filenames in os.walk(dir_path):
            for fname in filenames:
                fpath = os.path.join(root, fname)
                if self._is_supported(fpath):
                    files.append(fpath)
        files.sort()
        return files

    # ── Public API ────────────────────────────────────────────────

    async def ingest_path(self, path: str) -> int:

        path = os.path.abspath(path)
        if not os.path.exists(path):
            logger.error(f"Path does not exist: {path}")
            return 0

        if os.path.isdir(path):
            return await self._ingest_directory(path)

        return await self._ingest_single_file(path)

    async def full_reindex(self, new_model: str) -> int:

        ledger = self.vector_store.get_ledger()
        if not ledger:
            logger.warning("Ledger is empty, nothing to reindex")
            return 0

        new_collection_name = f"{self.vector_store.collection_name}_v{int(time.time())}"
        logger.info(
            f"Starting full reindex with model='{new_model}' "
            f"into collection '{new_collection_name}'"
        )

        new_collection = self.vector_store.create_new_collection(
            new_collection_name, new_model
        )

        old_collection = self.vector_store.collection
        self.vector_store.collection = new_collection

        count = 0
        for file_path in ledger:
            try:
                doc_id = str(uuid.uuid4())
                await self._process_file(file_path, doc_id)
                count += 1
            except Exception as e:
                logger.error(f"Failed to reindex {file_path}: {e}")

        self.vector_store.collection = old_collection
        self.vector_store.swap_collection(new_collection_name)
        self.vector_store.drop_collection(old_collection.name)

        self.vector_store.set_embedding_model(new_model)
        logger.info(
            f"Full reindex complete: {count}/{len(ledger)} files migrated "
            f"to collection '{new_collection_name}'"
        )
        return count

    def verify_model_consistency(self) -> bool:

        stored = self.vector_store.get_embedding_model()
        current = self._resolve_model_name()
        if stored is None:
            logger.info("No embedding model stored in collection metadata")
            return False
        if stored != current:
            logger.info(
                f"Embedding model mismatch: stored='{stored}', config='{current}'"
            )
            return False
        return True

    # ── Internal: Directory Scan ──────────────────────────────────

    async def _ingest_directory(self, dir_path: str) -> int:

        ledger = self.vector_store.get_ledger()
        files = self._scan_directory(dir_path)
        added = 0
        updated = 0
        skipped = 0

        for fpath in files:
            fpath_str = str(fpath)
            if fpath_str not in ledger:
                try:
                    doc_id = str(uuid.uuid4())
                    await self._process_file(fpath, doc_id)
                    added += 1
                except Exception as e:
                    logger.error(f"Failed to ingest {fpath}: {e}")
            else:
                entry = ledger[fpath_str]
                if await self._has_changed(fpath, entry):
                    try:
                        await self._update_file(fpath, entry)
                        updated += 1
                    except Exception as e:
                        logger.error(f"Failed to update {fpath}: {e}")
                else:
                    skipped += 1

        logger.info(
            f"Directory scan complete: {added} added, {updated} updated, "
            f"{skipped} skipped out of {len(files)} files"
        )
        return added + updated

    async def _has_changed(self, file_path: str, entry: dict) -> bool:

        try:
            current_hash = sha256_file(file_path)
            return current_hash != entry.get("sha256", "")
        except Exception:
            return True

    # ── Internal: Single File Processing ──────────────────────────

    async def _ingest_single_file(self, file_path: str) -> int:

        ledger = self.vector_store.get_ledger()
        fpath_str = str(file_path)
        if fpath_str in ledger:
            entry = ledger[fpath_str]
            if not await self._has_changed(file_path, entry):
                logger.info(f"File unchanged, skipping: {file_path}")
                return 0
        doc_id = str(uuid.uuid4())
        await self._process_file(file_path, doc_id)
        return 1

    async def _process_file(self, file_path: str, doc_id: str):

        filename = os.path.basename(file_path)
        logger.info(f"Processing file: {filename}")

        parse_start = time.time()
        full_text, pages = await self.core_engine.parse_file(file_path)
        parse_duration = time.time() - parse_start

        chunk_start = time.time()
        chunks = await self.core_engine.chunk_semantically(pages)
        chunk_duration = time.time() - chunk_start

        embed_start = time.time()
        chunk_texts = [c["text"] for c in chunks]
        embeddings = await self.llm_client.aembedding(chunk_texts)
        embed_duration = time.time() - embed_start

        chunk_ids = self.vector_store.add_chunks_batch(
            chunks=chunks,
            embeddings=embeddings,
            file_id=doc_id,
            source=file_path,
            filename=filename,
        )

        from rag_engine.core import Document
        doc = Document(
            doc_id=doc_id,
            content=full_text,
            metadata={"source": file_path, "filename": filename},
        )
        self.core_engine.parent_store[doc_id] = doc
        self.vector_store.save_parents(self.core_engine.parent_store)

        file_hash = sha256_file(file_path)
        file_mtime = get_mtime(file_path)
        self.vector_store.update_ledger_entry(
            file_path=str(file_path),
            file_id=doc_id,
            sha256=file_hash,
            last_modified=file_mtime,
            chunk_ids=chunk_ids,
        )

        total = time.time() - parse_start
        logger.info(
            f"Ingested '{filename}' in {total:.3f}s "
            f"(Parse: {parse_duration:.3f}s, Chunk: {chunk_duration:.3f}s, "
            f"Embed: {embed_duration:.3f}s)"
        )

    async def _update_file(self, file_path: str, entry: dict):

        file_id = entry["file_id"]
        self.vector_store.delete_document(file_id)
        self.vector_store.remove_ledger_entry(str(file_path))
        old_parent = self.core_engine.parent_store.pop(file_id, None)

        doc_id = str(uuid.uuid4())
        await self._process_file(file_path, doc_id)

        logger.info(f"Updated file: {file_path} (new file_id={doc_id})")

    # ── Internal: Model Name Resolution ───────────────────────────

    def _resolve_model_name(self) -> str:

        embedder = self.config.get("models", {}).get(
            "embedder_repo_id", "sentence-transformers/all-MiniLM-L6-v2"
        )
        return embedder
