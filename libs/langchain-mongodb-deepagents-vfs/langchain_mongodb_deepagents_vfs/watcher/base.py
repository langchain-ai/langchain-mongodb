"""Abstract base class for S3 bucket watchers."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import PurePosixPath

from pymongo.collection import Collection
from pymongo.operations import UpdateOne

from langchain_mongodb_deepagents_vfs.backends.base import ObjectStoreBackend
from langchain_mongodb_deepagents_vfs.chunker import Chunker
from langchain_mongodb_deepagents_vfs.dtypes import FileRecord
from langchain_mongodb_deepagents_vfs.embedder import Embedder
from langchain_mongodb_deepagents_vfs.errors import AdapterError

logger = logging.getLogger(__name__)


class S3Watcher(ABC):
    """Background watcher that keeps MongoDB in sync with S3 changes.

    Concrete implementations: PollingWatcher (default) and SQSWatcher.

    The shared callback logic lives here — on_created/on_updated reuse the
    same Chunker + Embedder pipeline as InitialSync. on_deleted issues
    delete_many({source_path: key}) directly.

    Args:
        store: Object store backend.
        chunker: Chunker instance.
        embedder: Embedder instance.
        collection: MongoDB collection to write into.
    """

    def __init__(
        self,
        store: ObjectStoreBackend,
        chunker: Chunker,
        embedder: Embedder,
        collection: Collection,  # type: ignore[type-arg]
    ) -> None:
        self._store = store
        self._chunker = chunker
        self._embedder = embedder
        self._col = collection

    # ------------------------------------------------------------------
    # Lifecycle (subclasses implement)
    # ------------------------------------------------------------------

    @abstractmethod
    def start(self) -> None:
        """Start the watcher in a background daemon thread."""

    @abstractmethod
    def stop(self) -> None:
        """Signal the watcher to stop and join the background thread."""

    # ------------------------------------------------------------------
    # Shared event callbacks (called by concrete implementations)
    # ------------------------------------------------------------------

    def on_created(self, key: str) -> None:
        """Handle a new object in S3."""
        etag = getattr(self._store, "get_etag", lambda k: "")(key) or ""
        if etag and self._col.count_documents(
            {"source_path": key, "etag": etag}, limit=1
        ):
            logger.debug(
                "Watcher: skipping already-ingested key '%s' (etag=%s)", key, etag
            )
            return
        self._ingest(key)

    def on_updated(self, key: str) -> None:
        """Handle an updated object in S3."""
        # Delete stale chunks first, then re-ingest
        self._col.delete_many({"source_path": key})
        self._ingest(key)

    def on_deleted(self, key: str) -> None:
        """Handle a deleted object — remove all its chunks from MongoDB."""
        try:
            result = self._col.delete_many({"source_path": key})
            logger.info("Deleted %d chunks for key '%s'", result.deleted_count, key)
        except Exception as exc:
            logger.error("Failed to delete chunks for key '%s': %s", key, exc)

    # ------------------------------------------------------------------
    # Shared ingestion pipeline
    # ------------------------------------------------------------------

    def _ingest(self, key: str) -> None:
        try:
            data = self._store.read(key)
        except AdapterError as exc:
            logger.error("Watcher: failed to download key '%s': %s", key, exc)
            return

        # Derive ETag from the store if possible
        etag = getattr(self._store, "get_etag", lambda k: "")(key) or ""

        try:
            chunks = self._chunker.chunk(key, data)
            if not chunks:
                return
            vectors = self._embedder.embed_batch(chunks)
        except AdapterError as exc:
            logger.error("Watcher: chunk/embed failed for key '%s': %s", key, exc)
            return

        filename = PurePosixPath(key).name
        ops = [
            UpdateOne(
                {"source_path": chunk.source_path, "chunk_index": chunk.chunk_index},
                {
                    "$set": FileRecord(
                        source_path=chunk.source_path,
                        chunk_index=chunk.chunk_index,
                        content=chunk.content,
                        embedding=vector,
                        page_number=chunk.page_number,
                        char_start=chunk.char_start,
                        char_end=chunk.char_end,
                        line_start=chunk.line_start,
                        etag=etag,
                        filename=filename,
                        metadata=chunk.metadata,
                    ).to_dict()
                },
                upsert=True,
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        try:
            self._col.bulk_write(ops, ordered=False)
            logger.info("Watcher: ingested %d chunks for key '%s'", len(ops), key)
        except Exception as exc:
            logger.error("Watcher: upsert failed for key '%s': %s", key, exc)
