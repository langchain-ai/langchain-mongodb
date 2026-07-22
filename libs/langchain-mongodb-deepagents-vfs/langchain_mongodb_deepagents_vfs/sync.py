"""Initial sync: copies every S3 object into MongoDB as chunked + embedded docs.

Idempotent: objects whose ETag hasn't changed since the last run are skipped,
so restarts after partial failures are cheap.

Concurrency: S3 downloads run in a ThreadPoolExecutor (I/O-bound);
embedding and bulk-write are sequential (CPU/API-bound).
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import PurePosixPath

from pymongo.collection import Collection
from pymongo.operations import UpdateOne

from langchain_mongodb_deepagents_vfs.backends.base import ObjectStoreBackend
from langchain_mongodb_deepagents_vfs.chunker import Chunker
from langchain_mongodb_deepagents_vfs.dtypes import FileRecord, SyncReport
from langchain_mongodb_deepagents_vfs.embedder import Embedder
from langchain_mongodb_deepagents_vfs.errors import AdapterError, ErrorCode

logger = logging.getLogger(__name__)

_DOWNLOAD_WORKERS = 16
_UPSERT_BATCH = 500


class InitialSync:
    """Paginates S3, downloads objects, chunks + embeds, upserts into Mongo.

    Args:
        store: Object store backend.
        chunker: Chunker instance.
        embedder: Embedder instance.
        collection: Target MongoDB collection.
        download_workers: Thread pool size for concurrent S3 downloads.
    """

    def __init__(
        self,
        store: ObjectStoreBackend,
        chunker: Chunker,
        embedder: Embedder,
        collection: Collection,  # type: ignore[type-arg]
        download_workers: int = _DOWNLOAD_WORKERS,
    ) -> None:
        self._store = store
        self._chunker = chunker
        self._embedder = embedder
        self._col = collection
        self._workers = download_workers

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, prefix: str = "", dry_run: bool = False) -> SyncReport:
        """Execute the initial sync.

        Args:
            prefix: Only process keys starting with this prefix.
            dry_run: If True, skip embedding and upsert (cost guardrail).

        Returns:
            SyncReport with seen / processed / skipped / failed counters.
        """
        report = SyncReport()
        keys_etags = list(self._store.list_keys(prefix))
        report.seen = len(keys_etags)
        logger.info(
            "InitialSync: %d objects found under prefix '%s'", report.seen, prefix
        )

        # Determine which keys need processing (ETag changed or new)
        to_process = self._filter_changed(keys_etags)
        report.skipped = report.seen - len(to_process)
        logger.info(
            "InitialSync: %d unchanged (skipped), %d to process",
            report.skipped,
            len(to_process),
        )

        if dry_run:
            logger.info("dry_run=True — skipping embedding and upsert.")
            report.processed = 0
            return report

        # Download concurrently
        raw_objects = self._download_all(to_process)

        # Process sequentially: chunk → embed → upsert batch
        pending_records: list[FileRecord] = []
        for key, etag, data in raw_objects:
            try:
                chunks = self._chunker.chunk(key, data)
                if not chunks:
                    continue
                vectors = self._embedder.embed_batch(chunks)
                filename = PurePosixPath(key).name
                for chunk, vector in zip(chunks, vectors, strict=True):
                    pending_records.append(
                        FileRecord(
                            source_path=key,
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
                        )
                    )
                if len(pending_records) >= _UPSERT_BATCH:
                    self._upsert(pending_records)
                    report.processed += len(pending_records)
                    pending_records = []
            except AdapterError as exc:
                logger.warning("Failed to process key '%s': %s", key, exc)
                report.failed += 1

        if pending_records:
            self._upsert(pending_records)
            report.processed += len(pending_records)

        logger.info(
            "InitialSync complete: processed=%d skipped=%d failed=%d",
            report.processed,
            report.skipped,
            report.failed,
        )
        return report

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _filter_changed(
        self, keys_etags: list[tuple[str, str]]
    ) -> list[tuple[str, str]]:
        """Return only (key, etag) pairs that differ from stored ETags.

        Uses $group so we transfer one document per *file* rather than one per
        *chunk* — O(files) instead of O(chunks) data transferred from MongoDB.
        """
        all_keys = [k for k, _ in keys_etags]
        stored: dict[str, str] = {}
        for doc in self._col.aggregate(
            [
                {"$match": {"source_path": {"$in": all_keys}}},
                {"$group": {"_id": "$source_path", "etag": {"$first": "$etag"}}},
            ]
        ):
            stored[doc["_id"]] = doc["etag"]

        return [(key, etag) for key, etag in keys_etags if stored.get(key) != etag]

    def _download_all(
        self, to_process: list[tuple[str, str]]
    ) -> list[tuple[str, str, bytes]]:
        results: list[tuple[str, str, bytes]] = []
        with ThreadPoolExecutor(max_workers=self._workers) as pool:
            future_map = {
                pool.submit(self._store.read, key): (key, etag)
                for key, etag in to_process
            }
            for future in as_completed(future_map):
                key, etag = future_map[future]
                try:
                    data = future.result()
                    results.append((key, etag, data))
                except Exception as exc:
                    logger.warning("Download failed for key '%s': %s", key, exc)
        return results

    def _upsert(self, records: list[FileRecord]) -> None:
        ops = [
            UpdateOne(
                {"source_path": r.source_path, "chunk_index": r.chunk_index},
                {"$set": r.to_dict()},
                upsert=True,
            )
            for r in records
        ]
        try:
            self._col.bulk_write(ops, ordered=False)
        except Exception as exc:
            raise AdapterError(ErrorCode.E3003_UPSERT_FAILED, str(exc)) from exc
