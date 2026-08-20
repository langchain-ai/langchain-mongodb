"""Polling-based S3 watcher.

Periodically lists all objects in the bucket, computes ETag diffs against a
local state snapshot, and fires on_created / on_updated / on_deleted callbacks.

Works with zero additional AWS infrastructure. Default watcher implementation.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from pymongo.collection import Collection

from langchain_mongodb_deepagents_vfs.backends.base import ObjectStoreBackend
from langchain_mongodb_deepagents_vfs.chunker import Chunker
from langchain_mongodb_deepagents_vfs.embedder import Embedder
from langchain_mongodb_deepagents_vfs.errors import AdapterError
from langchain_mongodb_deepagents_vfs.watcher.base import S3Watcher

logger = logging.getLogger(__name__)

_DEFAULT_INTERVAL = 10  # 10 seconds


class PollingWatcher(S3Watcher):
    """Watcher that polls S3 on a fixed interval and diffs ETags.

    Args:
        store: Object store backend.
        chunker: Chunker instance.
        embedder: Embedder instance.
        collection: MongoDB collection.
        interval_seconds: Poll interval in seconds (default 10).
        prefix: Only watch keys under this S3 prefix.
    """

    def __init__(
        self,
        store: ObjectStoreBackend,
        chunker: Chunker,
        embedder: Embedder,
        collection: Collection,  # type: ignore[type-arg]
        interval_seconds: int = _DEFAULT_INTERVAL,
        prefix: str = "",
    ) -> None:
        super().__init__(store, chunker, embedder, collection)
        self._interval = interval_seconds
        self._prefix = prefix
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        # key → etag snapshot of last poll
        self._state: dict[str, str] = {}
        # Health signal: last exception raised out of a poll iteration, if
        # any. The background thread swallows it to stay alive, so this is
        # the only way a caller can tell the watcher stopped syncing.
        self.last_error: Exception | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the polling loop in a background daemon thread."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop, name="PollingWatcher", daemon=True
        )
        self._thread.start()
        logger.info("PollingWatcher started (interval=%ds).", self._interval)

    def stop(self) -> None:
        """Signal the polling loop to stop and wait for it to exit."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=self._interval + 10)
        logger.info("PollingWatcher stopped.")

    def __enter__(self) -> PollingWatcher:
        self.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Internal loop
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        while not self._stop_event.wait(timeout=self._interval):
            try:
                self._poll()
            except Exception as exc:
                # A crash here (e.g. from on_created/on_updated issuing a
                # MongoDB command) must not silently end the background
                # thread — log it and record it so the caller can detect a
                # watcher that has stopped syncing.
                logger.error("PollingWatcher: unhandled error during poll: %s", exc)
                self.last_error = exc

    def _poll(self) -> None:
        try:
            current: dict[str, str] = dict(self._store.list_keys(self._prefix))
        except AdapterError as exc:
            logger.error("PollingWatcher: list_keys failed: %s", exc)
            self.last_error = exc
            return

        prev = self._state

        # Detect created / updated
        for key, etag in current.items():
            if key not in prev:
                logger.debug("PollingWatcher: created: %s", key)
                self.on_created(key)
            elif prev[key] != etag:
                logger.debug("PollingWatcher: updated: %s", key)
                self.on_updated(key)

        # Detect deleted
        for key in prev:
            if key not in current:
                logger.debug("PollingWatcher: deleted: %s", key)
                self.on_deleted(key)

        self._state = current
