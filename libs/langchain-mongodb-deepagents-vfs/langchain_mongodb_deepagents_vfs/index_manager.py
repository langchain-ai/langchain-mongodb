"""Atlas index provisioner.

Creates and waits for three indexes:
  1. Atlas Vector Search on ``embedding`` (knnVector, cosine similarity).
  2. Atlas Search on ``content`` + ``filename`` (full-text + wildcard).
  3. Compound native index on ``(source_path, chunk_index)`` (unique).

All operations are idempotent — calling ``ensure_indexes()`` twice is a no-op.
On non-Atlas MongoDB (mongomock, community server) the Atlas-specific calls
degrade to a warning so unit tests can run without real Atlas.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from pymongo.collection import Collection
from pymongo.errors import OperationFailure
from pymongo_search_utils import create_vector_search_index

from langchain_mongodb_deepagents_vfs.errors import AdapterError, ErrorCode

logger = logging.getLogger(__name__)

_DEFAULT_WAIT_SECONDS = 600  # 10 minutes — Atlas search index build can be slow


class IndexManager:
    """Manages Atlas and native indexes for the chunk collection.

    Args:
        collection: PyMongo Collection to operate on.
        embedding_dimensions: Vector dimensionality (must match the Embedder).
        wait_timeout: Seconds to poll before giving up on index readiness.
    """

    def __init__(
        self,
        collection: Collection,  # type: ignore[type-arg]
        embedding_dimensions: int = 1024,
        wait_timeout: int = _DEFAULT_WAIT_SECONDS,
    ) -> None:
        self._col = collection
        self._dims = embedding_dimensions
        self._timeout = wait_timeout
        self._is_atlas: bool | None = None  # resolved lazily

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ensure_indexes(self) -> None:
        """Create all required indexes if they don't already exist.

        Safe to call repeatedly — already-existing indexes are skipped.

        Raises:
            AdapterError(E1004): Index creation failed on a real Atlas cluster.
        """
        self._ensure_native_index()
        if self._check_atlas():
            self._ensure_vector_search_index()
            self._ensure_fulltext_search_index()
        else:
            logger.warning(
                "Non-Atlas MongoDB detected; skipping Vector Search and Full-Text Search index creation. "
                "grep and glob will fall back to regex queries."
            )

    def wait_until_queryable(self, timeout: int | None = None) -> bool:
        """Block until both Atlas search indexes report READY (queryable).

        Args:
            timeout: Override instance-level timeout (seconds).

        Returns:
            True if indexes became queryable within *timeout*, False otherwise.
        """
        if not self._check_atlas():
            return True  # no Atlas indexes to wait for

        deadline = time.monotonic() + (timeout or self._timeout)
        poll_interval = 10
        while time.monotonic() < deadline:
            if self._indexes_queryable():
                logger.info("Atlas search indexes are queryable.")
                return True
            remaining = deadline - time.monotonic()
            logger.debug("Waiting for indexes… %.0fs remaining", remaining)
            time.sleep(poll_interval)
        logger.warning(
            "Timed out waiting for Atlas search indexes to become queryable."
        )
        return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_atlas(self) -> bool:
        if self._is_atlas is not None:
            return self._is_atlas
        try:
            # Atlas exposes listSearchIndexes; community Mongo does not.
            self._col.list_search_indexes()
            self._is_atlas = True
        except (AttributeError, TypeError, OperationFailure):
            # mongomock raises TypeError; community Mongo raises OperationFailure
            self._is_atlas = False
        return self._is_atlas

    def _ensure_native_index(self) -> None:
        try:
            self._col.create_index(
                [("source_path", 1), ("chunk_index", 1)],
                unique=True,
                name="source_path_chunk_index_unique",
            )
            logger.info("Native compound index ensured.")
        except OperationFailure as exc:
            if "already exists" in str(exc).lower():
                pass
            else:
                raise AdapterError(
                    ErrorCode.E1004_INDEX_PROVISION_FAILED, str(exc)
                ) from exc

    def _ensure_vector_search_index(self) -> None:
        index_name = "vector_search_embedding"
        existing = {idx["name"] for idx in self._col.list_search_indexes()}
        if index_name in existing:
            logger.debug("Vector search index '%s' already exists.", index_name)
            return
        try:
            create_vector_search_index(
                collection=self._col,
                index_name=index_name,
                path="embedding",
                dimensions=self._dims,
                similarity="cosine",
                # Declare filterable fields so $vectorSearch filter clauses work
                filters=["source_path", "filename"],
            )
            logger.info("Vector search index '%s' creation initiated.", index_name)
        except OperationFailure as exc:
            raise AdapterError(
                ErrorCode.E1004_INDEX_PROVISION_FAILED, str(exc)
            ) from exc

    def _ensure_fulltext_search_index(self) -> None:
        index_name = "fulltext_search_content_filename"
        definition: dict[str, Any] = {
            "mappings": {
                "dynamic": False,
                "fields": {
                    "content": {"type": "string", "analyzer": "lucene.standard"},
                    # lucene.keyword: whole field is a single token; wildcard queries with
                    # allowAnalyzedField=true match against the full string value
                    "filename": {"type": "string", "analyzer": "lucene.keyword"},
                    "source_path": {"type": "string", "analyzer": "lucene.keyword"},
                },
            }
        }
        existing = {idx["name"]: idx for idx in self._col.list_search_indexes()}
        if index_name not in existing:
            try:
                self._col.create_search_index(
                    {"name": index_name, "type": "search", "definition": definition}
                )
                logger.info(
                    "Full-text search index '%s' creation initiated.", index_name
                )
            except OperationFailure as exc:
                raise AdapterError(
                    ErrorCode.E1004_INDEX_PROVISION_FAILED, str(exc)
                ) from exc
            return

        # Migrate: update if filename is not lucene.keyword or source_path is missing
        current_fields = (
            existing[index_name]
            .get("latestDefinition", {})
            .get("mappings", {})
            .get("fields", {})
        )
        current_filename = current_fields.get("filename", {})
        correct = (
            current_filename.get("type") == "string"
            and current_filename.get("analyzer") == "lucene.keyword"
            and "source_path" in current_fields
        )
        if not correct:
            try:
                self._col.update_search_index(index_name, definition)
                logger.info(
                    "Full-text search index '%s' updated: filename migrated to lucene.keyword.",
                    index_name,
                )
            except OperationFailure as exc:
                raise AdapterError(
                    ErrorCode.E1004_INDEX_PROVISION_FAILED, str(exc)
                ) from exc
        else:
            logger.debug(
                "Full-text search index '%s' already exists with correct definition.",
                index_name,
            )

    def _indexes_queryable(self) -> bool:
        try:
            # Check both queryable and status==READY: during an index update Atlas keeps
            # queryable=True (old definition active) but status==UPDATING; we must wait
            # until status returns to READY so the new definition is actually in use.
            states = {
                idx["name"]: (idx.get("queryable", False), idx.get("status", ""))
                for idx in self._col.list_search_indexes()
            }
            vector_ok = states.get("vector_search_embedding", (False, "")) == (
                True,
                "READY",
            )
            fulltext_ok = states.get(
                "fulltext_search_content_filename", (False, "")
            ) == (True, "READY")
            return vector_ok and fulltext_ok
        except Exception:
            return False
