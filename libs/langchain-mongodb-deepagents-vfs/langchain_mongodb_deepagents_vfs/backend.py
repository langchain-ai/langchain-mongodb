"""MongoFilesystemBackend — public entry point implementing BackendProtocol.

Construction is non-blocking: index provisioning, initial sync, and the
background watcher start in a daemon thread so the constructor returns
immediately. Searches block until the first sync completes.

Usage::

    from langchain_mongodb_deepagents_vfs import MongoFilesystemBackend

    backend = MongoFilesystemBackend(
        s3_bucket_name="my-bucket",
        mongodb_connection_string="mongodb+srv://...",
    )
    result = backend.grep("authentication flow", path="docs/")
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Literal

import pymongo
from deepagents.backends.protocol import BackendProtocol
from pymongo.collection import Collection
from pymongo.driver_info import DriverInfo

from langchain_mongodb_deepagents_vfs.backends.base import MAX_READ_BYTES
from langchain_mongodb_deepagents_vfs.backends.s3 import S3Backend
from langchain_mongodb_deepagents_vfs.chunker import Chunker
from langchain_mongodb_deepagents_vfs.dtypes import (
    EditResult,
    FileData,
    FileDownloadResponse,
    FileUploadResponse,
    GlobResult,
    GrepResult,
    LsResult,
    ReadResult,
    SyncReport,
    WriteResult,
)
from langchain_mongodb_deepagents_vfs.embedder import Embedder
from langchain_mongodb_deepagents_vfs.errors import (
    AdapterError,
    ErrorCode,
    adapter_boundary,
)
from langchain_mongodb_deepagents_vfs.index_manager import IndexManager
from langchain_mongodb_deepagents_vfs.search import SearchRouter
from langchain_mongodb_deepagents_vfs.sync import InitialSync
from langchain_mongodb_deepagents_vfs.watcher import (
    PollingWatcher,
    S3Watcher,
    SQSWatcher,
)

logger = logging.getLogger(__name__)

_VERSION: str | None
try:
    from importlib.metadata import version as get_version

    _VERSION = get_version("langchain_mongodb_deepagents_vfs")
except Exception:
    _VERSION = None

_DRIVER_INFO = DriverInfo(name="DeepAgents-MongoDB-FS", version=_VERSION)

_DB_NAME = "langchain_mongodb_deepagents_vfs"
_COLLECTION_NAME = "demo_chunks"

WatcherType = Literal["polling", "sqs"]


class MongoFilesystemBackend(BackendProtocol):
    """DeepAgents BackendProtocol implementation powered by MongoDB Atlas.

    Every public method returns the result type declared by
    ``deepagents.backends.protocol`` — no adapter-local result shapes.

    Args:
        s3_bucket_name: Name of the S3 bucket that is the source of truth.
        mongodb_connection_string: Atlas (or compatible) connection string.
        llm: Optional LangChain LLM instance (reserved for future use).
        embedding_model: LangChain Embeddings instance. Defaults to
            ``OpenAIEmbeddings(model='text-embedding-3-small', dimensions=1024)``.
        embedding_dimensions: Embedding vector size (default 1024).
        watcher: ``"polling"`` (default) or ``"sqs"``.
        sqs_queue_url: Required when ``watcher="sqs"``.
        aws_region: AWS region for S3 and SQS clients.
        s3_prefix: Only sync/watch objects under this S3 prefix.
        debug: If True, re-raise exceptions after logging (local dev only).
    """

    def __init__(
        self,
        s3_bucket_name: str,
        mongodb_connection_string: str,
        llm: Any = None,
        embedding_model: Any = None,
        embedding_dimensions: int = 1024,
        watcher: WatcherType = "polling",
        sqs_queue_url: str | None = None,
        aws_region: str | None = None,
        s3_prefix: str = "",
        debug: bool = False,
    ) -> None:
        if not s3_bucket_name:
            raise AdapterError(
                ErrorCode.E1001_MISSING_CONFIG, "s3_bucket_name is required"
            )
        if not mongodb_connection_string:
            raise AdapterError(
                ErrorCode.E1001_MISSING_CONFIG, "mongodb_connection_string is required"
            )
        if watcher == "sqs" and not sqs_queue_url:
            raise AdapterError(
                ErrorCode.E1001_MISSING_CONFIG,
                "sqs_queue_url is required when watcher='sqs'",
            )

        self.debug = debug
        self._prefix = s3_prefix

        # Build internal components
        self._store = S3Backend(bucket_name=s3_bucket_name, region_name=aws_region)
        # aws_region reaches the Embedder too, not just S3/SQS: without it a
        # caller passing aws_region= got S3 in that region and Bedrock in
        # whatever the environment happened to resolve.
        self._embedder = Embedder(
            model=embedding_model,
            dimensions=embedding_dimensions,
            region_name=aws_region,
        )
        self._chunker = Chunker()

        mongo_client: pymongo.MongoClient[dict[str, Any]] = pymongo.MongoClient(
            mongodb_connection_string, driver=_DRIVER_INFO
        )
        self._col: Collection = mongo_client[_DB_NAME][_COLLECTION_NAME]  # type: ignore[type-arg]

        self._index_manager = IndexManager(
            self._col, embedding_dimensions=embedding_dimensions
        )
        self._sync = InitialSync(self._store, self._chunker, self._embedder, self._col)
        self._search = SearchRouter(self._col, self._embedder)

        # Build watcher
        if watcher == "sqs":
            self._watcher: S3Watcher = SQSWatcher(
                store=self._store,
                chunker=self._chunker,
                embedder=self._embedder,
                collection=self._col,
                queue_url=sqs_queue_url,  # type: ignore[arg-type]
                region_name=aws_region,
                prefix=s3_prefix,
            )
        else:
            self._watcher = PollingWatcher(
                store=self._store,
                chunker=self._chunker,
                embedder=self._embedder,
                collection=self._col,
                prefix=s3_prefix,
            )

        # Gate that blocks grep/glob/ls until the first sync completes
        self._ready = threading.Event()

        # Outcome of background initialization, readable once _ready is set.
        # ``initial_sync_report`` is None if the sync raised outright; a report
        # with failed > 0 means some files never made it into MongoDB and will
        # be invisible to grep/glob.
        self.initial_sync_report: SyncReport | None = None
        self.init_errors: list[str] = []

        # Background init thread
        self._init_thread = threading.Thread(
            target=self._background_init, name="MongoFSInit", daemon=True
        )
        self._init_thread.start()

    # ------------------------------------------------------------------
    # Background initialization
    # ------------------------------------------------------------------

    def _background_init(self) -> None:
        try:
            logger.info("Provisioning indexes…")
            self._index_manager.ensure_indexes()
            self._index_manager.wait_until_queryable()
        except Exception as exc:
            logger.error("Index provisioning failed: %s", exc, exc_info=True)
            self.init_errors.append(f"index provisioning failed: {exc}")

        try:
            logger.info("Running initial sync…")
            report = self._sync.run(prefix=self._prefix)
            self.initial_sync_report = report
            if report.failed:
                # Not fatal — some files may have synced — but silence here is
                # what makes an empty collection look like a search bug.
                logger.error(
                    "Initial sync: %d of %d objects FAILED to index and will not "
                    "be searchable (processed=%d skipped=%d). See preceding "
                    "warnings for the per-key cause.",
                    report.failed,
                    report.seen,
                    report.processed,
                    report.skipped,
                )
                self.init_errors.append(
                    f"{report.failed} of {report.seen} objects failed to index"
                )
            else:
                logger.info("Initial sync complete: %s", report)
        except Exception as exc:
            logger.error("Initial sync failed: %s", exc, exc_info=True)
            self.init_errors.append(f"initial sync failed: {exc}")

        self._ready.set()

        try:
            logger.info("Starting watcher…")
            self._watcher.start()
        except Exception as exc:
            # Recorded like the two steps above, not just logged: a watcher that
            # never starts leaves a collection that looks correct and silently
            # goes stale, which is harder to spot than an empty one.
            logger.error("Watcher start failed: %s", exc, exc_info=True)
            self.init_errors.append(f"watcher start failed: {exc}")

    def _wait_ready(self) -> None:
        """Block until at least one sync pass has completed."""
        self._ready.wait()

    # ------------------------------------------------------------------
    # Search operations (route through SearchRouter after sync)
    # ------------------------------------------------------------------

    @adapter_boundary(ErrorCode.E5001_GREP_FAILED)
    def grep(
        self, pattern: str, path: str | None = None, glob: str | None = None
    ) -> GrepResult:
        """Search file contents using hybrid MongoDB search.

        Args:
            pattern: Natural-language or keyword query.
            path: Restrict search to this path prefix.
            glob: Restrict to filenames matching this glob pattern.

        Returns:
            GrepResult with protocol ``GrepMatch`` entries, most relevant
            first. Hybrid search ranks results, but the ranking is expressed
            by list order — ``GrepMatch`` has no score field and none is added.
        """
        self._wait_ready()
        return self._search.grep(pattern, path or "", glob or "")

    @adapter_boundary(ErrorCode.E5002_GLOB_FAILED)
    def glob(self, pattern: str, path: str | None = None) -> GlobResult:
        """Find files whose names match *pattern*.

        Args:
            pattern: Glob pattern (e.g. ``"*.pdf"``).
            path: Restrict to this path prefix.

        Returns:
            GlobResult with a ``FileInfo`` per matching file.
        """
        self._wait_ready()
        return self._search.glob(pattern, path or "")

    @adapter_boundary(ErrorCode.E5003_LS_FAILED)
    def ls(self, path: str) -> LsResult:
        """List immediate children of *path*.

        Args:
            path: Virtual directory path (e.g. ``"docs/"``).

        Returns:
            LsResult with file and directory entries.
        """
        self._wait_ready()
        return self._search.ls(path)

    # ------------------------------------------------------------------
    # Pass-through operations (go directly to the object store)
    # ------------------------------------------------------------------

    @adapter_boundary(ErrorCode.E2002_OBJECT_READ_FAILED)
    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        """Read a file from the object store.

        Args:
            file_path: Object key.
            offset: Line number to start reading from (0-indexed), per the
                protocol — not a byte offset.
            limit: Maximum number of lines to return.

        Returns:
            ReadResult carrying a ``FileData`` payload.
        """
        # HEAD first so an oversized object is refused before the GET transfers
        # anything. S3Backend.read() enforces the same cap as a backstop for
        # every other caller; this pre-flight only saves the wasted transfer.
        # Line-offset slicing needs the whole decoded object, so a byte Range
        # can't substitute here.
        size = self._store.get_size(file_path)
        if size > MAX_READ_BYTES:
            raise AdapterError(
                ErrorCode.E2002_OBJECT_READ_FAILED,
                f"Object is {size} bytes, exceeds read limit of {MAX_READ_BYTES} bytes",
            )
        data = self._store.read(file_path)
        lines = data.decode("utf-8", errors="replace").splitlines(keepends=True)
        end = offset + limit if limit >= 0 else None
        text = "".join(lines[offset:end])
        return ReadResult(file_data=FileData(content=text, encoding="utf-8"))

    @adapter_boundary(ErrorCode.E2003_OBJECT_WRITE_FAILED)
    def write(self, file_path: str, content: str) -> WriteResult:
        """Write *content* to *file_path* in the object store.

        Args:
            file_path: Object key.
            content: String content to write.

        Returns:
            WriteResult with the written path.
        """
        self._store.write(file_path, content.encode("utf-8"))
        return WriteResult(path=file_path)

    @adapter_boundary(ErrorCode.E2008_EDIT_CONFLICT)
    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        """Edit *file_path* by replacing *old_string* with *new_string*.

        Performs a conditional read-modify-write with ETag verification.

        Args:
            file_path: Object key.
            old_string: Substring to find.
            new_string: Replacement string.
            replace_all: Replace every occurrence if True, else only the first.

        Returns:
            EditResult with the edited path and replacement count.
        """
        occurrences = self._store.edit(file_path, old_string, new_string, replace_all)
        return EditResult(path=file_path, occurrences=occurrences)

    @adapter_boundary(ErrorCode.E2006_UPLOAD_FAILED)
    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Upload multiple files to the object store.

        Args:
            files: List of ``(path, bytes)`` tuples.

        Returns:
            One ``FileUploadResponse`` per input file, in input order.
        """
        return self._store.upload_files(files)

    @adapter_boundary(ErrorCode.E2007_DOWNLOAD_FAILED)
    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download multiple files from the object store.

        Args:
            paths: List of object keys to download.

        Returns:
            One ``FileDownloadResponse`` per input path, in input order.
        """
        return self._store.download_files(paths)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def stop(self) -> None:
        """Gracefully stop the background watcher."""
        self._watcher.stop()

    def __enter__(self) -> MongoFilesystemBackend:
        return self

    def __exit__(self, *_: Any) -> None:
        self.stop()
