"""Abstract base class for object-store backends.

Any future backend (Azure Blob, GCS, local disk) implements this ABC.
The rest of the adapter — Chunker, Embedder, IndexManager, Watcher,
SearchRouter — only depends on this interface, never on S3 specifics.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import PurePosixPath

from langchain_mongodb_deepagents_vfs.dtypes import (
    FileDownloadResponse,
    FileUploadResponse,
)


class ObjectStoreBackend(ABC):
    """Filesystem-like interface over an arbitrary object store."""

    # ------------------------------------------------------------------
    # Read / write primitives
    # ------------------------------------------------------------------

    @abstractmethod
    def read(self, path: str, offset: int = 0, limit: int = -1) -> bytes:
        """Return bytes for *path*, optionally sliced by *offset* / *limit*.

        Args:
            path: Object key / path.
            offset: Byte offset to start from (0 = beginning).
            limit: Maximum bytes to return (-1 = all remaining).

        Returns:
            Raw bytes.

        Raises:
            AdapterError(E2001): Object not found.
            AdapterError(E2002): Read failure.
        """

    @abstractmethod
    def write(self, path: str, content: bytes) -> None:
        """Write *content* to *path*, creating or replacing.

        Raises:
            AdapterError(E2003): Write failure.
        """

    @abstractmethod
    def edit(self, path: str, old: str, new: str, replace_all: bool = False) -> int:
        """Read-modify-write *path*, replacing *old* with *new*.

        Uses ETag / conditional-write semantics so concurrent edits are
        detected and surfaced as E2008.

        Returns:
            Number of occurrences replaced.

        Raises:
            AdapterError(E2001): Object not found.
            AdapterError(E2008): Concurrent modification conflict.
        """

    # ------------------------------------------------------------------
    # Bulk transfer helpers
    # ------------------------------------------------------------------

    @abstractmethod
    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Upload multiple ``(path, content)`` pairs.

        Partial success is expected: one response per input file, in input
        order, with ``error`` set on the ones that failed.
        """

    @abstractmethod
    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download multiple paths.

        Partial success is expected: one response per input path, in input
        order, with ``error`` set on the ones that failed.
        """

    # ------------------------------------------------------------------
    # Listing
    # ------------------------------------------------------------------

    @abstractmethod
    def list_keys(self, prefix: str = "") -> Iterator[tuple[str, str]]:
        """Yield ``(key, etag)`` tuples for all objects under *prefix*.

        Raises:
            AdapterError(E2005): Listing failure.
        """

    # ------------------------------------------------------------------
    # Helpers shared by all concrete backends
    # ------------------------------------------------------------------

    @staticmethod
    def normalize_key(path: str) -> str:
        """Normalize an OS path to a forward-slash S3-style key."""
        return str(PurePosixPath(path.replace("\\", "/")))
