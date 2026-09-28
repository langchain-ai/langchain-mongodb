"""Internal DTOs shared across all modules.

Result shapes returned from ``MongoFilesystemBackend`` are **not** redefined
here — they are imported straight from ``deepagents.backends.protocol`` so the
adapter can never drift from the protocol it implements. Only the types that
are genuinely ours (chunks, stored records, sync stats) live in this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from deepagents.backends.protocol import (
    EditResult,
    FileData,
    FileDownloadResponse,
    FileInfo,
    FileUploadResponse,
    GlobResult,
    GrepMatch,
    GrepResult,
    LsResult,
    ReadResult,
    WriteResult,
)

__all__ = [
    "Chunk",
    "EditResult",
    "FileData",
    "FileDownloadResponse",
    "FileInfo",
    "FileRecord",
    "FileUploadResponse",
    "GlobResult",
    "GrepMatch",
    "GrepResult",
    "LsResult",
    "ReadResult",
    "SearchHit",
    "SyncReport",
    "WriteResult",
]


@dataclass(frozen=True)
class Chunk:
    """A single chunk produced by the Chunker."""

    source_path: str
    chunk_index: int
    content: str
    page_number: int = 0
    char_start: int = 0
    char_end: int = 0
    line_start: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FileRecord:
    """MongoDB document shape for a stored chunk."""

    source_path: str
    chunk_index: int
    content: str
    embedding: list[float]
    page_number: int
    char_start: int
    char_end: int
    line_start: int
    etag: str
    filename: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "chunk_index": self.chunk_index,
            "content": self.content,
            "embedding": self.embedding,
            "page_number": self.page_number,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "line_start": self.line_start,
            "etag": self.etag,
            "filename": self.filename,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class SearchHit:
    """A single result entry returned by SearchRouter operations."""

    source_path: str
    chunk_index: int
    content: str
    line_start: int
    score: float = 0.0


@dataclass
class SyncReport:
    """Stats from one S3 → MongoDB sync pass. Internal, not a protocol type."""

    seen: int = 0
    processed: int = 0
    skipped: int = 0
    failed: int = 0
    error: str | None = None
