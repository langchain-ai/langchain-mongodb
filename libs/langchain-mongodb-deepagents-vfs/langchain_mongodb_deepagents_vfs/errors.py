"""Error classification layer for the adapter boundary.

No raw exception ever escapes MongoFilesystemBackend's public surface.
Every exception is caught here, mapped to a stable ErrorCode, logged with a
correlation ID, and returned as a user-friendly DTO field.
"""

from __future__ import annotations

import functools
import logging
import traceback
import uuid
from collections.abc import Callable
from enum import StrEnum
from typing import Any, TypeVar

from deepagents.backends.protocol import FILE_NOT_FOUND, PERMISSION_DENIED

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


class ErrorCode(StrEnum):
    """Stable error codes surfaced to callers of MongoFilesystemBackend."""

    # E1xxx — configuration / initialization errors
    E1001_MISSING_CONFIG = "E1001"
    E1002_INVALID_BUCKET = "E1002"
    E1003_INVALID_MONGO_URI = "E1003"
    E1004_INDEX_PROVISION_FAILED = "E1004"
    E1005_INITIAL_SYNC_FAILED = "E1005"
    E1006_WATCHER_START_FAILED = "E1006"

    # E2xxx — object store errors
    E2001_OBJECT_NOT_FOUND = "E2001"
    E2002_OBJECT_READ_FAILED = "E2002"
    E2003_OBJECT_WRITE_FAILED = "E2003"
    E2004_OBJECT_DELETE_FAILED = "E2004"
    E2005_LIST_FAILED = "E2005"
    E2006_UPLOAD_FAILED = "E2006"
    E2007_DOWNLOAD_FAILED = "E2007"
    E2008_EDIT_CONFLICT = "E2008"

    # E3xxx — MongoDB errors
    E3001_CONNECTION_FAILED = "E3001"
    E3002_QUERY_FAILED = "E3002"
    E3003_UPSERT_FAILED = "E3003"
    E3004_DELETE_FAILED = "E3004"
    E3005_INDEX_NOT_READY = "E3005"

    # E4xxx — embedding errors
    E4001_EMBEDDING_API_FAILED = "E4001"
    E4002_EMBEDDING_DIMENSION_MISMATCH = "E4002"
    E4003_EMBEDDING_RATE_LIMITED = "E4003"

    # E5xxx — search errors
    E5001_GREP_FAILED = "E5001"
    E5002_GLOB_FAILED = "E5002"
    E5003_LS_FAILED = "E5003"
    E5004_VECTOR_SEARCH_UNAVAILABLE = "E5004"

    # E6xxx — watcher errors
    E6001_SQS_RECEIVE_FAILED = "E6001"
    E6002_EVENT_PARSE_FAILED = "E6002"
    E6003_WATCHER_CRASHED = "E6003"

    # E9xxx — internal / unexpected errors
    E9001_INTERNAL_ERROR = "E9001"
    E9002_CHUNKER_FAILED = "E9002"
    E9003_FORMAT_UNSUPPORTED = "E9003"


_CODE_MESSAGES: dict[ErrorCode, str] = {
    ErrorCode.E1001_MISSING_CONFIG: "Required configuration parameter is missing.",
    ErrorCode.E1002_INVALID_BUCKET: "The specified S3 bucket does not exist or is not accessible.",
    ErrorCode.E1003_INVALID_MONGO_URI: "The MongoDB connection string is invalid or the cluster is unreachable.",
    ErrorCode.E1004_INDEX_PROVISION_FAILED: "Failed to create required Atlas search indexes.",
    ErrorCode.E1005_INITIAL_SYNC_FAILED: "Initial S3 → MongoDB sync did not complete successfully.",
    ErrorCode.E1006_WATCHER_START_FAILED: "The S3 watcher could not start.",
    ErrorCode.E2001_OBJECT_NOT_FOUND: "The requested object does not exist in the bucket.",
    ErrorCode.E2002_OBJECT_READ_FAILED: "Failed to read object from the object store.",
    ErrorCode.E2003_OBJECT_WRITE_FAILED: "Failed to write object to the object store.",
    ErrorCode.E2004_OBJECT_DELETE_FAILED: "Failed to delete object from the object store.",
    ErrorCode.E2005_LIST_FAILED: "Failed to list objects in the bucket.",
    ErrorCode.E2006_UPLOAD_FAILED: "One or more file uploads failed.",
    ErrorCode.E2007_DOWNLOAD_FAILED: "One or more file downloads failed.",
    ErrorCode.E2008_EDIT_CONFLICT: "Edit conflict: the object was modified concurrently.",
    ErrorCode.E3001_CONNECTION_FAILED: "Could not connect to MongoDB.",
    ErrorCode.E3002_QUERY_FAILED: "MongoDB query failed.",
    ErrorCode.E3003_UPSERT_FAILED: "MongoDB upsert operation failed.",
    ErrorCode.E3004_DELETE_FAILED: "MongoDB delete operation failed.",
    ErrorCode.E3005_INDEX_NOT_READY: "Atlas search index is not yet queryable.",
    ErrorCode.E4001_EMBEDDING_API_FAILED: "The embedding API returned an error.",
    ErrorCode.E4002_EMBEDDING_DIMENSION_MISMATCH: "Embedding dimensions do not match the configured value.",
    ErrorCode.E4003_EMBEDDING_RATE_LIMITED: "Embedding API rate limit exceeded.",
    ErrorCode.E5001_GREP_FAILED: "The grep search operation failed.",
    ErrorCode.E5002_GLOB_FAILED: "The glob search operation failed.",
    ErrorCode.E5003_LS_FAILED: "The ls operation failed.",
    ErrorCode.E5004_VECTOR_SEARCH_UNAVAILABLE: "Atlas Vector Search is not available on this cluster.",
    ErrorCode.E6001_SQS_RECEIVE_FAILED: "Failed to receive messages from SQS queue.",
    ErrorCode.E6002_EVENT_PARSE_FAILED: "Could not parse an S3 event notification from SQS.",
    ErrorCode.E6003_WATCHER_CRASHED: "The background watcher thread crashed unexpectedly.",
    ErrorCode.E9001_INTERNAL_ERROR: "An unexpected internal error occurred.",
    ErrorCode.E9002_CHUNKER_FAILED: "The document chunker encountered an error.",
    ErrorCode.E9003_FORMAT_UNSUPPORTED: "The file format is not supported by the chunker.",
}


class AdapterError(Exception):
    """Wraps a classified ErrorCode with its original cause."""

    def __init__(
        self, code: ErrorCode, message: str, cause: BaseException | None = None
    ) -> None:
        self.code = code
        self.message = message
        self.cause = cause
        super().__init__(f"[{code.value}] {message}")


def user_message(code: ErrorCode, detail: str = "") -> str:
    base = _CODE_MESSAGES.get(code, "An error occurred.")
    return f"[{code.value}] {base}" + (f" Detail: {detail}" if detail else "")


# Our codes that map onto deepagents' standardized per-file error literals.
# Anything absent falls back to the free-form message, which the protocol
# explicitly allows ("or a backend-specific error string").
_FILE_OPERATION_ERRORS: dict[ErrorCode, str] = {
    ErrorCode.E2001_OBJECT_NOT_FOUND: FILE_NOT_FOUND,
    ErrorCode.E1002_INVALID_BUCKET: PERMISSION_DENIED,
}


def file_operation_error(code: ErrorCode) -> str | None:
    """Return the protocol ``FileOperationError`` literal for *code*, if any."""
    return _FILE_OPERATION_ERRORS.get(code)


def adapter_boundary(
    code: ErrorCode = ErrorCode.E9001_INTERNAL_ERROR,
) -> Callable[[F], F]:
    """Decorator that catches all exceptions and converts them to error DTOs.

    The decorated method must return a DTO with an ``error`` field (str | None).
    On success the DTO is returned normally. On failure the ``error`` field is
    set to a user-friendly message and the DTO is returned (no raise).

    When the instance has ``debug=True``, the original exception is re-raised
    after logging so local development gets a full traceback.
    """

    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            correlation_id = str(uuid.uuid4())[:8]
            try:
                return fn(self, *args, **kwargs)
            except AdapterError as exc:
                logger.error(
                    "[%s] AdapterError %s in %s: %s",
                    correlation_id,
                    exc.code.value,
                    fn.__qualname__,
                    exc.message,
                    # exc_info=True, not exc.cause: AdapterError is nearly always
                    # raised with `from exc`, which populates __cause__ and leaves
                    # the explicit `cause` arg None. True renders the live
                    # exception plus its whole chain either way.
                    exc_info=True,
                )
                if getattr(self, "debug", False):
                    raise
                return _make_error_dto(fn, args, kwargs, exc.code, exc.message)
            except Exception as exc:
                tb = traceback.format_exc()
                logger.error(
                    "[%s] Unhandled exception %s in %s:\n%s",
                    correlation_id,
                    type(exc).__name__,
                    fn.__qualname__,
                    tb,
                )
                if getattr(self, "debug", False):
                    raise
                return _make_error_dto(fn, args, kwargs, code, str(exc))

        return wrapper  # type: ignore[return-value]

    return decorator


def _make_error_dto(
    fn: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    code: ErrorCode,
    detail: str,
) -> Any:
    """Return a zero-value protocol result with the error field set.

    Shapes here must stay identical to ``deepagents.backends.protocol`` — they
    are the same classes, imported, not re-declared.
    """
    from langchain_mongodb_deepagents_vfs.dtypes import (
        EditResult,
        FileDownloadResponse,
        FileUploadResponse,
        GlobResult,
        GrepResult,
        LsResult,
        ReadResult,
        WriteResult,
    )

    message = user_message(code, detail)
    name = fn.__name__

    if name == "ls":
        return LsResult(error=message, entries=None)
    if name == "read":
        return ReadResult(error=message, file_data=None)
    if name == "write":
        return WriteResult(error=message, path=None)
    if name == "edit":
        return EditResult(error=message, path=None, occurrences=None)
    if name == "grep":
        return GrepResult(error=message, matches=None)
    if name == "glob":
        return GlobResult(error=message, matches=None)
    # Batch operations have no aggregate error field: the protocol expects one
    # response per input, so fan the failure out across every requested path.
    if name == "upload_files":
        files = kwargs.get("files", args[0] if args else [])
        return [FileUploadResponse(path=p, error=message) for p, _ in files]
    if name == "download_files":
        paths = kwargs.get("paths", args[0] if args else [])
        return [FileDownloadResponse(path=p, error=message) for p in paths]
    return None
