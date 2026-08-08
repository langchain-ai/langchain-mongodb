"""S3 backend implementation using boto3."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

import boto3
from botocore.exceptions import ClientError

from langchain_mongodb_deepagents_vfs.backends.base import (
    MAX_READ_BYTES,
    ObjectStoreBackend,
)
from langchain_mongodb_deepagents_vfs.dtypes import (
    FileDownloadResponse,
    FileUploadResponse,
)
from langchain_mongodb_deepagents_vfs.errors import (
    AdapterError,
    ErrorCode,
    file_operation_error,
)

logger = logging.getLogger(__name__)


def _file_error(exc: AdapterError) -> str:
    """Map an AdapterError to a protocol ``FileOperationError`` where possible."""
    return file_operation_error(exc.code) or exc.message


class S3Backend(ObjectStoreBackend):
    """ObjectStoreBackend backed by an AWS S3 bucket.

    Args:
        bucket_name: Name of the S3 bucket.
        region_name: AWS region (defaults to AWS_DEFAULT_REGION env var).
        endpoint_url: Override endpoint for local testing (e.g. LocalStack).
        **boto_kwargs: Forwarded verbatim to ``boto3.client``.
    """

    def __init__(
        self,
        bucket_name: str,
        region_name: str | None = None,
        endpoint_url: str | None = None,
        **boto_kwargs: Any,
    ) -> None:
        self._bucket = bucket_name
        self._client: Any = boto3.client(
            "s3",
            region_name=region_name,
            endpoint_url=endpoint_url,
            **boto_kwargs,
        )
        self._verify_bucket()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _verify_bucket(self) -> None:
        try:
            self._client.head_bucket(Bucket=self._bucket)
        except ClientError as exc:
            code = exc.response["Error"]["Code"]
            if code in ("404", "NoSuchBucket"):
                raise AdapterError(
                    ErrorCode.E1002_INVALID_BUCKET, f"Bucket '{self._bucket}' not found"
                ) from exc
            raise AdapterError(ErrorCode.E2002_OBJECT_READ_FAILED, str(exc)) from exc

    def _key(self, path: str) -> str:
        return self.normalize_key(path).lstrip("/")

    # ------------------------------------------------------------------
    # ObjectStoreBackend implementation
    # ------------------------------------------------------------------

    def read(self, path: str, offset: int = 0, limit: int = -1) -> bytes:
        key = self._key(path)
        range_header: dict[str, str] = {}
        if offset > 0 or limit >= 0:
            end = "" if limit < 0 else str(offset + limit - 1)
            range_header["Range"] = f"bytes={offset}-{end}"
        try:
            response = self._client.get_object(
                Bucket=self._bucket, Key=key, **range_header
            )
            # Enforce the cap here, not at the call sites: InitialSync, the
            # watchers and download_files all funnel through this method, and
            # only the public read() ever did its own pre-flight check.
            # ContentLength reflects the ranged length when a Range was sent,
            # and arrives with the headers, so this costs no extra round trip
            # and rejects before any body bytes are transferred.
            length = response.get("ContentLength")
            if length is not None and length > MAX_READ_BYTES:
                raise AdapterError(
                    ErrorCode.E2002_OBJECT_READ_FAILED,
                    f"Object '{key}' is {length} bytes, exceeds the "
                    f"{MAX_READ_BYTES} byte read limit",
                )
            # Backstop for responses that omit ContentLength (chunked transfer):
            # read one byte past the cap so an oversized body is detected
            # without ever materialising more than the limit.
            body: bytes = response["Body"].read(MAX_READ_BYTES + 1)
            if len(body) > MAX_READ_BYTES:
                raise AdapterError(
                    ErrorCode.E2002_OBJECT_READ_FAILED,
                    f"Object '{key}' exceeds the {MAX_READ_BYTES} byte read limit",
                )
            return body
        except ClientError as exc:
            code = exc.response["Error"]["Code"]
            if code in ("NoSuchKey", "404"):
                raise AdapterError(
                    ErrorCode.E2001_OBJECT_NOT_FOUND, f"Key '{key}' not found"
                ) from exc
            raise AdapterError(ErrorCode.E2002_OBJECT_READ_FAILED, str(exc)) from exc

    def write(self, path: str, content: bytes) -> None:
        key = self._key(path)
        # Refuse to store what we could never read back: read() and edit() both
        # cap at MAX_READ_BYTES, so accepting a larger write would silently
        # create an object this backend can only fail on. Also denies the write
        # APIs as a way to plant oversized objects for other consumers.
        if len(content) > MAX_READ_BYTES:
            raise AdapterError(
                ErrorCode.E2003_OBJECT_WRITE_FAILED,
                f"Content is {len(content)} bytes, exceeds the "
                f"{MAX_READ_BYTES} byte limit",
            )
        try:
            self._client.put_object(Bucket=self._bucket, Key=key, Body=content)
        except ClientError as exc:
            raise AdapterError(ErrorCode.E2003_OBJECT_WRITE_FAILED, str(exc)) from exc

    def edit(self, path: str, old: str, new: str, replace_all: bool = False) -> int:
        key = self._key(path)
        try:
            head = self._client.head_object(Bucket=self._bucket, Key=key)
            etag = head["ETag"]
            # edit() is a read-modify-write, so it materialises the body just
            # like read() and needs the same ceiling. The HEAD above already
            # carries ContentLength, so this check is free.
            size = head.get("ContentLength")
            if size is not None and size > MAX_READ_BYTES:
                raise AdapterError(
                    ErrorCode.E2002_OBJECT_READ_FAILED,
                    f"Object '{key}' is {size} bytes, exceeds the "
                    f"{MAX_READ_BYTES} byte read limit",
                )
            body = self._client.get_object(Bucket=self._bucket, Key=key)["Body"].read(
                MAX_READ_BYTES + 1
            )
            if len(body) > MAX_READ_BYTES:
                raise AdapterError(
                    ErrorCode.E2002_OBJECT_READ_FAILED,
                    f"Object '{key}' exceeds the {MAX_READ_BYTES} byte read limit",
                )
        except ClientError as exc:
            code = exc.response["Error"]["Code"]
            if code in ("NoSuchKey", "404"):
                raise AdapterError(
                    ErrorCode.E2001_OBJECT_NOT_FOUND, f"Key '{key}' not found"
                ) from exc
            raise AdapterError(ErrorCode.E2002_OBJECT_READ_FAILED, str(exc)) from exc

        text = body.decode("utf-8")
        occurrences: int = text.count(old) if replace_all else min(text.count(old), 1)
        updated = text.replace(old, new) if replace_all else text.replace(old, new, 1)

        try:
            self._client.put_object(
                Bucket=self._bucket,
                Key=key,
                Body=updated.encode("utf-8"),
                # Conditional write: reject if object changed since we read it
                # (boto3 passes unknown params through; real S3 honours If-Match)
                **{"IfMatch": etag} if etag else {},
            )
        except ClientError as exc:
            code = exc.response["Error"]["Code"]
            if code == "PreconditionFailed":
                raise AdapterError(
                    ErrorCode.E2008_EDIT_CONFLICT, "Concurrent modification detected"
                ) from exc
            raise AdapterError(ErrorCode.E2003_OBJECT_WRITE_FAILED, str(exc)) from exc

        return occurrences

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        responses: list[FileUploadResponse] = []
        for path, content in files:
            try:
                self.write(path, content)
                responses.append(FileUploadResponse(path=path))
            except AdapterError as exc:
                logger.warning("Upload failed for path: %s", path)
                responses.append(FileUploadResponse(path=path, error=_file_error(exc)))
        return responses

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        responses: list[FileDownloadResponse] = []
        for path in paths:
            try:
                responses.append(
                    FileDownloadResponse(path=path, content=self.read(path))
                )
            except AdapterError as exc:
                logger.warning("Download failed for path: %s", path)
                responses.append(
                    FileDownloadResponse(path=path, error=_file_error(exc))
                )
        return responses

    def list_keys(self, prefix: str = "") -> Iterator[tuple[str, str]]:
        paginator = self._client.get_paginator("list_objects_v2")
        try:
            for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix):
                for obj in page.get("Contents", []):
                    yield obj["Key"], obj.get("ETag", "").strip('"')
        except ClientError as exc:
            raise AdapterError(ErrorCode.E2005_LIST_FAILED, str(exc)) from exc

    def get_etag(self, key: str) -> str | None:
        """Return the current ETag for *key*, or None if not found."""
        try:
            head = self._client.head_object(Bucket=self._bucket, Key=key)
            etag: str = head["ETag"].strip('"')
            return etag
        except ClientError:
            return None

    def get_size(self, path: str) -> int:
        """Return the object's size in bytes via a HEAD request (no body fetch)."""
        key = self._key(path)
        try:
            head = self._client.head_object(Bucket=self._bucket, Key=key)
            return int(head["ContentLength"])
        except ClientError as exc:
            code = exc.response["Error"]["Code"]
            if code in ("NoSuchKey", "404"):
                raise AdapterError(
                    ErrorCode.E2001_OBJECT_NOT_FOUND, f"Key '{key}' not found"
                ) from exc
            raise AdapterError(ErrorCode.E2002_OBJECT_READ_FAILED, str(exc)) from exc
