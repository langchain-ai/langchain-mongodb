"""SQS-based S3 event watcher (production-recommended).

Consumes S3 event notifications from an SQS queue. Each message carries the
S3 event type (ObjectCreated / ObjectRemoved) and the affected key.

Long-polling (WaitTimeSeconds=20) keeps AWS API calls minimal. Visibility
timeout management ensures a message is not re-processed if the ingest thread
crashes mid-flight.

Prerequisites:
  - Configure your S3 bucket to send event notifications to an SQS queue.
  - Grant the IAM role s3:GetObject + sqs:ReceiveMessage + sqs:DeleteMessage.
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Any

import boto3
from botocore.exceptions import ClientError
from pymongo.collection import Collection

from langchain_mongodb_deepagents_vfs.backends.base import ObjectStoreBackend
from langchain_mongodb_deepagents_vfs.chunker import Chunker
from langchain_mongodb_deepagents_vfs.embedder import Embedder
from langchain_mongodb_deepagents_vfs.errors import AdapterError, ErrorCode
from langchain_mongodb_deepagents_vfs.watcher.base import S3Watcher

logger = logging.getLogger(__name__)

_VISIBILITY_TIMEOUT = 120  # seconds — must be > ingest time for largest object
_LONG_POLL_WAIT = 20  # SQS long-polling max
_MAX_MESSAGES = 10


class SQSWatcher(S3Watcher):
    """Event-driven watcher backed by an SQS queue of S3 event notifications.

    Args:
        store: Object store backend.
        chunker: Chunker instance.
        embedder: Embedder instance.
        collection: MongoDB collection.
        queue_url: Full SQS queue URL.
        region_name: AWS region.
        visibility_timeout: How long (seconds) a received message stays
            invisible to other consumers while being processed.
        endpoint_url: Override for local testing (e.g. LocalStack).
    """

    def __init__(
        self,
        store: ObjectStoreBackend,
        chunker: Chunker,
        embedder: Embedder,
        collection: Collection,  # type: ignore[type-arg]
        queue_url: str,
        region_name: str | None = None,
        visibility_timeout: int = _VISIBILITY_TIMEOUT,
        endpoint_url: str | None = None,
    ) -> None:
        super().__init__(store, chunker, embedder, collection)
        self._queue_url = queue_url
        self._visibility_timeout = visibility_timeout
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._sqs = boto3.client(
            "sqs",
            region_name=region_name,
            endpoint_url=endpoint_url,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop, name="SQSWatcher", daemon=True
        )
        self._thread.start()
        logger.info("SQSWatcher started, consuming queue: %s", self._queue_url)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=_LONG_POLL_WAIT + 10)
        logger.info("SQSWatcher stopped.")

    def __enter__(self) -> SQSWatcher:
        self.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Internal loop
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._receive_and_process()
            except Exception as exc:
                logger.error("SQSWatcher: unhandled error in receive loop: %s", exc)

    def _receive_and_process(self) -> None:
        try:
            response = self._sqs.receive_message(
                QueueUrl=self._queue_url,
                MaxNumberOfMessages=_MAX_MESSAGES,
                WaitTimeSeconds=_LONG_POLL_WAIT,
                VisibilityTimeout=self._visibility_timeout,
            )
        except ClientError as exc:
            logger.error("SQSWatcher: ReceiveMessage failed: %s", exc)
            raise AdapterError(ErrorCode.E6001_SQS_RECEIVE_FAILED, str(exc)) from exc

        for msg in response.get("Messages", []):
            receipt = msg["ReceiptHandle"]
            try:
                self._process_message(msg["Body"])
                self._sqs.delete_message(
                    QueueUrl=self._queue_url, ReceiptHandle=receipt
                )
            except Exception as exc:
                logger.warning(
                    "SQSWatcher: message processing failed, will retry: %s", exc
                )

    def _process_message(self, body: str) -> None:
        try:
            envelope = json.loads(body)
            # SQS may wrap SNS notifications in an extra layer
            if "Message" in envelope:
                inner = json.loads(envelope["Message"])
            else:
                inner = envelope
            records = inner.get("Records", [])
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning("SQSWatcher: failed to parse event body: %s", exc)
            raise AdapterError(ErrorCode.E6002_EVENT_PARSE_FAILED, str(exc)) from exc

        for record in records:
            event_name: str = record.get("eventName", "")
            try:
                key = record["s3"]["object"]["key"]
            except KeyError:
                logger.warning(
                    "SQSWatcher: missing s3.object.key in record: %s", record
                )
                continue

            if event_name.startswith("ObjectCreated"):
                self.on_created(key)
            elif event_name.startswith("ObjectRemoved"):
                self.on_deleted(key)
            else:
                logger.debug(
                    "SQSWatcher: unrecognized event '%s', skipping.", event_name
                )
