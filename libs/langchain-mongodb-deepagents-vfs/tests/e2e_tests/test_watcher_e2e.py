"""E2E tests for PollingWatcher reacting to real S3 events.

Verifies that the watcher detects created, updated, and deleted objects
within a specific S3 prefix, and that objects outside the prefix are ignored.

Uses interval_seconds=10; each test allows up to WAIT_TIMEOUT=35 s for the
watcher to react (worst case: event arrives just after a poll fires).

Run with:
    export MONGODB_URI="mongodb+srv://..." S3_BUCKET_NAME="..." OPENAI_API_KEY="sk-..."
    pytest -m watcher_e2e -v -s
"""

from __future__ import annotations

import os
import threading
import time
import uuid

import boto3
import pymongo
import pytest

from langchain_mongodb_deepagents_vfs.backends.s3 import S3Backend
from langchain_mongodb_deepagents_vfs.chunker import Chunker
from langchain_mongodb_deepagents_vfs.embedder import Embedder
from langchain_mongodb_deepagents_vfs.watcher import PollingWatcher

_POLL_INTERVAL = 10
_WAIT_TIMEOUT = 35  # > 3 poll cycles to handle worst-case timing


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _wait_for(condition, timeout: float = _WAIT_TIMEOUT, step: float = 0.5) -> bool:
    """Spin until *condition()* is truthy or *timeout* seconds elapse."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(step)
    return False


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def watcher_prefix(e2e_prefix) -> str:
    """S3 prefix used exclusively by watcher tests, nested under the E2E prefix."""
    return f"{e2e_prefix}watcher/"


@pytest.fixture(scope="module")
def watcher_col(real_env):
    """Dedicated MongoDB collection for watcher E2E tests (isolated from main e2e)."""
    client = pymongo.MongoClient(real_env["MONGODB_URI"])
    col = client["langchain_mongodb_deepagents_vfs"]["watcher_e2e_chunks"]
    col.drop()
    yield col
    col.drop()


@pytest.fixture(scope="module")
def watcher_setup(real_env, s3_access, watcher_prefix, watcher_col):
    """Start a PollingWatcher with interval_seconds=10 against the real bucket.

    Injects tracking wrappers around on_created / on_updated / on_deleted so
    tests can assert which keys were detected.  Tears down the watcher and
    cleans up S3 objects on module exit.

    Yields a plain dict so individual tests don't need to destructure a fixture
    class.
    """
    bucket = real_env["S3_BUCKET_NAME"]
    region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    s3 = boto3.client("s3", region_name=region)

    store = S3Backend(bucket_name=bucket, region_name=region)
    embedder = Embedder()
    chunker = Chunker()

    created: list[str] = []
    updated: list[str] = []
    deleted: list[str] = []
    lock = threading.Lock()

    watcher = PollingWatcher(
        store=store,
        chunker=chunker,
        embedder=embedder,
        collection=watcher_col,
        interval_seconds=_POLL_INTERVAL,
        prefix=watcher_prefix,
    )

    # Wrap callbacks to record which keys triggered each event
    _orig_created = watcher.on_created
    _orig_updated = watcher.on_updated
    _orig_deleted = watcher.on_deleted

    def _track_created(key: str) -> None:
        with lock:
            created.append(key)
        _orig_created(key)

    def _track_updated(key: str) -> None:
        with lock:
            updated.append(key)
        _orig_updated(key)

    def _track_deleted(key: str) -> None:
        with lock:
            deleted.append(key)
        _orig_deleted(key)

    watcher.on_created = _track_created  # type: ignore[method-assign]
    watcher.on_updated = _track_updated  # type: ignore[method-assign]
    watcher.on_deleted = _track_deleted  # type: ignore[method-assign]

    watcher.start()

    yield {
        "watcher": watcher,
        "s3": s3,
        "bucket": bucket,
        "prefix": watcher_prefix,
        "region": region,
        "col": watcher_col,
        "created": created,
        "updated": updated,
        "deleted": deleted,
        "lock": lock,
    }

    watcher.stop()

    # Best-effort S3 cleanup
    try:
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=watcher_prefix):
            objs = [{"Key": o["Key"]} for o in page.get("Contents", [])]
            if objs:
                s3.delete_objects(Bucket=bucket, Delete={"Objects": objs})
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.real_e2e
@pytest.mark.watcher_e2e
class TestPollingWatcherE2E:
    """PollingWatcher reacts to S3 create / update / delete events in the prefix."""

    def test_detects_new_file(self, watcher_setup):
        """Uploading a new key under the watched prefix fires on_created."""
        s = watcher_setup
        key = f"{s['prefix']}create_{uuid.uuid4().hex[:6]}.txt"

        s["s3"].put_object(Bucket=s["bucket"], Key=key, Body=b"hello watcher")

        assert _wait_for(lambda: key in s["created"]), (
            f"on_created was not fired for {key!r} within {_WAIT_TIMEOUT}s. "
            f"created so far: {s['created']}"
        )

    def test_detects_updated_file(self, watcher_setup):
        """Re-uploading an existing key with new content fires on_updated (ETag changes)."""
        s = watcher_setup
        key = f"{s['prefix']}update_{uuid.uuid4().hex[:6]}.txt"

        # Upload v1 and wait for the watcher to establish its baseline
        s["s3"].put_object(Bucket=s["bucket"], Key=key, Body=b"version 1")
        assert _wait_for(lambda: key in s["created"]), (
            f"on_created not fired for {key!r} — cannot proceed to update assertion"
        )

        # Upload v2 (different body → different ETag)
        s["s3"].put_object(
            Bucket=s["bucket"], Key=key, Body=b"version 2 - changed content"
        )
        assert _wait_for(lambda: key in s["updated"]), (
            f"on_updated was not fired for {key!r} within {_WAIT_TIMEOUT}s. "
            f"updated so far: {s['updated']}"
        )

    def test_detects_deleted_file(self, watcher_setup):
        """Deleting a key fires on_deleted and removes its chunks from MongoDB."""
        s = watcher_setup
        key = f"{s['prefix']}delete_{uuid.uuid4().hex[:6]}.txt"

        s["s3"].put_object(Bucket=s["bucket"], Key=key, Body=b"will be deleted")
        assert _wait_for(lambda: key in s["created"]), (
            f"on_created not fired for {key!r} — cannot proceed to delete assertion"
        )

        s["s3"].delete_object(Bucket=s["bucket"], Key=key)
        assert _wait_for(lambda: key in s["deleted"]), (
            f"on_deleted was not fired for {key!r} within {_WAIT_TIMEOUT}s. "
            f"deleted so far: {s['deleted']}"
        )

        # Verify MongoDB chunks were cleaned up
        remaining = s["col"].count_documents({"source_path": key})
        assert remaining == 0, (
            f"Expected 0 MongoDB chunks for deleted key {key!r}, found {remaining}"
        )

    def test_prefix_isolation(self, watcher_setup, e2e_prefix):
        """A key uploaded outside the watched prefix must NOT trigger any callback."""
        s = watcher_setup
        outside_key = (
            f"{e2e_prefix}outside_watcher/isolation_{uuid.uuid4().hex[:6]}.txt"
        )

        s["s3"].put_object(
            Bucket=s["bucket"], Key=outside_key, Body=b"outside watched prefix"
        )

        # Wait two full poll cycles to be certain the watcher has had a chance to see it
        time.sleep(_POLL_INTERVAL * 2 + 3)

        with s["lock"]:
            all_seen = set(s["created"] + s["updated"] + s["deleted"])

        assert outside_key not in all_seen, (
            f"Key outside watched prefix was incorrectly detected: {outside_key!r}"
        )

        # Cleanup the out-of-prefix object
        try:
            s["s3"].delete_object(Bucket=s["bucket"], Key=outside_key)
        except Exception:
            pass
