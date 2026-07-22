"""Fixtures for real-service E2E tests.

Reads credentials from environment variables. Every fixture that touches a
real service starts with a ``pytest.skip`` if the required vars are absent so
the suite stays green in CI where credentials are not available.
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import boto3
import pytest
from pymongo import monitoring

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REQUIRED = {
    "MONGODB_URI": "MongoDB Atlas connection string",
    "S3_BUCKET_NAME": "S3 bucket name",
}

# Provider-specific credential check: at least one embedding provider must be configured.
_PROVIDER_CHECKS: dict[str, list[str]] = {
    "openai": ["OPENAI_API_KEY"],
    "bedrock": [],  # uses boto3 credential chain — no extra env vars needed
}

_TEST_PREFIX = os.getenv("S3_TEST_PREFIX", "langchain_mongodb_deepagents_vfs_e2e_test/")


def _require_env(*keys: str) -> dict[str, str]:
    """Return {key: value} for *keys*, skipping the test if any are missing."""
    missing = [k for k in keys if not os.getenv(k)]
    if missing:
        pytest.skip(f"Missing env vars: {', '.join(missing)}")
    return {k: os.environ[k] for k in keys}


# ---------------------------------------------------------------------------
# S3 cleanup helper
# ---------------------------------------------------------------------------


def _atlas_search_ready(backend: Any, keys: list[str], timeout: int = 60) -> None:
    """Poll until every key in *keys* is findable via glob, or *timeout* seconds elapse.

    Atlas Search indexes new documents asynchronously; this bridges the gap between
    MongoDB upsert completion (_ready) and the documents appearing in search results.
    """
    import fnmatch as _fnmatch

    deadline = time.monotonic() + timeout
    remaining = set(keys)
    while remaining and time.monotonic() < deadline:
        still_missing: set[str] = set()
        for key in remaining:
            filename = key.split("/")[-1]
            ext = filename.rsplit(".", 1)[-1] if "." in filename else ""
            pattern = f"*.{ext}" if ext else filename
            result = backend.glob(pattern)
            found = {
                m["path"]
                for m in (result.matches or [])
                if _fnmatch.fnmatch(m["path"].split("/")[-1], pattern)
            }
            if key not in found:
                still_missing.add(key)
        remaining = still_missing
        if remaining:
            time.sleep(2)


def _delete_prefix(bucket: str, prefix: str, region: str | None) -> None:
    """Delete all objects under *prefix* in *bucket* (best-effort cleanup)."""
    client = boto3.client("s3", region_name=region)
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        objs = [{"Key": o["Key"]} for o in page.get("Contents", [])]
        if objs:
            client.delete_objects(Bucket=bucket, Delete={"Objects": objs})


# ---------------------------------------------------------------------------
# MongoDB command logging (request/response capture)
# ---------------------------------------------------------------------------

_TRUNCATE_KEYS = {"embedding", "queryVector", "vector"}
_MAX_LIST_PREVIEW = 4


def _redact(obj: Any, depth: int = 0) -> Any:
    """Recursively shrink huge embedding vectors so logs stay readable."""
    if depth > 8:
        return "<...>"
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k in _TRUNCATE_KEYS and isinstance(v, list):
                out[k] = f"<{type(v).__name__} len={len(v)} sample={v[:3]}...>"
            else:
                out[k] = _redact(v, depth + 1)
        return out
    if isinstance(obj, list):
        if len(obj) > _MAX_LIST_PREVIEW and obj and isinstance(obj[0], (int, float)):
            return f"<numeric list len={len(obj)} sample={obj[:3]}...>"
        return [_redact(v, depth + 1) for v in obj[:50]]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return repr(obj)


class _MongoCommandLogger(monitoring.CommandListener):
    """pymongo.monitoring.CommandListener: prints every command + reply."""

    def __init__(self, sink) -> None:
        self.sink = sink
        # Mongo emits a heartbeat 'isMaster'/'hello' every ~10s; filter them out.
        self._noisy = {
            "hello",
            "ismaster",
            "ping",
            "endSessions",
            "buildInfo",
            "saslStart",
            "saslContinue",
            "getMore",
        }

    def _emit(self, kind: str, payload: dict) -> None:
        line = json.dumps(
            {
                "ts": datetime.utcnow().isoformat(timespec="milliseconds") + "Z",
                "kind": kind,
                **payload,
            },
            default=str,
        )
        self.sink.write(line + "\n")
        self.sink.flush()
        # Also surface to pytest's captured stdout (-s shows it live)
        print(f"[mongo {kind}] {line}", file=sys.stderr)

    def started(self, event) -> None:
        if event.command_name in self._noisy:
            return
        self._emit(
            "REQUEST",
            {
                "request_id": event.request_id,
                "db": event.database_name,
                "command_name": event.command_name,
                "command": _redact(dict(event.command)),
            },
        )

    def succeeded(self, event) -> None:
        if event.command_name in self._noisy:
            return
        self._emit(
            "RESPONSE",
            {
                "request_id": event.request_id,
                "command_name": event.command_name,
                "duration_ms": round(event.duration_micros / 1000, 2),
                "reply": _redact(dict(event.reply)),
            },
        )

    def failed(self, event) -> None:
        if event.command_name in self._noisy:
            return
        self._emit(
            "FAILED",
            {
                "request_id": event.request_id,
                "command_name": event.command_name,
                "duration_ms": round(event.duration_micros / 1000, 2),
                "failure": str(event.failure),
            },
        )


@pytest.fixture(scope="session", autouse=True)
def _mongo_command_log(request):
    """Auto-register a pymongo CommandListener for the whole real_e2e session.

    Activated whenever any test under tests/e2e/ runs.  Writes to
    ``MONGO_LOG_FILE`` if set, otherwise pytest's per-session tmp dir (so the
    repo working tree stays clean). Also mirrors to stderr.
    Set ``MONGO_LOG=0`` to disable.
    """
    if os.getenv("MONGO_LOG", "1") == "0":
        yield
        return

    import pymongo.monitoring as monitoring

    filename = f"mongo-e2e-{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}.log"
    default_path = request.getfixturevalue("tmp_path_factory").mktemp("mongo-log") / filename
    log_path = Path(os.getenv("MONGO_LOG_FILE", str(default_path))).resolve()
    sink = log_path.open("w", encoding="utf-8")
    listener = _MongoCommandLogger(sink)
    monitoring.register(listener)

    print(f"\n[mongo-log] capturing MongoDB commands to {log_path}", file=sys.stderr)
    try:
        yield listener
    finally:
        sink.close()
        print(f"[mongo-log] log written to {log_path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def real_env() -> dict[str, str]:
    """Skip session if any required credential is absent, or if the chosen
    embedding provider is missing its required env vars."""
    env = _require_env(*_REQUIRED)
    provider = os.getenv("EMBEDDING_PROVIDER", "openai").lower()
    required_for_provider = _PROVIDER_CHECKS.get(provider)
    if required_for_provider is None:
        pytest.skip(
            f"Unknown EMBEDDING_PROVIDER '{provider}'. "
            f"Supported: {', '.join(_PROVIDER_CHECKS)}"
        )
    missing = [k for k in required_for_provider if not os.getenv(k)]
    if missing:
        pytest.skip(f"EMBEDDING_PROVIDER={provider} requires: {', '.join(missing)}")
    return env


@pytest.fixture(scope="session")
def e2e_prefix(real_env) -> str:
    """Unique S3 prefix so parallel runs and reruns don't collide."""
    run_id = uuid.uuid4().hex[:8]
    return f"{_TEST_PREFIX}{run_id}/"


@pytest.fixture(scope="session")
def real_backend(real_env, e2e_prefix):
    """MongoFilesystemBackend connected to real Atlas, S3, and OpenAI.

    Seeds two files under *e2e_prefix*, waits for the initial sync, then
    yields the backend.  Cleans up Atlas chunks and S3 objects on teardown.
    """
    from langchain_mongodb_deepagents_vfs.backend import MongoFilesystemBackend

    bucket = real_env["S3_BUCKET_NAME"]
    region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

    # Seed test files directly via S3 so the initial sync picks them up
    s3 = boto3.client("s3", region_name=region)
    seed = {
        f"{e2e_prefix}docs/api.txt": b"authentication flow and token refresh",
        f"{e2e_prefix}docs/install.txt": b"installation prerequisites and setup guide",
        f"{e2e_prefix}src/main.py": b"def main():\n    print('hello world')\n",
    }
    for key, body in seed.items():
        s3.put_object(Bucket=bucket, Key=key, Body=body)

    backend = MongoFilesystemBackend(
        s3_bucket_name=bucket,
        mongodb_connection_string=real_env["MONGODB_URI"],
        aws_region=region,
        s3_prefix=e2e_prefix,
        debug=False,  # errors surface as DTO fields, not exceptions
    )
    # Block until initial sync + index setup completes (up to 5 min on cold Atlas)
    ready = backend._ready.wait(timeout=300)
    if not ready:
        backend.stop()
        pytest.fail(
            "Backend did not become ready within 300 s — check Atlas connectivity"
        )

    # Atlas Search indexes new documents asynchronously after MongoDB upsert.
    # Poll until every seeded file is visible in glob results (or timeout after 60 s).
    _atlas_search_ready(backend, list(seed.keys()), timeout=60)

    yield backend

    backend.stop()

    # Teardown: remove test chunks from Atlas and test objects from S3
    try:
        backend._col.delete_many({"source_path": {"$regex": f"^{e2e_prefix}"}})
    except Exception:
        pass
    _delete_prefix(bucket, e2e_prefix, region)


@pytest.fixture(scope="session")
def sync_ready(real_backend, e2e_prefix):
    """Skip any test that needs indexed MongoDB data if the initial sync
    produced no documents.  This happens when the embedding API quota is
    exceeded during sync — every file fails silently, _ready fires, but the
    collection stays empty.  Surfaces a clear message instead of confusing
    empty-result assertion failures.
    """
    count = real_backend._col.count_documents(
        {"source_path": {"$regex": f"^{e2e_prefix}"}}
    )
    if count == 0:
        provider = os.getenv("EMBEDDING_PROVIDER", "openai")
        pytest.skip(
            f"Initial sync indexed 0 documents (EMBEDDING_PROVIDER={provider}). "
            "Check embedding API quota/credentials in the logs above."
        )
