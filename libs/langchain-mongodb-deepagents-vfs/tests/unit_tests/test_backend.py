"""Unit tests for MongoFilesystemBackend initialization reporting."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock

import pytest

from langchain_mongodb_deepagents_vfs.backend import MongoFilesystemBackend
from langchain_mongodb_deepagents_vfs.dtypes import SyncReport


def _backend_with_stubs(watcher_start_error: Exception | None = None):
    """Build a backend far enough to run _background_init, without any I/O."""
    backend = MongoFilesystemBackend.__new__(MongoFilesystemBackend)
    backend._prefix = ""
    backend._ready = threading.Event()
    backend.initial_sync_report = None
    backend.init_errors = []
    backend._index_manager = MagicMock()
    backend._sync = MagicMock()
    backend._sync.run.return_value = SyncReport(seen=1, processed=1)
    backend._watcher = MagicMock()
    if watcher_start_error is not None:
        backend._watcher.start.side_effect = watcher_start_error
    return backend


@pytest.mark.unit
class TestBackgroundInitReporting:
    def test_watcher_start_failure_is_recorded(self):
        """A watcher that never starts must surface, not just log.

        It is the quietest initialization failure: the collection looks correct
        and simply stops tracking S3, unlike a failed sync which shows up
        immediately as an empty collection. init_errors is what the e2e
        real_backend fixture gates on, so an unrecorded failure is invisible.
        """
        backend = _backend_with_stubs(RuntimeError("no SQS permissions"))
        backend._background_init()

        assert any("watcher start failed" in e for e in backend.init_errors), (
            backend.init_errors
        )
        assert "no SQS permissions" in " ".join(backend.init_errors)
        # Still ready: sync succeeded, so reads and search work on what synced.
        assert backend._ready.is_set()

    def test_clean_init_records_no_errors(self):
        backend = _backend_with_stubs()
        backend._background_init()

        assert backend.init_errors == []
        assert backend.initial_sync_report is not None
        assert backend.initial_sync_report.failed == 0
        assert backend._ready.is_set()
