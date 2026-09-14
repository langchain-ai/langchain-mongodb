"""Unit tests for MongoFilesystemBackend initialization reporting."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock

import pytest

from langchain_mongodb_deepagents_vfs.backend import MongoFilesystemBackend
from langchain_mongodb_deepagents_vfs.dtypes import SyncReport


def _backend_with_stubs(
    watcher_start_error: Exception | None = None,
    index_provision_error: Exception | None = None,
    initial_sync_error: Exception | None = None,
):
    """Build a backend far enough to run _background_init, without any I/O."""
    backend = MongoFilesystemBackend.__new__(MongoFilesystemBackend)
    backend.debug = False
    backend._prefix = ""
    backend._ready = threading.Event()
    backend.initial_sync_report = None
    backend.init_errors = []
    backend._index_provisioning_failed = False
    backend._initial_sync_failed = False
    backend._index_manager = MagicMock()
    if index_provision_error is not None:
        backend._index_manager.ensure_indexes.side_effect = index_provision_error
    backend._sync = MagicMock()
    if initial_sync_error is not None:
        backend._sync.run.side_effect = initial_sync_error
    else:
        backend._sync.run.return_value = SyncReport(seen=1, processed=1)
    backend._watcher = MagicMock()
    if watcher_start_error is not None:
        backend._watcher.start.side_effect = watcher_start_error
    backend._search = MagicMock()
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


@pytest.mark.unit
class TestSearchSurfacesInitFailures:
    """grep/glob/ls must not return confidently-empty results after a fatal
    init failure.

    Before this fix, _wait_ready() only blocked until the ready event was
    set — it never checked whether provisioning or sync had actually
    succeeded, so a fatal init failure was recorded in init_errors but the
    search methods proceeded anyway and returned empty results indistinguishable
    from "nothing matched".
    """

    def test_grep_errors_when_index_provisioning_failed(self):
        backend = _backend_with_stubs(
            index_provision_error=RuntimeError("no search index permissions")
        )
        backend._background_init()

        result = backend.grep("query")

        assert result.error is not None
        assert "index" in result.error.lower()
        backend._search.grep.assert_not_called()

    def test_glob_errors_when_initial_sync_failed(self):
        backend = _backend_with_stubs(initial_sync_error=RuntimeError("S3 unreachable"))
        backend._background_init()

        result = backend.glob("*.pdf")

        assert result.error is not None
        assert "sync" in result.error.lower()
        backend._search.glob.assert_not_called()

    def test_ls_succeeds_when_only_watcher_start_failed(self):
        """Watcher failure alone doesn't invalidate already-synced data."""
        backend = _backend_with_stubs(watcher_start_error=RuntimeError("no perms"))
        backend._background_init()

        backend._search.ls.return_value = "ls-result"
        result = backend.ls("docs/")

        assert result == "ls-result"
