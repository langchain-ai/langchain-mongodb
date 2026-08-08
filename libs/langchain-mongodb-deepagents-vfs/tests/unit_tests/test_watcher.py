"""Unit tests for PollingWatcher and SQSWatcher."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from langchain_mongodb_deepagents_vfs.watcher.polling import PollingWatcher
from langchain_mongodb_deepagents_vfs.watcher.sqs import SQSWatcher


@pytest.mark.unit
class TestPollingWatcher:
    def _make_watcher(self, mongo_collection, mock_embedder, chunker):
        store = MagicMock()
        store.list_keys.return_value = iter([])
        store.read.return_value = b"content"
        watcher = PollingWatcher(
            store=store,
            chunker=chunker,
            embedder=mock_embedder,
            collection=mongo_collection,
            interval_seconds=1,
        )
        return watcher, store

    def test_start_stop(self, mongo_collection, mock_embedder, chunker):
        watcher, _ = self._make_watcher(mongo_collection, mock_embedder, chunker)
        watcher.start()
        assert watcher._thread is not None
        assert watcher._thread.is_alive()
        watcher.stop()
        assert not watcher._thread.is_alive()

    def test_context_manager(self, mongo_collection, mock_embedder, chunker):
        watcher, _ = self._make_watcher(mongo_collection, mock_embedder, chunker)
        with watcher:
            assert watcher._thread.is_alive()
        assert not watcher._thread.is_alive()

    def test_detects_created_key(self, mongo_collection, mock_embedder, chunker):
        store = MagicMock()
        store.read.return_value = b"new content"

        created_keys = []

        class TrackingWatcher(PollingWatcher):
            def on_created(self, key):
                created_keys.append(key)

        store.list_keys.side_effect = [
            iter([]),  # first poll: empty
            iter([("docs/a.txt", "etag1")]),  # second poll: new key
        ]

        watcher = TrackingWatcher(
            store=store,
            chunker=chunker,
            embedder=mock_embedder,
            collection=mongo_collection,
            interval_seconds=0,
        )
        # Manually call poll twice
        watcher._poll()
        watcher._poll()
        assert "docs/a.txt" in created_keys

    def test_detects_deleted_key(self, mongo_collection, mock_embedder, chunker):
        store = MagicMock()
        store.read.return_value = b"content"
        deleted_keys = []

        class TrackingWatcher(PollingWatcher):
            def on_deleted(self, key):
                deleted_keys.append(key)

        store.list_keys.side_effect = [
            iter([("docs/a.txt", "etag1")]),  # first poll: key exists
            iter([]),  # second poll: key gone
        ]

        watcher = TrackingWatcher(
            store=store,
            chunker=chunker,
            embedder=mock_embedder,
            collection=mongo_collection,
            interval_seconds=0,
        )
        watcher._poll()
        watcher._poll()
        assert "docs/a.txt" in deleted_keys

    def test_detects_updated_key(self, mongo_collection, mock_embedder, chunker):
        store = MagicMock()
        store.read.return_value = b"updated"
        updated_keys = []

        class TrackingWatcher(PollingWatcher):
            def on_updated(self, key):
                updated_keys.append(key)

        store.list_keys.side_effect = [
            iter([("docs/a.txt", "etag1")]),  # first poll
            iter([("docs/a.txt", "etag2")]),  # second poll: ETag changed
        ]

        watcher = TrackingWatcher(
            store=store,
            chunker=chunker,
            embedder=mock_embedder,
            collection=mongo_collection,
            interval_seconds=0,
        )
        watcher._poll()
        watcher._poll()
        assert "docs/a.txt" in updated_keys


@pytest.mark.unit
class TestSQSWatcherMessageParsing:
    def _make_sqs_watcher(
        self, mongo_collection, mock_embedder, chunker, sqs_client=None
    ):
        store = MagicMock()
        store.read.return_value = b"file content"
        watcher = SQSWatcher(
            store=store,
            chunker=chunker,
            embedder=mock_embedder,
            collection=mongo_collection,
            queue_url="https://sqs.us-east-1.amazonaws.com/123/test-queue",
            # Explicit region: SQSWatcher builds a boto3 client in __init__, and
            # these are pure message-parsing tests that make no AWS calls. Without
            # this they pass only where the environment happens to supply a
            # region, and fail on CI with NoRegionError.
            region_name="us-east-1",
        )
        if sqs_client:
            watcher._sqs = sqs_client
        return watcher, store

    def test_process_created_event(self, mongo_collection, mock_embedder, chunker):
        watcher, store = self._make_sqs_watcher(
            mongo_collection, mock_embedder, chunker
        )
        created_keys = []
        watcher.on_created = lambda key: created_keys.append(key)

        body = json.dumps(
            {
                "Records": [
                    {
                        "eventName": "ObjectCreated:Put",
                        "s3": {"object": {"key": "docs/newfile.txt"}},
                    }
                ]
            }
        )
        watcher._process_message(body)
        assert "docs/newfile.txt" in created_keys

    def test_process_deleted_event(self, mongo_collection, mock_embedder, chunker):
        watcher, store = self._make_sqs_watcher(
            mongo_collection, mock_embedder, chunker
        )
        deleted_keys = []
        watcher.on_deleted = lambda key: deleted_keys.append(key)

        body = json.dumps(
            {
                "Records": [
                    {
                        "eventName": "ObjectRemoved:Delete",
                        "s3": {"object": {"key": "docs/old.txt"}},
                    }
                ]
            }
        )
        watcher._process_message(body)
        assert "docs/old.txt" in deleted_keys

    def test_process_sns_wrapped_message(
        self, mongo_collection, mock_embedder, chunker
    ):
        watcher, store = self._make_sqs_watcher(
            mongo_collection, mock_embedder, chunker
        )
        created_keys = []
        watcher.on_created = lambda key: created_keys.append(key)

        inner = json.dumps(
            {
                "Records": [
                    {
                        "eventName": "ObjectCreated:Put",
                        "s3": {"object": {"key": "wrapped/file.txt"}},
                    }
                ]
            }
        )
        body = json.dumps({"Message": inner})
        watcher._process_message(body)
        assert "wrapped/file.txt" in created_keys

    def test_unknown_event_is_ignored(self, mongo_collection, mock_embedder, chunker):
        watcher, store = self._make_sqs_watcher(
            mongo_collection, mock_embedder, chunker
        )
        watcher.on_created = MagicMock()
        watcher.on_deleted = MagicMock()

        body = json.dumps(
            {
                "Records": [
                    {
                        "eventName": "ObjectRestore:Completed",
                        "s3": {"object": {"key": "x.txt"}},
                    }
                ]
            }
        )
        watcher._process_message(body)
        watcher.on_created.assert_not_called()
        watcher.on_deleted.assert_not_called()

    def test_malformed_json_raises_adapter_error(
        self, mongo_collection, mock_embedder, chunker
    ):
        from langchain_mongodb_deepagents_vfs.errors import AdapterError, ErrorCode

        watcher, _ = self._make_sqs_watcher(mongo_collection, mock_embedder, chunker)
        with pytest.raises(AdapterError) as exc_info:
            watcher._process_message("not json at all {{{")
        assert exc_info.value.code == ErrorCode.E6002_EVENT_PARSE_FAILED
