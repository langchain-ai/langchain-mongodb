"""Integration tests for watcher callbacks against moto S3 + mongomock."""

from __future__ import annotations

import json

import boto3
import pytest
from moto import mock_aws

from langchain_mongodb_deepagents_vfs.backends.s3 import S3Backend
from langchain_mongodb_deepagents_vfs.watcher.polling import PollingWatcher
from langchain_mongodb_deepagents_vfs.watcher.sqs import SQSWatcher


@pytest.mark.integration
class TestPollingWatcherIntegration:
    @pytest.fixture
    def live_store(self, aws_credentials):
        with mock_aws():
            client = boto3.client("s3", region_name="us-east-1")
            client.create_bucket(Bucket="watch-bucket")
            # Pre-populate
            client.put_object(
                Bucket="watch-bucket", Key="docs/a.txt", Body=b"initial content"
            )
            yield (
                S3Backend(
                    bucket_name="watch-bucket", region_name="us-east-1", prefix=""
                ),
                client,
            )

    def test_oversized_object_is_not_ingested(
        self, mongo_collection, mock_embedder, chunker, aws_credentials, monkeypatch
    ):
        """Watcher ingest must refuse an oversized object.

        Regression for the Corridor finding: _ingest() called store.read()
        with no size cap, so a large upload to the watched prefix would be
        loaded whole into the background daemon thread, then chunked and
        embedded. _ingest already handles AdapterError, so the guard in
        S3Backend.read() makes it skip and log rather than crash the thread.
        """
        import langchain_mongodb_deepagents_vfs.backends.s3 as s3_mod

        monkeypatch.setattr(s3_mod, "MAX_READ_BYTES", 100)
        with mock_aws():
            client = boto3.client("s3", region_name="us-east-1")
            client.create_bucket(Bucket="cap-watch")
            client.put_object(Bucket="cap-watch", Key="bomb.txt", Body=b"word " * 200)
            store = S3Backend(
                bucket_name="cap-watch", region_name="us-east-1", prefix=""
            )
            watcher = PollingWatcher(store, chunker, mock_embedder, mongo_collection)

            watcher.on_created("bomb.txt")  # must not raise

            assert mongo_collection.count_documents({"source_path": "bomb.txt"}) == 0
            mock_embedder.embed_batch.assert_not_called()

    def test_on_created_callback_inserts_chunks(
        self, live_store, mongo_collection, mock_embedder, chunker, aws_credentials
    ):
        with mock_aws():
            store, client = live_store
            client.create_bucket(Bucket="watch-bucket")
            client.put_object(
                Bucket="watch-bucket", Key="docs/new.txt", Body=b"new doc content here"
            )
            store2 = S3Backend(
                bucket_name="watch-bucket", region_name="us-east-1", prefix=""
            )
            watcher = PollingWatcher(store2, chunker, mock_embedder, mongo_collection)
            watcher.on_created("docs/new.txt")
            count = mongo_collection.count_documents({"source_path": "docs/new.txt"})
            assert count > 0

    def test_on_deleted_callback_removes_chunks(
        self, mongo_collection, mock_embedder, chunker, aws_credentials
    ):
        with mock_aws():
            client = boto3.client("s3", region_name="us-east-1")
            client.create_bucket(Bucket="del-bucket")
            client.put_object(
                Bucket="del-bucket", Key="docs/a.txt", Body=b"will be deleted"
            )
            store = S3Backend(
                bucket_name="del-bucket", region_name="us-east-1", prefix=""
            )
            watcher = PollingWatcher(store, chunker, mock_embedder, mongo_collection)
            # Ingest first
            watcher.on_created("docs/a.txt")
            assert mongo_collection.count_documents({"source_path": "docs/a.txt"}) > 0
            # Now delete
            watcher.on_deleted("docs/a.txt")
            assert mongo_collection.count_documents({"source_path": "docs/a.txt"}) == 0

    def test_on_updated_callback_replaces_chunks(
        self, mongo_collection, mock_embedder, chunker, aws_credentials
    ):
        with mock_aws():
            client = boto3.client("s3", region_name="us-east-1")
            client.create_bucket(Bucket="upd-bucket")
            client.put_object(Bucket="upd-bucket", Key="f.txt", Body=b"original")
            store = S3Backend(
                bucket_name="upd-bucket", region_name="us-east-1", prefix=""
            )
            watcher = PollingWatcher(store, chunker, mock_embedder, mongo_collection)
            watcher.on_created("f.txt")
            count_before = mongo_collection.count_documents({"source_path": "f.txt"})
            assert count_before > 0
            client.put_object(
                Bucket="upd-bucket", Key="f.txt", Body=b"updated content here"
            )
            watcher.on_updated("f.txt")
            count_after = mongo_collection.count_documents({"source_path": "f.txt"})
            assert count_after > 0


@pytest.mark.integration
class TestSQSWatcherIntegration:
    def test_sqs_watcher_enforces_prefix_scope(
        self, aws_credentials, mongo_collection, mock_embedder, chunker
    ):
        """Events for keys outside the configured prefix must be discarded.

        A queue can be subscribed to an entire shared bucket, so without this
        filter an out-of-scope object is indexed into the shared collection and
        becomes discoverable through grep/glob/ls.
        """
        with mock_aws():
            s3 = boto3.client("s3", region_name="us-east-1")
            s3.create_bucket(Bucket="tenant-bucket")
            s3.put_object(
                Bucket="tenant-bucket", Key="tenant_a/mine.txt", Body=b"in scope"
            )
            s3.put_object(
                Bucket="tenant-bucket", Key="tenant_b/secret.txt", Body=b"out of scope"
            )
            sqs = boto3.client("sqs", region_name="us-east-1")
            queue_url = sqs.create_queue(QueueName="scoped-queue")["QueueUrl"]

            store = S3Backend(
                bucket_name="tenant-bucket", region_name="us-east-1", prefix=""
            )
            watcher = SQSWatcher(
                store=store,
                chunker=chunker,
                embedder=mock_embedder,
                collection=mongo_collection,
                queue_url=queue_url,
                region_name="us-east-1",
                prefix="tenant_a/",
            )

            # Both records in one message: the filter is per-record, and one
            # long-poll keeps the test from costing two of them.
            sqs.send_message(
                QueueUrl=queue_url,
                MessageBody=json.dumps(
                    {
                        "Records": [
                            {
                                "eventName": "ObjectCreated:Put",
                                "s3": {"object": {"key": key}},
                            }
                            for key in ("tenant_a/mine.txt", "tenant_b/secret.txt")
                        ]
                    }
                ),
            )
            watcher._receive_and_process()

            assert (
                mongo_collection.count_documents({"source_path": "tenant_a/mine.txt"})
                > 0
            )
            assert (
                mongo_collection.count_documents({"source_path": "tenant_b/secret.txt"})
                == 0
            )

    def test_sqs_watcher_processes_create_event(
        self, aws_credentials, mongo_collection, mock_embedder, chunker
    ):
        with mock_aws():
            # Setup S3
            s3 = boto3.client("s3", region_name="us-east-1")
            s3.create_bucket(Bucket="sqs-bucket")
            s3.put_object(
                Bucket="sqs-bucket", Key="docs/file.txt", Body=b"SQS test content"
            )

            # Setup SQS
            sqs = boto3.client("sqs", region_name="us-east-1")
            q = sqs.create_queue(QueueName="events-queue")
            queue_url = q["QueueUrl"]

            store = S3Backend(
                bucket_name="sqs-bucket", region_name="us-east-1", prefix=""
            )
            watcher = SQSWatcher(
                store=store,
                chunker=chunker,
                embedder=mock_embedder,
                collection=mongo_collection,
                queue_url=queue_url,
                region_name="us-east-1",
            )

            # Send a fake S3 create event to SQS
            event = json.dumps(
                {
                    "Records": [
                        {
                            "eventName": "ObjectCreated:Put",
                            "s3": {"object": {"key": "docs/file.txt"}},
                        }
                    ]
                }
            )
            sqs.send_message(QueueUrl=queue_url, MessageBody=event)

            # Manually trigger receive-and-process
            watcher._receive_and_process()
            count = mongo_collection.count_documents({"source_path": "docs/file.txt"})
            assert count > 0
