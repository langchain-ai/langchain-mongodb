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
            yield S3Backend(bucket_name="watch-bucket", region_name="us-east-1"), client

    def test_on_created_callback_inserts_chunks(
        self, live_store, mongo_collection, mock_embedder, chunker, aws_credentials
    ):
        with mock_aws():
            store, client = live_store
            client.create_bucket(Bucket="watch-bucket")
            client.put_object(
                Bucket="watch-bucket", Key="docs/new.txt", Body=b"new doc content here"
            )
            store2 = S3Backend(bucket_name="watch-bucket", region_name="us-east-1")
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
            store = S3Backend(bucket_name="del-bucket", region_name="us-east-1")
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
            store = S3Backend(bucket_name="upd-bucket", region_name="us-east-1")
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

            store = S3Backend(bucket_name="sqs-bucket", region_name="us-east-1")
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
