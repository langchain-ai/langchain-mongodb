"""Integration tests for InitialSync against moto S3 + mongomock."""

from __future__ import annotations

import boto3
import pytest
from moto import mock_aws

from langchain_mongodb_deepagents_vfs.backends.s3 import S3Backend
from langchain_mongodb_deepagents_vfs.sync import InitialSync


@pytest.mark.integration
class TestInitialSyncBasic:
    @pytest.fixture
    def setup_bucket(self, aws_credentials):
        with mock_aws():
            client = boto3.client("s3", region_name="us-east-1")
            client.create_bucket(Bucket="sync-bucket")
            client.put_object(
                Bucket="sync-bucket", Key="docs/a.txt", Body=b"Content of A"
            )
            client.put_object(
                Bucket="sync-bucket", Key="docs/b.txt", Body=b"Content of B"
            )
            yield "sync-bucket"

    def test_sync_inserts_chunks(
        self, setup_bucket, mongo_collection, mock_embedder, chunker
    ):
        with mock_aws():
            store = S3Backend(bucket_name=setup_bucket, region_name="us-east-1")
            sync = InitialSync(store, chunker, mock_embedder, mongo_collection)
            report = sync.run()
            assert report.seen == 2
            assert report.failed == 0
            # At least one chunk per document should be stored
            count = mongo_collection.count_documents({})
            assert count > 0

    def test_sync_report_counters(
        self, setup_bucket, mongo_collection, mock_embedder, chunker
    ):
        with mock_aws():
            store = S3Backend(bucket_name=setup_bucket, region_name="us-east-1")
            sync = InitialSync(store, chunker, mock_embedder, mongo_collection)
            report = sync.run()
            assert report.seen == 2
            assert report.processed + report.skipped == report.seen - report.failed

    def test_download_failure_counts_as_failed(
        self, setup_bucket, mongo_collection, mock_embedder, chunker
    ):
        """A key we could not fetch is reported as failed, not silently dropped."""
        with mock_aws():
            store = S3Backend(bucket_name=setup_bucket, region_name="us-east-1")
            original_read = store.read

            def flaky_read(path, *args, **kwargs):
                if path.endswith("b.txt"):
                    raise OSError("boom")
                return original_read(path, *args, **kwargs)

            store.read = flaky_read  # type: ignore[method-assign]
            report = InitialSync(store, chunker, mock_embedder, mongo_collection).run()
            assert report.seen == 2
            assert report.failed == 1
            assert mongo_collection.count_documents({"source_path": "docs/b.txt"}) == 0

    def test_oversized_object_is_skipped_not_ingested(
        self, aws_credentials, mongo_collection, mock_embedder, chunker, monkeypatch
    ):
        """InitialSync must not pull an oversized object into memory.

        Regression for the Corridor finding: the 64 MiB cap was enforced only
        in MongoFilesystemBackend.read(), so the sync path could load an
        arbitrarily large object across 16 concurrent workers.
        """
        import langchain_mongodb_deepagents_vfs.backends.s3 as s3_mod

        monkeypatch.setattr(s3_mod, "MAX_READ_BYTES", 100)
        with mock_aws():
            client = boto3.client("s3", region_name="us-east-1")
            client.create_bucket(Bucket="cap-sync")
            client.put_object(Bucket="cap-sync", Key="big.txt", Body=b"word " * 200)
            client.put_object(Bucket="cap-sync", Key="ok.txt", Body=b"small body")
            store = S3Backend(bucket_name="cap-sync", region_name="us-east-1")

            report = InitialSync(store, chunker, mock_embedder, mongo_collection).run()

            assert report.seen == 2
            assert report.failed == 1  # the oversized key, counted not swallowed
            assert mongo_collection.count_documents({"source_path": "big.txt"}) == 0
            assert mongo_collection.count_documents({"source_path": "ok.txt"}) > 0

    def test_sync_is_idempotent(
        self, setup_bucket, mongo_collection, mock_embedder, chunker
    ):
        """Running sync twice doesn't create duplicate chunks."""
        with mock_aws():
            store = S3Backend(bucket_name=setup_bucket, region_name="us-east-1")
            sync = InitialSync(store, chunker, mock_embedder, mongo_collection)
            sync.run()
            count_after_first = mongo_collection.count_documents({})
            report2 = sync.run()
            count_after_second = mongo_collection.count_documents({})
            assert count_after_second == count_after_first
            # Second run should skip all (ETags unchanged)
            assert report2.skipped == report2.seen

    def test_sync_skips_unchanged_etags(
        self, setup_bucket, mongo_collection, mock_embedder, chunker
    ):
        with mock_aws():
            store = S3Backend(bucket_name=setup_bucket, region_name="us-east-1")
            sync = InitialSync(store, chunker, mock_embedder, mongo_collection)
            sync.run()
            embed_call_count_first = mock_embedder.embed_batch.call_count
            sync.run()
            embed_call_count_second = mock_embedder.embed_batch.call_count
            # No new embed calls on second run — everything skipped
            assert embed_call_count_second == embed_call_count_first

    def test_dry_run_does_not_embed_or_upsert(
        self, setup_bucket, mongo_collection, mock_embedder, chunker
    ):
        with mock_aws():
            store = S3Backend(bucket_name=setup_bucket, region_name="us-east-1")
            sync = InitialSync(store, chunker, mock_embedder, mongo_collection)
            report = sync.run(dry_run=True)
            mock_embedder.embed_batch.assert_not_called()
            assert mongo_collection.count_documents({}) == 0
            assert report.seen == 2

    def test_chunk_documents_have_required_fields(
        self, setup_bucket, mongo_collection, mock_embedder, chunker
    ):
        with mock_aws():
            store = S3Backend(bucket_name=setup_bucket, region_name="us-east-1")
            sync = InitialSync(store, chunker, mock_embedder, mongo_collection)
            sync.run()
            doc = mongo_collection.find_one({})
            required = {
                "source_path",
                "chunk_index",
                "content",
                "embedding",
                "line_start",
                "filename",
                "etag",
            }
            assert required.issubset(set(doc.keys()))

    def test_sync_with_prefix(
        self, aws_credentials, mongo_collection, mock_embedder, chunker
    ):
        with mock_aws():
            client = boto3.client("s3", region_name="us-east-1")
            client.create_bucket(Bucket="prefixed-bucket")
            client.put_object(Bucket="prefixed-bucket", Key="docs/a.txt", Body=b"Doc A")
            client.put_object(
                Bucket="prefixed-bucket", Key="images/b.png", Body=b"PNG data"
            )
            store = S3Backend(bucket_name="prefixed-bucket", region_name="us-east-1")
            sync = InitialSync(store, chunker, mock_embedder, mongo_collection)
            report = sync.run(prefix="docs/")
            assert report.seen == 1
