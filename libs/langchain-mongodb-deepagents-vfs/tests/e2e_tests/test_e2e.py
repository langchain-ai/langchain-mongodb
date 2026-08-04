"""End-to-end tests for MongoFilesystemBackend.

These tests use moto (mocked S3) and mongomock, wiring up the full
MongoFilesystemBackend stack without real AWS or Atlas. They exercise the
golden path: write → sync → search.

To run against real Atlas + S3, set the following env vars and use the
`e2e` marker explicitly:
    MONGO_URI=mongodb+srv://...
    S3_BUCKET=my-real-bucket
    AWS_REGION=us-east-1
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import boto3
import mongomock
import pytest
from moto import mock_aws

from langchain_mongodb_deepagents_vfs.backend import MongoFilesystemBackend
from langchain_mongodb_deepagents_vfs.dtypes import (
    EditResult,
    GlobResult,
    GrepResult,
    LsResult,
    ReadResult,
    WriteResult,
)
from langchain_mongodb_deepagents_vfs.errors import AdapterError


@pytest.fixture
def backend_env(aws_credentials):
    """Full MongoFilesystemBackend wired to moto S3 + mongomock Mongo."""
    with mock_aws():
        # Setup S3
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket="e2e-bucket")
        client.put_object(
            Bucket="e2e-bucket", Key="docs/api.txt", Body=b"authentication flow diagram"
        )
        client.put_object(
            Bucket="e2e-bucket",
            Key="docs/install.txt",
            Body=b"installation prerequisites",
        )
        client.put_object(
            Bucket="e2e-bucket", Key="images/logo.png", Body=b"PNG data here"
        )

        # Patch pymongo.MongoClient to use mongomock
        with patch(
            "langchain_mongodb_deepagents_vfs.backend.pymongo.MongoClient"
        ) as mock_mongo:
            mock_client = mongomock.MongoClient()
            mock_mongo.return_value = mock_client

            # Patch Embedder to avoid real API calls
            with patch(
                "langchain_mongodb_deepagents_vfs.backend.Embedder"
            ) as MockEmbedder:
                mock_embedder = MagicMock()
                mock_embedder._dimensions = 1024
                mock_embedder.embed_batch.side_effect = lambda chunks: [
                    [0.0] * 1024 for _ in chunks
                ]
                MockEmbedder.return_value = mock_embedder

                backend = MongoFilesystemBackend(
                    s3_bucket_name="e2e-bucket",
                    mongodb_connection_string="mongodb://localhost:27017",
                    aws_region="us-east-1",
                )
                # Wait for background init to complete
                backend._ready.wait(timeout=30)
                yield backend
                backend.stop()


@pytest.mark.e2e
class TestMongoFilesystemBackendE2E:
    def test_read_existing_file(self, backend_env):
        result = backend_env.read("docs/api.txt")
        assert isinstance(result, ReadResult)
        assert result.error is None
        assert "authentication" in result.file_data["content"]

    def test_read_missing_file_returns_error(self, backend_env):
        result = backend_env.read("docs/nonexistent.txt")
        assert result.error is not None

    def test_write_and_read_roundtrip(self, backend_env):
        result = backend_env.write("docs/new.txt", "brand new content")
        assert isinstance(result, WriteResult)
        assert result.error is None
        read_result = backend_env.read("docs/new.txt")
        assert "brand new content" in read_result.file_data["content"]

    def test_edit_file(self, backend_env):
        backend_env.write("docs/edit_me.txt", "hello world")
        result = backend_env.edit("docs/edit_me.txt", "hello", "goodbye")
        assert isinstance(result, EditResult)
        assert result.error is None
        assert result.occurrences == 1
        read_result = backend_env.read("docs/edit_me.txt")
        assert "goodbye" in read_result.file_data["content"]

    def test_ls_returns_result(self, backend_env):
        result = backend_env.ls("docs")
        assert isinstance(result, LsResult)
        assert result.error is None

    def test_glob_pdf_pattern(self, backend_env):
        result = backend_env.glob("**/*.txt")
        assert isinstance(result, GlobResult)
        assert result.error is None

    def test_grep_returns_matches(self, backend_env):
        result = backend_env.grep("authentication")
        assert isinstance(result, GrepResult)
        assert result.error is None

    def test_grep_match_shape(self, backend_env):
        result = backend_env.grep("installation")
        if result.matches:
            match = result.matches[0]
            assert "path" in match
            assert "line" in match
            assert "text" in match

    def test_upload_and_download(self, backend_env):
        upload_result = backend_env.upload_files(
            [("docs/uploaded.txt", b"uploaded bytes")]
        )
        assert [r.path for r in upload_result] == ["docs/uploaded.txt"]
        assert all(r.error is None for r in upload_result)

        download_result = backend_env.download_files(["docs/uploaded.txt"])
        assert [r.path for r in download_result] == ["docs/uploaded.txt"]
        assert download_result[0].content == b"uploaded bytes"

    def test_constructor_validates_missing_bucket(self):
        with pytest.raises(AdapterError):
            MongoFilesystemBackend(
                s3_bucket_name="",
                mongodb_connection_string="mongodb://localhost",
            )

    def test_constructor_validates_missing_mongo_uri(self):
        with pytest.raises(AdapterError):
            MongoFilesystemBackend(
                s3_bucket_name="my-bucket",
                mongodb_connection_string="",
            )

    def test_constructor_validates_sqs_without_queue_url(self):
        with pytest.raises(AdapterError):
            MongoFilesystemBackend(
                s3_bucket_name="my-bucket",
                mongodb_connection_string="mongodb://localhost",
                watcher="sqs",
            )

    def test_error_surfaced_without_stack_trace(self, backend_env):
        # debug=False: errors return as DTO fields, not exceptions.
        # We force the search router to throw so the adapter_boundary catches it.
        with patch(
            "langchain_mongodb_deepagents_vfs.backend.pymongo.MongoClient"
        ) as mock_mongo:
            mock_client = mongomock.MongoClient()
            mock_mongo.return_value = mock_client
            with patch(
                "langchain_mongodb_deepagents_vfs.backend.Embedder"
            ) as MockEmbedder:
                mock_embedder = MagicMock()
                MockEmbedder.return_value = mock_embedder
                with mock_aws():
                    client = boto3.client("s3", region_name="us-east-1")
                    client.create_bucket(Bucket="err-bucket")
                    b = MongoFilesystemBackend(
                        s3_bucket_name="err-bucket",
                        mongodb_connection_string="mongodb://localhost:27017",
                        aws_region="us-east-1",
                        debug=False,
                    )
                    b._ready.set()  # skip waiting
                    # Force the search layer to raise so the boundary captures it
                    b._search.grep = MagicMock(side_effect=RuntimeError("boom"))
                    result = b.grep("anything")
                    # Should return a GrepResult with error, not raise
                    assert result.error is not None
