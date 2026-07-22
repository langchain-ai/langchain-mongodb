"""Unit tests for S3Backend using moto."""

from __future__ import annotations

import boto3
import pytest
from moto import mock_aws

from langchain_mongodb_deepagents_vfs.backends.s3 import S3Backend
from langchain_mongodb_deepagents_vfs.errors import AdapterError, ErrorCode


@pytest.fixture
def backend(aws_credentials):
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket="test-bucket")
        yield S3Backend(bucket_name="test-bucket", region_name="us-east-1")


@pytest.mark.unit
class TestS3BackendInit:
    def test_invalid_bucket_raises(self, aws_credentials):
        with mock_aws():
            with pytest.raises(AdapterError) as exc_info:
                S3Backend(bucket_name="nonexistent-bucket", region_name="us-east-1")
            assert exc_info.value.code == ErrorCode.E1002_INVALID_BUCKET


@pytest.mark.unit
class TestS3BackendRead:
    def test_read_existing_object(self, aws_credentials):
        with mock_aws():
            client = boto3.client("s3", region_name="us-east-1")
            client.create_bucket(Bucket="test-bkt")
            client.put_object(Bucket="test-bkt", Key="file.txt", Body=b"hello world")
            backend = S3Backend(bucket_name="test-bkt", region_name="us-east-1")
            data = backend.read("file.txt")
            assert data == b"hello world"

    def test_read_missing_object_raises(self, aws_credentials):
        with mock_aws():
            client = boto3.client("s3", region_name="us-east-1")
            client.create_bucket(Bucket="test-bkt")
            backend = S3Backend(bucket_name="test-bkt", region_name="us-east-1")
            with pytest.raises(AdapterError) as exc_info:
                backend.read("missing.txt")
            assert exc_info.value.code == ErrorCode.E2001_OBJECT_NOT_FOUND

    def test_read_with_offset(self, aws_credentials):
        with mock_aws():
            client = boto3.client("s3", region_name="us-east-1")
            client.create_bucket(Bucket="test-bkt")
            client.put_object(Bucket="test-bkt", Key="f.txt", Body=b"0123456789")
            backend = S3Backend(bucket_name="test-bkt", region_name="us-east-1")
            data = backend.read("f.txt", offset=5)
            assert data == b"56789"


@pytest.mark.unit
class TestS3BackendWrite:
    def test_write_creates_object(self, aws_credentials):
        with mock_aws():
            client = boto3.client("s3", region_name="us-east-1")
            client.create_bucket(Bucket="test-bkt")
            backend = S3Backend(bucket_name="test-bkt", region_name="us-east-1")
            backend.write("new.txt", b"content")
            obj = client.get_object(Bucket="test-bkt", Key="new.txt")
            assert obj["Body"].read() == b"content"

    def test_write_overwrites_existing(self, aws_credentials):
        with mock_aws():
            client = boto3.client("s3", region_name="us-east-1")
            client.create_bucket(Bucket="test-bkt")
            client.put_object(Bucket="test-bkt", Key="f.txt", Body=b"old")
            backend = S3Backend(bucket_name="test-bkt", region_name="us-east-1")
            backend.write("f.txt", b"new")
            obj = client.get_object(Bucket="test-bkt", Key="f.txt")
            assert obj["Body"].read() == b"new"


@pytest.mark.unit
class TestS3BackendEdit:
    def test_edit_replaces_first_occurrence(self, aws_credentials):
        with mock_aws():
            client = boto3.client("s3", region_name="us-east-1")
            client.create_bucket(Bucket="test-bkt")
            client.put_object(Bucket="test-bkt", Key="f.txt", Body=b"foo foo foo")
            backend = S3Backend(bucket_name="test-bkt", region_name="us-east-1")
            assert backend.edit("f.txt", "foo", "bar") == 1
            obj = client.get_object(Bucket="test-bkt", Key="f.txt")
            assert obj["Body"].read() == b"bar foo foo"

    def test_edit_replace_all(self, aws_credentials):
        with mock_aws():
            client = boto3.client("s3", region_name="us-east-1")
            client.create_bucket(Bucket="test-bkt")
            client.put_object(Bucket="test-bkt", Key="f.txt", Body=b"foo foo foo")
            backend = S3Backend(bucket_name="test-bkt", region_name="us-east-1")
            assert backend.edit("f.txt", "foo", "bar", replace_all=True) == 3
            obj = client.get_object(Bucket="test-bkt", Key="f.txt")
            assert obj["Body"].read() == b"bar bar bar"

    def test_edit_missing_file_raises(self, aws_credentials):
        with mock_aws():
            client = boto3.client("s3", region_name="us-east-1")
            client.create_bucket(Bucket="test-bkt")
            backend = S3Backend(bucket_name="test-bkt", region_name="us-east-1")
            with pytest.raises(AdapterError) as exc_info:
                backend.edit("missing.txt", "a", "b")
            assert exc_info.value.code == ErrorCode.E2001_OBJECT_NOT_FOUND


@pytest.mark.unit
class TestS3BackendListKeys:
    def test_list_keys_returns_all(self, aws_credentials):
        with mock_aws():
            client = boto3.client("s3", region_name="us-east-1")
            client.create_bucket(Bucket="test-bkt")
            client.put_object(Bucket="test-bkt", Key="a.txt", Body=b"a")
            client.put_object(Bucket="test-bkt", Key="b.txt", Body=b"b")
            backend = S3Backend(bucket_name="test-bkt", region_name="us-east-1")
            keys = list(backend.list_keys())
            assert {k for k, _ in keys} == {"a.txt", "b.txt"}

    def test_list_keys_with_prefix(self, aws_credentials):
        with mock_aws():
            client = boto3.client("s3", region_name="us-east-1")
            client.create_bucket(Bucket="test-bkt")
            client.put_object(Bucket="test-bkt", Key="docs/a.txt", Body=b"a")
            client.put_object(Bucket="test-bkt", Key="images/b.png", Body=b"b")
            backend = S3Backend(bucket_name="test-bkt", region_name="us-east-1")
            keys = list(backend.list_keys(prefix="docs/"))
            assert all(k.startswith("docs/") for k, _ in keys)
            assert len(keys) == 1

    def test_list_keys_includes_etag(self, aws_credentials):
        with mock_aws():
            client = boto3.client("s3", region_name="us-east-1")
            client.create_bucket(Bucket="test-bkt")
            client.put_object(Bucket="test-bkt", Key="f.txt", Body=b"hello")
            backend = S3Backend(bucket_name="test-bkt", region_name="us-east-1")
            keys = list(backend.list_keys())
            assert len(keys) == 1
            key, etag = keys[0]
            assert etag != ""


@pytest.mark.unit
class TestS3BackendUploadDownload:
    def test_upload_multiple_files(self, aws_credentials):
        with mock_aws():
            client = boto3.client("s3", region_name="us-east-1")
            client.create_bucket(Bucket="test-bkt")
            backend = S3Backend(bucket_name="test-bkt", region_name="us-east-1")
            files = [("a.txt", b"aaa"), ("b.txt", b"bbb")]
            responses = backend.upload_files(files)
            assert [r.path for r in responses] == ["a.txt", "b.txt"]
            assert all(r.error is None for r in responses)

    def test_download_multiple_files(self, aws_credentials):
        with mock_aws():
            client = boto3.client("s3", region_name="us-east-1")
            client.create_bucket(Bucket="test-bkt")
            client.put_object(Bucket="test-bkt", Key="a.txt", Body=b"aaa")
            client.put_object(Bucket="test-bkt", Key="b.txt", Body=b"bbb")
            backend = S3Backend(bucket_name="test-bkt", region_name="us-east-1")
            results = backend.download_files(["a.txt", "b.txt"])
            data_map = {r.path: r.content for r in results}
            assert data_map["a.txt"] == b"aaa"
            assert data_map["b.txt"] == b"bbb"

    def test_download_partial_failure_keeps_successes(self, aws_credentials):
        """Protocol contract: one response per input, failures don't sink the batch."""
        with mock_aws():
            client = boto3.client("s3", region_name="us-east-1")
            client.create_bucket(Bucket="test-bkt")
            client.put_object(Bucket="test-bkt", Key="a.txt", Body=b"aaa")
            backend = S3Backend(bucket_name="test-bkt", region_name="us-east-1")
            results = backend.download_files(["a.txt", "missing.txt"])
            assert [r.path for r in results] == ["a.txt", "missing.txt"]
            assert results[0].error is None
            assert results[0].content == b"aaa"
            assert results[1].error == "file_not_found"
            assert results[1].content is None


@pytest.mark.unit
class TestS3BackendKeyNormalization:
    def test_normalize_windows_path(self):
        result = S3Backend.normalize_key("docs\\subdir\\file.txt")
        assert "\\" not in result
        assert "/" in result
