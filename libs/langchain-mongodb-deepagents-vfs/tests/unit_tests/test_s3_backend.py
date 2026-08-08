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
class TestS3BackendGetSize:
    def test_get_size_returns_byte_length(self, aws_credentials):
        with mock_aws():
            client = boto3.client("s3", region_name="us-east-1")
            client.create_bucket(Bucket="test-bkt")
            client.put_object(Bucket="test-bkt", Key="f.txt", Body=b"0123456789")
            backend = S3Backend(bucket_name="test-bkt", region_name="us-east-1")
            assert backend.get_size("f.txt") == 10

    def test_get_size_missing_raises(self, aws_credentials):
        with mock_aws():
            client = boto3.client("s3", region_name="us-east-1")
            client.create_bucket(Bucket="test-bkt")
            backend = S3Backend(bucket_name="test-bkt", region_name="us-east-1")
            with pytest.raises(AdapterError) as ei:
                backend.get_size("missing.txt")
            assert ei.value.code == ErrorCode.E2001_OBJECT_NOT_FOUND


@pytest.mark.unit
class TestReadSizeCap:
    def test_read_rejects_oversized_object(self):
        """backend.read HEADs first and refuses objects past the memory cap."""
        from langchain_mongodb_deepagents_vfs.backend import MongoFilesystemBackend
        from langchain_mongodb_deepagents_vfs.backends.base import MAX_READ_BYTES

        class _FakeStore:
            def get_size(self, path):
                return MAX_READ_BYTES + 1

            def read(self, path):  # pragma: no cover - must not be reached
                raise AssertionError("read() called despite oversized object")

        backend = MongoFilesystemBackend.__new__(MongoFilesystemBackend)
        backend._store = _FakeStore()
        backend.debug = True  # re-raise past the adapter boundary
        with pytest.raises(AdapterError) as ei:
            backend.read("bomb.txt", limit=1)
        assert ei.value.code == ErrorCode.E2002_OBJECT_READ_FAILED

    def test_store_read_enforces_cap_itself(self, aws_credentials, monkeypatch):
        """The cap lives in S3Backend.read, so every caller is covered.

        InitialSync, the watchers and download_files all call read() directly
        and never did their own pre-flight check.
        """
        import langchain_mongodb_deepagents_vfs.backends.s3 as s3_mod

        monkeypatch.setattr(s3_mod, "MAX_READ_BYTES", 100)
        with mock_aws():
            client = boto3.client("s3", region_name="us-east-1")
            client.create_bucket(Bucket="cap-bucket")
            client.put_object(Bucket="cap-bucket", Key="big.txt", Body=b"x" * 500)
            client.put_object(Bucket="cap-bucket", Key="small.txt", Body=b"x" * 50)
            backend = S3Backend(bucket_name="cap-bucket", region_name="us-east-1")

            with pytest.raises(AdapterError) as ei:
                backend.read("big.txt")
            assert ei.value.code == ErrorCode.E2002_OBJECT_READ_FAILED

            assert backend.read("small.txt") == b"x" * 50

    def test_edit_enforces_cap(self, aws_credentials, monkeypatch):
        """edit() is a read-modify-write and needs the same ceiling as read()."""
        import langchain_mongodb_deepagents_vfs.backends.s3 as s3_mod

        monkeypatch.setattr(s3_mod, "MAX_READ_BYTES", 100)
        with mock_aws():
            client = boto3.client("s3", region_name="us-east-1")
            client.create_bucket(Bucket="edit-cap")
            client.put_object(Bucket="edit-cap", Key="big.txt", Body=b"x" * 500)
            client.put_object(Bucket="edit-cap", Key="ok.txt", Body=b"hello world")
            backend = S3Backend(bucket_name="edit-cap", region_name="us-east-1")

            with pytest.raises(AdapterError) as ei:
                backend.edit("big.txt", "x", "y")
            assert ei.value.code == ErrorCode.E2002_OBJECT_READ_FAILED

            assert backend.edit("ok.txt", "hello", "goodbye") == 1

    def test_write_refuses_oversized_content(self, aws_credentials, monkeypatch):
        """Writing past the read cap would store an unreadable object."""
        import langchain_mongodb_deepagents_vfs.backends.s3 as s3_mod

        monkeypatch.setattr(s3_mod, "MAX_READ_BYTES", 100)
        with mock_aws():
            client = boto3.client("s3", region_name="us-east-1")
            client.create_bucket(Bucket="write-cap")
            backend = S3Backend(bucket_name="write-cap", region_name="us-east-1")

            with pytest.raises(AdapterError) as ei:
                backend.write("bomb.txt", b"x" * 500)
            assert ei.value.code == ErrorCode.E2003_OBJECT_WRITE_FAILED
            # nothing was stored
            assert client.list_objects_v2(Bucket="write-cap").get("KeyCount", 0) == 0

            backend.write("ok.txt", b"x" * 10)
            assert backend.read("ok.txt") == b"x" * 10

    def test_download_files_reports_oversized_per_path(
        self, aws_credentials, monkeypatch
    ):
        """download_files funnels through read(), so it inherits the cap."""
        import langchain_mongodb_deepagents_vfs.backends.s3 as s3_mod

        monkeypatch.setattr(s3_mod, "MAX_READ_BYTES", 100)
        with mock_aws():
            client = boto3.client("s3", region_name="us-east-1")
            client.create_bucket(Bucket="dl-bucket")
            client.put_object(Bucket="dl-bucket", Key="big.txt", Body=b"x" * 500)
            client.put_object(Bucket="dl-bucket", Key="ok.txt", Body=b"x" * 10)
            backend = S3Backend(bucket_name="dl-bucket", region_name="us-east-1")

            responses = backend.download_files(["big.txt", "ok.txt"])
            by_path = {r.path: r for r in responses}
            assert by_path["big.txt"].error is not None
            assert by_path["ok.txt"].error is None
            assert by_path["ok.txt"].content == b"x" * 10


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
